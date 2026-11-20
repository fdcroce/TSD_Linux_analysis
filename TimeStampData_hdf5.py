import os
import sys
import time
from datetime import datetime
import numpy as np
import tables as tb
import pandas as pd
import numba
from numba import njit, prange
from scipy.optimize import curve_fit


# ============================================================
# 1. BINARY TO HDF5 CONVERSION
# ============================================================

def convert_bin_to_hdf5(bin_path, hdf5_path, chunk_size=40_000_000, mode_override=None):
    """
    Converts raw CAEN binary timestamp files (.bin / .dat) into a compressed,
    channel-split HDF5 file using memory-mapped I/O and Blosc-LZ4 compression.
    """
    start_time = time.time()
    file_size_gb = os.path.getsize(bin_path) / (1024**3)
    print("=== STARTING OPTIMIZED CAEN TO HDF5 CONVERSION ===")
    print(f"Source file: {bin_path} ({file_size_gb:.2f} GB)")
    print(f"Destination file: {hdf5_path}")

    try:
        stat = os.stat(bin_path)
        file_ts = min(stat.st_mtime, getattr(stat, 'st_birthtime', stat.st_ctime))
    except AttributeError:
        file_ts = min(os.path.getmtime(bin_path), os.path.getctime(bin_path))
    measurement_start_iso = datetime.fromtimestamp(file_ts).isoformat()

    header_dtype = np.dtype([
        ('op_mode', 'u1'), ('cycles', '>u4'), ('cycle_len', '>u8'),
        ('agg_mode', 'u1'), ('chan_group_num', 'u1'), ('chan_mask', '>u4'),
        ('group_mask', '>u2'), ('group_size', 'u1')
    ])

    m = np.memmap(bin_path, dtype='uint8', mode='r')
    header_raw = m[:22].view(header_dtype)[0]

    agg_mode = mode_override if mode_override is not None else header_raw['agg_mode']
    has_channels = (agg_mode == 0)

    if has_channels:
        event_dtype = np.dtype([('timetag', '>u8'), ('channel', '>u2'), ('flags', '>u2')])
        print("--> Mode detected: CH_ALL (12 bytes per event)")
    else:
        event_dtype = np.dtype([('timetag', '>u8'), ('flags', '>u2')])
        print("--> Mode detected: CH_INDIVIDUAL / CH_GROUPED (10 bytes per event)")

    num_records = (m.size - 22) // event_dtype.itemsize
    data_view = m[22 : 22 + num_records * event_dtype.itemsize].view(event_dtype)
    print(f"Successfully mapped {num_records:,} events.")

    filters = tb.Filters(complevel=4, complib='blosc:lz4')
    chunk_schema = np.dtype([('timetag', 'u8'), ('flags', 'u2')])

    with tb.open_file(hdf5_path, mode='w', title="CAEN Repacked Data", filters=filters) as h5file:
        for field in header_dtype.names:
            h5file.root._v_attrs[field] = header_raw[field]
        h5file.root._v_attrs['agg_mode_effective'] = np.uint8(agg_mode)
        h5file.root._v_attrs['start_measurement_time'] = measurement_start_iso
        h5file.root._v_attrs['start_measurement_timestamp'] = float(file_ts)

        h5_tables = {}
        print("Streaming data sections to HDF5 tables...")

        for start in range(0, num_records, chunk_size):
            end = min(start + chunk_size, num_records)
            chunk = data_view[start:end]

            t_data = chunk['timetag'].astype('u8')
            f_data = chunk['flags'].astype('u2')

            if has_channels:
                c_data = chunk['channel'].astype('u2')
                order = np.argsort(c_data, kind='mergesort')
                c_sorted = c_data[order]
                t_sorted = t_data[order]
                f_sorted = f_data[order]

                unique_ch, split_idx = np.unique(c_sorted, return_index=True)
                split_idx = np.append(split_idx, c_sorted.size)

                for idx, ch in enumerate(unique_ch):
                    if ch not in h5_tables:
                        h5_tables[ch] = h5file.create_table(
                            '/', f'channel_{ch}', chunk_schema, f"Data Channel {ch}",
                            expectedrows=num_records
                        )
                    s, e = split_idx[idx], split_idx[idx + 1]
                    batch = np.empty(e - s, dtype=chunk_schema)
                    batch['timetag'] = t_sorted[s:e]
                    batch['flags'] = f_sorted[s:e]
                    h5_tables[ch].append(batch)
            else:
                if 0 not in h5_tables:
                    h5_tables[0] = h5file.create_table(
                        '/', 'channel_0', chunk_schema, "Data Channel 0",
                        expectedrows=num_records
                    )
                batch = np.empty(t_data.size, dtype=chunk_schema)
                batch['timetag'] = t_data
                batch['flags'] = f_data
                h5_tables[0].append(batch)

            print(f"  Progress: {(end / num_records) * 100:.1f}% | Processed {end:,} / {num_records:,} events")

        h5file.flush()

    print(f"=== CONVERSION COMPLETED IN {time.time() - start_time:.2f} SECONDS ===")


# ============================================================
# 2. NUMBA PARALLEL COMPUTATION KERNELS
# ============================================================

@njit(parallel=True)
def _fast_tid_core(chunk_tags, start_tick, max_delta_ticks, coarser_grouping, num_bins, include_overflow):
    """
    Thread-isolated Numba kernel for consecutive Time Interval Distribution (TID).
    """
    n_threads = numba.get_num_threads()
    thread_histograms = np.zeros((n_threads, num_bins), dtype=np.int64)
    num_events = chunk_tags.size

    for i in prange(num_events - 1):
        t0 = chunk_tags[i]
        t1 = chunk_tags[i + 1]
        if t1 < t0:
            continue

        dt = t1 - t0
        if dt >= start_tick:
            tid = numba.get_thread_id()
            if dt < max_delta_ticks:
                bin_idx = int((dt - start_tick) // coarser_grouping)
                if 0 <= bin_idx < num_bins:
                    thread_histograms[tid, bin_idx] += 1
            elif include_overflow:
                thread_histograms[tid, num_bins - 1] += 1

    global_counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        global_counts += thread_histograms[t]
    return global_counts


@njit(parallel=True)
def _rossi_alpha_window_core(timetags, active_elements, start_tick, gate_ticks, coarser_grouping, num_bins):
    """
    Thread-isolated Numba kernel for Auto-Correlation Rossi-Alpha (single channel).
    """
    n_threads = numba.get_num_threads()
    thread_histograms = np.zeros((n_threads, num_bins), dtype=np.int64)

    for i in prange(active_elements):
        t_trigger = timetags[i]
        tid = numba.get_thread_id()
        j = i + 1

        while j < timetags.size:
            t_future = timetags[j]
            if t_future < t_trigger:
                j += 1
                continue

            dt = t_future - t_trigger
            if dt >= gate_ticks:
                break

            if dt >= start_tick:
                bin_idx = int((dt - start_tick) // coarser_grouping)
                if 0 <= bin_idx < num_bins:
                    thread_histograms[tid, bin_idx] += 1
            j += 1

    global_counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        global_counts += thread_histograms[t]
    return global_counts


@njit(parallel=True)
def _rossi_alpha_cross_core(trigger_tags, response_tags, start_tick, gate_ticks, coarser_grouping, num_bins):
    """
    Thread-isolated Numba kernel for Cross-Correlation Rossi-Alpha.
    Triggers on `trigger_tags` and histograms forward pulses in `response_tags`
    within [start_tick, gate_ticks).
    """
    n_threads = numba.get_num_threads()
    thread_histograms = np.zeros((n_threads, num_bins), dtype=np.int64)
    n_trig = trigger_tags.size
    n_resp = response_tags.size

    for i in prange(n_trig):
        t_trigger = trigger_tags[i]
        tid = numba.get_thread_id()

        target = t_trigger + np.uint64(start_tick)
        lo = 0
        hi = n_resp
        while lo < hi:
            mid = (lo + hi) // 2
            if response_tags[mid] < target:
                lo = mid + 1
            else:
                hi = mid

        j = lo
        while j < n_resp:
            t_future = response_tags[j]
            if t_future < t_trigger:
                j += 1
                continue

            dt = t_future - t_trigger
            if dt >= gate_ticks:
                break

            if dt >= start_tick:
                bin_idx = int((dt - start_tick) // coarser_grouping)
                if 0 <= bin_idx < num_bins:
                    thread_histograms[tid, bin_idx] += 1
            j += 1

    global_counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        global_counts += thread_histograms[t]
    return global_counts


# ============================================================
# 3. STREAMING ANALYSIS FUNCTIONS (PYTABLES NODES)
# ============================================================

def _binary_search_timetag(table_node, target_tick, side="left"):
    """O(log N) binary search on an on-disk PyTables 'timetag' column."""
    lo, hi = 0, len(table_node)
    while lo < hi:
        mid = (lo + hi) // 2
        val = table_node[mid]['timetag']
        if (val < target_tick) if side == "left" else (val <= target_tick):
            lo = mid + 1
        else:
            hi = mid
    return lo


def calculate_count_rate(
    h5_channel,
    bin_size_seconds,
    time_resolution_sec=10e-9,
    chunk_size=50_000_000,
    filter_flags=True,
    absolute_time=False,
    drop_zeros=False,
    drop_last_bin=True,
    start_idx=0,
    stop_idx=None,
):
    """
    Computes count rate (Hz) over time by streaming chunks out of an HDF5 channel table.
    Drops the incomplete trailing bin by default (`drop_last_bin=True`).
    """
    stop_idx = len(h5_channel) if stop_idx is None else min(stop_idx, len(h5_channel))
    total_events = max(0, stop_idx - start_idx)
    if total_events == 0:
        out = pd.Series(dtype=np.float64, name="count_rate")
        out.index.name = "time" if absolute_time else "time_seconds"
        return out

    bin_size_ticks = bin_size_seconds / time_resolution_sec
    first_tick = h5_channel[start_idx]['timetag']
    last_tick = h5_channel[stop_idx - 1]['timetag']
    total_ticks = last_tick - first_tick

    if drop_last_bin:
        num_bins = int(total_ticks // bin_size_ticks)
    else:
        num_bins = int(np.ceil(total_ticks / bin_size_ticks))

    if num_bins <= 0:
        out = pd.Series(dtype=np.float64, name="count_rate")
        out.index.name = "time" if absolute_time else "time_seconds"
        return out

    global_counts = np.zeros(num_bins, dtype=np.int64)

    for start in range(start_idx, stop_idx, chunk_size):
        end = min(start + chunk_size, stop_idx)
        chunk_tags = h5_channel.read(start, end, field='timetag')

        if filter_flags:
            chunk_flags = h5_channel.read(start, end, field='flags')
            chunk_tags = chunk_tags[chunk_flags == 0]
            if chunk_tags.size == 0:
                continue

        bin_indices = ((chunk_tags - first_tick) / bin_size_ticks).astype(np.int64)
        valid = (bin_indices >= 0) & (bin_indices < num_bins)
        chunk_counts = np.bincount(bin_indices[valid], minlength=num_bins)
        global_counts += chunk_counts[:num_bins]

    if drop_zeros:
        idx = np.nonzero(global_counts)[0]
        rate_hz = global_counts[idx] / bin_size_seconds
        relative_times = idx * bin_size_seconds
    else:
        rate_hz = global_counts / bin_size_seconds
        relative_times = np.arange(num_bins) * bin_size_seconds

    if absolute_time:
        root_attrs = h5_channel._v_file.root._v_attrs
        if 'start_measurement_time' not in root_attrs:
            raise ValueError("HDF5 file does not contain 'start_measurement_time' attribute.")
        file_first_tick = h5_channel[0]['timetag']
        offset_sec = (first_tick - file_first_tick) * time_resolution_sec
        base_time = (
            np.datetime64(root_attrs['start_measurement_time'], 'us')
            + np.timedelta64(int(round(offset_sec * 1_000_000)), 'us')
        )
        time_deltas = (relative_times * 1_000_000).astype('timedelta64[us]')
        time_axis = base_time + time_deltas
        index_name = "time"
    else:
        time_axis = relative_times
        index_name = "time_seconds"

    out = pd.Series(rate_hz, index=time_axis, name="count_rate")
    out.index.name = index_name
    return out


def compute_time_interval_distribution(
    h5_channel,
    chunk_size=50_000_000,
    max_delta_ticks=5000,
    coarser_grouping=1,
    start_tick=0,
    filter_flags=True,
    include_overflow=False,
    drop_zeros=False,
    start_idx=0,
    stop_idx=None,
):
    """
    Computes consecutive Time Interval Distribution (TID) across chunk boundaries.
    """
    stop_idx = len(h5_channel) if stop_idx is None else min(stop_idx, len(h5_channel))
    tid_bins = np.arange(start_tick, max_delta_ticks + coarser_grouping, coarser_grouping)
    num_bins = len(tid_bins) - 1
    global_counts = np.zeros(num_bins, dtype=np.int64)

    carryover_tag = None

    for start in range(start_idx, stop_idx, chunk_size):
        end = min(start + chunk_size, stop_idx)
        chunk_tags = h5_channel.read(start, end, field='timetag')

        if filter_flags:
            chunk_flags = h5_channel.read(start, end, field='flags')
            chunk_tags = chunk_tags[chunk_flags == 0]

        if chunk_tags.size == 0:
            continue

        if carryover_tag is not None:
            chunk_tags = np.concatenate((np.array([carryover_tag], dtype=np.uint64), chunk_tags))

        carryover_tag = chunk_tags[-1]

        if chunk_tags.size > 1:
            global_counts += _fast_tid_core(
                chunk_tags, start_tick, max_delta_ticks, coarser_grouping, num_bins, include_overflow
            )

    edges = tid_bins[:-1]
    if drop_zeros:
        nz = np.nonzero(global_counts)[0]
        out = pd.Series(global_counts[nz], index=edges[nz], name="time_interval_distribution")
    else:
        out = pd.Series(global_counts, index=edges, name="time_interval_distribution")

    out.index.name = "time_interval"
    return out


def compute_rossi_alpha(
    h5_channel,
    h5_trigger=None,
    chunk_size=50_000_000,
    gate=50e-6,
    coarser_grouping=1,
    time_resolution_sec=10e-9,
    start_tick=0,
    filter_flags=True,
    start_idx=0,
    stop_idx=None,
    trigger_start_idx=0,
    trigger_stop_idx=None,
):
    """
    Computes Triggered Forward-Window Rossi-Alpha distribution.

    - If `h5_trigger is None`: computes auto-correlation on `h5_channel`.
    - If `h5_trigger` is provided: triggers on events in `h5_trigger` and histograms
      forward follower events in `h5_channel` within `gate` seconds.
    """
    if isinstance(h5_trigger, HDF5ChannelHandle):
        trigger_start_idx = h5_trigger.start_idx
        trigger_stop_idx = h5_trigger.stop_idx
        h5_trigger = h5_trigger.table

    stop_idx = len(h5_channel) if stop_idx is None else min(stop_idx, len(h5_channel))
    gate_ticks = int(round(gate / time_resolution_sec))
    if gate_ticks <= coarser_grouping:
        raise ValueError("The gate time must be larger than the coarser_grouping width.")

    rossi_bins_ticks = np.arange(start_tick, gate_ticks + coarser_grouping, coarser_grouping)
    num_bins = len(rossi_bins_ticks) - 1
    rossi_counts = np.zeros(num_bins, dtype=np.int64)

    # MODE A: AUTO-CORRELATION
    if h5_trigger is None:
        for start in range(start_idx, stop_idx, chunk_size):
            end = min(start + chunk_size, stop_idx)
            chunk_tags = h5_channel.read(start, end, field='timetag')

            if filter_flags:
                chunk_flags = h5_channel.read(start, end, field='flags')
                chunk_tags = chunk_tags[chunk_flags == 0]

            active_elements = chunk_tags.size
            if active_elements == 0:
                continue

            cutoff_tick = chunk_tags[-1] + np.uint64(gate_ticks)

            pad_step = 5_000_000
            pad_end = end
            padding_chunks = []
            while pad_end < stop_idx:
                next_end = min(pad_end + pad_step, stop_idx)
                pad_slice = h5_channel.read(pad_end, next_end, field='timetag')

                if filter_flags:
                    pad_flags = h5_channel.read(pad_end, next_end, field='flags')
                    pad_slice = pad_slice[pad_flags == 0]

                pad_end = next_end
                if pad_slice.size > 0:
                    padding_chunks.append(pad_slice)
                    if pad_slice[-1] >= cutoff_tick:
                        break

            if padding_chunks:
                padding_tags = np.concatenate(padding_chunks) if len(padding_chunks) > 1 else padding_chunks[0]
                idx_in_padding = np.searchsorted(padding_tags, cutoff_tick, side='right')
                extended_tags = np.concatenate((chunk_tags, padding_tags[:idx_in_padding]))
            else:
                extended_tags = chunk_tags

            rossi_counts += _rossi_alpha_window_core(
                extended_tags, active_elements, start_tick, gate_ticks, coarser_grouping, num_bins
            )

    # MODE B: CROSS-CORRELATION (Trigger = h5_trigger, Response = h5_channel)
    else:
        trigger_stop_idx = (
            len(h5_trigger) if trigger_stop_idx is None else min(trigger_stop_idx, len(h5_trigger))
        )
        if start_idx >= stop_idx or trigger_start_idx >= trigger_stop_idx:
            rossi_bins_seconds = rossi_bins_ticks[:-1] * time_resolution_sec
            out = pd.Series(rossi_counts, index=rossi_bins_seconds, name="rossi_alpha_distribution")
            out.index.name = "time_interval"
            return out

        for start in range(trigger_start_idx, trigger_stop_idx, chunk_size):
            end = min(start + chunk_size, trigger_stop_idx)
            trig_tags = h5_trigger.read(start, end, field='timetag')

            if filter_flags:
                trig_flags = h5_trigger.read(start, end, field='flags')
                trig_tags = trig_tags[trig_flags == 0]

            if trig_tags.size == 0:
                continue

            min_tick = trig_tags[0] + np.uint64(start_tick)
            max_tick = trig_tags[-1] + np.uint64(gate_ticks)

            resp_s = _binary_search_timetag(h5_channel, min_tick, side="left")
            resp_e = _binary_search_timetag(h5_channel, max_tick, side="right")

            resp_s = max(start_idx, min(resp_s, stop_idx))
            resp_e = max(resp_s, min(resp_e, stop_idx))

            if resp_s >= resp_e:
                continue

            resp_tags = h5_channel.read(resp_s, resp_e, field='timetag')
            if filter_flags:
                resp_flags = h5_channel.read(resp_s, resp_e, field='flags')
                resp_tags = resp_tags[resp_flags == 0]

            if resp_tags.size == 0:
                continue

            rossi_counts += _rossi_alpha_cross_core(
                trig_tags, resp_tags, start_tick, gate_ticks, coarser_grouping, num_bins
            )

    rossi_bins_seconds = rossi_bins_ticks[:-1] * time_resolution_sec
    out = pd.Series(rossi_counts, index=rossi_bins_seconds, name="rossi_alpha_distribution")
    out.index.name = "time_interval"
    return out


# ============================================================
# 4. ROSSI-ALPHA MULTI-EXPONENTIAL FITTER
# ============================================================

def fit_rossi_alpha(
    data,
    n_exp=1,
    x_col="time_interval",
    y_col="rossi_alpha_distribution",
    p0=None,
):
    """
    Fits y(x) = sum_{k=1..n_exp} A_k * exp(B_k * x) + C to a Rossi-Alpha curve
    using unweighted least squares (`sigma=None`, `absolute_sigma=False`).

    Returns:
    --------
    popt, cov, r2, chi2red, model
    """
    if n_exp < 1:
        raise ValueError("n_exp must be >= 1.")

    if isinstance(data, pd.Series):
        x = data.index.to_numpy(dtype=np.float64)
        y = data.to_numpy(dtype=np.float64)
    else:
        x = data[x_col].to_numpy(dtype=np.float64)
        y = data[y_col].to_numpy(dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]

    if p0 is None:
        C0 = float(np.min(y))
        A_base = float(max(y[0] - C0, 1.0))

        y_net = y - C0
        pos = y_net > 0
        if np.count_nonzero(pos) >= 3:
            slope, intercept = np.polyfit(x[pos], np.log(y_net[pos]), deg=1)
            B_base = min(float(slope), -1.0)
            A_base = float(np.exp(intercept))
        else:
            span = max(float(x[-1] - x[0]), 1e-9)
            B_base = -5.0 / span

        if n_exp == 1:
            p0 = [A_base, B_base, C0]
        else:
            multipliers = np.geomspace(3.0, 0.4, n_exp)
            amp_weights = np.geomspace(0.6, 0.4, n_exp)
            amp_weights /= np.sum(amp_weights)

            p0 = []
            for m, w in zip(multipliers, amp_weights):
                p0.extend([A_base * w, B_base * m])
            p0.append(C0)

    def multi_exp_model(t, *params):
        C = params[-1]
        total = np.full_like(t, C, dtype=np.float64)
        for k in range(n_exp):
            total += params[2 * k] * np.exp(params[2 * k + 1] * t)
        return total

    lower_bounds = [0.0, -np.inf] * n_exp + [0.0]
    upper_bounds = [np.inf, 0.0] * n_exp + [np.inf]

    popt, cov = curve_fit(
        multi_exp_model,
        x,
        y,
        p0=p0,
        sigma=None,
        absolute_sigma=False,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    comp_indices = np.argsort([popt[2 * k + 1] for k in range(n_exp)])
    perm = []
    for idx in comp_indices:
        perm.extend([2 * idx, 2 * idx + 1])
    perm.append(len(popt) - 1)
    perm = np.array(perm, dtype=int)

    popt = popt[perm]
    cov = cov[np.ix_(perm, perm)]

    y_fit = multi_exp_model(x, *popt)

    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    dof = max(1, len(x) - len(popt))
    chi2red = ss_res / dof

    def model(x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        return multi_exp_model(x_new, *popt)

    return popt, cov, r2, chi2red, model


# ============================================================
# 5. OBJECT-ORIENTED HDF5 LOADER & CHANNEL HANDLES
# ============================================================

class HDF5ChannelHandle:
    """
    Represents a single channel (or cut slice of a channel) inside an HDF5TimestampData container.
    Supports method chaining via `.cut(start_idx, stop_idx)` or `.cut_time(t_start, t_stop)`.
    """
    def __init__(self, table_node, channel_id, start_idx=0, stop_idx=None):
        self.table = table_node
        self.channel_id = channel_id
        self.start_idx = start_idx
        self.stop_idx = len(table_node) if stop_idx is None else min(stop_idx, len(table_node))

    def __len__(self):
        return max(0, self.stop_idx - self.start_idx)

    def __repr__(self):
        return (
            f"<HDF5ChannelHandle channel={self.channel_id} "
            f"events={len(self):,} [{self.start_idx}:{self.stop_idx}]>"
        )

    def cut(self, start_idx=0, stop_idx=None):
        """
        Returns a new zero-copy `HDF5ChannelHandle` restricted to relative event indices
        `[start_idx, stop_idx)` within the current handle's range.
        """
        n = len(self)
        if start_idx is None:
            start_idx = 0
        elif start_idx < 0:
            start_idx = max(0, n + start_idx)
        else:
            start_idx = min(start_idx, n)

        if stop_idx is None:
            stop_idx = n
        elif stop_idx < 0:
            stop_idx = max(0, n + stop_idx)
        else:
            stop_idx = min(stop_idx, n)

        new_start = self.start_idx + start_idx
        new_stop = max(new_start, self.start_idx + stop_idx)
        return HDF5ChannelHandle(self.table, self.channel_id, new_start, new_stop)

    def cut_time(self, t_start=None, t_stop=None, time_resolution_sec=10e-9):
        """
        Returns a new zero-copy `HDF5ChannelHandle` restricted to relative seconds
        `[t_start, t_stop]` measured from the start of this channel handle.
        """
        if len(self) == 0:
            return self.cut(0, 0)

        t0_tick = int(self.table[self.start_idx]['timetag'])
        max_tick = int(self.table[self.stop_idx - 1]['timetag'])

        start_tick = t0_tick if t_start is None else max(0, t0_tick + int(round(t_start / time_resolution_sec)))
        stop_tick = max_tick + 1 if t_stop is None else max(0, t0_tick + int(round(t_stop / time_resolution_sec)))

        s_idx = _binary_search_timetag(self.table, np.uint64(start_tick), side="left")
        e_idx = _binary_search_timetag(self.table, np.uint64(stop_tick), side="left")

        s_idx = max(self.start_idx, min(s_idx, self.stop_idx))
        e_idx = max(s_idx, min(e_idx, self.stop_idx))
        return HDF5ChannelHandle(self.table, self.channel_id, s_idx, e_idx)

    def read_slice(self, start=0, stop=None):
        """Reads raw structured records relative to this handle's window."""
        abs_start = self.start_idx + start
        abs_stop = self.stop_idx if stop is None else min(self.start_idx + stop, self.stop_idx)
        return self.table[abs_start:abs_stop]

    def count_rate(self, bin_size_seconds=0.1, **kwargs):
        return calculate_count_rate(
            self.table,
            bin_size_seconds=bin_size_seconds,
            start_idx=self.start_idx,
            stop_idx=self.stop_idx,
            **kwargs,
        )

    def time_interval_distribution(self, **kwargs):
        return compute_time_interval_distribution(
            self.table,
            start_idx=self.start_idx,
            stop_idx=self.stop_idx,
            **kwargs,
        )

    def rossi_alpha(self, **kwargs):
        return compute_rossi_alpha(
            self.table,
            # h5_trigger=trigger,
            start_idx=self.start_idx,
            stop_idx=self.stop_idx,
            **kwargs,
        )


class HDF5TimestampData:
    """
    Object-oriented loader for converted CAEN HDF5 files.
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5 = tb.open_file(h5_path, mode="r")
        self.metadata = {
            name: self._h5.root._v_attrs[name]
            for name in self._h5.root._v_attrs._f_list("user")
        }
        self.start_measurement_time = self.metadata.get("start_measurement_time")

        self.channels = {}
        for node in self._h5.list_nodes("/", classname="Table"):
            if node._v_name.startswith("channel_"):
                ch_id = int(node._v_name.split("_")[1])
                self.channels[ch_id] = HDF5ChannelHandle(node, ch_id)

    @classmethod
    def from_bin(cls, bin_path, hdf5_path, chunk_size=40_000_000, mode_override=None, overwrite=False):
        if overwrite or not os.path.exists(hdf5_path):
            convert_bin_to_hdf5(
                bin_path, hdf5_path, chunk_size=chunk_size, mode_override=mode_override
            )
        return cls(hdf5_path)

    def __getitem__(self, ch_id):
        return self.channels[ch_id]

    def __repr__(self):
        ch_summary = ", ".join(f"{k}: {len(v):,}" for k, v in sorted(self.channels.items()))
        return f"<HDF5TimestampData '{self.h5_path}' | channels={{ {ch_summary} }}>"

    def close(self):
        if self._h5.isopen:
            self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# 6. CLI ENTRYPOINT & USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python caen_hdf5_pipeline.py <input_file.dat> <output_file.h5>")
        sys.exit(1)

    bin_file = sys.argv[1]
    h5_file = sys.argv[2]

    with HDF5TimestampData.from_bin(bin_file, h5_file) as data:
        print("\nLoaded dataset:", data)
        print("Measurement start:", data.start_measurement_time)

        channels = sorted(data.channels.keys())
        ch0 = data[channels[0]]

        # 1. Count rate (drops incomplete trailing bin by default)
        cr = ch0.count_rate(bin_size_seconds=0.5, absolute_time=True)
        print(f"\nChannel {channels[0]} Count Rate (head):\n", cr.head())

        # 2. Time Interval Distribution on a sliced event subset
        tid = ch0.cut(0, 1_000_000).time_interval_distribution(max_delta_ticks=5000, coarser_grouping=5)
        print(f"\nChannel {channels[0]} TID (head):\n", tid.head())

        # 3. Auto-correlation Rossi-Alpha + fit
        ra_auto = ch0.cut_time(0, 10.0).rossi_alpha(gate=1.0e-3, coarser_grouping=50)
        popt, cov, r2, chi2red, model = fit_rossi_alpha(ra_auto, n_exp=1)
        print(f"\nChannel {channels[0]} Auto Rossi-Alpha fit: alpha={-popt[1]:.2f} s^-1, R2={r2:.4f}, chi2red={chi2red:.2f}")

        # 4. Cross-correlation Rossi-Alpha (if at least 2 channels are present)
        if len(channels) > 1:
            ch1 = data[channels[1]]
            ra_cross = ch0.cut_time(0, 10.0).rossi_alpha(
                trigger=ch1.cut_time(0, 10.0),
                gate=1.0e-3,
                coarser_grouping=50,
            )
            popt_c, cov_c, r2_c, chi2red_c, model_c = fit_rossi_alpha(ra_cross, n_exp=1)
            print(f"Cross Rossi-Alpha ({channels[1]} -> {channels[0]}): alpha={-popt_c[1]:.2f} s^-1, R2={r2_c:.4f}, chi2red={chi2red_c:.2f}")