import os
from datetime import datetime
from typing import Callable, Optional, Tuple, Union

import numba
from numba import njit, prange
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tables as tb

_UINT64_MAX = np.uint64(np.iinfo(np.uint64).max)
_UINT64_MAX_LITERAL = np.uint64(0xFFFFFFFFFFFFFFFF)


# ============================================================
# 1. LOW-LEVEL NUMBA KERNELS (CACHE-SAFE, ZERO-COPY, THREAD-ISOLATED)
# ============================================================

@njit(inline="always")
def _bswap16(val: np.uint16) -> np.uint16:
    v = np.uint16(val)
    return np.uint16(((v & np.uint16(0x00FF)) << 8) | ((v & np.uint16(0xFF00)) >> 8))


@njit(inline="always")
def _bswap64(val: np.uint64) -> np.uint64:
    v = np.uint64(val)
    return (
        ((v & np.uint64(0x00000000000000FF)) << 56)
        | ((v & np.uint64(0x000000000000FF00)) << 40)
        | ((v & np.uint64(0x0000000000FF0000)) << 24)
        | ((v & np.uint64(0x00000000FF000000)) << 8)
        | ((v & np.uint64(0x000000FF00000000)) >> 8)
        | ((v & np.uint64(0x0000FF0000000000)) >> 24)
        | ((v & np.uint64(0x00FF000000000000)) >> 40)
        | ((v & np.uint64(0xFF00000000000000)) >> 56)
    )


@njit(cache=True)
def _count_channel_frequencies(raw_channels: np.ndarray, counts: np.ndarray) -> None:
    """O(N) single-pass scan counting events per big-endian channel ID."""
    for i in range(raw_channels.size):
        true_ch = _bswap16(raw_channels[i])
        counts[true_ch] += 1


@njit(cache=True)
def _extract_single_channel_linear(
    raw_timetags: np.ndarray,
    raw_channels: np.ndarray,
    raw_flags: np.ndarray,
    target_channel: int,
    out_timetags: np.ndarray,
    out_flags: np.ndarray,
    start_cursor: int,
) -> int:
    """Extracts and byte-swaps events for a single channel linearly into pre-allocated buffers."""
    cursor = start_cursor
    target = np.uint16(target_channel)
    for i in range(raw_channels.size):
        if _bswap16(raw_channels[i]) == target:
            out_timetags[cursor] = _bswap64(raw_timetags[i])
            out_flags[cursor] = _bswap16(raw_flags[i])
            cursor += 1
    return cursor


@njit(cache=True)
def _accumulate_count_rate(
    timetags: np.ndarray,
    flags: np.ndarray,
    filter_flags: bool,
    t0_ticks: np.uint64,
    dt_ticks: np.uint64,
    counts: np.ndarray,
) -> None:
    """Accumulates binned event counts into `counts` with zero intermediate array copies."""
    n_bins = counts.size
    for i in range(timetags.size):
        if filter_flags and flags[i] != np.uint16(0):
            continue
        t_curr = timetags[i]
        if t_curr >= t0_ticks:
            b = np.int64((t_curr - t0_ticks) // dt_ticks)
            if 0 <= b < n_bins:
                counts[b] += 1


@njit(parallel=True, cache=True)
def _fast_tid_core(
    timetags: np.ndarray,
    flags: np.ndarray,
    filter_flags: bool,
    start_tick: int,
    max_delta_ticks: int,
    coarser_grouping: int,
    num_bins: int,
    include_overflow: bool,
    n_threads: int,
) -> np.ndarray:
    """
    Thread-isolated parallel kernel for consecutive Time Interval Distribution (TID).
    Uses explicit `n_threads` partitioning so Numba caches cleanly without dynamic thread globals.
    """
    thread_histograms = np.zeros((n_threads, num_bins), dtype=np.int64)
    n = timetags.size
    if n < 2:
        return np.zeros(num_bins, dtype=np.int64)

    n_pairs = n - 1
    chunk_per_thread = (n_pairs + n_threads - 1) // n_threads
    start_bound = np.uint64(start_tick)
    max_bound = np.uint64(max_delta_ticks)
    group = np.uint64(coarser_grouping)

    for tid in prange(n_threads):
        i_start = tid * chunk_per_thread
        i_end = i_start + chunk_per_thread
        if i_end > n_pairs:
            i_end = n_pairs

        for i in range(i_start, i_end):
            if filter_flags and flags[i] != np.uint16(0):
                continue

            t0 = timetags[i]
            j = i + 1
            while j < n and (filter_flags and flags[j] != np.uint16(0)):
                j += 1
            if j >= n:
                continue

            t1 = timetags[j]
            if t1 < t0:
                continue

            dt = t1 - t0
            if dt >= start_bound:
                if dt < max_bound:
                    bin_idx = int((dt - start_bound) // group)
                    if 0 <= bin_idx < num_bins:
                        thread_histograms[tid, bin_idx] += 1
                elif include_overflow:
                    thread_histograms[tid, num_bins - 1] += 1

    global_counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        global_counts += thread_histograms[t]
    return global_counts


@njit(parallel=True, cache=True)
def _rossi_alpha_auto_core(
    timetags: np.ndarray,
    flags: np.ndarray,
    filter_flags: bool,
    active_elements: int,
    start_tick: int,
    gate_ticks: int,
    coarser_grouping: int,
    num_bins: int,
    n_threads: int,
) -> np.ndarray:
    """
    Thread-isolated parallel kernel for Auto-Correlation Rossi-Alpha.
    Uses strided trigger assignment (`range(tid, active_elements, n_threads)`) to balance bursts across cores.
    """
    thread_histograms = np.zeros((n_threads, num_bins), dtype=np.int64)
    n_total = timetags.size
    start_bound = np.uint64(start_tick)
    gate_bound = np.uint64(gate_ticks)
    group = np.uint64(coarser_grouping)

    for tid in prange(n_threads):
        for i in range(tid, active_elements, n_threads):
            if filter_flags and flags[i] != np.uint16(0):
                continue

            t_trigger = timetags[i]
            j = i + 1
            while j < n_total:
                if filter_flags and flags[j] != np.uint16(0):
                    j += 1
                    continue

                t_future = timetags[j]
                if t_future < t_trigger:
                    j += 1
                    continue

                dt = t_future - t_trigger
                if dt >= gate_bound:
                    break

                if dt >= start_bound:
                    bin_idx = int((dt - start_bound) // group)
                    if 0 <= bin_idx < num_bins:
                        thread_histograms[tid, bin_idx] += 1
                j += 1

    global_counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        global_counts += thread_histograms[t]
    return global_counts


@njit(parallel=True, cache=True)
def _rossi_alpha_cross_core(
    trigger_tags: np.ndarray,
    trigger_flags: np.ndarray,
    response_tags: np.ndarray,
    response_flags: np.ndarray,
    filter_flags: bool,
    start_tick: int,
    gate_ticks: int,
    coarser_grouping: int,
    num_bins: int,
    n_threads: int,
) -> np.ndarray:
    """
    Thread-isolated parallel kernel for Cross-Correlation Rossi-Alpha.
    Triggers on `trigger_tags` and histograms forward pulses in `response_tags`
    within `[start_tick, gate_ticks)`.
    """
    thread_histograms = np.zeros((n_threads, num_bins), dtype=np.int64)
    n_trig = trigger_tags.size
    n_resp = response_tags.size
    if n_trig == 0 or n_resp == 0:
        return np.zeros(num_bins, dtype=np.int64)

    start_bound = np.uint64(start_tick)
    gate_bound = np.uint64(gate_ticks)
    group = np.uint64(coarser_grouping)
    u64_max = np.uint64(0xFFFFFFFFFFFFFFFF)

    for tid in prange(n_threads):
        for i in range(tid, n_trig, n_threads):
            if filter_flags and trigger_flags[i] != np.uint16(0):
                continue

            t_trigger = trigger_tags[i]
            target = t_trigger + start_bound if (u64_max - t_trigger) >= start_bound else u64_max

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
                if filter_flags and response_flags[j] != np.uint16(0):
                    j += 1
                    continue

                t_future = response_tags[j]
                if t_future < t_trigger:
                    j += 1
                    continue

                dt = t_future - t_trigger
                if dt >= gate_bound:
                    break

                if dt >= start_bound:
                    bin_idx = int((dt - start_bound) // group)
                    if 0 <= bin_idx < num_bins:
                        thread_histograms[tid, bin_idx] += 1
                j += 1

    global_counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        global_counts += thread_histograms[t]
    return global_counts


# ============================================================
# 2. BINARY CONVERSION & INDEXING HELPERS
# ============================================================

def _extract_file_start_time(filepath: str) -> datetime:
    """Extracts earliest filesystem timestamp (mtime / ctime / birthtime) as start time."""
    try:
        st = os.stat(filepath)
        ts = min(st.st_mtime, getattr(st, "st_birthtime", st.st_ctime))
    except AttributeError:
        ts = min(os.path.getmtime(filepath), os.path.getctime(filepath))
    return datetime.fromtimestamp(ts)


def _resolve_table_and_bounds(
    node_or_handle,
    start_idx: Optional[int] = None,
    stop_idx: Optional[int] = None,
) -> Tuple[tb.Table, int, int]:
    """Normalizes an HDF5ChannelHandle or PyTables Table into (table, start_idx, stop_idx)."""
    if isinstance(node_or_handle, HDF5ChannelHandle):
        table = node_or_handle.table
        s0 = node_or_handle.start_idx if start_idx is None else start_idx
        s1 = node_or_handle.stop_idx if stop_idx is None else stop_idx
    else:
        table = node_or_handle
        s0 = 0 if start_idx is None else start_idx
        s1 = len(table) if stop_idx is None else stop_idx

    total_rows = len(table)
    s0 = max(0, min(int(s0), total_rows))
    s1 = max(s0, min(int(s1), total_rows))
    return table, s0, s1


def _binary_search_timetag(
    table_node: tb.Table,
    target_tick: int,
    start_idx: int = 0,
    stop_idx: Optional[int] = None,
    side: str = "left",
) -> int:
    """O(log N) binary search on an on-disk PyTables 'timetag' column within [start_idx, stop_idx)."""
    lo = int(start_idx)
    hi = len(table_node) if stop_idx is None else int(stop_idx)
    target = np.uint64(target_tick)

    while lo < hi:
        mid = (lo + hi) // 2
        val = np.uint64(table_node[mid]["timetag"])
        cond = (val < target) if side == "left" else (val <= target)
        if cond:
            lo = mid + 1
        else:
            hi = mid
    return lo


def convert_bin_to_hdf5(
    filepath: str,
    h5_output_path: str,
    chunk_size: int = 100_000_000,
    mode_override: Optional[int] = None,
    complevel: int = 1,
    complib: str = "blosc:lz4",
) -> None:
    """
    Memory-mapped O(1) header + chunked conversion of raw CAEN timestamp binary files
    into a compressed PyTables HDF5 file with one table per active channel.
    """
    start_time = _extract_file_start_time(filepath)

    header_dtype = np.dtype([
        ("op_mode", "u1"),
        ("cycles", ">u4"),
        ("cycle_len", ">u8"),
        ("agg_mode", "u1"),
        ("chan_group_num", "u1"),
        ("chan_mask", ">u4"),
        ("group_mask", ">u2"),
        ("group_size", "u1"),
    ])

    m = np.memmap(filepath, dtype="uint8", mode="r")
    if m.size < 22:
        raise ValueError(f"File '{filepath}' is too small ({m.size} bytes) to contain a valid 22-byte header.")

    header_raw = m[:22].view(header_dtype)[0]
    agg_mode = int(mode_override) if mode_override is not None else int(header_raw["agg_mode"])
    has_channels = (agg_mode == 0)

    if has_channels:
        event_dtype = np.dtype([("timetag", ">u8"), ("channel", ">u2"), ("flags", ">u2")])
    else:
        event_dtype = np.dtype([("timetag", ">u8"), ("flags", ">u2")])

    data_bytes = m.size - 22
    num_records = data_bytes // event_dtype.itemsize
    if num_records == 0:
        raise ValueError(f"No complete event records found in '{filepath}'.")

    data_view = m[22 : 22 + num_records * event_dtype.itemsize].view(event_dtype)

    class ChannelEventRow(tb.IsDescription):
        timetag = tb.UInt64Col(pos=0)
        flags = tb.UInt16Col(pos=1)

    filters = tb.Filters(complevel=complevel, complib=complib)
    out_dir = os.path.dirname(os.path.abspath(h5_output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with tb.open_file(h5_output_path, mode="w", title="CAEN Timestamp Data") as h5file:
        attrs = h5file.root._v_attrs
        attrs.start_measurement_time = start_time.isoformat()
        attrs.source_file = os.path.abspath(filepath)
        attrs.op_mode = int(header_raw["op_mode"])
        attrs.cycles = int(header_raw["cycles"])
        attrs.cycle_len = int(header_raw["cycle_len"])
        attrs.agg_mode = agg_mode
        attrs.chan_group_num = int(header_raw["chan_group_num"])
        attrs.chan_mask = int(header_raw["chan_mask"])
        attrs.group_mask = int(header_raw["group_mask"])
        attrs.group_size = int(header_raw["group_size"])
        attrs.total_raw_records = int(num_records)

        h5_tables = {}
        expected_rows_per_ch = max(1, num_records // 4)

        for start in range(0, num_records, chunk_size):
            end = min(start + chunk_size, num_records)
            chunk = data_view[start:end]

            if has_channels:
                raw_t = chunk["timetag"].view("uint64")
                raw_c = chunk["channel"].view("uint16")
                raw_f = chunk["flags"].view("uint16")

                ch_freqs = np.zeros(65536, dtype=np.int64)
                _count_channel_frequencies(raw_c, ch_freqs)
                active_chs = np.flatnonzero(ch_freqs)

                for ch in active_chs:
                    ch_int = int(ch)
                    cnt = int(ch_freqs[ch_int])
                    if ch_int not in h5_tables:
                        h5_tables[ch_int] = h5file.create_table(
                            "/",
                            f"channel_{ch_int}",
                            ChannelEventRow,
                            f"Timestamp Data Channel {ch_int}",
                            filters=filters,
                            expectedrows=expected_rows_per_ch,
                        )

                    out_t = np.empty(cnt, dtype=np.uint64)
                    out_f = np.empty(cnt, dtype=np.uint16)
                    _extract_single_channel_linear(raw_t, raw_c, raw_f, ch_int, out_t, out_f, 0)

                    rec_arr = np.empty(cnt, dtype=[("timetag", "<u8"), ("flags", "<u2")])
                    rec_arr["timetag"] = out_t
                    rec_arr["flags"] = out_f
                    h5_tables[ch_int].append(rec_arr)
            else:
                ch_int = 0
                if ch_int not in h5_tables:
                    h5_tables[ch_int] = h5file.create_table(
                        "/",
                        f"channel_{ch_int}",
                        ChannelEventRow,
                        "Timestamp Data Channel 0",
                        filters=filters,
                        expectedrows=num_records,
                    )
                cnt = chunk.size
                rec_arr = np.empty(cnt, dtype=[("timetag", "<u8"), ("flags", "<u2")])
                rec_arr["timetag"] = chunk["timetag"].byteswap().newbyteorder("<")
                rec_arr["flags"] = chunk["flags"].byteswap().newbyteorder("<")
                h5_tables[ch_int].append(rec_arr)

        for tbl in h5_tables.values():
            tbl.flush()

    del m


# ============================================================
# 3. CORE ANALYSIS FUNCTIONS (COUNT RATE, TID, ROSSI-ALPHA)
# ============================================================

def calculate_count_rate(
    h5_channel,
    bin_size_seconds: float = 0.1,
    time_resolution_sec: float = 10e-9,
    filter_flags: bool = True,
    absolute_time: bool = False,
    start_measurement_time: Optional[Union[str, datetime]] = None,
    drop_zeros: bool = False,
    chunk_size: int = 100_000_000,
    start_idx: Optional[int] = None,
    stop_idx: Optional[int] = None,
) -> pd.Series:
    """
    Computes count rate (events/s) over time bins for `[start_idx, stop_idx)`.
    Returns a `pd.Series` indexed by relative time (seconds) or wall-clock `datetime64`.
    """
    table, s0, s1 = _resolve_table_and_bounds(h5_channel, start_idx, stop_idx)
    if s1 <= s0:
        idx = pd.DatetimeIndex([], name="time") if absolute_time else pd.Index([], dtype=float, name="time")
        return pd.Series([], dtype=np.float64, index=idx, name="count_rate")

    dt_ticks = max(1, int(round(bin_size_seconds / time_resolution_sec)))
    actual_bin_sec = dt_ticks * time_resolution_sec

    first_tags = table.read(s0, min(s0 + 1000, s1), field="timetag")
    first_flags = table.read(s0, min(s0 + 1000, s1), field="flags") if filter_flags else None
    if filter_flags:
        valid_first = first_tags[first_flags == 0]
        t0_ticks = np.uint64(valid_first[0] if valid_first.size > 0 else first_tags[0])
    else:
        t0_ticks = np.uint64(first_tags[0])

    tail_start = max(s0, s1 - 1000)
    last_tags = table.read(tail_start, s1, field="timetag")
    last_flags = table.read(tail_start, s1, field="flags") if filter_flags else None
    if filter_flags:
        valid_last = last_tags[last_flags == 0]
        t_max_ticks = np.uint64(valid_last[-1] if valid_last.size > 0 else last_tags[-1])
    else:
        t_max_ticks = np.uint64(last_tags[-1])

    if t_max_ticks < t0_ticks:
        t_max_ticks = t0_ticks

    num_bins = int((t_max_ticks - t0_ticks) // np.uint64(dt_ticks)) + 1
    counts = np.zeros(num_bins, dtype=np.int64)
    dt_ticks_u64 = np.uint64(dt_ticks)
    empty_flags = np.empty(0, dtype=np.uint16)

    for start in range(s0, s1, chunk_size):
        end = min(start + chunk_size, s1)
        chunk_tags = table.read(start, end, field="timetag")
        chunk_flags = table.read(start, end, field="flags") if filter_flags else empty_flags
        _accumulate_count_rate(chunk_tags, chunk_flags, filter_flags, t0_ticks, dt_ticks_u64, counts)

    rel_times = np.arange(num_bins, dtype=np.float64) * actual_bin_sec
    rates = counts.astype(np.float64) / actual_bin_sec

    if absolute_time:
        base_dt = start_measurement_time
        if base_dt is None and isinstance(h5_channel, HDF5ChannelHandle):
            base_dt = h5_channel.start_measurement_time
        if base_dt is None:
            try:
                base_dt = table._v_file.root._v_attrs["start_measurement_time"]
            except Exception as exc:
                raise ValueError(
                    "absolute_time=True requires `start_measurement_time` or an HDF5 attribute."
                ) from exc
        if isinstance(base_dt, str):
            base_dt = datetime.fromisoformat(base_dt)
        base_ns = np.datetime64(base_dt, "ns")
        t0_offset_ns = np.timedelta64(int(round(float(t0_ticks) * time_resolution_sec * 1e9)), "ns")
        bin_offsets_ns = (rel_times * 1e9).astype("timedelta64[ns]")
        time_index = pd.DatetimeIndex(base_ns + t0_offset_ns + bin_offsets_ns, name="time")
    else:
        time_index = pd.Index(rel_times, dtype=np.float64, name="time")

    series = pd.Series(rates, index=time_index, name="count_rate")
    if drop_zeros:
        series = series[series > 0]
    return series


def compute_time_interval_distribution(
    h5_channel,
    max_delta_ticks: int = 10_000,
    coarser_grouping: int = 1,
    start_tick: int = 0,
    time_resolution_sec: float = 10e-9,
    filter_flags: bool = True,
    include_overflow: bool = False,
    drop_zeros: bool = False,
    chunk_size: int = 100_000_000,
    start_idx: Optional[int] = None,
    stop_idx: Optional[int] = None,
) -> pd.Series:
    """
    Computes consecutive inter-arrival Time Interval Distribution (TID) over `[start_idx, stop_idx)`.
    Returns a `pd.Series` indexed by interval in ticks.
    """
    if coarser_grouping < 1:
        raise ValueError("`coarser_grouping` must be >= 1.")
    if max_delta_ticks <= start_tick:
        raise ValueError("`max_delta_ticks` must be greater than `start_tick`.")

    table, s0, s1 = _resolve_table_and_bounds(h5_channel, start_idx, stop_idx)
    num_bins = (max_delta_ticks - start_tick + coarser_grouping - 1) // coarser_grouping
    if include_overflow:
        num_bins += 1

    total_counts = np.zeros(num_bins, dtype=np.int64)
    if s1 - s0 < 2:
        bin_edges_ticks = np.arange(start_tick, start_tick + num_bins * coarser_grouping, coarser_grouping, dtype=np.int64)
        idx = pd.Index(bin_edges_ticks[:num_bins], name="time_interval")
        return pd.Series(total_counts, index=idx, name="time_interval_distribution")

    n_threads = numba.get_num_threads()
    prev_last_tag: Optional[np.uint64] = None
    prev_last_flag: np.uint16 = np.uint16(0)
    empty_flags = np.empty(0, dtype=np.uint16)

    for start in range(s0, s1, chunk_size):
        end = min(start + chunk_size, s1)
        chunk_tags = table.read(start, end, field="timetag")
        chunk_flags = table.read(start, end, field="flags") if filter_flags else empty_flags

        if prev_last_tag is not None and chunk_tags.size > 0:
            boundary_tags = np.array([prev_last_tag, chunk_tags[0]], dtype=np.uint64)
            boundary_flags = (
                np.array([prev_last_flag, chunk_flags[0]], dtype=np.uint16)
                if filter_flags
                else empty_flags
            )
            total_counts += _fast_tid_core(
                boundary_tags,
                boundary_flags,
                filter_flags,
                start_tick,
                max_delta_ticks,
                coarser_grouping,
                num_bins,
                include_overflow,
                1,
            )

        total_counts += _fast_tid_core(
            chunk_tags,
            chunk_flags,
            filter_flags,
            start_tick,
            max_delta_ticks,
            coarser_grouping,
            num_bins,
            include_overflow,
            n_threads,
        )

        if chunk_tags.size > 0:
            prev_last_tag = np.uint64(chunk_tags[-1])
            if filter_flags:
                prev_last_flag = np.uint16(chunk_flags[-1])

    bin_ticks = start_tick + np.arange(num_bins, dtype=np.int64) * coarser_grouping
    series = pd.Series(total_counts, index=pd.Index(bin_ticks, name="time_interval"), name="time_interval_distribution")
    if drop_zeros:
        series = series[series > 0]
    return series


def compute_rossi_alpha(
    h5_channel,
    h5_trigger=None,
    gate: float = 50e-6,
    coarser_grouping: int = 1,
    start_tick: int = 0,
    time_resolution_sec: float = 10e-9,
    filter_flags: bool = True,
    chunk_size: int = 100_000_000,
    start_idx: Optional[int] = None,
    stop_idx: Optional[int] = None,
    trigger_start_idx: Optional[int] = None,
    trigger_stop_idx: Optional[int] = None,
) -> pd.Series:
    """
    Computes triggered forward-window Rossi-Alpha distribution:
    - `h5_trigger is None`: auto-correlation on `h5_channel` over `[start_idx, stop_idx)`.
    - `h5_trigger is not None`: cross-correlation triggering on `h5_trigger` over
      `[trigger_start_idx, trigger_stop_idx)` against follower events in `h5_channel`
      over `[start_idx, stop_idx)`.

    Returns a `pd.Series` indexed by time lag in seconds.
    """
    if coarser_grouping < 1:
        raise ValueError("`coarser_grouping` must be >= 1.")

    gate_ticks = int(round(gate / time_resolution_sec))
    if gate_ticks <= start_tick:
        raise ValueError("`gate` must be larger than `start_tick * time_resolution_sec`.")

    num_bins = (gate_ticks - start_tick + coarser_grouping - 1) // coarser_grouping
    bin_ticks = start_tick + np.arange(num_bins, dtype=np.int64) * coarser_grouping
    bin_seconds = bin_ticks.astype(np.float64) * time_resolution_sec

    resp_table, r_s0, r_s1 = _resolve_table_and_bounds(h5_channel, start_idx, stop_idx)
    if r_s1 <= r_s0:
        return pd.Series(
            np.zeros(num_bins, dtype=np.int64),
            index=pd.Index(bin_seconds, name="time_interval"),
            name="rossi_alpha_distribution",
        )

    n_threads = numba.get_num_threads()
    rossi_counts = np.zeros(num_bins, dtype=np.int64)
    gate_ticks_u64 = np.uint64(gate_ticks)
    empty_flags = np.empty(0, dtype=np.uint16)

    if h5_trigger is None:
        pad_step = 10_000_000
        for start in range(r_s0, r_s1, chunk_size):
            end = min(start + chunk_size, r_s1)
            chunk_tags = resp_table.read(start, end, field="timetag")
            if chunk_tags.size == 0:
                continue
            chunk_flags = resp_table.read(start, end, field="flags") if filter_flags else empty_flags
            active_elements = int(chunk_tags.size)

            last_t = np.uint64(chunk_tags[-1])
            cutoff_tick = last_t + gate_ticks_u64 if (_UINT64_MAX - last_t) >= gate_ticks_u64 else _UINT64_MAX

            pad_start = end
            pad_blocks = []
            while pad_start < r_s1:
                pad_end = min(pad_start + pad_step, r_s1)
                p_tags = resp_table.read(pad_start, pad_end, field="timetag")
                if p_tags.size == 0:
                    break
                p_flags = resp_table.read(pad_start, pad_end, field="flags") if filter_flags else empty_flags
                idx_cut = int(np.searchsorted(p_tags, cutoff_tick, side="right"))
                pad_blocks.append((p_tags[:idx_cut], p_flags[:idx_cut] if filter_flags else empty_flags))
                pad_start = pad_end
                if idx_cut < p_tags.size:
                    break

            if pad_blocks:
                extended_tags = np.concatenate([chunk_tags] + [b[0] for b in pad_blocks])
                extended_flags = (
                    np.concatenate([chunk_flags] + [b[1] for b in pad_blocks])
                    if filter_flags
                    else empty_flags
                )
            else:
                extended_tags = chunk_tags
                extended_flags = chunk_flags

            rossi_counts += _rossi_alpha_auto_core(
                extended_tags,
                extended_flags,
                filter_flags,
                active_elements,
                start_tick,
                gate_ticks,
                coarser_grouping,
                num_bins,
                n_threads,
            )
    else:
        trig_table, t_s0, t_s1 = _resolve_table_and_bounds(h5_trigger, trigger_start_idx, trigger_stop_idx)
        if t_s1 <= t_s0:
            return pd.Series(
                np.zeros(num_bins, dtype=np.int64),
                index=pd.Index(bin_seconds, name="time_interval"),
                name="rossi_alpha_distribution",
            )

        for start in range(t_s0, t_s1, chunk_size):
            end = min(start + chunk_size, t_s1)
            trig_tags = trig_table.read(start, end, field="timetag")
            if trig_tags.size == 0:
                continue
            trig_flags = trig_table.read(start, end, field="flags") if filter_flags else empty_flags

            t_min = np.uint64(trig_tags[0])
            last_trig = np.uint64(trig_tags[-1])
            t_max = last_trig + gate_ticks_u64 if (_UINT64_MAX - last_trig) >= gate_ticks_u64 else _UINT64_MAX

            resp_lo = _binary_search_timetag(resp_table, int(t_min), r_s0, r_s1, side="left")
            resp_hi = _binary_search_timetag(resp_table, int(t_max), resp_lo, r_s1, side="right")

            if resp_hi > resp_lo:
                resp_tags = resp_table.read(resp_lo, resp_hi, field="timetag")
                resp_flags = resp_table.read(resp_lo, resp_hi, field="flags") if filter_flags else empty_flags
            else:
                resp_tags = np.empty(0, dtype=np.uint64)
                resp_flags = empty_flags

            rossi_counts += _rossi_alpha_cross_core(
                trig_tags,
                trig_flags,
                resp_tags,
                resp_flags,
                filter_flags,
                start_tick,
                gate_ticks,
                coarser_grouping,
                num_bins,
                n_threads,
            )

    return pd.Series(rossi_counts, index=pd.Index(bin_seconds, name="time_interval"), name="rossi_alpha_distribution")


# ============================================================
# 4. MULTI-EXPONENTIAL ROSSI-ALPHA FITTER (USER-SPECIFIED n_exp)
# ============================================================

def fit_rossi_alpha(
    x: np.ndarray,
    y: np.ndarray,
    n_exp: int = 1,
    sigma: Optional[np.ndarray] = None,
    maxfev: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, float, float, Callable[[np.ndarray], np.ndarray]]:
    """
    Fits `y(x) = sum_{k=1}^{n_exp} A_k * exp(B_k * x) + C` with user-specified `n_exp`.
    Initializes `C` directly from `np.min(y)`.

    Returns:
        (popt, cov, r2, chi2red, model_fn)
    """
    if n_exp < 1:
        raise ValueError("`n_exp` must be >= 1.")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if sigma is not None:
        s_arr = np.asarray(sigma, dtype=np.float64)
        mask &= np.isfinite(s_arr) & (s_arr > 0)
        s_clean = s_arr[mask]
    else:
        s_clean = None

    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    if x_clean.size < (2 * n_exp + 2):
        raise ValueError("Not enough valid data points to fit the requested number of exponentials.")

    order = np.argsort(x_clean)
    x_clean = x_clean[order]
    y_clean = y_clean[order]
    if s_clean is not None:
        s_clean = s_clean[order]

    def multi_exp_model(x_val: np.ndarray, *params: float) -> np.ndarray:
        c_val = params[-1]
        out = np.full_like(x_val, c_val, dtype=np.float64)
        for k in range(n_exp):
            a_k = params[2 * k]
            b_k = params[2 * k + 1]
            out += a_k * np.exp(b_k * x_val)
        return out

    c_init = float(np.min(y_clean))
    span_x = max(float(x_clean[-1] - x_clean[0]), 1e-12)
    amp_total = max(float(np.max(y_clean) - c_init), 1.0)
    a_each = amp_total / n_exp

    b_fast = -5.0 / span_x
    b_slow = -0.2 / span_x
    if n_exp == 1:
        b_inits = [-1.0 / span_x]
    else:
        b_inits = list(-np.geomspace(abs(b_slow), abs(b_fast), n_exp))

    p0 = []
    lower_bounds = []
    upper_bounds = []
    for b_0 in b_inits:
        p0.extend([a_each, b_0])
        lower_bounds.extend([-np.inf, -np.inf])
        upper_bounds.extend([np.inf, 0.0])

    p0.append(c_init)
    lower_bounds.append(-np.inf)
    upper_bounds.append(np.inf)

    popt, cov = curve_fit(
        multi_exp_model,
        x_clean,
        y_clean,
        p0=np.asarray(p0, dtype=np.float64),
        sigma=s_clean,
        absolute_sigma=(s_clean is not None),
        bounds=(np.asarray(lower_bounds, dtype=np.float64), np.asarray(upper_bounds, dtype=np.float64)),
        maxfev=maxfev,
    )

    y_pred = multi_exp_model(x_clean, *popt)
    residuals = y_clean - y_pred

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    dof = max(1, x_clean.size - len(popt))
    if s_clean is not None:
        chi2 = float(np.sum((residuals / s_clean) ** 2))
    else:
        chi2 = ss_res
    chi2red = chi2 / dof

    def model_fn(x_new: np.ndarray) -> np.ndarray:
        return multi_exp_model(np.asarray(x_new, dtype=np.float64), *popt)

    return popt, cov, r2, chi2red, model_fn


# ============================================================
# 5. OBJECT-ORIENTED HDF5 CONTAINER & LAZY .cut() SLICING
# ============================================================

class HDF5ChannelHandle:
    """
    Lightweight handle for a single channel table inside an HDF5 file.
    `cut(start_idx, stop_idx)` is lazy (zero-copy) and returns a bounded view.
    """

    def __init__(
        self,
        table_node: tb.Table,
        channel_id: int,
        start_idx: int = 0,
        stop_idx: Optional[int] = None,
        start_measurement_time: Optional[str] = None,
    ):
        self.table = table_node
        self.channel_id = int(channel_id)
        total = len(table_node)
        self.start_idx = max(0, min(int(start_idx), total))
        self.stop_idx = total if stop_idx is None else max(self.start_idx, min(int(stop_idx), total))
        self.start_measurement_time = start_measurement_time

    def __len__(self) -> int:
        return max(0, self.stop_idx - self.start_idx)

    def cut(self, start_idx: int = 0, stop_idx: Optional[int] = None) -> "HDF5ChannelHandle":
        """
        Returns a new HDF5ChannelHandle with row bounds relative to this handle's current view.
        Does not copy or load table data into memory.
        """
        new_start = self.start_idx + max(0, int(start_idx))
        if stop_idx is None:
            new_stop = self.stop_idx
        else:
            new_stop = min(self.stop_idx, self.start_idx + max(0, int(stop_idx)))
        return HDF5ChannelHandle(
            self.table,
            self.channel_id,
            start_idx=new_start,
            stop_idx=new_stop,
            start_measurement_time=self.start_measurement_time,
        )

    def read_slice(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        """Reads a slice relative to `[self.start_idx, self.stop_idx)` as a NumPy structured array."""
        abs_start = self.start_idx + max(0, int(start))
        abs_stop = self.stop_idx if stop is None else min(self.stop_idx, self.start_idx + max(0, int(stop)))
        if abs_stop <= abs_start:
            return np.empty(0, dtype=self.table.dtype)
        return self.table.read(abs_start, abs_stop)

    def count_rate(self, bin_size_seconds: float = 0.1, **kwargs) -> pd.Series:
        return calculate_count_rate(
            self.table,
            bin_size_seconds=bin_size_seconds,
            start_measurement_time=self.start_measurement_time,
            start_idx=self.start_idx,
            stop_idx=self.stop_idx,
            **kwargs,
        )

    def time_interval_distribution(self, **kwargs) -> pd.Series:
        return compute_time_interval_distribution(
            self.table,
            start_idx=self.start_idx,
            stop_idx=self.stop_idx,
            **kwargs,
        )

    def rossi_alpha(self, **kwargs) -> pd.Series:
        return compute_rossi_alpha(
            self.table,
            start_idx=self.start_idx,
            stop_idx=self.stop_idx,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"<HDF5ChannelHandle ch={self.channel_id} | rows=[{self.start_idx}, {self.stop_idx}) "
            f"({len(self):,} events)>"
        )


class HDF5TimestampData:
    """
    Context-managed wrapper around a converted CAEN timestamp HDF5 file.
    Provides direct integer channel indexing (`data[ch_id]`) and lazy slicing methods.
    """

    def __init__(self, h5_path: str):
        self.h5_path = str(h5_path)
        self._h5 = tb.open_file(self.h5_path, mode="r")
        user_attrs = self._h5.root._v_attrs._f_list("user")
        self.metadata = {name: self._h5.root._v_attrs[name] for name in user_attrs}
        self.start_measurement_time = self.metadata.get("start_measurement_time")

        self.channels = {}
        for node in self._h5.list_nodes("/", classname="Table"):
            if node._v_name.startswith("channel_"):
                ch_id = int(node._v_name.split("_")[1])
                self.channels[ch_id] = HDF5ChannelHandle(
                    node,
                    ch_id,
                    start_measurement_time=self.start_measurement_time,
                )

    @classmethod
    def from_bin(
        cls,
        bin_path: str,
        h5_path: Optional[str] = None,
        overwrite: bool = False,
        **convert_kwargs,
    ) -> "HDF5TimestampData":
        """Converts `bin_path` to `h5_path` if missing or `overwrite=True`, then opens it."""
        base, _ = os.path.splitext(bin_path)
        target_h5 = h5_path or f"{base}.h5"
        if overwrite or not os.path.exists(target_h5):
            convert_bin_to_hdf5(bin_path, target_h5, **convert_kwargs)
        return cls(target_h5)

    def cut(self, start_idx: int = 0, stop_idx: Optional[int] = None) -> dict:
        """Returns a `{ch_id: HDF5ChannelHandle}` mapping sliced to `[start_idx, stop_idx)` per channel."""
        return {ch_id: handle.cut(start_idx, stop_idx) for ch_id, handle in self.channels.items()}

    def slice_time(
        self,
        t_start_sec: float,
        t_stop_sec: float,
        time_resolution_sec: float = 10e-9,
    ) -> dict:
        """
        Slices each channel to `[t_start_sec, t_stop_sec)` relative to that channel's first timestamp
        using O(log N) binary search on the HDF5 table.
        """
        sliced = {}
        for ch_id, handle in self.channels.items():
            if len(handle) == 0:
                sliced[ch_id] = handle.cut(0, 0)
                continue
            first_tick = int(handle.table[handle.start_idx]["timetag"])
            tick_lo = first_tick + int(round(t_start_sec / time_resolution_sec))
            tick_hi = first_tick + int(round(t_stop_sec / time_resolution_sec))
            idx_lo = _binary_search_timetag(handle.table, tick_lo, handle.start_idx, handle.stop_idx, side="left")
            idx_hi = _binary_search_timetag(handle.table, tick_hi, idx_lo, handle.stop_idx, side="right")
            sliced[ch_id] = HDF5ChannelHandle(
                handle.table,
                ch_id,
                start_idx=idx_lo,
                stop_idx=idx_hi,
                start_measurement_time=handle.start_measurement_time,
            )
        return sliced

    def __getitem__(self, ch_id: int) -> HDF5ChannelHandle:
        return self.channels[ch_id]

    def close(self) -> None:
        if self._h5.isopen:
            self._h5.close()

    def __enter__(self) -> "HDF5TimestampData":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        ch_summary = {k: len(v) for k, v in self.channels.items()}
        return f"<HDF5TimestampData '{self.h5_path}' | channels={ch_summary}>"