import os
import sys
import time
import pickle
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless Linux/Windows CLI execution
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def _bootstrap_lib_path() -> None:
    """Adds --lib-dir (if passed via CLI) or default network path before local imports."""
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--lib-dir" and i + 1 < len(sys.argv):
            sys.path.insert(0, str(Path(sys.argv[i + 1]).expanduser().resolve()))
            return
        if arg.startswith("--lib-dir="):
            sys.path.insert(0, str(Path(arg.split("=", 1)[1]).expanduser().resolve()))
            return
    sys.path.append(r"\\sckstor9\NMS_DATA\VENUS-F\Knowledge\PhD\DiCroceFederico\PROCESSING")


_bootstrap_lib_path()

from DetectorSetup_utils import DetectorChannel, ExperimentSetup
import TID_analysis as tid
import TimeStampData_hdf5 as tsd


# ============================================================
# 1. HELPER, FITTING & PLOTTING FUNCTIONS
# ============================================================

def simulate_chain(events: np.ndarray, arrangement: str, popt: np.ndarray) -> np.ndarray:
    """Applies single or series dead-time models to a raw event timestamp array."""
    arr = arrangement.upper().strip()
    if arr == "NE":
        return tid.apply_non_paralyzable(events, popt[1])
    if arr == "E":
        return tid.apply_paralyzable(events, popt[1])
    if arr == "NE-E":
        return tid.apply_paralyzable(tid.apply_non_paralyzable(events, popt[1]), popt[2])
    if arr == "E-NE":
        return tid.apply_non_paralyzable(tid.apply_paralyzable(events, popt[1]), popt[2])
    if arr == "NE-NE":
        return tid.apply_non_paralyzable(tid.apply_non_paralyzable(events, popt[1]), popt[2])
    if arr == "E-E":
        return tid.apply_paralyzable(tid.apply_paralyzable(events, popt[1]), popt[2])
    raise ValueError(f"Unknown dead-time arrangement: '{arrangement}'")


def slice_rossi_gate(
    ra_max: pd.Series,
    gate_sec: float,
    coarser_grouping: int = 1,
) -> pd.Series:
    """
    Extracts a shorter gate (and optional coarser rebinning) from a precomputed
    max-gate Rossi-alpha Series in memory without re-reading the HDF5 file.
    """
    sub = ra_max[ra_max.index < gate_sec - 1e-13]

    if coarser_grouping > 1:
        n_coarse = len(sub) // coarser_grouping
        trimmed = sub.iloc[: n_coarse * coarser_grouping]
        new_index = trimmed.index.to_numpy()[::coarser_grouping]
        new_counts = trimmed.to_numpy().reshape(n_coarse, coarser_grouping).sum(axis=1)
        sub = pd.Series(new_counts, index=new_index, name=ra_max.name)
        sub.index.name = ra_max.index.name

    return sub


def fit_multi_exp(x: np.ndarray, y: np.ndarray, n_exp: int = 1, p0=None, bounds=(0, np.inf)):
    """
    Fits y = c + sum_{i=1}^{n_exp} A_i * exp(-lambda_i * x).

    Returns:
        popt: Fitted parameters [c, A1, lambda1, A2, lambda2, ...]
        pcov: Covariance matrix of parameters
        y_fit: Evaluated model at x
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def model(x_val, *params):
        res = np.full_like(x_val, params[0], dtype=float)
        for i in range(n_exp):
            amp = params[1 + 2 * i]
            lam = params[2 + 2 * i]
            res += amp * np.exp(-lam * x_val)
        return res

    if p0 is None:
        c_guess = max(float(np.min(y)), 0.0)
        amp_total = max(float(np.max(y)) - c_guess, 1e-6)
        x_span = max(float(np.max(x) - np.min(x)), 1e-6)

        p0 = [c_guess]
        for i in range(n_exp):
            p0.append(amp_total / n_exp)
            p0.append((i + 1) * (2.0 / x_span))

    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
    y_fit = model(x, *popt)
    return popt, pcov, y_fit


def plot_countrate_evolution(countrate: pd.Series, channel=None, ax=None):
    """Plots normalized count rate evolution (rate / mean) with mean & std annotations."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(dpi=200, figsize=(7, 3))

    mean_val = float(countrate.mean())
    std_val = float(countrate.std())
    norm_rate = countrate / mean_val

    norm_rate.plot(ax=ax, color="steelblue", lw=0.9, label="Normalized Rate")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2, label="Mean (1.0)")
    ax.grid(False)
    ax.set_xlim(countrate.index.min(), countrate.index.max())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count Rate / Mean")

    title_prefix = f"Channel {channel} — " if channel is not None else ""
    ax.set_title(f"{title_prefix}Normalized Count Rate Evolution")

    stats_text = f"Mean = {mean_val:.2f} cps\nStd = {std_val:.2f} cps ({std_val / mean_val * 100:.2f}%)"
    ax.text(
        0.98,
        0.93,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    if created_fig:
        plt.tight_layout()

    return ax


def plot_tid_full_and_closeup(
    tid_,
    popt,
    arrangement: str = "NE-E",
    channel=None,
    duration: float = None,
    axes=None,
    logy_full: bool = False,
):
    """
    Plots a 2-column comparison of the single-channel TID PDF:
    - Left:  Full inter-arrival time distribution (bounded by max TID interval)
    - Right: Close-up up to x_max = 3 * (tau_1 + tau_2) with right-aligned y-axis
    """
    popt = np.asarray(popt, dtype=float)

    if isinstance(tid_, pd.Series):
        df = pd.DataFrame({
            "interval": tid_.index.to_numpy(dtype=float),
            "time_interval_distribution": tid_.to_numpy(dtype=float),
        })
    else:
        df = tid_.copy()
        if "channel" in df.columns:
            if channel is None:
                unique_chs = df["channel"].unique()
                if len(unique_chs) > 1:
                    raise ValueError(
                        f"`tid_` contains multiple channels ({list(unique_chs)}). Pass `channel=...`."
                    )
                channel = unique_chs[0]
            df = df[df["channel"] == channel].copy()

    df = df.sort_values("interval").reset_index(drop=True)

    has_two_taus = len(popt) > 2 and popt[2] is not None and not np.isnan(popt[2])
    total_tau = popt[1] + popt[2] if has_two_taus else popt[1]
    xlim_closeup = 3.0 * total_tau

    dt_emp = np.median(np.diff(np.sort(df["interval"].unique())))
    total_counts = df["time_interval_distribution"].sum()
    df["pdf"] = df["time_interval_distribution"] / (total_counts * dt_emp)

    if duration is None:
        duration = max(total_counts / popt[0], 1.0)

    max_raw_events = 2_000_000
    max_sim_duration = max_raw_events / max(float(popt[0]), 1.0)
    sim_duration = min(float(duration), max_sim_duration, 30.0)
    sim_duration = max(sim_duration, 0.1)

    raw_events = tid.generate_poisson_events(popt[0], sim_duration)
    recorded_events = simulate_chain(raw_events, arrangement, popt)
    intervals = np.diff(recorded_events)

    bin_width = dt_emp if (np.isfinite(dt_emp) and dt_emp > 0) else max(xlim_closeup / 150.0, 1e-8)
    bin_edges = np.arange(0, intervals.max() + bin_width, bin_width)
    counts, edges = np.histogram(intervals, bins=bin_edges, density=True)

    created_fig = axes is None
    if created_fig:
        fig, axes = plt.subplots(
            1, 2,
            dpi=200,
            figsize=(13.5, 4.8),
            sharex=False,
            sharey=False,
        )
    ax_full, ax_zoom = axes

    # Cut consistently at the maximum interval of the empirical TID (not the simulated/resampled max)
    xlim_full = float(df["interval"].max())

    def _plot_layers(ax, x_upper, n_grid):
        df_sub = df[df["interval"] <= x_upper]
        ax.step(
            df_sub["interval"],
            df_sub["pdf"],
            where="post",
            alpha=0.75,
            color="steelblue",
            lw=1.3,
            label="Empirical",
        )

        mask = edges[:-1] <= x_upper
        ax.step(
            edges[:-1][mask],
            counts[mask],
            where="post",
            color="crimson",
            lw=1.2,
            label=f"Reconstructed ({arrangement})",
        )

        t_grid = np.linspace(0, x_upper, n_grid)
        ax.plot(
            t_grid,
            tid.evaluate_series_tid(t_grid, arrangement, *popt),
            "k--",
            lw=1.4,
            label=f"Analytical ({arrangement})",
        )

        ax.axvline(
            popt[1],
            color="darkorange",
            ls=":",
            lw=1.6,
            label=rf"$\tau_1$ = {popt[1]*1e6:.2f} $\mu$s",
        )
        if has_two_taus:
            ax.axvline(
                popt[1] + popt[2],
                color="seagreen",
                ls="-.",
                lw=1.6,
                label=rf"$\tau_1 + \tau_2$ = {(popt[1] + popt[2])*1e6:.2f} $\mu$s",
            )

    _plot_layers(ax_full, xlim_full, n_grid=1000)
    ax_full.set_xlim(0, xlim_full)
    if logy_full:
        ax_full.set_yscale("log")
    ax_full.set_xlabel("Inter-arrival Time $t$ (s)")
    ax_full.set_ylabel(r"Probability Density ($\mathrm{s}^{-1}$)")
    ax_full.set_title("Full Distribution")
    ax_full.legend(loc="upper right", frameon=True)
    ax_full.grid(True, alpha=0.25)

    _plot_layers(ax_zoom, xlim_closeup, n_grid=500)
    ax_zoom.set_xlim(0, xlim_closeup)
    ax_zoom.yaxis.tick_right()
    ax_zoom.yaxis.set_label_position("right")
    ax_zoom.set_xlabel("Inter-arrival Time $t$ (s)")
    ax_zoom.set_ylabel(r"Probability Density ($\mathrm{s}^{-1}$)")

    tau_formula = r"3(\tau_1 + \tau_2)" if has_two_taus else r"3\tau_1"
    ax_zoom.set_title(rf"Dead-Time Close-Up ($0$ to ${tau_formula} = {xlim_closeup*1e6:.2f}\,\mu\mathrm{{s}}$)")
    ax_zoom.grid(True, alpha=0.25)

    param_names = [r"\lambda", r"\tau_1", r"\tau_2"] if has_two_taus else [r"\lambda", r"\tau_1"]
    param_label = "\n".join(fr"${name}$ = {val:.3e}" for name, val in zip(param_names, popt))
    ax_zoom.text(
        0.97,
        0.95,
        param_label,
        transform=ax_zoom.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    if created_fig:
        if channel is not None:
            fig.suptitle(f"Channel {channel} — Time Interval Distribution ({arrangement})", y=1.02)
        plt.tight_layout()

    return axes


def plot_tid_closeup(
    tid_,
    popt,
    arrangement: str = "NE-E",
    channel=None,
    duration: float = None,
    ax=None,
):
    """
    Plots a single-channel close-up of the empirical TID PDF, Monte Carlo reconstructed PDF,
    and analytical dead-time model up to x_max = 3 * (tau_1 + tau_2).
    """
    popt = np.asarray(popt, dtype=float)

    if isinstance(tid_, pd.Series):
        df = pd.DataFrame({
            "interval": tid_.index.to_numpy(dtype=float),
            "time_interval_distribution": tid_.to_numpy(dtype=float),
        })
    else:
        df = tid_.copy()
        if "channel" in df.columns:
            if channel is None:
                unique_chs = df["channel"].unique()
                if len(unique_chs) > 1:
                    raise ValueError(
                        f"`tid_` contains multiple channels ({list(unique_chs)}). Pass `channel=...`."
                    )
                channel = unique_chs[0]
            df = df[df["channel"] == channel].copy()

    df = df.sort_values("interval").reset_index(drop=True)

    has_two_taus = len(popt) > 2 and popt[2] is not None and not np.isnan(popt[2])
    total_tau = popt[1] + popt[2] if has_two_taus else popt[1]
    xlim_ = 3.0 * total_tau

    dt_emp = np.median(np.diff(np.sort(df["interval"].unique())))
    total_counts = df["time_interval_distribution"].sum()
    df["pdf"] = df["time_interval_distribution"] / (total_counts * dt_emp)
    df_zoom = df[df["interval"] <= xlim_]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(dpi=200, figsize=(7.5, 4.8))

    ax.step(
        df_zoom["interval"],
        df_zoom["pdf"],
        where="post",
        alpha=0.75,
        color="steelblue",
        lw=1.3,
        label="Empirical",
    )

    if duration is None:
        duration = max(total_counts / popt[0], 1.0)

    max_raw_events = 2_000_000
    max_sim_duration = max_raw_events / max(float(popt[0]), 1.0)
    sim_duration = min(float(duration), max_sim_duration, 30.0)
    sim_duration = max(sim_duration, 0.1)

    raw_events = tid.generate_poisson_events(popt[0], sim_duration)
    recorded_events = simulate_chain(raw_events, arrangement, popt)
    intervals = np.diff(recorded_events)

    bin_width = dt_emp if (np.isfinite(dt_emp) and dt_emp > 0) else max(xlim_ / 150.0, 1e-8)
    bin_edges = np.arange(0, intervals.max() + bin_width, bin_width)
    counts, edges = np.histogram(intervals, bins=bin_edges, density=True)

    mask = edges[:-1] <= xlim_
    ax.step(
        edges[:-1][mask],
        counts[mask],
        where="post",
        color="crimson",
        lw=1.2,
        label=f"Reconstructed ({arrangement})",
    )

    t_grid = np.linspace(0, xlim_, 500)
    ax.plot(
        t_grid,
        tid.evaluate_series_tid(t_grid, arrangement, *popt),
        "k--",
        lw=1.4,
        label=f"Analytical ({arrangement})",
    )

    ax.axvline(
        popt[1],
        color="darkorange",
        ls=":",
        lw=1.6,
        label=rf"$\tau_1$ = {popt[1]*1e6:.2f} $\mu$s",
    )
    if has_two_taus:
        ax.axvline(
            popt[1] + popt[2],
            color="seagreen",
            ls="-.",
            lw=1.6,
            label=rf"$\tau_1 + \tau_2$ = {(popt[1] + popt[2])*1e6:.2f} $\mu$s",
        )

    param_names = [r"\lambda", r"\tau_1", r"\tau_2"] if has_two_taus else [r"\lambda", r"\tau_1"]
    param_label = "\n".join(fr"${name}$ = {val:.3e}" for name, val in zip(param_names, popt))
    ax.text(
        0.97,
        0.60,
        param_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    ch_prefix = f"Channel {channel} — " if channel is not None else ""
    tau_formula = r"3(\tau_1 + \tau_2)" if has_two_taus else r"3\tau_1"
    ax.set_xlim(0, xlim_)
    ax.set_xlabel("Inter-arrival Time $t$ (s)")
    ax.set_ylabel(r"Probability Density ($\mathrm{s}^{-1}$)")
    ax.set_title(rf"{ch_prefix}Dead-Time Close-Up ($0$ to ${tau_formula} = {xlim_*1e6:.2f}\,\mu\mathrm{{s}}$)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.25)

    if created_fig:
        plt.tight_layout()

    return ax


def plot_rossi_alpha_fit(
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    popt: np.ndarray,
    n_exp: int,
    ch_id: int,
    ch_tr: int,
    gate: float,
    min_interval: float,
    rebin_width: float = 1e-6,
    ax=None,
):
    """Plots empirical Rossi-alpha distribution (rebinned to 1 us by default) alongside the multi-exponential fit."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(dpi=200, figsize=(7, 3.5))

    x_all = np.asarray(x_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)
    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)

    dt_native = float(np.median(np.diff(x_all))) if len(x_all) > 1 else 0.0
    if rebin_width is not None and dt_native > 0 and rebin_width > dt_native * 1.01:
        factor = int(round(rebin_width / dt_native))
        n_coarse = len(y_all) // factor
        if n_coarse > 0:
            x_plot = x_all[: n_coarse * factor].reshape(n_coarse, factor)[:, 0]
            y_plot = y_all[: n_coarse * factor].reshape(n_coarse, factor).sum(axis=1)
            y_fit_plot = y_fit * factor
            bin_label = f"Empirical Data ({factor * dt_native * 1e6:.2g} $\\mu$s bins)"
        else:
            x_plot, y_plot, y_fit_plot = x_all, y_all, y_fit
            bin_label = "Empirical Data"
    else:
        x_plot, y_plot, y_fit_plot = x_all, y_all, y_fit
        bin_label = "Empirical Data"

    ax.plot(x_plot * 1e6, y_plot, color="steelblue", lw=1.0, alpha=0.8, label=bin_label)

    alphas = [popt[2 + 2 * i] for i in range(n_exp)]
    alpha_str = ", ".join(f"{a:.2e}" for a in alphas)
    ax.plot(
        x_fit * 1e6,
        y_fit_plot,
        color="crimson",
        lw=1.6,
        label=rf"Fit ($\alpha = {alpha_str}\ \mathrm{{s}}^{{-1}}$)",
    )

    if min_interval > 0:
        ax.axvline(
            min_interval * 1e6,
            color="gray",
            ls=":",
            lw=1.1,
            label=rf"Fit start ({min_interval*1e6:.1f} $\mu$s)",
        )

    ax.set_xlabel(r"Delay Time ($\mu$s)")
    ax.set_ylabel("Rossi-$\\alpha$ Counts")
    ax.set_title(f"Rossi-$\\alpha$ | Signal Ch {ch_id}, Trigger Ch {ch_tr} | Gate = {gate * 1e6:.0f} $\\mu$s")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25)

    if created_fig:
        plt.tight_layout()

    return ax


# ============================================================
# 2. MAIN PROCESSING PIPELINE
# ============================================================

def main(
    file_in: str,
    dt_structure: str = "NE-E",
    output_dir: str = "processed_data",
    overwrite_h5: bool = False,
    bin_size_seconds: float = 0.1,
    min_duration: float = 10.0,
    max_ticks: int = 20_000,
    chunk_size: int = 500_000,
    n_exp: int = 1,
    rossi_min_interval: float = 50e-6,
    rossi_plot_bin_us: float = 1.0,
) -> None:
    t_pipeline_start = time.perf_counter()
    print(f"[TIME] Numba threads available: {tsd.numba.get_num_threads()}")

    in_path = Path(file_in).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: '{in_path}'")

    t_step = time.perf_counter()
    if in_path.suffix.lower() in {".h5", ".hdf5"}:
        h5_path = in_path
        print(f"[TIME] Using existing HDF5 file: {h5_path} (0.00 s)")
    else:
        h5_path = in_path.with_suffix(".h5")
        if overwrite_h5 or not h5_path.exists():
            print("[TIME] Step 1 (BIN -> HDF5 conversion): starting...")
            tsd.convert_bin_to_hdf5(str(in_path), str(h5_path), chunk_size=chunk_size)
            print(f"[TIME] Step 1 (BIN -> HDF5 conversion): {time.perf_counter() - t_step:.2f} s")
        else:
            print(f"[TIME] Step 1 (HDF5 already exists, skipped conversion): {time.perf_counter() - t_step:.2f} s — {h5_path}")

    # --------------------------------------------------------
    # A. Compute Count Rate and TID per active channel
    # --------------------------------------------------------
    t_step_a = time.perf_counter()
    results = {}
    with tsd.HDF5TimestampData(str(h5_path)) as data:
        for ch_id, ch_handle in data.channels.items():
            print(f"\nProcessing channel {ch_id} ({len(ch_handle):,} events)...")

            t_cr = time.perf_counter()
            cr_series = ch_handle.count_rate(bin_size_seconds=bin_size_seconds, chunk_size=chunk_size)
            print(f"[TIME]   Ch {ch_id} count_rate: {time.perf_counter() - t_cr:.2f} s")

            t_tid = time.perf_counter()
            tid_series = ch_handle.time_interval_distribution(
                max_delta_ticks=max_ticks,
                coarser_grouping=1,
                chunk_size=chunk_size,
            )
            print(f"[TIME]   Ch {ch_id} time_interval_distribution (max_ticks={max_ticks}, chunk_size={chunk_size:,}): {time.perf_counter() - t_tid:.2f} s")

            results[ch_id] = {
                "countrate": cr_series,
                "tid": tid_series,
            }
    print(f"[TIME] Step 2A (All channels count_rate + TID): {time.perf_counter() - t_step_a:.2f} s")

    if not results:
        print(f"No channels found in {h5_path}. Exiting.")
        return

    multichannel_acquisition = len(results) > 1
    print("Multichannel acquisition detected." if multichannel_acquisition else "Single-channel acquisition detected.")

    suitable_channels = []
    for ch_id, ch_res in results.items():
        cr = ch_res["countrate"]
        recorded_time = float(cr.index.max() - cr.index.min()) if len(cr) > 1 else 0.0

        if recorded_time < min_duration:
            label = f"Channel {ch_id}" if multichannel_acquisition else f"File {h5_path.name}"
            print(f"{label} is empty or too short (recorded_time = {recorded_time:.2f} s) — skipping.")
        else:
            suitable_channels.append(ch_id)

    if not suitable_channels:
        print(f"No channels met the minimum {min_duration:.1f} s recording duration. Exiting.")
        return

    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # B. Per-channel Count Rate & Dead-Time (TID) Analysis
    # --------------------------------------------------------
    t_step_b = time.perf_counter()
    for ch_id in suitable_channels:
        t_ch = time.perf_counter()
        suffix = f"_ch{ch_id}" if multichannel_acquisition else ""
        ch_label = ch_id if multichannel_acquisition else None
        print(f"\n--- Analyzing {f'Channel {ch_id}' if multichannel_acquisition else 'Single Channel'} ---")

        # 1. Count rate evolution
        t_sub = time.perf_counter()
        print("Saving and plotting count rate evolution...")
        countrate = results[ch_id]["countrate"]
        countrate.to_csv(out_path / f"countrate{suffix}.csv")

        fig_cr, ax_cr = plt.subplots(dpi=200, figsize=(7, 3))
        plot_countrate_evolution(countrate, channel=ch_label, ax=ax_cr)
        fig_cr.tight_layout()
        fig_cr.savefig(out_path / f"countrate{suffix}.png", bbox_inches="tight")
        plt.close(fig_cr)
        print(f"[TIME]   Ch {ch_id} count-rate save & plot: {time.perf_counter() - t_sub:.2f} s")

        # 2. Time Interval Distribution (TID)
        t_sub = time.perf_counter()
        print("Saving TID...")
        tid_ = results[ch_id]["tid"].to_frame().reset_index(names="interval")
        tid_["interval"] = tid_["interval"] * 1e-8  # Convert 10 ns ticks to seconds
        tid_.to_csv(out_path / f"tid{suffix}.csv", index=False)
        print(f"[TIME]   Ch {ch_id} TID CSV save: {time.perf_counter() - t_sub:.2f} s")

        # 3. Dead-time histogram fitting
        t_sub = time.perf_counter()
        print(f"Fitting TID with a {dt_structure} deadtime model...")
        dt_tick_sec = tid_["interval"].diff().mean()
        p0 = (
            (countrate.mean(), 10e-8)
            if dt_structure.upper() in {"NE", "E"}
            else (countrate.mean(), 10e-8, 10e-8)
        )

        tid_fitting = tid.fit_deadtime_histogram(
            tid_["interval"] + dt_tick_sec / 2.0,
            tid_["time_interval_distribution"],
            dt_structure,
            p0,
        )

        with open(out_path / f"tid_fitting{suffix}.pkl", "wb") as f:
            pickle.dump(tid_fitting, f)
        print(f"[TIME]   Ch {ch_id} TID fit & pickle: {time.perf_counter() - t_sub:.2f} s")

        # 4. Close-up dead-time plot
        t_sub = time.perf_counter()
        print("Plotting TID distribution and estimated deadtime constants...")
        recorded_duration = float(countrate.index.max() - countrate.index.min())

        fig_tid, axes_tid = plt.subplots(
            1, 2, figsize=(13.5, 4.8), dpi=200,
            sharex=False, sharey=False,
            gridspec_kw={"width_ratios": [2, 1]},
        )
        plot_tid_full_and_closeup(
            tid_,
            tid_fitting["popt"],
            arrangement=dt_structure,
            channel=ch_label,
            duration=recorded_duration,
            axes=axes_tid,
        )
        fig_tid.tight_layout()
        fig_tid.savefig(out_path / f"tid_dtc{suffix}.png", bbox_inches="tight")
        plt.close(fig_tid)

        fig_tid, ax_tid = plt.subplots(dpi=200, figsize=(7.5, 4.8))
        plot_tid_closeup(
            tid_,
            tid_fitting["popt"],
            arrangement=dt_structure,
            channel=ch_label,
            duration=recorded_duration,
            ax=ax_tid,
        )
        fig_tid.tight_layout()
        fig_tid.savefig(out_path / f"tid_closeup_dtc{suffix}.png", bbox_inches="tight")
        plt.close(fig_tid)
        print(f"[TIME]   Ch {ch_id} TID plots (full + closeup): {time.perf_counter() - t_sub:.2f} s")
        print(f"[TIME]   Ch {ch_id} total Step 2B time: {time.perf_counter() - t_ch:.2f} s")

    print(f"[TIME] Step 2B (All channels post-processing/fitting/plotting): {time.perf_counter() - t_step_b:.2f} s")

    # --------------------------------------------------------
    # C. Single-Pass Max-Gate Rossi-Alpha Computation & Fitting
    # --------------------------------------------------------
    print("\n--- Computing Rossi-Alpha Distributions (Single-Pass Max Gate) ---")
    t_step_c = time.perf_counter()
    gates_sec = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]) * 1e-6
    max_gate = float(np.max(gates_sec))
    rossi = {}

    channel_pairs = (
        list(product(suitable_channels, repeat=2))
        if len(suitable_channels) > 1
        else [(suitable_channels[0], suitable_channels[0])]
    )

    with tsd.HDF5TimestampData(str(h5_path)) as data:
        for ch_id, ch_tr in channel_pairs:
            t_pair = time.perf_counter()
            print(
                f"Computing max-gate Rossi-alpha ({max_gate * 1e6:.0f} us): "
                f"signal Ch {ch_id}, trigger Ch {ch_tr}..."
            )
            data_id = data.channels[ch_id]
            trig_handle = None if ch_id == ch_tr else data.channels[ch_tr]

            ra_full = data_id.rossi_alpha(
                h5_trigger=trig_handle,
                gate=max_gate,
                coarser_grouping=1,
                chunk_size=chunk_size,
            )

            for g in gates_sec:
                rossi[(float(g), ch_id, ch_tr)] = slice_rossi_gate(
                    ra_full, gate_sec=float(g), coarser_grouping=1
                )

            print(
                f"[TIME]   Pair (sig={ch_id}, trig={ch_tr}) max-gate + {len(gates_sec)} slices: "
                f"{time.perf_counter() - t_pair:.2f} s"
            )

    print(f"[TIME] Step 2C-1 (All Rossi-Alpha computations): {time.perf_counter() - t_step_c:.2f} s")

    t_pkl = time.perf_counter()
    with open(out_path / "rossialpha_data.pkl", "wb") as f:
        pickle.dump(rossi, f)
    print(f"[TIME] Saving rossialpha_data.pkl: {time.perf_counter() - t_pkl:.2f} s")

    # Fit (native resolution) and plot (rebinned to `rossi_plot_bin_us`) each Rossi-alpha distribution
    print("Fitting and plotting Rossi-alpha distributions...")
    t_fit_plot = time.perf_counter()
    rossi_plots_dir = out_path / "rossi_alpha_plots"
    rossi_plots_dir.mkdir(parents=True, exist_ok=True)

    fit_results = {}
    fit_summary_rows = []
    rebin_width_sec = rossi_plot_bin_us * 1e-6 if rossi_plot_bin_us and rossi_plot_bin_us > 0 else None

    for (g, ch_id, ch_tr), rossi_raw in rossi.items():
        if isinstance(rossi_raw, pd.Series):
            df_rossi = rossi_raw.to_frame(name="rossi_alpha_distribution").reset_index(names="interval")
        else:
            df_rossi = rossi_raw.reset_index() if "interval" not in rossi_raw.columns else rossi_raw.copy()

        # Ensure interval is in seconds (convert if stored in 10 ns ticks)
        if df_rossi["interval"].max() > 1.0:
            df_rossi["interval"] = df_rossi["interval"] * 1e-8

        df_rossi = df_rossi.sort_values("interval").reset_index(drop=True)
        sub_df = df_rossi[df_rossi["interval"] > rossi_min_interval]

        if len(sub_df) < (1 + 2 * n_exp):
            print(
                f"Skipping fit for Ch {ch_id} -> Tr {ch_tr}, Gate {g*1e6:.0f} us: "
                f"insufficient points above {rossi_min_interval*1e6:.1f} us."
            )
            continue

        x_fit_data = sub_df["interval"].to_numpy(dtype=float)
        y_fit_data = sub_df["rossi_alpha_distribution"].to_numpy(dtype=float)

        try:
            popt, pcov, y_fit = fit_multi_exp(x_fit_data, y_fit_data, n_exp=n_exp)
        except RuntimeError as err:
            print(f"Fit did not converge for Ch {ch_id} -> Tr {ch_tr}, Gate {g*1e6:.0f} us: {err}")
            continue

        perr = np.sqrt(np.diag(pcov))
        fit_results[(g, ch_id, ch_tr)] = {
            "popt": popt,
            "pcov": pcov,
            "perr": perr,
            "x_fit": x_fit_data,
            "y_fit": y_fit,
        }

        row = {
            "channel": ch_id,
            "trigger_channel": ch_tr,
            "gate_s": g,
            "gate_us": g * 1e6,
            "c": popt[0],
            "c_err": perr[0],
        }
        for i in range(n_exp):
            row[f"A_{i+1}"] = popt[1 + 2 * i]
            row[f"A_{i+1}_err"] = perr[1 + 2 * i]
            row[f"alpha_{i+1}_s_inv"] = popt[2 + 2 * i]
            row[f"alpha_{i+1}_err"] = perr[2 + 2 * i]
        fit_summary_rows.append(row)

        fig_ra, ax_ra = plt.subplots(dpi=200, figsize=(7, 3.5))
        plot_rossi_alpha_fit(
            x_all=df_rossi["interval"].to_numpy(dtype=float),
            y_all=df_rossi["rossi_alpha_distribution"].to_numpy(dtype=float),
            x_fit=x_fit_data,
            y_fit=y_fit,
            popt=popt,
            n_exp=n_exp,
            ch_id=ch_id,
            ch_tr=ch_tr,
            gate=g,
            min_interval=rossi_min_interval,
            rebin_width=rebin_width_sec,
            ax=ax_ra,
        )
        ax_ra.set_xlim( 0, max(df_rossi["interval"].max(), x_fit_data.max()) * 1.05)
        fig_ra.tight_layout()
        plot_name = f"rossi_alpha_ch{ch_id}_tr{ch_tr}_g{int(round(g * 1e6))}us.png"
        fig_ra.savefig(rossi_plots_dir / plot_name, bbox_inches="tight")
        plt.close(fig_ra)

    if fit_summary_rows:
        pd.DataFrame(fit_summary_rows).to_csv(out_path / "rossi_alpha_curve_fit.csv", index=False)

    with open(out_path / "rossi_alpha_fitting.pkl", "wb") as f:
        pickle.dump(fit_results, f)

    print(f"[TIME] Step 2C-2 (All Rossi-Alpha fits & plots): {time.perf_counter() - t_fit_plot:.2f} s")
    print(f"\n[TIME] TOTAL PIPELINE RUNTIME: {time.perf_counter() - t_pipeline_start:.2f} s")
    print(f"Processing complete. Outputs saved to '{out_path}'.")


# ============================================================
# 3. COMMAND-LINE INTERFACE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-platform (Windows/Linux) CAEN timestamp conversion & Count Rate + TID + Rossi-alpha analysis."
    )
    parser.add_argument(
        "file_in",
        help="Path to the input raw binary (.bin/.dat) or pre-converted (.h5) file.",
    )
    parser.add_argument(
        "--dt",
        "--dt-structure",
        dest="dt_structure",
        default="NE-E",
        choices=["NE", "E", "NE-E", "E-NE", "NE-NE", "E-E"],
        help="Dead-time model arrangement to fit (default: NE-E).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="output_dir",
        default="processed_data",
        help="Directory where CSVs, pickles, and PNG plots will be saved (default: processed_data).",
    )
    parser.add_argument(
        "--lib-dir",
        default=None,
        help="Optional path to the PROCESSING folder containing TID_analysis.py and TimeStampData_hdf5.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-conversion of .bin/.dat to .h5 even if the .h5 file already exists.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.1,
        help="Count rate bin width in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum channel recording duration in seconds to process (default: 10.0).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=20_000,
        help="Maximum inter-arrival time interval in ticks for time_interval_distribution (default: 20000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Number of events per processing chunk for conversion and HDF5 analysis (default: 500000).",
    )
    parser.add_argument(
        "--n-exp",
        type=int,
        default=1,
        help="Number of exponential decay components for Rossi-alpha fitting (default: 1).",
    )
    parser.add_argument(
        "--rossi-min-interval",
        type=float,
        default=50e-6,
        help="Lower time cutoff in seconds for Rossi-alpha fitting to exclude dead-time distortion (default: 50e-6).",
    )
    parser.add_argument(
        "--rossi-plot-bin-us",
        type=float,
        default=1.0,
        help="Visualization bin width in microseconds for Rossi-alpha plots (default: 1.0; set <=0 for native bins).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        file_in=args.file_in,
        dt_structure=args.dt_structure,
        output_dir=args.output_dir,
        overwrite_h5=args.overwrite,
        bin_size_seconds=args.bin_size,
        min_duration=args.min_duration,
        max_ticks=args.max_ticks,
        chunk_size=args.chunk_size,
        n_exp=args.n_exp,
        rossi_min_interval=args.rossi_min_interval,
        rossi_plot_bin_us=args.rossi_plot_bin_us,
    )