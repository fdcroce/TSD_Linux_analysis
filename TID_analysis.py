import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit


# =====================================================================
# 1. MONTE CARLO EVENT STREAM & DEAD-TIME ALGORITHMS
# =====================================================================

def generate_poisson_events(rate: float, duration: float, rng=None) -> np.ndarray:
    """Generate arrival timestamps of a homogeneous Poisson process."""
    if rng is None:
        rng = np.random.default_rng()
    n_expected = rate * duration
    n_samples = int(n_expected + 10 * np.sqrt(max(n_expected, 1)) + 100)
    inter_arrivals = rng.exponential(scale=1.0 / rate, size=n_samples)
    timestamps = np.cumsum(inter_arrivals)
    while timestamps[-1] < duration:
        extra = rng.exponential(scale=1.0 / rate, size=max(100, int(0.1 * n_samples)))
        timestamps = np.concatenate([timestamps, timestamps[-1] + np.cumsum(extra)])
    return timestamps[timestamps <= duration]


def generate_rossi_alpha_events(
    total_rate: float,
    duration: float,
    rossi_alpha: float,
    chain_fraction: float = 0.3,
    events_per_chain: float = 3.0,
    background_rate: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate arrival timestamps with Rossi-alpha correlated fission bursts.

    Each correlated burst starts at a Poisson trigger time and spawns
    `events_per_chain` correlated events whose offsets after the trigger
    follow an exponential distribution with decay rate `rossi_alpha` (1/s).

    Parameters
    ----------
    total_rate : float
        Target overall event rate (events/s) used when `background_rate` is None.
    duration : float
        Simulation window length in seconds.
    rossi_alpha : float
        Rossi-alpha decay constant alpha in s^-1 (must be > 0).
    chain_fraction : float
        Approximate fraction of total events belonging to correlated chains in [0, 1).
    events_per_chain : float
        Mean number of correlated events per chain trigger (> 0).
    background_rate : float, optional
        Explicit uncorrelated Poisson background rate (events/s). If None, it is
        inferred from `total_rate * (1 - chain_fraction)`.
    rng : np.random.Generator, optional
        NumPy random generator instance.

    Returns
    -------
    np.ndarray
        Sorted 1D array of event timestamps in [0, duration].
    """
    if rng is None:
        rng = np.random.default_rng()
    if duration <= 0:
        return np.array([], dtype=float)
    if rossi_alpha <= 0:
        raise ValueError("rossi_alpha must be positive.")
    if not (0.0 <= chain_fraction < 1.0):
        raise ValueError("chain_fraction must be in [0, 1).")
    if events_per_chain <= 0:
        raise ValueError("events_per_chain must be positive.")

    if background_rate is None:
        bg_rate = total_rate * (1.0 - chain_fraction)
        correlated_rate = total_rate * chain_fraction
    else:
        bg_rate = max(0.0, float(background_rate))
        correlated_rate = max(0.0, float(total_rate) - bg_rate)

    parts: list[np.ndarray] = []

    # 1. Uncorrelated (Poisson) background events
    if bg_rate > 0:
        bg_events = generate_poisson_events(bg_rate, duration, rng=rng)
        if bg_events.size > 0:
            parts.append(bg_events)

    # 2. Correlated fission-chain triggers + exponentially delayed events (~ exp(-alpha * dt))
    trigger_rate = correlated_rate / events_per_chain
    if trigger_rate > 0:
        triggers = generate_poisson_events(trigger_rate, duration, rng=rng)
        if triggers.size > 0:
            # Draw number of correlated events per trigger (Poisson around events_per_chain)
            counts = rng.poisson(lam=events_per_chain, size=triggers.size)
            valid = counts > 0
            if np.any(valid):
                t_valid = triggers[valid]
                c_valid = counts[valid]
                repeated_triggers = np.repeat(t_valid, c_valid)
                delays = rng.exponential(scale=1.0 / rossi_alpha, size=repeated_triggers.size)
                chain_events = repeated_triggers + delays
                chain_events = chain_events[chain_events <= duration]
                if chain_events.size > 0:
                    parts.append(chain_events)

    if not parts:
        return np.array([], dtype=float)

    return np.sort(np.concatenate(parts))

def apply_non_paralyzable(timestamps: np.ndarray, tau: float) -> np.ndarray:
    """Non-paralyzable (NE) dead-time filter."""
    if len(timestamps) == 0 or tau <= 0:
        return timestamps.copy()
    accepted = []
    dead_until = -np.inf
    for t in timestamps:
        if t >= dead_until:
            accepted.append(t)
            dead_until = t + tau
    return np.array(accepted)


def apply_paralyzable(timestamps: np.ndarray, tau: float) -> np.ndarray:
    """Paralyzable (extendable / E) dead-time filter."""
    if len(timestamps) == 0 or tau <= 0:
        return timestamps.copy()
    ts = np.sort(timestamps)
    accepted = []
    dead_until = -np.inf
    for t in ts:
        if t >= dead_until:
            accepted.append(t)
        # Every arriving event extends dead time
        dead_until = t + tau
    return np.array(accepted, dtype=float)


# =====================================================================
# 2. ANALYTICAL TIME-INTERVAL DISTRIBUTION (TID) FUNCTIONS
# =====================================================================

def tid_single_ne(t: np.ndarray, lam: float, tau: float) -> np.ndarray:
    """Analytical TID for a single Non-Paralyzable dead time."""
    t = np.asarray(t, dtype=float)
    pdf = np.zeros_like(t)
    mask = t >= tau
    pdf[mask] = lam * np.exp(-lam * (t[mask] - tau))
    return pdf

def tid_single_e(t: np.ndarray, lam: float, tau: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    pdf = np.zeros_like(t)
    if tau <= 0 or np.max(t) < tau:
        return tid_single_ne(t, lam, 0.0) if tau <= 0 else pdf

    max_k = int(np.floor(np.max(t) / tau))
    # Guard against unphysically tiny trial tau during optimization
    if max_k > 500:
        return np.zeros_like(t)

    for k in range(1, max_k + 1):
        mask = t >= k * tau
        tk = t[mask] - k * tau
        sign = -1.0 if (k - 1) % 2 == 1 else 1.0

        if k == 1:
            term = lam * np.exp(-lam * tau)
            pdf[mask] += sign * term
        else:
            pos = tk > 0
            log_term = (
                k * np.log(lam)
                - math.lgamma(k)
                - k * lam * tau
                + (k - 1) * np.log(tk[pos])
            )
            sub_pdf = np.zeros_like(tk)
            sub_pdf[pos] = np.exp(log_term)
            pdf[mask] += sign * sub_pdf

    return np.maximum(pdf, 0.0)


def tid_ne_e(t: np.ndarray, lam: float, tau1: float, tau2: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if tau1 >= tau2:
        return tid_single_ne(t, lam, tau1)

    pdf = np.zeros_like(t)
    if np.max(t) < tau2:
        return pdf
    if tau1 <= 1e-12:
        return tid_single_e(t, lam, tau2)

    max_k = int(np.floor((np.max(t) - tau2) / tau1)) + 1
    # Guard against unphysically tiny trial tau1 during optimization
    if max_k > 500:
        return np.zeros_like(t)

    for k in range(1, max_k + 1):
        shift = tau2 + (k - 1) * tau1
        mask = t >= shift
        tk = t[mask] - shift
        exponent = -lam * (tau2 - (k - 1) * (tau2 - tau1))
        sign = -1.0 if (k - 1) % 2 == 1 else 1.0

        if k == 1:
            pdf[mask] += sign * lam * np.exp(exponent)
        else:
            pos = tk > 0
            log_term = (
                k * np.log(lam)
                - math.lgamma(k)
                + exponent
                + (k - 1) * np.log(tk[pos])
            )
            sub_pdf = np.zeros_like(tk)
            sub_pdf[pos] = np.exp(log_term)
            pdf[mask] += sign * sub_pdf

    return np.maximum(pdf, 0.0)


def tid_e_ne(t: np.ndarray, lam: float, tau1: float, tau2: float) -> np.ndarray:
    """Analytical TID for Paralyzable (tau1) -> Non-Paralyzable (tau2) in series."""
    t = np.asarray(t, dtype=float)
    if tau1 >= tau2:
        return tid_single_e(t, lam, tau1)

    pdf = np.zeros_like(t)
    r1 = lam * np.exp(-lam * tau1)
    mask = t >= tau2
    pdf[mask] = r1 * np.exp(-r1 * (t[mask] - tau2))
    return pdf


def tid_ne_ne(t: np.ndarray, lam: float, tau1: float, tau2: float) -> np.ndarray:
    """Analytical TID for Non-Paralyzable (tau1) -> Non-Paralyzable (tau2) in series."""
    return tid_single_ne(t, lam, max(tau1, tau2))


def tid_e_e(t: np.ndarray, lam: float, tau1: float, tau2: float) -> np.ndarray:
    """Analytical TID for Paralyzable (tau1) -> Paralyzable (tau2) in series."""
    return tid_single_e(t, lam, max(tau1, tau2))


def evaluate_series_tid(
    t: np.ndarray,
    arrangement: str,
    lam: float,
    tau1: float,
    tau2: float = 0.0,
) -> np.ndarray:
    """
    Pure deterministic analytical PDF evaluator for any dead-time arrangement.
    Used both inside `analyze_series_deadtime` and by the fitting routines.
    """
    arr = arrangement.upper().strip()
    if arr == "NE-E":
        return tid_ne_e(t, lam, tau1, tau2)
    elif arr == "E-NE":
        return tid_e_ne(t, lam, tau1, tau2)
    elif arr == "NE-NE":
        return tid_ne_ne(t, lam, tau1, tau2)
    elif arr == "E-E":
        return tid_e_e(t, lam, tau1, tau2)
    elif arr == "NE":
        return tid_single_ne(t, lam, tau1)
    elif arr == "E":
        return tid_single_e(t, lam, tau1)
    else:
        raise ValueError(
            f"Unknown arrangement '{arrangement}'. "
            "Choose from 'NE-E', 'E-NE', 'NE-NE', 'E-E', 'NE', or 'E'."
        )


# =====================================================================
# 3. SIMULATION + ANALYTICAL GENERATOR (`analyze_series_deadtime`)
# =====================================================================

def analyze_series_deadtime(
    arrangement: str,
    lam: float,
    tau1: float,
    tau2: float = 0.0,
    duration: float = 1000.0,
    t_grid: np.ndarray = None,
    seed: int = 42,
) -> dict:
    """
    Simulate a Poisson stream through a chosen dead-time arrangement and evaluate
    both empirical inter-arrival intervals and the exact analytical PDF.
    """
    arr = arrangement.upper().strip()
    rng = np.random.default_rng(seed)

    if t_grid is None:
        t_max = max(tau1, tau2) + 6.0 / lam
        t_grid = np.linspace(0, t_max, 500)

    raw_events = generate_poisson_events(lam, duration, rng=rng)

    if arr == "NE-E":
        stage1 = apply_non_paralyzable(raw_events, tau1)
        recorded = apply_paralyzable(stage1, tau2)
    elif arr == "E-NE":
        stage1 = apply_paralyzable(raw_events, tau1)
        recorded = apply_non_paralyzable(stage1, tau2)
    elif arr == "NE-NE":
        stage1 = apply_non_paralyzable(raw_events, tau1)
        recorded = apply_non_paralyzable(stage1, tau2)
    elif arr == "E-E":
        stage1 = apply_paralyzable(raw_events, tau1)
        recorded = apply_paralyzable(stage1, tau2)
    elif arr == "NE":
        recorded = apply_non_paralyzable(raw_events, tau1)
    elif arr == "E":
        recorded = apply_paralyzable(raw_events, tau1)
    else:
        raise ValueError(
            f"Unknown arrangement '{arrangement}'. "
            "Choose from 'NE-E', 'E-NE', 'NE-NE', 'E-E', 'NE', or 'E'."
        )

    pdf = evaluate_series_tid(t_grid, arr, lam, tau1, tau2)
    intervals = np.diff(recorded)
    observed_rate = len(recorded) / duration

    return {
        "arrangement": arr,
        "raw_events": raw_events,
        "recorded_events": recorded,
        "intervals": intervals,
        "observed_rate": observed_rate,
        "t_grid": t_grid,
        "analytical_pdf": pdf,
    }


# =====================================================================
# 4. EXPERIMENTAL DATA FITTING FUNCTIONS (MLE & HISTOGRAM CURVE_FIT)
# =====================================================================

def fit_deadtime_mle(
    intervals: np.ndarray,
    arrangement: str,
    p0: tuple = None,
) -> dict:
    """
    Unbinned Maximum Likelihood Estimation (MLE) for (lam, tau1) on single dead times
    ('NE', 'E') or (lam, tau1, tau2) on series dead times ('NE-E', 'E-NE', etc.).
    """
    arr = arrangement.upper().strip()
    is_single = arr in {"NE", "E"}

    intervals = np.asarray(intervals, dtype=float)
    intervals = intervals[intervals > 0]
    t_min = np.min(intervals)
    mean_dt = np.mean(intervals)

    if is_single:
        if p0 is None:
            tau_guess = 0.98 * t_min
            lam_guess = 1.0 / max(mean_dt - tau_guess, 1e-6)
            p0 = (lam_guess, tau_guess)
        else:
            p0 = p0[:2]  # Use only (lam, tau1) if a 3-tuple was passed

        def neg_log_likelihood(params):
            lam, tau1 = params
            if lam <= 0 or tau1 < 0 or tau1 > t_min:
                return 1e18
            pdf = evaluate_series_tid(intervals, arr, lam, tau1)
            if np.any(pdf <= 0) or np.any(~np.isfinite(pdf)):
                return 1e18
            return -np.sum(np.log(pdf))

        res = minimize(
            neg_log_likelihood,
            x0=np.array(p0, dtype=float),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
        )
        return {
            "popt": res.x,          # (lam, tau1)
            "lam": res.x[0],
            "tau1": res.x[1],
            "tau2": None,
            "success": res.success,
            "nll": res.fun,
        }

    # --- Two-stage series arrangement ---
    if p0 is None:
        tau_guess = 0.98 * t_min
        lam_guess = 1.0 / max(mean_dt - tau_guess, 1e-6)
        p0 = (lam_guess, 0.5 * tau_guess, tau_guess)

    def neg_log_likelihood(params):
        lam, tau1, tau2 = params
        if lam <= 0 or tau1 < 0 or tau2 < 0 or max(tau1, tau2) > t_min:
            return 1e18
        pdf = evaluate_series_tid(intervals, arr, lam, tau1, tau2)
        if np.any(pdf <= 0) or np.any(~np.isfinite(pdf)):
            return 1e18
        return -np.sum(np.log(pdf))

    res = minimize(
        neg_log_likelihood,
        x0=np.array(p0, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )
    return {
        "popt": res.x,              # (lam, tau1, tau2)
        "lam": res.x[0],
        "tau1": res.x[1],
        "tau2": res.x[2],
        "success": res.success,
        "nll": res.fun,
    }


def fit_deadtime_histogram(
    dt: np.ndarray,
    counts: np.ndarray,
    arrangement: str,
    p0: tuple = None,
    method: str = "binned_mle",
) -> dict:
    """
    Fit dead-time parameters directly from histogram data (dt, counts).

    Parameters
    ----------
    dt : np.ndarray
        Bin centers of inter-arrival time intervals.
    counts : np.ndarray
        Raw event counts in each bin (does NOT need to be pre-normalized).
    arrangement : str
        'NE', 'E', 'NE-E', 'E-NE', 'NE-NE', or 'E-E'.
    p0 : tuple, optional
        Initial guess (lam, tau1) for single or (lam, tau1, tau2) for series.
    method : str
        'binned_mle' (recommended: minimizes binned Poisson deviance, handles
        sharp edges & empty tail bins cleanly) or 'lsq' (weighted least-squares).
    """
    arr = arrangement.upper().strip()
    is_single = arr in {"NE", "E"}

    dt = np.asarray(dt, dtype=float)
    counts = np.asarray(counts, dtype=float)

    # Infer uniform bin width (or handle variable bin widths)
    bin_width = np.median(np.diff(dt))
    total_counts = np.sum(counts)
    density = counts / (total_counts * bin_width)

    # Estimate smart initial guesses from the histogram if p0 is omitted
    if p0 is None:
        nonzero = dt[counts > 0]
        tau_guess = nonzero[0] if len(nonzero) > 0 else dt[0]
        mean_dt = np.sum(dt * counts) / total_counts
        lam_guess = 1.0 / max(mean_dt - tau_guess, bin_width)
        p0 = (lam_guess, tau_guess) if is_single else (lam_guess, 0.5 * tau_guess, tau_guess)
    elif is_single:
        p0 = p0[:2]

    def loss(params):
        if is_single:
            lam, tau1 = params
            if lam <= 0 or tau1 < 0:
                return 1e18
            pdf = evaluate_series_tid(dt, arr, lam, tau1)
        else:
            lam, tau1, tau2 = params
            if lam <= 0 or tau1 < 0 or tau2 < 0:
                return 1e18
            pdf = evaluate_series_tid(dt, arr, lam, tau1, tau2)

        if np.any(~np.isfinite(pdf)):
            return 1e18

        if method == "binned_mle":
            # Expected counts per bin: mu_i = N * pdf(dt_i) * bin_width
            mu = np.maximum(total_counts * pdf * bin_width, 1e-12)
            # Baker-Cousins Poisson deviance (exact even when counts == 0 in tail)
            pos = counts > 0
            deviance = 2.0 * np.sum(mu - counts)
            deviance -= 2.0 * np.sum(counts[pos] * np.log(mu[pos] / counts[pos]))
            return deviance
        else:
            # Weighted least-squares (chi-square)
            sigma = np.sqrt(np.maximum(counts, 1.0)) / (total_counts * bin_width)
            return np.sum(((density - pdf) / sigma) ** 2)

    res = minimize(
        loss,
        x0=np.array(p0, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 8000},
    )

    return {
        "popt": res.x,
        "lam": res.x[0],
        "tau1": res.x[1],
        "tau2": None if is_single else res.x[2],
        "dt": dt,
        "density": density,
        "success": res.success,
    }

# # =====================================================================
# # 5. END-TO-END EXAMPLE: PRODUCE DATA -> FIT -> VERIFY
# # =====================================================================
# if __name__ == "__main__":
#     # 1. Produce synthetic experimental data using analyze_series_deadtime
#     sim = analyze_series_deadtime(
#         arrangement="NE-E",
#         lam=65.0,
#         tau1=0.008,
#         tau2=0.022,
#         duration=1500.0,
#         seed=7,
#     )
#     exp_intervals = sim["intervals"]

#     # 2. Fit the 3 parameters (lam, tau1, tau2) from the intervals
#     fit = fit_deadtime_mle(exp_intervals, arrangement="NE-E")
#     popt = fit["popt"]

#     # 3. Plot experimental histogram, true analytical PDF, and fitted PDF
#     fig, ax = plt.subplots(figsize=(7, 3.5))
#     ax.hist(exp_intervals, bins=80, density=True, alpha=0.45, color="steelblue", label="Simulated / Exp. Data")
#     ax.plot(sim["t_grid"], sim["analytical_pdf"], "k-", lw=1.5, label="True Analytical PDF")
#     ax.plot(sim["t_grid"], evaluate_series_tid(sim["t_grid"], "NE-E", *popt), "r--", lw=2, label="Fitted PDF")

#     param_names = [r"\lambda", r"\tau_1", r"\tau_2"]
#     label = "\n".join(fr"${name}$ = {val:.5f}" for name, val in zip(param_names, popt))
#     ax.text(
#         0.97, 0.93, label,
#         transform=ax.transAxes, ha="right", va="top",
#         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
#     )
#     ax.set_xlabel("Inter-arrival Time $t$ (s)")
#     ax.set_ylabel("Probability Density")
#     ax.legend(loc="center right")
#     plt.tight_layout()
#     plt.show()