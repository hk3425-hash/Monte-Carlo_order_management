"""
features.py — Feature engineering: tick ranges, EWMA/EWMV, causal state labels.

Functions
---------
compute_ranges          : add integer tick-count R, R_up, R_dn and ret columns
ewma_ewmv               : Algorithm 1 — strictly causal EWMA and EWMV
quantile_states_causal  : causal quantile-bin assignment (no look-ahead)
add_states              : compute all EWMA features and assign state labels
"""
import numpy as np
import pandas as pd
from bisect import insort, bisect_right

import config


def compute_ranges(df: pd.DataFrame, eps: float, verbose: bool = True) -> pd.DataFrame:
    """Add integer tick-count columns R, R_up, R_dn and the raw return ret."""
    d = df.copy()
    R_raw   = ((d["high"] - d["low"])  / eps).round()
    Rup_raw = ((d["high"] - d["open"]) / eps).round()
    Rdn_raw = ((d["open"] - d["low"])  / eps).round()

    if verbose:
        for name, raw in [("R", R_raw), ("R_up", Rup_raw), ("R_dn", Rdn_raw)]:
            n_neg = (raw < 0).sum()
            if n_neg > 0:
                print(f"  WARNING: {n_neg} bars have negative {name} — investigate")

    d["R"]    = R_raw.clip(lower=0).astype(int)
    d["R_up"] = Rup_raw.clip(lower=0).astype(int)
    d["R_dn"] = Rdn_raw.clip(lower=0).astype(int)
    d["ret"]  = d["close"] - d["open"]

    clean_mask = (R_raw >= 0) & (Rup_raw >= 0) & (Rdn_raw >= 0)
    mismatch = ((d.loc[clean_mask, "R"] - d.loc[clean_mask, "R_up"]
                                        - d.loc[clean_mask, "R_dn"]).abs() > 1).sum()
    if verbose:
        if mismatch:
            print(f"  Warning: {mismatch} clean bars where R ≠ R_up + R_dn (rounding artefact)")
        else:
            print("  Range identity R = R_up + R_dn  ✓")
    return d


def ewma_ewmv(series: np.ndarray, lam: float):
    """
    Algorithm 1 from the project spec.
    Returns (ewma, ewmv) arrays of length n. At bar j the value uses η_{j-1}
    only — strictly causal, no look-ahead.
    """
    n = len(series)
    ewma_arr = np.zeros(n)
    ewmv_arr = np.zeros(n)
    sumW = sumWX = sumWSS = 0.0

    for j in range(1, n):
        eta = series[j - 1]
        if j == 1:
            sumW        = 1.0
            sumWX       = eta
            ewma_arr[j] = sumWX / sumW
            sumWSS      = (eta - ewma_arr[j]) ** 2
            ewmv_arr[j] = np.sqrt(sumWSS / sumW)
        else:
            sumW        = lam * sumW   + 1
            sumWX       = lam * sumWX  + eta
            ewma_arr[j] = sumWX / sumW
            sumWSS      = lam * sumWSS + (eta - ewma_arr[j]) ** 2
            ewmv_arr[j] = np.sqrt(sumWSS / sumW)
    return ewma_arr, ewmv_arr


def quantile_states_causal(series: pd.Series, n_states: int,
                           min_obs: int = 30) -> pd.Series:
    """Strictly causal quantile-bin assignment. At bar j the bin edges are
    derived from series[0..j-1] only — pd.qcut would use the full series and
    leak. Bars before min_obs get label 0."""
    values = series.values
    n      = len(values)
    states = np.zeros(n, dtype=int)
    fracs  = np.linspace(0, 1, n_states + 1)[1:-1]

    hist = []
    for j in range(n):
        x = values[j]
        m = len(hist)
        if j >= min_obs and m >= n_states and not np.isnan(x):
            thresholds = sorted({hist[int(f * m)] for f in fracs})
            states[j]  = min(bisect_right(thresholds, x), n_states - 1)
        if not np.isnan(x):
            insort(hist, x)
    return pd.Series(states, index=series.index, dtype=int)


def add_states(df_tau: pd.DataFrame,
               m_vol: int = None,
               n_sig: int = None,
               k_dir: int = None) -> pd.DataFrame:
    """Compute EWMA features and assign causal state labels in-place."""
    if m_vol is None: m_vol = config.M_VOL_STATES
    if n_sig is None: n_sig = config.N_SIG_STATES
    if k_dir is None: k_dir = config.K_DIR_STATES

    d = df_tau.copy()
    lam = config.LAM
    ewma_vol, ewmv_vol = ewma_ewmv(d["volume"].values, lam)
    ewma_rng, ewmv_rng = ewma_ewmv(d["R"].values,      lam)
    ewma_ret, ewmv_ret = ewma_ewmv(d["ret"].values,    lam)

    d["ewma_vol"] = ewma_vol
    d["ewmv_vol"] = ewmv_vol
    d["ewma_rng"] = ewma_rng
    d["ewmv_rng"] = ewmv_rng
    d["ewma_ret"] = ewma_ret
    d["ewmv_ret"] = ewmv_ret
    d["delta_x"]  = d["open"].diff()

    d["state_vol"] = quantile_states_causal(d["ewma_vol"].shift(1), m_vol)
    d["state_sig"] = quantile_states_causal(d["ewmv_rng"].shift(1), n_sig)
    d["state_dir"] = quantile_states_causal(d["delta_x"].shift(1),  k_dir)
    return d
