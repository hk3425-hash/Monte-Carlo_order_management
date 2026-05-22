"""
epdf.py — Empirical PDF machinery: unconditional ePDF, fill probability,
KL divergence, running conditional ePDF table, and walk-forward builder.

Classes
-------
ConditionalEPDF         : running count table counts[m][n][k][ell]

Functions
---------
epdf_from_array         : unconditional P(R = ell) from a flat array
fill_prob_from_pmf      : P(R >= k) survival function
kl_div                  : KL(p || q) with log-zero guard
build_rolling_epdfs     : walk-forward OOS ePDF builder
full_cond_epdf          : full-sample conditional ePDF over all state cells
"""
import numpy as np
import pandas as pd

import config


def epdf_from_array(vals: np.ndarray, max_ell: int = None) -> np.ndarray:
    """Unconditional P(R = ell) for ell = 0..max_ell from a flat array."""
    if max_ell is None:
        max_ell = config.MAX_SPREADS
    counts = np.bincount(np.clip(vals.astype(int), 0, max_ell), minlength=max_ell + 1)
    return counts / counts.sum() if counts.sum() > 0 else counts


def fill_prob_from_pmf(pmf: np.ndarray) -> np.ndarray:
    """P(R >= k) survival function for k = 0..len(pmf)-1."""
    return np.array([pmf[k:].sum() for k in range(len(pmf))])


def kl_div(p, q, eps=1e-12):
    """KL(p || q) with a small floor to avoid log(0)."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


class ConditionalEPDF:
    """Running count table: counts[m][n][k][ell] += 1 each time R=ell is observed
    when state=(vol=m, sig=n, dir=k). Conditioning on direction is optional —
    set K_DIR_STATES=1 and the dimension collapses."""

    def __init__(self, m: int, n: int, k: int, max_ell: int = None):
        if max_ell is None:
            max_ell = config.MAX_SPREADS
        self.counts  = np.zeros((m, n, k, max_ell + 1), dtype=float)
        self.max_ell = max_ell

    def update(self, sv: int, ss: int, sd: int, ell: int):
        self.counts[sv, ss, sd, min(ell, self.max_ell)] += 1

    def pmf(self, sv: int, ss: int, sd: int) -> np.ndarray:
        c = self.counts[sv, ss, sd]
        total = c.sum()
        if total < 10:
            return np.ones(self.max_ell + 1) / (self.max_ell + 1)
        return c / total

    def fill_prob(self, sv: int, ss: int, sd: int, ell: int) -> float:
        p = self.pmf(sv, ss, sd)
        return float(p[ell:].sum())


def build_rolling_epdfs(df: pd.DataFrame, j_start: int = None):
    """Walk forward bar by bar: query the ePDF for the upcoming bar's
    fill-probabilities (OOS), then update the count table with the realised
    outcome. Returns three ConditionalEPDF objects and a per-bar DataFrame."""
    if j_start is None:
        j_start = config.J_START

    m = config.M_VOL_STATES
    n = config.N_SIG_STATES
    k = config.K_DIR_STATES

    epdf_R   = ConditionalEPDF(m, n, k)
    epdf_Rup = ConditionalEPDF(m, n, k)
    epdf_Rdn = ConditionalEPDF(m, n, k)

    sv_arr = df["state_vol"].values
    ss_arr = df["state_sig"].values
    sd_arr = df["state_dir"].values
    R_arr  = df["R"].values
    Ru_arr = df["R_up"].values
    Rd_arr = df["R_dn"].values

    records = []
    for j in range(1, len(df)):
        sv, ss, sd = int(sv_arr[j]), int(ss_arr[j]), int(sd_arr[j])

        if j >= j_start:
            row = {
                "timestamp"  : df.index[j],
                "sv": sv, "ss": ss, "sd": sd,
                "R_actual"   : R_arr[j],
                "Rup_actual" : Ru_arr[j],
                "Rdn_actual" : Rd_arr[j],
                "open"       : df["open"].iloc[j],
                "close"      : df["close"].iloc[j],
            }
            for ell in range(1, 11):
                row[f"fp_rup_{ell}"] = epdf_Rup.fill_prob(sv, ss, sd, ell)
                row[f"fp_rdn_{ell}"] = epdf_Rdn.fill_prob(sv, ss, sd, ell)
            records.append(row)

        epdf_R.update(sv, ss, sd,   R_arr[j])
        epdf_Rup.update(sv, ss, sd, Ru_arr[j])
        epdf_Rdn.update(sv, ss, sd, Rd_arr[j])

    bt = pd.DataFrame(records).set_index("timestamp") if records else pd.DataFrame()
    return epdf_R, epdf_Rup, epdf_Rdn, bt


def full_cond_epdf(df: pd.DataFrame, target: str = "R_dn") -> dict:
    """Compute the conditional ePDF and fill-prob for every (vol, sig, dir) cell."""
    out = {}
    for m in range(config.M_VOL_STATES):
        for n in range(config.N_SIG_STATES):
            for k in range(config.K_DIR_STATES):
                mask = (
                    (df["state_vol"] == m) &
                    (df["state_sig"] == n) &
                    (df["state_dir"] == k)
                )
                vals = df.loc[mask, target].values
                p    = epdf_from_array(vals) if len(vals) > 0 else np.zeros(config.MAX_SPREADS + 1)
                fp   = fill_prob_from_pmf(p)
                out[(m, n, k)] = (p, fp, int(mask.sum()))
    return out
