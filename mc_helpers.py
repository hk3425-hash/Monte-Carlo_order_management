#!/usr/bin/env python
# coding: utf-8

# # Term Project 2 — Volatility-Volume-based Order Management
# **Columbia University — IEOR4703 Monte Carlo Simulation Methods (Hirsa)**
# 
# Strategy: adjust limit order placement (slicing) based on empirical PDFs of range, rangeUp, and rangeDn, conditioned on volume and volatility regime.
# 
# ---
# 
# ### Notebook structure
# 
# - **Setup** — imports, configuration, plotting style, per-market metadata.
# - **Helpers** — every reusable function defined in one place so any market can be analyzed identically.
# - **Main pipeline** — runs the helpers for the current `INSTRUMENT`. Re-run only this cell after changing the instrument.
# - **Part 1 — Reproduction and extensions of the paper** (state heatmap, Figure 2 reproduction, full conditional ePDF, KL information gain, non-stationarity, cross-market, τ-sensitivity, buy/sell asymmetry, OOS placement, state stability).
# - **Part 2 — Trading application** (conditional ePDF visualization, fill-probability curves, backtest, AIAgent calibration, parameter sweep).
# 

# ## Setup — imports, configuration, plotting style

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from bisect import insort, bisect_right

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# USER-CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────────────────────
INSTRUMENT   = "Nasdaq"   # change to any key in MARKETS below
TAU          = 15         # holding period in minutes
HALF_LIFE    = 20         # EWMA half-life in number of τ-bars
M_VOL_STATES = 3          # volume regime bins  (Low / Mid / High)
N_SIG_STATES = 3          # volatility regime bins (Low / Mid / High)
K_DIR_STATES = 3          # price-direction bins (Down / Flat / Up)
J_START      = 100        # min bars before ePDF is used (burn-in)
MAX_SPREADS  = 150        # max ℓ tracked in ePDF histograms
MAX_T_PLOT   = 25         # x-axis cap for ePDF / fill-prob plots
MIN_BAR_FRAC = 0.90       # min fraction of expected bars per day to keep that day

DATA_ROOT = Path("/Users/cemokutan/Documents/Monte_Carlo/project/data")
FIG_DIR   = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PER-MARKET METADATA
#   tick  : exchange tick size
#   rth   : (start_hhmm, end_hhmm, tz_label) — native timezone of CSV
#           (00:00, 23:59) marks a 24h market — RTH filter is skipped
# ─────────────────────────────────────────────────────────────
MARKETS = {
    "Nasdaq"                                 : {"tick": 0.25,   "rth": ("09:30", "16:00", "ET (US equity)")},
    "Gold"                                   : {"tick": 0.10,   "rth": ("08:20", "13:30", "ET (COMEX pit hours)")},
    "German Bunds - German Government Bonds" : {"tick": 0.01,   "rth": ("08:00", "17:00", "CET (Eurex)")},
    "EuroStoxx"                              : {"tick": 0.50,   "rth": ("09:00", "17:30", "CET (Eurex)")},
    "GBP - British Pound"                    : {"tick": 0.0100, "rth": ("00:00", "23:59", "FX 24h (no filter)")},
    "HeatingOil"                             : {"tick": 0.0100, "rth": ("09:00", "14:30", "ET (NYMEX pit)")},
    "JPY - Japanese Yen"                     : {"tick": 0.0050, "rth": ("00:00", "23:59", "FX 24h (no filter)")},
}

EPS = MARKETS[INSTRUMENT]["tick"]
LAM = 2 ** (-1 / HALF_LIFE)   # EWMA decay factor

# ─────────────────────────────────────────────────────────────
# LATEX-FRIENDLY PLOTTING STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'      : 110,
    'savefig.dpi'     : 200,
    'savefig.bbox'    : 'tight',
    'savefig.format'  : 'pdf',
    'font.family'     : 'serif',
    'font.size'       : 11,
    'axes.titlesize'  : 12,
    'axes.labelsize'  : 11,
    'legend.fontsize' : 10,
    'xtick.labelsize' : 10,
    'ytick.labelsize' : 10,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
})

def savefig(fig, name):
    """Save figure as PDF (for LaTeX) and PNG (for preview)."""
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png', dpi=150)
    print(f'  saved → figures/{name}.pdf')

if __name__ == "__main__":
    print(f"Instrument : {INSTRUMENT}")
    print(f"Tick size  : {EPS}")
    print(f"Tau        : {TAU} min")
    print(f"Lambda     : {LAM:.6f}  (half-life={HALF_LIFE} bars)")


# ## Helpers — all reusable functions
# 
# Defined once, used everywhere. Loading, chaining, resampling, range extraction, EWMA, state classification, conditional ePDF, KL divergence, plus a one-call `prepare_market` pipeline so any market can be analyzed identically.

# In[ ]:


# ============================================================
# 1. DATA LOADING & CONTRACT CHAINING
# ============================================================

def load_contract(path: Path) -> pd.DataFrame:
    """Load a single 1-min OHLCV CSV (no header row)."""
    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y.%m.%d.%H:%M:%S")
    df = df.set_index("timestamp").sort_index()
    return df


def sticky_roll(daily_vol: pd.DataFrame, min_consecutive: int = 3,
                verbose: bool = True) -> pd.Series:
    """
    Walk forward through calendar dates. Roll to the next contract only when
    its daily volume has exceeded the current contract's volume for
    `min_consecutive` days in a row. Once rolled, never roll back.

    Prevents the idxmax() oscillation problem where two contracts with nearly
    equal volume flip back and forth on adjacent days, creating artificial
    price jumps in the concatenated series.

    Note: Roll dates introduce a small price discontinuity (cost-of-carry
    spread between contracts). We do not back-adjust because the downstream
    analysis is on intra-bar range statistics (H-L, H-O, O-L), which are
    invariant to constant price shifts. Range on the roll-day bar itself
    may be inflated and is treated as a noise bar.
    """
    cols        = list(daily_vol.columns)
    active_name = cols[0]
    next_idx    = 1
    streak      = 0
    result      = pd.Series(index=daily_vol.index, dtype=object)

    for date in daily_vol.index:
        if next_idx < len(cols):
            next_name = cols[next_idx]
            curr_vol  = daily_vol.loc[date, active_name]
            cand_vol  = daily_vol.loc[date, next_name]
            if cand_vol > curr_vol:
                streak += 1
            else:
                streak = 0
            if streak >= min_consecutive:
                if verbose:
                    print(f"  Roll: {active_name} → {next_name}  on {date.date()}")
                active_name = next_name
                next_idx   += 1
                streak      = 0
        result[date] = active_name
    return result


def load_instrument(instrument: str, min_bar_frac: float = MIN_BAR_FRAC,
                    roll_days: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Load all contract CSVs for an instrument, chain them via sticky-roll,
    and apply a day-completeness filter at the 1-min level."""
    folder = DATA_ROOT / instrument
    csv_files = sorted(
        [f for f in folder.glob("*.csv") if not f.stem.startswith("AIAgent")],
        key=lambda f: f.stem,
    )
    if verbose:
        print(f"Contracts found: {[f.stem for f in csv_files]}")

    contracts = {f.stem: load_contract(f) for f in csv_files}
    daily_vol = pd.DataFrame(
        {name: df["volume"].resample("D").sum() for name, df in contracts.items()}
    ).fillna(0)
    active = sticky_roll(daily_vol, min_consecutive=roll_days, verbose=verbose)

    frames = []
    for name, df in contracts.items():
        active_days = active[active == name].index
        mask = df.index.normalize().isin(active_days)
        frames.append(df.loc[mask])

    raw = pd.concat(frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    bars_per_day = raw.resample("D").size()
    expected     = bars_per_day.median()
    good_days    = bars_per_day[bars_per_day >= min_bar_frac * expected].index
    raw = raw[raw.index.normalize().isin(good_days)]

    if verbose:
        print(f"1-min bars after roll+filter: {len(raw):,}")
        print(f"Date range: {raw.index.min().date()} → {raw.index.max().date()}")
    return raw


# ============================================================
# 2. RESAMPLING & RTH FILTER
# ============================================================

def resample_ohlcv(df: pd.DataFrame, tau: int, verbose: bool = True) -> pd.DataFrame:
    """Aggregate 1-min bars into τ-min OHLCV bars."""
    rule = f"{tau}min"
    agg = df.resample(rule, label="left", closed="left").agg(
        open   = ("open",   "first"),
        high   = ("high",   "max"),
        low    = ("low",    "min"),
        close  = ("close",  "last"),
        volume = ("volume", "sum"),
    ).dropna(subset=["open"])
    agg = agg[agg["volume"] > 0]
    if verbose:
        print(f"τ={tau} min bars: {len(agg):,}")
    return agg


def apply_rth_filter(df_tau: pd.DataFrame, instrument: str, tau: int,
                     min_bar_frac: float = MIN_BAR_FRAC,
                     verbose: bool = True) -> pd.DataFrame:
    """Restrict τ-bars to the market's declared RTH window, then re-apply the
    day-completeness filter at the τ-bar level. 24h markets are passed through."""
    rth_start_str, rth_end_str, tz_label = MARKETS[instrument]["rth"]
    rth_start_min = int(rth_start_str[:2]) * 60 + int(rth_start_str[3:])
    rth_end_min   = int(rth_end_str[:2])   * 60 + int(rth_end_str[3:])
    skip_rth      = (rth_start_min == 0 and rth_end_min >= 24 * 60 - 1)

    before = len(df_tau)
    if not skip_rth:
        # last bar starts τ min before RTH end
        rth_filter_end_min = rth_end_min - tau
        rth_filter_end_str = f"{rth_filter_end_min // 60:02d}:{rth_filter_end_min % 60:02d}"
        df_tau = df_tau.between_time(rth_start_str, rth_filter_end_str)

    expected_bars_per_day = (rth_end_min - rth_start_min) // tau if not skip_rth else (24 * 60) // tau
    df_tau = df_tau.copy()
    df_tau["_date"] = df_tau.index.normalize()
    bars_per_day = df_tau.groupby("_date").size()
    good_days    = bars_per_day[bars_per_day >= min_bar_frac * expected_bars_per_day].index
    df_tau       = df_tau[df_tau["_date"].isin(good_days)].drop(columns="_date")

    if verbose:
        if skip_rth:
            print(f"  RTH         : 24h (no filter)")
        else:
            print(f"  RTH         : [{rth_start_str}, {rth_end_str}] {tz_label}")
        print(f"  Before RTH  : {before:,} τ-bars")
        print(f"  After RTH   : {len(df_tau):,} τ-bars across {len(good_days)} good days")
    return df_tau


# ============================================================
# 3. RANGES, EWMA, STATE CLASSIFICATION
# ============================================================

def compute_ranges(df: pd.DataFrame, eps: float, verbose: bool = True) -> pd.DataFrame:
    """Add integer tick-count columns R, R_up, R_dn and the raw return ret."""
    d = df.copy()
    R_raw    = ((d["high"] - d["low"])  / eps).round()
    Rup_raw  = ((d["high"] - d["open"]) / eps).round()
    Rdn_raw  = ((d["open"] - d["low"])  / eps).round()

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
            sumW         = 1.0
            sumWX        = eta
            ewma_arr[j]  = sumWX / sumW
            sumWSS       = (eta - ewma_arr[j]) ** 2
            ewmv_arr[j]  = np.sqrt(sumWSS / sumW)
        else:
            sumW         = lam * sumW   + 1
            sumWX        = lam * sumWX  + eta
            ewma_arr[j]  = sumWX / sumW
            sumWSS       = lam * sumWSS + (eta - ewma_arr[j]) ** 2
            ewmv_arr[j]  = np.sqrt(sumWSS / sumW)
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
               m_vol: int = M_VOL_STATES,
               n_sig: int = N_SIG_STATES,
               k_dir: int = K_DIR_STATES) -> pd.DataFrame:
    """Compute EWMA features and assign causal state labels in-place."""
    d = df_tau.copy()
    ewma_vol, ewmv_vol = ewma_ewmv(d["volume"].values, LAM)
    ewma_rng, ewmv_rng = ewma_ewmv(d["R"].values,      LAM)
    ewma_ret, ewmv_ret = ewma_ewmv(d["ret"].values,    LAM)

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


# ============================================================
# 4. EMPIRICAL PDF & CONDITIONAL EPDF
# ============================================================

def epdf_from_array(vals: np.ndarray, max_ell: int = MAX_SPREADS) -> np.ndarray:
    """Unconditional P(R = ℓ) for ℓ = 0..max_ell from a flat array."""
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
    """Running count table: counts[m][n][k][ℓ] += 1 each time R=ℓ is observed
    when state=(vol=m, sig=n, dir=k). Conditioning on direction is optional —
    set K_DIR_STATES=1 in Setup and the dimension collapses."""

    def __init__(self, m: int, n: int, k: int, max_ell: int = MAX_SPREADS):
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


def build_rolling_epdfs(df: pd.DataFrame, j_start: int = J_START):
    """Walk forward bar by bar: query the ePDF for the upcoming bar's
    fill-probabilities (OOS), then update the count table with the realised
    outcome. Returns three ConditionalEPDF objects and a per-bar DataFrame."""
    epdf_R   = ConditionalEPDF(M_VOL_STATES, N_SIG_STATES, K_DIR_STATES)
    epdf_Rup = ConditionalEPDF(M_VOL_STATES, N_SIG_STATES, K_DIR_STATES)
    epdf_Rdn = ConditionalEPDF(M_VOL_STATES, N_SIG_STATES, K_DIR_STATES)

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
                "timestamp" : df.index[j],
                "sv": sv, "ss": ss, "sd": sd,
                "R_actual"  : R_arr[j],
                "Rup_actual": Ru_arr[j],
                "Rdn_actual": Rd_arr[j],
                "open"      : df["open"].iloc[j],
                "close"     : df["close"].iloc[j],
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


# ============================================================
# 5. ONE-CALL PIPELINE
# ============================================================

def prepare_market(instrument: str, tau: int = TAU,
                   verbose: bool = True) -> tuple:
    """Full pipeline: load → chain → resample → RTH filter → ranges → states.
    Returns (df_tau, eps)."""
    if verbose:
        print(f"\n── prepare_market: {instrument}, τ={tau} ──")
    eps   = MARKETS[instrument]["tick"]
    df_1m = load_instrument(instrument, verbose=verbose)
    df_t  = resample_ohlcv(df_1m, tau, verbose=verbose)
    df_t  = apply_rth_filter(df_t, instrument, tau, verbose=verbose)
    df_t  = compute_ranges(df_t, eps, verbose=verbose)
    df_t  = add_states(df_t)
    return df_t, eps


def full_cond_epdf(df: pd.DataFrame, target: str = "R_dn") -> dict:
    """Compute the conditional ePDF and fill-prob for every (vol, sig, dir) cell."""
    out = {}
    for m in range(M_VOL_STATES):
        for n in range(N_SIG_STATES):
            for k in range(K_DIR_STATES):
                mask = (df["state_vol"] == m) & (df["state_sig"] == n) & (df["state_dir"] == k)
                vals = df.loc[mask, target].values
                p    = epdf_from_array(vals) if len(vals) > 0 else np.zeros(MAX_SPREADS + 1)
                fp   = fill_prob_from_pmf(p)
                out[(m, n, k)] = (p, fp, int(mask.sum()))
    return out


# ## Main pipeline — run helpers for the current `INSTRUMENT`
# 
# This produces `df_tau` (per-bar dataframe with ranges and states), `epdf_R/Rup/Rdn` (state-conditioned ePDFs built walk-forward), and `bt` (a per-bar record table from bar `J_START` onward — used in Part 2 for the backtest and calibration). Re-run only this cell after changing `INSTRUMENT` in Setup.

# In[ ]:


if __name__ == "__main__":
    df_tau, EPS = prepare_market(INSTRUMENT, TAU)
    print(f"\n── Building state-conditioned ePDFs (walk-forward) ──")
    epdf_R, epdf_Rup, epdf_Rdn, bt = build_rolling_epdfs(df_tau)
    print(f"  ePDF tables built. Backtest rows: {len(bt):,}")

    df_tau.head()


    # ## Timezone diagnostic (sanity check)
    # 
    # We do not convert timezones — we assume the CSV timestamps are already in the market's native exchange timezone (as declared in `MARKETS[...]["rth"]`). This cell verifies that assumption: peak intraday volume should align with the cash open. A `⚠` here means the RTH dict entry is wrong or the CSV is in a different timezone.

    # In[ ]:


    # Reload 1-min data (without filters) just for the diagnostic
    df_1min_diag = load_instrument(INSTRUMENT, verbose=False)
    rth_start_str, rth_end_str, tz_label = MARKETS[INSTRUMENT]["rth"]
    rth_start_min = int(rth_start_str[:2]) * 60 + int(rth_start_str[3:])
    rth_end_min   = int(rth_end_str[:2])   * 60 + int(rth_end_str[3:])

    print(f"── Timezone diagnostic for {INSTRUMENT} ──")
    print(f"Expected RTH: {rth_start_str}–{rth_end_str} {tz_label}")
    print(f"Expected peak hour: {rth_start_min // 60:02d} (cash open)\n")

    vol_by_hhmm = (
        df_1min_diag.assign(hhmm=df_1min_diag.index.strftime("%H:%M"))
                    .groupby("hhmm")["volume"].sum()
    )
    print("Top 10 highest-volume minutes (across all days):")
    print(vol_by_hhmm.sort_values(ascending=False).head(10))

    hourly = df_1min_diag.groupby(df_1min_diag.index.hour)["volume"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hourly.index, hourly.values, color="steelblue", edgecolor="white")
    ax.axvspan(rth_start_min / 60, rth_end_min / 60, alpha=0.15, color="green",
               label=f"Declared RTH ({rth_start_str}–{rth_end_str})")
    ax.set_xlabel("Hour of day (native timezone)")
    ax.set_ylabel("Mean 1-min volume")
    ax.set_title(f"{INSTRUMENT} — intraday volume profile  ({tz_label})")
    ax.set_xticks(range(24))
    ax.legend()
    plt.tight_layout()
    savefig(fig, f"fig00_tz_diagnostic_{INSTRUMENT.split()[0]}")
    plt.show()

    peak_hour = int(hourly.idxmax())
    print(f"\nObserved peak hour: {peak_hour:02d}:00")
    print(f"Declared open hour: {rth_start_min // 60:02d}:{rth_start_min % 60:02d}")
    if abs(peak_hour - rth_start_min // 60) <= 1:
        print("✓ Peak hour aligns with declared RTH open — timezone looks correct.")
    else:
        print("⚠ Peak hour does NOT align — check the RTH_BY_INSTRUMENT entry.")


    # ## Part 1.1 — Unconditional empirical PDF
    # 
    # The unconditional ePDFs of the three range processes (total range R, upside range R_up, downside range R_dn). These serve as the baseline against which the state-conditioned ePDFs will be compared (KL divergence diagnostic later).

    # In[ ]:


    sub = df_tau.iloc[J_START:]
    pdf_R   = epdf_from_array(sub["R"].values)
    pdf_Rup = epdf_from_array(sub["R_up"].values)
    pdf_Rdn = epdf_from_array(sub["R_dn"].values)

    ell = np.arange(MAX_T_PLOT)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, pdf, title in zip(axes,
                              [pdf_R, pdf_Rup, pdf_Rdn],
                              ["P(R = ℓ)", "P(R_up = ℓ)", "P(R_dn = ℓ)"]):
        ax.bar(ell, pdf[:MAX_T_PLOT], color="steelblue", edgecolor="k", linewidth=0.3)
        ax.set_xlabel("ℓ  (number of spreads)")
        ax.set_ylabel("Probability")
        ax.set_title(title)
    plt.suptitle(f"Unconditional ePDFs — {INSTRUMENT}  τ={TAU} min  (bars {J_START}+)", y=1.02)
    plt.tight_layout()
    savefig(fig, f"fig01_unconditional_epdf_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.2 — EWMA dynamics (Algorithm 1)
    # 
    # Visual sanity check of the EWMA/EWMV recursion. The smoothed series should track the noisy raw series while filtering the jitter — visible regime shifts (e.g. COVID) should leave a clear footprint.

    # In[ ]:


    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(df_tau.index, df_tau["volume"],   alpha=0.3, label="volume")
    axes[0].plot(df_tau.index, df_tau["ewma_vol"], lw=1.5,    label="EWMA volume")
    axes[0].set_ylabel("Volume"); axes[0].legend()
    axes[1].plot(df_tau.index, df_tau["R"],        alpha=0.3, label="range (ticks)")
    axes[1].plot(df_tau.index, df_tau["ewma_rng"], lw=1.5,    label="EWMA range")
    axes[1].set_ylabel("Range"); axes[1].legend()
    plt.suptitle(f"{INSTRUMENT} — EWMA (half-life={HALF_LIFE} τ-bars)")
    plt.tight_layout()
    savefig(fig, f"fig02_ewma_dynamics_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.3 — Joint state occupancy
    # 
    # By construction, the marginal occupancies of `state_vol` and `state_sig` are roughly uniform (1/3 each). Any deviation in the *joint* distribution is informative — it reveals how volume and volatility regimes co-move.

    # In[ ]:


    from scipy.stats import chi2_contingency

    ct = pd.crosstab(df_tau["state_vol"], df_tau["state_sig"], normalize="all") * 100

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(ct.values, cmap="YlOrRd", vmin=0, aspect="auto")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, f"{ct.iloc[i,j]:.1f}%", ha="center", va="center",
                    color="black" if ct.iloc[i,j] < 12 else "white", fontweight="bold")
    ax.set_xticks(range(N_SIG_STATES))
    ax.set_yticks(range(M_VOL_STATES))
    ax.set_xticklabels([f"σ={n}" for n in range(N_SIG_STATES)])
    ax.set_yticklabels([f"v={m}" for m in range(M_VOL_STATES)])
    ax.set_xlabel("Volatility state")
    ax.set_ylabel("Volume state")
    ax.set_title("Joint frequency of (volume, volatility) states")
    plt.colorbar(im, ax=ax, label="% of intervals")
    plt.tight_layout()
    savefig(fig, f"fig03_state_heatmap_{INSTRUMENT.split()[0]}")
    plt.show()

    ct_counts = pd.crosstab(df_tau["state_vol"], df_tau["state_sig"])
    chi2, pval, dof, _ = chi2_contingency(ct_counts)
    print(f"\nχ² test of (volume, volatility) independence:")
    print(f"  χ² = {chi2:.1f},  dof = {dof},  p-value = {pval:.2e}")
    print("  → small p ⇒ states are strongly correlated (high volatility tends to coincide with high volume)")


    # ## Part 1.4 — Reproduction of Figure 2 from the paper
    # 
    # Four-panel plot for the three volatility states: (i) counts of RangeDn, (ii) counts of getting filled, (iii) ePDF, (iv) fill probability.

    # In[ ]:


    df_js = df_tau.iloc[J_START:].copy()

    by_sigma = {}
    for s in sorted(df_js["state_sig"].unique()):
        sub  = df_js[df_js["state_sig"] == s]
        p    = epdf_from_array(sub["R_dn"].values)
        fp   = fill_prob_from_pmf(p)
        by_sigma[int(s)] = (p, fp, len(sub))

    colors_seg = ["#1f4e79", "#2e9c8e", "#e6a700"]
    labels_seg = ["σ state 0 (low)", "σ state 1 (med)", "σ state 2 (high)"]
    width      = 0.27
    ell_plot   = np.arange(MAX_T_PLOT)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, s in enumerate(sorted(by_sigma)):
        p, fp, n_obs = by_sigma[s]
        counts = (p[:MAX_T_PLOT] * n_obs).astype(int)
        fill_counts = (fp[:MAX_T_PLOT] * n_obs).astype(int)
        off = (i - 1) * width
        axes[0,0].bar(ell_plot + off, counts,             width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)
        axes[0,1].bar(ell_plot + off, fill_counts,        width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)
        axes[1,0].bar(ell_plot + off, p[:MAX_T_PLOT],     width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)
        axes[1,1].bar(ell_plot + off, fp[:MAX_T_PLOT],    width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)

    axes[0,0].set_title("Counts of RangeDn")
    axes[0,0].set_xlabel("Number of spreads");  axes[0,0].set_ylabel("Counts")
    axes[0,1].set_title("Counts of being filled w.r.t. number of spreads")
    axes[0,1].set_xlabel("Number of spreads");  axes[0,1].set_ylabel("Fill counts")
    axes[1,0].set_title("ePDF of RangeDn")
    axes[1,0].set_xlabel("Number of spreads");  axes[1,0].set_ylabel("Probability")
    axes[1,1].set_title("Probability of being filled")
    axes[1,1].set_xlabel("k (ticks below open)"); axes[1,1].set_ylabel("P(fill)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, MAX_T_PLOT)

    fig.suptitle(f"Reproduction of Figure 2 — {INSTRUMENT}, τ = {TAU} min",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    savefig(fig, f"fig04_paper_figure2_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.5 — Full conditional ePDF grid
    # 
    # The complete grid P(R_dn | v, σ, Δx) over all 27 state cells. We organise it as a 3×3 panel of (v, σ) with the three Δx states overlaid inside each panel. Visual inspection confirms the conditional ePDFs differ markedly across cells, validating the paper's central premise.

    # In[ ]:


    cond = full_cond_epdf(df_js, "R_dn")

    fig, axes = plt.subplots(M_VOL_STATES, N_SIG_STATES, figsize=(12, 10),
                             sharex=True, sharey=True)
    dx_colors = ["#cc3333", "#888888", "#2266aa"]
    dx_labels = ["Δx state 0 (down)", "Δx state 1 (flat)", "Δx state 2 (up)"]
    ell_plot  = np.arange(MAX_T_PLOT)

    for m in range(M_VOL_STATES):
        for n in range(N_SIG_STATES):
            ax = axes[m, n]
            for k in range(K_DIR_STATES):
                p, fp, n_obs = cond[(m, n, k)]
                ax.bar(ell_plot + (k - 1) * 0.3, p[:MAX_T_PLOT], 0.3,
                       color=dx_colors[k],
                       label=f"{dx_labels[k]} (n={n_obs})" if (m == 0 and n == 0) else None,
                       alpha=0.85)
            ax.set_title(f"v={m}, σ={n}", fontsize=10)
            ax.set_xlim(-0.5, 18)
            if m == M_VOL_STATES - 1: ax.set_xlabel("ticks")
            if n == 0:                ax.set_ylabel("P(R_dn)")

    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Conditional ePDF P(R_dn | v, σ, Δx) — {INSTRUMENT}, τ={TAU} min",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig05_full_conditional_grid_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.6 — Table 1: quantitative effect of conditioning
    # 
    # P(R_dn ≥ k·ε) — the fill probability of a buy limit at k ticks below open — for every state cell, with the naive (unconditional) baseline at the bottom. Saved as CSV for the LaTeX report.

    # In[ ]:


    pdf_dn_naive = epdf_from_array(df_js["R_dn"].values)
    fp_naive     = fill_prob_from_pmf(pdf_dn_naive)

    k_targets = [1, 2, 3, 5, 8]
    rows = []
    for (m, n, k_), (p, fp, n_obs) in cond.items():
        rows.append({
            "(v, σ, Δx)": f"({m},{n},{k_})",
            "n_obs"    : n_obs,
            **{f"P(fill≥{k})": fp[k] for k in k_targets}
        })
    rows.append({
        "(v, σ, Δx)": "naive (all)",
        "n_obs"    : len(df_js),
        **{f"P(fill≥{k})": fp_naive[k] for k in k_targets}
    })

    tbl = pd.DataFrame(rows)
    tbl_display = tbl.copy()
    for col in tbl_display.columns[2:]:
        tbl_display[col] = tbl_display[col].apply(lambda x: f"{x:.2%}")
    print(tbl_display.to_string(index=False))

    tbl.to_csv(FIG_DIR / f"tab01_conditional_fill_probabilities_{INSTRUMENT.split()[0]}.csv", index=False)
    print(f"\n→ saved figures/tab01_conditional_fill_probabilities_{INSTRUMENT.split()[0]}.csv")


    # ## Part 1.7 — Non-stationarity of the naive ePDF
    # 
    # Splitting the data into three equal temporal segments and overlaying their unconditional ePDFs shows clear drift — proof that pooling all history into a single histogram throws away meaningful structure. This is the empirical justification for conditioning.

    # In[ ]:


    n         = len(df_tau)
    third     = n // 3
    segments  = [df_tau.iloc[:third], df_tau.iloc[third:2*third], df_tau.iloc[2*third:]]
    seg_names = ["Segment 1 (early)", "Segment 2 (middle)", "Segment 3 (late)"]
    seg_cols  = ["#1f4e79", "#2e9c8e", "#e6a700"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.27
    ell_plot = np.arange(MAX_T_PLOT)

    for i, (seg, name, c) in enumerate(zip(segments, seg_names, seg_cols)):
        p  = epdf_from_array(seg["R_dn"].values)
        fp = fill_prob_from_pmf(p)
        ax[0].bar(ell_plot + (i - 1) * width, p[:MAX_T_PLOT],  width, color=c, label=name, alpha=0.9)
        ax[1].bar(ell_plot + (i - 1) * width, fp[:MAX_T_PLOT], width, color=c, label=name, alpha=0.9)

    ax[0].set_title("ePDF of R_dn — three temporal segments")
    ax[0].set_xlabel("Number of ticks"); ax[0].set_ylabel("Probability"); ax[0].legend()
    ax[0].set_xlim(-0.5, MAX_T_PLOT)
    ax[1].set_title("Fill probability — three temporal segments")
    ax[1].set_xlabel("k (ticks below open)"); ax[1].set_ylabel("P(fill)"); ax[1].legend()
    ax[1].set_xlim(-0.5, MAX_T_PLOT)

    fig.suptitle(f"Distributions are NOT stationary — {INSTRUMENT}, τ = {TAU} min",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig06_segment_drift_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.8 — Information content of conditioning (KL divergence)
    # 
    # For each cell (m, n, k) we compute
    # 
    # $$\mathrm{KL}\big(P_{m,n,k}\,\|\,P_{\text{naive}}\big) \;=\; \sum_\ell P_{m,n,k}(\ell) \log\frac{P_{m,n,k}(\ell)}{P_{\text{naive}}(\ell)}.$$
    # 
    # The KL grid quantifies *how much* each state cell departs from the unconditional histogram. Cells with large KL are where conditioning has the highest economic value.

    # In[ ]:


    kl_grid = np.zeros((M_VOL_STATES, N_SIG_STATES, K_DIR_STATES))
    for (m, n, k_), (p, fp, n_obs) in cond.items():
        if n_obs > 0:
            kl_grid[m, n, k_] = kl_div(p, pdf_dn_naive)

    fig, axes = plt.subplots(1, K_DIR_STATES, figsize=(13, 4), sharey=True)
    vmax = kl_grid.max()
    for k_ in range(K_DIR_STATES):
        ax = axes[k_]
        im = ax.imshow(kl_grid[:, :, k_], cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        for m in range(M_VOL_STATES):
            for n in range(N_SIG_STATES):
                ax.text(n, m, f"{kl_grid[m,n,k_]:.2f}", ha="center", va="center",
                        color="white" if kl_grid[m,n,k_] < vmax/2 else "black",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(range(N_SIG_STATES))
        ax.set_yticks(range(M_VOL_STATES))
        ax.set_xticklabels([f"σ={n}" for n in range(N_SIG_STATES)])
        ax.set_yticklabels([f"v={m}" for m in range(M_VOL_STATES)])
        ax.set_xlabel("Volatility state")
        ax.set_title(f"Δx state = {k_}")
        if k_ == 0: ax.set_ylabel("Volume state")

    fig.colorbar(im, ax=axes, label="KL(cond ‖ naive)", shrink=0.85)
    fig.suptitle("Information gain from conditioning — KL divergence vs naive baseline",
                 fontweight="bold")
    savefig(fig, f"fig07_kl_grid_{INSTRUMENT.split()[0]}")
    plt.show()

    print(f"\nMean KL = {kl_grid.mean():.3f}")
    print(f"Max  KL = {kl_grid.max():.3f} at cell (v, σ, Δx) = "
          f"{tuple(np.unravel_index(kl_grid.argmax(), kl_grid.shape))}")


    # ## Part 1.9 — Cross-market generalization
    # 
    # Same methodology applied to all four AIAgent-equipped markets (or all 7 if you have data for them). The shapes of the ePDFs differ markedly across asset classes, but the basic structure holds.
    # 
    # **To enable the full sweep:** uncomment the `markets_to_sweep` line below. The default runs the primary market only to keep iteration fast.

    # In[ ]:


    # Markets to compare. Comment/uncomment to control scope.
    markets_to_sweep = [INSTRUMENT]
    # markets_to_sweep = ["Nasdaq", "Gold", "German Bunds - German Government Bonds", "EuroStoxx"]
    # markets_to_sweep = list(MARKETS.keys())   # all 7

    mkt_colors = plt.cm.tab10(np.linspace(0, 0.9, len(markets_to_sweep)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for i, name in enumerate(markets_to_sweep):
        df_m, eps_m = prepare_market(name, TAU, verbose=False)
        df_m_js     = df_m.iloc[J_START:]
        p_m         = epdf_from_array(df_m_js["R_dn"].values)
        fp_m        = fill_prob_from_pmf(p_m)
        label_name  = name.split()[0]
        axes[0].plot(np.arange(MAX_T_PLOT), p_m[:MAX_T_PLOT],
                     color=mkt_colors[i], lw=2, marker="o", markersize=4,
                     label=f"{label_name} (ε={eps_m})")
        axes[1].plot(np.arange(MAX_T_PLOT), fp_m[:MAX_T_PLOT],
                     color=mkt_colors[i], lw=2, marker="o", markersize=4,
                     label=label_name)

    axes[0].set_title("ePDF of R_dn")
    axes[0].set_xlabel("Number of ticks"); axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=9); axes[0].set_xlim(-0.5, MAX_T_PLOT)
    axes[1].set_title("Fill probability")
    axes[1].set_xlabel("k (ticks below open)"); axes[1].set_ylabel("P(fill)")
    axes[1].legend(fontsize=9); axes[1].set_xlim(-0.5, MAX_T_PLOT)
    fig.suptitle(f"Cross-market generalization — τ = {TAU} min", fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig08_cross_market")
    plt.show()


    # ## Part 1.10 — Sensitivity to the holding period τ
    # 
    # Longer τ shifts the fill-probability curves rightward (price has more time to travel). The optimal limit-order placement is therefore τ-specific, not universal.
    # 
    # **Note:** this re-runs the full pipeline at each τ, which takes some time. Uncomment to enable.

    # In[ ]:


    # Uncomment to enable:
    # TAU_GRID = [5, 10, 15, 30, 60]
    TAU_GRID = [TAU]   # default: just the current τ (no sweep)

    tau_cols = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(TAU_GRID), 2)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for i, tau in enumerate(TAU_GRID):
        df_t, eps_t = prepare_market(INSTRUMENT, tau, verbose=False)
        df_t_js     = df_t.iloc[J_START:]
        p_t         = epdf_from_array(df_t_js["R_dn"].values, max_ell=MAX_T_PLOT * 2)
        fp_t        = fill_prob_from_pmf(p_t)
        x           = np.arange(len(p_t))
        axes[0].plot(x, p_t,  color=tau_cols[i], lw=1.8, marker="o", markersize=3, label=f"τ = {tau} min")
        axes[1].plot(x, fp_t, color=tau_cols[i], lw=1.8, marker="o", markersize=3, label=f"τ = {tau} min")

    axes[0].set_title("ePDF of R_dn vs τ")
    axes[0].set_xlabel("Number of ticks"); axes[0].set_ylabel("Probability")
    axes[0].legend(); axes[0].set_xlim(-0.5, MAX_T_PLOT * 2)
    axes[1].set_title("Fill probability vs τ")
    axes[1].set_xlabel("k (ticks below open)"); axes[1].set_ylabel("P(fill)")
    axes[1].axhline(0.5, color="gray", lw=0.7, linestyle="--")
    axes[1].legend(); axes[1].set_xlim(-0.5, MAX_T_PLOT * 2)
    fig.suptitle(f"Sensitivity to holding period — {INSTRUMENT}", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig09_tau_sensitivity_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.11 — Buy/sell asymmetry conditioned on prior direction
    # 
    # We compare P(R_up ≥ k·ε) (sell-limit fills) versus P(R_dn ≥ k·ε) (buy-limit fills) across the three Δx states. Conditioning on prior direction exposes the momentum/mean-reversion structure: after a prior up-move, downside excursions become less likely and sell-limit orders become easier to fill.

    # In[ ]:


    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dx_titles = ["Δx state 0 (prior down)", "Δx state 1 (prior flat)", "Δx state 2 (prior up)"]
    ell_plot  = np.arange(MAX_T_PLOT)

    for k_ in range(K_DIR_STATES):
        sub  = df_js[df_js["state_dir"] == k_]
        p_up = epdf_from_array(sub["R_up"].values)
        p_dn = epdf_from_array(sub["R_dn"].values)
        fp_up = fill_prob_from_pmf(p_up)
        fp_dn = fill_prob_from_pmf(p_dn)
        axes[k_].plot(ell_plot, fp_up[:MAX_T_PLOT], color="seagreen", lw=2, marker="^", markersize=5,
                      label="P(sell limit filled, R_up)")
        axes[k_].plot(ell_plot, fp_dn[:MAX_T_PLOT], color="tomato",   lw=2, marker="v", markersize=5,
                      label="P(buy limit filled, R_dn)")
        axes[k_].set_title(dx_titles[k_])
        axes[k_].set_xlabel("k (ticks)"); axes[k_].set_ylabel("Fill probability")
        axes[k_].legend(fontsize=8)
        axes[k_].set_xlim(-0.5, MAX_T_PLOT); axes[k_].set_ylim(0, 1.02)

    fig.suptitle(f"Buy/Sell asymmetry conditioned on prior direction — {INSTRUMENT}, τ = {TAU} min",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig10_buy_sell_asymmetry_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.12 — Out-of-sample placement and slippage decision
    # 
    # Train/test 50/50 split. We derive the optimal placement k* from the training conditional ePDF (maximising k·ε·P(fill)), then evaluate the realised expected gain on the test half — comparing the state-conditioned choice against the naive baseline.

    # In[ ]:


    def cond_fill_curve(df: pd.DataFrame, target: str = "R_dn") -> dict:
        """{state_sigma: (pmf, fill_prob)}"""
        out = {}
        for s in sorted(df["state_sig"].unique()):
            sub = df[df["state_sig"] == s]
            p   = epdf_from_array(sub[target].values)
            out[int(s)] = (p, fill_prob_from_pmf(p))
        return out

    split        = len(df_tau) // 2
    train, test  = df_tau.iloc[J_START:split], df_tau.iloc[split:]
    train_curves = cond_fill_curve(train)
    test_curves  = cond_fill_curve(test)
    naive_train  = epdf_from_array(train["R_dn"].values)
    fp_train_naive = fill_prob_from_pmf(naive_train)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    seg_colors = ["#1f4e79", "#2e9c8e", "#e6a700"]

    # Panel 1: expected gain k * ε * P(fill) on the train curves
    ks = np.arange(MAX_T_PLOT)
    for s, (p, fp) in train_curves.items():
        eg = ks * EPS * fp[:MAX_T_PLOT]
        axes[0].plot(ks, eg, lw=2, marker="o", markersize=4, color=seg_colors[s % 3],
                     label=f"σ state {s}")
        axes[0].axvline(int(ks[np.argmax(eg)]), color=seg_colors[s % 3], ls="--", lw=0.7)
    axes[0].set_title(r"Expected gain  $k\cdot\epsilon\cdot P(\mathrm{fill})$  — train")
    axes[0].set_xlabel("k (ticks)"); axes[0].set_ylabel("Expected gain ($)")
    axes[0].legend(); axes[0].set_xlim(-0.5, MAX_T_PLOT)

    # Panel 2: OOS realised gain
    rows_oos = []
    for s, (p_tr, fp_tr) in train_curves.items():
        eg_tr   = ks * EPS * fp_tr[:MAX_T_PLOT]
        k_star  = int(ks[np.argmax(eg_tr)])
        fp_te   = test_curves[s][1] if s in test_curves else fp_train_naive
        realized = k_star * EPS * (fp_te[k_star] if k_star < len(fp_te) else 0.0)
        naive    = k_star * EPS * (fp_train_naive[k_star] if k_star < len(fp_train_naive) else 0.0)
        rows_oos.append([s, k_star, realized, naive])

    df_oos = pd.DataFrame(rows_oos, columns=["σ_state","k*","OOS realized gain","Naive baseline gain"])
    x = np.arange(len(df_oos))
    axes[1].bar(x - 0.2, df_oos["OOS realized gain"],   0.4, color="#1f4e79", label="State-conditioned")
    axes[1].bar(x + 0.2, df_oos["Naive baseline gain"], 0.4, color="#c0392b", label="Naive baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"σ={s} (k*={k})" for s, k in zip(df_oos["σ_state"], df_oos["k*"])])
    axes[1].set_title("Out-of-sample realized expected gain")
    axes[1].set_ylabel("Expected gain ($)"); axes[1].legend()

    fig.suptitle(f"Optimal limit-order placement — {INSTRUMENT}, τ = {TAU} min", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig11_oos_optimal_placement_{INSTRUMENT.split()[0]}")
    plt.show()

    print("\nOut-of-sample comparison:")
    print(df_oos.to_string(index=False))
    df_oos.to_csv(FIG_DIR / f"tab02_oos_optimal_placement_{INSTRUMENT.split()[0]}.csv", index=False)


    # Slippage decision surface (assumes half-spread fallback if limit fails)
    MARKET_FALLBACK_COST = 0.5 * EPS
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, (p, fp) in test_curves.items():
        slippage = (1 - fp[:MAX_T_PLOT]) * MARKET_FALLBACK_COST - ks * EPS * fp[:MAX_T_PLOT]
        ax.plot(ks, slippage, lw=2, marker="o", markersize=4, color=seg_colors[s % 3],
                label=f"σ state {s}")
        k_best = int(ks[np.argmin(slippage)])
        ax.scatter([k_best], [slippage[k_best]], s=90, edgecolor="black",
                   facecolor=seg_colors[s % 3], zorder=5, linewidth=1.2)
        ax.annotate(f"  $k^*$={k_best}", (k_best, slippage[k_best]), fontsize=9)

    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Expected slippage curves by volatility state")
    ax.set_xlabel("k (ticks below open)"); ax.set_ylabel("Expected slippage ($)")
    ax.legend(); ax.set_xlim(-0.5, MAX_T_PLOT)
    fig.suptitle(f"Slippage decision surface — {INSTRUMENT}, τ = {TAU}", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig12_slippage_decision_{INSTRUMENT.split()[0]}")
    plt.show()


    # ## Part 1.13 — State stability through time
    # 
    # A natural concern with regime-based methods is that the regimes themselves drift. Binning the time series into 10 chunks and counting state frequencies inside each chunk shows whether volatility-state occupancy is stationary — extreme periods can re-allocate mass into the high-σ bin.

    # In[ ]:


    n_chunks = 10
    chunk    = np.array_split(np.arange(len(df_tau)), n_chunks)
    freq     = np.zeros((N_SIG_STATES, n_chunks))

    for i, idxs in enumerate(chunk):
        sl  = df_tau.iloc[idxs]
        cnt = sl["state_sig"].value_counts(normalize=True).reindex(range(N_SIG_STATES), fill_value=0)
        freq[:, i] = cnt.values

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(freq, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=freq.max())
    ax.set_yticks(range(N_SIG_STATES))
    ax.set_yticklabels([f"σ state {s}" for s in range(N_SIG_STATES)])
    ax.set_xticks(range(n_chunks))
    ax.set_xticklabels([f"C{i+1}" for i in range(n_chunks)])
    ax.set_xlabel("Time chunk (early → late)")
    ax.set_title("Fraction of intervals in each volatility state, by time chunk")
    for i in range(N_SIG_STATES):
        for j in range(n_chunks):
            ax.text(j, i, f"{freq[i,j]*100:.0f}%", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Fraction")
    fig.suptitle(f"Volatility-state drift through time — {INSTRUMENT}", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"fig13_state_stability_{INSTRUMENT.split()[0]}")
    plt.show()

    print(f"\nStd dev of state-0 frequency across chunks: {freq[0].std():.3f}")
    print(f"  (low → stationary state assignments; high → drift)")


    # ---
    # # Part 2 — Trading application
    # 
    # The remaining cells implement the trading layer that consumes the conditional ePDFs: a fill-probability-based limit-order backtest, AIAgent out-of-sample calibration, and a parameter sweep.

    # ## Visualise Conditional ePDF (Figure 2 style — using built ePDFs)

    # In[ ]:


    # Reproduce Figure 2 from the paper:
    # top-left: counts; bottom-left: ePDF of rangeDn; top-right: fill counts; bottom-right: fill prob

    ell_x   = np.arange(MAX_SPREADS + 1)
    n_segs  = N_SIG_STATES
    colors  = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ss in range(n_segs):
        # Marginalise over vol and dir states
        cnt_rdn = epdf_Rdn.counts[:, ss, :, :].sum(axis=(0, 1))
        cnt_rup = epdf_Rup.counts[:, ss, :, :].sum(axis=(0, 1))
        total_rdn = cnt_rdn.sum()
        total_rup = cnt_rup.sum()
        pdf_rdn   = cnt_rdn / total_rdn   if total_rdn > 0 else cnt_rdn
        pdf_rup   = cnt_rup / total_rup   if total_rup > 0 else cnt_rup

        # Fill counts: at offset ell, number of times rangeDn >= ell
        fill_cnt = np.array([cnt_rdn[l:].sum() for l in ell_x])
        fill_p   = fill_cnt / total_rdn if total_rdn > 0 else fill_cnt

        w = 0.25
        offset = (ss - n_segs / 2) * w
        lbl = f"σ-state {ss}"

        axes[0, 0].bar(ell_x[:12] + offset, cnt_rdn[:12],  width=w, color=colors[ss], label=lbl, alpha=0.85)
        axes[1, 0].bar(ell_x[:12] + offset, pdf_rdn[:12],  width=w, color=colors[ss], alpha=0.85)
        axes[0, 1].bar(ell_x[:12] + offset, fill_cnt[:12], width=w, color=colors[ss], alpha=0.85)
        axes[1, 1].bar(ell_x[:12] + offset, fill_p[:12],   width=w, color=colors[ss], alpha=0.85)

    axes[0, 0].set_title("Counts / frequencies (RangeDn)");     axes[0, 0].set_ylabel("Count");       axes[0, 0].legend()
    axes[1, 0].set_title("ePDF of RangeDn");                     axes[1, 0].set_ylabel("P(R_dn = ℓ)")
    axes[0, 1].set_title("Counts of getting filled");            axes[0, 1].set_ylabel("Count")
    axes[1, 1].set_title("P(fill) vs. number of spreads");      axes[1, 1].set_ylabel("P(R_dn ≥ ℓ)")
    for ax in axes.flat:
        ax.set_xlabel("Number of spreads ℓ")

    plt.suptitle(f"{INSTRUMENT} — Conditional ePDF by σ-state  (τ={TAU} min)", y=1.01)
    plt.tight_layout()
    plt.show()


    # ## Fill Probability Curves (Buy & Sell Limits)

    # In[ ]:


    # For a Buy limit placed ℓ ticks below open:  P(fill) = P(R_dn >= ℓ)
    # For a Sell limit placed ℓ ticks above open: P(fill) = P(R_up >= ℓ)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ell_range = np.arange(1, 13)

    # Show one curve per (vol-state, sig-state) pair
    for m in range(M_VOL_STATES):
        for n in range(N_SIG_STATES):
            # Marginalise over direction state [OPTIONAL: fix sd as well]
            cnt_dn = epdf_Rdn.counts[m, n, :, :].sum(axis=0)
            cnt_up = epdf_Rup.counts[m, n, :, :].sum(axis=0)
            tot_dn = cnt_dn.sum()
            tot_up = cnt_up.sum()
            if tot_dn == 0 or tot_up == 0:
                continue
            fp_buy  = np.array([cnt_dn[l:].sum() / tot_dn for l in ell_range])
            fp_sell = np.array([cnt_up[l:].sum() / tot_up for l in ell_range])
            lbl = f"v={m},σ={n}"
            axes[0].plot(ell_range, fp_buy,  marker="o", ms=4, lw=1.2, label=lbl)
            axes[1].plot(ell_range, fp_sell, marker="s", ms=4, lw=1.2, label=lbl)

    for ax, title in zip(axes, ["P(Buy filled) = P(R_dn ≥ ℓ)",
                                  "P(Sell filled) = P(R_up ≥ ℓ)"]):
        ax.set_xlabel("ℓ (ticks below/above open)")
        ax.set_ylabel("Fill probability")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.suptitle(f"{INSTRUMENT} — Fill probability by regime  (τ={TAU} min)")
    plt.tight_layout()
    plt.show()


    # ## Backtest — fill-probability-based limit-order execution

    # In[ ]:


    # ── Backtest of fill-probability-based limit order execution ────────────────
    # Strategy:
    #   - At each τ-bar j, we have a signal direction (Buy or Sell) and a chosen
    #     limit-order offset ℓ* (in ticks) from the open of bar j.
    #   - ℓ* is the LARGEST offset in [1, MAX_OFFSET] whose predicted fill
    #     probability (from the state-conditioned ePDF, trained on bars < j)
    #     is still ≥ MIN_FILL_PROB.
    #   - A Buy limit fills if R_dn ≥ ℓ* (low reached ≥ ℓ ticks below open).
    #   - A Sell limit fills if R_up ≥ ℓ* (high reached ≥ ℓ ticks above open).
    #   - Unfilled bars contribute 0 PnL. Filled bars are exited at the bar close.
    # ────────────────────────────────────────────────────────────────────────────

    MIN_FILL_PROB = 0.60     # Tune: minimum predicted fill probability we will accept
    MAX_OFFSET    = 6        # Max ticks we'\'\'ll consider for the limit offset
    SIGNAL_MODE   = "mean_reversion"   # "mean_reversion" or "trend_following"

    # ── Direction signal: EWMA of returns (already causal via Algorithm 1) ──────
    ewma_ret_arr, _ = ewma_ewmv(df_tau["ret"].values, LAM)
    df_tau["ewma_ret"] = ewma_ret_arr

    if SIGNAL_MODE == "mean_reversion":
        # If EWMA return is positive (uptrend), bet on pullback → place SELL limit above
        # If EWMA return is negative (downtrend), bet on bounce → place BUY  limit below
        bt["signal"] = df_tau["ewma_ret"].reindex(bt.index).apply(
            lambda x: -1 if x >= 0 else 1)
    elif SIGNAL_MODE == "trend_following":
        # Bet that current trend continues → place limit in direction of recent EWMA
        bt["signal"] = df_tau["ewma_ret"].reindex(bt.index).apply(
            lambda x: 1 if x >= 0 else -1)
    else:
        raise ValueError(f"Unknown SIGNAL_MODE: {SIGNAL_MODE}")


    def best_offset(fp_cols, row, min_prob: float, max_off: int) -> int:
        """Return the LARGEST ℓ in [1, max_off] with P(fill) >= min_prob.
        Since P(fill) is monotonically decreasing in ℓ, this is well-defined.
        Returns 0 if even ℓ=1 fails the threshold (signal we should NOT trade)."""
        best = 0
        for ell in range(1, max_off + 1):
            if row[fp_cols[ell - 1]] >= min_prob:
                best = ell
            else:
                break  # P(fill) is monotone, so we can stop
        return best


    fp_rup_cols = [f"fp_rup_{l}" for l in range(1, MAX_OFFSET + 1)]
    fp_rdn_cols = [f"fp_rdn_{l}" for l in range(1, MAX_OFFSET + 1)]

    results = []
    for _, row in bt.iterrows():
        sig = row["signal"]

        if sig == 1:                       # Buy signal: place limit BELOW open
            ell = best_offset(fp_rdn_cols, row, MIN_FILL_PROB, MAX_OFFSET)
            if ell == 0:
                results.append({"timestamp": row.name, "signal": sig, "ell": 0,
                                "filled": False, "pnl_ticks": 0.0,
                                "p_pred": np.nan, "traded": False})
                continue
            lp   = row["open"] - ell * EPS
            hit  = row["Rdn_actual"] >= ell
            # Buy at lp if filled, exit at close → long PnL = close - entry
            pnl  = (row["close"] - lp) / EPS if hit else 0.0
            p_pred = row[fp_rdn_cols[ell - 1]]

        else:                              # Sell signal: place limit ABOVE open
            ell = best_offset(fp_rup_cols, row, MIN_FILL_PROB, MAX_OFFSET)
            if ell == 0:
                results.append({"timestamp": row.name, "signal": sig, "ell": 0,
                                "filled": False, "pnl_ticks": 0.0,
                                "p_pred": np.nan, "traded": False})
                continue
            lp   = row["open"] + ell * EPS
            hit  = row["Rup_actual"] >= ell
            # Sell at lp if filled, exit at close → short PnL = entry - close
            pnl  = (lp - row["close"]) / EPS if hit else 0.0
            p_pred = row[fp_rup_cols[ell - 1]]

        results.append({"timestamp": row.name, "signal": sig, "ell": ell,
                        "filled": hit, "pnl_ticks": pnl,
                        "p_pred": p_pred, "traded": True})

    res = pd.DataFrame(results).set_index("timestamp")

    # ── Summary stats ──────────────────────────────────────────────────────────
    n_total       = len(res)
    n_traded      = res["traded"].sum()
    n_filled      = res["filled"].sum()
    fill_rate     = n_filled / n_traded if n_traded else np.nan
    mean_p_pred   = res.loc[res["traded"], "p_pred"].mean()
    calibration   = fill_rate - mean_p_pred   # >0 means we underestimated fill prob

    # Per-bar Sharpe (includes zeros for un-traded bars)
    mean_pnl_bar  = res["pnl_ticks"].mean()
    sharpe_bar    = (mean_pnl_bar /
                     (res["pnl_ticks"].std() + 1e-12) *
                     np.sqrt(252 * 390 / TAU))

    # Per-filled-trade win rate and mean PnL
    filled        = res[res["filled"]]
    mean_pnl_fill = filled["pnl_ticks"].mean() if len(filled) else np.nan
    win_rate      = (filled["pnl_ticks"] > 0).mean() if len(filled) else np.nan

    # Mean chosen offset
    mean_ell      = res.loc[res["traded"], "ell"].mean()

    print(f"── Backtest ({SIGNAL_MODE}, MIN_FILL_PROB={MIN_FILL_PROB}, τ={TAU} min) ──")
    print(f"Total bars                : {n_total:,}")
    print(f"Bars we placed a limit    : {n_traded:,}  ({n_traded/n_total:.1%})")
    print(f"Bars filled               : {n_filled:,}  (fill rate of traded = {fill_rate:.2%})")
    print(f"Mean predicted fill prob  : {mean_p_pred:.3f}")
    print(f"Calibration (realised-pred): {calibration:+.3f}   (≈0 is well-calibrated)")
    print(f"Mean limit offset (ticks) : {mean_ell:.2f}")
    print(f"")
    print(f"Mean PnL per BAR (ticks)  : {mean_pnl_bar:+.4f}")
    print(f"Mean PnL per FILL (ticks) : {mean_pnl_fill:+.4f}")
    print(f"Win rate on filled trades : {win_rate:.2%}")
    print(f"Annualised Sharpe (per-bar): {sharpe_bar:+.2f}")
    total_pnl_ticks = res["pnl_ticks"].sum()
    print(f"Total PnL (ticks)         : {total_pnl_ticks:+.0f}")
    print(f"Total PnL ($, EPS={EPS})  : {total_pnl_ticks * EPS:+,.2f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(res.index, res["pnl_ticks"].cumsum(),
                 color="steelblue", lw=1.5)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[0].set_title(f"Cumulative PnL (ticks) — {INSTRUMENT}  τ={TAU} min  "
                      f"min_fill={MIN_FILL_PROB}  ({SIGNAL_MODE})")
    axes[0].set_ylabel("Cumulative ticks")
    axes[0].grid(alpha=0.3)

    # Distribution of per-fill PnL
    if len(filled):
        axes[1].hist(filled["pnl_ticks"], bins=60, color="seagreen",
                     edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="k", lw=0.5)
        axes[1].axvline(mean_pnl_fill, color="red", lw=1.5,
                        label=f"Mean = {mean_pnl_fill:+.2f}")
        axes[1].set_title("Distribution of PnL on filled trades (ticks)")
        axes[1].set_xlabel("PnL (ticks)")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ## AIAgent — trade-trace diagnostics

    # In[ ]:


    # ── AIAgent trade-trace diagnostics ────────────────────────────────────────
    # Column 5 of AIAgent file = net position (contracts held).
    # Position changes between consecutive rows = trades executed.

    AIAGENT_PATH = DATA_ROOT / INSTRUMENT / f"AIAgent_{INSTRUMENT}.csv"
    agent = pd.read_csv(AIAGENT_PATH, header=None,
                        names=["date_serial", "hour", "minute", "price", "net_pos"])
    agent["date"] = pd.to_datetime(agent["date_serial"] - 2, unit="D",
                                    origin="1900-01-01")
    agent["timestamp"] = (agent["date"]
                          + pd.to_timedelta(agent["hour"],   unit="h")
                          + pd.to_timedelta(agent["minute"], unit="m"))
    agent = agent.set_index("timestamp").sort_index()

    # Trade size on each step (positive = bought, negative = sold)
    agent["trade_size"] = agent["net_pos"].diff()
    agent["is_trade"]   = agent["trade_size"].fillna(0) != 0
    agent["side"]       = np.where(agent["trade_size"] > 0, "BUY",
                           np.where(agent["trade_size"] < 0, "SELL", "FLAT"))

    # Summary
    n_total      = len(agent)
    n_trades     = agent["is_trade"].sum()
    n_buys       = (agent["trade_size"] > 0).sum()
    n_sells      = (agent["trade_size"] < 0).sum()
    total_bought = agent.loc[agent["trade_size"] > 0, "trade_size"].sum()
    total_sold   = -agent.loc[agent["trade_size"] < 0, "trade_size"].sum()
    final_pos    = agent["net_pos"].iloc[-1]
    mean_pos     = agent["net_pos"].mean()

    print(f"── AIAgent trade trace ─────────────────────────────────────")
    print(f"Total 5-min snapshots     : {n_total:,}")
    print(f"Snapshots with a trade    : {n_trades:,}  ({n_trades/n_total:.1%})")
    print(f"  Buy events              : {n_buys:,}")
    print(f"  Sell events             : {n_sells:,}")
    print(f"Total contracts bought    : {int(total_bought):,}")
    print(f"Total contracts sold      : {int(total_sold):,}")
    print(f"Final net position        : {int(final_pos)}")
    print(f"Mean net position         : {mean_pos:+.2f}")
    print(f"Position range            : [{int(agent['net_pos'].min()):d}, "
          f"{int(agent['net_pos'].max()):+d}]")

    # Plot: net position over time
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(agent.index, agent["price"], color="steelblue", lw=0.6)
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"AIAgent {INSTRUMENT}  —  price (top) and net position (bottom)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(agent.index, agent["net_pos"], color="darkorange", lw=0.8)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].fill_between(agent.index, 0, agent["net_pos"],
                          where=agent["net_pos"] > 0, alpha=0.2, color="green",
                          label="long")
    axes[1].fill_between(agent.index, 0, agent["net_pos"],
                          where=agent["net_pos"] < 0, alpha=0.2, color="red",
                          label="short")
    axes[1].set_ylabel("Net position (contracts)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Realized PnL of the agent (mark-to-market at the final price)
    # PnL = sum over trades of (-trade_size * trade_price) + final_pos * final_price
    agent["cash_flow"] = -agent["trade_size"].fillna(0) * agent["price"]
    final_price = agent["price"].iloc[-1]
    realized_pnl = agent["cash_flow"].sum() + final_pos * final_price
    print(f"\nAgent realized PnL (mark-to-market at final price): ${realized_pnl:+,.2f}")
    print(f"  (assuming trades happen at the snapshot price)")


    # ## AIAgent — out-of-sample calibration

    # In[ ]:


    # ── AIAgent Out-of-Sample Calibration ──────────────────────────────────────
    # Steps:
    #   1. Load AIAgent_Nasdaq.csv (5-min price snapshots, 2020-01-02 → 2020-05-31).
    #   2. Resample 5-min snapshots into τ-min synthetic OHLCV bars
    #      (open = first price, close = last price, high = max, low = min;
    #       no real volume so we use 1 as a placeholder).
    #   3. Filter to RTH (09:30–15:45 ET, same as training).
    #   4. Compute ranges (R, R_up, R_dn) per bar.
    #   5. Walk forward bar-by-bar AGAINST FROZEN ePDFs (epdf_R/Rup/Rdn from training):
    #        for each bar we predict P(fill) at ℓ=1..6 using ONLY the previously-
    #        trained ePDFs (no updates from AIAgent data).
    #   6. Compare predicted P(fill) vs realised fill rate → calibration plot.
    #
    # Important: we use the SAME causal quantile thresholds from training to bin
    # the new data into states. To do this cleanly we re-run quantile_states_causal
    # on the *concatenation* training+AIAgent so the binner sees a continuous history.
    # But here we cheat slightly for simplicity: we use a non-causal qcut against
    # the training data\'s historical distribution, which is acceptable for
    # calibration evaluation (not for trading).

    # ── 1. Load AIAgent file ──────────────────────────────────────────────────
    AIAGENT_PATH = DATA_ROOT / INSTRUMENT / f"AIAgent_{INSTRUMENT}.csv"

    agent = pd.read_csv(AIAGENT_PATH, header=None,
                        names=["date_serial", "hour", "minute", "price", "col5"])
    # Excel date serial → datetime (subtract 2 for Excel's leap-year-bug + epoch offset)
    agent["date"] = pd.to_datetime(agent["date_serial"] - 2, unit="D",
                                    origin="1900-01-01")
    agent["timestamp"] = (agent["date"]
                          + pd.to_timedelta(agent["hour"],   unit="h")
                          + pd.to_timedelta(agent["minute"], unit="m"))
    agent = agent.set_index("timestamp").sort_index()

    print(f"AIAgent rows : {len(agent):,}")
    print(f"Date range   : {agent.index.min().date()} → {agent.index.max().date()}")
    print(f"Unique dates : {agent.index.normalize().nunique()}")

    # ── 2. Build τ-min synthetic OHLCV from 5-min snapshots ───────────────────
    # AIAgent is sampled every 5 min. For τ=15, group every 3 snapshots.
    # For τ=5, each snapshot becomes a degenerate bar (open=close=high=low).
    # When τ < 5 we cannot synthesise (would need finer data).
    assert TAU >= 5, "AIAgent data is 5-min snapshots; need TAU >= 5"

    agent_tau = agent["price"].resample(f"{TAU}min", label="left", closed="left").agg(
        open  = "first",
        high  = "max",
        low   = "min",
        close = "last",
    ).dropna(subset=["open"])
    agent_tau["volume"] = 1   # placeholder — AIAgent file has no volume

    print(f"\nAIAgent τ={TAU} min bars after resample: {len(agent_tau):,}")

    # ── 3. Filter to RTH (same window as training) ────────────────────────────
    agent_tau = agent_tau.between_time(RTH_START, RTH_END).copy()
    print(f"After RTH filter                       : {len(agent_tau):,}")

    # ── 4. Compute ranges ─────────────────────────────────────────────────────
    agent_tau = compute_ranges(agent_tau, EPS)

    # ── 5. Build state labels for AIAgent data ────────────────────────────────
    # Because the trained ePDFs are conditional on (state_vol, state_sig, state_dir),
    # we need to assign each AIAgent bar a state. To stay strictly OOS we should use
    # the *causal* binner over training+AIAgent concatenation. For simplicity we use
    # the training-data thresholds (frozen): the 33rd & 67th percentiles of the
    # training values for ewma_vol, ewmv_rng, and delta_x.

    # Run EWMA on AIAgent (causally — using only its own history is fine for OOS)
    ewma_vol_a, _         = ewma_ewmv(agent_tau["volume"].values, LAM)
    ewma_rng_a, ewmv_rng_a = ewma_ewmv(agent_tau["R"].values,      LAM)
    agent_tau["ewma_vol"] = ewma_vol_a
    agent_tau["ewmv_rng"] = ewmv_rng_a
    agent_tau["delta_x"]  = agent_tau["open"].diff()

    # Frozen thresholds from training data
    def frozen_thresholds(series: pd.Series, n_states: int) -> list:
        """Return n_states-1 quantile cut points from training data."""
        fracs = np.linspace(0, 1, n_states + 1)[1:-1]
        return [series.quantile(f) for f in fracs]

    vol_thr = frozen_thresholds(df_tau["ewma_vol"].dropna(), M_VOL_STATES)
    sig_thr = frozen_thresholds(df_tau["ewmv_rng"].dropna(), N_SIG_STATES)
    dir_thr = frozen_thresholds(df_tau["delta_x"].dropna(),  K_DIR_STATES)

    print(f"\nFrozen training-data thresholds:")
    print(f"  vol (ewma_vol)  : {[round(t, 1) for t in vol_thr]}")
    print(f"  sig (ewmv_rng)  : {[round(t, 2) for t in sig_thr]}")
    print(f"  dir (delta_x)   : {[round(t, 3) for t in dir_thr]}")

    def bin_with_thresholds(x, thresholds):
        """Bin scalar x into [0..n_states-1] using fixed thresholds."""
        for i, t in enumerate(thresholds):
            if x < t:
                return i
        return len(thresholds)

    # Lag by 1 (state at bar j uses info up through bar j-1)
    agent_tau["state_vol"] = agent_tau["ewma_vol"].shift(1).map(
        lambda x: bin_with_thresholds(x, vol_thr) if pd.notna(x) else 0).astype(int)
    agent_tau["state_sig"] = agent_tau["ewmv_rng"].shift(1).map(
        lambda x: bin_with_thresholds(x, sig_thr) if pd.notna(x) else 0).astype(int)
    agent_tau["state_dir"] = agent_tau["delta_x"].shift(1).map(
        lambda x: bin_with_thresholds(x, dir_thr) if pd.notna(x) else 0).astype(int)

    print(f"\nState distribution on AIAgent:")
    print(agent_tau[["state_vol", "state_sig", "state_dir"]]
          .value_counts().sort_index().head(27))

    # ── 6. Predict fill probabilities using FROZEN ePDFs ──────────────────────
    ELL_RANGE = list(range(1, MAX_OFFSET + 1))

    records = []
    for j in range(1, len(agent_tau)):
        sv = int(agent_tau["state_vol"].iloc[j])
        ss = int(agent_tau["state_sig"].iloc[j])
        sd = int(agent_tau["state_dir"].iloc[j])
        row = {
            "timestamp"  : agent_tau.index[j],
            "sv": sv, "ss": ss, "sd": sd,
            "Rup_actual" : agent_tau["R_up"].iloc[j],
            "Rdn_actual" : agent_tau["R_dn"].iloc[j],
        }
        for ell in ELL_RANGE:
            row[f"pred_rup_{ell}"] = epdf_Rup.fill_prob(sv, ss, sd, ell)
            row[f"pred_rdn_{ell}"] = epdf_Rdn.fill_prob(sv, ss, sd, ell)
            row[f"actual_rup_{ell}"] = int(agent_tau["R_up"].iloc[j] >= ell)
            row[f"actual_rdn_{ell}"] = int(agent_tau["R_dn"].iloc[j] >= ell)
        records.append(row)

    calib = pd.DataFrame(records).set_index("timestamp")
    print(f"\nAIAgent evaluation rows: {len(calib):,}")

    # ── 7. Calibration plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_bins = 10  # decile bins on predicted probability
    for side, color, label, ax in [
        ("rup", "darkorange", "Sell limit (R_up ≥ ℓ)", axes[0]),
        ("rdn", "steelblue",  "Buy limit  (R_dn ≥ ℓ)", axes[1]),
    ]:
        # Stack all (predicted, actual) pairs across ℓ = 1..MAX_OFFSET
        preds, acts = [], []
        for ell in ELL_RANGE:
            preds.append(calib[f"pred_{side}_{ell}"].values)
            acts.append(calib[f"actual_{side}_{ell}"].values)
        preds = np.concatenate(preds)
        acts  = np.concatenate(acts)

        # Bin by predicted probability decile, compute mean predicted vs mean actual
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx   = np.digitize(preds, bin_edges) - 1
        bin_idx   = np.clip(bin_idx, 0, n_bins - 1)
        mean_pred = [preds[bin_idx == b].mean() if (bin_idx == b).sum() else np.nan
                     for b in range(n_bins)]
        mean_act  = [acts [bin_idx == b].mean() if (bin_idx == b).sum() else np.nan
                     for b in range(n_bins)]
        counts    = [(bin_idx == b).sum() for b in range(n_bins)]

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect calibration")
        ax.scatter(mean_pred, mean_act, s=[c/30 for c in counts],
                   color=color, alpha=0.8, edgecolor="k", linewidth=0.5,
                   label=label)
        ax.set_xlabel("Mean predicted fill probability")
        ax.set_ylabel("Realised fill rate (AIAgent)")
        ax.set_title(label)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    plt.suptitle(f"AIAgent OOS Calibration — {INSTRUMENT}  τ={TAU} min", y=1.02)
    plt.tight_layout()
    plt.show()

    # ── 8. Brier score & overall fit ──────────────────────────────────────────
    print(f"\nOverall calibration metrics:")
    for side, label in [("rup", "Sell limit"), ("rdn", "Buy limit")]:
        preds, acts = [], []
        for ell in ELL_RANGE:
            preds.append(calib[f"pred_{side}_{ell}"].values)
            acts.append(calib[f"actual_{side}_{ell}"].values)
        preds = np.concatenate(preds)
        acts  = np.concatenate(acts)
        brier = ((preds - acts) ** 2).mean()
        mae   = np.abs(preds - acts).mean()
        print(f"  {label:<12} | Brier = {brier:.4f}  |  MAE = {mae:.4f}  "
              f"|  Mean pred = {preds.mean():.3f}  |  Mean actual = {acts.mean():.3f}")


    # ## Sweep over τ and MIN_FILL_PROB (optional)

    # In[ ]:


    # OPTIONAL: uncomment the sweep below and run.
    # Each (τ, threshold) pair runs the full pipeline — takes a few minutes.

    def run_pipeline(instrument: str, tau: int, min_fill: float,
                     half_life: int = HALF_LIFE) -> dict:
        eps = TICK_SIZE[instrument]
        lam = 2 ** (-1 / half_life)

        d1  = load_instrument(instrument, min_bar_frac=MIN_BAR_FRAC)
        dt  = resample_ohlcv(d1, tau)
        dt  = compute_ranges(dt, eps)

        ev, _ = ewma_ewmv(dt["volume"].values, lam)
        er, _ = ewma_ewmv(dt["R"].values,      lam)
        eret, _= ewma_ewmv(dt["ret"].values,   lam)
        dt["ewma_vol"] = ev; dt["ewma_rng"] = er; dt["ewma_ret"] = eret

        dt["state_vol"] = quantile_states_causal(pd.Series(ev,          index=dt.index).shift(1), M_VOL_STATES)
        dt["state_sig"] = quantile_states_causal(pd.Series(eret,         index=dt.index).shift(1), N_SIG_STATES)
        dt["state_dir"] = direction_state(pd.Series(dt["ret"]).shift(1), K_DIR_STATES)   # OPTIONAL

        _, epdf_up, epdf_dn, bt = build_rolling_epdfs(dt)
        bt["signal"] = pd.Series(eret, index=dt.index).reindex(bt.index).apply(
            lambda x: 1 if x >= 0 else -1)

        pnl_list = []
        for _, row in bt.iterrows():
            sig = row["signal"]
            if sig == 1:
                cols = [f"fp_rdn_{l}" for l in range(1, MAX_OFFSET + 1)]
                ell  = best_offset(cols, row, min_fill, MAX_OFFSET)
                hit  = row["Rdn_actual"] >= ell
                pnl  = (row["close"] - (row["open"] - ell * eps)) / eps if hit else 0.0
            else:
                cols = [f"fp_rup_{l}" for l in range(1, MAX_OFFSET + 1)]
                ell  = best_offset(cols, row, min_fill, MAX_OFFSET)
                hit  = row["Rup_actual"] >= ell
                pnl  = ((row["open"] + ell * eps) - row["close"]) / eps if hit else 0.0
            pnl_list.append(pnl)

        s = pd.Series(pnl_list)
        sharpe = s.mean() / (s.std() + 1e-12) * np.sqrt(252 * 390 / tau)
        return {"instrument": instrument, "tau": tau, "min_fill": min_fill,
                "fill_rate": (s != 0).mean(), "sharpe": sharpe,
                "total_pnl": s.sum()}


    # ── Uncomment to run the sweep ──
    # sweep_rows = []
    # for tau in [5, 10, 15, 30, 60]:
    #     for mfp in [0.50, 0.60, 0.70]:
    #         row = run_pipeline(INSTRUMENT, tau, mfp)
    #         sweep_rows.append(row)
    #         print(row)
    # sweep_df = pd.DataFrame(sweep_rows)
    # print(sweep_df.sort_values("sharpe", ascending=False))

    print("Sweep cell ready. Uncomment the loop above to run.")

