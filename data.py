"""
data.py — Data loading, contract chaining, resampling, and RTH filtering.

Functions
---------
load_contract     : read a single 1-min OHLCV CSV
sticky_roll       : choose the active contract for each calendar date
load_instrument   : chain all contracts for one instrument via sticky-roll
resample_ohlcv    : aggregate 1-min bars to τ-min OHLCV
apply_rth_filter  : restrict to regular trading hours and drop incomplete days
"""
import numpy as np
import pandas as pd
from pathlib import Path

import config


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
    equal volume flip back and forth, creating artificial price jumps.

    Roll dates introduce a small price discontinuity (cost-of-carry spread).
    We do not back-adjust because downstream analysis uses intra-bar range
    statistics (H-L, H-O, O-L), which are invariant to constant price shifts.
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


def load_instrument(instrument: str, min_bar_frac: float = None,
                    roll_days: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Load all contract CSVs for an instrument, chain via sticky-roll, and
    apply a day-completeness filter at the 1-min level."""
    if min_bar_frac is None:
        min_bar_frac = config.MIN_BAR_FRAC
    folder = config.DATA_ROOT / instrument
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


def resample_ohlcv(df: pd.DataFrame, tau: int, verbose: bool = True) -> pd.DataFrame:
    """Aggregate 1-min bars into τ-min OHLCV bars."""
    agg = df.resample(f"{tau}min", label="left", closed="left").agg(
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
                     min_bar_frac: float = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Restrict τ-bars to the market's declared RTH window, then re-apply the
    day-completeness filter at the τ-bar level. 24-h markets are passed through."""
    if min_bar_frac is None:
        min_bar_frac = config.MIN_BAR_FRAC
    rth_start_str, rth_end_str, tz_label = config.MARKETS[instrument]["rth"]
    rth_start_min = int(rth_start_str[:2]) * 60 + int(rth_start_str[3:])
    rth_end_min   = int(rth_end_str[:2])   * 60 + int(rth_end_str[3:])
    skip_rth      = (rth_start_min == 0 and rth_end_min >= 24 * 60 - 1)

    before = len(df_tau)
    if not skip_rth:
        rth_filter_end_min = rth_end_min - tau
        rth_filter_end_str = f"{rth_filter_end_min // 60:02d}:{rth_filter_end_min % 60:02d}"
        df_tau = df_tau.between_time(rth_start_str, rth_filter_end_str)

    expected_bars_per_day = (
        (rth_end_min - rth_start_min) // tau if not skip_rth else (24 * 60) // tau
    )
    df_tau = df_tau.copy()
    df_tau["_date"] = df_tau.index.normalize()
    bars_per_day = df_tau.groupby("_date").size()
    good_days    = bars_per_day[bars_per_day >= min_bar_frac * expected_bars_per_day].index
    df_tau       = df_tau[df_tau["_date"].isin(good_days)].drop(columns="_date")

    if verbose:
        if skip_rth:
            print("  RTH         : 24h (no filter)")
        else:
            print(f"  RTH         : [{rth_start_str}, {rth_end_str}] {tz_label}")
        print(f"  Before RTH  : {before:,} τ-bars")
        print(f"  After RTH   : {len(df_tau):,} τ-bars across {len(good_days)} good days")
    return df_tau
