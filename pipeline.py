"""
pipeline.py — One-call market preparation pipeline.

Functions
---------
prepare_market : load → chain → resample → RTH filter → ranges → states
"""
import config
from data import load_instrument, resample_ohlcv, apply_rth_filter
from features import compute_ranges, add_states


def prepare_market(instrument: str, tau: int = None, verbose: bool = True) -> tuple:
    """Full pipeline: load → chain → resample → RTH filter → ranges → states.
    Returns (df_tau, eps)."""
    if tau is None:
        tau = config.TAU
    if verbose:
        print(f"\n── prepare_market: {instrument}, τ={tau} ──")
    eps   = config.MARKETS[instrument]["tick"]
    df_1m = load_instrument(instrument, verbose=verbose)
    df_t  = resample_ohlcv(df_1m, tau, verbose=verbose)
    df_t  = apply_rth_filter(df_t, instrument, tau, verbose=verbose)
    df_t  = compute_ranges(df_t, eps, verbose=verbose)
    df_t  = add_states(df_t)
    return df_t, eps
