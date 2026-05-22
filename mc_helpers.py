"""
mc_helpers.py — Backward-compatibility shim.

Re-exports everything from the split modules so that existing code that does
    from mc_helpers import prepare_market, ConditionalEPDF, ...
or sets
    mc_helpers.LAM = value
continues to work without modification.

New code should import directly from the source modules:
    config, data, features, epdf, pipeline
"""
from config import *       # INSTRUMENT, TAU, LAM, EPS, MARKETS, savefig, ...
from data import *         # load_contract, sticky_roll, load_instrument, resample_ohlcv, apply_rth_filter
from features import *     # compute_ranges, ewma_ewmv, quantile_states_causal, add_states
from epdf import *         # epdf_from_array, fill_prob_from_pmf, kl_div, ConditionalEPDF, build_rolling_epdfs, full_cond_epdf
from pipeline import *     # prepare_market
