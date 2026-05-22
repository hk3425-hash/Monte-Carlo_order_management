"""
config.py — Global constants, market metadata, and mutable pipeline parameters.

All functions in data.py / features.py / epdf.py read their parameters from
this module at call time. To change a parameter, set it here:

    import config
    config.LAM          = 2 ** (-1 / 20)
    config.M_VOL_STATES = 3
"""
import os
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Default pipeline parameters ───────────────────────────────────────────────
INSTRUMENT   = "Nasdaq"
TAU          = 15
HALF_LIFE    = 20
M_VOL_STATES = 3
N_SIG_STATES = 3
K_DIR_STATES = 3
J_START      = 100
MAX_SPREADS  = 150
MAX_T_PLOT   = 25
MIN_BAR_FRAC = 0.90


def _resolve_data_root() -> Path:
    """Locate the data directory in priority order:
    1. MC_DATA_ROOT environment variable
    2. data/ folder inside this project directory
    3. Original hardcoded path (fallback for original author's setup)
    """
    env_path = os.environ.get("MC_DATA_ROOT")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    here = Path(__file__).parent
    local = here / "data"
    if local.exists() and any(local.iterdir()):
        return local
    return Path("/Users/cemokutan/Documents/Monte_Carlo/project/data")


DATA_ROOT = _resolve_data_root()
FIG_DIR   = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Per-market metadata ───────────────────────────────────────────────────────
#   tick : exchange tick size
#   rth  : (start_hhmm, end_hhmm, tz_label) in the market's native timezone.
#          (00:00, 23:59) marks a 24-h market — RTH filter is skipped.
MARKETS = {
    "Nasdaq"                                 : {"tick": 0.25,   "rth": ("09:30", "16:00", "ET (US equity)")},
    "Gold"                                   : {"tick": 0.10,   "rth": ("08:20", "13:30", "ET (COMEX pit hours)")},
    "German Bunds - German Government Bonds" : {"tick": 0.01,   "rth": ("08:00", "17:00", "CET (Eurex)")},
    "EuroStoxx"                              : {"tick": 0.50,   "rth": ("09:00", "17:30", "CET (Eurex)")},
    "GBP - British Pound"                    : {"tick": 0.0100, "rth": ("00:00", "23:59", "FX 24h (no filter)")},
    "HeatingOil"                             : {"tick": 0.0100, "rth": ("09:00", "14:30", "ET (NYMEX pit)")},
    "JPY - Japanese Yen"                     : {"tick": 0.0050, "rth": ("00:00", "23:59", "FX 24h (no filter)")},
}

# ── AIAgent filename mapping ──────────────────────────────────────────────────
# Maps each MARKETS key to the actual AIAgent CSV filename in that instrument's
# data subfolder. Long instrument names don't map to valid filenames directly.
AIAGENT_FILENAME = {
    "Nasdaq"                                 : "AIAgent_Nasdaq.csv",
    "Gold"                                   : "AIAgent_Gold.csv",
    "German Bunds - German Government Bonds" : "AIAgent_Bunds.csv",
    "EuroStoxx"                              : "AIAgent_EuroStoxx.csv",
    "GBP - British Pound"                    : "AIAgent_GBPUSD.csv",
    "HeatingOil"                             : "AIAgent_HeatingOil.csv",
    "JPY - Japanese Yen"                     : "AIAgent_JPY.csv",
}

# ── Derived globals — updated by app.py or analysis scripts at runtime ────────
EPS = MARKETS[INSTRUMENT]["tick"]
LAM = 2 ** (-1 / HALF_LIFE)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"      : 110,
    "savefig.dpi"     : 200,
    "savefig.bbox"    : "tight",
    "savefig.format"  : "pdf",
    "font.family"     : "serif",
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 10,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
})


def savefig(fig, name: str) -> None:
    """Save figure as PDF (for LaTeX) and PNG (for preview)."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    print(f"  saved → figures/{name}.pdf")
