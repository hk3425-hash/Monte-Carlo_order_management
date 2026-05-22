# IEOR4703 Monte Carlo — Volatility-Volume Order Management

Term Project 2 for Columbia IEOR4703. A Streamlit dashboard exploring state-conditioned empirical PDFs of intraday price ranges, applied to limit-order execution.

## What it does

**Part 1: Regime Analysis** — Builds conditional ePDFs `P(R_dn | volume regime, volatility regime, prior direction)` from historical futures OHLCV data using strictly causal EWMA recursion. Demonstrates via KL divergence that regime conditioning carries meaningful information.

**Part 2: Execution Comparison** — Given a directional signal (synthetic mean-reversion, trend-following, or external AIAgent trades), compares three execution policies on the same signal bars:
- Method A: Market orders at bar open (baseline)
- Method B: Naive fixed-offset limit orders
- Method C: State-conditioned ePDF-guided adaptive limit orders

## Setup

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data setup

The application expects futures data organized as:

```
data/
  Nasdaq/
    NQH20.csv
    NQM20.csv
    ...
    AIAgent_Nasdaq.csv
  Gold/
    GCG24.csv
    ...
    AIAgent_Gold.csv
  ...
```

Each instrument folder holds per-contract 1-minute OHLCV CSVs (no header row, columns: timestamp, open, high, low, close, volume) and an optional AIAgent CSV.

The application looks for the data folder in this order:

1. `$MC_DATA_ROOT` environment variable (if set and path exists)
2. `data/` folder inside the project directory (`monte-carlo-app/data/`)
3. A hardcoded fallback path (`/Users/cemokutan/Documents/Monte_Carlo/project/data`)

**Quickest setup:** drop the course-provided data into `monte-carlo-app/data/`.

**Alternative:** set the environment variable before launching:
```bash
export MC_DATA_ROOT=/path/to/your/data
streamlit run app.py
```

The resolved data path is displayed at the bottom of the sidebar for verification.

### AIAgent filenames

Each instrument's AIAgent file uses a short name regardless of the full instrument key:

| Instrument | AIAgent file |
|---|---|
| Nasdaq | `AIAgent_Nasdaq.csv` |
| Gold | `AIAgent_Gold.csv` |
| German Bunds - German Government Bonds | `AIAgent_Bunds.csv` |
| EuroStoxx | `AIAgent_EuroStoxx.csv` |
| GBP - British Pound | `AIAgent_GBPUSD.csv` |
| HeatingOil | `AIAgent_HeatingOil.csv` |
| JPY - Japanese Yen | `AIAgent_JPY.csv` |

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Supported markets

Nasdaq, Gold, German Bunds, EuroStoxx, GBP, HeatingOil, JPY — any market with chained futures contracts and an optional AIAgent trade file.

## Authors

Cem Okutan — Columbia University, Spring 2026
