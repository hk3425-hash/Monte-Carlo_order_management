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

## Configure data path

Edit `DATA_ROOT` in `mc_helpers.py` to point to your local copy of the futures data folder.

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Supported markets

Nasdaq, Gold, German Bunds, EuroStoxx, GBP, HeatingOil, JPY — any market with chained futures contracts and an optional AIAgent trade file.

## Authors

Cem Okutan — Columbia University, Spring 2026
