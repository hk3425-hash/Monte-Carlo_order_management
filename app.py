"""
Columbia IEOR4703 Monte Carlo Simulation — Volatility-Volume Order Management
Streamlit dashboard with two tabs:
  Part 1 — Regime Analysis: joint state heatmap, unconditional + conditional ePDFs,
            KL divergence, and fill-probability tables.
  Part 2 — Trading Application: fill-probability curves and a walk-forward limit-order
            backtest that uses state-conditioned ePDFs to choose the optimal offset.

Data source options (sidebar):
  • Continuous  — sticky-roll chained contracts via load_instrument().
  • Single contract — one CSV file, no chaining.
  • AIAgent file — synthetic OHLCV from 5-min price snapshots; shows price/position
                   chart, trade-trace summary, and OOS fill-prob calibration against
                   frozen ePDFs built in Continuous mode.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import streamlit as st

import mc_helpers
from mc_helpers import (
    MARKETS,
    load_instrument, load_contract, resample_ohlcv, apply_rth_filter,
    compute_ranges, add_states, build_rolling_epdfs,
    epdf_from_array, fill_prob_from_pmf, kl_div, full_cond_epdf,
    ewma_ewmv,
)

MAX_T_PLOT  = 25     # x-axis cap on ePDF / fill-prob plots
MAX_SPREADS = 150    # histogram buckets in ConditionalEPDF
ELL_CAL     = list(range(1, 7))  # offsets used in OOS calibration


# ── Small utilities ────────────────────────────────────────────────────────────

def _fig_selector(label: str, options: list, key: str) -> str:
    """Horizontal figure picker: st.pills if available, else st.radio."""
    try:
        sel = st.pills(label, options, default=options[0], key=key)
        return sel if sel is not None else options[0]
    except AttributeError:
        return st.radio(label, options, horizontal=True, key=key)


def _frozen_thresholds(series: pd.Series, n_states: int) -> list:
    """Return n_states-1 quantile cut points (used for frozen state assignment)."""
    fracs = np.linspace(0, 1, n_states + 1)[1:-1]
    return [float(series.dropna().quantile(f)) for f in fracs]


def _bin(x, thresholds: list) -> int:
    for i, t in enumerate(thresholds):
        if x < t:
            return i
    return len(thresholds)


# ── Execution helpers (used by Part 2) ────────────────────────────────────────

def _exec_market(sig: int, row, eps: float) -> dict:
    pnl = (row["close"] - row["open"]) / eps if sig == 1 else (row["open"] - row["close"]) / eps
    return {"pnl": float(pnl), "filled": True, "ell": 0}


def _exec_naive(sig: int, row, eps: float, ell_fixed: int) -> dict:
    if sig == 1:
        filled = int(row["Rdn_actual"]) >= ell_fixed
        pnl = (row["close"] - (row["open"] - ell_fixed * eps)) / eps if filled else -0.5
    else:
        filled = int(row["Rup_actual"]) >= ell_fixed
        pnl = ((row["open"] + ell_fixed * eps) - row["close"]) / eps if filled else -0.5
    return {"pnl": float(pnl), "filled": bool(filled), "ell": ell_fixed}


def _exec_epdf(sig: int, row, eps: float, epdf_Rup, epdf_Rdn,
               max_offset: int, min_fill_prob: float) -> dict:
    sv, ss, sd = int(row["sv"]), int(row["ss"]), int(row["sd"])
    if sig == 1:
        ell = 0
        for l in range(1, max_offset + 1):
            c = f"fp_rdn_{l}"
            prob = row[c] if c in row.index else epdf_Rdn.fill_prob(sv, ss, sd, l)
            if prob >= min_fill_prob:
                ell = l
            else:
                break
        if ell == 0:
            return {"pnl": 0.0, "filled": False, "ell": 0, "traded": False}
        filled = int(row["Rdn_actual"]) >= ell
        pnl = (row["close"] - (row["open"] - ell * eps)) / eps if filled else -0.5
    else:
        ell = 0
        for l in range(1, max_offset + 1):
            c = f"fp_rup_{l}"
            prob = row[c] if c in row.index else epdf_Rup.fill_prob(sv, ss, sd, l)
            if prob >= min_fill_prob:
                ell = l
            else:
                break
        if ell == 0:
            return {"pnl": 0.0, "filled": False, "ell": 0, "traded": False}
        filled = int(row["Rup_actual"]) >= ell
        pnl = ((row["open"] + ell * eps) - row["close"]) / eps if filled else -0.5
    return {"pnl": float(pnl), "filled": bool(filled), "ell": ell, "traded": True}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IEOR4703 — MC Simulation", layout="wide")
st.title("IEOR4703 Monte Carlo — Volatility-Volume Order Management")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Parameters")

    data_source = st.radio(
        "Data source",
        ["Continuous (sticky-roll chained)", "Single contract", "AIAgent file"],
        help=(
            "Continuous: chains all contracts via sticky-roll.  "
            "Single contract: loads one CSV directly.  "
            "AIAgent file: loads the provided 5-min price snapshot file."
        ),
    )

    instrument = st.selectbox("Instrument", list(MARKETS.keys()))

    # Contract picker — only shown for Single contract mode
    contract_name: str | None = None
    if data_source == "Single contract":
        data_dir = mc_helpers.DATA_ROOT / instrument
        csv_stems = sorted(
            f.stem for f in data_dir.glob("*.csv")
            if not f.stem.startswith("AIAgent")
        )
        if csv_stems:
            contract_name = st.selectbox("Contract", csv_stems)
        else:
            st.warning(f"No CSV files found in {data_dir}")

    tau       = st.slider("τ holding period (min)", 5, 60, 15)
    half_life = st.slider("EWMA half-life (bars)", 5, 60, 20)
    st.markdown("**State bins**")
    m_vol  = int(st.number_input("M_VOL_STATES (volume)",     min_value=2, max_value=5, value=3))
    n_sig  = int(st.number_input("N_SIG_STATES (volatility)", min_value=2, max_value=5, value=3))
    k_dir  = int(st.number_input("K_DIR_STATES (direction)",  min_value=1, max_value=5, value=3))
    j_start = int(st.number_input("J_START (burn-in bars)", min_value=10, max_value=500, value=100))


# ── Sync module globals ────────────────────────────────────────────────────────
# Helper functions read LAM, M/N/K_*_STATES, MAX_SPREADS directly from the
# mc_helpers namespace; we keep them in sync with the sidebar values.
lam = 2.0 ** (-1.0 / half_life)
mc_helpers.LAM          = lam
mc_helpers.M_VOL_STATES = m_vol
mc_helpers.N_SIG_STATES = n_sig
mc_helpers.K_DIR_STATES = k_dir
mc_helpers.J_START      = j_start
mc_helpers.MAX_SPREADS  = MAX_SPREADS
eps = MARKETS[instrument]["tick"]


# ── Cached pipeline functions ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_continuous(instrument, tau, half_life, m_vol, n_sig, k_dir, j_start):
    """Sticky-roll chained pipeline. Cache key covers all inputs."""
    _lam = 2.0 ** (-1.0 / half_life)
    mc_helpers.LAM = _lam; mc_helpers.M_VOL_STATES = m_vol
    mc_helpers.N_SIG_STATES = n_sig; mc_helpers.K_DIR_STATES = k_dir
    mc_helpers.J_START = j_start; mc_helpers.MAX_SPREADS = MAX_SPREADS

    _eps = MARKETS[instrument]["tick"]
    df_1m = load_instrument(instrument, verbose=False)
    df_t  = resample_ohlcv(df_1m, tau, verbose=False)
    df_t  = apply_rth_filter(df_t, instrument, tau, verbose=False)
    df_t  = compute_ranges(df_t, _eps, verbose=False)
    df_t  = add_states(df_t, m_vol=m_vol, n_sig=n_sig, k_dir=k_dir)
    epdf_R, epdf_Rup, epdf_Rdn, bt = build_rolling_epdfs(df_t, j_start=j_start)
    return df_t, epdf_R, epdf_Rup, epdf_Rdn, bt


@st.cache_data(show_spinner=False)
def _run_single(instrument, contract_name, tau, half_life, m_vol, n_sig, k_dir, j_start):
    """Single-contract pipeline (no chaining). Cache key includes contract_name."""
    _lam = 2.0 ** (-1.0 / half_life)
    mc_helpers.LAM = _lam; mc_helpers.M_VOL_STATES = m_vol
    mc_helpers.N_SIG_STATES = n_sig; mc_helpers.K_DIR_STATES = k_dir
    mc_helpers.J_START = j_start; mc_helpers.MAX_SPREADS = MAX_SPREADS

    _eps = MARKETS[instrument]["tick"]
    path = mc_helpers.DATA_ROOT / instrument / f"{contract_name}.csv"
    df_1m = load_contract(path)
    df_t  = resample_ohlcv(df_1m, tau, verbose=False)
    df_t  = apply_rth_filter(df_t, instrument, tau, verbose=False)
    df_t  = compute_ranges(df_t, _eps, verbose=False)
    df_t  = add_states(df_t, m_vol=m_vol, n_sig=n_sig, k_dir=k_dir)
    epdf_R, epdf_Rup, epdf_Rdn, bt = build_rolling_epdfs(df_t, j_start=j_start)
    return df_t, epdf_R, epdf_Rup, epdf_Rdn, bt


@st.cache_data(show_spinner=False)
def _load_aiagent(instrument, tau):
    """Load AIAgent CSV → (raw agent DataFrame with net_pos, τ-min OHLCV with ranges)."""
    path = mc_helpers.DATA_ROOT / instrument / f"AIAgent_{instrument}.csv"
    agent = pd.read_csv(path, header=None,
                        names=["date_serial", "hour", "minute", "price", "net_pos"])
    agent["date"] = pd.to_datetime(agent["date_serial"] - 2, unit="D",
                                    origin="1900-01-01")
    agent["timestamp"] = (agent["date"]
                          + pd.to_timedelta(agent["hour"],   unit="h")
                          + pd.to_timedelta(agent["minute"], unit="m"))
    agent = agent.set_index("timestamp").sort_index()

    _eps = MARKETS[instrument]["tick"]
    agent_tau = agent["price"].resample(f"{tau}min", label="left", closed="left").agg(
        open="first", high="max", low="min", close="last"
    ).dropna(subset=["open"])
    agent_tau["volume"] = 1   # placeholder (no real volume in AIAgent file)
    agent_tau = apply_rth_filter(agent_tau, instrument, tau,
                                  min_bar_frac=0.5, verbose=False)
    agent_tau = compute_ranges(agent_tau, _eps, verbose=False)
    return agent, agent_tau


# ── Route to the right pipeline ────────────────────────────────────────────────

if data_source == "Continuous (sticky-roll chained)":
    with st.spinner("Loading data and building ePDFs…"):
        df_tau, epdf_R, epdf_Rup, epdf_Rdn, bt = _run_continuous(
            instrument, tau, half_life, m_vol, n_sig, k_dir, j_start
        )
    # Persist frozen ePDFs so AIAgent mode can use them for OOS calibration.
    st.session_state["frozen"] = dict(
        df_tau=df_tau, epdf_Rup=epdf_Rup, epdf_Rdn=epdf_Rdn,
        instrument=instrument, m_vol=m_vol, n_sig=n_sig, k_dir=k_dir,
    )

elif data_source == "Single contract":
    if contract_name is None:
        st.error("No contract selected — pick an instrument with CSV files.")
        st.stop()
    with st.spinner(f"Loading {contract_name} and building ePDFs…"):
        df_tau, epdf_R, epdf_Rup, epdf_Rdn, bt = _run_single(
            instrument, contract_name, tau, half_life, m_vol, n_sig, k_dir, j_start
        )

else:  # AIAgent file — data loaded inside the tab, not here
    df_tau = epdf_R = epdf_Rup = epdf_Rdn = bt = None

df_js = df_tau.iloc[j_start:].copy() if df_tau is not None else None


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Part 1: Regime Analysis", "Part 2: Trading Application"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — REGIME ANALYSIS  (or AIAgent views)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── AIAgent branch ─────────────────────────────────────────────────────────
    if data_source == "AIAgent file":

        aiagent_views = ["Price & Position", "Trade Summary", "OOS Calibration"]
        sel1 = _fig_selector("View", aiagent_views, key="tab1_aiagent")

        try:
            with st.spinner("Loading AIAgent data…"):
                agent_raw, agent_tau = _load_aiagent(instrument, tau)
        except FileNotFoundError:
            st.error(f"AIAgent_{instrument}.csv not found in the data directory.")
            st.stop()

        # ── Price & Position ──────────────────────────────────────────────────
        if sel1 == "Price & Position":
            st.subheader(f"AIAgent — Price and Net Position  ({instrument})")
            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            axes[0].plot(agent_raw.index, agent_raw["price"],
                         color="steelblue", lw=0.7)
            axes[0].set_ylabel("Price")
            axes[0].set_title("Price")
            axes[0].grid(alpha=0.3)

            axes[1].plot(agent_raw.index, agent_raw["net_pos"],
                         color="darkorange", lw=0.8)
            axes[1].axhline(0, color="k", lw=0.5)
            axes[1].fill_between(agent_raw.index, 0, agent_raw["net_pos"],
                                  where=agent_raw["net_pos"] > 0,
                                  alpha=0.2, color="green", label="long")
            axes[1].fill_between(agent_raw.index, 0, agent_raw["net_pos"],
                                  where=agent_raw["net_pos"] < 0,
                                  alpha=0.2, color="red", label="short")
            axes[1].set_ylabel("Net position (contracts)")
            axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "Top: the AIAgent's price series (5-min snapshots). "
                "Bottom: cumulative net position — green fills mark long exposure, "
                "red marks short. Position changes between consecutive rows indicate trades."
            )

        # ── Trade Summary ─────────────────────────────────────────────────────
        elif sel1 == "Trade Summary":
            st.subheader(f"AIAgent — Trade Trace Summary  ({instrument})")
            agent_raw["trade_size"] = agent_raw["net_pos"].diff()
            agent_raw["is_trade"]   = agent_raw["trade_size"].fillna(0) != 0
            n_snaps  = len(agent_raw)
            n_trades = int(agent_raw["is_trade"].sum())
            n_buys   = int((agent_raw["trade_size"] > 0).sum())
            n_sells  = int((agent_raw["trade_size"] < 0).sum())
            tot_bought = agent_raw.loc[agent_raw["trade_size"] > 0, "trade_size"].sum()
            tot_sold   = -agent_raw.loc[agent_raw["trade_size"] < 0, "trade_size"].sum()
            final_pos  = agent_raw["net_pos"].iloc[-1]
            mean_pos   = agent_raw["net_pos"].mean()
            final_price = agent_raw["price"].iloc[-1]
            cash_flows  = (-agent_raw["trade_size"].fillna(0) * agent_raw["price"]).sum()
            mtm_pnl     = cash_flows + final_pos * final_price

            c1, c2, c3 = st.columns(3)
            c1.metric("Total 5-min snapshots", f"{n_snaps:,}")
            c1.metric("Snapshots with a trade", f"{n_trades:,}  ({n_trades/n_snaps:.1%})")
            c2.metric("Buy events", f"{n_buys:,}")
            c2.metric("Sell events", f"{n_sells:,}")
            c3.metric("Contracts bought", f"{int(tot_bought):,}")
            c3.metric("Contracts sold", f"{int(tot_sold):,}")
            st.divider()
            c4, c5, c6 = st.columns(3)
            c4.metric("Final net position", f"{int(final_pos)}")
            c5.metric("Mean net position", f"{mean_pos:+.2f}")
            c6.metric("Mark-to-market PnL", f"${mtm_pnl:+,.2f}")
            st.caption(
                "Mark-to-market PnL assumes all trades filled at the 5-min snapshot "
                "price and the open position is closed at the final snapshot price."
            )

            st.subheader("Trade size distribution")
            trades = agent_raw.loc[agent_raw["is_trade"], "trade_size"]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(trades, bins=40, color="steelblue", edgecolor="white")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Trade size (contracts, + = buy)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of individual trade sizes")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        # ── OOS Calibration ───────────────────────────────────────────────────
        elif sel1 == "OOS Calibration":
            frozen = st.session_state.get("frozen")
            if frozen is None or frozen.get("instrument") != instrument:
                st.info(
                    "ℹ️ OOS calibration requires frozen ePDFs from the **Continuous** pipeline.  "
                    "Switch the Data source to **Continuous (sticky-roll chained)**, let it load, "
                    "then come back to AIAgent mode."
                )
            else:
                st.subheader(f"AIAgent — OOS Fill-Prob Calibration  ({instrument})")

                f_df_tau  = frozen["df_tau"]
                f_epdf_Rup = frozen["epdf_Rup"]
                f_epdf_Rdn = frozen["epdf_Rdn"]
                f_m = frozen["m_vol"]; f_n = frozen["n_sig"]; f_k = frozen["k_dir"]

                # Compute EWMA features on AIAgent τ-bars (causal, own history)
                _lam_a = lam
                ev_a, _   = ewma_ewmv(agent_tau["volume"].values, _lam_a)
                er_a, ev2_a = ewma_ewmv(agent_tau["R"].values,   _lam_a)
                agent_tau = agent_tau.copy()
                agent_tau["ewma_vol"] = ev_a
                agent_tau["ewmv_rng"] = ev2_a
                agent_tau["delta_x"]  = agent_tau["open"].diff()

                # Frozen thresholds from training df_tau
                vol_thr = _frozen_thresholds(f_df_tau["ewma_vol"], f_m)
                sig_thr = _frozen_thresholds(f_df_tau["ewmv_rng"], f_n)
                dir_thr = _frozen_thresholds(f_df_tau["delta_x"],  f_k)

                def _assign(col, thr):
                    return (agent_tau[col].shift(1)
                            .map(lambda x: _bin(x, thr) if pd.notna(x) else 0)
                            .astype(int))

                agent_tau["state_vol"] = _assign("ewma_vol", vol_thr)
                agent_tau["state_sig"] = _assign("ewmv_rng", sig_thr)
                agent_tau["state_dir"] = _assign("delta_x",  dir_thr)

                # Collect (predicted, actual) pairs for each ell
                preds_dn, acts_dn = [], []
                preds_up, acts_up = [], []
                for j in range(1, len(agent_tau)):
                    sv = int(agent_tau["state_vol"].iloc[j])
                    ss = int(agent_tau["state_sig"].iloc[j])
                    sd = int(agent_tau["state_dir"].iloc[j])
                    r_dn = int(agent_tau["R_dn"].iloc[j])
                    r_up = int(agent_tau["R_up"].iloc[j])
                    for ell in ELL_CAL:
                        preds_dn.append(f_epdf_Rdn.fill_prob(sv, ss, sd, ell))
                        acts_dn.append(1 if r_dn >= ell else 0)
                        preds_up.append(f_epdf_Rup.fill_prob(sv, ss, sd, ell))
                        acts_up.append(1 if r_up >= ell else 0)

                preds_dn = np.array(preds_dn); acts_dn = np.array(acts_dn)
                preds_up = np.array(preds_up); acts_up = np.array(acts_up)

                # Calibration plot: bin by predicted prob, compare to realised rate
                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)

                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                for ax, preds, acts, title, color in [
                    (axes[0], preds_dn, acts_dn, "Buy limit  P(R_dn >= ell)", "steelblue"),
                    (axes[1], preds_up, acts_up, "Sell limit  P(R_up >= ell)", "darkorange"),
                ]:
                    bidx = np.clip(np.digitize(preds, bin_edges) - 1, 0, n_bins - 1)
                    mp = [preds[bidx == b].mean() if (bidx == b).any() else np.nan
                          for b in range(n_bins)]
                    ma = [acts[bidx == b].mean()  if (bidx == b).any() else np.nan
                          for b in range(n_bins)]
                    cnts = [(bidx == b).sum() for b in range(n_bins)]
                    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
                            label="Perfect calibration")
                    ax.scatter(mp, ma, s=[max(c / 20, 20) for c in cnts],
                               color=color, alpha=0.8, edgecolor="k", lw=0.5,
                               label="Decile bins (size ∝ count)")
                    ax.set_xlabel("Mean predicted fill probability")
                    ax.set_ylabel("Realised fill rate")
                    ax.set_title(title)
                    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
                    ax.grid(alpha=0.3); ax.legend(fontsize=9)

                    brier = float(np.mean((preds - acts) ** 2))
                    mae   = float(np.mean(np.abs(preds - acts)))
                    ax.set_xlabel(
                        f"Mean predicted  (Brier={brier:.4f}, MAE={mae:.4f})"
                    )

                fig.suptitle(
                    f"OOS Calibration — {instrument}  τ={tau} min  "
                    f"(frozen ePDFs vs AIAgent realised fills)",
                    fontweight="bold",
                )
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)
                st.caption(
                    "Each dot is a decile bin of predicted fill probabilities. "
                    "Dots lying on the dashed diagonal mean the model is perfectly "
                    "calibrated. Dots above the line mean actual fills exceeded "
                    "predictions (conservative model); below means over-predicted. "
                    "Dot size is proportional to the number of observations in that bin."
                )

    # ── Normal pipeline branch (Continuous / Single contract) ──────────────────
    else:
        FIG_OPTIONS_1 = [
            "Dataset Summary",
            "Joint State Heatmap",
            "Unconditional ePDFs",
            "Conditional ePDF Grid",
            "KL Divergence Grid",
            "Fill Probability Table",
        ]
        sel1 = _fig_selector("View", FIG_OPTIONS_1, key="tab1_fig")

        # ── Dataset Summary ───────────────────────────────────────────────────
        if sel1 == "Dataset Summary":
            st.subheader("Dataset Summary")
            n_good_days = df_tau.index.normalize().nunique()
            d_min = df_tau.index.min().date()
            d_max = df_tau.index.max().date()
            st.caption(f"Date range: **{d_min}** to **{d_max}** ({n_good_days} good trading days)")
            c1, c2, c3 = st.columns(3)
            c1.metric("τ-bars (total)", f"{len(df_tau):,}")
            c2.metric("Good trading days", f"{n_good_days:,}")
            c3.metric("Post-burn-in bars", f"{len(df_js):,}")
            src_note = (
                f"Single contract: **{contract_name}**"
                if data_source == "Single contract"
                else "Sticky-roll chained contracts"
            )
            st.caption(
                f"{src_note}. The pipeline resamples to τ-min OHLCV, restricts to RTH, "
                "drops incomplete days, and discards the first J_START bars (burn-in). "
                "Only post-burn-in bars feed the ePDF statistics."
            )

        # ── Joint State Heatmap ───────────────────────────────────────────────
        elif sel1 == "Joint State Heatmap":
            st.subheader("Joint (Volume, Volatility) State Heatmap")
            ct     = pd.crosstab(df_tau["state_vol"], df_tau["state_sig"],
                                  normalize="all") * 100
            ct_cnt = pd.crosstab(df_tau["state_vol"], df_tau["state_sig"])
            chi2_stat, pval, dof, _ = chi2_contingency(ct_cnt)

            fig, ax = plt.subplots(figsize=(5, 3.6))
            im = ax.imshow(ct.values, cmap="YlOrRd", vmin=0, aspect="auto")
            vmax_ct = ct.values.max()
            for i in range(ct.shape[0]):
                for jj in range(ct.shape[1]):
                    ax.text(jj, i, f"{ct.iloc[i, jj]:.1f}%",
                            ha="center", va="center", fontweight="bold", fontsize=10,
                            color="white" if ct.iloc[i, jj] > vmax_ct * 0.65 else "black")
            ax.set_xticks(range(n_sig)); ax.set_yticks(range(m_vol))
            ax.set_xticklabels([f"σ={n}" for n in range(n_sig)])
            ax.set_yticklabels([f"v={m}" for m in range(m_vol)])
            ax.set_xlabel("Volatility state"); ax.set_ylabel("Volume state")
            ax.set_title(f"Joint frequency (%) — {instrument}")
            plt.colorbar(im, ax=ax, label="% of τ-bars")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close(fig)

            corr = "strongly correlated" if pval < 0.05 else "consistent with independence"
            st.info(
                f"χ²={chi2_stat:.1f},  dof={dof},  p={pval:.2e}  → states are **{corr}**.  "
                "Off-diagonal mass reveals that high-σ and high-v intervals tend to coincide."
            )
            st.caption(
                "Each cell shows what fraction of τ-bars fell in that (v, σ) pair.  "
                "A perfectly independent pair would show 1/(M×N)% everywhere."
            )

        # ── Unconditional ePDFs ───────────────────────────────────────────────
        elif sel1 == "Unconditional ePDFs":
            st.subheader("Unconditional Empirical PDFs")
            pdf_R   = epdf_from_array(df_js["R"].values)
            pdf_Rup = epdf_from_array(df_js["R_up"].values)
            pdf_Rdn = epdf_from_array(df_js["R_dn"].values)
            ell_arr = np.arange(MAX_T_PLOT)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, pdf, title in zip(
                axes,
                [pdf_R, pdf_Rup, pdf_Rdn],
                ["P(R = ell)  — total range",
                 "P(R_up = ell)  — high - open",
                 "P(R_dn = ell)  — open - low"],
            ):
                ax.bar(ell_arr, pdf[:MAX_T_PLOT], color="steelblue",
                       edgecolor="k", linewidth=0.3)
                ax.set_xlabel("ell  (ticks)"); ax.set_ylabel("Probability")
                ax.set_title(title)
            fig.suptitle(
                f"Unconditional ePDFs — {instrument}  τ={tau} min  (bars {j_start}+)",
                y=1.02,
            )
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "Naive (unconditional) histograms with no regime conditioning.  "
                "Right-skewed: most bars are small; large excursions are rare.  "
                "R = R_up + R_dn (up to rounding).  These are the baseline for KL divergence."
            )

        # ── Conditional ePDF Grid ─────────────────────────────────────────────
        elif sel1 == "Conditional ePDF Grid":
            st.subheader("Conditional ePDF Grid  —  P(R_dn | v, σ, Δx)")
            cond = full_cond_epdf(df_js, "R_dn")

            _base_colors = ["#cc3333", "#888888", "#2266aa", "#e6a700", "#2a9d2a"]
            dx_colors = [_base_colors[k_ % len(_base_colors)] for k_ in range(k_dir)]
            ell_plot = np.arange(MAX_T_PLOT)
            bar_w    = 0.80 / k_dir

            fig, axes = plt.subplots(
                m_vol, n_sig,
                figsize=(4.0 * n_sig, 2.8 * m_vol),
                sharex=True, sharey=True,
            )
            if m_vol == 1 and n_sig == 1:
                axes = np.array([[axes]])
            elif m_vol == 1:
                axes = axes[np.newaxis, :]
            elif n_sig == 1:
                axes = axes[:, np.newaxis]

            for m in range(m_vol):
                for n in range(n_sig):
                    ax = axes[m, n]
                    for k_ in range(k_dir):
                        p, _, n_obs = cond.get(
                            (m, n, k_),
                            (np.zeros(MAX_SPREADS + 1), np.zeros(MAX_SPREADS + 1), 0),
                        )
                        ax.bar(
                            ell_plot + (k_ - k_dir / 2 + 0.5) * bar_w,
                            p[:MAX_T_PLOT], bar_w, color=dx_colors[k_], alpha=0.85,
                            label=f"Δx={k_} (n={n_obs})" if (m == 0 and n == 0) else None,
                        )
                    ax.set_title(f"v={m}, σ={n}", fontsize=10)
                    ax.set_xlim(-0.5, 18)
                    if m == m_vol - 1: ax.set_xlabel("ticks")
                    if n == 0:         ax.set_ylabel("P(R_dn)")

            axes[0, 0].legend(fontsize=8, loc="upper right")
            fig.suptitle(
                f"P(R_dn | v, σ, Δx) — {instrument}  τ={tau} min",
                fontweight="bold",
            )
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "Each panel is one (volume, volatility) regime; bars within are Δx states.  "
                "High-σ / high-v cells typically have heavier tails — fills further from "
                "the open are more probable, which is the core insight of the paper."
            )

        # ── KL Divergence Grid ────────────────────────────────────────────────
        elif sel1 == "KL Divergence Grid":
            st.subheader("Information Gain — KL Divergence vs Naive Baseline")
            cond = full_cond_epdf(df_js, "R_dn")
            pdf_dn_naive = epdf_from_array(df_js["R_dn"].values)

            kl_grid = np.zeros((m_vol, n_sig, k_dir))
            for (m, n, k_), (p, _, n_obs) in cond.items():
                if n_obs > 0:
                    kl_grid[m, n, k_] = kl_div(p, pdf_dn_naive)

            fig, axes = plt.subplots(
                1, k_dir, figsize=(4.8 * k_dir, 4.2),
                sharey=True, constrained_layout=True,
            )
            if k_dir == 1:
                axes = [axes]
            vmax_kl = max(kl_grid.max(), 1e-6)
            for k_ in range(k_dir):
                ax = axes[k_]
                im = ax.imshow(kl_grid[:, :, k_], cmap="magma",
                               vmin=0, vmax=vmax_kl, aspect="auto")
                for m in range(m_vol):
                    for n in range(n_sig):
                        ax.text(n, m, f"{kl_grid[m, n, k_]:.3f}",
                                ha="center", va="center", fontsize=10, fontweight="bold",
                                color="white" if kl_grid[m, n, k_] < vmax_kl * 0.55
                                else "black")
                ax.set_xticks(range(n_sig)); ax.set_yticks(range(m_vol))
                ax.set_xticklabels([f"σ={n}" for n in range(n_sig)])
                ax.set_yticklabels([f"v={m}" for m in range(m_vol)])
                ax.set_xlabel("Volatility state"); ax.set_title(f"Δx state = {k_}")
                if k_ == 0: ax.set_ylabel("Volume state")

            fig.colorbar(im, ax=axes, label="KL(cond || naive)")
            fig.suptitle("KL divergence: information gain from conditioning on regime",
                         fontweight="bold")
            st.pyplot(fig); plt.close(fig)

            col_kl1, col_kl2 = st.columns(2)
            col_kl1.metric("Mean KL", f"{kl_grid.mean():.4f}")
            argmax = tuple(int(x) for x in np.unravel_index(kl_grid.argmax(), kl_grid.shape))
            col_kl2.metric("Max KL cell  (v, σ, Δx)", f"{kl_grid.max():.4f}  @ {argmax}")
            st.caption(
                "KL(P_cell ‖ P_naive) measures how much a cell departs from the "
                "unconditional baseline.  Brighter = higher economic value of conditioning."
            )

        # ── Fill Probability Table ────────────────────────────────────────────
        elif sel1 == "Fill Probability Table":
            st.subheader("Fill Probability Table  —  P(R_dn ≥ k·ε)")
            cond = full_cond_epdf(df_js, "R_dn")
            pdf_dn_naive = epdf_from_array(df_js["R_dn"].values)
            fp_naive  = fill_prob_from_pmf(pdf_dn_naive)
            k_targets = [1, 2, 3, 5, 8]

            rows = []
            for (m, n, k_), (p, fp, n_obs) in sorted(cond.items()):
                rows.append({
                    "(v, σ, Δx)": f"({m},{n},{k_})",
                    "n_obs": f"{n_obs:,}",
                    **{f"P(fill≥{k})": f"{fp[k]:.2%}" if k < len(fp) else "—"
                       for k in k_targets},
                })
            rows.append({
                "(v, σ, Δx)": "naive (all)",
                "n_obs": f"{len(df_js):,}",
                **{f"P(fill≥{k})": f"{fp_naive[k]:.2%}" if k < len(fp_naive) else "—"
                   for k in k_targets},
            })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "P(R_dn ≥ k·ε): probability a buy limit k ticks below open fills.  "
                "Naive row (bottom) is the unconditional baseline with no regime conditioning."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADING APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    FIG_OPTIONS_2 = ["Fill Probability Curves", "Execution Comparison"]
    sel2 = _fig_selector("View", FIG_OPTIONS_2, key="tab2_fig")

    # ── Fill Probability Curves ───────────────────────────────────────────────
    if sel2 == "Fill Probability Curves":
        if data_source == "AIAgent file":
            st.info(
                "Fill Probability Curves require the **Continuous** or **Single contract** "
                "pipeline. Switch the Data source to view them."
            )
        else:
            st.subheader("Fill Probability Curves by Regime")
            ell_range = np.arange(1, MAX_T_PLOT)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for m in range(m_vol):
                for n in range(n_sig):
                    cnt_dn = epdf_Rdn.counts[m, n, :, :].sum(axis=0)
                    cnt_up = epdf_Rup.counts[m, n, :, :].sum(axis=0)
                    tot_dn, tot_up = cnt_dn.sum(), cnt_up.sum()
                    if tot_dn == 0 or tot_up == 0:
                        continue
                    fp_buy  = np.array([cnt_dn[l:].sum() / tot_dn for l in ell_range])
                    fp_sell = np.array([cnt_up[l:].sum() / tot_up for l in ell_range])
                    lbl = f"v={m},σ={n}"
                    axes[0].plot(ell_range, fp_buy,  marker="o", ms=4, lw=1.2, label=lbl)
                    axes[1].plot(ell_range, fp_sell, marker="s", ms=4, lw=1.2, label=lbl)

            for ax, title in zip(
                axes,
                ["P(Buy filled) = P(R_dn >= ell)", "P(Sell filled) = P(R_up >= ell)"],
            ):
                ax.set_xlabel("ell  (ticks from open)")
                ax.set_ylabel("Fill probability")
                ax.set_title(title)
                ax.legend(fontsize=7, ncol=2)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0.5, MAX_T_PLOT - 0.5)

            fig.suptitle(f"{instrument} — Fill probability curves  (τ={tau} min)")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "Each curve shows how fill probability decreases as the limit moves "
                "further from the open.  High-σ/high-v regimes have flatter curves "
                "(price travels further), enabling profitable deep placement."
            )

    # ── Execution Comparison ──────────────────────────────────────────────────
    elif sel2 == "Execution Comparison":
        st.subheader("Market vs. Naive Limit vs. ePDF-guided Limit, across signal sources")
        st.markdown(
            "**Execution layer vs. signal layer.** This project's contribution is the "
            "*execution layer*: given a directional decision from any signal source, how "
            "should we place limit orders to improve execution quality? The signal source "
            "itself is exogenous — it could be a model, a rule, or external trades. "
            "Choose a placeholder signal below, then compare three execution methods on "
            "every signal-bar."
        )

        signal_source = st.radio(
            "Signal source",
            [
                "EWMA-based (mean reversion)",
                "EWMA-based (trend following)",
                "AIAgent trades",
            ],
            help=(
                "Mean reversion: bet that recent moves reverse  (signal = −sign(ewma_ret)). "
                "Placeholder directional signal.  ‖  "
                "Trend following: bet that recent moves continue  (signal = +sign(ewma_ret)). "
                "Placeholder directional signal.  ‖  "
                "AIAgent trades: realized trades from AIAgent_{INSTRUMENT}.csv. "
                "External signal source."
            ),
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            min_fill_prob = st.slider(
                "MIN_FILL_PROB (Method C)", 0.40, 0.90, 0.60, step=0.05,
                help="Method C only places a limit when predicted fill prob ≥ this threshold.",
            )
        with col_b:
            max_offset = int(st.slider(
                "MAX_OFFSET (Method C, ticks)", 2, 10, 6,
                help="Largest offset from open Method C will consider.",
            ))
        with col_c:
            ell_fixed = int(st.slider(
                "Fixed offset ℓ (Method B, ticks)", 1, 10, 3,
                help="Fixed limit placement for Method B; fills if price travels this far.",
            ))

        # Ensure Continuous pipeline is available for bt, ePDFs, and R_dn/R_up actuals.
        # In Continuous/Single mode these are already loaded; AIAgent mode auto-loads.
        if data_source == "AIAgent file":
            with st.spinner("Building ePDFs from continuous history (one-time, ~30 s)…"):
                _c_df_tau, _, _c_epdf_Rup, _c_epdf_Rdn, _c_bt = _run_continuous(
                    instrument, tau, half_life, m_vol, n_sig, k_dir, j_start
                )
            _bt_ec, _epdf_Rup_ec, _epdf_Rdn_ec, _df_tau_ec = (
                _c_bt, _c_epdf_Rup, _c_epdf_Rdn, _c_df_tau
            )
            if not st.session_state.get("auto_continuous_tab2_explained", False):
                st.info(
                    "ℹ️ Note: Execution Comparison requires the Continuous pipeline for "
                    "realized R_dn/R_up values and ePDFs. These were loaded automatically. "
                    "Switching instruments or τ will re-trigger this build."
                )
                st.session_state["auto_continuous_tab2_explained"] = True
        else:
            _bt_ec, _epdf_Rup_ec, _epdf_Rdn_ec, _df_tau_ec = bt, epdf_Rup, epdf_Rdn, df_tau

        # Mini fill-prob reference chart showing both method thresholds
        ell_range = np.arange(1, MAX_T_PLOT)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        for m in range(m_vol):
            for n in range(n_sig):
                cnt_dn = _epdf_Rdn_ec.counts[m, n, :, :].sum(axis=0)
                cnt_up = _epdf_Rup_ec.counts[m, n, :, :].sum(axis=0)
                tot_dn, tot_up = cnt_dn.sum(), cnt_up.sum()
                if tot_dn == 0 or tot_up == 0:
                    continue
                axes[0].plot(ell_range,
                              [cnt_dn[l:].sum() / tot_dn for l in ell_range],
                              lw=1.0, alpha=0.7, label=f"v={m},σ={n}")
                axes[1].plot(ell_range,
                              [cnt_up[l:].sum() / tot_up for l in ell_range],
                              lw=1.0, alpha=0.7)
        for ax, title in zip(axes, ["P(Buy filled)", "P(Sell filled)"]):
            ax.axhline(min_fill_prob, color="red", ls="--", lw=1.4, alpha=0.9,
                       label=f"C threshold={min_fill_prob:.2f}")
            ax.axvline(ell_fixed,   color="darkorange", ls=":",  lw=1.4,
                       label=f"B ell={ell_fixed}")
            ax.axvline(max_offset,  color="steelblue",  ls=":",  lw=1.4,
                       label=f"C max={max_offset}")
            ax.set_xlim(0.5, 15); ax.set_ylim(0, 1.05)
            ax.set_xlabel("ell (ticks)"); ax.set_title(title)
            ax.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        if st.button("▶  Run Execution Comparison", type="primary"):
            with st.spinner("Running execution comparison…"):

                # ── Build signal series and evaluation universe ───────────
                if signal_source.startswith("EWMA"):
                    ewma_ret_arr, _ = ewma_ewmv(_df_tau_ec["ret"].values, lam)
                    ewma_ret = pd.Series(ewma_ret_arr, index=_df_tau_ec.index)
                    if "mean reversion" in signal_source:
                        sig_series = ewma_ret.reindex(_bt_ec.index).apply(
                            lambda x: -1 if x >= 0 else 1
                        )
                    else:
                        sig_series = ewma_ret.reindex(_bt_ec.index).apply(
                            lambda x: 1 if x >= 0 else -1
                        )
                    eval_bt = _bt_ec

                else:  # AIAgent trades
                    try:
                        agent_raw, _ = _load_aiagent(instrument, tau)
                    except FileNotFoundError:
                        st.error(
                            f"AIAgent_{instrument}.csv not found in the data directory."
                        )
                        st.stop()
                    pos_tau = (agent_raw["net_pos"]
                               .resample(f"{tau}min", label="left", closed="left")
                               .last())
                    pos_delta = pos_tau.diff().reindex(_bt_ec.index).fillna(0)
                    mask = pos_delta.abs() > 0
                    if not mask.any():
                        st.warning(
                            f"No AIAgent trades align with Continuous τ-bars "
                            f"({instrument}, τ={tau} min). Try a different τ."
                        )
                        st.stop()
                    eval_bt   = _bt_ec[mask]
                    sig_series = pos_delta[mask].apply(lambda x: 1 if x > 0 else -1)

                # ── Three-method loop ─────────────────────────────────────
                recs = []
                for idx, row in eval_bt.iterrows():
                    sig = int(sig_series.loc[idx])
                    rA = _exec_market(sig, row, eps)
                    rB = _exec_naive(sig, row, eps, ell_fixed)
                    rC = _exec_epdf(sig, row, eps, _epdf_Rup_ec, _epdf_Rdn_ec,
                                    max_offset, min_fill_prob)
                    recs.append({
                        "pnl_A": rA["pnl"],
                        "pnl_B": rB["pnl"], "filled_B": rB["filled"],
                        "pnl_C": rC["pnl"], "filled_C": rC["filled"],
                        "traded_C": rC.get("traded", True),
                    })
                res = pd.DataFrame(recs, index=eval_bt.index)

            # ── 1. Cumulative PnL overlay ─────────────────────────────────
            fig, ax = plt.subplots(figsize=(13, 3.5))
            for col, label, color in [
                ("pnl_A", "A — Market",                   "gray"),
                ("pnl_B", f"B — Naive limit (ell={ell_fixed})", "darkorange"),
                ("pnl_C", "C — ePDF-guided limit",         "steelblue"),
            ]:
                ax.plot(res.index, res[col].cumsum(), lw=1.6, label=label, color=color)
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_ylabel("Cumulative ticks"); ax.set_xlabel("Date")
            ax.set_title(
                f"Execution Comparison — {instrument}  τ={tau} min  "
                f"({signal_source})"
            )
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "All three methods applied to the same signal-bars. "
                "Method A is market execution at open; B places a fixed-offset limit with "
                "market fallback on miss; C adapts the offset per regime (sits out when no "
                "offset meets the fill-prob threshold)."
            )

            # ── 2. Per-method PnL distributions ──────────────────────────
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
            for ax, col, mask_col, label, color in [
                (axes[0], "pnl_A", None,        "A — Market",                   "gray"),
                (axes[1], "pnl_B", None,        f"B — Naive (ell={ell_fixed})", "darkorange"),
                (axes[2], "pnl_C", "traded_C",  "C — ePDF-guided",              "steelblue"),
            ]:
                data = res.loc[res[mask_col], col] if mask_col else res[col]
                if len(data) == 0:
                    ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                            transform=ax.transAxes)
                else:
                    ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
                    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
                    mv = data.mean()
                    ax.axvline(mv, color="red", lw=1.6, label=f"Mean={mv:+.2f}")
                    ax.legend(fontsize=8)
                ax.set_title(label); ax.set_xlabel("PnL (ticks)")
                if ax is axes[0]:
                    ax.set_ylabel("Count")
            fig.suptitle("Per-trade PnL distribution by execution method",
                         fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "Method C histogram covers only bars where an offset qualified "
                "(traded_C=True); non-traded bars contribute PnL=0 to the running total "
                "but are excluded from this distribution."
            )

            # ── 3. Summary table ──────────────────────────────────────────
            n_signal_bars = len(res)
            tbl_rows = []
            for label, col_pnl, is_c in [
                ("A — Market order",               "pnl_A", False),
                (f"B — Naive limit (ell={ell_fixed})", "pnl_B", False),
                ("C — ePDF-guided limit",           "pnl_C", True),
            ]:
                pnl_s = res[col_pnl]
                if is_c:
                    traded_mask = res["traded_C"]
                    filled_mask = res["filled_C"] & traded_mask
                    n_traded    = int(traded_mask.sum())
                    fill_rate_v = filled_mask[traded_mask].mean() if n_traded else np.nan
                    win_rate_v  = (pnl_s[traded_mask] > 0).mean() if n_traded else np.nan
                else:
                    n_traded    = n_signal_bars
                    filled_mask = res["filled_B"] if col_pnl == "pnl_B" else pd.Series(
                        True, index=res.index)
                    fill_rate_v = filled_mask.mean()
                    win_rate_v  = (pnl_s > 0).mean()

                total_ticks_v = pnl_s.sum()
                mean_pnl_v    = pnl_s.mean()
                sharpe_v      = (mean_pnl_v / (pnl_s.std() + 1e-12)
                                 * np.sqrt(252 * 390 / tau))
                improvement_v = (pnl_s - res["pnl_A"]).mean()

                tbl_rows.append({
                    "Method":                        label,
                    "Bars traded":                   f"{n_traded:,}",
                    "Fill rate":                     f"{fill_rate_v:.1%}",
                    "Improvement vs A (ticks/bar)":  f"{improvement_v:+.3f}",
                    "Total PnL (ticks)":             f"{total_ticks_v:+.0f}",
                    "Total PnL ($)":                 f"${total_ticks_v * eps:+,.2f}",
                    "Mean PnL/bar":                  f"{mean_pnl_v:+.4f}",
                    "Win rate":                      f"{win_rate_v:.1%}" if not np.isnan(win_rate_v) else "—",
                    "Sharpe":                        f"{sharpe_v:+.2f}",
                })

            st.subheader("Execution Comparison Summary")
            st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

            c_vs_a = (res["pnl_C"] - res["pnl_A"]).mean()
            c_vs_b = (res["pnl_C"] - res["pnl_B"]).mean()
            st.caption(
                f"On signal source **{signal_source}**: Method C improves average entry by "
                f"**{c_vs_a:+.3f} ticks/bar** over market orders (A) and "
                f"**{c_vs_b:+.3f} ticks/bar** over the fixed-offset baseline (B)."
            )
            st.caption(
                "Method A fills at open. "
                f"Method B places limit {ell_fixed} tick(s) from open; if missed, "
                "falls back to market at close (−0.5 tick slippage). "
                "Method C picks the largest ell where P(fill|regime) ≥ MIN_FILL_PROB; "
                "if no ell qualifies the bar is skipped (PnL=0); if placed but missed, "
                "same −0.5 tick fallback applies. All methods exit at bar close."
            )
