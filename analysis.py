"""
analysis.py — Stand-alone analysis script (extracted from mc_helpers __main__ block).

Run directly:   python analysis.py
Or import:      import analysis   (runs all cells on import — intended for scripting)

All configuration comes from config.py. Change INSTRUMENT, TAU, etc. there before
running, or override them after importing config:

    import config; config.INSTRUMENT = "Gold"; config.TAU = 30
    import analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

import config
from config import (INSTRUMENT, TAU, HALF_LIFE, EPS, LAM,
                    M_VOL_STATES, N_SIG_STATES, K_DIR_STATES,
                    J_START, MAX_SPREADS, MAX_T_PLOT, MIN_BAR_FRAC,
                    DATA_ROOT, FIG_DIR, MARKETS, AIAGENT_FILENAME, savefig)
from data import load_instrument, resample_ohlcv
from features import (compute_ranges, ewma_ewmv, quantile_states_causal, add_states)
from epdf import (epdf_from_array, fill_prob_from_pmf, kl_div,
                  build_rolling_epdfs, full_cond_epdf)
from pipeline import prepare_market


# ── Main pipeline ─────────────────────────────────────────────────────────────

print(f"Instrument : {INSTRUMENT}")
print(f"Tick size  : {EPS}")
print(f"Tau        : {TAU} min")
print(f"Lambda     : {LAM:.6f}  (half-life={HALF_LIFE} bars)")

df_tau, EPS = prepare_market(INSTRUMENT, TAU)
print(f"\n── Building state-conditioned ePDFs (walk-forward) ──")
epdf_R, epdf_Rup, epdf_Rdn, bt = build_rolling_epdfs(df_tau)
print(f"  ePDF tables built. Backtest rows: {len(bt):,}")

df_tau.head()


# ── Timezone diagnostic ───────────────────────────────────────────────────────

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


# ── Part 1.1 — Unconditional empirical PDF ────────────────────────────────────

sub = df_tau.iloc[J_START:]
pdf_R   = epdf_from_array(sub["R"].values)
pdf_Rup = epdf_from_array(sub["R_up"].values)
pdf_Rdn = epdf_from_array(sub["R_dn"].values)

ell = np.arange(MAX_T_PLOT)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, pdf, title in zip(axes,
                          [pdf_R, pdf_Rup, pdf_Rdn],
                          ["P(R = l)", "P(R_up = l)", "P(R_dn = l)"]):
    ax.bar(ell, pdf[:MAX_T_PLOT], color="steelblue", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("l  (number of spreads)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
plt.suptitle(f"Unconditional ePDFs — {INSTRUMENT}  tau={TAU} min  (bars {J_START}+)", y=1.02)
plt.tight_layout()
savefig(fig, f"fig01_unconditional_epdf_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.2 — EWMA dynamics ──────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(df_tau.index, df_tau["volume"],   alpha=0.3, label="volume")
axes[0].plot(df_tau.index, df_tau["ewma_vol"], lw=1.5,    label="EWMA volume")
axes[0].set_ylabel("Volume"); axes[0].legend()
axes[1].plot(df_tau.index, df_tau["R"],        alpha=0.3, label="range (ticks)")
axes[1].plot(df_tau.index, df_tau["ewma_rng"], lw=1.5,    label="EWMA range")
axes[1].set_ylabel("Range"); axes[1].legend()
plt.suptitle(f"{INSTRUMENT} — EWMA (half-life={HALF_LIFE} tau-bars)")
plt.tight_layout()
savefig(fig, f"fig02_ewma_dynamics_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.3 — Joint state occupancy ─────────────────────────────────────────

ct = pd.crosstab(df_tau["state_vol"], df_tau["state_sig"], normalize="all") * 100

fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(ct.values, cmap="YlOrRd", vmin=0, aspect="auto")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, f"{ct.iloc[i,j]:.1f}%", ha="center", va="center",
                color="black" if ct.iloc[i,j] < 12 else "white", fontweight="bold")
ax.set_xticks(range(N_SIG_STATES))
ax.set_yticks(range(M_VOL_STATES))
ax.set_xticklabels([f"sigma={n}" for n in range(N_SIG_STATES)])
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
print(f"\nchi^2 test of (volume, volatility) independence:")
print(f"  chi^2 = {chi2:.1f},  dof = {dof},  p-value = {pval:.2e}")
print("  → small p => states are strongly correlated")


# ── Part 1.4 — Reproduction of Figure 2 from the paper ───────────────────────

df_js = df_tau.iloc[J_START:].copy()

by_sigma = {}
for s in sorted(df_js["state_sig"].unique()):
    sub  = df_js[df_js["state_sig"] == s]
    p    = epdf_from_array(sub["R_dn"].values)
    fp   = fill_prob_from_pmf(p)
    by_sigma[int(s)] = (p, fp, len(sub))

colors_seg = ["#1f4e79", "#2e9c8e", "#e6a700"]
labels_seg = ["sigma state 0 (low)", "sigma state 1 (med)", "sigma state 2 (high)"]
width      = 0.27
ell_plot   = np.arange(MAX_T_PLOT)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, s in enumerate(sorted(by_sigma)):
    p, fp, n_obs = by_sigma[s]
    counts = (p[:MAX_T_PLOT] * n_obs).astype(int)
    fill_counts = (fp[:MAX_T_PLOT] * n_obs).astype(int)
    off = (i - 1) * width
    axes[0,0].bar(ell_plot + off, counts,          width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)
    axes[0,1].bar(ell_plot + off, fill_counts,     width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)
    axes[1,0].bar(ell_plot + off, p[:MAX_T_PLOT],  width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)
    axes[1,1].bar(ell_plot + off, fp[:MAX_T_PLOT], width, color=colors_seg[i], label=labels_seg[i], alpha=0.9)

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

fig.suptitle(f"Reproduction of Figure 2 — {INSTRUMENT}, tau = {TAU} min",
             fontweight="bold", fontsize=13)
plt.tight_layout()
savefig(fig, f"fig04_paper_figure2_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.5 — Full conditional ePDF grid ────────────────────────────────────

cond = full_cond_epdf(df_js, "R_dn")

fig, axes = plt.subplots(M_VOL_STATES, N_SIG_STATES, figsize=(12, 10),
                         sharex=True, sharey=True)
dx_colors = ["#cc3333", "#888888", "#2266aa"]
dx_labels = ["dx state 0 (down)", "dx state 1 (flat)", "dx state 2 (up)"]
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
        ax.set_title(f"v={m}, sigma={n}", fontsize=10)
        ax.set_xlim(-0.5, 18)
        if m == M_VOL_STATES - 1: ax.set_xlabel("ticks")
        if n == 0:                ax.set_ylabel("P(R_dn)")

axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"Conditional ePDF P(R_dn | v, sigma, dx) — {INSTRUMENT}, tau={TAU} min",
             fontweight="bold")
plt.tight_layout()
savefig(fig, f"fig05_full_conditional_grid_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.6 — Table 1: quantitative effect of conditioning ──────────────────

pdf_dn_naive = epdf_from_array(df_js["R_dn"].values)
fp_naive     = fill_prob_from_pmf(pdf_dn_naive)

k_targets = [1, 2, 3, 5, 8]
rows = []
for (m, n, k_), (p, fp, n_obs) in cond.items():
    rows.append({
        "(v, sigma, dx)": f"({m},{n},{k_})",
        "n_obs"         : n_obs,
        **{f"P(fill>={k})": fp[k] for k in k_targets}
    })
rows.append({
    "(v, sigma, dx)": "naive (all)",
    "n_obs"         : len(df_js),
    **{f"P(fill>={k})": fp_naive[k] for k in k_targets}
})

tbl = pd.DataFrame(rows)
tbl_display = tbl.copy()
for col in tbl_display.columns[2:]:
    tbl_display[col] = tbl_display[col].apply(lambda x: f"{x:.2%}")
print(tbl_display.to_string(index=False))

tbl.to_csv(FIG_DIR / f"tab01_conditional_fill_probabilities_{INSTRUMENT.split()[0]}.csv", index=False)
print(f"\n→ saved figures/tab01_conditional_fill_probabilities_{INSTRUMENT.split()[0]}.csv")


# ── Part 1.7 — Non-stationarity of the naive ePDF ────────────────────────────

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

fig.suptitle(f"Distributions are NOT stationary — {INSTRUMENT}, tau = {TAU} min",
             fontweight="bold")
plt.tight_layout()
savefig(fig, f"fig06_segment_drift_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.8 — Information content of conditioning (KL divergence) ────────────

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
    ax.set_xticklabels([f"sigma={n}" for n in range(N_SIG_STATES)])
    ax.set_yticklabels([f"v={m}" for m in range(M_VOL_STATES)])
    ax.set_xlabel("Volatility state")
    ax.set_title(f"dx state = {k_}")
    if k_ == 0: ax.set_ylabel("Volume state")

fig.colorbar(im, ax=axes, label="KL(cond || naive)", shrink=0.85)
fig.suptitle("Information gain from conditioning — KL divergence vs naive baseline",
             fontweight="bold")
savefig(fig, f"fig07_kl_grid_{INSTRUMENT.split()[0]}")
plt.show()

print(f"\nMean KL = {kl_grid.mean():.3f}")
print(f"Max  KL = {kl_grid.max():.3f} at cell (v, sigma, dx) = "
      f"{tuple(int(x) for x in np.unravel_index(kl_grid.argmax(), kl_grid.shape))}")


# ── Part 1.9 — Cross-market generalization ────────────────────────────────────

markets_to_sweep = [INSTRUMENT]
# markets_to_sweep = ["Nasdaq", "Gold", "German Bunds - German Government Bonds", "EuroStoxx"]
# markets_to_sweep = list(MARKETS.keys())

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
                 label=f"{label_name} (eps={eps_m})")
    axes[1].plot(np.arange(MAX_T_PLOT), fp_m[:MAX_T_PLOT],
                 color=mkt_colors[i], lw=2, marker="o", markersize=4,
                 label=label_name)

axes[0].set_title("ePDF of R_dn")
axes[0].set_xlabel("Number of ticks"); axes[0].set_ylabel("Probability")
axes[0].legend(fontsize=9); axes[0].set_xlim(-0.5, MAX_T_PLOT)
axes[1].set_title("Fill probability")
axes[1].set_xlabel("k (ticks below open)"); axes[1].set_ylabel("P(fill)")
axes[1].legend(fontsize=9); axes[1].set_xlim(-0.5, MAX_T_PLOT)
fig.suptitle(f"Cross-market generalization — tau = {TAU} min", fontweight="bold")
plt.tight_layout()
savefig(fig, "fig08_cross_market")
plt.show()


# ── Part 1.10 — Sensitivity to the holding period tau ────────────────────────

# TAU_GRID = [5, 10, 15, 30, 60]   # Uncomment to run full sweep
TAU_GRID = [TAU]

tau_cols = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(TAU_GRID), 2)))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for i, tau in enumerate(TAU_GRID):
    df_t, eps_t = prepare_market(INSTRUMENT, tau, verbose=False)
    df_t_js     = df_t.iloc[J_START:]
    p_t         = epdf_from_array(df_t_js["R_dn"].values, max_ell=MAX_T_PLOT * 2)
    fp_t        = fill_prob_from_pmf(p_t)
    x           = np.arange(len(p_t))
    axes[0].plot(x, p_t,  color=tau_cols[i], lw=1.8, marker="o", markersize=3, label=f"tau = {tau} min")
    axes[1].plot(x, fp_t, color=tau_cols[i], lw=1.8, marker="o", markersize=3, label=f"tau = {tau} min")

axes[0].set_title("ePDF of R_dn vs tau")
axes[0].set_xlabel("Number of ticks"); axes[0].set_ylabel("Probability")
axes[0].legend(); axes[0].set_xlim(-0.5, MAX_T_PLOT * 2)
axes[1].set_title("Fill probability vs tau")
axes[1].set_xlabel("k (ticks below open)"); axes[1].set_ylabel("P(fill)")
axes[1].axhline(0.5, color="gray", lw=0.7, linestyle="--")
axes[1].legend(); axes[1].set_xlim(-0.5, MAX_T_PLOT * 2)
fig.suptitle(f"Sensitivity to holding period — {INSTRUMENT}", fontweight="bold")
plt.tight_layout()
savefig(fig, f"fig09_tau_sensitivity_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.11 — Buy/sell asymmetry conditioned on prior direction ─────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
dx_titles = ["dx state 0 (prior down)", "dx state 1 (prior flat)", "dx state 2 (prior up)"]
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

fig.suptitle(f"Buy/Sell asymmetry conditioned on prior direction — {INSTRUMENT}, tau = {TAU} min",
             fontweight="bold")
plt.tight_layout()
savefig(fig, f"fig10_buy_sell_asymmetry_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.12 — Out-of-sample placement and slippage decision ────────────────

def cond_fill_curve(df: pd.DataFrame, target: str = "R_dn") -> dict:
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

ks = np.arange(MAX_T_PLOT)
for s, (p, fp) in train_curves.items():
    eg = ks * EPS * fp[:MAX_T_PLOT]
    axes[0].plot(ks, eg, lw=2, marker="o", markersize=4, color=seg_colors[s % 3],
                 label=f"sigma state {s}")
    axes[0].axvline(int(ks[np.argmax(eg)]), color=seg_colors[s % 3], ls="--", lw=0.7)
axes[0].set_title(r"Expected gain  k * eps * P(fill)  — train")
axes[0].set_xlabel("k (ticks)"); axes[0].set_ylabel("Expected gain ($)")
axes[0].legend(); axes[0].set_xlim(-0.5, MAX_T_PLOT)

rows_oos = []
for s, (p_tr, fp_tr) in train_curves.items():
    eg_tr   = ks * EPS * fp_tr[:MAX_T_PLOT]
    k_star  = int(ks[np.argmax(eg_tr)])
    fp_te   = test_curves[s][1] if s in test_curves else fp_train_naive
    realized = k_star * EPS * (fp_te[k_star] if k_star < len(fp_te) else 0.0)
    naive    = k_star * EPS * (fp_train_naive[k_star] if k_star < len(fp_train_naive) else 0.0)
    rows_oos.append([s, k_star, realized, naive])

df_oos = pd.DataFrame(rows_oos, columns=["sigma_state","k*","OOS realized gain","Naive baseline gain"])
x = np.arange(len(df_oos))
axes[1].bar(x - 0.2, df_oos["OOS realized gain"],   0.4, color="#1f4e79", label="State-conditioned")
axes[1].bar(x + 0.2, df_oos["Naive baseline gain"], 0.4, color="#c0392b", label="Naive baseline")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"sigma={s} (k*={k})" for s, k in zip(df_oos["sigma_state"], df_oos["k*"])])
axes[1].set_title("Out-of-sample realized expected gain")
axes[1].set_ylabel("Expected gain ($)"); axes[1].legend()

fig.suptitle(f"Optimal limit-order placement — {INSTRUMENT}, tau = {TAU} min", fontweight="bold")
plt.tight_layout()
savefig(fig, f"fig11_oos_optimal_placement_{INSTRUMENT.split()[0]}")
plt.show()

print("\nOut-of-sample comparison:")
print(df_oos.to_string(index=False))
df_oos.to_csv(FIG_DIR / f"tab02_oos_optimal_placement_{INSTRUMENT.split()[0]}.csv", index=False)

MARKET_FALLBACK_COST = 0.5 * EPS
fig, ax = plt.subplots(figsize=(8, 4.5))
for s, (p, fp) in test_curves.items():
    slippage = (1 - fp[:MAX_T_PLOT]) * MARKET_FALLBACK_COST - ks * EPS * fp[:MAX_T_PLOT]
    ax.plot(ks, slippage, lw=2, marker="o", markersize=4, color=seg_colors[s % 3],
            label=f"sigma state {s}")
    k_best = int(ks[np.argmin(slippage)])
    ax.scatter([k_best], [slippage[k_best]], s=90, edgecolor="black",
               facecolor=seg_colors[s % 3], zorder=5, linewidth=1.2)
    ax.annotate(f"  k*={k_best}", (k_best, slippage[k_best]), fontsize=9)

ax.axhline(0, color="gray", lw=0.7, ls="--")
ax.set_title("Expected slippage curves by volatility state")
ax.set_xlabel("k (ticks below open)"); ax.set_ylabel("Expected slippage ($)")
ax.legend(); ax.set_xlim(-0.5, MAX_T_PLOT)
fig.suptitle(f"Slippage decision surface — {INSTRUMENT}, tau = {TAU}", fontweight="bold")
plt.tight_layout()
savefig(fig, f"fig12_slippage_decision_{INSTRUMENT.split()[0]}")
plt.show()


# ── Part 1.13 — State stability through time ──────────────────────────────────

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
ax.set_yticklabels([f"sigma state {s}" for s in range(N_SIG_STATES)])
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


# ── Part 2 — Visualise Conditional ePDF (Figure 2 style) ─────────────────────

ell_x   = np.arange(MAX_SPREADS + 1)
n_segs  = N_SIG_STATES
colors  = ["steelblue", "darkorange", "seagreen"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ss in range(n_segs):
    cnt_rdn   = epdf_Rdn.counts[:, ss, :, :].sum(axis=(0, 1))
    cnt_rup   = epdf_Rup.counts[:, ss, :, :].sum(axis=(0, 1))
    total_rdn = cnt_rdn.sum()
    total_rup = cnt_rup.sum()
    pdf_rdn   = cnt_rdn / total_rdn if total_rdn > 0 else cnt_rdn
    pdf_rup   = cnt_rup / total_rup if total_rup > 0 else cnt_rup

    fill_cnt = np.array([cnt_rdn[l:].sum() for l in ell_x])
    fill_p   = fill_cnt / total_rdn if total_rdn > 0 else fill_cnt

    w      = 0.25
    offset = (ss - n_segs / 2) * w
    lbl    = f"sigma-state {ss}"

    axes[0, 0].bar(ell_x[:12] + offset, cnt_rdn[:12],  width=w, color=colors[ss], label=lbl, alpha=0.85)
    axes[1, 0].bar(ell_x[:12] + offset, pdf_rdn[:12],  width=w, color=colors[ss], alpha=0.85)
    axes[0, 1].bar(ell_x[:12] + offset, fill_cnt[:12], width=w, color=colors[ss], alpha=0.85)
    axes[1, 1].bar(ell_x[:12] + offset, fill_p[:12],   width=w, color=colors[ss], alpha=0.85)

axes[0, 0].set_title("Counts / frequencies (RangeDn)"); axes[0, 0].set_ylabel("Count"); axes[0, 0].legend()
axes[1, 0].set_title("ePDF of RangeDn");                axes[1, 0].set_ylabel("P(R_dn = l)")
axes[0, 1].set_title("Counts of getting filled");       axes[0, 1].set_ylabel("Count")
axes[1, 1].set_title("P(fill) vs. number of spreads"); axes[1, 1].set_ylabel("P(R_dn >= l)")
for ax in axes.flat:
    ax.set_xlabel("Number of spreads l")

plt.suptitle(f"{INSTRUMENT} — Conditional ePDF by sigma-state  (tau={TAU} min)", y=1.01)
plt.tight_layout()
plt.show()


# ── Part 2 — Fill Probability Curves ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ell_range = np.arange(1, 13)

for m in range(M_VOL_STATES):
    for n in range(N_SIG_STATES):
        cnt_dn = epdf_Rdn.counts[m, n, :, :].sum(axis=0)
        cnt_up = epdf_Rup.counts[m, n, :, :].sum(axis=0)
        tot_dn = cnt_dn.sum()
        tot_up = cnt_up.sum()
        if tot_dn == 0 or tot_up == 0:
            continue
        fp_buy  = np.array([cnt_dn[l:].sum() / tot_dn for l in ell_range])
        fp_sell = np.array([cnt_up[l:].sum() / tot_up for l in ell_range])
        lbl = f"v={m},sigma={n}"
        axes[0].plot(ell_range, fp_buy,  marker="o", ms=4, lw=1.2, label=lbl)
        axes[1].plot(ell_range, fp_sell, marker="s", ms=4, lw=1.2, label=lbl)

for ax, title in zip(axes, ["P(Buy filled) = P(R_dn >= l)",
                              "P(Sell filled) = P(R_up >= l)"]):
    ax.set_xlabel("l (ticks below/above open)")
    ax.set_ylabel("Fill probability")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

plt.suptitle(f"{INSTRUMENT} — Fill probability by regime  (tau={TAU} min)")
plt.tight_layout()
plt.show()


# ── Part 2 — Backtest ─────────────────────────────────────────────────────────

MIN_FILL_PROB = 0.60
MAX_OFFSET    = 6
SIGNAL_MODE   = "mean_reversion"   # "mean_reversion" or "trend_following"

ewma_ret_arr, _ = ewma_ewmv(df_tau["ret"].values, LAM)
df_tau["ewma_ret"] = ewma_ret_arr

if SIGNAL_MODE == "mean_reversion":
    bt["signal"] = df_tau["ewma_ret"].reindex(bt.index).apply(
        lambda x: -1 if x >= 0 else 1)
elif SIGNAL_MODE == "trend_following":
    bt["signal"] = df_tau["ewma_ret"].reindex(bt.index).apply(
        lambda x: 1 if x >= 0 else -1)
else:
    raise ValueError(f"Unknown SIGNAL_MODE: {SIGNAL_MODE}")


def best_offset(fp_cols, row, min_prob: float, max_off: int) -> int:
    best = 0
    for ell in range(1, max_off + 1):
        if row[fp_cols[ell - 1]] >= min_prob:
            best = ell
        else:
            break
    return best


fp_rup_cols = [f"fp_rup_{l}" for l in range(1, MAX_OFFSET + 1)]
fp_rdn_cols = [f"fp_rdn_{l}" for l in range(1, MAX_OFFSET + 1)]

results = []
for _, row in bt.iterrows():
    sig = row["signal"]

    if sig == 1:
        ell = best_offset(fp_rdn_cols, row, MIN_FILL_PROB, MAX_OFFSET)
        if ell == 0:
            results.append({"timestamp": row.name, "signal": sig, "ell": 0,
                            "filled": False, "pnl_ticks": 0.0,
                            "p_pred": np.nan, "traded": False})
            continue
        lp    = row["open"] - ell * EPS
        hit   = row["Rdn_actual"] >= ell
        pnl   = (row["close"] - lp) / EPS if hit else 0.0
        p_pred = row[fp_rdn_cols[ell - 1]]
    else:
        ell = best_offset(fp_rup_cols, row, MIN_FILL_PROB, MAX_OFFSET)
        if ell == 0:
            results.append({"timestamp": row.name, "signal": sig, "ell": 0,
                            "filled": False, "pnl_ticks": 0.0,
                            "p_pred": np.nan, "traded": False})
            continue
        lp    = row["open"] + ell * EPS
        hit   = row["Rup_actual"] >= ell
        pnl   = (lp - row["close"]) / EPS if hit else 0.0
        p_pred = row[fp_rup_cols[ell - 1]]

    results.append({"timestamp": row.name, "signal": sig, "ell": ell,
                    "filled": hit, "pnl_ticks": pnl,
                    "p_pred": p_pred, "traded": True})

res = pd.DataFrame(results).set_index("timestamp")

n_total       = len(res)
n_traded      = res["traded"].sum()
n_filled      = res["filled"].sum()
fill_rate     = n_filled / n_traded if n_traded else np.nan
mean_p_pred   = res.loc[res["traded"], "p_pred"].mean()
calibration   = fill_rate - mean_p_pred
mean_pnl_bar  = res["pnl_ticks"].mean()
sharpe_bar    = (mean_pnl_bar /
                 (res["pnl_ticks"].std() + 1e-12) *
                 np.sqrt(252 * 390 / TAU))
filled_rows   = res[res["filled"]]
mean_pnl_fill = filled_rows["pnl_ticks"].mean() if len(filled_rows) else np.nan
win_rate      = (filled_rows["pnl_ticks"] > 0).mean() if len(filled_rows) else np.nan
mean_ell      = res.loc[res["traded"], "ell"].mean()

print(f"── Backtest ({SIGNAL_MODE}, MIN_FILL_PROB={MIN_FILL_PROB}, tau={TAU} min) ──")
print(f"Total bars                : {n_total:,}")
print(f"Bars we placed a limit    : {n_traded:,}  ({n_traded/n_total:.1%})")
print(f"Bars filled               : {n_filled:,}  (fill rate of traded = {fill_rate:.2%})")
print(f"Mean predicted fill prob  : {mean_p_pred:.3f}")
print(f"Calibration (realised-pred): {calibration:+.3f}")
print(f"Mean limit offset (ticks) : {mean_ell:.2f}")
print(f"Mean PnL per BAR (ticks)  : {mean_pnl_bar:+.4f}")
print(f"Mean PnL per FILL (ticks) : {mean_pnl_fill:+.4f}")
print(f"Win rate on filled trades : {win_rate:.2%}")
print(f"Annualised Sharpe (per-bar): {sharpe_bar:+.2f}")
total_pnl_ticks = res["pnl_ticks"].sum()
print(f"Total PnL (ticks)         : {total_pnl_ticks:+.0f}")
print(f"Total PnL ($, EPS={EPS})  : {total_pnl_ticks * EPS:+,.2f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(res.index, res["pnl_ticks"].cumsum(), color="steelblue", lw=1.5)
axes[0].axhline(0, color="k", lw=0.5, alpha=0.5)
axes[0].set_title(f"Cumulative PnL (ticks) — {INSTRUMENT}  tau={TAU} min  "
                  f"min_fill={MIN_FILL_PROB}  ({SIGNAL_MODE})")
axes[0].set_ylabel("Cumulative ticks")
axes[0].grid(alpha=0.3)

if len(filled_rows):
    axes[1].hist(filled_rows["pnl_ticks"], bins=60, color="seagreen",
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].axvline(mean_pnl_fill, color="red", lw=1.5, label=f"Mean = {mean_pnl_fill:+.2f}")
    axes[1].set_title("Distribution of PnL on filled trades (ticks)")
    axes[1].set_xlabel("PnL (ticks)"); axes[1].set_ylabel("Count")
    axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ── Part 2 — AIAgent trade-trace diagnostics ──────────────────────────────────

AIAGENT_PATH = DATA_ROOT / INSTRUMENT / AIAGENT_FILENAME[INSTRUMENT]
agent = pd.read_csv(AIAGENT_PATH, header=None,
                    names=["date_serial", "hour", "minute", "price", "net_pos"])
agent["date"] = pd.to_datetime(agent["date_serial"] - 2, unit="D",
                                origin="1900-01-01")
agent["timestamp"] = (agent["date"]
                      + pd.to_timedelta(agent["hour"],   unit="h")
                      + pd.to_timedelta(agent["minute"], unit="m"))
agent = agent.set_index("timestamp").sort_index()

agent["trade_size"] = agent["net_pos"].diff()
agent["is_trade"]   = agent["trade_size"].fillna(0) != 0
agent["side"]       = np.where(agent["trade_size"] > 0, "BUY",
                       np.where(agent["trade_size"] < 0, "SELL", "FLAT"))

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

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(agent.index, agent["price"], color="steelblue", lw=0.6)
axes[0].set_ylabel("Price")
axes[0].set_title(f"AIAgent {INSTRUMENT}  —  price (top) and net position (bottom)")
axes[0].grid(alpha=0.3)

axes[1].plot(agent.index, agent["net_pos"], color="darkorange", lw=0.8)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].fill_between(agent.index, 0, agent["net_pos"],
                      where=agent["net_pos"] > 0, alpha=0.2, color="green", label="long")
axes[1].fill_between(agent.index, 0, agent["net_pos"],
                      where=agent["net_pos"] < 0, alpha=0.2, color="red",   label="short")
axes[1].set_ylabel("Net position (contracts)")
axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

agent["cash_flow"] = -agent["trade_size"].fillna(0) * agent["price"]
final_price = agent["price"].iloc[-1]
realized_pnl = agent["cash_flow"].sum() + final_pos * final_price
print(f"\nAgent realized PnL (mark-to-market at final price): ${realized_pnl:+,.2f}")
print(f"  (assuming trades happen at the snapshot price)")


# ── Part 2 — AIAgent Out-of-Sample Calibration ───────────────────────────────

AIAGENT_PATH = DATA_ROOT / INSTRUMENT / AIAGENT_FILENAME[INSTRUMENT]

agent = pd.read_csv(AIAGENT_PATH, header=None,
                    names=["date_serial", "hour", "minute", "price", "col5"])
agent["date"] = pd.to_datetime(agent["date_serial"] - 2, unit="D",
                                origin="1900-01-01")
agent["timestamp"] = (agent["date"]
                      + pd.to_timedelta(agent["hour"],   unit="h")
                      + pd.to_timedelta(agent["minute"], unit="m"))
agent = agent.set_index("timestamp").sort_index()

print(f"AIAgent rows : {len(agent):,}")
print(f"Date range   : {agent.index.min().date()} → {agent.index.max().date()}")
print(f"Unique dates : {agent.index.normalize().nunique()}")

assert TAU >= 5, "AIAgent data is 5-min snapshots; need TAU >= 5"

agent_tau = agent["price"].resample(f"{TAU}min", label="left", closed="left").agg(
    open  = "first",
    high  = "max",
    low   = "min",
    close = "last",
).dropna(subset=["open"])
agent_tau["volume"] = 1

print(f"\nAIAgent tau={TAU} min bars after resample: {len(agent_tau):,}")

_rth_start, _rth_end, _ = MARKETS[INSTRUMENT]["rth"]
agent_tau = agent_tau.between_time(_rth_start, _rth_end).copy()
print(f"After RTH filter                       : {len(agent_tau):,}")

agent_tau = compute_ranges(agent_tau, EPS)

ewma_vol_a, _          = ewma_ewmv(agent_tau["volume"].values, LAM)
ewma_rng_a, ewmv_rng_a = ewma_ewmv(agent_tau["R"].values,      LAM)
agent_tau["ewma_vol"]  = ewma_vol_a
agent_tau["ewmv_rng"]  = ewmv_rng_a
agent_tau["delta_x"]   = agent_tau["open"].diff()


def frozen_thresholds(series: pd.Series, n_states: int) -> list:
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
    for i, t in enumerate(thresholds):
        if x < t:
            return i
    return len(thresholds)

agent_tau["state_vol"] = agent_tau["ewma_vol"].shift(1).map(
    lambda x: bin_with_thresholds(x, vol_thr) if pd.notna(x) else 0).astype(int)
agent_tau["state_sig"] = agent_tau["ewmv_rng"].shift(1).map(
    lambda x: bin_with_thresholds(x, sig_thr) if pd.notna(x) else 0).astype(int)
agent_tau["state_dir"] = agent_tau["delta_x"].shift(1).map(
    lambda x: bin_with_thresholds(x, dir_thr) if pd.notna(x) else 0).astype(int)

print(f"\nState distribution on AIAgent:")
print(agent_tau[["state_vol", "state_sig", "state_dir"]]
      .value_counts().sort_index().head(27))

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
        row[f"pred_rup_{ell}"]    = epdf_Rup.fill_prob(sv, ss, sd, ell)
        row[f"pred_rdn_{ell}"]    = epdf_Rdn.fill_prob(sv, ss, sd, ell)
        row[f"actual_rup_{ell}"]  = int(agent_tau["R_up"].iloc[j] >= ell)
        row[f"actual_rdn_{ell}"]  = int(agent_tau["R_dn"].iloc[j] >= ell)
    records.append(row)

calib = pd.DataFrame(records).set_index("timestamp")
print(f"\nAIAgent evaluation rows: {len(calib):,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_bins = 10
for side, color, label, ax in [
    ("rup", "darkorange", "Sell limit (R_up >= l)", axes[0]),
    ("rdn", "steelblue",  "Buy limit  (R_dn >= l)", axes[1]),
]:
    preds, acts = [], []
    for ell in ELL_RANGE:
        preds.append(calib[f"pred_{side}_{ell}"].values)
        acts.append(calib[f"actual_{side}_{ell}"].values)
    preds = np.concatenate(preds)
    acts  = np.concatenate(acts)

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
               color=color, alpha=0.8, edgecolor="k", linewidth=0.5, label=label)
    ax.set_xlabel("Mean predicted fill probability")
    ax.set_ylabel("Realised fill rate (AIAgent)")
    ax.set_title(label)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3); ax.legend(loc="lower right")

plt.suptitle(f"AIAgent OOS Calibration — {INSTRUMENT}  tau={TAU} min", y=1.02)
plt.tight_layout()
plt.show()

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


# ── Sweep over tau and MIN_FILL_PROB (optional — uncomment to run) ────────────

def run_pipeline(instrument: str, tau: int, min_fill: float,
                 half_life: int = None) -> dict:
    if half_life is None:
        half_life = config.HALF_LIFE
    import config as _cfg
    _cfg.LAM = 2 ** (-1 / half_life)

    eps = config.MARKETS[instrument]["tick"]
    d1  = load_instrument(instrument, min_bar_frac=config.MIN_BAR_FRAC)
    dt  = resample_ohlcv(d1, tau)
    dt  = compute_ranges(dt, eps)
    dt  = add_states(dt)

    _, epdf_up, epdf_dn, bt_sw = build_rolling_epdfs(dt)
    ewma_ret_s, _ = ewma_ewmv(dt["ret"].values, config.LAM)
    bt_sw["signal"] = pd.Series(ewma_ret_s, index=dt.index).reindex(bt_sw.index).apply(
        lambda x: 1 if x >= 0 else -1)

    pnl_list = []
    for _, row in bt_sw.iterrows():
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


# sweep_rows = []
# for tau in [5, 10, 15, 30, 60]:
#     for mfp in [0.50, 0.60, 0.70]:
#         row = run_pipeline(INSTRUMENT, tau, mfp)
#         sweep_rows.append(row)
#         print(row)
# sweep_df = pd.DataFrame(sweep_rows)
# print(sweep_df.sort_values("sharpe", ascending=False))

print("Sweep cell ready. Uncomment the loop above to run.")
