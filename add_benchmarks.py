"""
add_benchmarks.py
=================
Adds SPY and equal-weight benchmarks to the cumulative returns chart.
Reads existing simulation_returns.csv — no retraining needed.

Place in your ranker/ folder alongside:
  xgb_ranker_v2_results/simulation_returns.csv
  S&P500 Data/spy_market_return.csv
  sp500_alpha_target_panel.parquet  (for equal-weight universe return)

Run:
  python add_benchmarks.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ── edit if your paths differ ──────
SIM_PATH   = "xgb_ranker_v3_results/simulation_returns.csv"
SPY_PATH   = "S&P500 Data/spy_market_return.csv"
PANEL_PATH = "sp500_alpha_target_panel.parquet"   # for equal-weight; set to None to skip
OUT_PATH   = "xgb_ranker_v3_results/cumulative_returns_with_benchmarks.png"

TOP_N      = 50   # must match what you used in v2
# ─────────────────────────────────────────────

BLUE   = "#2563EB"
GREEN  = "#16A34A"
ORANGE = "#F59E0B"
GREY   = "#6B7280"
PURPLE = "#7C3AED"

print("Loading simulation results...")
sim = pd.read_csv(SIM_PATH, parse_dates=["date"]).set_index("date")
start = sim.index.min()
end   = sim.index.max()
print(f"  Sim period: {start.date()} → {end.date()}  ({len(sim)} days)")

# ── SPY benchmark ────────────────────────────
print("Loading SPY...")
spy = pd.read_csv(SPY_PATH, parse_dates=["date"])
spy = spy.sort_values("date").set_index("date")["market_return"]
spy = spy.reindex(sim.index)
if spy.isna().all():
    print("  WARNING: SPY data doesn't cover the test period — downloading from yfinance...")
    import yfinance as yf
    spy_raw = yf.download("SPY", start=str(start.date()), end=str(end.date()), 
                           auto_adjust=True, progress=False)
    spy = spy_raw["Close"]
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]  # flatten MultiIndex
    spy = spy.pct_change().dropna()
    spy.index = pd.to_datetime(spy.index.date)
    spy = spy.reindex(sim.index).fillna(0)
else:
    spy = spy.fillna(0)
cum_spy = (1 + spy).cumprod()
cum_spy = cum_spy / cum_spy.iloc[0]
print(f"  SPY total return over period: {(cum_spy.iloc[-1]-1):.1%}")

# ── Equal-weight universe ────────────────────
cum_ew = None
if PANEL_PATH and os.path.exists(PANEL_PATH):
    print("Loading panel for equal-weight benchmark...")
    df = pd.read_parquet(PANEL_PATH) if PANEL_PATH.endswith(".parquet") else pd.read_csv(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    ew = df[df["date"].isin(sim.index)].groupby("date")["ret_fwd_5d"].mean()
    ew = ew.reindex(sim.index).fillna(0)
    cum_ew = (1 + ew).cumprod()
    cum_ew = cum_ew / cum_ew.iloc[0]
    print(f"  Equal-weight total return: {(cum_ew.iloc[-1]-1):.1%}")
else:
    print("  Skipping equal-weight (panel not found)")

# ── Plot ─────────────────────────────────────
print("Plotting...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                          gridspec_kw={"hspace": 0.08})
fig.patch.set_facecolor("#F8FAFC")
for ax in axes: ax.set_facecolor("#F8FAFC")

# Top: cumulative returns
ax = axes[0]
ax.plot(sim.index, sim["cum_long"], color=BLUE,   lw=2,   label=f"Long Top-{TOP_N}")
ax.plot(sim.index, sim["cum_ls"],   color=GREEN,  lw=2,   label="Long-Short")
ax.plot(sim.index, cum_spy,         color=ORANGE, lw=1.8, ls="--", label="SPY (buy & hold)")
if cum_ew is not None:
    ax.plot(sim.index, cum_ew,      color=GREY,   lw=1.5, ls=":",  label="Equal-weight S&P 500")
ax.axhline(1.0, color="#D1D5DB", lw=0.8, ls="-")
ax.set_ylabel("Cumulative Return", fontsize=11)
ax.set_title("XGBoost Ranker v2 — Out-of-Sample Performance vs Benchmarks",
             fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Add final return annotations on right edge
for label, series, color in [
    (f"Long Top-{TOP_N}", sim["cum_long"], BLUE),
    ("Long-Short",        sim["cum_ls"],   GREEN),
    ("SPY",               cum_spy,         ORANGE),
]:
    val = series.iloc[-1]
    ax.annotate(f"{val-1:+.1%}", xy=(sim.index[-1], val),
                xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=9, va="center", fontweight="500")
if cum_ew is not None:
    val = cum_ew.iloc[-1]
    ax.annotate(f"{val-1:+.1%}", xy=(sim.index[-1], val),
                xytext=(6, 0), textcoords="offset points",
                color=GREY, fontsize=9, va="center")

# Bottom: drawdown
ax2 = axes[1]
for col, label, color, lw, ls in [
    ("long_ret", f"Long Top-{TOP_N}", BLUE,  1.5, "-"),
    ("ls_ret",   "Long-Short",        GREEN, 1.5, "-"),
]:
    cum = (1 + sim[col]).cumprod()
    dd  = (cum / cum.cummax() - 1) * 100
    ax2.fill_between(sim.index, dd, 0, alpha=0.2, color=color)
    ax2.plot(sim.index, dd, color=color, lw=lw, ls=ls, label=label)

# SPY drawdown
spy_cum_dd = (1 + spy).cumprod()
spy_dd = (spy_cum_dd / spy_cum_dd.cummax() - 1) * 100
ax2.plot(sim.index, spy_dd, color=ORANGE, lw=1.5, ls="--", label="SPY")

# Mark regime filter days
if "regime_scale" in sim.columns:
    regime_dates = sim[sim["regime_scale"] < 1.0].index
    if len(regime_dates):
        ax2.axvspan(regime_dates.min(), regime_dates.max(),
                    alpha=0.08, color=PURPLE, label="Regime filter active")

ax2.set_ylabel("Drawdown (%)", fontsize=11)
ax2.set_title("Underwater Equity Curve", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

# ── Performance table in chart ────────────────
def perf(r):
    r  = r.dropna()
    ar = (1+r).prod()**(252/len(r))-1
    vol= r.std()*np.sqrt(252)
    sh = ar/vol if vol>0 else np.nan
    mdd= ((1+r).cumprod()/((1+r).cumprod()).cummax()-1).min()
    return ar, vol, sh, mdd

rows = [
    (f"Long Top-{TOP_N}", *perf(sim["long_ret"])),
    ("Long-Short",        *perf(sim["ls_ret"])),
    ("SPY",               *perf(spy)),
]
if cum_ew is not None:
    ew_ret = ew.reindex(sim.index).fillna(0)
    rows.append(("Equal-weight", *perf(ew_ret)))

tbl_text = f"{'Strategy':<22} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>7} {'MaxDD':>8}\n"
tbl_text += "─" * 56 + "\n"
for name, ar, vol, sh, mdd in rows:
    tbl_text += f"{name:<22} {ar:>8.1%} {vol:>8.1%} {sh:>7.2f} {mdd:>8.1%}\n"

fig.text(0.12, -0.04, tbl_text, fontfamily="monospace", fontsize=9,
         verticalalignment="top", color="#374151")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Saved: {OUT_PATH}")