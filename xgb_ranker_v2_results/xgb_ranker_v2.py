"""
xgb_ranker_v2.py  — improved version
=====================================
Changes from v1:
  1. ICIR-based CV scoring (mean_ic / std_ic) instead of raw mean IC
  2. Correlation-based redundancy filter (drops features with |corr| > CORR_THRESHOLD)
  3. Regime filter — scales position size based on SPY rolling return
  4. Wider portfolio (TOP_N=50) with rank-score weighting option
  5. Sharpe-based CV scoring option (toggle USE_SHARPE_CV)

Place in same folder as:
  sp500_alpha_target_panel.parquet
  alpha_rank_ic_summary_5d.csv
  spy_market_return.csv   ← needed for regime filter (optional)

Run:
  pip install xgboost pandas numpy matplotlib scipy pyarrow
  python xgb_ranker_v2.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════

PANEL_PATH      = "/Users/yiweipwang/Documents/Icarus Fund/ranker/sp500_alpha_target_panel.parquet"
IC_CSV_PATH     = "alpha_rank_ic_summary_5d.csv"
SPY_PATH        = "spy_market_return.csv"        # set to None to disable regime filter
OUTPUT_DIR      = "xgb_ranker_v2_results"

TARGET_COL      = "ret_fwd_5d"
DATE_COL        = "date"
TICKER_COL      = "ticker"
FEATURE_SUFFIX  = "_z"
T_STAT_MIN      = 2.0

# CV scoring method: "icir" (recommended), "ic", or "sharpe"
CV_SCORE_METHOD = "icir"

# Feature redundancy filter — drop one of any pair with |corr| > this
CORR_THRESHOLD  = 0.85

# Walk-forward CV
TRAIN_MONTHS    = 12
VAL_MONTHS      = 3

# Portfolio
TOP_N           = 50          # long top N, short bottom N (wider = less concentration risk)
WEIGHT_BY_RANK  = True        # True = score-weighted, False = equal weight
TRANS_COST      = 0.001       # per side on rebalance days
REBAL_FREQ      = "W"

# Regime filter — if SPY n-day return < threshold, scale positions by REGIME_SCALE
REGIME_LOOKBACK = 20          # days to measure market trend
REGIME_THRESHOLD = -0.03      # if SPY 20-day ret < -3%, reduce size
REGIME_SCALE    = 0.5         # scale positions to 50% in bad regime

# Tighter grid — focus on what worked
PARAM_GRID = {
    "n_estimators":     [200, 300, 500],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.03, 0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
    "reg_lambda": [5.0, 10.0, 20.0],  # push lambda higher than before
}
# ══════════════════════════════════════════════


# ── 1. LOAD & FEATURE SELECTION ────────────────

def load_panel():
    print("=" * 60)
    print("LOADING PANEL")
    print("=" * 60)
    path = PANEL_PATH if os.path.exists(PANEL_PATH) else PANEL_PATH.replace(".parquet",".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Panel not found: {PANEL_PATH}")
    print(f"  Reading {path} ...")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"  Rows: {len(df):,}  |  Dates: {df[DATE_COL].nunique()}  |  Tickers: {df[TICKER_COL].nunique()}")
    print(f"  Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")

    # Feature selection from IC summary
    if os.path.exists(IC_CSV_PATH):
        ic = pd.read_csv(IC_CSV_PATH)
        passing = ic[ic["t_stat"].abs() >= T_STAT_MIN]["alpha"].tolist()
        feature_cols = [c for c in passing if c in df.columns]
        print(f"  Features passing |t|≥{T_STAT_MIN}: {len(feature_cols)}")
    else:
        feature_cols = [c for c in df.columns if c.endswith(FEATURE_SUFFIX)
                        and c not in (TARGET_COL, DATE_COL, TICKER_COL)]

    df = df.dropna(subset=[TARGET_COL]).dropna(subset=feature_cols, how="all")
    return df, feature_cols

EXCLUDE_FEATURES = [
    "alpha_55_average_true_range_z",   # overfitting — near-zero IC, noisy decile pattern
    "alpha_68_adv_trend_z",           # weak, non-monotone decile returns
]



def remove_correlated_features(df, feature_cols):
    """Drop one of any pair with cross-sectional correlation > CORR_THRESHOLD."""
    print(f"\n  Running redundancy filter (|corr| > {CORR_THRESHOLD}) ...")
    # Compute cross-sectional mean correlation across all dates
    sample_dates = pd.Series(df[DATE_COL].unique()).sample(min(100, df[DATE_COL].nunique()), random_state=42)
    sample = df[df[DATE_COL].isin(sample_dates)][feature_cols].fillna(0)
    corr = sample.corr().abs()

    to_drop = set()
    cols = list(feature_cols)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i+1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.loc[cols[i], cols[j]] > CORR_THRESHOLD:
                to_drop.add(cols[j])   # keep i (higher up = higher ICIR from IC summary)

    kept = [c for c in feature_cols if c not in to_drop]
    print(f"  Dropped {len(to_drop)} correlated features: {[c.replace('_z','') for c in to_drop]}")
    print(f"  Remaining features: {len(kept)}")
    return kept


# ── 2. REGIME FILTER ──────────────────────────

def load_spy():
    if SPY_PATH is None or not os.path.exists(SPY_PATH):
        return None
    spy = pd.read_csv(SPY_PATH, parse_dates=["date"])
    spy = spy.sort_values("date").set_index("date")
    spy["cum_ret"] = (1 + spy["market_return"]).cumprod()
    spy["roll_ret"] = spy["cum_ret"] / spy["cum_ret"].shift(REGIME_LOOKBACK) - 1
    print(f"  SPY loaded: {spy.index.min().date()} → {spy.index.max().date()}")
    return spy["roll_ret"]


def regime_scale(date, spy_series):
    if spy_series is None:
        return 1.0
    try:
        r = spy_series.asof(pd.Timestamp(date))
        return REGIME_SCALE if (not np.isnan(r) and r < REGIME_THRESHOLD) else 1.0
    except:
        return 1.0


# ── 3. CV SCORING ──────────────────────────────

def rank_ic_score(y_true, y_pred):
    if len(y_true) < 5:
        return 0.0
    r, _ = spearmanr(y_pred, y_true)
    return float(r) if not np.isnan(r) else 0.0


def cv_score_from_ics(ic_values):
    """Convert list of daily ICs to a single CV score based on CV_SCORE_METHOD."""
    ic_arr = np.array(ic_values)
    if len(ic_arr) == 0:
        return -np.inf
    mean_ic = np.mean(ic_arr)
    std_ic  = np.std(ic_arr) if len(ic_arr) > 1 else 1e-6
    if CV_SCORE_METHOD == "icir":
        return mean_ic / (std_ic + 1e-8)         # penalizes inconsistency
    elif CV_SCORE_METHOD == "sharpe":
        # treat daily IC as a "return" and compute Sharpe
        return mean_ic / (std_ic + 1e-8) * np.sqrt(252)
    else:  # raw ic
        return mean_ic


def build_folds(df):
    months = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="MS")
    folds  = []
    for i in range(0, len(months) - TRAIN_MONTHS - VAL_MONTHS + 1, VAL_MONTHS):
        tr = df[(df[DATE_COL] >= months[i]) & (df[DATE_COL] < months[i + TRAIN_MONTHS])]
        va = df[(df[DATE_COL] >= months[i + TRAIN_MONTHS]) &
                (df[DATE_COL] <  months[min(i+TRAIN_MONTHS+VAL_MONTHS, len(months)-1)])]
        if len(tr) > 200 and len(va) > 50:
            folds.append((tr, va))
    return folds


def score_params(params, folds, feature_cols):
    import xgboost as xgb
    xgb_params = {
        "objective":"rank:pairwise", "eval_metric":"ndcg",
        "eta":params["learning_rate"], "max_depth":params["max_depth"],
        "subsample":params["subsample"], "colsample_bytree":params["colsample_bytree"],
        "lambda":params["reg_lambda"], "seed":42, "nthread":-1, "verbosity":0,
    }
    fold_scores = []
    for tr, va in folds:
        g_tr = tr.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)
        g_va = va.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)
        dtrain = xgb.DMatrix(tr[feature_cols].fillna(0).values,
                             label=tr[TARGET_COL].values, feature_names=feature_cols)
        dtrain.set_group(g_tr)
        dval = xgb.DMatrix(va[feature_cols].fillna(0).values, feature_names=feature_cols)
        dval.set_group(g_va)
        model = xgb.train(xgb_params, dtrain,
                          num_boost_round=params["n_estimators"], verbose_eval=False)
        preds  = model.predict(dval)
        cursor, ics = 0, []
        for g in g_va:
            ics.append(rank_ic_score(va[TARGET_COL].values[cursor:cursor+g],
                                     preds[cursor:cursor+g]))
            cursor += g
        fold_scores.append(cv_score_from_ics(ics))
    return np.mean(fold_scores), np.std(fold_scores)


def grid_search(df, feature_cols):
    print("\n" + "=" * 60)
    print(f"WALK-FORWARD GRID SEARCH  (scoring: {CV_SCORE_METHOD.upper()})")
    print("=" * 60)
    folds  = build_folds(df)
    print(f"  Folds: {len(folds)}")
    keys   = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f"  Combos: {len(combos)} × {len(folds)} folds = {len(combos)*len(folds)} fits")

    records, best_score, best_params = [], -np.inf, None
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        mean_s, std_s = score_params(params, folds, feature_cols)
        records.append({**params, f"mean_{CV_SCORE_METHOD}": mean_s, "std_score": std_s})
        if mean_s > best_score:
            best_score, best_params = mean_s, params.copy()
        step = max(1, len(combos) // 8)
        if (i+1) % step == 0 or i == len(combos)-1:
            print(f"  [{i+1:3d}/{len(combos)}]  best={best_score:.4f}  {best_params}")

    gs_df = pd.DataFrame(records).sort_values(f"mean_{CV_SCORE_METHOD}", ascending=False).reset_index(drop=True)
    print(f"\n  ✓ Best {CV_SCORE_METHOD.upper()} : {best_score:.4f}")
    print(f"  ✓ Best params       : {best_params}")
    return best_params, gs_df


# ── 4. FINAL MODEL ─────────────────────────────

def train_final(df, feature_cols, best_params):
    import xgboost as xgb
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    dates  = sorted(df[DATE_COL].unique())
    split  = dates[int(len(dates) * 0.8)]
    train  = df[df[DATE_COL] <  split]
    test   = df[df[DATE_COL] >= split]
    print(f"  Train: {train[DATE_COL].min().date()} → {train[DATE_COL].max().date()}  ({len(train):,} rows)")
    print(f"  Test : {test[DATE_COL].min().date()}  → {test[DATE_COL].max().date()}   ({len(test):,} rows)")

    g_tr = train.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)
    dtrain = xgb.DMatrix(train[feature_cols].fillna(0).values,
                         label=train[TARGET_COL].values, feature_names=feature_cols)
    dtrain.set_group(g_tr)
    xgb_params = {
        "objective":"rank:pairwise","eval_metric":"ndcg",
        "eta":best_params["learning_rate"],"max_depth":best_params["max_depth"],
        "subsample":best_params["subsample"],"colsample_bytree":best_params["colsample_bytree"],
        "lambda":best_params["reg_lambda"],"seed":42,"nthread":-1,"verbosity":0,
    }
    model = xgb.train(xgb_params, dtrain,
                      num_boost_round=best_params["n_estimators"], verbose_eval=False)
    test  = test.copy()
    test["xgb_score"] = model.predict(
        xgb.DMatrix(test[feature_cols].fillna(0).values, feature_names=feature_cols))

    fi = model.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature":list(fi.keys()),"gain":list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)
    fi_df["short_name"] = fi_df["feature"].apply(
        lambda c: "_".join(c.replace("_z","").split("_")[2:]))
    return model, test, fi_df


# ── 5. SIMULATION ──────────────────────────────

def simulate(test_df, spy_series=None):
    print("\n" + "=" * 60)
    print("PORTFOLIO SIMULATION")
    print("=" * 60)
    df = test_df.copy().sort_values(DATE_COL)
    all_dates = sorted(df[DATE_COL].unique())
    rebal_set = set(pd.Series(all_dates, index=all_dates)
                    .resample(REBAL_FREQ).last().dropna().values)

    records, longs, shorts, long_scores, short_scores = [], [], [], [], []
    for date in all_dates:
        day = df[df[DATE_COL] == date].dropna(subset=["xgb_score", TARGET_COL])
        if date in rebal_set and len(day) >= TOP_N * 2:
            ranked      = day.sort_values("xgb_score", ascending=False).reset_index(drop=True)
            longs       = ranked.head(TOP_N)[TICKER_COL].tolist()
            shorts      = ranked.tail(TOP_N)[TICKER_COL].tolist()
            long_scores = ranked.head(TOP_N)["xgb_score"].values
            short_scores= ranked.tail(TOP_N)["xgb_score"].values
        if not longs:
            continue

        ret_map = day.set_index(TICKER_COL)[TARGET_COL].to_dict()
        scale   = regime_scale(date, spy_series)

        if WEIGHT_BY_RANK and len(long_scores) > 0:
            lw = np.abs(long_scores); lw = lw / lw.sum()
            sw = np.abs(short_scores); sw = sw / sw.sum()
            lr = sum(lw[i] * ret_map[t] for i, t in enumerate(longs)  if t in ret_map)
            sr = sum(sw[i] * ret_map[t] for i, t in enumerate(shorts) if t in ret_map)
        else:
            lr = np.mean([ret_map[t] for t in longs  if t in ret_map])
            sr = np.mean([ret_map[t] for t in shorts if t in ret_map])

        tc = 2 * TRANS_COST if date in rebal_set else 0.0
        ls = (lr - sr - tc) * scale
        records.append({"date":date,"long_ret":lr*scale,"short_ret":sr,
                          "ls_ret":ls,"regime_scale":scale})

    sim = pd.DataFrame(records).set_index("date")
    sim.index = pd.to_datetime(sim.index)
    sim["cum_long"] = (1 + sim["long_ret"]).cumprod()
    sim["cum_ls"]   = (1 + sim["ls_ret"]).cumprod()

    def metrics(r):
        r = r.dropna(); n = len(r)
        ar  = (1+r).prod()**(252/n)-1
        vol = r.std()*np.sqrt(252)
        sh  = ar/vol if vol>0 else np.nan
        cum = (1+r).cumprod()
        mdd = (cum/cum.cummax()-1).min()
        return ar, vol, sh, mdd, (r>0).mean()

    print(f"\n  {'Metric':<20} {'Long Top-'+str(TOP_N):>14} {'Long-Short':>12}")
    print(f"  {'-'*48}")
    for label, col in [(f'Long Top-{TOP_N}','long_ret'),('Long-Short','ls_ret')]:
        ar, vol, sh, mdd, hit = metrics(sim[col])
        print(f"  {'Annual Return':<20} {ar:>14.2%}" if label==f'Long Top-{TOP_N}'
              else f"  {'Annual Return':<20} {'':>14} {ar:>12.2%}")
    for k, fmt in [("Annual Vol",":.2%"),("Sharpe",":.3f"),("Max Drawdown",":.2%"),("Hit Rate",":.2%")]:
        vals = [metrics(sim[c]) for c in ['long_ret','ls_ret']]
        idx  = {"Annual Vol":1,"Sharpe":2,"Max Drawdown":3,"Hit Rate":4}[k]
        print(f"  {k:<20} {format(vals[0][idx], fmt[1:]):>14} {format(vals[1][idx], fmt[1:]):>12}")

    regime_days = (sim["regime_scale"] < 1.0).sum()
    if regime_days > 0:
        print(f"\n  Regime filter triggered: {regime_days} days ({regime_days/len(sim):.1%} of period)")
    return sim


def rank_ic_series(test_df):
    rows = []
    for date, grp in test_df.groupby(DATE_COL):
        g = grp.dropna(subset=["xgb_score", TARGET_COL])
        if len(g) < 5: continue
        r, _ = spearmanr(g["xgb_score"].values, g[TARGET_COL].values)
        rows.append({"date": date, "rank_ic": float(r) if not np.isnan(r) else 0.0})
    ic = pd.DataFrame(rows).set_index("date")
    ic.index = pd.to_datetime(ic.index)
    ic["rolling20"] = ic["rank_ic"].rolling(20).mean()
    icir = ic["rank_ic"].mean() / (ic["rank_ic"].std() + 1e-8)
    print(f"\n  OOS mean Rank IC : {ic['rank_ic'].mean():.4f}")
    print(f"  OOS ICIR         : {icir:.3f}")
    print(f"  OOS % positive   : {(ic['rank_ic']>0).mean():.1%}")
    return ic


# ── 6. PLOTS ───────────────────────────────────

BLUE,GREEN,RED,ORANGE,GREY = "#2563EB","#16A34A","#DC2626","#F59E0B","#6B7280"

def save_plot(fig, name, out):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")

def plot_returns(sim, out):
    fig, axes = plt.subplots(2,1,figsize=(14,10),sharex=True,gridspec_kw={"hspace":0.08})
    ax = axes[0]
    ax.plot(sim.index, sim["cum_long"], color=BLUE,  lw=2, label=f"Long Top-{TOP_N}")
    ax.plot(sim.index, sim["cum_ls"],   color=GREEN, lw=2, label="Long-Short (regime-adjusted)")
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_ylabel("Cumulative Return"); ax.legend(); ax.grid(axis="y",alpha=0.3)
    ax.set_title("XGBoost Ranker v2 — Out-of-Sample Performance", fontweight="bold")
    ax2 = axes[1]
    for col,label,color in [("long_ret",f"Long Top-{TOP_N}",BLUE),("ls_ret","Long-Short",GREEN)]:
        cum = (1+sim[col]).cumprod()
        dd  = (cum/cum.cummax()-1)*100
        ax2.fill_between(sim.index,dd,0,alpha=0.25,color=color)
        ax2.plot(sim.index,dd,color=color,lw=1.5,label=label)
    # Mark regime-filtered days
    if "regime_scale" in sim.columns:
        regime_dates = sim[sim["regime_scale"]<1.0].index
        if len(regime_dates):
            ax2.axvspan(regime_dates.min(), regime_dates.max(), alpha=0.1, color="orange",
                        label="Regime filter active")
    ax2.set_ylabel("Drawdown (%)"); ax2.legend(); ax2.grid(axis="y",alpha=0.3)
    ax2.set_title("Underwater Equity Curve")
    save_plot(fig, "cumulative_returns.png", out)

def plot_feature_importance(fi_df, out):
    top = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(10, max(4,len(top)*0.5)))
    ax.barh(top["short_name"][::-1], top["gain"][::-1], color=BLUE, alpha=0.85)
    ax.set_xlabel("Gain"); ax.set_title("Feature Importance — Top 15 (Gain)", fontweight="bold")
    ax.grid(axis="x",alpha=0.3)
    save_plot(fig, "feature_importance.png", out)

def plot_rank_ic(ic_df, out):
    fig, ax = plt.subplots(figsize=(14,5))
    colors = np.where(ic_df["rank_ic"]>=0, GREEN, RED)
    ax.bar(ic_df.index, ic_df["rank_ic"], color=colors, alpha=0.5, width=2)
    ax.plot(ic_df.index, ic_df["rolling20"], color=BLUE, lw=2, label="20-day Rolling IC")
    m = ic_df["rank_ic"].mean()
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(m, color=ORANGE, lw=1.5, ls="--", label=f"Mean IC={m:.4f}")
    ax.set_title("Out-of-Sample Rank IC Over Time", fontweight="bold")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    save_plot(fig, "rank_ic_over_time.png", out)

def plot_annual(sim, out):
    annual = sim["ls_ret"].resample("YE").apply(lambda r: (1+r).prod()-1)
    fig, ax = plt.subplots(figsize=(12,5))
    colors = [GREEN if v>=0 else RED for v in annual.values]
    ax.bar([str(d.year) for d in annual.index], annual.values*100, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    for i,v in enumerate(annual.values):
        ax.text(i, v+(0.5 if v>=0 else -1.5), f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Annual Return (%)"); ax.set_title("Long-Short Annual Returns", fontweight="bold")
    ax.grid(axis="y",alpha=0.3)
    save_plot(fig, "annual_returns.png", out)


# ── MAIN ───────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, feature_cols = load_panel()

    # Redundancy filter
    feature_cols = remove_correlated_features(df, feature_cols)
    feature_cols = [f for f in feature_cols if f not in EXCLUDE_FEATURES]
    print(f"  After exclusions: {len(feature_cols)} features")

    # SPY regime data
    spy_series = load_spy()

    # Grid search with ICIR scoring
    best_params, gs_df = grid_search(df, feature_cols)
    gs_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Final model
    model, test_df, fi_df = train_final(df, feature_cols, best_params)
    model.save_model(os.path.join(OUTPUT_DIR, "model.ubj"))
    fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    # Simulation
    sim = simulate(test_df, spy_series)
    sim.to_csv(os.path.join(OUTPUT_DIR, "simulation_returns.csv"))

    # Rank IC
    ic_df = rank_ic_series(test_df)
    ic_df.to_csv(os.path.join(OUTPUT_DIR, "rank_ic_series.csv"))

    # Plots
    print("\n" + "=" * 60)
    print("SAVING PLOTS")
    print("=" * 60)
    plot_returns(sim, OUTPUT_DIR)
    plot_feature_importance(fi_df, OUTPUT_DIR)
    plot_rank_ic(ic_df, OUTPUT_DIR)
    plot_annual(sim, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DONE —", os.path.abspath(OUTPUT_DIR))
    print("=" * 60)

if __name__ == "__main__":
    main()