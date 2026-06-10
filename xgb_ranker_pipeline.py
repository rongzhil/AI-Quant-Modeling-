"""
xgb_ranker_pipeline.py
=======================
XGBoost Ranking Model + Walk-Forward Grid Search + Return Simulation
Quantitative Multi-Factor Model — S&P 500 Cross-Sectional Alpha

INPUT
-----
  alpha_panel.parquet  — output of build_cross_sectional_alpha_panel.py
      Required columns:
        date              : YYYY-MM-DD string
        ticker            : str
        ret_fwd_5d        : float  (5-day forward return label)
        alpha_XX_name_z_cs_rank   : cross-sectional rank [0,1]  ← primary features
        (also available: _cs_zscore, _sector_neutral_rank, _sector_neutral_value)
      OR:
  alpha_panel.csv      — same schema as CSV fallback

OUTPUTS  (written to ./xgb_ranker_results/)
-------
  model.ubj                    XGBoost model
  gridsearch_results.csv       all CV fold scores
  best_params.json             best hyperparameters
  feature_importance.csv       gain-based importance
  simulation_returns.csv       daily portfolio return series
  plots/
    cumulative_returns.png
    feature_importance.png
    rank_ic_over_time.png
    annual_returns.png
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
#  CONFIGURATION  ← edit paths / settings here
# ══════════════════════════════════════════════

PANEL_PATH   = "alpha_panel.parquet"   # or "alpha_panel.csv"
IC_CSV_PATH  = "alpha_rank_ic_summary_5d.csv"
OUTPUT_DIR   = "xgb_ranker_results"

TARGET_COL   = "ret_fwd_5d"
DATE_COL     = "date"
TICKER_COL   = "ticker"

# Which column suffix to use as features from the panel
# Options: "_cs_rank"  |  "_cs_zscore"  |  "_sector_neutral_rank"
FEATURE_SUFFIX = "_cs_rank"

# Feature selection threshold: keep alphas where |t_stat| >= this
T_STAT_THRESHOLD = 2.0

# Walk-forward CV
TRAIN_MONTHS  = 18
VAL_MONTHS    = 3

# Portfolio simulation
TOP_N         = 20      # long top N, short bottom N
TRANS_COST    = 0.001   # per side, applied on rebalance days
REBAL_FREQ    = "W"     # "W"=weekly, "ME"=month-end, "D"=daily

# Grid search space
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
    "reg_lambda":       [1.0, 5.0],
}
# ══════════════════════════════════════════════


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def select_features(ic_csv, panel_cols):
    """Pick feature columns from panel based on IC summary t-stat filter."""
    ic = pd.read_csv(ic_csv)
    passing = ic[(ic["t_stat"].abs() >= T_STAT_THRESHOLD) & ic["t_stat"].notna()]["alpha"].tolist()
    # Each alpha in IC summary is e.g. "alpha_32_12_1_month_momentum_z"
    # Panel column is  "alpha_32_12_1_month_momentum_z" + FEATURE_SUFFIX
    feature_cols = [f"{a}{FEATURE_SUFFIX}" for a in passing if f"{a}{FEATURE_SUFFIX}" in panel_cols]
    print(f"  Features from IC filter (|t|≥{T_STAT_THRESHOLD}): {len(feature_cols)}")
    if not feature_cols:
        # fallback: all _cs_rank columns
        feature_cols = [c for c in panel_cols if c.endswith(FEATURE_SUFFIX)]
        print(f"  Fallback: all {FEATURE_SUFFIX} columns → {len(feature_cols)}")
    return feature_cols


def load_panel():
    print("=" * 60)
    print("LOADING PANEL")
    print("=" * 60)
    if os.path.exists(PANEL_PATH):
        path = PANEL_PATH
    elif os.path.exists(PANEL_PATH.replace(".parquet", ".csv")):
        path = PANEL_PATH.replace(".parquet", ".csv")
    else:
        raise FileNotFoundError(
            f"Panel not found at '{PANEL_PATH}' or CSV equivalent.\n"
            "Run build_cross_sectional_alpha_panel.py first, then copy the output here."
        )

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"  Rows      : {len(df):,}")
    print(f"  Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    print(f"  Tickers   : {df[TICKER_COL].nunique()}")

    feature_cols = select_features(IC_CSV_PATH, df.columns.tolist())

    # Drop rows with missing target or all features NaN
    df = df.dropna(subset=[TARGET_COL], how="any")
    df = df.dropna(subset=feature_cols, how="all")
    print(f"  Rows after dropna: {len(df):,}")
    return df, feature_cols


# ──────────────────────────────────────────────
# 2. WALK-FORWARD GRID SEARCH
# ──────────────────────────────────────────────

def rank_ic(y_true, y_pred):
    if len(y_true) < 3:
        return 0.0
    r, _ = spearmanr(y_pred, y_true)
    return r if not np.isnan(r) else 0.0


def build_folds(df):
    dates = sorted(df[DATE_COL].unique())
    months = pd.date_range(dates[0], dates[-1], freq="MS")
    folds = []
    for i in range(0, len(months) - TRAIN_MONTHS - VAL_MONTHS + 1, VAL_MONTHS):
        tr_start = months[i]
        tr_end   = months[i + TRAIN_MONTHS]
        va_end   = months[min(i + TRAIN_MONTHS + VAL_MONTHS, len(months) - 1)]
        tr = df[(df[DATE_COL] >= tr_start) & (df[DATE_COL] <  tr_end)]
        va = df[(df[DATE_COL] >= tr_end)   & (df[DATE_COL] <  va_end)]
        if len(tr) > 200 and len(va) > 50:
            folds.append((tr, va))
    return folds


def score_params(params, folds, feature_cols):
    import xgboost as xgb
    xgb_params = {
        "objective":        "rank:pairwise",
        "eval_metric":      "ndcg",
        "eta":              params["learning_rate"],
        "max_depth":        params["max_depth"],
        "subsample":        params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "lambda":           params["reg_lambda"],
        "seed":             42,
        "nthread":          -1,
        "verbosity":        0,
    }
    fold_ics = []
    for tr, va in folds:
        X_tr = tr[feature_cols].fillna(0).values
        y_tr = tr[TARGET_COL].values
        g_tr = tr.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)

        X_va = va[feature_cols].fillna(0).values
        g_va = va.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dtrain.set_group(g_tr)
        dval   = xgb.DMatrix(X_va, feature_names=feature_cols)
        dval.set_group(g_va)

        model = xgb.train(xgb_params, dtrain,
                          num_boost_round=params["n_estimators"],
                          verbose_eval=False)
        preds = model.predict(dval)

        # Rank IC per date, then average
        date_arr = va[DATE_COL].values
        unique_d = np.unique(date_arr)
        ics = []
        cursor = 0
        for g in g_va:
            mask = np.zeros(len(va), dtype=bool)
            # use group sizes directly
            ics.append(rank_ic(
                va[TARGET_COL].values[cursor:cursor+g],
                preds[cursor:cursor+g]
            ))
            cursor += g
        fold_ics.append(np.mean(ics))
    return np.mean(fold_ics), np.std(fold_ics)


def grid_search(df, feature_cols):
    print("\n" + "=" * 60)
    print("WALK-FORWARD GRID SEARCH")
    print("=" * 60)
    folds = build_folds(df)
    print(f"  Folds built: {len(folds)}")
    if len(folds) < 2:
        raise ValueError("Too few folds. Check date range / TRAIN_MONTHS / VAL_MONTHS.")

    keys   = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f"  Param combos: {len(combos)}  ×  {len(folds)} folds = {len(combos)*len(folds)} fits")

    records, best_score, best_params = [], -np.inf, None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        mean_ic, std_ic = score_params(params, folds, feature_cols)
        records.append({**params, "mean_rank_ic": mean_ic, "std_rank_ic": std_ic})

        if mean_ic > best_score:
            best_score, best_params = mean_ic, params.copy()

        step = max(1, len(combos) // 8)
        if (i + 1) % step == 0 or i == len(combos) - 1:
            print(f"  [{i+1:3d}/{len(combos)}]  best IC={best_score:.4f}  {best_params}")

    gs_df = pd.DataFrame(records).sort_values("mean_rank_ic", ascending=False).reset_index(drop=True)
    print(f"\n  ✓ Best Rank IC : {best_score:.4f}")
    print(f"  ✓ Best params  : {best_params}")
    return best_params, gs_df


# ──────────────────────────────────────────────
# 3. FINAL MODEL TRAINING
# ──────────────────────────────────────────────

def train_final(df, feature_cols, best_params):
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    dates      = sorted(df[DATE_COL].unique())
    split_idx  = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    train = df[df[DATE_COL] <  split_date]
    test  = df[df[DATE_COL] >= split_date]
    print(f"  Train: {train[DATE_COL].min().date()} → {train[DATE_COL].max().date()}  ({len(train):,} rows)")
    print(f"  Test : {test[DATE_COL].min().date()}  → {test[DATE_COL].max().date()}   ({len(test):,} rows)")

    X_tr = train[feature_cols].fillna(0).values
    y_tr = train[TARGET_COL].values
    g_tr = train.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dtrain.set_group(g_tr)

    xgb_params = {
        "objective":        "rank:pairwise",
        "eval_metric":      "ndcg",
        "eta":              best_params["learning_rate"],
        "max_depth":        best_params["max_depth"],
        "subsample":        best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "lambda":           best_params["reg_lambda"],
        "seed":             42,
        "nthread":          -1,
        "verbosity":        0,
    }

    model = xgb.train(xgb_params, dtrain,
                      num_boost_round=best_params["n_estimators"],
                      verbose_eval=False)

    # Score full test set
    X_te = xgb.DMatrix(test[feature_cols].fillna(0).values, feature_names=feature_cols)
    test = test.copy()
    test["xgb_score"] = model.predict(X_te)

    # Feature importance
    fi   = model.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)

    # Readable short name
    def short(col):
        parts = col.replace(FEATURE_SUFFIX, "").split("_")
        return "_".join(parts[2:]) if len(parts) > 2 else col
    fi_df["short_name"] = fi_df["feature"].apply(short)

    return model, test, fi_df


# ──────────────────────────────────────────────
# 4. RETURN SIMULATION
# ──────────────────────────────────────────────

def simulate(test_df):
    print("\n" + "=" * 60)
    print("PORTFOLIO SIMULATION")
    print("=" * 60)

    df = test_df.copy().sort_values(DATE_COL)
    all_dates   = sorted(df[DATE_COL].unique())
    rebal_set   = set(pd.Series(all_dates, index=all_dates)
                      .resample(REBAL_FREQ).last().dropna().values)

    records, longs, shorts = [], [], []

    for date in all_dates:
        day = df[df[DATE_COL] == date].dropna(subset=["xgb_score", TARGET_COL])
        if date in rebal_set and len(day) >= TOP_N * 2:
            ranked  = day.sort_values("xgb_score", ascending=False)
            longs   = ranked.head(TOP_N)[TICKER_COL].tolist()
            shorts  = ranked.tail(TOP_N)[TICKER_COL].tolist()

        if not longs:
            continue

        ret_map  = day.set_index(TICKER_COL)[TARGET_COL].to_dict()
        lr = np.mean([ret_map[t] for t in longs  if t in ret_map]) if longs  else 0.0
        sr = np.mean([ret_map[t] for t in shorts if t in ret_map]) if shorts else 0.0
        tc = 2 * TRANS_COST if date in rebal_set else 0.0
        records.append({"date": date, "long_ret": lr, "short_ret": sr,
                         "ls_ret": lr - sr - tc, "rebal": date in rebal_set})

    sim = pd.DataFrame(records).set_index("date")
    sim.index = pd.to_datetime(sim.index)
    sim["cum_long"] = (1 + sim["long_ret"]).cumprod()
    sim["cum_short"]= (1 + sim["short_ret"]).cumprod()
    sim["cum_ls"]   = (1 + sim["ls_ret"]).cumprod()

    def metrics(r):
        r = r.dropna()
        n   = len(r)
        ar  = (1 + r).prod() ** (252 / n) - 1 if n > 0 else np.nan
        vol = r.std() * np.sqrt(252)
        sh  = ar / vol if vol > 0 else np.nan
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        hit = (r > 0).mean()
        return {"Annual Return": ar, "Annual Vol": vol, "Sharpe": sh,
                "Max Drawdown": mdd, "Hit Rate": hit}

    m_long = metrics(sim["long_ret"])
    m_ls   = metrics(sim["ls_ret"])

    print(f"\n  {'Metric':<20} {'Long Top-{}'.format(TOP_N):>12} {'Long-Short':>12}")
    print(f"  {'-'*46}")
    for k in ["Annual Return", "Annual Vol", "Sharpe", "Max Drawdown", "Hit Rate"]:
        vl = f"{m_long[k]:.2%}" if k != "Sharpe" else f"{m_long[k]:.2f}"
        vs = f"{m_ls[k]:.2%}"   if k != "Sharpe" else f"{m_ls[k]:.2f}"
        print(f"  {k:<20} {vl:>12} {vs:>12}")

    return sim, m_long, m_ls


def rank_ic_series(test_df):
    rows = []
    for date, grp in test_df.groupby(DATE_COL):
        g = grp.dropna(subset=["xgb_score", TARGET_COL])
        if len(g) < 5:
            continue
        rows.append({"date": date, "rank_ic": rank_ic(g[TARGET_COL].values, g["xgb_score"].values)})
    ic = pd.DataFrame(rows).set_index("date")
    ic.index = pd.to_datetime(ic.index)
    ic["rolling20"] = ic["rank_ic"].rolling(20).mean()
    icir = ic["rank_ic"].mean() / ic["rank_ic"].std()
    print(f"\n  Out-of-sample mean Rank IC : {ic['rank_ic'].mean():.4f}")
    print(f"  Out-of-sample ICIR         : {icir:.3f}")
    return ic


# ──────────────────────────────────────────────
# 5. PLOTS
# ──────────────────────────────────────────────

BLUE    = "#2563EB"
GREEN   = "#16A34A"
RED     = "#DC2626"
ORANGE  = "#F59E0B"
GREY    = "#6B7280"
BG      = "#F8FAFC"


def plot_returns(sim, out):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)

    # Cumulative
    ax = axes[0]
    ax.plot(sim.index, sim["cum_long"], color=BLUE,  lw=2,   label=f"Long Top-{TOP_N}")
    ax.plot(sim.index, sim["cum_ls"],   color=GREEN, lw=2,   label="Long-Short")
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.set_title("XGBoost Ranker — Out-of-Sample Portfolio Performance", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # Drawdown
    ax2 = axes[1]
    for col, label, color in [("long_ret", f"Long Top-{TOP_N}", BLUE), ("ls_ret", "Long-Short", GREEN)]:
        cum = (1 + sim[col]).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        ax2.fill_between(sim.index, dd, 0, alpha=0.25, color=color)
        ax2.plot(sim.index, dd, color=color, lw=1.5, label=label)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_title("Underwater Equity Curve", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out, "cumulative_returns.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def plot_feature_importance(fi_df, out):
    top = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.55)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(top["short_name"][::-1], top["gain"][::-1], color=BLUE, alpha=0.85)
    for bar, val in zip(bars, top["gain"][::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Gain", fontsize=11)
    ax.set_title("Feature Importance — Top 15 (Gain)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, "feature_importance.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def plot_rank_ic(ic_df, out):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    colors = np.where(ic_df["rank_ic"] >= 0, GREEN, RED)
    ax.bar(ic_df.index, ic_df["rank_ic"], color=colors, alpha=0.5, width=2)
    ax.plot(ic_df.index, ic_df["rolling20"], color=BLUE, lw=2, label="20-day Rolling IC")
    mean_ic = ic_df["rank_ic"].mean()
    ax.axhline(0,       color="black", lw=0.8)
    ax.axhline(mean_ic, color=ORANGE,  lw=1.5, ls="--", label=f"Mean IC = {mean_ic:.4f}")
    ax.set_ylabel("Rank IC", fontsize=11)
    ax.set_title("Out-of-Sample Rank IC Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, "rank_ic_over_time.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def plot_annual(sim, out):
    annual = sim["ls_ret"].resample("YE").apply(lambda r: (1 + r).prod() - 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    colors = [GREEN if v >= 0 else RED for v in annual.values]
    ax.bar([str(d.year) for d in annual.index], annual.values * 100, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Annual Return (%)", fontsize=11)
    ax.set_title("Long-Short Portfolio — Annual Returns", fontsize=12, fontweight="bold")
    for i, v in enumerate(annual.values):
        ax.text(i, v + (0.5 if v >= 0 else -1.5), f"{v:.1%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, "annual_returns.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load
    df, feature_cols = load_panel()

    # 2. Grid search
    best_params, gs_df = grid_search(df, feature_cols)
    gs_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # 3. Final model
    model, test_df, fi_df = train_final(df, feature_cols, best_params)
    model.save_model(os.path.join(OUTPUT_DIR, "model.ubj"))
    fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    # 4. Simulation
    sim, m_long, m_ls = simulate(test_df)
    sim.to_csv(os.path.join(OUTPUT_DIR, "simulation_returns.csv"))

    # 5. Rank IC
    ic_df = rank_ic_series(test_df)
    ic_df.to_csv(os.path.join(OUTPUT_DIR, "rank_ic_series.csv"))

    # 6. Plots
    print("\n" + "=" * 60)
    print("SAVING PLOTS")
    print("=" * 60)
    for fn in [plot_returns, plot_feature_importance, plot_rank_ic, plot_annual]:
        try:
            args = (sim, plots_dir) if fn in (plot_returns, plot_annual) else \
                   (fi_df, plots_dir) if fn == plot_feature_importance else \
                   (ic_df, plots_dir)
            p = fn(*args)
            print(f"  ✓ {p}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")

    # 7. Summary table
    print("\n" + "=" * 60)
    print("COMPLETE — Output summary")
    print("=" * 60)
    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size  = os.path.getsize(fpath)
            print(f"  {fpath}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
