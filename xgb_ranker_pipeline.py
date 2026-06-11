"""
xgb_ranker_local.py
===================
Run this script from your Mac. Place it in your project folder alongside
sp500_alpha_target_panel.parquet (or .csv).

Install dependencies first:
    pip install xgboost scikit-learn pandas numpy matplotlib scipy pyarrow

Then run:
    python xgb_ranker_local.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed — saves PNGs directly
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  CONFIGURATION  ← edit these if needed
# ══════════════════════════════════════════════

PANEL_PATH      = "sp500_alpha_target_panel.parquet"  # or .csv
IC_CSV_PATH     = "alpha_rank_ic_summary_5d.csv"
OUTPUT_DIR      = "xgb_ranker_results"

TARGET_COL      = "ret_fwd_5d"
DATE_COL        = "date"
TICKER_COL      = "ticker"
FEATURE_SUFFIX  = "_z"          # use the z-scored columns as features
T_STAT_MIN      = 2.0           # only keep alphas with |t_stat| >= this

# Walk-forward CV
TRAIN_MONTHS    = 12
VAL_MONTHS      = 3

# Portfolio simulation
TOP_N           = 20            # long top N, short bottom N
TRANS_COST      = 0.001         # per side on rebalance days
REBAL_FREQ      = "W"           # "W"=weekly  "ME"=month-end  "D"=daily

# Grid search space — reduce to speed up, expand for thoroughness
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
    "reg_lambda":       [1.0, 5.0],
}
# ══════════════════════════════════════════════


# ── 1. LOAD ────────────────────────────────────

def load_panel():
    print("=" * 60)
    print("LOADING PANEL")
    print("=" * 60)

    if os.path.exists(PANEL_PATH):
        path = PANEL_PATH
    elif os.path.exists(PANEL_PATH.replace(".parquet", ".csv")):
        path = PANEL_PATH.replace(".parquet", ".csv")
    else:
        raise FileNotFoundError(f"Panel not found: {PANEL_PATH}")

    print(f"  Reading {path} ...")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"  Rows      : {len(df):,}")
    print(f"  Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    print(f"  Tickers   : {df[TICKER_COL].nunique()}")

    # Select features from IC summary
    if os.path.exists(IC_CSV_PATH):
        ic = pd.read_csv(IC_CSV_PATH)
        passing = ic[ic["t_stat"].abs() >= T_STAT_MIN]["alpha"].str.replace("_z$", "", regex=False) + "_z"
        # IC summary has cols like "alpha_32_12_1_month_momentum_z" already
        passing = ic[ic["t_stat"].abs() >= T_STAT_MIN]["alpha"].tolist()
        feature_cols = [c for c in passing if c in df.columns]
        print(f"  Features (|t|≥{T_STAT_MIN}): {len(feature_cols)}")
    else:
        feature_cols = [c for c in df.columns if c.endswith(FEATURE_SUFFIX)
                        and c not in (TARGET_COL, DATE_COL, TICKER_COL)]
        print(f"  Features (all _z): {len(feature_cols)}")

    df = df.dropna(subset=[TARGET_COL]).dropna(subset=feature_cols, how="all")
    print(f"  Rows after dropna: {len(df):,}")
    return df, feature_cols


# ── 2. WALK-FORWARD GRID SEARCH ────────────────

def rank_ic_score(y_true, y_pred):
    if len(y_true) < 3:
        return 0.0
    r, _ = spearmanr(y_pred, y_true)
    return float(r) if not np.isnan(r) else 0.0


def build_folds(df):
    months = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="MS")
    folds = []
    for i in range(0, len(months) - TRAIN_MONTHS - VAL_MONTHS + 1, VAL_MONTHS):
        tr = df[(df[DATE_COL] >= months[i]) & (df[DATE_COL] < months[i + TRAIN_MONTHS])]
        va = df[(df[DATE_COL] >= months[i + TRAIN_MONTHS]) &
                (df[DATE_COL] < months[min(i + TRAIN_MONTHS + VAL_MONTHS, len(months)-1)])]
        if len(tr) > 200 and len(va) > 50:
            folds.append((tr, va))
    return folds


def score_params(params, folds, feature_cols):
    import xgboost as xgb
    xgb_params = {
        "objective": "rank:pairwise", "eval_metric": "ndcg",
        "eta": params["learning_rate"], "max_depth": params["max_depth"],
        "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
        "lambda": params["reg_lambda"], "seed": 42, "nthread": -1, "verbosity": 0,
    }
    fold_ics = []
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
        preds = model.predict(dval)
        cursor, ics = 0, []
        for g in g_va:
            ics.append(rank_ic_score(va[TARGET_COL].values[cursor:cursor+g], preds[cursor:cursor+g]))
            cursor += g
        fold_ics.append(np.mean(ics))
    return np.mean(fold_ics), np.std(fold_ics)


def grid_search(df, feature_cols):
    print("\n" + "=" * 60)
    print("WALK-FORWARD GRID SEARCH")
    print("=" * 60)
    folds = build_folds(df)
    print(f"  Folds: {len(folds)}")
    keys   = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f"  Combos: {len(combos)}  ×  {len(folds)} folds = {len(combos)*len(folds)} fits")

    records, best_score, best_params = [], -np.inf, None
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        mean_ic, std_ic = score_params(params, folds, feature_cols)
        records.append({**params, "mean_rank_ic": mean_ic, "std_rank_ic": std_ic})
        if mean_ic > best_score:
            best_score, best_params = mean_ic, params.copy()
        step = max(1, len(combos) // 8)
        if (i+1) % step == 0 or i == len(combos)-1:
            print(f"  [{i+1:3d}/{len(combos)}]  best IC={best_score:.4f}  {best_params}")

    gs_df = pd.DataFrame(records).sort_values("mean_rank_ic", ascending=False).reset_index(drop=True)
    print(f"\n  ✓ Best Rank IC : {best_score:.4f}")
    print(f"  ✓ Best params  : {best_params}")
    return best_params, gs_df


# ── 3. FINAL MODEL ─────────────────────────────

def train_final(df, feature_cols, best_params):
    import xgboost as xgb
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    dates     = sorted(df[DATE_COL].unique())
    split     = dates[int(len(dates) * 0.8)]
    train     = df[df[DATE_COL] <  split]
    test      = df[df[DATE_COL] >= split]
    print(f"  Train: {train[DATE_COL].min().date()} → {train[DATE_COL].max().date()}  ({len(train):,} rows)")
    print(f"  Test : {test[DATE_COL].min().date()}  → {test[DATE_COL].max().date()}   ({len(test):,} rows)")

    g_tr = train.groupby(DATE_COL)[TICKER_COL].count().values.astype(np.int32)
    dtrain = xgb.DMatrix(train[feature_cols].fillna(0).values,
                         label=train[TARGET_COL].values, feature_names=feature_cols)
    dtrain.set_group(g_tr)

    xgb_params = {
        "objective": "rank:pairwise", "eval_metric": "ndcg",
        "eta": best_params["learning_rate"], "max_depth": best_params["max_depth"],
        "subsample": best_params["subsample"], "colsample_bytree": best_params["colsample_bytree"],
        "lambda": best_params["reg_lambda"], "seed": 42, "nthread": -1, "verbosity": 0,
    }
    model = xgb.train(xgb_params, dtrain,
                      num_boost_round=best_params["n_estimators"], verbose_eval=False)

    test = test.copy()
    test["xgb_score"] = model.predict(
        xgb.DMatrix(test[feature_cols].fillna(0).values, feature_names=feature_cols))

    fi = model.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)
    fi_df["short_name"] = fi_df["feature"].apply(
        lambda c: "_".join(c.replace("_z","").split("_")[2:]))

    return model, test, fi_df


# ── 4. SIMULATION ──────────────────────────────

def simulate(test_df):
    print("\n" + "=" * 60)
    print("PORTFOLIO SIMULATION")
    print("=" * 60)
    df = test_df.copy().sort_values(DATE_COL)
    all_dates = sorted(df[DATE_COL].unique())
    rebal_set = set(pd.Series(all_dates, index=all_dates)
                    .resample(REBAL_FREQ).last().dropna().values)

    records, longs, shorts = [], [], []
    for date in all_dates:
        day = df[df[DATE_COL] == date].dropna(subset=["xgb_score", TARGET_COL])
        if date in rebal_set and len(day) >= TOP_N * 2:
            ranked = day.sort_values("xgb_score", ascending=False)
            longs  = ranked.head(TOP_N)[TICKER_COL].tolist()
            shorts = ranked.tail(TOP_N)[TICKER_COL].tolist()
        if not longs:
            continue
        ret_map = day.set_index(TICKER_COL)[TARGET_COL].to_dict()
        lr = np.mean([ret_map[t] for t in longs  if t in ret_map])
        sr = np.mean([ret_map[t] for t in shorts if t in ret_map])
        tc = 2 * TRANS_COST if date in rebal_set else 0.0
        records.append({"date": date, "long_ret": lr, "short_ret": sr,
                         "ls_ret": lr - sr - tc})

    sim = pd.DataFrame(records).set_index("date")
    sim.index = pd.to_datetime(sim.index)
    sim["cum_long"] = (1 + sim["long_ret"]).cumprod()
    sim["cum_ls"]   = (1 + sim["ls_ret"]).cumprod()

    def metrics(r):
        r  = r.dropna()
        ar = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sh  = ar / vol if vol > 0 else np.nan
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        return {"Annual Return": ar, "Annual Vol": vol,
                "Sharpe": sh, "Max Drawdown": mdd, "Hit Rate": (r > 0).mean()}

    m_long = metrics(sim["long_ret"])
    m_ls   = metrics(sim["ls_ret"])

    print(f"\n  {'Metric':<20} {'Long Top-'+str(TOP_N):>12} {'Long-Short':>12}")
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
        if len(g) < 5: continue
        r, _ = spearmanr(g["xgb_score"].values, g[TARGET_COL].values)
        rows.append({"date": date, "rank_ic": float(r) if not np.isnan(r) else 0.0})
    ic = pd.DataFrame(rows).set_index("date")
    ic.index = pd.to_datetime(ic.index)
    ic["rolling20"] = ic["rank_ic"].rolling(20).mean()
    icir = ic["rank_ic"].mean() / ic["rank_ic"].std()
    print(f"\n  Out-of-sample mean Rank IC : {ic['rank_ic'].mean():.4f}")
    print(f"  Out-of-sample ICIR         : {icir:.3f}")
    return ic


# ── 5. PLOTS ───────────────────────────────────

BLUE, GREEN, RED, ORANGE, GREY = "#2563EB","#16A34A","#DC2626","#F59E0B","#6B7280"

def save_plot(fig, name, out):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")

def plot_returns(sim, out):
    fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True,
                              gridspec_kw={"hspace":0.08})
    ax = axes[0]
    ax.plot(sim.index, sim["cum_long"], color=BLUE,  lw=2, label=f"Long Top-{TOP_N}")
    ax.plot(sim.index, sim["cum_ls"],   color=GREEN, lw=2, label="Long-Short")
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_ylabel("Cumulative Return"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("XGBoost Ranker — Out-of-Sample Performance", fontweight="bold")
    ax2 = axes[1]
    for col, label, color in [("long_ret",f"Long Top-{TOP_N}",BLUE),("ls_ret","Long-Short",GREEN)]:
        cum = (1 + sim[col]).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        ax2.fill_between(sim.index, dd, 0, alpha=0.25, color=color)
        ax2.plot(sim.index, dd, color=color, lw=1.5, label=label)
    ax2.set_ylabel("Drawdown (%)"); ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Underwater Equity Curve")
    save_plot(fig, "cumulative_returns.png", out)

def plot_feature_importance(fi_df, out):
    top = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top)*0.5)))
    ax.barh(top["short_name"][::-1], top["gain"][::-1], color=BLUE, alpha=0.85)
    ax.set_xlabel("Gain"); ax.set_title("Feature Importance — Top 15", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_plot(fig, "feature_importance.png", out)

def plot_rank_ic(ic_df, out):
    fig, ax = plt.subplots(figsize=(14,5))
    colors = np.where(ic_df["rank_ic"] >= 0, GREEN, RED)
    ax.bar(ic_df.index, ic_df["rank_ic"], color=colors, alpha=0.5, width=2)
    ax.plot(ic_df.index, ic_df["rolling20"], color=BLUE, lw=2, label="20-day Rolling IC")
    mean_ic = ic_df["rank_ic"].mean()
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(mean_ic, color=ORANGE, lw=1.5, ls="--", label=f"Mean IC={mean_ic:.4f}")
    ax.set_title("Out-of-Sample Rank IC Over Time", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    save_plot(fig, "rank_ic_over_time.png", out)

def plot_annual(sim, out):
    annual = sim["ls_ret"].resample("YE").apply(lambda r: (1+r).prod()-1)
    fig, ax = plt.subplots(figsize=(12,5))
    colors = [GREEN if v >= 0 else RED for v in annual.values]
    ax.bar([str(d.year) for d in annual.index], annual.values*100, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(annual.values):
        ax.text(i, v+(0.5 if v>=0 else -1.5), f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Annual Return (%)"); ax.set_title("Long-Short Annual Returns", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    save_plot(fig, "annual_returns.png", out)


# ── MAIN ───────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    plot_returns(sim, OUTPUT_DIR)
    plot_feature_importance(fi_df, OUTPUT_DIR)
    plot_rank_ic(ic_df, OUTPUT_DIR)
    plot_annual(sim, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DONE — all outputs in:", os.path.abspath(OUTPUT_DIR))
    print("=" * 60)

if __name__ == "__main__":
    main()
