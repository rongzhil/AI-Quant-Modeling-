"""Two-step feature selection over the raw alphas.

Step 1 - significance filter:
    Drop any alpha whose |Rank-IC t-stat| < RANK_IC_T_STAT_THRESHOLD. These add
    no reliable signal. Absolute value is used so reversal alphas (negative IC)
    survive when significant.

Step 2 - redundancy clustering:
    Among the survivors, compute pairwise rank correlation (pooled Spearman over
    the train segment). Alphas with |r| >= REDUNDANCY_CORR_THRESHOLD are mutually
    redundant; greedily cluster them and keep only the highest-|ICIR| member of
    each cluster (the cluster head), dropping the rest.

Both steps use the TRAIN segment only, consistent with the Rank-IC evaluation,
so the test set never influences which features are chosen.

Inputs:
    * alpha_panel : long panel with date, ticker and the raw alpha columns
      (the same panel fed to evaluate; only train rows are used here).
    * ic_summary  : the table from evaluate.compute_rank_ic (alpha, mean_ic,
      std_ic, icir, t_stat, n_days).

Outputs (written by ``run_feature_selection``):
    * selected_features.csv      : one row per kept alpha, with its stats.
    * feature_selection_report.txt : human-readable account of both steps,
      including every cluster and why each alpha was kept or dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, evaluate


def _train_only(panel: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(panel["date"]).dt.normalize()
    return panel[d <= pd.Timestamp(config.TRAIN_END_DATE)]


def significance_filter(ic_summary: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Step 1: split alphas into survivors and dropped by |t_stat|.

    Returns (survivors, dropped), each a list of alpha column names. Alphas with
    NaN t_stat (never enough cross-section to evaluate) are dropped.
    """
    thr = config.RANK_IC_T_STAT_THRESHOLD
    survivors, dropped = [], []
    for _, row in ic_summary.iterrows():
        t = row["t_stat"]
        if pd.notna(t) and abs(t) >= thr:
            survivors.append(row["alpha"])
        else:
            dropped.append(row["alpha"])
    return survivors, dropped


def rank_correlation_matrix(panel: pd.DataFrame, alpha_cols: list[str]) -> pd.DataFrame:
    """Pooled Spearman correlation among the given alphas on the train segment.

    Pooled over all (date, ticker) rows: this measures whether two alphas carry
    the same cross-sectional information overall. Spearman = rank-based, so it is
    immune to each alpha's scale.
    """
    train = _train_only(panel)
    data = train[alpha_cols].apply(pd.to_numeric, errors="coerce")
    # Spearman handles NaNs pairwise and is scale-free.
    return data.corr(method="spearman")


def redundancy_cluster(
    survivors: list[str],
    corr: pd.DataFrame,
    icir_by_alpha: dict[str, float],
) -> tuple[list[str], list[dict]]:
    """Step 2: greedily cluster |r| >= threshold, keep highest-|ICIR| per cluster.

    Processing alphas from strongest to weakest |ICIR|, each not-yet-assigned
    alpha starts a new cluster as its head; any not-yet-assigned alpha correlated
    with the head at |r| >= threshold joins that cluster and is dropped.

    Returns (kept, clusters) where ``kept`` is the list of cluster-head columns
    and ``clusters`` is a list of dicts describing each cluster for the report.
    """
    thr = config.REDUNDANCY_CORR_THRESHOLD
    # Order survivors by descending |ICIR| so the strongest becomes each head.
    ordered = sorted(
        survivors,
        key=lambda a: (abs(icir_by_alpha.get(a, np.nan)) if pd.notna(icir_by_alpha.get(a, np.nan)) else -np.inf),
        reverse=True,
    )

    assigned: set[str] = set()
    kept: list[str] = []
    clusters: list[dict] = []

    for head in ordered:
        if head in assigned:
            continue
        assigned.add(head)
        members = [head]
        for other in ordered:
            if other in assigned or other == head:
                continue
            r = corr.loc[head, other] if (head in corr.index and other in corr.columns) else np.nan
            if pd.notna(r) and abs(r) >= thr:
                assigned.add(other)
                members.append(other)
        kept.append(head)
        clusters.append({
            "head": head,
            "members": members,
            "dropped": [m for m in members if m != head],
        })

    return kept, clusters


def select_features(alpha_panel: pd.DataFrame, ic_summary: pd.DataFrame) -> dict:
    """Run both steps and return a structured result.

    Result keys:
        selected            : list of kept alpha columns
        dropped_significance : list dropped in step 1
        dropped_redundancy   : list dropped in step 2
        clusters             : cluster descriptions from step 2
        survivors_step1      : survivors after step 1
        ic_summary           : the input summary (for writing the csv)
    """
    survivors, dropped_sig = significance_filter(ic_summary)

    icir_by_alpha = dict(zip(ic_summary["alpha"], ic_summary["icir"]))

    if survivors:
        corr = rank_correlation_matrix(alpha_panel, survivors)
        kept, clusters = redundancy_cluster(survivors, corr, icir_by_alpha)
    else:
        corr = pd.DataFrame()
        kept, clusters = [], []

    dropped_red = [m for c in clusters for m in c["dropped"]]

    return {
        "selected": kept,
        "dropped_significance": dropped_sig,
        "dropped_redundancy": dropped_red,
        "clusters": clusters,
        "survivors_step1": survivors,
        "ic_summary": ic_summary,
        "corr": corr,
    }


def _format_report(result: dict) -> str:
    """Build the human-readable feature-selection report."""
    ic = result["ic_summary"].set_index("alpha")
    thr_t = config.RANK_IC_T_STAT_THRESHOLD
    thr_r = config.REDUNDANCY_CORR_THRESHOLD
    lines: list[str] = []

    def stat(alpha: str) -> str:
        if alpha in ic.index:
            row = ic.loc[alpha]
            return (f"mean_ic={row['mean_ic']:+.4f} icir={row['icir']:+.3f} "
                    f"t_stat={row['t_stat']:+.2f} n_days={int(row['n_days'])}")
        return "(no stats)"

    lines.append("=" * 72)
    lines.append("FEATURE SELECTION REPORT")
    lines.append("=" * 72)
    lines.append(f"Train cutoff           : {config.TRAIN_END_DATE}")
    lines.append(f"Significance threshold : |t_stat| >= {thr_t}")
    lines.append(f"Redundancy threshold   : |rank-corr| >= {thr_r}")
    lines.append("")
    lines.append(f"Alphas in              : {len(ic)}")
    lines.append(f"Survived step 1        : {len(result['survivors_step1'])}")
    lines.append(f"Final selected         : {len(result['selected'])}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("STEP 1 - SIGNIFICANCE FILTER")
    lines.append("-" * 72)
    lines.append(f"Dropped ({len(result['dropped_significance'])}) for |t_stat| < {thr_t}:")
    for a in result["dropped_significance"]:
        lines.append(f"  - {a:42s} {stat(a)}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("STEP 2 - REDUNDANCY CLUSTERS")
    lines.append("-" * 72)
    lines.append(f"{len(result['clusters'])} cluster(s); cluster head (highest |ICIR|) is kept.")
    lines.append("")
    for i, c in enumerate(result["clusters"], 1):
        size = len(c["members"])
        tag = "singleton" if size == 1 else f"{size} members"
        lines.append(f"Cluster {i} [{tag}] - KEEP {c['head']}")
        lines.append(f"    head : {c['head']:42s} {stat(c['head'])}")
        for m in c["dropped"]:
            r = result["corr"].loc[c["head"], m] if (not result["corr"].empty) else np.nan
            lines.append(f"    drop : {m:42s} {stat(m)}  |r_to_head|={abs(r):.3f}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("FINAL SELECTED FEATURES")
    lines.append("-" * 72)
    for a in result["selected"]:
        lines.append(f"  + {a:42s} {stat(a)}")
    lines.append("")
    return "\n".join(lines)


def run_feature_selection(
    alpha_panel: pd.DataFrame,
    ic_summary: pd.DataFrame,
    write: bool = True,
) -> dict:
    """End-to-end: select features, optionally write csv + report, return result."""
    result = select_features(alpha_panel, ic_summary)

    ic = ic_summary.set_index("alpha")
    selected_rows = ic.loc[[a for a in result["selected"] if a in ic.index]].reset_index()
    result["selected_table"] = selected_rows
    report_text = _format_report(result)
    result["report_text"] = report_text

    if write:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        selected_rows.to_csv(config.SELECTED_FEATURES_PATH, index=False)
        config.SELECTION_REPORT_PATH.write_text(report_text)

    return result