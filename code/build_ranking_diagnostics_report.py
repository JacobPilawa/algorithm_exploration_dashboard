#!/usr/bin/env python3
"""Build ranking-system diagnostics for the corrected JPAR dataset.

Outputs are intentionally curated:
- one master rankings CSV
- one largest-disagreements CSV
- one compact markdown summary
- one multipage PDF with plots/tables
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jpar_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


LOWER_BETTER = {
    "jpar",
    "raw_jpar_update",
    "mean_event_jpar",
    "mean_adjusted_event_jpar",
    "mean_log_zscore",
    "weighted_log_zscore",
    "mean_zscore",
    "weighted_zscore",
    "mean_event_percentile",
    "weighted_event_percentile",
    "mean_normalized_rank",
    "weighted_normalized_rank",
    "robust_log_zscore",
    "conservative_log_zscore",
    "best3_mean_log_zscore",
}
HIGHER_BETTER = {"elo_rating", "trueskill_conservative", "msp_like_score"}
SYSTEMS = [
    "jpar",
    "raw_jpar_update",
    "mean_adjusted_event_jpar",
    "weighted_log_zscore",
    "mean_log_zscore",
    "conservative_log_zscore",
    "robust_log_zscore",
    "best3_mean_log_zscore",
    "weighted_event_percentile",
    "mean_event_percentile",
    "weighted_normalized_rank",
    "mean_normalized_rank",
    "weighted_zscore",
    "mean_zscore",
    "elo_rating",
    "trueskill_conservative",
    "msp_like_score",
]


def keyify(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def mode_name(values: pd.Series) -> str:
    values = values.dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return "" if values.empty else values.mode().iloc[0]


def format_hhmmss(value: object) -> str:
    seconds = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(seconds):
        return ""
    total = int(round(float(seconds)))
    if total < 0:
        return ""
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def load_calc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"event_id": "string", "resolved_member_id": "string"})
    numeric_cols = [
        "completion_seconds",
        "event_mean_completion_seconds",
        "event_participant_count",
        "eligible_participant_count",
        "event_jpar",
        "adjusted_event_jpar",
        "previous_jpar",
        "jpar_out",
        "latest_jpar",
        "pieces_assembled",
        "piece_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["member_key"] = df["resolved_member_id"].apply(keyify)
    df = df.dropna(subset=["event_date", "event_id", "completion_seconds"])
    df = df[df["completion_seconds"].gt(0)].copy()
    return df.sort_values(["event_date", "event_id", "completion_seconds", "member_key"]).reset_index(drop=True)


def add_event_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_completion_seconds"] = np.log(out["completion_seconds"])
    stats = (
        out.groupby("event_id", dropna=False)
        .agg(
            event_std_seconds=("completion_seconds", "std"),
            event_mean_log_seconds=("log_completion_seconds", "mean"),
            event_std_log_seconds=("log_completion_seconds", "std"),
        )
        .reset_index()
    )
    out = out.merge(stats, on="event_id", how="left")
    out["event_zscore"] = (out["completion_seconds"] - out["event_mean_completion_seconds"]) / out["event_std_seconds"]
    out["event_log_zscore"] = (out["log_completion_seconds"] - out["event_mean_log_seconds"]) / out["event_std_log_seconds"]
    out.loc[out["event_std_seconds"].le(0) | out["event_std_seconds"].isna(), "event_zscore"] = np.nan
    out.loc[out["event_std_log_seconds"].le(0) | out["event_std_log_seconds"].isna(), "event_log_zscore"] = np.nan

    out["event_rank"] = out.groupby("event_id")["completion_seconds"].rank(method="average", ascending=True)
    out["event_percentile"] = out.groupby("event_id")["completion_seconds"].rank(pct=True, ascending=True)
    n_minus_1 = out["event_participant_count"] - 1
    out["event_normalized_rank"] = np.where(n_minus_1.gt(0), (out["event_rank"] - 1) / n_minus_1, np.nan)
    return out


def add_running_updates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["event_date", "event_id", "completion_seconds", "member_key"]).copy()
    running: dict[str, dict[str, float]] = defaultdict(dict)
    mapping = {
        "raw_jpar_update_out": "event_jpar",
        "weighted_zscore_out": "event_zscore",
        "weighted_log_zscore_out": "event_log_zscore",
        "weighted_event_percentile_out": "event_percentile",
        "weighted_normalized_rank_out": "event_normalized_rank",
    }
    values: dict[str, list[float]] = {col: [] for col in mapping}
    for _, row in out.iterrows():
        key = row["member_key"]
        for out_col, in_col in mapping.items():
            value = row[in_col]
            if not key or pd.isna(value):
                values[out_col].append(np.nan)
                continue
            previous = running[key].get(out_col)
            current = float(value) if previous is None or pd.isna(previous) else (float(previous) + float(value)) / 2
            running[key][out_col] = current
            values[out_col].append(current)
    for col, col_values in values.items():
        out[col] = col_values
    return out


def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings: dict[str, float] = defaultdict(lambda: 1500.0)
    events_seen: dict[str, int] = defaultdict(int)
    history_rows: list[dict[str, object]] = []
    for (event_date, event_id), event in df.groupby(["event_date", "event_id"], sort=True):
        event = event[event["member_key"].ne("")].sort_values("completion_seconds")
        keys = event["member_key"].tolist()
        if len(keys) < 2:
            continue
        deltas = defaultdict(float)
        for i, winner in enumerate(keys):
            for loser in keys[i + 1 :]:
                rw, rl = ratings[winner], ratings[loser]
                expected_w = 1 / (1 + 10 ** ((rl - rw) / 400))
                k_w = 24 / math.sqrt(max(1, events_seen[winner] + 1))
                k_l = 24 / math.sqrt(max(1, events_seen[loser] + 1))
                scale = max(1, len(keys) - 1)
                deltas[winner] += k_w * (1 - expected_w) / scale
                deltas[loser] += k_l * (0 - (1 - expected_w)) / scale
        for key, delta in deltas.items():
            ratings[key] += delta
        for key in keys:
            events_seen[key] += 1
            history_rows.append({"event_date": event_date, "event_id": event_id, "member_key": key, "elo_rating_out": ratings[key]})
    return pd.DataFrame({"member_key": list(ratings), "elo_rating": list(ratings.values())}), pd.DataFrame(history_rows)


def compute_trueskill(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import trueskill
    except Exception:
        return pd.DataFrame(columns=["member_key", "trueskill_conservative"]), pd.DataFrame()

    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings: dict[str, object] = defaultdict(env.create_rating)
    history_rows: list[dict[str, object]] = []
    for (event_date, event_id), event in df.groupby(["event_date", "event_id"], sort=True):
        event = event[event["member_key"].ne("")].sort_values("completion_seconds")
        keys = event["member_key"].tolist()
        if len(keys) < 2:
            continue
        new_groups = env.rate([(ratings[key],) for key in keys], ranks=list(range(len(keys))))
        for key, group in zip(keys, new_groups):
            ratings[key] = group[0]
            history_rows.append(
                {
                    "event_date": event_date,
                    "event_id": event_id,
                    "member_key": key,
                    "trueskill_mu_out": group[0].mu,
                    "trueskill_sigma_out": group[0].sigma,
                    "trueskill_conservative_out": group[0].mu - 3 * group[0].sigma,
                }
            )
    ratings_df = pd.DataFrame(
        {
            "member_key": list(ratings),
            "trueskill_mu": [rating.mu for rating in ratings.values()],
            "trueskill_sigma": [rating.sigma for rating in ratings.values()],
            "trueskill_conservative": [rating.mu - 3 * rating.sigma for rating in ratings.values()],
        }
    )
    return ratings_df, pd.DataFrame(history_rows)


def build_master(df: pd.DataFrame, elo: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    ranked = df[df["member_key"].ne("")].copy()
    name_col = "member_full_name" if "member_full_name" in ranked.columns else "full_name"
    people = (
        ranked.groupby("member_key", as_index=False)
        .agg(
            full_name=(name_col, mode_name),
            resolved_member_id=("resolved_member_id", "first"),
            events=("event_id", "nunique"),
            rows=("event_id", "size"),
            first_event_date=("event_date", "min"),
            latest_event_date=("event_date", "max"),
            jpar=("latest_jpar", "last"),
            mean_event_jpar=("event_jpar", "mean"),
            mean_adjusted_event_jpar=("adjusted_event_jpar", "mean"),
            mean_completion_seconds=("completion_seconds", "mean"),
            mean_zscore=("event_zscore", "mean"),
            mean_log_zscore=("event_log_zscore", "mean"),
            median_log_zscore=("event_log_zscore", "median"),
            mean_event_percentile=("event_percentile", "mean"),
            mean_normalized_rank=("event_normalized_rank", "mean"),
            best_event_percentile=("event_percentile", "min"),
        )
    )
    weighted = (
        ranked.groupby("member_key", as_index=False)
        .agg(
            raw_jpar_update=("raw_jpar_update_out", "last"),
            weighted_zscore=("weighted_zscore_out", "last"),
            weighted_log_zscore=("weighted_log_zscore_out", "last"),
            weighted_event_percentile=("weighted_event_percentile_out", "last"),
            weighted_normalized_rank=("weighted_normalized_rank_out", "last"),
        )
    )
    best3 = (
        ranked.sort_values("event_log_zscore")
        .groupby("member_key")
        .head(3)
        .groupby("member_key", as_index=False)["event_log_zscore"]
        .mean()
        .rename(columns={"event_log_zscore": "best3_mean_log_zscore"})
    )
    people = people.merge(weighted, on="member_key", how="left").merge(best3, on="member_key", how="left")
    people = people.merge(elo, on="member_key", how="left").merge(ts, on="member_key", how="left")
    people["robust_log_zscore"] = 0.65 * people["median_log_zscore"] + 0.35 * people["mean_log_zscore"]
    people["conservative_log_zscore"] = people["mean_log_zscore"] + 0.35 / np.sqrt(people["events"].clip(lower=1))
    people["msp_like_score"] = 1000 - 115 * people["mean_log_zscore"] - 35 * np.log1p(people["events"]) - 20 * (1 - people["best_event_percentile"])
    people["mean_time"] = people["mean_completion_seconds"].apply(format_hhmmss)
    return people


def add_ranks(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    for system in SYSTEMS:
        if system not in out.columns:
            continue
        out[f"{system}_rank"] = out[system].rank(method="min", ascending=system in LOWER_BETTER, na_option="bottom").astype("Int64")
    rank_cols = [f"{system}_rank" for system in SYSTEMS if f"{system}_rank" in out.columns]
    fixed = ["jpar_rank", "full_name", "resolved_member_id", "events", "mean_time", "first_event_date", "latest_event_date", "jpar"]
    other = [c for c in out.columns if c not in set(fixed + rank_cols + ["member_key"])]
    out = out.sort_values(["jpar_rank", "full_name"]).reset_index(drop=True)
    return out[fixed + [c for c in rank_cols if c != "jpar_rank"] + other]


def asof_percentiles(history: pd.DataFrame, value_col: str, higher_is_better: bool = False) -> pd.DataFrame:
    as_of: dict[str, float] = {}
    rows = []
    for seq, event in enumerate(history[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"]).to_dict("records"), start=1):
        event_rows = history[history["event_date"].eq(event["event_date"]) & history["event_id"].eq(event["event_id"])].dropna(subset=[value_col])
        as_of.update(dict(zip(event_rows["member_key"], event_rows[value_col])))
        values = pd.Series(as_of.values(), dtype="float64").dropna()
        if values.empty:
            continue
        if higher_is_better:
            values = -values
        rows.append(
            {
                "event_seq": seq,
                "event_date": event["event_date"],
                "event_id": event["event_id"],
                "n_ranked": len(values),
                "p10": values.quantile(0.10),
                "p25": values.quantile(0.25),
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
            }
        )
    return pd.DataFrame(rows)


def drift_tables(df: pd.DataFrame, elo_hist: pd.DataFrame, ts_hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = df[df["member_key"].ne("")].sort_values(["event_date", "event_id", "completion_seconds"])
    tables = {
        "JPAR": asof_percentiles(base.rename(columns={"jpar_out": "value"}), "value"),
        "Raw JPAR Update": asof_percentiles(base.rename(columns={"raw_jpar_update_out": "value"}), "value"),
        "Weighted Log Z": asof_percentiles(base.rename(columns={"weighted_log_zscore_out": "value"}), "value"),
        "Weighted Percentile": asof_percentiles(base.rename(columns={"weighted_event_percentile_out": "value"}), "value"),
        "Elo": asof_percentiles(elo_hist.rename(columns={"elo_rating_out": "value"}), "value", higher_is_better=True),
    }
    if not ts_hist.empty:
        tables["TrueSkill"] = asof_percentiles(ts_hist.rename(columns={"trueskill_conservative_out": "value"}), "value", higher_is_better=True)
    return tables


def event_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_date, event_id), sub in df.groupby(["event_date", "event_id"], sort=True):
        mean_expected = sub["mean_expected_event_average"].dropna()
        event_mean = sub["event_mean_completion_seconds"].dropna()
        rows.append(
            {
                "event_date": event_date,
                "event_id": event_id,
                "event_name": mode_name(sub.get("event_name", pd.Series(dtype=object))),
                "rows": len(sub),
                "eligible": sub["eligible_participant_count"].max(),
                "new_share": sub["previous_jpar"].isna().mean(),
                "median_event_jpar": sub["event_jpar"].median(),
                "median_adjusted_event_jpar": sub["adjusted_event_jpar"].median(),
                "calibration_multiplier": event_mean.iloc[0] / mean_expected.iloc[0] if len(event_mean) and len(mean_expected) and mean_expected.iloc[0] else np.nan,
                "raw_log_std": sub["event_log_zscore"].std(),
            }
        )
    return pd.DataFrame(rows)


def drift_decomposition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = df[df["member_key"].ne("")].dropna(subset=["jpar_out"]).copy()
    ranked["event_month"] = ranked["event_date"].dt.to_period("M").astype(str)
    first_seen = (
        ranked.sort_values(["event_date", "event_id"])
        .groupby("member_key", as_index=False)
        .first()[["member_key", "event_date", "event_id", "event_month", "jpar_out"]]
        .rename(
            columns={
                "event_date": "first_event_date",
                "event_id": "first_event_id",
                "event_month": "first_event_month",
                "jpar_out": "first_jpar",
            }
        )
    )
    ranked = ranked.merge(first_seen, on="member_key", how="left")
    ranked["is_first_event_for_member"] = ranked["event_date"].eq(ranked["first_event_date"]) & ranked["event_id"].eq(ranked["first_event_id"])

    as_of: dict[str, float] = {}
    snapshots: dict[str, dict[str, float]] = {}
    monthly_rows = []
    for month in sorted(ranked["event_month"].dropna().unique()):
        month_rows = ranked[ranked["event_month"].eq(month)]
        latest_month = month_rows.sort_values(["event_date", "event_id"]).groupby("member_key", sort=False).tail(1)
        as_of.update(dict(zip(latest_month["member_key"], latest_month["jpar_out"])))
        snapshots[month] = dict(as_of)
        values = pd.Series(as_of.values(), dtype="float64").dropna()
        firsts = first_seen[first_seen["first_event_month"].eq(month)]
        monthly_rows.append(
            {
                "month": month,
                "asof_n": len(values),
                "new_members": firsts["member_key"].nunique(),
                "new_member_share": firsts["member_key"].nunique() / len(values) if len(values) else np.nan,
                "asof_p10": values.quantile(0.10),
                "asof_p25": values.quantile(0.25),
                "asof_median_jpar": values.quantile(0.50),
                "asof_p75": values.quantile(0.75),
                "asof_p90": values.quantile(0.90),
                "new_entrant_first_jpar_median": firsts["first_jpar"].median(),
                "new_entrant_first_jpar_p75": firsts["first_jpar"].quantile(0.75),
                "event_rows": len(month_rows),
                "event_new_row_share": month_rows["is_first_event_for_member"].mean(),
                "median_event_jpar": month_rows["event_jpar"].median(),
                "median_adjusted_event_jpar": month_rows["adjusted_event_jpar"].median(),
                "median_calibration_multiplier": month_rows.groupby("event_id")["event_mean_completion_seconds"].first().median()
                / month_rows.groupby("event_id")["mean_expected_event_average"].first().median()
                if month_rows["mean_expected_event_average"].notna().any()
                else np.nan,
            }
        )

    cohort_rows = []
    cohorts = sorted(first_seen["first_event_month"].dropna().unique())
    for month, snap in snapshots.items():
        for cohort in cohorts:
            ids = set(first_seen.loc[first_seen["first_event_month"].eq(cohort), "member_key"])
            values = pd.Series([snap[mid] for mid in ids if mid in snap], dtype="float64").dropna()
            if values.empty:
                continue
            cohort_rows.append(
                {
                    "month": month,
                    "cohort_month": cohort,
                    "n": len(values),
                    "median_jpar": values.median(),
                    "p75_jpar": values.quantile(0.75),
                }
            )
    return pd.DataFrame(monthly_rows), pd.DataFrame(cohort_rows)


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, title, fontsize=20, weight="bold", va="top")
    y = 0.88
    for line in lines:
        fig.text(0.06, y, line, fontsize=10.5, va="top")
        y -= 0.032
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, font_size: float = 8.5) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", loc="left", pad=16)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_colored_rank_workbook(master: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    display_cols = [
        "full_name",
        "events",
        "mean_time",
        "jpar_rank",
        "weighted_log_zscore_rank",
        "conservative_log_zscore_rank",
        "weighted_event_percentile_rank",
        "elo_rating_rank",
        "trueskill_conservative_rank",
        "msp_like_score_rank",
        "jpar",
    ]
    display = master[[c for c in display_cols if c in master.columns]].head(150).copy()
    display = display.rename(
        columns={
            "full_name": "Name",
            "events": "Events",
            "mean_time": "Mean Time",
            "jpar_rank": "JPAR",
            "weighted_log_zscore_rank": "Weighted Log Z",
            "conservative_log_zscore_rank": "Conservative Log Z",
            "weighted_event_percentile_rank": "Weighted Percentile",
            "elo_rating_rank": "Elo H2H",
            "trueskill_conservative_rank": "TrueSkill",
            "msp_like_score_rank": "MSP-like",
            "jpar": "JPAR Score",
        }
    )
    rank_cols = [c for c in display.columns if c not in {"Name", "Events", "Mean Time", "JPAR Score"}]

    def color_rank(value: object) -> str:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(v):
            return ""
        if v <= 10:
            return "background-color: #14532d; color: white"
        if v <= 25:
            return "background-color: #22c55e; color: #052e16"
        if v <= 50:
            return "background-color: #bbf7d0; color: #052e16"
        if v <= 100:
            return "background-color: #fef3c7; color: #422006"
        return "background-color: #fee2e2; color: #450a0a"

    styled = (
        display.style.applymap(color_rank, subset=rank_cols)
        .format({"JPAR Score": "{:.6f}"})
        .set_caption("Top 150: Rank Comparison Across Systems")
    )
    xlsx_path = output_dir / "colored_rank_comparison_top150.xlsx"
    html_path = output_dir / "colored_rank_comparison_top150.html"
    styled.to_excel(xlsx_path, index=False, engine="openpyxl")
    styled.to_html(html_path, index=False)
    return xlsx_path, html_path


def plot_score_distributions(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()
    axes[0].hist(df["latest_jpar"].dropna(), bins=45, color="#2563eb", alpha=0.8)
    axes[0].set_title("Latest JPAR Distribution")
    axes[1].scatter(df["event_mean_completion_seconds"], df["event_jpar"], s=8, alpha=0.25, color="#0f766e")
    axes[1].set_title("Event Mean Time vs Raw Event JPAR")
    axes[1].set_xlabel("event mean seconds")
    axes[1].set_ylabel("event_jpar")
    axes[2].hist(df["completion_seconds"].dropna(), bins=50, color="#f59e0b", alpha=0.8)
    axes[2].set_title("Raw Completion Seconds")
    axes[3].hist(np.log(df["completion_seconds"].dropna()), bins=50, color="#7c3aed", alpha=0.8)
    axes[3].set_title("Log Completion Seconds")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Core Score And Time Diagnostics", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def plot_corr(pdf: PdfPages, master: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [f"{system}_rank" for system in SYSTEMS if f"{system}_rank" in master.columns]
    corr = master[rank_cols].astype(float).corr(method="spearman")
    labels = [c[:-5] for c in rank_cols]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman Rank Correlation Across Systems", fontsize=16, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    corr.index = labels
    corr.columns = labels
    return corr


def plot_rank_scatter(pdf: PdfPages, master: pd.DataFrame) -> None:
    cols = ["weighted_log_zscore_rank", "conservative_log_zscore_rank", "weighted_event_percentile_rank", "elo_rating_rank", "trueskill_conservative_rank", "msp_like_score_rank"]
    cols = [c for c in cols if c in master.columns]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), squeeze=False)
    x = master["jpar_rank"].astype(float)
    for ax, col in zip(axes.ravel(), cols):
        y = master[col].astype(float)
        rho = x.corr(y, method="spearman")
        ax.scatter(x, y, s=12, alpha=0.45, color="#2563eb", edgecolors="none")
        lim = max(x.max(), y.max())
        ax.plot([1, lim], [1, lim], color="#991b1b", linewidth=1)
        ax.set_title(f"{col[:-5]} vs JPAR (rho={rho:.2f})", fontsize=10)
        ax.set_xlabel("JPAR rank")
        ax.set_ylabel("system rank")
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Selected Alternative Ranks vs JPAR", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def plot_rank_vs_mean_time(pdf: PdfPages, master: pd.DataFrame) -> None:
    rank_cols = [f"{system}_rank" for system in SYSTEMS if f"{system}_rank" in master.columns]
    pages = [rank_cols[i : i + 6] for i in range(0, len(rank_cols), 6)]
    x = master["mean_completion_seconds"].astype(float) / 60.0
    for page_num, cols in enumerate(pages, start=1):
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), squeeze=False)
        for ax, col in zip(axes.ravel(), cols):
            y = master[col].astype(float)
            colors = np.where(master["events"].ge(3), "#2563eb", "#f97316")
            ax.scatter(x, y, s=12 + np.minimum(master["events"], 10) * 2, c=colors, alpha=0.42, edgecolors="none")
            rho = x.corr(y, method="spearman")
            ax.set_title(f"{col[:-5]} rank vs mean time\nrho={rho:.2f}; orange=<3 events", fontsize=9.5)
            ax.set_xlabel("mean completion time (minutes)")
            ax.set_ylabel("rank")
            ax.invert_yaxis()
            ax.grid(alpha=0.22)
        for ax in axes.ravel()[len(cols):]:
            ax.axis("off")
        fig.suptitle(f"Rank vs Mean Time by Person ({page_num}/{len(pages)})", fontsize=16, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


def plot_drift(pdf: PdfPages, tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), squeeze=False)
    for ax, (name, table) in zip(axes.ravel(), tables.items()):
        t = table.copy()
        t["event_date"] = pd.to_datetime(t["event_date"])
        ax.fill_between(t["event_date"], t["p10"], t["p90"], color="#bfdbfe", alpha=0.55)
        ax.fill_between(t["event_date"], t["p25"], t["p75"], color="#60a5fa", alpha=0.55)
        ax.plot(t["event_date"], t["p50"], color="#111827", linewidth=2)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[len(tables):]:
        ax.axis("off")
    fig.suptitle("As-Of Distribution Drift (lower is better; Elo/TrueSkill negated)", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def plot_event_calibration(pdf: PdfPages, events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    sc = axes[0].scatter(events["event_date"], events["calibration_multiplier"], s=20 + events["rows"] * 0.4, c=events["new_share"], cmap="viridis", alpha=0.85)
    axes[0].axhline(1.0, color="#991b1b", linestyle="--", linewidth=1)
    axes[0].set_title("JPAR Calibration Multiplier")
    axes[0].set_xlabel("event date")
    axes[0].set_ylabel("event mean / expected event average")
    axes[0].tick_params(axis="x", rotation=35)
    fig.colorbar(sc, ax=axes[0], label="share without prior JPAR")
    axes[1].scatter(events["new_share"], events["median_adjusted_event_jpar"], s=20 + events["rows"] * 0.4, alpha=0.8, color="#0f766e")
    axes[1].set_title("New Entrant Share vs Adjusted Event JPAR")
    axes[1].set_xlabel("share without prior JPAR")
    axes[1].set_ylabel("median adjusted_event_jpar")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Event-Level Calibration Diagnostics", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def plot_drift_decomposition(pdf: PdfPages, monthly: pd.DataFrame, cohorts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=False)
    x = monthly["month"]
    axes[0].plot(x, monthly["asof_median_jpar"], marker="o", linewidth=2.3, label="All ranked as-of median")
    axes[0].plot(x, monthly["new_entrant_first_jpar_median"], marker="o", linewidth=2.0, label="New entrants' first-JPAR median")
    first_cohort = cohorts["cohort_month"].min() if not cohorts.empty else None
    if first_cohort is not None:
        fixed = cohorts[cohorts["cohort_month"].eq(first_cohort)]
        axes[0].plot(fixed["month"], fixed["median_jpar"], marker="o", linewidth=2.0, label=f"Fixed cohort median ({first_cohort})")
    axes[0].set_title("JPAR Drift Decomposition")
    axes[0].set_ylabel("JPAR")
    axes[0].legend(frameon=False, ncol=1)
    axes[0].grid(alpha=0.22)
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(x, monthly["new_members"], color="#93c5fd", alpha=0.8, label="New members")
    ax2 = axes[1].twinx()
    ax2.plot(x, monthly["new_member_share"], color="#b45309", marker="s", label="New member share")
    ax2.plot(x, monthly["event_new_row_share"], color="#0f766e", marker="^", label="New row share")
    axes[1].set_title("Population Mix Over Time")
    axes[1].set_ylabel("New member count")
    ax2.set_ylabel("Share")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].tick_params(axis="x", rotation=35)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def write_drift_inference(output_dir: Path, monthly: pd.DataFrame, events: pd.DataFrame, corr: pd.DataFrame) -> Path:
    first_median = monthly["asof_median_jpar"].dropna().iloc[0]
    last_median = monthly["asof_median_jpar"].dropna().iloc[-1]
    drift = last_median - first_median
    event_corr = events[["calibration_multiplier", "new_share", "median_adjusted_event_jpar", "rows"]].corr(numeric_only=True)
    cal_new_corr = event_corr.loc["calibration_multiplier", "new_share"] if "calibration_multiplier" in event_corr.index else np.nan
    lines = [
        "# JPAR Drift Inference",
        "",
        f"As-of median JPAR moved from `{first_median:.4f}` to `{last_median:.4f}` across the corrected cutoff run, a change of `{drift:+.4f}`.",
        "",
        "## Mechanism",
        "",
        "JPAR is not anchored to an invariant external scale. Each event first computes `event_jpar = completion_seconds / event_mean_completion_seconds`, then sometimes recalibrates that event using returning players' prior JPARs: `adjusted_event_jpar = completion_seconds / mean_expected_event_average`. Finally, a person's score is updated as a half-average of previous JPAR and current adjusted event JPAR.",
        "",
        "This means the system is path-dependent. If the population entering the ranking changes, or if returning players at an event are not representative of the event field, the calibration factor can move the entire event up or down. New players inherit that shifted event scale as their initial JPAR, and then they become future calibration anchors. That feedback loop is the root reason drift can persist.",
        "",
        "## Diagnostics From This Run",
        "",
        f"- Correlation between event calibration multiplier and new-row share: `{cal_new_corr:.3f}`.",
        "- Compare the PDF drift decomposition page: if the fixed early cohort is stable while the all-ranked median drifts, population mix is a major driver. If the fixed cohort also drifts, the update/calibration mechanics are changing scores for existing people.",
        "- Large one-event disagreements indicate that current JPAR can be strongly affected by a person's first event context, especially before enough history accumulates.",
        "",
        "## Mitigation Options",
        "",
        "1. Use log-z or percentile event scores as the primary event-normalized input. These are anchored to within-event distributions rather than raw event average ratios.",
        "2. Keep the running update, but update toward a stable within-event metric (`weighted_log_zscore` or `weighted_event_percentile`) instead of recalibrated JPAR ratios.",
        "3. If preserving JPAR, constrain calibration multipliers: shrink event calibration toward 1.0 unless there are enough returning anchors and their prior ratings are representative.",
        "4. Add uncertainty or minimum-event rules for publication: provisional until 2 or 3 events; use conservative rank for one-event participants.",
        "5. Publish both a performance rating and an uncertainty/experience field. This prevents one-event stars and one-event poor showings from being overinterpreted.",
    ]
    path = output_dir / "jpar_drift_inference.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def largest_disagreements(master: pd.DataFrame) -> pd.DataFrame:
    systems = ["weighted_log_zscore", "conservative_log_zscore", "weighted_event_percentile", "elo_rating", "trueskill_conservative", "msp_like_score"]
    rows = []
    for system in systems:
        col = f"{system}_rank"
        if col not in master.columns:
            continue
        temp = master[["full_name", "events", "jpar_rank", col, "jpar"]].copy()
        temp["system"] = system
        temp = temp.rename(columns={col: "system_rank"})
        temp["rank_delta_vs_jpar"] = temp["system_rank"].astype(float) - temp["jpar_rank"].astype(float)
        rows.append(temp)
    out = pd.concat(rows, ignore_index=True)
    return out.reindex(out["rank_delta_vs_jpar"].abs().sort_values(ascending=False).index).head(250)


def write_summary(path: Path, master: pd.DataFrame, corr: pd.DataFrame, events: pd.DataFrame) -> None:
    jpar_corr = corr["jpar"].sort_values(ascending=False).drop("jpar", errors="ignore")
    lines = [
        "# Corrected Dataset Ranking Diagnostics",
        "",
        f"Ranked members: {len(master):,}",
        f"Events analyzed: {events['event_id'].nunique():,}",
        f"Date range: {events['event_date'].min().date()} to {events['event_date'].max().date()}",
        "",
        "## Highest Correlations With JPAR",
        "",
    ]
    for system, value in jpar_corr.head(10).items():
        lines.append(f"- `{system}`: {value:.3f}")
    lines += [
        "",
        "## Notes",
        "",
        "- `weighted_log_zscore`, `weighted_event_percentile`, Elo, and TrueSkill are useful comparison systems because they reduce or avoid direct dependence on JPAR's event calibration multiplier.",
        "- Positive `rank_delta_vs_jpar` in the disagreement CSV means the alternative system ranks someone worse than JPAR; negative means it ranks them better.",
        "- The PDF is the primary review artifact; CSVs are included only for drill-down.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corrected ranking diagnostics report")
    parser.add_argument("--input", default="data/data_jpar_v2/source_of_truth_calculation_df.csv")
    parser.add_argument("--output-dir", default="outputs/corrected_ranking_diagnostics")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_running_updates(add_event_metrics(load_calc(input_path)))
    elo, elo_hist = compute_elo(df)
    ts, ts_hist = compute_trueskill(df)
    master = add_ranks(build_master(df, elo, ts))
    events = event_diagnostics(df)
    drifts = drift_tables(df, elo_hist, ts_hist)
    monthly_drift, cohort_drift = drift_decomposition(df)
    disagreements = largest_disagreements(master)

    master_path = output_dir / "master_ranking_systems.csv"
    disagree_path = output_dir / "largest_rank_disagreements.csv"
    summary_path = output_dir / "summary.md"
    pdf_path = output_dir / "ranking_diagnostics_report.pdf"
    master.to_csv(master_path, index=False)
    disagreements.to_csv(disagree_path, index=False)
    events.to_csv(output_dir / "event_calibration_diagnostics.csv", index=False)
    monthly_drift.to_csv(output_dir / "monthly_jpar_drift_decomposition.csv", index=False)
    cohort_drift.to_csv(output_dir / "cohort_jpar_drift_decomposition.csv", index=False)
    colored_xlsx, colored_html = write_colored_rank_workbook(master, output_dir)

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Corrected JPAR Ranking Diagnostics",
            [
                f"Input: {input_path}",
                f"Rows analyzed: {len(df):,}",
                f"Events analyzed: {df['event_id'].nunique():,}",
                f"Ranked members: {len(master):,}",
                f"Date range: {df['event_date'].min().date()} to {df['event_date'].max().date()}",
                "",
                "Systems compared: current JPAR, raw JPAR update, raw/log z-scores, percentiles, normalized ranks, Elo, TrueSkill, and conservative blends.",
                "The goal is diagnostic: identify drift, rank disagreements, one-event sensitivity, and event-calibration effects.",
            ],
        )
        top = master.head(25)[["jpar_rank", "full_name", "events", "mean_time", "jpar", "weighted_log_zscore_rank", "elo_rating_rank", "trueskill_conservative_rank"]].copy()
        top["jpar"] = top["jpar"].round(6)
        add_table_page(pdf, "Current JPAR Top 25 With Selected Alternative Ranks", top)
        plot_score_distributions(pdf, df)
        corr = plot_corr(pdf, master)
        plot_rank_scatter(pdf, master)
        plot_rank_vs_mean_time(pdf, master)
        plot_drift(pdf, drifts)
        plot_drift_decomposition(pdf, monthly_drift, cohort_drift)
        plot_event_calibration(pdf, events)
        disagree_table = disagreements.head(25)[["system", "full_name", "events", "jpar_rank", "system_rank", "rank_delta_vs_jpar", "jpar"]].copy()
        disagree_table["jpar"] = disagree_table["jpar"].round(6)
        add_table_page(pdf, "Largest Rank Disagreements vs JPAR", disagree_table, font_size=7.5)

    write_summary(summary_path, master, corr, events)
    drift_inference_path = write_drift_inference(output_dir, monthly_drift, events, corr)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {master_path}")
    print(f"Wrote {disagree_path}")
    print(f"Wrote {colored_xlsx}")
    print(f"Wrote {colored_html}")
    print(f"Wrote {drift_inference_path}")
    print(master.head(15)[["jpar_rank", "full_name", "events", "mean_time", "jpar", "weighted_log_zscore_rank", "elo_rating_rank", "trueskill_conservative_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
