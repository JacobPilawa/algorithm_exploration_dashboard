#!/usr/bin/env python3
"""Build a sortable interactive ranking comparison HTML table."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXTERNAL_OUTPUT_DIR: Path | None = Path("colleague_ranking_systems/outputs")
EXTERNAL_SYSTEMS = [
    ("external_logtime", "Log-Time Volatility (Mean)"),
    ("external_logtime_conservative", "Log-Time Volatility (Conservative)"),
    ("external_logtime_no_tier", "Log-Time Volatility (No Tier Boost)"),
    ("external_bayesian", "Bayesian Skill Mu"),
    ("external_bayesian_conservative", "Bayesian Skill (Conservative)"),
    ("external_nationals", "Nationals-Constrained Log-Time"),
]

DISPLAY_COLUMNS = [
    ("full_name", "Name", "text"),
    ("events", "Events", "integer"),
    ("mean_time", "Mean Time", "text"),
    ("jpar", "JPAR Score", "number"),
    ("jpar_rank", "JPAR Rank", "rank"),
    ("raw_jpar_update_rank", "Raw JPAR Update", "rank"),
    ("mean_adjusted_event_jpar_rank", "Mean Adjusted JPAR", "rank"),
    ("mean_log_zscore_rank", "Mean Log Z Score", "rank"),
    ("mean_zscore_rank", "Mean Raw Z Score", "rank"),
    ("weighted_log_zscore_rank", "Weighted Log Z Score", "rank"),
    ("weighted_zscore_rank", "Weighted Raw Z Score", "rank"),
    ("sof_weighted_log_zscore_rank", "SOF Weighted Log Z", "rank"),
    ("sof_mean_log_zscore_rank", "SOF Mean Log Z", "rank"),
    ("sof_weighted_zscore_rank", "SOF Weighted Raw Z", "rank"),
    ("sof_mean_zscore_rank", "SOF Mean Raw Z", "rank"),
    ("sof_weighted_centered_log_rank", "SOF Weighted Centered Log", "rank"),
    ("sof_mean_centered_log_rank", "SOF Mean Centered Log", "rank"),
    ("weighted_event_percentile_rank", "Weighted Percentile", "rank"),
    ("mean_event_percentile_rank", "Mean Percentile", "rank"),
    ("weighted_normalized_rank_rank", "Weighted Norm Rank", "rank"),
    ("mean_normalized_rank_rank", "Mean Norm Rank", "rank"),
    ("elo_rating_rank", "Elo H2H", "rank"),
    ("trueskill_conservative_rank", "TrueSkill", "rank"),
    ("msp_like_score_rank", "MSP-like", "rank"),
    ("external_logtime_rank", "Log-Time Volatility Mean", "rank"),
    ("external_logtime_conservative_rank", "Log-Time Volatility Conservative", "rank"),
    ("external_logtime_no_tier_rank", "Log-Time Volatility — No Tier Boost", "rank"),
    ("external_bayesian_rank", "Bayesian Skill", "rank"),
    ("external_bayesian_conservative_rank", "Bayesian Skill Conservative", "rank"),
    ("external_nationals_rank", "Nationals-Constrained", "rank"),
]

# Preserved in the analysis code for future re-enabling, but intentionally
# hidden from every current page while these systems receive further testing.
DISABLED_SYSTEM_RANK_KEYS = {
    "raw_jpar_update_rank",
    "mean_adjusted_event_jpar_rank",
    "weighted_event_percentile_rank",
    "mean_event_percentile_rank",
    "weighted_normalized_rank_rank",
    "mean_normalized_rank_rank",
    "elo_rating_rank",
    "trueskill_conservative_rank",
    "msp_like_score_rank",
    "sof_weighted_log_zscore_rank",
    "sof_mean_log_zscore_rank",
    "sof_weighted_zscore_rank",
    "sof_mean_zscore_rank",
    "sof_weighted_centered_log_rank",
    "sof_mean_centered_log_rank",
    "external_logtime_no_tier_rank",
}

SYSTEM_DESCRIPTIONS = {
    "jpar_rank": "Current pipeline JPAR rank. Lower score/rank is better. Uses calibrated adjusted_event_jpar and a running half-update: first event uses adjusted_event_jpar, later events use (previous_jpar + adjusted_event_jpar) / 2.",
    "raw_jpar_update_rank": "Same running half-update as JPAR, but uses raw event_jpar = completion_seconds / event_mean_completion_seconds instead of calibrated adjusted_event_jpar.",
    "mean_adjusted_event_jpar_rank": "Simple average of adjusted_event_jpar across a person's events. Keeps JPAR's event calibration but removes the running recency/path update.",
    "weighted_log_zscore_rank": "Running half-update applied to within-event log-time z-scores. Event score is zscore(log(completion_seconds)); lower is better.",
    "mean_log_zscore_rank": "Simple average of within-event z-scores of log(completion_seconds). Lower means consistently faster than the event field after log-normalizing time.",
    "weighted_event_percentile_rank": "Running half-update applied to within-event percentile rank. Lower is better.",
    "mean_event_percentile_rank": "Average within-event percentile rank. Lower is better.",
    "weighted_normalized_rank_rank": "Running half-update applied to normalized within-event rank: (event_rank - 1) / (event_participant_count - 1), where 0 is first and 1 is last.",
    "mean_normalized_rank_rank": "Average normalized within-event rank: (event_rank - 1) / (event_participant_count - 1). Lower is better.",
    "weighted_zscore_rank": "Running half-update applied to raw-time event z-scores: (completion_seconds - event_mean_seconds) / event_std_seconds. Lower is better.",
    "mean_zscore_rank": "Average raw-time event z-score. Lower is better. More sensitive to skewed time distributions than log-z.",
    "sof_weighted_log_zscore_rank": "Strength-of-field-adjusted running half-update of log-time event z-scores. Before each event, the entrants' incoming average score is added to their ordinary event z-score, so an average result in a strong field remains stronger than an average result in a weak field. Unrated entrants use a zero prior. Lower is better.",
    "sof_mean_log_zscore_rank": "Strength-of-field-adjusted running mean of log-time event z-scores. Adjusted event score = ordinary log-time event z-score + the field's incoming average score, with zero used for debut entrants. Lower is better.",
    "sof_weighted_zscore_rank": "Strength-of-field-adjusted running half-update of raw-time event z-scores. Adjusted event score = ordinary raw-time event z-score + the field's incoming average score, with zero used for debut entrants. Lower is better.",
    "sof_mean_zscore_rank": "Strength-of-field-adjusted running mean of raw-time event z-scores. Adjusted event score = ordinary raw-time event z-score + the field's incoming average score, with zero used for debut entrants. Lower is better.",
    "sof_weighted_centered_log_rank": "Strength-of-field-adjusted running half-update of centered log time. Event observation = log(time) minus the event mean log time plus the field's incoming average score. It deliberately does not divide by event standard deviation; debut entrants use a zero prior. Lower is better.",
    "sof_mean_centered_log_rank": "Strength-of-field-adjusted running mean of centered log time. Event observation = log(time) minus the event mean log time plus the field's incoming average score. It deliberately does not divide by event standard deviation; debut entrants use a zero prior. Lower is better.",
    "elo_rating_rank": "Pairwise head-to-head Elo. In each event, every faster solver is treated as beating every slower solver. Higher Elo score is better, converted here to rank.",
    "trueskill_conservative_rank": "TrueSkill multiplayer race model, ranked by conservative score mu - 3*sigma. Higher score is better, converted here to rank.",
    "msp_like_score_rank": "Invented points-like diagnostic score: 1000 - 115*mean_log_zscore - 35*log1p(events) - 20*(1 - best_event_percentile). Higher score is better, converted here to rank.",
    "external_logtime_rank": "Dynamic log-time model. Each result is expressed relative to an estimated event difficulty, then a Kalman-style update tracks the puzzler's speed, uncertainty, volatility, inactivity, and recent surprises. Higher score is better.",
    "external_logtime_conservative_rank": "The same Log-Time Volatility state, ranked by the conservative score −(ability mean + 2×uncertainty SD). This favors estimates supported by lower uncertainty; higher score is better.",
    "external_logtime_no_tier_rank": "The same dynamic log-time volatility model, but every event uses the ordinary observation variance. USA Nationals preliminaries and finals receive no additional precision or importance. Higher score is better.",
    "external_bayesian_rank": "Finish-order Bayesian-style rating. A puzzler's expected percentile is computed from entrant ratings and mu moves toward the actual percentile. This view ranks by plain mu; higher is better.",
    "external_bayesian_conservative_rank": "The same Bayesian Skill state, ranked by mu minus two sigma. Because sigma shrinks with appearances, this view favors ratings supported by more event history; higher is better.",
    "external_nationals_rank": "Uses the log-time measurement model, with the most recent USA Nationals finish order enforced among that Nationals field while other placements retain their measurement-based slots. Higher score is better.",
}

PLOT_SYSTEMS = [
    ("jpar", "JPAR Rank", "jpar_out", "jpar"),
    ("raw_jpar_update", "Raw JPAR Update", "raw_jpar_update_out", "raw_jpar_update"),
    ("weighted_log_zscore", "Weighted Log Z Score", "weighted_log_zscore_out", "weighted_log_zscore"),
    ("mean_log_zscore", "Mean Log Z Score", "mean_log_zscore_out", "mean_log_zscore"),
    ("weighted_zscore", "Weighted Raw Z Score", "weighted_zscore_out", "weighted_zscore"),
    ("mean_zscore", "Mean Raw Z Score", "mean_zscore_out", "mean_zscore"),
    ("mean_adjusted_event_jpar", "Mean Adjusted JPAR", "mean_adjusted_event_jpar_out", "mean_adjusted_event_jpar"),
    ("weighted_event_percentile", "Weighted Percentile", "weighted_event_percentile_out", "weighted_event_percentile"),
    ("mean_event_percentile", "Mean Percentile", "mean_event_percentile_out", "mean_event_percentile"),
    ("weighted_normalized_rank", "Weighted Norm Rank", "weighted_normalized_rank_out", "weighted_normalized_rank"),
    ("mean_normalized_rank", "Mean Norm Rank", "mean_normalized_rank_out", "mean_normalized_rank"),
    ("sof_weighted_log_zscore", "SOF Weighted Log Z", "sof_weighted_log_zscore_out", "sof_weighted_log_zscore"),
    ("sof_mean_log_zscore", "SOF Mean Log Z", "sof_mean_log_zscore_out", "sof_mean_log_zscore"),
    ("sof_weighted_zscore", "SOF Weighted Raw Z", "sof_weighted_zscore_out", "sof_weighted_zscore"),
    ("sof_mean_zscore", "SOF Mean Raw Z", "sof_mean_zscore_out", "sof_mean_zscore"),
    ("sof_weighted_centered_log", "SOF Weighted Centered Log", "sof_weighted_centered_log_out", "sof_weighted_centered_log"),
    ("sof_mean_centered_log", "SOF Mean Centered Log", "sof_mean_centered_log_out", "sof_mean_centered_log"),
    ("external_logtime", "Log-Time Volatility (Mean)", "external_logtime", "external_logtime"),
    ("external_logtime_conservative", "Log-Time Volatility (Conservative)", "external_logtime_conservative", "external_logtime_conservative"),
    ("external_logtime_no_tier", "Log-Time Volatility — No Tier Boost", "external_logtime_no_tier", "external_logtime_no_tier"),
    ("external_bayesian", "Bayesian Skill Mu", "external_bayesian", "external_bayesian"),
    ("external_bayesian_conservative", "Bayesian Skill (Conservative)", "external_bayesian_conservative", "external_bayesian_conservative"),
    ("external_nationals", "Nationals-Constrained Log-Time", "external_nationals", "external_nationals"),
]

DEFAULT_PLOT_SYSTEM_KEYS = {
    "jpar",
    "raw_jpar_update",
    "weighted_log_zscore",
    "mean_log_zscore",
    "weighted_zscore",
    "mean_zscore",
}

SCATTER_RANK_SYSTEMS = [
    ("jpar_rank", "JPAR Rank"),
    ("raw_jpar_update_rank", "Raw JPAR Update"),
    ("weighted_log_zscore_rank", "Weighted Log Z Score"),
    ("mean_log_zscore_rank", "Mean Log Z Score"),
    ("weighted_zscore_rank", "Weighted Raw Z Score"),
    ("mean_zscore_rank", "Mean Raw Z Score"),
    ("external_logtime_rank", "Log-Time Volatility (Mean)"),
    ("external_logtime_conservative_rank", "Log-Time Volatility (Conservative)"),
    ("external_logtime_no_tier_rank", "Log-Time Volatility — No Tier Boost"),
    ("external_bayesian_rank", "Bayesian Skill Mu"),
    ("external_bayesian_conservative_rank", "Bayesian Skill (Conservative)"),
    ("external_nationals_rank", "Nationals-Constrained Log-Time"),
]

INDIVIDUAL_SYSTEMS = [
    ("jpar", "JPAR Rank", "jpar_out"),
    ("raw_jpar_update", "Raw JPAR Update", "raw_jpar_update_out"),
    ("weighted_log_zscore", "Weighted Log Z Score", "weighted_log_zscore_out"),
    ("mean_log_zscore", "Mean Log Z Score", "mean_log_zscore_out"),
    ("weighted_zscore", "Weighted Raw Z Score", "weighted_zscore_out"),
    ("mean_zscore", "Mean Raw Z Score", "mean_zscore_out"),
    ("sof_weighted_log_zscore", "SOF Weighted Log Z", "sof_weighted_log_zscore_out"),
    ("sof_mean_log_zscore", "SOF Mean Log Z", "sof_mean_log_zscore_out"),
    ("sof_weighted_zscore", "SOF Weighted Raw Z", "sof_weighted_zscore_out"),
    ("sof_mean_zscore", "SOF Mean Raw Z", "sof_mean_zscore_out"),
    ("sof_weighted_centered_log", "SOF Weighted Centered Log", "sof_weighted_centered_log_out"),
    ("sof_mean_centered_log", "SOF Mean Centered Log", "sof_mean_centered_log_out"),
    ("external_logtime", "Log-Time Volatility (Mean)", "external_logtime"),
    ("external_logtime_conservative", "Log-Time Volatility (Conservative)", "external_logtime_conservative"),
    ("external_logtime_no_tier", "Log-Time Volatility — No Tier Boost", "external_logtime_no_tier"),
    ("external_bayesian", "Bayesian Skill Mu", "external_bayesian"),
    ("external_bayesian_conservative", "Bayesian Skill (Conservative)", "external_bayesian_conservative"),
    ("external_nationals", "Nationals-Constrained Log-Time", "external_nationals"),
]

PROFILE_SYSTEMS = [
    ("jpar", "JPAR Rank", "jpar_out", "adjusted_event_jpar", "adjusted_ratio"),
    ("raw_jpar_update", "Raw JPAR Update", "raw_jpar_update_out", "event_jpar", "raw_ratio"),
    ("mean_adjusted_event_jpar", "Mean Adjusted JPAR", "mean_adjusted_event_jpar_out", "adjusted_event_jpar", "adjusted_ratio"),
    ("weighted_log_zscore", "Weighted Log Z Score", "weighted_log_zscore_out", "event_log_zscore", "log_z"),
    ("mean_log_zscore", "Mean Log Z Score", "mean_log_zscore_out", "event_log_zscore", "log_z"),
    ("weighted_zscore", "Weighted Raw Z Score", "weighted_zscore_out", "event_zscore", "raw_z"),
    ("mean_zscore", "Mean Raw Z Score", "mean_zscore_out", "event_zscore", "raw_z"),
    ("weighted_event_percentile", "Weighted Percentile", "weighted_event_percentile_out", "event_percentile", "quantile"),
    ("mean_event_percentile", "Mean Percentile", "mean_event_percentile_out", "event_percentile", "quantile"),
    ("weighted_normalized_rank", "Weighted Norm Rank", "weighted_normalized_rank_out", "event_normalized_rank", "quantile"),
    ("mean_normalized_rank", "Mean Norm Rank", "mean_normalized_rank_out", "event_normalized_rank", "quantile"),
    ("sof_weighted_log_zscore", "SOF Weighted Log Z", "sof_weighted_log_zscore_out", "sof_weighted_log_zscore_event", "sof_log_z"),
    ("sof_mean_log_zscore", "SOF Mean Log Z", "sof_mean_log_zscore_out", "sof_mean_log_zscore_event", "sof_log_z"),
    ("sof_weighted_zscore", "SOF Weighted Raw Z", "sof_weighted_zscore_out", "sof_weighted_zscore_event", "sof_raw_z"),
    ("sof_mean_zscore", "SOF Mean Raw Z", "sof_mean_zscore_out", "sof_mean_zscore_event", "sof_raw_z"),
    ("sof_weighted_centered_log", "SOF Weighted Centered Log", "sof_weighted_centered_log_out", "sof_weighted_centered_log_event", "sof_centered_log"),
    ("sof_mean_centered_log", "SOF Mean Centered Log", "sof_mean_centered_log_out", "sof_mean_centered_log_event", "sof_centered_log"),
]

EXTERNAL_PROFILE_SYSTEMS = [
    ("external_logtime", "Log-Time Volatility (Mean)"),
    ("external_logtime_conservative", "Log-Time Volatility (Conservative)"),
    ("external_logtime_no_tier", "Log-Time Volatility — No Tier Boost"),
    ("external_bayesian", "Bayesian Skill Mu"),
    ("external_bayesian_conservative", "Bayesian Skill (Conservative)"),
    ("external_nationals", "Nationals-Constrained Log-Time"),
]


def external_csv(name: str) -> pd.DataFrame:
    if EXTERNAL_OUTPUT_DIR is None:
        return pd.DataFrame()
    path = EXTERNAL_OUTPUT_DIR / name
    return pd.read_csv(path, low_memory=False, dtype={"event_id": "string", "member_key": "string"}) if path.exists() else pd.DataFrame()


def as_value(value: object, kind: str) -> object:
    if pd.isna(value):
        return None
    if kind in {"number", "integer", "rank"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return None
        if kind in {"integer", "rank"}:
            return int(numeric)
        return float(numeric)
    return str(value)


def keyify(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_calc_for_plots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"event_id": "string", "resolved_member_id": "string"})
    for col in ["completion_seconds", "event_mean_completion_seconds", "event_jpar", "adjusted_event_jpar", "jpar_out"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["member_key"] = df["resolved_member_id"].apply(keyify)
    df = df.dropna(subset=["event_date", "event_id", "completion_seconds"])
    df = df[df["completion_seconds"].gt(0)].copy()
    df = df.sort_values(["event_date", "event_id", "completion_seconds", "member_key"]).reset_index(drop=True)

    df["log_completion_seconds"] = np.log(df["completion_seconds"])
    stats = (
        df.groupby("event_id", dropna=False)
        .agg(
            event_std_seconds=("completion_seconds", "std"),
            event_mean_log_seconds=("log_completion_seconds", "mean"),
            event_std_log_seconds=("log_completion_seconds", "std"),
        )
        .reset_index()
    )
    df = df.merge(stats, on="event_id", how="left")
    df["event_zscore"] = (df["completion_seconds"] - df["event_mean_completion_seconds"]) / df["event_std_seconds"]
    df["event_log_zscore"] = (df["log_completion_seconds"] - df["event_mean_log_seconds"]) / df["event_std_log_seconds"]
    df["event_centered_log_time"] = df["log_completion_seconds"] - df["event_mean_log_seconds"]
    df.loc[df["event_std_seconds"].le(0) | df["event_std_seconds"].isna(), "event_zscore"] = np.nan
    df.loc[df["event_std_log_seconds"].le(0) | df["event_std_log_seconds"].isna(), "event_log_zscore"] = np.nan
    df["event_rank"] = df.groupby("event_id")["completion_seconds"].rank(method="average", ascending=True)
    df["event_percentile"] = df.groupby("event_id")["completion_seconds"].rank(pct=True, ascending=True)
    participant_count = pd.to_numeric(df.get("event_participant_count"), errors="coerce")
    if participant_count.isna().all():
        participant_count = df.groupby("event_id")["completion_seconds"].transform("size")
    df["event_normalized_rank"] = np.where(participant_count.gt(1), (df["event_rank"] - 1) / (participant_count - 1), np.nan)
    # Event-relative metrics use the full event field; only resolved people enter ranking state.
    ranked = add_plot_running_scores(df[df["member_key"].ne("")].copy())
    ranked = add_strength_of_field_zscores(ranked)
    grouped = ranked.groupby("member_key", sort=False)
    for source, target in [
        ("adjusted_event_jpar", "mean_adjusted_event_jpar_out"),
        ("event_percentile", "mean_event_percentile_out"),
        ("event_normalized_rank", "mean_normalized_rank_out"),
    ]:
        ranked[target] = grouped[source].expanding().mean().reset_index(level=0, drop=True).sort_index()
    return ranked


def add_plot_running_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    running: dict[str, dict[str, float | list[float]]] = {}
    values = {col: [] for col in ["raw_jpar_update_out", "weighted_log_zscore_out", "weighted_zscore_out", "weighted_event_percentile_out", "weighted_normalized_rank_out", "mean_log_zscore_out", "mean_zscore_out"]}
    for _, row in out.iterrows():
        key = row["member_key"]
        state = running.setdefault(
            key,
            {
                "raw_jpar_update_out": np.nan,
                "weighted_log_zscore_out": np.nan,
                "weighted_zscore_out": np.nan,
                "weighted_event_percentile_out": np.nan,
                "weighted_normalized_rank_out": np.nan,
                "log_values": [],
                "raw_z_values": [],
            },
        )
        for out_col, in_col in [
            ("raw_jpar_update_out", "event_jpar"),
            ("weighted_log_zscore_out", "event_log_zscore"),
            ("weighted_zscore_out", "event_zscore"),
            ("weighted_event_percentile_out", "event_percentile"),
            ("weighted_normalized_rank_out", "event_normalized_rank"),
        ]:
            value = row[in_col]
            previous = state[out_col]
            if pd.isna(value):
                values[out_col].append(np.nan)
            else:
                current = float(value) if pd.isna(previous) else (float(previous) + float(value)) / 2.0
                state[out_col] = current
                values[out_col].append(current)

        log_value = row["event_log_zscore"]
        raw_z_value = row["event_zscore"]
        if pd.notna(log_value):
            state["log_values"].append(float(log_value))
        if pd.notna(raw_z_value):
            state["raw_z_values"].append(float(raw_z_value))
        values["mean_log_zscore_out"].append(float(np.mean(state["log_values"])) if state["log_values"] else np.nan)
        values["mean_zscore_out"].append(float(np.mean(state["raw_z_values"])) if state["raw_z_values"] else np.nan)

    for col, col_values in values.items():
        out[col] = col_values
    return out


def add_strength_of_field_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Add online field-strength-adjusted raw/log Z-score histories.

    Each system uses only its entrants' incoming state. Missing/debut ratings use
    the neutral zero prior. All entrants in an event are frozen to the same
    pre-event field mean before any participant updates are applied.
    """
    out = df.copy()
    specs = {
        "sof_weighted_log_zscore": ("event_log_zscore", "half"),
        "sof_mean_log_zscore": ("event_log_zscore", "mean"),
        "sof_weighted_zscore": ("event_zscore", "half"),
        "sof_mean_zscore": ("event_zscore", "mean"),
        "sof_weighted_centered_log": ("event_centered_log_time", "half"),
        "sof_mean_centered_log": ("event_centered_log_time", "mean"),
    }
    states: dict[str, dict[str, float]] = {key: {} for key in specs}
    counts: dict[str, dict[str, int]] = {key: {} for key in specs}
    event_values = {f"{key}_event": pd.Series(np.nan, index=out.index, dtype="float64") for key in specs}
    post_values = {f"{key}_out": pd.Series(np.nan, index=out.index, dtype="float64") for key in specs}

    for (_, _), event_rows in out.groupby(["event_date", "event_id"], sort=True):
        member_keys = event_rows["member_key"].astype(str).tolist()
        for key, (source_col, update_kind) in specs.items():
            state = states[key]
            field_mean = float(np.mean([state.get(member_key, 0.0) for member_key in member_keys])) if member_keys else 0.0
            pending: list[tuple[int, str, float]] = []
            for index, row in event_rows.iterrows():
                source = row.get(source_col)
                if pd.isna(source):
                    continue
                member_key = str(row["member_key"])
                observation = float(source) + field_mean
                event_values[f"{key}_event"].loc[index] = observation
                pending.append((index, member_key, observation))
            for index, member_key, observation in pending:
                previous = state.get(member_key)
                if previous is None:
                    current = observation
                    counts[key][member_key] = 1
                elif update_kind == "half":
                    current = (previous + observation) / 2.0
                    counts[key][member_key] = counts[key].get(member_key, 1) + 1
                else:
                    count = counts[key].get(member_key, 1) + 1
                    current = previous + (observation - previous) / count
                    counts[key][member_key] = count
                state[member_key] = current
                post_values[f"{key}_out"].loc[index] = current

    for column, values in {**event_values, **post_values}.items():
        out[column] = values
    return out


def percentile_history(df: pd.DataFrame, value_col: str) -> list[dict[str, object]]:
    as_of: dict[str, float] = {}
    event_counts: dict[str, int] = {}
    rows = []
    events = df[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"])
    for _, event in events.iterrows():
        raw_event_rows = df[df["event_date"].eq(event["event_date"]) & df["event_id"].eq(event["event_id"])]
        for member_key in raw_event_rows["member_key"].dropna().unique():
            event_counts[member_key] = event_counts.get(member_key, 0) + 1
        event_rows = raw_event_rows.dropna(subset=[value_col])
        as_of.update(dict(zip(event_rows["member_key"], event_rows[value_col])))
        samples = [
            {"score": float(score), "events": int(event_counts.get(member_key, 0))}
            for member_key, score in as_of.items()
            if pd.notna(score)
        ]
        values = pd.Series([sample["score"] for sample in samples], dtype="float64").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "date": event["event_date"].strftime("%Y-%m-%d"),
                "event": str(event["event_id"]),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p50": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "samples": samples,
            }
        )
    return rows


def histogram(values: pd.Series, bins: int = 36) -> dict[str, list[float]]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {"bins": [], "counts": []}
    lo, hi = clean.quantile(0.01), clean.quantile(0.99)
    clipped = clean.clip(lo, hi)
    counts, edges = np.histogram(clipped, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).round(6)
    return {"bins": centers.tolist(), "counts": counts.astype(int).tolist()}


def score_samples(master: pd.DataFrame, score_col: str) -> list[dict[str, float | int]]:
    samples = []
    for _, row in master[[score_col, "events"]].dropna(subset=[score_col]).iterrows():
        samples.append({"score": float(row[score_col]), "events": int(row["events"])})
    return samples


def augment_master_with_strength_of_field_zscores(master: pd.DataFrame, calc_path: Path) -> pd.DataFrame:
    calc = load_calc_for_plots(calc_path)
    score_columns = [
        "sof_weighted_log_zscore",
        "sof_mean_log_zscore",
        "sof_weighted_zscore",
        "sof_mean_zscore",
        "sof_weighted_centered_log",
        "sof_mean_centered_log",
    ]
    history_columns = [f"{key}_out" for key in score_columns]
    final = calc.groupby("member_key", as_index=False)[history_columns].last()
    final = final.rename(columns={f"{key}_out": key for key in score_columns})
    out = master.copy()
    if "member_key" not in out:
        out["member_key"] = out["resolved_member_id"].apply(keyify)
    out = out.drop(columns=[column for column in score_columns if column in out], errors="ignore")
    out = out.drop(columns=[f"{key}_rank" for key in score_columns if f"{key}_rank" in out], errors="ignore")
    out = out.merge(final, on="member_key", how="left")
    for key in score_columns:
        out[f"{key}_rank"] = pd.to_numeric(out[key], errors="coerce").rank(method="min", ascending=True)
    return out


def build_plot_data(master: pd.DataFrame, calc_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    calc = load_calc_for_plots(calc_path)
    external_history = external_csv("state_history.csv")
    drift = {}
    hists = {}
    for key, label, history_col, score_col in PLOT_SYSTEMS:
        if f"{key}_rank" in DISABLED_SYSTEM_RANK_KEYS:
            continue
        if EXTERNAL_OUTPUT_DIR is None and key.startswith("external_"):
            continue
        if key.startswith("external_"):
            rows = []
            if not external_history.empty and history_col in external_history:
                for (event_date, event_id), snapshot in external_history.groupby(["event_date", "event_id"], sort=False):
                    values = pd.to_numeric(snapshot[history_col], errors="coerce").dropna()
                    samples = [
                        {"score": float(row[history_col]), "events": int(row["events"])}
                        for _, row in snapshot.dropna(subset=[history_col]).iterrows()
                    ]
                    if not values.empty:
                        rows.append({
                            "date": str(event_date), "event": str(event_id),
                            "p10": float(values.quantile(.10)), "p25": float(values.quantile(.25)),
                            "p50": float(values.quantile(.50)), "p75": float(values.quantile(.75)),
                            "p90": float(values.quantile(.90)), "samples": samples,
                        })
            drift[key] = {"label": label, "rows": rows}
        else:
            drift[key] = {"label": label, "rows": percentile_history(calc, history_col)}
        if score_col in master.columns:
            hists[key] = {"label": label, **histogram(master[score_col]), "samples": score_samples(master, score_col)}
    return drift, hists


def ordinal_ranks(values: dict[str, float]) -> dict[str, int]:
    """Return lower-is-better ranks for a snapshot of active score values."""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return {member_key: position for position, (member_key, _) in enumerate(ordered, start=1)}


def ranks_descending(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return {member_key: position for position, (member_key, _) in enumerate(ordered, start=1)}


def build_individual_data(calc_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Reconstruct every ranking system after each event for person-level audit views."""
    active_individual_systems = [
        system for system in INDIVIDUAL_SYSTEMS
        if f"{system[0]}_rank" not in DISABLED_SYSTEM_RANK_KEYS
        and (EXTERNAL_OUTPUT_DIR is not None or not system[0].startswith("external_"))
    ]
    calc = load_calc_for_plots(calc_path)
    impact_source = pd.read_csv(calc_path, low_memory=False, dtype={"event_id": "string"})
    impact_source["event_date"] = pd.to_datetime(impact_source["event_date"], errors="coerce")
    required = ["event_jpar", "adjusted_event_jpar", "mean_expected_event_average", "event_mean_completion_seconds"]
    for col in required:
        calc[col] = pd.to_numeric(calc.get(col), errors="coerce")
        impact_source[col] = pd.to_numeric(impact_source.get(col), errors="coerce")
    impact_source["expected_event_average"] = pd.to_numeric(impact_source.get("expected_event_average"), errors="coerce")

    people: dict[str, dict[str, object]] = {}
    as_of = {key: {} for key, _, _ in active_individual_systems}
    events = calc[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"])
    impact_rows: list[dict[str, object]] = []

    for _, event in events.iterrows():
        event_rows = calc[calc["event_date"].eq(event["event_date"]) & calc["event_id"].eq(event["event_id"])].copy()
        event_rows = event_rows.sort_values(["completion_seconds", "member_key"])
        for key, _, score_col in active_individual_systems:
            if score_col not in event_rows:
                continue
            usable = event_rows.dropna(subset=[score_col])
            as_of[key].update(dict(zip(usable["member_key"], usable[score_col])))
        ranks = {key: ordinal_ranks(values) for key, values in as_of.items()}

        placement = event_rows["completion_seconds"].rank(method="min", ascending=True)
        for idx, row in event_rows.iterrows():
            member_key = row["member_key"]
            person = people.setdefault(
                member_key,
                {"name": str(row.get("full_name") or row.get("member_full_name") or member_key), "events": [], "trends": {key: [] for key, _, _ in active_individual_systems}},
            )
            rank_values = {key: ranks[key].get(member_key) for key, _, _ in active_individual_systems}
            record = {
                "date": event["event_date"].strftime("%Y-%m-%d"),
                "event": str(event["event_id"]),
                "event_name": str(row.get("event_name") or event["event_id"]),
                "place": int(placement.loc[idx]) if pd.notna(placement.loc[idx]) else None,
                "time_seconds": float(row["completion_seconds"]),
                "ranks": rank_values,
            }
            person["events"].append(record)
            for key, _, score_col in active_individual_systems:
                score = row.get(score_col)
                rank = rank_values[key]
                if pd.notna(score) and rank is not None:
                    person["trends"][key].append({"date": record["date"], "event": record["event"], "score": float(score), "rank": rank})

    for (event_date, event_id), event_rows in impact_source.dropna(subset=["event_date", "event_id"]).groupby(["event_date", "event_id"], sort=True):
        qualified = event_rows.dropna(subset=["event_jpar"])
        raw_mean = qualified["event_mean_completion_seconds"].dropna().iloc[0] if qualified["event_mean_completion_seconds"].notna().any() else np.nan
        expected_mean = qualified["mean_expected_event_average"].dropna().iloc[0] if qualified["mean_expected_event_average"].notna().any() else np.nan
        multiplier = raw_mean / expected_mean if pd.notna(raw_mean) and pd.notna(expected_mean) and expected_mean > 0 else np.nan
        delta = qualified["adjusted_event_jpar"] - qualified["event_jpar"]
        impact_rows.append({
            "date": event_date.strftime("%Y-%m-%d"),
            "event": str(event_id),
            "event_name": str(event_rows["event_name"].dropna().iloc[0]) if event_rows["event_name"].notna().any() else str(event_id),
            "participants": int(len(qualified)),
            "returning_anchors": int(qualified["expected_event_average"].notna().sum()),
            "raw_mean_seconds": float(raw_mean) if pd.notna(raw_mean) else None,
            "anchor_implied_mean_seconds": float(expected_mean) if pd.notna(expected_mean) else None,
            "calibration_multiplier": float(multiplier) if pd.notna(multiplier) else None,
            "median_jpar_shift": float(delta.median()) if delta.notna().any() else None,
        })
    external_history = external_csv("state_history.csv")
    if not external_history.empty:
        event_records = {
            (member_key, str(record["event"])): record
            for member_key, person in people.items()
            for record in person["events"]
        }
        for (_, event_id), snapshot in external_history.groupby(["event_date", "event_id"], sort=False):
            for key, _, score_col in active_individual_systems:
                if not key.startswith("external_") or score_col not in snapshot:
                    continue
                scores = pd.to_numeric(snapshot.set_index("member_key")[score_col], errors="coerce").dropna().to_dict()
                ranks = ranks_descending(scores)
                for member_key, score in scores.items():
                    member_key = str(member_key)
                    if member_key not in people:
                        continue
                    people[member_key]["trends"].setdefault(key, []).append({
                        "date": str(snapshot["event_date"].iloc[0]), "event": str(event_id),
                        "score": float(score), "rank": ranks[member_key],
                    })
                    record = event_records.get((member_key, str(event_id)))
                    if record is not None:
                        record["ranks"][key] = ranks[member_key]
    return people, impact_rows


def build_ranking_timeline(calc_path: Path) -> list[dict[str, object]]:
    """Build compact per-event score updates used to reconstruct historical leaderboards."""
    calc = load_calc_for_plots(calc_path)
    numeric_cols = ["adjusted_event_jpar", "jpar_out"]
    for col in numeric_cols:
        calc[col] = pd.to_numeric(calc.get(col), errors="coerce")
    calc = calc.sort_values(["event_date", "event_id", "completion_seconds", "member_key"]).copy()
    grouped = calc.groupby("member_key", sort=False)
    calc["events_asof"] = grouped.cumcount() + 1
    for source, target in [
        ("completion_seconds", "mean_completion_seconds_asof"),
        ("adjusted_event_jpar", "mean_adjusted_event_jpar"),
        ("event_log_zscore", "mean_log_zscore"),
        ("event_zscore", "mean_zscore"),
        ("event_percentile", "mean_event_percentile"),
        ("event_normalized_rank", "mean_normalized_rank"),
    ]:
        calc[target] = grouped[source].expanding().mean().reset_index(level=0, drop=True).sort_index()
    calc["best_event_percentile_asof"] = grouped["event_percentile"].cummin()
    calc["msp_like_score"] = (
        1000
        - 115 * calc["mean_log_zscore"]
        - 35 * np.log1p(calc["events_asof"])
        - 20 * (1 - calc["best_event_percentile_asof"])
    )

    merge_keys = ["event_date", "event_id", "member_key"]
    # Elo and TrueSkill remain implemented in build_ranking_diagnostics_report.py,
    # but their timeline work is disabled while those systems are hidden.
    if not {"elo_rating_rank", "trueskill_conservative_rank"}.issubset(DISABLED_SYSTEM_RANK_KEYS):
        from build_ranking_diagnostics_report import compute_elo, compute_trueskill

        _, elo_history = compute_elo(calc)
        _, trueskill_history = compute_trueskill(calc)
        if not elo_history.empty:
            calc = calc.merge(elo_history[merge_keys + ["elo_rating_out"]], on=merge_keys, how="left")
        if not trueskill_history.empty:
            calc = calc.merge(trueskill_history[merge_keys + ["trueskill_conservative_out"]], on=merge_keys, how="left")

    score_columns = {
        "jpar": "jpar_out",
        "raw_jpar_update": "raw_jpar_update_out",
        "mean_adjusted_event_jpar": "mean_adjusted_event_jpar",
        "mean_log_zscore": "mean_log_zscore",
        "mean_zscore": "mean_zscore",
        "weighted_log_zscore": "weighted_log_zscore_out",
        "weighted_zscore": "weighted_zscore_out",
        "weighted_event_percentile": "weighted_event_percentile_out",
        "mean_event_percentile": "mean_event_percentile",
        "weighted_normalized_rank": "weighted_normalized_rank_out",
        "mean_normalized_rank": "mean_normalized_rank",
        "sof_weighted_log_zscore": "sof_weighted_log_zscore_out",
        "sof_mean_log_zscore": "sof_mean_log_zscore_out",
        "sof_weighted_zscore": "sof_weighted_zscore_out",
        "sof_mean_zscore": "sof_mean_zscore_out",
        "sof_weighted_centered_log": "sof_weighted_centered_log_out",
        "sof_mean_centered_log": "sof_mean_centered_log_out",
        "elo_rating": "elo_rating_out",
        "trueskill_conservative": "trueskill_conservative_out",
        "msp_like_score": "msp_like_score",
    }
    score_columns = {key: value for key, value in score_columns.items() if f"{key}_rank" not in DISABLED_SYSTEM_RANK_KEYS}
    timeline = []
    for (event_date, event_id), event_rows in calc.groupby(["event_date", "event_id"], sort=True):
        updates = []
        for _, row in event_rows.iterrows():
            mean_seconds = row["mean_completion_seconds_asof"]
            hours, remainder = divmod(int(round(mean_seconds)), 3600)
            minutes, seconds = divmod(remainder, 60)
            update = {
                "_member_key": row["member_key"],
                "_state_only": False,
                "full_name": str(row.get("full_name") or row.get("member_full_name") or row["member_key"]),
                "events": int(row["events_asof"]),
                "mean_time": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                "event_place": float(row["event_rank"]) if pd.notna(row["event_rank"]) else None,
                "event_time_seconds": float(row["completion_seconds"]),
            }
            for output_key, source_key in score_columns.items():
                value = row.get(source_key)
                update[output_key] = float(value) if pd.notna(value) else None
            updates.append(update)
        event_name = event_rows["event_name"].dropna().iloc[0] if "event_name" in event_rows and event_rows["event_name"].notna().any() else event_id
        timeline.append({"date": event_date.strftime("%Y-%m-%d"), "event": str(event_id), "event_name": str(event_name), "updates": updates})

    external_history = external_csv("state_history.csv")
    if not external_history.empty:
        by_event = {str(event["event"]): event for event in timeline}
        for (_, event_id), snapshot in external_history.groupby(["event_date", "event_id"], sort=False):
            event = by_event.get(str(event_id))
            if event is None:
                continue
            updates = {str(update["_member_key"]): update for update in event["updates"]}
            for _, row in snapshot.iterrows():
                member_key = keyify(row["member_key"])
                update = updates.get(member_key)
                if update is None:
                    update = {
                        "_member_key": member_key,
                        "_state_only": True,
                        "full_name": str(row.get("full_name") or member_key),
                    }
                    event["updates"].append(update)
                    updates[member_key] = update
                for key, _ in EXTERNAL_SYSTEMS:
                    value = row.get(key)
                    update[key] = float(value) if pd.notna(value) else None
    return timeline


def build_puzzler_profiles(calc_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Build leakage-tolerant person diagnostics from pre-event system states."""
    calc = load_calc_for_plots(calc_path)
    for col in ["mean_expected_event_average", "event_mean_completion_seconds"]:
        calc[col] = pd.to_numeric(calc.get(col), errors="coerce")

    people: dict[str, dict[str, object]] = {}
    event_payloads: dict[str, dict[str, object]] = {}
    as_of: dict[str, dict[str, float]] = {key: {} for key, *_ in PROFILE_SYSTEMS}
    events = calc[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"])

    for _, event in events.iterrows():
        event_rows = calc[
            calc["event_date"].eq(event["event_date"]) & calc["event_id"].eq(event["event_id"])
        ].copy()
        event_rows = event_rows.sort_values(["completion_seconds", "member_key"])
        if event_rows.empty:
            continue

        event_id = str(event["event_id"])
        event_name = str(event_rows["event_name"].dropna().iloc[0]) if event_rows["event_name"].notna().any() else event_id
        times = event_rows["completion_seconds"].dropna().astype(float).sort_values().to_numpy()
        raw_mean = float(times.mean()) if len(times) else np.nan
        log_times = np.log(times) if len(times) else np.array([], dtype=float)
        log_mean = float(log_times.mean()) if len(log_times) else np.nan
        raw_sd = float(times.std(ddof=1)) if len(times) > 1 else np.nan
        log_sd = float(log_times.std(ddof=1)) if len(log_times) > 1 else np.nan
        anchor_values = event_rows["mean_expected_event_average"].dropna()
        anchor_mean = float(anchor_values.iloc[0]) if not anchor_values.empty else raw_mean
        event_payloads[event_id] = {
            "date": event["event_date"].strftime("%Y-%m-%d"),
            "name": event_name,
            "times": [float(value) for value in times],
            "mean": raw_mean if pd.notna(raw_mean) else None,
            "median": float(np.median(times)) if len(times) else None,
            "anchor_mean": anchor_mean if pd.notna(anchor_mean) else None,
        }

        entrant_keys = [str(value) for value in event_rows["member_key"] if value]
        profile_field_means = {
            key: float(np.mean([as_of[key].get(member_key, 0.0) for member_key in entrant_keys])) if entrant_keys else 0.0
            for key, *_ in PROFILE_SYSTEMS
            if key.startswith("sof_")
        }

        pending_records: list[tuple[str, dict[str, object], dict[str, object]]] = []
        for _, row in event_rows.iterrows():
            member_key = row["member_key"]
            if not member_key:
                continue
            systems: dict[str, object] = {}
            for key, _, _, actual_col, conversion in PROFILE_SYSTEMS:
                incoming = as_of[key].get(member_key)
                actual_score = row.get(actual_col)
                predicted_time = np.nan
                if incoming is not None:
                    if conversion == "adjusted_ratio" and pd.notna(anchor_mean):
                        predicted_time = incoming * anchor_mean
                    elif conversion == "raw_ratio" and pd.notna(raw_mean):
                        predicted_time = incoming * raw_mean
                    elif conversion == "log_z" and pd.notna(log_mean) and pd.notna(log_sd):
                        predicted_time = float(np.exp(log_mean + incoming * log_sd))
                    elif conversion == "raw_z" and pd.notna(raw_mean) and pd.notna(raw_sd):
                        predicted_time = raw_mean + incoming * raw_sd
                    elif conversion == "quantile" and len(times):
                        predicted_time = float(np.quantile(times, np.clip(incoming, 0, 1)))
                    elif conversion == "sof_log_z" and pd.notna(log_mean) and pd.notna(log_sd):
                        event_z = incoming - profile_field_means.get(key, 0.0)
                        predicted_time = float(np.exp(log_mean + event_z * log_sd))
                    elif conversion == "sof_raw_z" and pd.notna(raw_mean) and pd.notna(raw_sd):
                        event_z = incoming - profile_field_means.get(key, 0.0)
                        predicted_time = raw_mean + event_z * raw_sd
                    elif conversion == "sof_centered_log" and pd.notna(log_mean):
                        centered_log = incoming - profile_field_means.get(key, 0.0)
                        predicted_time = float(np.exp(log_mean + centered_log))
                    elif conversion in {"log_z", "raw_z", "sof_log_z", "sof_raw_z", "sof_centered_log"} and len(times):
                        # Degenerate fields have no usable spread; the event median is
                        # the only defensible time-scale fallback for this diagnostic.
                        predicted_time = float(np.median(times))
                systems[key] = {
                    "predicted_score": float(incoming) if incoming is not None else None,
                    "actual_score": float(actual_score) if pd.notna(actual_score) else None,
                    "predicted_time": float(max(1, predicted_time)) if pd.notna(predicted_time) else None,
                    "predicted_rank": None,
                    "actual_rank": None,
                }

            record = {
                "date": event["event_date"].strftime("%Y-%m-%d"),
                "event": event_id,
                "event_name": event_name,
                "actual_time": float(row["completion_seconds"]),
                "place": int(row["event_rank"]) if pd.notna(row["event_rank"]) else None,
                "systems": systems,
            }
            person = people.setdefault(
                member_key,
                {"name": str(row.get("full_name") or row.get("member_full_name") or member_key), "events": []},
            )
            pending_records.append((member_key, person, record))

        for key, *_ in PROFILE_SYSTEMS:
            eligible = [
                (member_key, record)
                for member_key, _, record in pending_records
                if record["systems"][key]["predicted_score"] is not None
            ]
            predicted = pd.Series(
                {member_key: record["systems"][key]["predicted_score"] for member_key, record in eligible},
                dtype="float64",
            ).rank(method="min", ascending=True)
            actual = pd.Series(
                {member_key: record["actual_time"] for member_key, record in eligible},
                dtype="float64",
            ).rank(method="min", ascending=True)
            for member_key, record in eligible:
                record["systems"][key]["predicted_rank"] = int(predicted[member_key])
                record["systems"][key]["actual_rank"] = int(actual[member_key])

        for _, person, record in pending_records:
            person["events"].append(record)

        for key, _, post_col, _, _ in PROFILE_SYSTEMS:
            usable = event_rows.dropna(subset=[post_col])
            as_of[key].update({str(member): float(score) for member, score in zip(usable["member_key"], usable[post_col])})

    external_predictions = external_csv("rolling_predictions.csv")
    external_history = external_csv("state_history.csv")
    if not external_predictions.empty:
        record_index = {
            (member_key, str(record["event"])): record
            for member_key, person in people.items()
            for record in person["events"]
        }
        post_scores = {}
        if not external_history.empty:
            for _, row in external_history.iterrows():
                for key, _ in EXTERNAL_SYSTEMS:
                    post_scores[(keyify(row["member_key"]), str(row["event_id"]), key)] = row.get(key)
        for _, row in external_predictions.iterrows():
            member_key, event_id, key = keyify(row["member_key"]), str(row["event_id"]), str(row["system"])
            record = record_index.get((member_key, event_id))
            if record is None:
                continue
            actual_score = post_scores.get((member_key, event_id, key))
            record["systems"][key] = {
                "predicted_score": float(row["incoming_score"]),
                "actual_score": float(actual_score) if pd.notna(actual_score) else None,
                "predicted_time": float(row["predicted_time_diagnostic"]) if pd.notna(row["predicted_time_diagnostic"]) else None,
                "predicted_rank": int(row["predicted_rank"]),
                "actual_rank": int(row["actual_rank_common_cohort"]),
            }
        for person in people.values():
            for record in person["events"]:
                for key, _ in EXTERNAL_PROFILE_SYSTEMS:
                    record["systems"].setdefault(key, {
                        "predicted_score": None, "actual_score": None, "predicted_time": None,
                        "predicted_rank": None, "actual_rank": None,
                    })
    return people, event_payloads


def build_cumulative_calibration_data(calc_path: Path, master: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize the calibration component of each person's JPAR history."""
    calc = pd.read_csv(calc_path, low_memory=False, dtype={"resolved_member_id": "string"})
    for col in ["event_jpar", "adjusted_event_jpar", "mean_expected_event_average"]:
        calc[col] = pd.to_numeric(calc.get(col), errors="coerce")
    calc["member_key"] = calc["resolved_member_id"].apply(keyify)
    calc = calc[calc["member_key"].ne("")].copy()
    calc["event_delta"] = calc["adjusted_event_jpar"] - calc["event_jpar"]
    grouped = (
        calc.groupby("member_key", as_index=False)
        .agg(
            full_name=("full_name", "last"),
            total_events=("event_id", "nunique"),
            calibrated_events=("mean_expected_event_average", lambda s: int(s.notna().sum())),
            cumulative_event_delta=("event_delta", "sum"),
            mean_event_delta=("event_delta", "mean"),
        )
    )
    finals = master[["resolved_member_id", "jpar", "raw_jpar_update"]].copy()
    finals["member_key"] = finals["resolved_member_id"].apply(keyify)
    finals["final_calibration_effect"] = finals["jpar"] - finals["raw_jpar_update"]
    grouped = grouped.merge(finals[["member_key", "jpar", "final_calibration_effect"]], on="member_key", how="left")
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["final_calibration_effect"])
    return [
        {
            "name": str(row["full_name"]),
            "events": int(row["total_events"]),
            "calibrated_events": int(row["calibrated_events"]),
            "jpar": float(row["jpar"]),
            "cumulative_event_delta": float(row["cumulative_event_delta"]),
            "mean_event_delta": float(row["mean_event_delta"]),
            "final_calibration_effect": float(row["final_calibration_effect"]),
        }
        for _, row in grouped.sort_values("final_calibration_effect").iterrows()
    ]


def build_calculation_overview(calc_path: Path, input_path: Path) -> list[dict[str, object]]:
    """Return the full prepared input with a flag for rows that received a JPAR update."""
    source = pd.read_csv(input_path, low_memory=False, dtype={"event_id": "string", "event_row_id": "string"})
    calc = pd.read_csv(calc_path, low_memory=False, dtype={"event_id": "string", "resolved_member_id": "string", "event_row_id": "string"})
    calc["event_date"] = pd.to_datetime(calc["event_date"], errors="coerce")
    for col in ["completion_seconds", "event_jpar", "adjusted_event_jpar", "previous_jpar", "jpar_out", "event_participant_count"]:
        calc[col] = pd.to_numeric(calc.get(col), errors="coerce")
    calc["member_key"] = calc["resolved_member_id"].apply(keyify)
    included = calc.dropna(subset=["event_date", "event_id", "completion_seconds", "jpar_out"])
    included = included[included["member_key"].ne("") & included["completion_seconds"].gt(0)].copy()
    included["place"] = included.groupby("event_id")["completion_seconds"].rank(method="min", ascending=True)
    cols = ["event_row_id", "place", "completion_seconds", "event_participant_count", "event_jpar", "adjusted_event_jpar", "previous_jpar", "jpar_out"]
    source = source.merge(included[cols], on="event_row_id", how="left")
    source["event_date"] = pd.to_datetime(source["event_date"], errors="coerce")
    source["completion_time_seconds"] = pd.to_numeric(source["completion_time_seconds"], errors="coerce")
    source = source.sort_values(["event_date", "event_id", "completion_time_seconds"], na_position="last")
    return [
        {
            "date": row["event_date"].strftime("%Y-%m-%d") if pd.notna(row["event_date"]) else "",
            "event": str(row["event_id"]),
            "event_name": str(row.get("event_name") or row["event_id"]),
            "name": str(row.get("full_name") or row.get("member_full_name") or row.get("member_id") or ""),
            "included": pd.notna(row["jpar_out"]),
            "place": int(row["place"]) if pd.notna(row["place"]) else None,
            "time_seconds": float(row["completion_seconds"]) if pd.notna(row["completion_seconds"]) else (float(row["completion_time_seconds"]) if pd.notna(row["completion_time_seconds"]) else None),
            "players": int(row["event_participant_count"]) if pd.notna(row["event_participant_count"]) else None,
            "event_jpar": float(row["event_jpar"]) if pd.notna(row["event_jpar"]) else None,
            "adjusted_event_jpar": float(row["adjusted_event_jpar"]) if pd.notna(row["adjusted_event_jpar"]) else None,
            "previous_jpar": float(row["previous_jpar"]) if pd.notna(row["previous_jpar"]) else None,
            "jpar_out": float(row["jpar_out"]) if pd.notna(row["jpar_out"]) else None,
        }
        for _, row in source.iterrows()
    ]


def build_feedback_data(calc_path: Path) -> list[dict[str, object]]:
    """Trace how initially unanchored entrants later become JPAR calibration anchors."""
    calc = pd.read_csv(calc_path, low_memory=False, dtype={"event_id": "string", "resolved_member_id": "string"})
    calc["event_date"] = pd.to_datetime(calc["event_date"], errors="coerce")
    for col in ["completion_seconds", "event_jpar", "adjusted_event_jpar", "previous_jpar", "jpar_out", "event_mean_completion_seconds", "mean_expected_event_average"]:
        calc[col] = pd.to_numeric(calc.get(col), errors="coerce")
    calc["member_key"] = calc["resolved_member_id"].apply(keyify)
    calc = calc.dropna(subset=["event_date", "event_id", "jpar_out"])
    calc = calc[calc["member_key"].ne("")].copy()
    events = calc[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"]).reset_index(drop=True)
    event_index = {(row.event_date, row.event_id): index for index, row in events.iterrows()}
    calc["event_index"] = [event_index[(row.event_date, row.event_id)] for _, row in calc.iterrows()]
    calc = calc.sort_values(["event_index", "member_key"])

    future_anchors: set[str] = set()
    propagated_by_event: dict[int, int] = {}
    for index in reversed(range(len(events))):
        event_rows = calc[calc["event_index"].eq(index)]
        entrants = event_rows[event_rows["previous_jpar"].isna()]["member_key"].unique()
        propagated_by_event[index] = sum(member in future_anchors for member in entrants)
        future_anchors.update(event_rows.loc[event_rows["previous_jpar"].notna(), "member_key"].unique())

    rows = []
    as_of_jpar: dict[str, float] = {}
    for index, event in events.iterrows():
        event_rows = calc[calc["event_index"].eq(index)]
        as_of_jpar.update(dict(zip(event_rows["member_key"], event_rows["jpar_out"])))
        raw_mean = event_rows["event_mean_completion_seconds"].dropna().iloc[0] if event_rows["event_mean_completion_seconds"].notna().any() else np.nan
        expected_mean = event_rows["mean_expected_event_average"].dropna().iloc[0] if event_rows["mean_expected_event_average"].notna().any() else np.nan
        multiplier = raw_mean / expected_mean if pd.notna(raw_mean) and pd.notna(expected_mean) and expected_mean > 0 else np.nan
        delta = event_rows["adjusted_event_jpar"] - event_rows["event_jpar"]
        rows.append({
            "date": event.event_date.strftime("%Y-%m-%d"),
            "event": str(event.event_id),
            "event_name": str(event_rows["event_name"].dropna().iloc[0]) if event_rows["event_name"].notna().any() else str(event.event_id),
            "participants": int(event_rows["member_key"].nunique()),
            "anchors": int(event_rows["previous_jpar"].notna().sum()),
            "entrants": int(event_rows["previous_jpar"].isna().sum()),
            "entrants_later_anchor": int(propagated_by_event[index]),
            "multiplier": float(multiplier) if pd.notna(multiplier) else None,
            "raw_median": float(event_rows["event_jpar"].median()) if event_rows["event_jpar"].notna().any() else None,
            "adjusted_median": float(event_rows["adjusted_event_jpar"].median()) if event_rows["adjusted_event_jpar"].notna().any() else None,
            "as_of_jpar_median": float(pd.Series(list(as_of_jpar.values()), dtype="float64").median()) if as_of_jpar else None,
            "median_delta": float(delta.median()) if delta.notna().any() else None,
        })
    return rows


def build_html(df: pd.DataFrame, output_path: Path, calc_path: Path) -> None:
    cols = [(key, label, kind) for key, label, kind in DISPLAY_COLUMNS if key in df.columns and key not in DISABLED_SYSTEM_RANK_KEYS]
    rows = []
    for _, row in df.iterrows():
        item = {}
        for key, label, kind in cols:
            item[key] = as_value(row[key], kind)
        item["_member_key"] = keyify(row.get("resolved_member_id"))
        rows.append(item)

    labels = [{"key": key, "label": label, "kind": kind} for key, label, kind in cols]
    descriptions = {
        key: SYSTEM_DESCRIPTIONS[key]
        for key, _, kind in cols
        if kind == "rank" and key in SYSTEM_DESCRIPTIONS
    }
    data_json = json.dumps(rows, ensure_ascii=False)
    columns_json = json.dumps(labels, ensure_ascii=False)
    descriptions_json = json.dumps(descriptions, ensure_ascii=False)
    drift_data, hist_data = build_plot_data(df, calc_path)
    individual_data, impact_data = build_individual_data(calc_path)
    ranking_timeline = build_ranking_timeline(calc_path)
    puzzler_profiles, puzzler_events = build_puzzler_profiles(calc_path)
    cumulative_calibration_data = build_cumulative_calibration_data(calc_path, df)
    calculation_overview_data = build_calculation_overview(calc_path, Path("data/data_event_results/source_of_truth_jpar_input.csv"))
    feedback_data = build_feedback_data(calc_path)
    cumulative_jpar_max = 4.0
    drift_json = json.dumps(drift_data, ensure_ascii=False)
    hist_json = json.dumps(hist_data, ensure_ascii=False)
    individual_json = json.dumps(individual_data, ensure_ascii=False)
    ranking_timeline_json = json.dumps(ranking_timeline, ensure_ascii=False)
    puzzler_profiles_json = json.dumps(puzzler_profiles, ensure_ascii=False)
    puzzler_events_json = json.dumps(puzzler_events, ensure_ascii=False)
    impact_json = json.dumps(impact_data, ensure_ascii=False)
    cumulative_calibration_json = json.dumps(cumulative_calibration_data, ensure_ascii=False)
    calculation_overview_json = json.dumps(calculation_overview_data, ensure_ascii=False)
    feedback_json = json.dumps(feedback_data, ensure_ascii=False)
    active_individual_systems = [
        system for system in INDIVIDUAL_SYSTEMS
        if f"{system[0]}_rank" not in DISABLED_SYSTEM_RANK_KEYS
        and (EXTERNAL_OUTPUT_DIR is not None or not system[0].startswith("external_"))
    ]
    active_profile_systems = [
        system for system in PROFILE_SYSTEMS
        if f"{system[0]}_rank" not in DISABLED_SYSTEM_RANK_KEYS
    ]
    active_external_profiles = [
        system for system in EXTERNAL_PROFILE_SYSTEMS
        if EXTERNAL_OUTPUT_DIR is not None and f"{system[0]}_rank" not in DISABLED_SYSTEM_RANK_KEYS
    ]
    individual_systems_json = json.dumps([{"key": key, "label": label} for key, label, _ in active_individual_systems])
    profile_systems_json = json.dumps(
        [{"key": key, "label": label} for key, label, *_ in active_profile_systems]
        + [{"key": key, "label": label} for key, label in active_external_profiles]
    )
    scatter_columns = [
        {"key": key, "label": label}
        for key, label, kind in cols
        if kind == "rank" and key != "jpar_rank"
    ]
    scatter_columns_json = json.dumps(scatter_columns, ensure_ascii=False)
    default_plot_systems_json = json.dumps(sorted(DEFAULT_PLOT_SYSTEM_KEYS))
    available_rank_keys = {column["key"] for column in scatter_columns}
    default_scatter_systems_json = json.dumps([
        key for key, _ in SCATTER_RANK_SYSTEMS
        if key != "jpar_rank" and key not in DISABLED_SYSTEM_RANK_KEYS and key in available_rank_keys
    ])

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JPAR and Ranking Exploration</title>
  <style>
    :root {{
      --border: #d7dce2;
      --text: #111827;
      --muted: #6b7280;
      --header: #f3f4f6;
      --row: #ffffff;
      --row-alt: #f9fafb;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #ffffff;
    }}
    main {{
      padding: 0 20px 28px;
      max-width: 1800px;
      margin: 0 auto;
    }}
    .page-header {{
      padding: 22px 0 15px;
    }}
    .page-header h1 {{
      margin-bottom: 5px;
      font-size: 26px;
    }}
    .tab-bar {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      gap: 4px;
      overflow-x: auto;
      margin: 0 -20px 22px;
      padding: 8px 20px 0;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
    }}
    .tab-button {{
      flex: 0 0 auto;
      margin-bottom: -1px;
      padding: 10px 14px 11px;
      border: 1px solid transparent;
      border-bottom: 2px solid transparent;
      border-radius: 7px 7px 0 0;
      color: var(--muted);
      font-weight: 650;
    }}
    .tab-button:hover {{
      color: var(--text);
      background: #f8fafc;
    }}
    .tab-button[aria-selected="true"] {{
      color: #1d4ed8;
      border-bottom-color: #2563eb;
      background: #eff6ff;
    }}
    .tab-panel[hidden] {{
      display: none;
    }}
    .tab-intro {{
      margin: -4px 0 18px;
      color: var(--muted);
      line-height: 1.45;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 24px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }}
    input, select, button {{
      font: inherit;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      padding: 7px 9px;
    }}
    input {{
      min-width: 260px;
    }}
    button {{
      cursor: pointer;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-left: auto;
    }}
    details {{
      border: 1px solid var(--border);
      border-radius: 6px;
      margin: 0 0 12px;
      background: #fbfdff;
    }}
    summary {{
      cursor: pointer;
      padding: 9px 11px;
      font-weight: 650;
    }}
    .explain {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 8px 14px;
      padding: 0 11px 12px;
      font-size: 13px;
      line-height: 1.35;
    }}
    .explain-item {{
      border-top: 1px solid #edf0f3;
      padding-top: 8px;
    }}
    .explain-name {{
      font-weight: 700;
      margin-bottom: 3px;
    }}
    .table-wrap {{
      border: 1px solid var(--border);
      overflow: auto;
      max-height: calc(100vh - 118px);
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .plot-card {{
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 6px;
      padding: 8px;
    }}
    .plot-title {{
      font-weight: 700;
      font-size: 13px;
      margin: 0 0 4px;
    }}
    svg {{
      width: 100%;
      display: block;
    }}
    #driftPlots {{
      grid-template-columns: repeat(3, minmax(280px, 1fr));
    }}
    #driftPlots svg {{
      height: 340px;
    }}
    #histPlots svg {{
      height: 230px;
    }}
    #scatterPlots svg {{
      height: 330px;
    }}
    #individualPlots svg {{
      height: 270px;
    }}
    #feedbackPlots, #feedbackParticipationPlot {{
      grid-template-columns: minmax(0, 1fr);
    }}
    #customRankPlot {{
      grid-template-columns: minmax(0, 1fr);
    }}
    #feedbackPlots svg {{
      height: 360px;
    }}
    #feedbackParticipationPlot svg {{
      height: 300px;
    }}
    .individual-table-wrap {{
      border: 1px solid var(--border);
      overflow: auto;
      max-height: calc(50vh - 80px);
    }}
    tr.selectable-row {{
      cursor: pointer;
    }}
    tr.selectable-row:hover td {{
      outline: 2px solid rgba(37, 99, 235, 0.45);
      outline-offset: -2px;
    }}
    tr.selected-row td {{
      box-shadow: inset 0 2px 0 #2563eb, inset 0 -2px 0 #2563eb;
    }}
    .impact-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 12px;
    }}
    .impact-table-wrap {{
      border: 1px solid var(--border);
      overflow: auto;
      max-height: 390px;
    }}
    .prediction-system-picker {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 10px 0 14px;
    }}
    .prediction-system-picker label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 9px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #f8fafc;
      font-size: 13px;
      cursor: pointer;
    }}
    .prediction-system-picker input {{
      min-width: auto;
      margin: 0;
    }}
    .diagnostic-system-picker {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 12px;
    }}
    .diagnostic-system-picker label {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 5px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #f8fafc;
      font-size: 12px;
      cursor: pointer;
    }}
    .diagnostic-system-picker input {{ min-width: auto; margin: 0; }}
    .prediction-view-mode[aria-pressed="true"] {{
      background: #7c3aed !important;
      border-color: #7c3aed !important;
      color: white;
    }}
    .page-note {{
      margin-top: 34px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      font-style: italic;
    }}
    .prediction-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.8fr);
      gap: 14px;
      align-items: start;
    }}
    #predictionPlot svg {{
      height: 560px;
    }}
    .puzzler-plot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .puzzler-plot-grid svg, #puzzlerEventHistogram svg {{
      height: 390px;
    }}
    .time-method-details {{
      margin: 8px 0 0;
      border-color: #c4b5fd;
      background: #faf9ff;
    }}
    .time-method-details summary {{
      color: #5b21b6;
      font-size: 12px;
    }}
    .assumption-note {{
      padding: 0 11px 11px;
      color: #4c1d95;
      font-size: 12px;
      line-height: 1.45;
    }}
    .assumption-item {{
      padding: 9px 0;
    }}
    .assumption-item + .assumption-item {{
      border-top: 1px solid #ddd6fe;
    }}
    .assumption-item strong {{
      display: block;
      margin-bottom: 2px;
      color: #3b0764;
    }}
    #puzzlerEventHistogram {{
      max-width: 1100px;
      margin-top: 10px;
    }}
    @media (max-width: 980px) {{
      .prediction-layout {{ grid-template-columns: 1fr; }}
      .puzzler-plot-grid {{ grid-template-columns: 1fr; }}
      .plot-grid, #driftPlots {{ grid-template-columns: 1fr; }}
    }}
    .mode-toggle button[aria-pressed="true"] {{
      background: #1d4ed8;
      border-color: #1d4ed8;
      color: white;
    }}
    .section-divider {{
      border: 0;
      border-top: 3px solid #334155;
      margin: 34px 0 22px;
    }}
    .hover-target {{
      cursor: crosshair;
      pointer-events: all;
    }}
    .hover-slice {{
      fill: transparent;
    }}
    .hover-slice:hover {{
      fill: rgba(37, 99, 235, 0.08);
    }}
    #tooltip {{
      position: fixed;
      z-index: 1000;
      display: none;
      pointer-events: none;
      max-width: 270px;
      padding: 8px 9px;
      border: 1px solid #111827;
      border-radius: 6px;
      background: rgba(17, 24, 39, 0.94);
      color: #fff;
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-line;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.20);
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
    }}
    th, td {{
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
      white-space: nowrap;
      text-align: right;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--header);
      cursor: pointer;
      user-select: none;
      font-weight: 650;
    }}
    th:first-child, td:first-child {{
      position: sticky;
      left: 0;
      z-index: 1;
      text-align: left;
      background: inherit;
      min-width: 185px;
    }}
    th:first-child {{
      z-index: 3;
      background: var(--header);
    }}
    tr:nth-child(odd) {{
      background: var(--row);
    }}
    tr:nth-child(even) {{
      background: var(--row-alt);
    }}
    .rank-cell {{
      font-weight: 650;
      text-align: center;
      border-radius: 0;
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
<main>
  <header class="page-header">
    <h1>JPAR and Ranking Exploration</h1>
  </header>
  <nav class="tab-bar" role="tablist" aria-label="JPAR analysis sections">
    <button class="tab-button" id="tab-raw" role="tab" aria-controls="panel-raw" aria-selected="true" data-tab="raw">Raw Data</button>
    <button class="tab-button" id="tab-rankings" role="tab" aria-controls="panel-rankings" aria-selected="false" data-tab="rankings" tabindex="-1">Ranking Comparison</button>
    <button class="tab-button" id="tab-puzzler" role="tab" aria-controls="panel-puzzler" aria-selected="false" data-tab="puzzler" tabindex="-1">Puzzler Data</button>
    <button class="tab-button" id="tab-predictions" role="tab" aria-controls="panel-predictions" aria-selected="false" data-tab="predictions" tabindex="-1">Predictions</button>
    <button class="tab-button" id="tab-misc" role="tab" aria-controls="panel-misc" aria-selected="false" data-tab="misc" tabindex="-1">Miscellaneous</button>
  </nav>
  <section class="tab-panel" id="panel-raw" role="tabpanel" aria-labelledby="tab-raw" data-panel="raw">
  <h1>JPAR Calculation Data</h1>
  <div class="tab-intro">The actual event results and intermediate values entering the JPAR calculation. The default filter shows only member-resolved rows that received a JPAR update in this cutoff run.</div>
  <div class="controls" style="margin-top:10px">
    <label><input id="overviewIncludedOnly" type="checkbox" checked style="min-width:auto"> Included in JPAR only</label>
    <label>Search <select id="overviewSearchColumn"><option value="all">All columns</option></select></label>
    <input id="overviewSearch" type="search" placeholder="Search calculation results...">
    <label>Rows <select id="overviewLimit"><option value="100">100</option><option value="250">250</option><option value="500">500</option><option value="99999" selected>All</option></select></label>
    <span id="overviewMeta" class="meta"></span>
  </div>
  <div class="table-wrap" style="max-height:520px"><table id="overviewTable"></table></div>
  <div class="page-note">Note: Some ranking systems are currently disabled pending further testing.</div>
  </section>
  <section class="tab-panel" id="panel-rankings" role="tabpanel" aria-labelledby="tab-rankings" data-panel="rankings" hidden>
  <h1>JPAR Ranking System Comparison</h1>
  <div class="tab-intro">Compare JPAR with alternative ranking systems across final ranks, historical drift, score distributions, and one-to-one rank plots. Click a person row to highlight them throughout.</div>
  <div class="controls" style="align-items:flex-end; margin-bottom:16px">
    <label style="flex:1 1 520px">Rankings after <strong id="rankingCutoffLabel"></strong><br>
      <input id="rankingCutoff" type="range" min="0" max="{max(0, len(ranking_timeline) - 1)}" step="1" value="{max(0, len(ranking_timeline) - 1)}" style="width:100%; min-width:260px; padding:0; border:0">
    </label>
    <span id="rankingCutoffPosition" class="meta" style="margin-left:0"></span>
  </div>
  <div class="controls">
    <input id="search" type="search" placeholder="Search name...">
    <label>Min events <input id="minEvents" type="number" min="1" step="1" value="1" style="width:70px; min-width:70px"></label>
    <label>Rows
      <select id="limit">
        <option value="50">50</option>
        <option value="100" selected>100</option>
        <option value="250">250</option>
        <option value="500">500</option>
        <option value="99999">All</option>
      </select>
    </label>
    <button id="reset">Reset</button>
    <span id="meta" class="meta"></span>
  </div>
  <details>
    <summary>Ranking system descriptions</summary>
    <div id="descriptions" class="explain"></div>
  </details>
  <div class="table-wrap">
    <table id="rankTable"></table>
  </div>
  <h2>Drift Percentiles Over Time</h2>
  <div class="muted">Lines show p10, p25, median, p75, and p90 of each system's as-of score distribution after each event. Median is black.</div>
  <div id="driftSystemPicker" class="diagnostic-system-picker"></div>
  <div class="controls">
    <label>Drift min events <input id="driftMinEvents" type="number" min="1" step="1" value="1" style="width:70px; min-width:70px"></label>
    <span class="mode-toggle">
      <button class="individual-mode" data-mode="score" aria-pressed="true">Score</button>
      <button class="individual-mode" data-mode="rank" aria-pressed="false">Rank</button>
    </span>
    <span id="driftMeta" class="meta"></span>
  </div>
  <div id="driftPlots" class="plot-grid"></div>
  <h2>Score Distributions</h2>
  <div class="muted">Histograms use the final person-level scores for each system, clipped to the 1st and 99th percentiles to keep long tails readable.</div>
  <div id="histSystemPicker" class="diagnostic-system-picker"></div>
  <div class="controls">
    <label>Histogram min events <input id="histMinEvents" type="number" min="1" step="1" value="1" style="width:70px; min-width:70px"></label>
    <span id="histMeta" class="meta"></span>
  </div>
  <div id="histPlots" class="plot-grid"></div>
  <h2>Rank Comparisons vs JPAR</h2>
  <div class="muted">Each scatter plot compares original fiducial JPAR rank on the x-axis against another system's rank on the y-axis. Lower rank is better, so closer to the upper-left is stronger.</div>
  <div id="scatterSystemPicker" class="diagnostic-system-picker"></div>
  <div class="controls">
    <label>Scatter min events <input id="scatterMinEvents" type="number" min="1" step="1" value="1" style="width:70px; min-width:70px"></label>
    <span id="scatterMeta" class="meta"></span>
  </div>
  <div id="scatterPlots" class="plot-grid"></div>
  <section id="selectedEventsPanel" hidden>
    <h2 id="selectedEventsTitle"></h2>
    <div class="muted">Post-event rank in each system for the selected person.</div>
    <div class="table-wrap" style="max-height:430px; margin-top:10px"><table id="individualEventTable"></table></div>
  </section>
  <hr class="section-divider">
  <h2>Custom Rank Comparison</h2>
  <div class="muted">Choose any two enabled ranking systems for a direct one-to-one comparison at the currently selected ranking cutoff.</div>
  <div class="controls" style="margin-top:10px">
    <label>X-axis <select id="customRankX"></select></label>
    <label>Y-axis <select id="customRankY"></select></label>
    <label>Min events <input id="customRankMinEvents" type="number" min="1" step="1" value="1" style="width:70px; min-width:70px"></label>
    <span id="customRankMeta" class="meta"></span>
  </div>
  <div id="customRankPlot" class="plot-grid"></div>
  <div class="page-note">Note: Some ranking systems are currently disabled pending further testing.</div>
  </section>
  <section class="tab-panel" id="panel-puzzler" role="tabpanel" aria-labelledby="tab-puzzler" data-panel="puzzler" hidden>
  <h1>Puzzler Data</h1>
  <div class="tab-intro">Inspect one puzzler's incoming system score against what happened next. Rank comparisons are genuinely pre-event. Converted time estimates deliberately use the observed event distribution and are diagnostic, not leakage-free forecasts.</div>
  <div class="controls" style="align-items:flex-end">
    <label>Puzzler<br><select id="puzzlerSelect" style="min-width:300px"></select></label>
    <label>Detail system<br><select id="puzzlerSystem" style="min-width:240px"></select></label>
    <span id="puzzlerMeta" class="meta"></span>
  </div>
  <div class="muted">Systems shown on both scatterplots</div>
  <div id="puzzlerScatterSystemPicker" class="prediction-system-picker"></div>
  <div class="puzzler-plot-grid">
    <div><div class="plot-card"><div class="plot-title">Diagnostic predicted vs actual time</div><div id="puzzlerTimePlot"></div></div><details class="time-method-details"><summary>How predicted times are derived</summary><div id="puzzlerTimeAssumption" class="assumption-note"></div></details></div>
    <div class="plot-card"><div class="plot-title">Pre-event predicted vs actual rank</div><div id="puzzlerRankPlot"></div></div>
  </div>
  <h2>Selected Event</h2>
  <div class="controls">
    <label>Event <select id="puzzlerEvent" style="min-width:420px"></select></label>
    <span id="puzzlerEventMeta" class="meta"></span>
  </div>
  <div id="puzzlerEventHistogram" class="plot-card"></div>
  <h2>Prediction History</h2>
  <div class="table-wrap" style="max-height:480px"><table id="puzzlerHistoryTable"></table></div>
  </section>
  <section class="tab-panel" id="panel-predictions" role="tabpanel" aria-labelledby="tab-predictions" data-panel="predictions" hidden>
  <h1>Event Predictions</h1>
  <div class="tab-intro">Compare the ranking systems immediately before an event with the entrants' actual finish order. Incoming ranks are recalculated within each event's eligible field, so the one-to-one line represents a perfect prediction.</div>
  <div class="controls" style="align-items:flex-end">
    <label style="flex:1 1 480px">Event <strong id="predictionEventLabel"></strong><br>
      <input id="predictionEventSlider" type="range" min="0" max="{max(0, len(ranking_timeline) - 1)}" step="1" value="{max(0, len(ranking_timeline) - 1)}" style="width:100%; min-width:260px; padding:0; border:0">
    </label>
    <label style="flex:1 1 360px">Select event<br><select id="predictionEventSelect" style="width:100%; min-width:280px"></select></label>
  </div>
  <div class="muted">Systems to compare</div>
  <div id="predictionSystemPicker" class="prediction-system-picker"></div>
  <div class="controls">
    <span class="mode-toggle" aria-label="Prediction event scope">
      <button class="prediction-scope-mode" data-mode="event" aria-pressed="true">Selected Event</button>
      <button class="prediction-scope-mode" data-mode="all" aria-pressed="false">All Events</button>
    </span>
  </div>
  <div id="predictionSummary" class="muted" style="margin-bottom:10px"></div>
  <div class="prediction-layout">
    <div id="predictionPlot"></div>
    <div>
      <div class="plot-title">Prediction accuracy</div>
      <div class="table-wrap" style="max-height:560px"><table id="predictionMetricsTable"></table></div>
      <div class="muted" style="margin-top:14px">Plot style</div>
      <span class="mode-toggle" aria-label="Prediction plot style" style="display:inline-flex; margin-top:6px">
        <button class="prediction-view-mode" data-mode="bands" aria-pressed="true">Rolling Bands</button>
        <button class="prediction-view-mode" data-mode="scatter" aria-pressed="false">Scatter</button>
      </span>
    </div>
  </div>
  <h2>Entrants in Selected Event</h2>
  <div class="muted">Entrants with an incoming JPAR appear first. Competitors without one are retained at the bottom and excluded from JPAR's prediction metrics.</div>
  <div class="table-wrap" style="max-height:620px; margin-top:10px"><table id="predictionEntrantsTable"></table></div>
  <div class="page-note">Note: Some ranking systems are currently disabled pending further testing.</div>
  </section>
  <section class="tab-panel" id="panel-misc" role="tabpanel" aria-labelledby="tab-misc" data-panel="misc" hidden>
  <h1>Raw vs Adjusted JPAR by Event</h1>
  <div class="muted">Raw event JPAR is computed from the event mean; adjusted event JPAR applies JPAR's calibration multiplier. The divergence between their medians is the calibration intervention for that event. Green shows the as-of median latest JPAR after each event.</div>
  <div id="feedbackPlots" class="plot-grid"></div>
  <div id="feedbackParticipationPlot" class="plot-grid" style="margin-top:12px"></div>
  <hr class="section-divider">
  <h1>Event Impacts in JPAR</h1>
  <div class="muted">JPAR's adjustment factor is <code>event mean time / anchor-implied mean time</code>. Below 1.00 inflates every participant's raw event JPAR (a relative penalty); above 1.00 lowers it (a relative boost). These are calibration shifts, not a claim that a puzzle itself was objectively easy or hard.</div>
  <div class="muted" style="margin-top:6px">Median JPAR delta = <code>adjusted_event_jpar - raw_event_jpar</code>: a positive value increases JPAR scores (worse); a negative value decreases them (better).</div>
  <div id="impactMeta" class="muted" style="margin-top:8px"></div>
  <div class="impact-grid" style="margin-top:10px">
    <div><div class="plot-title">Largest relative JPAR penalties</div><div class="impact-table-wrap"><table id="penaltyEventsTable"></table></div></div>
    <div><div class="plot-title">Largest relative JPAR boosts</div><div class="impact-table-wrap"><table id="boostEventsTable"></table></div></div>
  </div>
  <h2>Cumulative JPAR Calibration Impact</h2>
  <div class="muted">Negative final calibration effect means JPAR ended lower (better) than the same running update using raw event JPAR; positive means it ended higher (worse). Click headers to sort.</div>
  <div class="controls" style="margin-top:10px">
    <label>Min events <input id="cumulativeMinEvents" type="number" min="1" step="1" value="1" style="width:70px; min-width:70px"></label>
    <label style="min-width:300px">Include final JPAR up to <strong id="cumulativeMaxJparValue"></strong><br><input id="cumulativeMaxJpar" type="range" min="0" max="{cumulative_jpar_max:.4f}" step="0.01" value="{cumulative_jpar_max:.4f}" style="width:300px; min-width:300px; padding:0; border:0"></label>
    <span id="cumulativeMeta" class="meta"></span>
  </div>
  <svg id="cumulativeJparHistogram" viewBox="0 0 620 90" style="width:min(620px, 100%); height:90px; margin:0 0 8px"></svg>
  <div class="table-wrap" style="max-height:430px; margin-top:10px">
    <table id="cumulativeCalibrationTable"></table>
  </div>
  <div class="page-note">Note: Some ranking systems are currently disabled pending further testing.</div>
  </section>
  <div id="tooltip"></div>
</main>
<script>
const rows = {data_json};
const columns = {columns_json};
const descriptions = {descriptions_json};
const driftData = {drift_json};
const histData = {hist_json};
const individualData = {individual_json};
const rankingTimeline = {ranking_timeline_json};
const puzzlerProfiles = {puzzler_profiles_json};
const puzzlerEvents = {puzzler_events_json};
const impactData = {impact_json};
const cumulativeCalibrationData = {cumulative_calibration_json};
const calculationOverviewData = {calculation_overview_json};
const feedbackData = {feedback_json};
const cumulativeJparMax = {cumulative_jpar_max:.8f};
const individualSystems = {individual_systems_json};
const profileSystems = {profile_systems_json};
const scatterColumns = {scatter_columns_json};
const defaultPlotSystems = new Set({default_plot_systems_json});
const selectedDriftSystems = new Set(defaultPlotSystems);
const selectedHistSystems = new Set(defaultPlotSystems);
const selectedScatterSystems = new Set({default_scatter_systems_json});
const higherIsBetterScores = new Set(["elo_rating", "trueskill_conservative", "msp_like_score", "external_logtime", "external_logtime_conservative", "external_logtime_no_tier", "external_bayesian", "external_bayesian_conservative", "external_nationals"]);
let sortKey = "jpar_rank";
let sortDir = "asc";
let selectedMemberKey = null;
let individualView = "score";
let cumulativeSortKey = "final_calibration_effect";
let cumulativeSortDir = "asc";
let overviewSortKey = "date";
let overviewSortDir = "asc";
let currentRankingRows = rows;
let rankingFrame = null;
const defaultPredictionSystems = new Set(["jpar_rank", "mean_log_zscore_rank", "mean_zscore_rank"]);
const selectedPredictionSystems = new Set();
const selectedPuzzlerScatterSystems = new Set(["jpar"]);
const puzzlerScatterColors = ["#2563eb", "#f97316", "#16a34a", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04", "#475569", "#db2777", "#65a30d", "#4f46e5"];
let predictionViewMode = "bands";
let predictionScopeMode = "event";
let predictionEntrantSortKey = "incoming_jpar_relative";
let predictionEntrantSortDir = "asc";
const renderedTabs = new Set();

function rebuildRankingRows(cutoffIndex) {{
  const state = new Map();
  for (let eventIndex = 0; eventIndex <= cutoffIndex; eventIndex += 1) {{
    const event = rankingTimeline[eventIndex];
    if (!event) continue;
    event.updates.forEach(update => {{
      const previous = state.get(update._member_key) || {{_member_key: update._member_key}};
      state.set(update._member_key, Object.assign(previous, update));
    }});
  }}
  const snapshot = [...state.values()];
  columns.filter(column => column.kind === "rank").forEach(column => {{
    const scoreKey = column.key.replace(/_rank$/, "");
    const direction = higherIsBetterScores.has(scoreKey) ? -1 : 1;
    const ranked = snapshot
      .filter(row => row[scoreKey] != null && Number.isFinite(Number(row[scoreKey])))
      .sort((a, b) => direction * (Number(a[scoreKey]) - Number(b[scoreKey])) || String(a.full_name).localeCompare(String(b.full_name)));
    let previousScore = null;
    let previousRank = null;
    ranked.forEach((row, index) => {{
      const score = Number(row[scoreKey]);
      const rank = previousScore !== null && score === previousScore ? previousRank : index + 1;
      row[column.key] = rank;
      previousScore = score;
      previousRank = rank;
    }});
  }});
  currentRankingRows = snapshot;
  const cutoff = rankingTimeline[cutoffIndex];
  document.getElementById("rankingCutoffLabel").textContent = cutoff ? `${{cutoff.date}} · ${{cutoff.event_name}} · ${{cutoff.event}}` : "No event";
  document.getElementById("rankingCutoffPosition").textContent = cutoff ? `Event ${{cutoffIndex + 1}} of ${{rankingTimeline.length}} · ${{snapshot.length}} ranked puzzlers` : "";
}}

function initializeRankingSlider() {{
  const slider = document.getElementById("rankingCutoff");
  rebuildRankingRows(Number(slider.value));
}}

function renderTab(tab) {{
  if (renderedTabs.has(tab)) return;
  if (tab === "raw") {{
    renderCalculationOverview();
  }} else if (tab === "rankings") {{
    initializeRankingSlider();
    initializeDiagnosticPickers();
    initializeCustomRankComparison();
    renderDescriptions();
    render();
    renderDriftPlots();
    renderHistPlots();
    renderRankScatterPlots();
    renderCustomRankComparison();
    renderSelectedPersonEvents();
  }} else if (tab === "puzzler") {{
    initializePuzzlerProfile();
  }} else if (tab === "predictions") {{
    initializePredictions();
  }} else if (tab === "misc") {{
    renderFeedbackSection();
    renderEventImpacts();
    renderCumulativeCalibrationTable();
  }}
  renderedTabs.add(tab);
  attachPlotTooltips();
}}

function activateTab(tab, updateHash = true) {{
  const target = document.querySelector(`[data-panel="${{tab}}"]`);
  if (!target) tab = "raw";
  document.querySelectorAll(".tab-button").forEach(button => {{
    const active = button.dataset.tab === tab;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }});
  document.querySelectorAll(".tab-panel").forEach(panel => {{
    panel.hidden = panel.dataset.panel !== tab;
  }});
  renderTab(tab);
  if (updateHash && window.location.hash !== `#${{tab}}`) history.replaceState(null, "", `#${{tab}}`);
}}

const overviewColumns = [
  ["included", "In JPAR", "boolean"], ["date", "Date", "text"], ["event", "Event ID", "text"], ["event_name", "Event", "text"], ["name", "Name", "text"],
  ["place", "Place", "number"], ["time_seconds", "Time", "time"], ["players", "Players", "number"],
  ["event_jpar", "Raw Event JPAR", "score"], ["adjusted_event_jpar", "Adjusted Event JPAR", "score"],
  ["previous_jpar", "Previous JPAR", "score"], ["jpar_out", "JPAR After Event", "score"],
];

function renderCalculationOverview() {{
  const searchColumn = document.getElementById("overviewSearchColumn");
  if (searchColumn.options.length === 1) {{
    overviewColumns.forEach(([key, label]) => {{
      const option = document.createElement("option"); option.value = key; option.textContent = label; searchColumn.appendChild(option);
    }});
  }}
  const query = document.getElementById("overviewSearch").value.trim().toLowerCase();
  const selectedColumn = searchColumn.value;
  const limit = Number(document.getElementById("overviewLimit").value);
  const includedOnly = document.getElementById("overviewIncludedOnly").checked;
  const filtered = calculationOverviewData.filter(row => {{
    if (includedOnly && !row.included) return false;
    if (!query) return true;
    const keys = selectedColumn === "all" ? overviewColumns.map(([key]) => key) : [selectedColumn];
    return keys.some(key => String(row[key] ?? "").toLowerCase().includes(query));
  }}).sort((a, b) => {{
    const av = a[overviewSortKey], bv = b[overviewSortKey];
    if (av == null) return 1; if (bv == null) return -1;
    const cmp = typeof av === "string" ? String(av).localeCompare(String(bv)) : Number(av) - Number(bv);
    return cmp * (overviewSortDir === "asc" ? 1 : -1);
  }});
  const visible = filtered.slice(0, limit);
  const value = (row, key, kind) => {{
    if (row[key] == null) return "";
    if (kind === "boolean") return row[key] ? "Yes" : "No";
    if (kind === "time") return formatTime(row[key]);
    if (kind === "score") return Number(row[key]).toFixed(4);
    return escapeAttr(row[key]);
  }};
  const table = document.getElementById("overviewTable");
  table.innerHTML = `<thead><tr>${{overviewColumns.map(([key, label]) => `<th data-overview-key="${{key}}">${{label}}${{key === overviewSortKey ? (overviewSortDir === "asc" ? " ▲" : " ▼") : ""}}</th>`).join("")}}</tr></thead><tbody>${{visible.map(row => `<tr>${{overviewColumns.map(([key, , kind]) => `<td>${{value(row, key, kind)}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
  document.getElementById("overviewMeta").textContent = `Showing ${{visible.length}} of ${{filtered.length}} qualified rows; ${{calculationOverviewData.length}} total`;
  table.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {{
    const key = th.dataset.overviewKey;
    if (overviewSortKey === key) overviewSortDir = overviewSortDir === "asc" ? "desc" : "asc";
    else {{ overviewSortKey = key; overviewSortDir = "asc"; }}
    renderCalculationOverview();
  }}));
}}

function rankStyle(value) {{
  if (value == null) return "";
  const colorMaxRank = 150;
  const t = Math.max(0, Math.min(1, (Number(value) - 1) / Math.max(1, colorMaxRank - 1)));
  // Continuous green -> yellow -> red scale.
  const hue = 130 - (130 * t);
  const sat = 72;
  const light = 34 + (54 * Math.pow(t, 0.72));
  const text = t < 0.34 ? "white" : "#111827";
  return `background-color: hsl(${{hue}}, ${{sat}}%, ${{light}}%); color: ${{text}};`;
}}

function formatValue(value, kind) {{
  if (value == null || Number.isNaN(value)) return "";
  if (kind === "integer") return String(Math.round(Number(value)));
  if (kind === "number") {{
    return Number(value).toFixed(value < 10 ? 6 : 2);
  }}
  return String(value);
}}

function compare(a, b, col) {{
  const av = a[col.key];
  const bv = b[col.key];
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (col.kind === "text") return String(av).localeCompare(String(bv));
  return Number(av) - Number(bv);
}}

function filteredRows() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  const minEvents = Number(document.getElementById("minEvents").value || 1);
  const limit = Number(document.getElementById("limit").value);
  const col = columns.find(c => c.key === sortKey) || columns[0];
  let out = currentRankingRows.filter(r => {{
    const nameOk = !q || String(r.full_name || "").toLowerCase().includes(q);
    const eventsOk = Number(r.events || 0) >= minEvents;
    return nameOk && eventsOk;
  }});
  out.sort((a, b) => compare(a, b, col) * (sortDir === "asc" ? 1 : -1));
  return [out, out.slice(0, limit)];
}}

function render() {{
  const table = document.getElementById("rankTable");
  const [allFiltered, visible] = filteredRows();
  const thead = `<thead><tr>${{columns.map(c => {{
    const marker = c.key === sortKey ? (sortDir === "asc" ? " ▲" : " ▼") : "";
    return `<th data-key="${{c.key}}">${{c.label}}${{marker}}</th>`;
  }}).join("")}}</tr></thead>`;
  const rowHtml = r => `<tr class="selectable-row ${{r._member_key === selectedMemberKey ? "selected-row" : ""}}" data-member-key="${{escapeAttr(r._member_key)}}">${{columns.map(c => {{
    const value = r[c.key];
    const cls = c.kind === "rank" ? "rank-cell" : "";
    const style = c.kind === "rank" ? rankStyle(value) : "";
    return `<td class="${{cls}}" style="${{style}}">${{formatValue(value, c.kind)}}</td>`;
  }}).join("")}}</tr>`;
  const tbody = `<tbody>${{visible.map(rowHtml).join("")}}</tbody>`;
  table.innerHTML = thead + tbody;
  document.getElementById("meta").textContent = `Showing ${{visible.length}} of ${{allFiltered.length}} filtered; ${{currentRankingRows.length}} at cutoff`;
  document.querySelectorAll("#rankTable th").forEach(th => {{
    th.addEventListener("click", () => {{
      const key = th.dataset.key;
      if (sortKey === key) sortDir = sortDir === "asc" ? "desc" : "asc";
      else {{
        sortKey = key;
        const col = columns.find(c => c.key === key);
        sortDir = col && col.kind === "text" ? "asc" : "asc";
      }}
      render();
    }});
  }});
  document.querySelectorAll("tr[data-member-key]").forEach(tr => {{
    tr.addEventListener("click", () => {{
      selectedMemberKey = selectedMemberKey === tr.dataset.memberKey ? null : tr.dataset.memberKey;
      render();
      renderDriftPlots();
      renderHistPlots();
      renderRankScatterPlots();
      renderSelectedPersonEvents();
      attachPlotTooltips();
    }});
  }});
}}

function renderDescriptions() {{
  const container = document.getElementById("descriptions");
  container.innerHTML = columns
    .filter(c => descriptions[c.key])
    .map(c => `<div class="explain-item"><div class="explain-name">${{c.label}}</div><div class="muted">${{descriptions[c.key]}}</div></div>`)
    .join("");
}}

function scale(value, domainMin, domainMax, rangeMin, rangeMax) {{
  if (domainMax === domainMin) return (rangeMin + rangeMax) / 2;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}}

function niceTicks(min, max, count = 4) {{
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({{length: count}}, (_, i) => min + i * step);
}}

function quantile(sortedValues, q) {{
  if (!sortedValues.length) return null;
  const pos = (sortedValues.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (pos - lo);
}}

function percentileRowsFor(payload, minEvents) {{
  return (payload.rows || []).map(row => {{
    const values = (row.samples || [])
      .filter(sample => Number(sample.events || 0) >= minEvents)
      .map(sample => Number(sample.score))
      .filter(v => Number.isFinite(v))
      .sort((a, b) => a - b);
    if (!values.length) return null;
    return {{
      date: row.date,
      event: row.event,
      n: values.length,
      p10: quantile(values, 0.10),
      p25: quantile(values, 0.25),
      p50: quantile(values, 0.50),
      p75: quantile(values, 0.75),
      p90: quantile(values, 0.90),
    }};
  }}).filter(Boolean);
}}

function histogramFor(payload, minEvents, bins = 36) {{
  const values = (payload.samples || [])
    .filter(sample => Number(sample.events || 0) >= minEvents)
    .map(sample => Number(sample.score))
    .filter(v => Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!values.length) return {{bins: [], counts: [], n: 0}};
  const lo = quantile(values, 0.01);
  const hi = quantile(values, 0.99);
  if (lo === hi) return {{bins: [lo], counts: [values.length], n: values.length}};
  const counts = Array.from({{length: bins}}, () => 0);
  const centers = Array.from({{length: bins}}, (_, i) => lo + ((i + 0.5) * (hi - lo)) / bins);
  values.forEach(v => {{
    const clipped = Math.max(lo, Math.min(hi, v));
    const idx = Math.min(bins - 1, Math.floor(((clipped - lo) / (hi - lo)) * bins));
    counts[idx] += 1;
  }});
  return {{bins: centers, counts, n: values.length}};
}}

function dateTicks(minTime, maxTime, count = 4) {{
  const step = (maxTime - minTime) / (count - 1);
  return Array.from({{length: count}}, (_, i) => new Date(minTime + i * step));
}}

function fmtDate(d) {{
  return d.toISOString().slice(0, 10);
}}

function scoreMeanFor(system, minEvents = 1) {{
  const values = rows
    .filter(r => Number(r.events || 0) >= minEvents)
    .map(r => r[system])
    .filter(v => Number.isFinite(v));
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}}

function escapeAttr(value) {{
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}}

function tooltipText(...lines) {{
  return escapeAttr(lines.filter(Boolean).join("\\n"));
}}

function buildDiagnosticPicker(containerId, options, selectedSet, renderFunction) {{
  const container = document.getElementById(containerId);
  if (container.children.length) return;
  options.forEach(option => {{
    const checked = selectedSet.has(option.key);
    container.insertAdjacentHTML("beforeend", `<label><input type="checkbox" data-system="${{option.key}}" ${{checked ? "checked" : ""}}> ${{option.label}}</label>`);
  }});
  container.querySelectorAll("input").forEach(input => input.addEventListener("change", () => {{
    if (input.checked) selectedSet.add(input.dataset.system);
    else selectedSet.delete(input.dataset.system);
    renderFunction();
    attachPlotTooltips();
  }}));
}}

function initializeDiagnosticPickers() {{
  const plotOptions = Object.entries(driftData).map(([key, payload]) => ({{key, label: payload.label}}));
  buildDiagnosticPicker("driftSystemPicker", plotOptions, selectedDriftSystems, renderDriftPlots);
  buildDiagnosticPicker("histSystemPicker", plotOptions.filter(option => histData[option.key]), selectedHistSystems, renderHistPlots);
  buildDiagnosticPicker("scatterSystemPicker", scatterColumns, selectedScatterSystems, renderRankScatterPlots);
}}

function initializeCustomRankComparison() {{
  const xSelect = document.getElementById("customRankX");
  const ySelect = document.getElementById("customRankY");
  if (!xSelect.options.length) {{
    columns.filter(column => column.kind === "rank").forEach(column => {{
      [xSelect, ySelect].forEach(select => {{
        const option = document.createElement("option");
        option.value = column.key;
        option.textContent = column.label;
        select.appendChild(option);
      }});
    }});
    xSelect.value = "jpar_rank";
    ySelect.value = "mean_log_zscore_rank";
  }}
}}

function renderCustomRankComparison() {{
  const container = document.getElementById("customRankPlot");
  const xKey = document.getElementById("customRankX").value;
  const yKey = document.getElementById("customRankY").value;
  const xColumn = columns.find(column => column.key === xKey);
  const yColumn = columns.find(column => column.key === yKey);
  const minEvents = Number(document.getElementById("customRankMinEvents").value || 1);
  const points = currentRankingRows.map(row => ({{
    name: row.full_name,
    events: Number(row.events || 0),
    x: Number(row[xKey]),
    y: Number(row[yKey]),
  }})).filter(point => point.events >= minEvents && Number.isFinite(point.x) && Number.isFinite(point.y));
  document.getElementById("customRankMeta").textContent = `${{points.length}} people`;
  if (!points.length || !xColumn || !yColumn) {{ container.innerHTML = ""; return; }}
  const width = 760, height = 500, pad = {{left: 54, right: 18, top: 18, bottom: 50}};
  const rankMax = Math.max(100, Math.ceil(Math.max(...points.flatMap(point => [point.x, point.y])) / 100) * 100);
  const x = value => scale(value, 1, rankMax, pad.left, width - pad.right);
  const y = value => scale(value, 1, rankMax, height - pad.bottom, pad.top);
  const ticks = niceTicks(1, rankMax, 5);
  const grid = ticks.map(value => `<line x1="${{x(value).toFixed(1)}}" y1="${{pad.top}}" x2="${{x(value).toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#e5e7eb"/><line x1="${{pad.left}}" y1="${{y(value).toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y(value).toFixed(1)}}" stroke="#e5e7eb"/><text x="${{x(value).toFixed(1)}}" y="${{height - 22}}" text-anchor="middle" font-size="11" fill="#6b7280">${{Math.round(value)}}</text><text x="${{pad.left - 9}}" y="${{(y(value) + 4).toFixed(1)}}" text-anchor="end" font-size="11" fill="#6b7280">${{Math.round(value)}}</text>`).join("");
  const dots = points.map(point => `<circle class="hover-target" cx="${{x(point.x).toFixed(1)}}" cy="${{y(point.y).toFixed(1)}}" r="3.2" fill="#2563eb" opacity="0.5" data-tooltip="${{tooltipText(point.name, `events: ${{point.events}}`, `${{xColumn.label}}: ${{Math.round(point.x)}}`, `${{yColumn.label}}: ${{Math.round(point.y)}}`)}}"></circle>`).join("");
  container.innerHTML = `<div class="plot-card"><div class="plot-title">${{xColumn.label}} vs ${{yColumn.label}}</div><svg viewBox="0 0 ${{width}} ${{height}}">${{grid}}<line x1="${{x(1)}}" y1="${{y(1)}}" x2="${{x(rankMax)}}" y2="${{y(rankMax)}}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5 5"/>${{dots}}<text x="${{width / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="12">${{xColumn.label}}</text><text x="15" y="${{height / 2}}" text-anchor="middle" font-size="12" transform="rotate(-90 15 ${{height / 2}})">${{yColumn.label}}</text></svg></div>`;
}}

function pathFor(points, xMin, xMax, yMin, yMax, width, height, pad, key) {{
  return points.map((p, i) => {{
    const x = scale(new Date(p.date).getTime(), xMin, xMax, pad.left, width - pad.right);
    const y = scale(p[key], yMin, yMax, height - pad.bottom, pad.top);
    return `${{i === 0 ? "M" : "L"}}${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(" ");
}}

function renderDriftPlots() {{
  const container = document.getElementById("driftPlots");
  const colors = {{p10: "#2563eb", p25: "#60a5fa", p50: "#111827", p75: "#f97316", p90: "#dc2626"}};
  const keys = ["p10", "p25", "p50", "p75", "p90"];
  const minEvents = Number(document.getElementById("driftMinEvents").value || 1);
  const selectedPerson = selectedMemberKey ? individualData[selectedMemberKey] : null;
  let latestN = 0;
  container.innerHTML = Object.entries(driftData).filter(([system]) => selectedDriftSystems.has(system)).map(([system, payload]) => {{
    const pts = percentileRowsFor(payload, minEvents);
    if (!pts.length) return "";
    const selectedTrend = selectedPerson ? (selectedPerson.trends[system] || []) : [];
    const rankMode = individualView === "rank" && selectedTrend.length > 0;
    latestN = Math.max(latestN, pts[pts.length - 1].n || 0);
    const width = 420, height = 260, pad = {{left: 44, right: 12, top: 16, bottom: 36}};
    const times = pts.map(p => new Date(p.date).getTime());
    const vals = pts.flatMap(p => keys.map(k => p[k]))
      .concat(rankMode ? [] : selectedTrend.map(point => Number(point.score)))
      .filter(v => Number.isFinite(v));
    const xMin = Math.min(...times), xMax = Math.max(...times);
    let yMin = rankMode ? 1 : Math.min(...vals), yMax = rankMode ? Math.max(100, Math.ceil(Math.max(...selectedTrend.map(p => Number(p.rank))) / 50) * 50) : Math.max(...vals);
    const currentMean = rankMode || system === "jpar" ? null : scoreMeanFor(system, minEvents);
    if (currentMean != null) {{
      yMin = Math.min(yMin, currentMean);
      yMax = Math.max(yMax, currentMean);
    }}
    if (!rankMode) {{ const yPad = (yMax - yMin) * 0.08 || 1; yMin -= yPad; yMax += yPad; }}
    const x0 = pad.left, x1 = width - pad.right, y0 = height - pad.bottom, y1 = pad.top;
    const paths = rankMode ? "" : keys.map(k => `<path d="${{pathFor(pts, xMin, xMax, yMin, yMax, width, height, pad, k)}}" fill="none" stroke="${{colors[k]}}" stroke-width="${{k === "p50" ? 2.6 : 1.5}}" opacity="${{k === "p50" ? 1 : 0.9}}"/>`).join("");
    const selectedPath = !selectedTrend.length ? "" : individualTrendPath(selectedTrend, xMin, xMax, yMin, yMax, width, height, pad, rankMode ? "rank" : "score", height - pad.bottom, pad.top);
    const selectedDots = selectedTrend.map(point => {{
      const y = scale(Number(rankMode ? point.rank : point.score), yMin, yMax, height - pad.bottom, pad.top);
      const x = scale(new Date(point.date).getTime(), xMin, xMax, pad.left, width - pad.right);
      const tip = tooltipText(selectedPerson.name, payload.label, `${{point.date}} event ${{point.event}}`, `score: ${{Number(point.score).toFixed(4)}}`, `post-event rank: #${{point.rank}}`);
      return `<circle class="hover-target" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="4.5" fill="#7c3aed" stroke="#fff" stroke-width="1.2" data-tooltip="${{tip}}"></circle>`;
    }}).join("");
    const sliceXs = pts.map(p => scale(new Date(p.date).getTime(), xMin, xMax, pad.left, width - pad.right));
    const slices = pts.map((p, i) => {{
      const x = sliceXs[i];
      const left = i === 0 ? pad.left : (sliceXs[i - 1] + x) / 2;
      const right = i === sliceXs.length - 1 ? width - pad.right : (x + sliceXs[i + 1]) / 2;
      const tip = tooltipText(
        payload.label,
        `${{p.date}} event ${{p.event}}`,
        `people: ${{p.n}}`,
        ...keys.map(k => `${{k}}: ${{Number(p[k]).toFixed(4)}}`)
      );
      return `<rect class="hover-target hover-slice" x="${{left.toFixed(1)}}" y="${{pad.top}}" width="${{Math.max(4, right - left).toFixed(1)}}" height="${{(height - pad.bottom - pad.top).toFixed(1)}}" data-tooltip="${{tip}}"></rect><line x1="${{x.toFixed(1)}}" y1="${{pad.top}}" x2="${{x.toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#2563eb" stroke-width="1" opacity="0.05"/>`;
    }}).join("");
    const yTicks = niceTicks(yMin, yMax, 4).map(t => {{
      const y = scale(t, yMin, yMax, height - pad.bottom, pad.top);
      return `<line x1="${{x0}}" y1="${{y.toFixed(1)}}" x2="${{x1}}" y2="${{y.toFixed(1)}}" stroke="#e5e7eb"/><text x="4" y="${{(y + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{rankMode ? "#" + Math.round(t) : t.toFixed(2)}}</text>`;
    }}).join("");
    const xTicks = dateTicks(xMin, xMax, 4).map(d => {{
      const x = scale(d.getTime(), xMin, xMax, pad.left, width - pad.right);
      return `<line x1="${{x.toFixed(1)}}" y1="${{y0}}" x2="${{x.toFixed(1)}}" y2="${{y0 + 4}}" stroke="#6b7280"/><text x="${{(x - 28).toFixed(1)}}" y="${{height - 10}}" font-size="10" fill="#6b7280">${{fmtDate(d).slice(5)}}</text>`;
    }}).join("");
    const meanLine = currentMean == null ? "" : (() => {{
      const y = scale(currentMean, yMin, yMax, height - pad.bottom, pad.top);
      const tip = tooltipText(payload.label, `current mean score: ${{currentMean.toFixed(4)}}`);
      return `<line class="hover-target" x1="${{x0}}" y1="${{y.toFixed(1)}}" x2="${{x1}}" y2="${{y.toFixed(1)}}" stroke="#111827" stroke-width="5" stroke-dasharray="3 4" opacity="0.22" data-tooltip="${{tip}}"></line><line x1="${{x0}}" y1="${{y.toFixed(1)}}" x2="${{x1}}" y2="${{y.toFixed(1)}}" stroke="#111827" stroke-width="1" stroke-dasharray="3 4" opacity="0.55"></line><text x="${{x1 - 72}}" y="${{(y - 4).toFixed(1)}}" font-size="9" fill="#374151">mean ${{currentMean.toFixed(2)}}</text>`;
    }})();
    const labels = rankMode ? `<text x="${{pad.left}}" y="10" font-size="10" fill="#7c3aed">selected person's post-event rank</text>` : `${{keys.map((k, i) => `<text x="${{pad.left + i * 46}}" y="10" font-size="10" fill="${{colors[k]}}">${{k}}</text>`).join("")}}${{selectedTrend.length ? `<text x="${{pad.left + 240}}" y="10" font-size="10" fill="#7c3aed">selected person</text>` : ""}}`;
    return `<div class="plot-card"><div class="plot-title">${{payload.label}}</div><svg viewBox="0 0 ${{width}} ${{height}}">
      ${{yTicks}}
      <line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y0}}" stroke="#9ca3af"/>
      <line x1="${{x0}}" y1="${{y0}}" x2="${{x0}}" y2="${{y1}}" stroke="#9ca3af"/>
      ${{xTicks}}${{meanLine}}${{paths}}${{selectedPath ? `<path d="${{selectedPath}}" fill="none" stroke="#7c3aed" stroke-width="3.4"/>` : ""}}${{slices}}${{selectedDots}}${{labels}}
    </svg></div>`;
  }}).join("");
  document.getElementById("driftMeta").textContent = latestN
    ? `latest event distribution includes up to ${{latestN}} people with at least ${{minEvents}} event${{minEvents === 1 ? "" : "s"}} as of that date`
    : `no drift samples with at least ${{minEvents}} event${{minEvents === 1 ? "" : "s"}}`;
}}

function renderHistPlots() {{
  const container = document.getElementById("histPlots");
  const minEvents = Number(document.getElementById("histMinEvents").value || 1);
  const selectedRow = selectedMemberKey ? rows.find(row => row._member_key === selectedMemberKey) : null;
  let latestN = 0;
  container.innerHTML = Object.entries(histData).filter(([system]) => selectedHistSystems.has(system)).map(([system, payload]) => {{
    const hist = histogramFor(payload, minEvents);
    const bins = hist.bins || [], counts = hist.counts || [];
    if (!bins.length) return "";
    latestN = Math.max(latestN, hist.n || 0);
    const width = 420, height = 260, pad = {{left: 42, right: 12, top: 12, bottom: 34}};
    const selectedScore = selectedRow ? Number(selectedRow[system]) : NaN;
    const domainScores = Number.isFinite(selectedScore) ? bins.concat([selectedScore]) : bins;
    const xMin = Math.min(...domainScores), xMax = Math.max(...domainScores), yMax = Math.max(...counts);
    const barW = Math.max(1, (width - pad.left - pad.right) / bins.length - 1);
    const bars = bins.map((b, i) => {{
      const x = scale(b, xMin, xMax, pad.left, width - pad.right);
      const y = scale(counts[i], 0, yMax, height - pad.bottom, pad.top);
      const h = height - pad.bottom - y;
      const tip = tooltipText(payload.label, `people: ${{hist.n}}`, `score bin: ${{Number(b).toFixed(4)}}`, `count: ${{counts[i]}}`);
      return `<rect class="hover-target" x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{barW.toFixed(1)}}" height="${{h.toFixed(1)}}" fill="#2563eb" opacity="0.78" data-tooltip="${{tip}}"></rect>`;
    }}).join("");
    const yTicks = niceTicks(0, yMax, 4).map(t => {{
      const y = scale(t, 0, yMax, height - pad.bottom, pad.top);
      return `<line x1="${{pad.left}}" y1="${{y.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y.toFixed(1)}}" stroke="#e5e7eb"/><text x="4" y="${{(y + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{Math.round(t)}}</text>`;
    }}).join("");
    const xTicks = niceTicks(xMin, xMax, 4).map(t => {{
      const x = scale(t, xMin, xMax, pad.left, width - pad.right);
      return `<line x1="${{x.toFixed(1)}}" y1="${{height - pad.bottom}}" x2="${{x.toFixed(1)}}" y2="${{height - pad.bottom + 4}}" stroke="#6b7280"/><text x="${{(x - 15).toFixed(1)}}" y="${{height - 10}}" font-size="10" fill="#6b7280">${{t.toFixed(1)}}</text>`;
    }}).join("");
    const selectedLine = Number.isFinite(selectedScore) ? (() => {{
      const x = scale(Math.max(xMin, Math.min(xMax, selectedScore)), xMin, xMax, pad.left, width - pad.right);
      const tip = tooltipText(selectedRow.full_name, payload.label, `score: ${{selectedScore.toFixed(4)}}`);
      return `<line class="hover-target" x1="${{x.toFixed(1)}}" y1="${{pad.top}}" x2="${{x.toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#7c3aed" stroke-width="3" data-tooltip="${{tip}}"></line>`;
    }})() : "";
    return `<div class="plot-card"><div class="plot-title">${{payload.label}}</div><svg viewBox="0 0 ${{width}} ${{height}}">
      ${{yTicks}}
      <line x1="${{pad.left}}" y1="${{height - pad.bottom}}" x2="${{width - pad.right}}" y2="${{height - pad.bottom}}" stroke="#9ca3af"/>
      <line x1="${{pad.left}}" y1="${{height - pad.bottom}}" x2="${{pad.left}}" y2="${{pad.top}}" stroke="#9ca3af"/>
      ${{xTicks}}${{bars}}${{selectedLine}}
    </svg></div>`;
  }}).join("");
  document.getElementById("histMeta").textContent = latestN
    ? `${{latestN}} people with at least ${{minEvents}} event${{minEvents === 1 ? "" : "s"}}`
    : `no histogram samples with at least ${{minEvents}} event${{minEvents === 1 ? "" : "s"}}`;
}}

function renderRankScatterPlots() {{
  const container = document.getElementById("scatterPlots");
  const jparCol = columns.find(c => c.key === "jpar_rank");
  const rankCols = scatterColumns.filter(c => columns.some(col => col.key === c.key) && selectedScatterSystems.has(c.key));
  const minEvents = Number(document.getElementById("scatterMinEvents").value || 1);
  const scatterRows = rows.filter(r => Number(r.events || 0) >= minEvents || r._member_key === selectedMemberKey);
  document.getElementById("scatterMeta").textContent = `${{scatterRows.length}} people with at least ${{minEvents}} event${{minEvents === 1 ? "" : "s"}}`;
  if (!jparCol || !rankCols.length) {{
    container.innerHTML = "";
    return;
  }}
  container.innerHTML = rankCols.map(col => {{
    const pts = scatterRows
      .map(r => ({{
        memberKey: r._member_key,
        name: r.full_name || "",
        events: Number(r.events || 0),
        x: Number(r.jpar_rank),
        y: Number(r[col.key]),
      }}))
      .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!pts.length) return "";
    const width = 520, height = 360, pad = {{left: 46, right: 14, top: 12, bottom: 42}};
    const maxRank = Math.ceil(Math.max(...pts.flatMap(p => [p.x, p.y])) / 100) * 100;
    const rankMax = Math.max(100, maxRank);
    const x0 = pad.left, x1 = width - pad.right, yTop = pad.top, yBottom = height - pad.bottom;
    const xTicks = niceTicks(1, rankMax, 4).map(t => {{
      const x = scale(t, 1, rankMax, x0, x1);
      return `<line x1="${{x.toFixed(1)}}" y1="${{yBottom}}" x2="${{x.toFixed(1)}}" y2="${{yBottom + 4}}" stroke="#6b7280"/><text x="${{(x - 12).toFixed(1)}}" y="${{height - 12}}" font-size="10" fill="#6b7280">${{Math.round(t)}}</text>`;
    }}).join("");
    const yTicks = niceTicks(1, rankMax, 4).map(t => {{
      const y = scale(t, 1, rankMax, yBottom, yTop);
      return `<line x1="${{x0}}" y1="${{y.toFixed(1)}}" x2="${{x1}}" y2="${{y.toFixed(1)}}" stroke="#e5e7eb"/><text x="4" y="${{(y + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{Math.round(t)}}</text>`;
    }}).join("");
    const diagonal = `<line x1="${{x0}}" y1="${{yBottom}}" x2="${{x1}}" y2="${{yTop}}" stroke="#dc2626" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.82"/>`;
    const points = pts.map(p => {{
      const x = scale(p.x, 1, rankMax, x0, x1);
      const y = scale(p.y, 1, rankMax, yBottom, yTop);
      const tip = tooltipText(p.name, `events: ${{p.events}}`, `JPAR rank: ${{Math.round(p.x)}}`, `${{col.label}} rank: ${{Math.round(p.y)}}`);
      const selected = p.memberKey === selectedMemberKey;
      return `<circle class="hover-target" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="${{selected ? 7 : 3}}" fill="${{selected ? "#7c3aed" : "#2563eb"}}" stroke="${{selected ? "#111827" : "none"}}" stroke-width="${{selected ? 1.4 : 0}}" opacity="${{selected ? 1 : 0.45}}" data-tooltip="${{tip}}"></circle>`;
    }}).join("");
    return `<div class="plot-card"><div class="plot-title">JPAR vs ${{col.label}}</div><svg viewBox="0 0 ${{width}} ${{height}}">
      ${{yTicks}}
      <line x1="${{x0}}" y1="${{yBottom}}" x2="${{x1}}" y2="${{yBottom}}" stroke="#9ca3af"/>
      <line x1="${{x0}}" y1="${{yBottom}}" x2="${{x0}}" y2="${{yTop}}" stroke="#9ca3af"/>
      ${{xTicks}}${{diagonal}}${{points}}
      <text x="${{(width / 2 - 34).toFixed(1)}}" y="${{height - 2}}" font-size="10" fill="#374151">JPAR rank</text>
      <text transform="translate(10 ${{(height / 2 + 34).toFixed(1)}}) rotate(-90)" font-size="10" fill="#374151">${{col.label}}</text>
    </svg></div>`;
  }}).join("");
}}

function formatTime(seconds) {{
  const total = Math.round(Number(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return [h, m, s].map(v => String(v).padStart(2, "0")).join(":");
}}

function individualTrendPath(points, xMin, xMax, yMin, yMax, width, height, pad, valueKey, rangeTop, rangeBottom) {{
  return points.map((p, i) => {{
    const x = scale(new Date(p.date).getTime(), xMin, xMax, pad.left, width - pad.right);
    const y = scale(p[valueKey], yMin, yMax, rangeTop, rangeBottom);
    return `${{i === 0 ? "M" : "L"}}${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(" ");
}}

function unusedRenderIndividualData() {{
  const person = selectedMemberKey ? individualData[selectedMemberKey] : null;
  const title = document.getElementById("individualTitle");
  const meta = document.getElementById("individualMeta");
  const plotContainer = document.getElementById("individualPlots");
  const eventTable = document.getElementById("individualEventTable");
  if (!person) {{
    title.textContent = "Select a person";
    meta.textContent = "";
    plotContainer.innerHTML = "";
    eventTable.innerHTML = "";
    return;
  }}
  title.textContent = person.name;
  meta.textContent = individualView === "score"
    ? `${{person.events.length}} ranked event${{person.events.length === 1 ? "" : "s"}}. Colored percentile lines use the full as-of population; the thick blue line is this person's score.`
    : `${{person.events.length}} ranked event${{person.events.length === 1 ? "" : "s"}}. Rank mode shows only this person's post-event trajectory; lower rank is better.`;
  const colors = {{p10: "#2563eb", p25: "#60a5fa", p50: "#111827", p75: "#f97316", p90: "#dc2626"}};
  const percentileKeys = ["p10", "p25", "p50", "p75", "p90"];
  plotContainer.innerHTML = Object.entries(driftData).map(([system, payload]) => {{
    const trend = person.trends[system] || [];
    const pts = percentileRowsFor(payload, 1);
    if (!pts.length || !trend.length) return "";
    const width = 460, height = 300, pad = {{left: 44, right: 44, top: 16, bottom: 38}};
    const times = pts.map(p => new Date(p.date).getTime());
    const allScores = pts.flatMap(p => percentileKeys.map(k => p[k])).concat(trend.map(p => Number(p.score))).filter(Number.isFinite);
    const xMin = Math.min(...times), xMax = Math.max(...times);
    const rankMax = Math.max(10, Math.ceil(Math.max(...trend.map(p => Number(p.rank))) / 50) * 50);
    const x0 = pad.left, x1 = width - pad.right, yTop = pad.top, yBottom = height - pad.bottom;
    let yMin = Math.min(...allScores), yMax = Math.max(...allScores);
    const yPad = (yMax - yMin) * 0.08 || 1;
    yMin -= yPad; yMax += yPad;
    const scoreMode = individualView === "score";
    const percentilePaths = scoreMode ? percentileKeys.map(k => `<path d="${{pathFor(pts, xMin, xMax, yMin, yMax, width, height, pad, k)}}" fill="none" stroke="${{colors[k]}}" stroke-width="${{k === "p50" ? 2.4 : 1.3}}" opacity="${{k === "p50" ? 0.8 : 0.64}}"/>`).join("") : "";
    const personalPath = scoreMode
      ? individualTrendPath(trend, xMin, xMax, yMin, yMax, width, height, pad, "score", yBottom, yTop)
      : individualTrendPath(trend, xMin, xMax, 1, rankMax, width, height, pad, "rank", yTop, yBottom);
    const yTicks = scoreMode
      ? niceTicks(yMin, yMax, 4).map(t => {{ const y = scale(t, yMin, yMax, yBottom, yTop); return `<line x1="${{x0}}" y1="${{y.toFixed(1)}}" x2="${{x1}}" y2="${{y.toFixed(1)}}" stroke="#e5e7eb"/><text x="3" y="${{(y + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{t.toFixed(2)}}</text>`; }}).join("")
      : niceTicks(1, rankMax, 4).map(t => {{ const y = scale(t, 1, rankMax, yTop, yBottom); return `<line x1="${{x0}}" y1="${{y.toFixed(1)}}" x2="${{x1}}" y2="${{y.toFixed(1)}}" stroke="#e5e7eb"/><text x="3" y="${{(y + 3).toFixed(1)}}" font-size="10" fill="#6b7280">#${{Math.round(t)}}</text>`; }}).join("");
    const xTicks = dateTicks(xMin, xMax, 4).map(d => {{ const x = scale(d.getTime(), xMin, xMax, x0, x1); return `<line x1="${{x.toFixed(1)}}" y1="${{yBottom}}" x2="${{x.toFixed(1)}}" y2="${{yBottom + 4}}" stroke="#6b7280"/><text x="${{(x - 25).toFixed(1)}}" y="${{height - 11}}" font-size="10" fill="#6b7280">${{fmtDate(d).slice(5)}}</text>`; }}).join("");
    const dots = trend.map(p => {{
      const x = scale(new Date(p.date).getTime(), xMin, xMax, x0, x1);
      const y = scoreMode ? scale(Number(p.score), yMin, yMax, yBottom, yTop) : scale(Number(p.rank), 1, rankMax, yTop, yBottom);
      const impact = system === "jpar" ? impactData.find(e => e.event === p.event) : null;
      const calibration = impact && Number.isFinite(Number(impact.calibration_multiplier))
        ? [`calibration multiplier: ${{Number(impact.calibration_multiplier).toFixed(3)}}`, `median JPAR delta: ${{Number(impact.median_jpar_shift).toFixed(3)}}`, `anchors: ${{impact.returning_anchors}}`]
        : [];
      const fill = "#1d4ed8";
      const tip = tooltipText(person.name, `${{payload.label}}`, `${{p.date}} event ${{p.event}}`, `score: ${{Number(p.score).toFixed(4)}}`, `post-event rank: #${{p.rank}}`, ...calibration);
      return `<circle class="hover-target" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="4" fill="${{fill}}" data-tooltip="${{tip}}"></circle>`;
    }}).join("");
    const legend = scoreMode ? `<text x="${{x0}}" y="10" font-size="10" fill="#1d4ed8">selected score</text>` : `<text x="${{x0}}" y="10" font-size="10" fill="#1d4ed8">selected post-event rank</text>`;
    return `<div class="plot-card"><div class="plot-title">${{payload.label}}</div><svg viewBox="0 0 ${{width}} ${{height}}">${{yTicks}}<line x1="${{x0}}" y1="${{yBottom}}" x2="${{x1}}" y2="${{yBottom}}" stroke="#9ca3af"/><line x1="${{x0}}" y1="${{yBottom}}" x2="${{x0}}" y2="${{yTop}}" stroke="#9ca3af"/>${{xTicks}}${{percentilePaths}}<path d="${{personalPath}}" fill="none" stroke="#1d4ed8" stroke-width="3.5" opacity="0.95"/>${{dots}}${{legend}}</svg></div>`;
  }}).join("");
  const header = `<thead><tr><th>Date</th><th>Event</th><th>Place</th><th>Time</th><th>JPAR multiplier</th><th>Median JPAR delta</th>${{individualSystems.map(s => `<th>${{s.label}} rank after event</th>`).join("")}}</tr></thead>`;
  const body = `<tbody>${{person.events.map(e => {{ const impact = impactData.find(x => x.event === e.event); return `<tr><td>${{e.date}}</td><td title="${{escapeAttr(e.event_name)}}">${{escapeAttr(e.event_name)}}</td><td>${{e.place == null ? "" : e.place}}</td><td>${{formatTime(e.time_seconds)}}</td><td>${{impact && Number.isFinite(Number(impact.calibration_multiplier)) ? Number(impact.calibration_multiplier).toFixed(3) : ""}}</td><td>${{impact && Number.isFinite(Number(impact.median_jpar_shift)) ? Number(impact.median_jpar_shift).toFixed(3) : ""}}</td>${{individualSystems.map(s => `<td>${{e.ranks[s.key] == null ? "" : "#" + e.ranks[s.key]}}</td>`).join("")}}</tr>`; }}).join("")}}</tbody>`;
  eventTable.innerHTML = header + body;
}}

function renderSelectedPersonEvents() {{
  const panel = document.getElementById("selectedEventsPanel");
  const eventTable = document.getElementById("individualEventTable");
  const person = selectedMemberKey ? individualData[selectedMemberKey] : null;
  if (!person) {{
    panel.hidden = true;
    eventTable.innerHTML = "";
    return;
  }}
  panel.hidden = false;
  document.getElementById("selectedEventsTitle").textContent = `${{person.name}}: Event History`;
  const header = `<thead><tr><th>Date</th><th>Event</th><th>Place</th><th>Time</th><th>JPAR multiplier</th><th>Median JPAR delta</th>${{individualSystems.map(s => `<th>${{s.label}} rank after event</th>`).join("")}}</tr></thead>`;
  const body = `<tbody>${{person.events.map(e => {{ const impact = impactData.find(x => x.event === e.event); return `<tr><td>${{e.date}}</td><td title="${{escapeAttr(e.event_name)}}">${{escapeAttr(e.event_name)}}</td><td>${{e.place == null ? "" : e.place}}</td><td>${{formatTime(e.time_seconds)}}</td><td>${{impact && Number.isFinite(Number(impact.calibration_multiplier)) ? Number(impact.calibration_multiplier).toFixed(3) : ""}}</td><td>${{impact && Number.isFinite(Number(impact.median_jpar_shift)) ? Number(impact.median_jpar_shift).toFixed(3) : ""}}</td>${{individualSystems.map(s => `<td>${{e.ranks[s.key] == null ? "" : "#" + e.ranks[s.key]}}</td>`).join("")}}</tr>`; }}).join("")}}</tbody>`;
  eventTable.innerHTML = header + body;
}}

function renderEventImpacts() {{
  const valid = impactData.filter(e => Number(e.participants || 0) > 0 && Number.isFinite(Number(e.calibration_multiplier)));
  const penalty = [...valid].sort((a, b) => a.calibration_multiplier - b.calibration_multiplier).slice(0, 10);
  const boost = [...valid].sort((a, b) => b.calibration_multiplier - a.calibration_multiplier).slice(0, 10);
  const renderTable = (id, data) => {{
    document.getElementById(id).innerHTML = `<thead><tr><th>Date</th><th>Event</th><th>Players</th><th>Anchors</th><th>Multiplier</th><th>Median JPAR delta</th></tr></thead><tbody>${{data.map(e => `<tr><td>${{e.date}}</td><td title="${{escapeAttr(e.event_name)}}">${{escapeAttr(e.event_name)}}</td><td>${{e.participants}}</td><td>${{e.returning_anchors}}</td><td>${{Number(e.calibration_multiplier).toFixed(3)}}</td><td>${{Number(e.median_jpar_shift).toFixed(3)}}</td></tr>`).join("")}}</tbody>`;
  }};
  renderTable("penaltyEventsTable", penalty);
  renderTable("boostEventsTable", boost);
  document.getElementById("impactMeta").textContent = `${{valid.length}} of ${{impactData.length}} events had returning anchors and therefore a non-default JPAR calibration. Multiplier = 1.00 means no calibration shift.`;
}}

function calibrationEffectStyle(value) {{
  if (!Number.isFinite(Number(value))) return "";
  const magnitude = Math.min(1, Math.abs(Number(value)) / 0.5);
  const hue = Number(value) < 0 ? 142 : 4;
  const light = 96 - magnitude * 30;
  return `background-color: hsl(${{hue}}, 70%, ${{light}}%); font-weight: 650;`;
}}

function renderCumulativeCalibrationTable() {{
  const table = document.getElementById("cumulativeCalibrationTable");
  const minEvents = Number(document.getElementById("cumulativeMinEvents").value || 1);
  const maxJpar = Number(document.getElementById("cumulativeMaxJpar").value || cumulativeJparMax);
  const cols = [
    ["name", "Name", "text"],
    ["events", "Events", "number"],
    ["calibrated_events", "Calibrated Events", "number"],
    ["cumulative_event_delta", "Cumulative Event Delta", "number"],
    ["mean_event_delta", "Mean Event Delta", "number"],
    ["final_calibration_effect", "Final Calibration Effect", "number"],
  ];
  const data = cumulativeCalibrationData
    .filter(row => Number(row.events) >= minEvents && (!Number.isFinite(maxJpar) || Number(row.jpar) <= maxJpar))
    .sort((a, b) => {{
    const av = a[cumulativeSortKey], bv = b[cumulativeSortKey];
    const cmp = typeof av === "string" ? av.localeCompare(bv) : Number(av) - Number(bv);
    return cmp * (cumulativeSortDir === "asc" ? 1 : -1);
  }});
  const header = `<thead><tr>${{cols.map(([key, label]) => `<th data-cumulative-key="${{key}}">${{label}}${{key === cumulativeSortKey ? (cumulativeSortDir === "asc" ? " ▲" : " ▼") : ""}}</th>`).join("")}}</tr></thead>`;
  const body = `<tbody>${{data.map(row => `<tr>${{cols.map(([key, , kind]) => {{ const value = row[key]; const style = key.includes("delta") || key === "final_calibration_effect" ? calibrationEffectStyle(value) : ""; return `<td style="${{style}}">${{kind === "number" && key !== "events" && key !== "calibrated_events" ? Number(value).toFixed(4) : value}}</td>`; }}).join("")}}</tr>`).join("")}}</tbody>`;
  table.innerHTML = header + body;
  document.getElementById("cumulativeMeta").textContent = `${{data.length}} of ${{cumulativeCalibrationData.length}} people`;
  document.getElementById("cumulativeMaxJparValue").textContent = maxJpar.toFixed(2);
  renderCumulativeJparHistogram(minEvents, maxJpar);
  table.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {{
    const key = th.dataset.cumulativeKey;
    if (cumulativeSortKey === key) cumulativeSortDir = cumulativeSortDir === "asc" ? "desc" : "asc";
    else {{ cumulativeSortKey = key; cumulativeSortDir = key === "name" ? "asc" : "asc"; }}
    renderCumulativeCalibrationTable();
  }}));
}}

function renderCumulativeJparHistogram(minEvents, maxJpar) {{
  const svg = document.getElementById("cumulativeJparHistogram");
  const values = cumulativeCalibrationData
    .filter(row => Number(row.events) >= minEvents)
    .map(row => Number(row.jpar))
    .filter(value => Number.isFinite(value) && value <= cumulativeJparMax);
  const width = 620, height = 90, pad = {{left: 32, right: 12, top: 8, bottom: 22}}, bins = 40;
  const counts = Array.from({{length: bins}}, () => 0);
  values.forEach(value => {{
    const index = Math.min(bins - 1, Math.max(0, Math.floor((value / cumulativeJparMax) * bins)));
    counts[index] += 1;
  }});
  const yMax = Math.max(1, ...counts);
  const barWidth = (width - pad.left - pad.right) / bins;
  const bars = counts.map((count, index) => {{
    const x = pad.left + index * barWidth;
    const y = scale(count, 0, yMax, height - pad.bottom, pad.top);
    const cutoff = ((index + 0.5) / bins) * cumulativeJparMax;
    const fill = cutoff <= maxJpar ? "#2563eb" : "#cbd5e1";
    return `<rect x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{Math.max(1, barWidth - 1).toFixed(1)}}" height="${{(height - pad.bottom - y).toFixed(1)}}" fill="${{fill}}"/>`;
  }}).join("");
  const cutoffX = scale(maxJpar, 0, cumulativeJparMax, pad.left, width - pad.right);
  svg.innerHTML = `<line x1="${{pad.left}}" y1="${{height - pad.bottom}}" x2="${{width - pad.right}}" y2="${{height - pad.bottom}}" stroke="#94a3b8"/>${{bars}}<line x1="${{cutoffX.toFixed(1)}}" y1="${{pad.top}}" x2="${{cutoffX.toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#111827" stroke-width="2"/><text x="${{pad.left}}" y="${{height - 5}}" font-size="10" fill="#64748b">0</text><text x="${{width - pad.right - 28}}" y="${{height - 5}}" font-size="10" fill="#64748b">${{cumulativeJparMax.toFixed(1)}}</text><text x="${{Math.min(width - 100, cutoffX + 5).toFixed(1)}}" y="${{pad.top + 10}}" font-size="10" fill="#111827">cutoff ${{maxJpar.toFixed(2)}}</text>`;
}}

function renderFeedbackSection() {{
  const container = document.getElementById("feedbackPlots");
  const participationContainer = document.getElementById("feedbackParticipationPlot");
  const rows = feedbackData.filter(row => row.date);
  if (!rows.length) return;
  const width = 1100, height = 360, pad = {{left: 52, right: 18, top: 24, bottom: 42}};
  const times = rows.map(row => new Date(row.date).getTime());
  const xMin = Math.min(...times), xMax = Math.max(...times);
  const x = row => scale(new Date(row.date).getTime(), xMin, xMax, pad.left, width - pad.right);
  const xTicks = dateTicks(xMin, xMax, 4).map(d => {{ const px = scale(d.getTime(), xMin, xMax, pad.left, width - pad.right); return `<text x="${{(px - 25).toFixed(1)}}" y="${{height - 10}}" font-size="10" fill="#6b7280">${{fmtDate(d).slice(5)}}</text>`; }}).join("");
  const slices = rows.map((row, index) => {{
    const current = x(row), previous = index ? x(rows[index - 1]) : pad.left, next = index < rows.length - 1 ? x(rows[index + 1]) : width - pad.right;
    const left = index ? (previous + current) / 2 : pad.left;
    const right = index < rows.length - 1 ? (current + next) / 2 : width - pad.right;
    const tip = tooltipText(row.event_name, `${{row.date}} event ${{row.event}}`, `participants: ${{row.participants}}`, `returning anchors: ${{row.anchors}}`, row.raw_median == null ? "" : `raw event_jpar median: ${{Number(row.raw_median).toFixed(3)}}`, row.adjusted_median == null ? "" : `adjusted event_jpar median: ${{Number(row.adjusted_median).toFixed(3)}}`, row.as_of_jpar_median == null ? "" : `as-of latest JPAR median: ${{Number(row.as_of_jpar_median).toFixed(3)}}`);
    return `<rect class="hover-target hover-slice" x="${{left.toFixed(1)}}" y="${{pad.top}}" width="${{Math.max(3, right - left).toFixed(1)}}" height="${{(height - pad.top - pad.bottom).toFixed(1)}}" data-tooltip="${{tip}}"></rect>`;
  }}).join("");
  const timeline = (key, label, center, color) => {{
    const values = rows.map(row => Number(row[key])).filter(Number.isFinite);
    let yMin = Math.min(...values, center), yMax = Math.max(...values, center);
    const range = (yMax - yMin) * 0.10 || 1;
    yMin -= range; yMax += range;
    const path = rows.filter(row => Number.isFinite(Number(row[key]))).map((row, index) => `${{index ? "L" : "M"}}${{x(row).toFixed(1)}},${{scale(Number(row[key]), yMin, yMax, height - pad.bottom, pad.top).toFixed(1)}}`).join(" ");
    const ticks = niceTicks(yMin, yMax, 4).map(t => {{ const py = scale(t, yMin, yMax, height - pad.bottom, pad.top); return `<line x1="${{pad.left}}" y1="${{py.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{py.toFixed(1)}}" stroke="#e5e7eb"/><text x="2" y="${{(py + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{t.toFixed(2)}}</text>`; }}).join("");
    const centerY = scale(center, yMin, yMax, height - pad.bottom, pad.top);
    return `<div class="plot-card"><div class="plot-title">${{label}}</div><svg viewBox="0 0 ${{width}} ${{height}}">${{ticks}}<line x1="${{pad.left}}" y1="${{centerY.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{centerY.toFixed(1)}}" stroke="#111827" stroke-dasharray="3 4" opacity="0.45"/><path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="2.6"/>${{slices}}${{xTicks}}</svg></div>`;
  }};
  const medianMin = 0.6, medianMax = 3.0;
  const medianPath = key => rows.filter(row => Number.isFinite(Number(row[key]))).map((row, index) => {{ const value = Math.max(medianMin, Math.min(medianMax, Number(row[key]))); return `${{index ? "L" : "M"}}${{x(row).toFixed(1)}},${{scale(value, medianMin, medianMax, height - pad.bottom, pad.top).toFixed(1)}}`; }}).join(" ");
  const medianDots = (key, color) => rows.filter(row => Number.isFinite(Number(row[key]))).map(row => {{ const value = Math.max(medianMin, Math.min(medianMax, Number(row[key]))); return `<circle cx="${{x(row).toFixed(1)}}" cy="${{scale(value, medianMin, medianMax, height - pad.bottom, pad.top).toFixed(1)}}" r="5" fill="${{color}}" stroke="#ffffff" stroke-width="1.4"/>`; }}).join("");
  const medianTicks = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0].map(t => {{ const py = scale(t, medianMin, medianMax, height - pad.bottom, pad.top); return `<line x1="${{pad.left}}" y1="${{py.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{py.toFixed(1)}}" stroke="#e5e7eb"/><text x="2" y="${{(py + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{t.toFixed(2)}}</text>`; }}).join("");
  const medians = `<div class="plot-card"><div class="plot-title">Raw Event Scores vs Calibration-Adjusted Scores</div><svg viewBox="0 0 ${{width}} ${{height}}">${{medianTicks}}<line x1="${{pad.left}}" y1="${{scale(1, medianMin, medianMax, height - pad.bottom, pad.top).toFixed(1)}}" x2="${{width - pad.right}}" y2="${{scale(1, medianMin, medianMax, height - pad.bottom, pad.top).toFixed(1)}}" stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.85"/><path d="${{medianPath("raw_median")}}" fill="none" stroke="#2563eb" stroke-width="2.8"/><path d="${{medianPath("adjusted_median")}}" fill="none" stroke="#f97316" stroke-width="2.8"/><path d="${{medianPath("as_of_jpar_median")}}" fill="none" stroke="#16a34a" stroke-width="2.8"/>${{medianDots("raw_median", "#2563eb")}}${{medianDots("adjusted_median", "#f97316")}}${{medianDots("as_of_jpar_median", "#16a34a")}}${{slices}}${{xTicks}}<text x="${{pad.left}}" y="10" font-size="10" fill="#2563eb">raw event_jpar median</text><text x="${{pad.left + 132}}" y="10" font-size="10" fill="#f97316">adjusted event_jpar median</text><text x="${{pad.left + 300}}" y="10" font-size="10" fill="#16a34a">as-of latest JPAR median</text></svg></div>`;
  const entrantsMax = Math.max(1, ...rows.map(row => row.entrants));
  const barWidth = Math.max(3, (width - pad.left - pad.right) / rows.length - 2);
  const propagationBars = rows.map(row => {{
    const px = x(row) - barWidth / 2;
    const totalY = scale(row.entrants, 0, entrantsMax, height - pad.bottom, pad.top);
    const laterY = scale(row.entrants_later_anchor, 0, entrantsMax, height - pad.bottom, pad.top);
    return `<rect x="${{px.toFixed(1)}}" y="${{totalY.toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{(height - pad.bottom - totalY).toFixed(1)}}" fill="#cbd5e1"/><rect x="${{px.toFixed(1)}}" y="${{laterY.toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{(height - pad.bottom - laterY).toFixed(1)}}" fill="#2563eb"/>`;
  }}).join("");
  const propagationTicks = niceTicks(0, entrantsMax, 4).map(t => {{ const py = scale(t, 0, entrantsMax, height - pad.bottom, pad.top); return `<line x1="${{pad.left}}" y1="${{py.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{py.toFixed(1)}}" stroke="#e5e7eb"/><text x="4" y="${{(py + 3).toFixed(1)}}" font-size="10" fill="#6b7280">${{Math.round(t)}}</text>`; }}).join("");
  const propagation = `<div class="plot-card"><div class="plot-title">Unanchored Entrants Who Later Anchor</div><svg viewBox="0 0 ${{width}} ${{height}}">${{propagationTicks}}${{propagationBars}}${{slices}}${{xTicks}}<text x="${{pad.left}}" y="10" font-size="10" fill="#2563eb">blue: later anchors</text><text x="${{pad.left + 100}}" y="10" font-size="10" fill="#64748b">gray: all entrants</text></svg></div>`;
  container.innerHTML = medians;
  const barHeight = 300, barPad = pad;
  const barRows = Array.from(rows.reduce((byDate, row) => {{
    const existing = byDate.get(row.date) || {{date: row.date, participants: 0, anchors: 0, entrants: 0, events: [], names: []}};
    existing.participants += Number(row.participants || 0);
    existing.anchors += Number(row.anchors || 0);
    existing.entrants += Number(row.entrants || 0);
    existing.events.push(row.event);
    existing.names.push(row.event_name);
    byDate.set(row.date, existing);
    return byDate;
  }}, new Map()).values());
  const participantMax = Math.max(1, ...barRows.map(row => Number(row.entrants || 0) + Number(row.anchors || 0)));
  const participantBarWidth = Math.max(2, Math.min(12, ((width - barPad.left - barPad.right) / barRows.length) * 0.42));
  const participationBars = barRows.map(row => {{
    const returning = Number(row.anchors || 0), newcomers = Number(row.entrants || 0);
    const px = scale(new Date(row.date).getTime(), xMin, xMax, barPad.left, width - barPad.right) - participantBarWidth / 2;
    const returningY = scale(returning, 0, participantMax, barHeight - barPad.bottom, barPad.top);
    const totalY = scale(returning + newcomers, 0, participantMax, barHeight - barPad.bottom, barPad.top);
    return `<rect x="${{px.toFixed(1)}}" y="${{returningY.toFixed(1)}}" width="${{participantBarWidth.toFixed(1)}}" height="${{(barHeight - barPad.bottom - returningY).toFixed(1)}}" fill="#16a34a"/><rect x="${{px.toFixed(1)}}" y="${{totalY.toFixed(1)}}" width="${{participantBarWidth.toFixed(1)}}" height="${{(returningY - totalY).toFixed(1)}}" fill="#dc2626"/>`;
  }}).join("");
  const participantTicks = niceTicks(0, participantMax, 4).map(t => {{ const py = scale(t, 0, participantMax, barHeight - barPad.bottom, barPad.top); return `<line x1="${{barPad.left}}" y1="${{py.toFixed(1)}}" x2="${{width - barPad.right}}" y2="${{py.toFixed(1)}}" stroke="#e5e7eb"/><text x="5" y="${{(py + 3).toFixed(1)}}" font-size="11" fill="#6b7280">${{Math.round(t)}}</text>`; }}).join("");
  const participationSlices = barRows.map((row, index) => {{
    const current = scale(new Date(row.date).getTime(), xMin, xMax, barPad.left, width - barPad.right);
    const previous = index ? scale(new Date(barRows[index - 1].date).getTime(), xMin, xMax, barPad.left, width - barPad.right) : barPad.left;
    const next = index < barRows.length - 1 ? scale(new Date(barRows[index + 1].date).getTime(), xMin, xMax, barPad.left, width - barPad.right) : width - barPad.right;
    const left = index ? (previous + current) / 2 : barPad.left;
    const right = index < rows.length - 1 ? (current + next) / 2 : width - barPad.right;
    const tip = tooltipText(`${{row.date}} (${{row.events.length}} event${{row.events.length === 1 ? "" : "s"}})`, `events: ${{row.events.join(", ")}}`, `new participants: ${{row.entrants}}`, `returning participants: ${{row.anchors}}`, `total participants: ${{row.participants}}`);
    return `<rect class="hover-target hover-slice" x="${{left.toFixed(1)}}" y="${{barPad.top}}" width="${{Math.max(3, right - left).toFixed(1)}}" height="${{(barHeight - barPad.top - barPad.bottom).toFixed(1)}}" data-tooltip="${{tip}}"></rect>`;
  }}).join("");
  const participantXTicks = dateTicks(xMin, xMax, 4).map(d => {{ const px = scale(d.getTime(), xMin, xMax, barPad.left, width - barPad.right); return `<text x="${{(px - 25).toFixed(1)}}" y="${{barHeight - 10}}" font-size="11" fill="#6b7280">${{fmtDate(d).slice(5)}}</text>`; }}).join("");
  participationContainer.innerHTML = `<div class="plot-card"><div class="plot-title">New vs Returning Puzzlers by Event</div><svg viewBox="0 0 ${{width}} ${{barHeight}}">${{participantTicks}}${{participationBars}}${{participationSlices}}${{participantXTicks}}<text x="${{barPad.left}}" y="12" font-size="11" fill="#dc2626">red: new, no prior JPAR</text><text x="${{barPad.left + 150}}" y="12" font-size="11" fill="#16a34a">green: returning</text></svg></div>`;
  const sorted = [...rows].sort((a, b) => b.entrants_later_anchor - a.entrants_later_anchor || b.entrants - a.entrants);
  const feedbackTable = document.getElementById("feedbackTable");
  if (feedbackTable) feedbackTable.innerHTML = `<thead><tr><th>Date</th><th>Event</th><th>Players</th><th>Anchors</th><th>Unanchored Entrants</th><th>Entrants Later Anchoring</th><th>Multiplier</th><th>Median Delta</th></tr></thead><tbody>${{sorted.map(row => `<tr><td>${{row.date}}</td><td title="${{escapeAttr(row.event_name)}}">${{escapeAttr(row.event_name)}}</td><td>${{row.participants}}</td><td>${{row.anchors}}</td><td>${{row.entrants}}</td><td>${{row.entrants_later_anchor}}</td><td>${{row.multiplier == null ? "" : Number(row.multiplier).toFixed(3)}}</td><td>${{row.median_delta == null ? "" : Number(row.median_delta).toFixed(3)}}</td></tr>`).join("")}}</tbody>`;
}}

const predictionColors = ["#2563eb", "#f97316", "#16a34a", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04", "#475569", "#db2777", "#65a30d", "#9333ea", "#0f766e", "#c2410c", "#4f46e5"];

function rankMap(items, valueGetter, higherIsBetter = false) {{
  const ordered = items
    .filter(item => valueGetter(item) != null && Number.isFinite(Number(valueGetter(item))))
    .sort((a, b) => (higherIsBetter ? -1 : 1) * (Number(valueGetter(a)) - Number(valueGetter(b))) || String(a.full_name).localeCompare(String(b.full_name)));
  const ranks = new Map();
  let previousValue = null;
  let previousRank = null;
  ordered.forEach((item, index) => {{
    const value = Number(valueGetter(item));
    const rank = previousValue !== null && value === previousValue ? previousRank : index + 1;
    ranks.set(item._member_key, rank);
    previousValue = value;
    previousRank = rank;
  }});
  return ranks;
}}

function incomingPredictionState(eventIndex) {{
  const state = new Map();
  for (let index = 0; index < eventIndex; index += 1) {{
    rankingTimeline[index].updates.forEach(update => {{
      const previous = state.get(update._member_key) || {{_member_key: update._member_key}};
      state.set(update._member_key, Object.assign(previous, update));
    }});
  }}
  return state;
}}

function selectedPredictionColumns() {{
  return columns.filter(column => column.kind === "rank" && selectedPredictionSystems.has(column.key));
}}

function predictionMetrics(points) {{
  const errors = points.map(point => point.predicted - point.actual);
  return {{
    mae: errors.length ? errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length : null,
    rmse: errors.length ? Math.sqrt(errors.reduce((sum, value) => sum + value * value, 0) / errors.length) : null,
    maxError: errors.length ? Math.max(...errors.map(Math.abs)) : null,
    bias: errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null,
  }};
}}

function buildPredictionComparison(eventIndex) {{
  const event = rankingTimeline[eventIndex];
  const incoming = incomingPredictionState(eventIndex);
  const entrants = event ? event.updates.filter(update => !update._state_only).map(update => ({{...update, incoming: incoming.get(update._member_key) || null}})) : [];
  const fullIncoming = [...incoming.values()];
  const systems = selectedPredictionColumns();
  const comparisons = systems.map((column, colorIndex) => {{
    const scoreKey = column.key.replace(/_rank$/, "");
    const higher = higherIsBetterScores.has(scoreKey);
    const eligible = entrants.filter(entrant => entrant.incoming && entrant.incoming[scoreKey] != null);
    const incomingRelative = rankMap(eligible, entrant => entrant.incoming[scoreKey], higher);
    const actualRelative = rankMap(eligible, entrant => entrant.event_time_seconds, false);
    const absoluteIncoming = rankMap(fullIncoming, row => row[scoreKey], higher);
    const points = eligible.map(entrant => ({{
      key: entrant._member_key,
      name: entrant.full_name,
      predicted: incomingRelative.get(entrant._member_key),
      actual: actualRelative.get(entrant._member_key),
      absolute: absoluteIncoming.get(entrant._member_key),
    }}));
    return {{column, scoreKey, color: predictionColors[colorIndex % predictionColors.length], points, ...predictionMetrics(points)}};
  }});
  return {{event, entrants, incoming, comparisons}};
}}

function buildAllPredictionComparisons() {{
  const systems = selectedPredictionColumns();
  const pooled = new Map(systems.map((column, index) => [column.key, {{column, scoreKey: column.key.replace(/_rank$/, ""), color: predictionColors[index % predictionColors.length], points: []}}]));
  rankingTimeline.forEach((event, eventIndex) => {{
    const result = buildPredictionComparison(eventIndex);
    result.comparisons.forEach(comparison => {{
      const target = pooled.get(comparison.column.key);
      const n = comparison.points.length;
      comparison.points.forEach(point => {{
        target.points.push({{
          ...point,
          predicted: n <= 1 ? 50 : 100 * (point.predicted - 1) / (n - 1),
          actual: n <= 1 ? 50 : 100 * (point.actual - 1) / (n - 1),
          event: event.event,
          eventName: event.event_name,
          eventDate: event.date,
        }});
      }});
    }});
  }});
  return [...pooled.values()].map(comparison => ({{...comparison, ...predictionMetrics(comparison.points)}}));
}}

function rollingPredictionBand(points, domainMin, domainMax) {{
  const sorted = [...points].sort((a, b) => a.predicted - b.predicted || a.actual - b.actual);
  if (!sorted.length) return [];
  let windowSize = Math.min(sorted.length, Math.max(5, Math.min(31, Math.round(Math.sqrt(sorted.length) * 2))));
  if (windowSize > 1 && windowSize % 2 === 0) windowSize -= 1;
  const half = Math.floor(windowSize / 2);
  return sorted.map((point, index) => {{
    // Keep the window centered and let it shrink naturally at both edges.
    // Shifting a full window inward makes several edge estimates reuse the
    // same observations, which creates an artificial flat shelf.
    const start = Math.max(0, index - half);
    const end = Math.min(sorted.length, index + half + 1);
    const window = sorted.slice(start, end);
    // Smooth residuals around the one-to-one line, not raw actual ranks.
    // Smoothing raw y-values forces an edge window spanning predicted ranks
    // 1–11 toward their middle y-value even when every point is perfectly on
    // the diagonal. Residual smoothing correctly preserves that diagonal.
    const residuals = window.map(item => item.actual - item.predicted).sort((a, b) => a - b);
    const middle = Math.floor(residuals.length / 2);
    const medianResidual = residuals.length % 2 ? residuals[middle] : (residuals[middle - 1] + residuals[middle]) / 2;
    const meanResidual = residuals.reduce((sum, value) => sum + value, 0) / residuals.length;
    const residualSd = Math.sqrt(residuals.reduce((sum, value) => sum + (value - meanResidual) ** 2, 0) / residuals.length);
    const center = point.predicted + medianResidual;
    return {{
      x: point.predicted,
      median: Math.max(domainMin, Math.min(domainMax, center)),
      low: Math.max(domainMin, center - residualSd),
      high: Math.min(domainMax, center + residualSd),
    }};
  }});
}}

function profileSystemLabel(key) {{
  return profileSystems.find(system => system.key === key)?.label || key;
}}

function profileTimeAssumption(key) {{
  const notes = {{
    jpar: "The incoming JPAR is the puzzler's running JPAR score immediately before this event. Because an adjusted event JPAR equals completion time divided by the event's anchor-implied typical time, that calculation is reversed: predicted time = incoming JPAR × this event's anchor-implied mean time. If the event has no anchor baseline, its observed mean time is used.",
    raw_jpar_update: "The incoming score is the puzzler's running half-update of earlier raw event time ratios. Each raw event ratio is completion time divided by that event's mean time, so the calculation is reversed: predicted time = incoming score × this event's observed mean time.",
    mean_adjusted_event_jpar: "The incoming score is the simple mean of the puzzler's earlier adjusted event JPAR ratios. It is converted back to seconds as predicted time = incoming mean ratio × this event's anchor-implied mean time. If the anchor baseline is unavailable, the observed event mean is used.",
    weighted_log_zscore: "The incoming score is the running half-update of the puzzler's earlier within-event log-time z-scores. It is placed into this event's observed log-time distribution: predicted time = exp(event mean log time + incoming score × event log-time standard deviation).",
    mean_log_zscore: "The incoming score is the simple mean of the puzzler's earlier within-event log-time z-scores. It is placed into this event's observed log-time distribution: predicted time = exp(event mean log time + incoming score × event log-time standard deviation).",
    weighted_zscore: "The incoming score is the running half-update of the puzzler's earlier raw-time event z-scores. It is placed into this event's observed raw-time distribution: predicted time = event mean time + incoming score × event time standard deviation.",
    mean_zscore: "The incoming score is the simple mean of the puzzler's earlier raw-time event z-scores. It is placed into this event's observed raw-time distribution: predicted time = event mean time + incoming score × event time standard deviation.",
    weighted_event_percentile: "The incoming score is the running half-update of the puzzler's earlier within-event percentile ranks. That percentile is looked up in this event's observed completion-time distribution; the corresponding empirical time quantile becomes the predicted time.",
    mean_event_percentile: "The incoming score is the simple mean of the puzzler's earlier within-event percentile ranks. That percentile is looked up in this event's observed completion-time distribution; the corresponding empirical time quantile becomes the predicted time.",
    weighted_normalized_rank: "The incoming score is the running half-update of earlier normalized finishing ranks, where 0 is fastest and 1 is slowest. That position is looked up in this event's observed completion-time distribution; the corresponding empirical time quantile becomes the predicted time.",
    mean_normalized_rank: "The incoming score is the simple mean of earlier normalized finishing ranks, where 0 is fastest and 1 is slowest. That position is looked up in this event's observed completion-time distribution; the corresponding empirical time quantile becomes the predicted time."
    ,sof_weighted_log_zscore: "The incoming score is the running half-update of strength-adjusted log-time z-scores. The entrants' incoming average score, using zero for debut entrants, is removed to recover the expected within-event z-score; that z-score is then placed into this event's observed log-time distribution."
    ,sof_mean_log_zscore: "The incoming score is the running mean of strength-adjusted log-time z-scores. The entrants' incoming average score, using zero for debut entrants, is removed to recover the expected within-event z-score; that z-score is then placed into this event's observed log-time distribution."
    ,sof_weighted_zscore: "The incoming score is the running half-update of strength-adjusted raw-time z-scores. The entrants' incoming average score, using zero for debut entrants, is removed to recover the expected within-event z-score; that z-score is then placed into this event's observed raw-time distribution."
    ,sof_mean_zscore: "The incoming score is the running mean of strength-adjusted raw-time z-scores. The entrants' incoming average score, using zero for debut entrants, is removed to recover the expected within-event z-score; that z-score is then placed into this event's observed raw-time distribution."
    ,sof_weighted_centered_log: "The incoming score is the running half-update of strength-adjusted centered log times. The entrants' incoming average score, using zero for debut entrants, is removed to recover the expected log-time deviation; predicted time = exp(event mean log time + that deviation). No event standard deviation is used."
    ,sof_mean_centered_log: "The incoming score is the running mean of strength-adjusted centered log times. The entrants' incoming average score, using zero for debut entrants, is removed to recover the expected log-time deviation; predicted time = exp(event mean log time + that deviation). No event standard deviation is used."
    ,external_logtime: "The incoming rating is the negative of the model's estimated difficulty-adjusted log time, so higher is faster. For this display it is converted to seconds with predicted time = exp(observed event difficulty − incoming rating). The event difficulty is estimated from this event's actual times."
    ,external_logtime_conservative: "This uses the same incoming Log-Time state but subtracts two uncertainty standard deviations from the rating. Diagnostic time = exp(observed event difficulty − conservative rating), so it is a deliberately slower uncertainty-adjusted estimate rather than the mean estimate."
    ,external_logtime_no_tier: "The incoming rating is produced by the same log-time volatility model, but USA Nationals results use the same observation variance as ordinary events. Diagnostic time = exp(observed event difficulty − incoming rating); the event difficulty is estimated from this event's actual times."
    ,external_bayesian: "The incoming score is plain Bayesian Skill mu. Entrants are ordered by mu, and the resulting predicted position is mapped to the same quantile of this event's observed time distribution."
    ,external_bayesian_conservative: "The incoming score is Bayesian Skill mu minus two sigma. Entrants are ordered by that conservative score, and the resulting predicted position is mapped to the same quantile of this event's observed time distribution."
    ,external_nationals: "The incoming ordering combines the log-time measurement with the most recent USA Nationals order among Nationals entrants. Its predicted entrant position is converted to seconds using the observed event difficulty and the underlying incoming log-time measurement."
  }};
  return `${{notes[key] || "The incoming system score is mapped onto this event's observed time distribution."}} This is an outcome-informed diagnostic, not a leakage-free time forecast.`;
}}

function profileRows(systemKey = document.getElementById("puzzlerSystem").value) {{
  const profile = puzzlerProfiles[document.getElementById("puzzlerSelect").value];
  return (profile?.events || []).map(event => ({{...event, diagnostic: event.systems[systemKey]}}));
}}

function renderProfileScatter(containerId, series, mode) {{
  const container = document.getElementById(containerId);
  const points = series.flatMap(item => item.points);
  if (!points.length) {{
    container.innerHTML = `<div class="muted" style="padding:28px 8px">Select at least one system with an incoming score.</div>`;
    return;
  }}
  const width = 620, height = 390, pad = {{left: 62, right: 18, top: Math.max(28, 18 + Math.ceil(series.length / 2) * 15), bottom: 52}};
  const all = points.flatMap(point => [point.predicted, point.actual]);
  const minimum = mode === "rank" ? 1 : Math.min(...all) * 0.94;
  const maximum = Math.max(...all) * (mode === "rank" ? 1.04 : 1.06);
  const x = value => scale(value, minimum, maximum, pad.left, width - pad.right);
  const y = value => scale(value, minimum, maximum, height - pad.bottom, pad.top);
  const ticks = niceTicks(minimum, maximum, 5);
  const tickText = value => mode === "time" ? formatTime(value) : `#${{Math.max(1, Math.round(value))}}`;
  const grid = ticks.map(value => `<line x1="${{x(value).toFixed(1)}}" y1="${{pad.top}}" x2="${{x(value).toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#e5e7eb"/><line x1="${{pad.left}}" y1="${{y(value).toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y(value).toFixed(1)}}" stroke="#e5e7eb"/><text x="${{x(value).toFixed(1)}}" y="${{height - 25}}" text-anchor="middle" font-size="10" fill="#6b7280">${{tickText(value)}}</text><text x="${{pad.left - 8}}" y="${{(y(value) + 3).toFixed(1)}}" text-anchor="end" font-size="10" fill="#6b7280">${{tickText(value)}}</text>`).join("");
  const dots = series.map(item => item.points.map(point => `<circle class="hover-target" cx="${{x(point.predicted).toFixed(1)}}" cy="${{y(point.actual).toFixed(1)}}" r="4.8" fill="${{item.color}}" stroke="white" stroke-width="0.8" opacity="0.78" data-tooltip="${{tooltipText(item.label, point.event_name, point.date, mode === "time" ? `diagnostic time: ${{formatTime(point.predicted)}}` : `predicted rank: #${{point.predicted}}`, mode === "time" ? `actual time: ${{formatTime(point.actual)}}` : `actual rank: #${{point.actual}}`, `incoming score: ${{Number(point.score).toFixed(4)}}`, `event score: ${{Number(point.actualScore).toFixed(4)}}`)}}"></circle>`).join("")).join("");
  const legend = series.map((item, index) => `<g transform="translate(${{pad.left + (index % 2) * 270}}, ${{13 + Math.floor(index / 2) * 15}})"><circle cx="4" cy="0" r="4" fill="${{item.color}}"/><text x="12" y="4" font-size="10" fill="#374151">${{item.label}}</text></g>`).join("");
  const xLabel = mode === "time" ? "Diagnostic predicted time" : "Pre-event predicted rank";
  const yLabel = mode === "time" ? "Actual time" : "Actual rank among eligible entrants";
  container.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}">${{grid}}<line x1="${{x(minimum)}}" y1="${{y(minimum)}}" x2="${{x(maximum)}}" y2="${{y(maximum)}}" stroke="#dc2626" stroke-width="1.7" stroke-dasharray="5 5"/>${{dots}}${{legend}}<text x="${{width / 2}}" y="${{height - 5}}" text-anchor="middle" font-size="12">${{xLabel}}</text><text x="15" y="${{height / 2}}" text-anchor="middle" font-size="12" transform="rotate(-90 15 ${{height / 2}})">${{yLabel}}</text></svg>`;
}}

function renderPuzzlerEventHistogram() {{
  const eventId = document.getElementById("puzzlerEvent").value;
  const selectedSystem = document.getElementById("puzzlerSystem").value;
  const row = profileRows().find(event => event.event === eventId);
  const event = puzzlerEvents[eventId];
  const container = document.getElementById("puzzlerEventHistogram");
  if (!row || !event?.times?.length) {{ container.innerHTML = ""; return; }}
  const times = event.times.map(Number).filter(Number.isFinite);
  const width = 1000, height = 330, pad = {{left: 52, right: 20, top: 28, bottom: 48}}, bins = Math.min(32, Math.max(12, Math.round(Math.sqrt(times.length) * 2)));
  const predicted = Number(row.diagnostic?.predicted_time), dataMin = Math.min(...times), dataMax = Math.max(...times);
  const min = Number.isFinite(predicted) ? Math.min(dataMin, predicted) : dataMin;
  const max = Number.isFinite(predicted) ? Math.max(dataMax, predicted) : dataMax;
  const counts = Array.from({{length: bins}}, () => 0);
  times.forEach(value => {{ const index = Math.min(bins - 1, Math.floor(((value - dataMin) / Math.max(1, dataMax - dataMin)) * bins)); counts[index] += 1; }});
  const x = value => scale(value, min, max, pad.left, width - pad.right), yMax = Math.max(...counts);
  const y = value => scale(value, 0, yMax, height - pad.bottom, pad.top);
  const bars = counts.map((count, index) => {{
    const left = dataMin + index * (dataMax - dataMin) / bins, right = dataMin + (index + 1) * (dataMax - dataMin) / bins;
    const top = y(count);
    return `<rect class="hover-target" x="${{x(left).toFixed(1)}}" y="${{top.toFixed(1)}}" width="${{Math.max(1, x(right) - x(left) - 1).toFixed(1)}}" height="${{(height - pad.bottom - top).toFixed(1)}}" fill="#93c5fd" data-tooltip="${{tooltipText(`${{formatTime(left)}}–${{formatTime(right)}}`, `${{count}} puzzler${{count === 1 ? "" : "s"}}`)}}"></rect>`;
  }}).join("");
  const ticks = niceTicks(min, max, 6).map(value => `<line x1="${{x(value).toFixed(1)}}" y1="${{height - pad.bottom}}" x2="${{x(value).toFixed(1)}}" y2="${{height - pad.bottom + 5}}" stroke="#64748b"/><text x="${{x(value).toFixed(1)}}" y="${{height - 18}}" text-anchor="middle" font-size="10" fill="#6b7280">${{formatTime(value)}}</text>`).join("");
  const yStep = Math.max(1, Math.ceil(yMax / 4));
  const yTickValues = Array.from({{length: Math.floor(yMax / yStep) + 1}}, (_, index) => index * yStep);
  if (yTickValues[yTickValues.length - 1] !== yMax) yTickValues.push(yMax);
  const yTicks = yTickValues.map(value => `<line x1="${{pad.left}}" y1="${{y(value).toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y(value).toFixed(1)}}" stroke="#e5e7eb"/><line x1="${{pad.left - 5}}" y1="${{y(value).toFixed(1)}}" x2="${{pad.left}}" y2="${{y(value).toFixed(1)}}" stroke="#64748b"/><text x="${{pad.left - 9}}" y="${{(y(value) + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#6b7280">${{value}}</text>`).join("");
  const actualX = x(row.actual_time);
  const predictedLine = Number.isFinite(predicted) ? `<line x1="${{x(predicted).toFixed(1)}}" y1="${{pad.top}}" x2="${{x(predicted).toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#7c3aed" stroke-width="3"/><text x="${{Math.min(width - 155, x(predicted) + 5).toFixed(1)}}" y="${{pad.top + 25}}" font-size="11" fill="#7c3aed">diagnostic ${{formatTime(predicted)}}</text>` : "";
  container.innerHTML = `<div class="plot-title">${{event.name}} · ${{event.date}} · ${{times.length}} results</div><svg viewBox="0 0 ${{width}} ${{height}}">${{yTicks}}${{bars}}<line x1="${{pad.left}}" y1="${{height - pad.bottom}}" x2="${{width - pad.right}}" y2="${{height - pad.bottom}}" stroke="#64748b"/><line x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{height - pad.bottom}}" stroke="#64748b"/>${{ticks}}${{predictedLine}}<line x1="${{actualX.toFixed(1)}}" y1="${{pad.top}}" x2="${{actualX.toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#dc2626" stroke-width="3"/><text x="${{Math.min(width - 120, actualX + 5).toFixed(1)}}" y="${{pad.top + 10}}" font-size="11" fill="#dc2626">actual ${{formatTime(row.actual_time)}}</text><text x="${{width / 2}}" y="${{height - 3}}" text-anchor="middle" font-size="12">Completion time</text><text x="15" y="${{height / 2}}" text-anchor="middle" font-size="12" transform="rotate(-90 15 ${{height / 2}})">Puzzlers</text></svg>`;
  const d = row.diagnostic;
  document.getElementById("puzzlerEventMeta").textContent = d?.predicted_rank == null
    ? `${{formatTime(row.actual_time)}} actual · no incoming ${{profileSystemLabel(selectedSystem)}} score`
    : `${{formatTime(row.actual_time)}} actual · ${{formatTime(d.predicted_time)}} diagnostic · predicted #${{d.predicted_rank}}, actual #${{d.actual_rank}} among eligible entrants`;
}}

function renderPuzzlerProfile() {{
  const personKey = document.getElementById("puzzlerSelect").value;
  const system = document.getElementById("puzzlerSystem").value;
  const profile = puzzlerProfiles[personKey];
  if (!profile) return;
  const rows = profileRows();
  const scatterSystems = profileSystems.filter(candidate => selectedPuzzlerScatterSystems.has(candidate.key));
  const scatterSeries = scatterSystems.map(candidate => {{
    const candidateRows = profileRows(candidate.key);
    const color = puzzlerScatterColors[profileSystems.findIndex(item => item.key === candidate.key) % puzzlerScatterColors.length];
    const timePoints = candidateRows.filter(row => Number.isFinite(Number(row.diagnostic?.predicted_time))).map(row => ({{...row, predicted:Number(row.diagnostic.predicted_time), actual:Number(row.actual_time), score:row.diagnostic.predicted_score, actualScore:row.diagnostic.actual_score}}));
    const rankPoints = candidateRows.filter(row => row.diagnostic?.predicted_rank != null && row.diagnostic?.actual_rank != null).map(row => ({{...row, predicted:Number(row.diagnostic.predicted_rank), actual:Number(row.diagnostic.actual_rank), score:row.diagnostic.predicted_score, actualScore:row.diagnostic.actual_score}}));
    return {{...candidate, color, timePoints, rankPoints}};
  }});
  const detailRankCount = rows.filter(row => row.diagnostic?.predicted_rank != null).length;
  document.getElementById("puzzlerMeta").textContent = `${{rows.length}} events · ${{detailRankCount}} with an incoming ${{profileSystemLabel(system)}} score · ${{scatterSeries.length}} scatter system${{scatterSeries.length === 1 ? "" : "s"}}`;
  document.getElementById("puzzlerTimeAssumption").innerHTML = scatterSystems.length
    ? scatterSystems.map(candidate => `<div class="assumption-item"><strong>${{candidate.label}}</strong>${{profileTimeAssumption(candidate.key)}}</div>`).join("")
    : `<div class="assumption-item">Select one or more scatterplot systems to see their time derivations.</div>`;
  renderProfileScatter("puzzlerTimePlot", scatterSeries.map(item => ({{label:item.label,color:item.color,points:item.timePoints}})), "time");
  renderProfileScatter("puzzlerRankPlot", scatterSeries.map(item => ({{label:item.label,color:item.color,points:item.rankPoints}})), "rank");
  const tableRows = rows.filter(row => row.diagnostic?.predicted_score != null).slice().reverse();
  document.getElementById("puzzlerHistoryTable").innerHTML = `<thead><tr><th>Date</th><th>Event</th><th>Actual Time</th><th>Diagnostic Time</th><th>Incoming Score</th><th>Event Score</th><th>Predicted Rank</th><th>Actual Rank</th></tr></thead><tbody>${{tableRows.map(row => {{ const d=row.diagnostic; return `<tr><td>${{row.date}}</td><td>${{escapeAttr(row.event_name)}}</td><td>${{formatTime(row.actual_time)}}</td><td>${{formatTime(d.predicted_time)}}</td><td>${{Number(d.predicted_score).toFixed(4)}}</td><td>${{d.actual_score == null ? "" : Number(d.actual_score).toFixed(4)}}</td><td>#${{d.predicted_rank}}</td><td>#${{d.actual_rank}}</td></tr>`; }}).join("")}}</tbody>`;
  renderPuzzlerEventHistogram();
  attachPlotTooltips();
}}

function populatePuzzlerEvents() {{
  const select = document.getElementById("puzzlerEvent");
  const rows = profileRows();
  const previous = select.value;
  select.innerHTML = rows.map(row => `<option value="${{escapeAttr(row.event)}}">${{row.date}} · ${{escapeAttr(row.event_name)}} · ${{row.event}}</option>`).join("");
  select.value = rows.some(row => row.event === previous) ? previous : (rows[rows.length - 1]?.event || "");
}}

function initializePuzzlerProfile() {{
  const personSelect = document.getElementById("puzzlerSelect");
  const systemSelect = document.getElementById("puzzlerSystem");
  if (!personSelect.options.length) {{
    Object.entries(puzzlerProfiles).sort((a,b) => a[1].name.localeCompare(b[1].name)).forEach(([key, profile]) => {{
      personSelect.insertAdjacentHTML("beforeend", `<option value="${{escapeAttr(key)}}">${{escapeAttr(profile.name)}} (${{profile.events.length}})</option>`);
    }});
    profileSystems.forEach(system => systemSelect.insertAdjacentHTML("beforeend", `<option value="${{system.key}}">${{system.label}}</option>`));
    const picker = document.getElementById("puzzlerScatterSystemPicker");
    profileSystems.forEach((system, index) => {{
      const checked = selectedPuzzlerScatterSystems.has(system.key);
      picker.insertAdjacentHTML("beforeend", `<label><input type="checkbox" data-puzzler-scatter-system="${{system.key}}" ${{checked ? "checked" : ""}}><span style="color:${{puzzlerScatterColors[index % puzzlerScatterColors.length]}}">●</span> ${{system.label}}</label>`);
    }});
    picker.querySelectorAll("input").forEach(input => input.addEventListener("change", () => {{
      if (input.checked) selectedPuzzlerScatterSystems.add(input.dataset.puzzlerScatterSystem);
      else selectedPuzzlerScatterSystems.delete(input.dataset.puzzlerScatterSystem);
      renderPuzzlerProfile();
    }}));
    systemSelect.value = "jpar";
    personSelect.addEventListener("change", () => {{ populatePuzzlerEvents(); renderPuzzlerProfile(); }});
    systemSelect.addEventListener("change", renderPuzzlerProfile);
    document.getElementById("puzzlerEvent").addEventListener("change", () => {{ renderPuzzlerEventHistogram(); attachPlotTooltips(); }});
  }}
  populatePuzzlerEvents();
  renderPuzzlerProfile();
}}

function renderPredictionPlot(comparisons, normalized = false) {{
  const container = document.getElementById("predictionPlot");
  const populated = comparisons.filter(comparison => comparison.points.length);
  if (!populated.length) {{
    container.innerHTML = `<div class="plot-card"><div class="muted" style="padding:28px">No selected system has incoming ranks for this event.</div></div>`;
    return;
  }}
  const width = 760, height = 560, pad = {{left: 62, right: 22, top: Math.max(48, 24 + Math.ceil(populated.length / 3) * 16), bottom: 54}};
  const domainMin = normalized ? 0 : 1;
  const domainMax = normalized ? 100 : Math.max(2, ...populated.flatMap(comparison => comparison.points.flatMap(point => [point.predicted, point.actual])));
  const x = value => scale(value, domainMin, domainMax, pad.left, width - pad.right);
  const y = value => scale(value, domainMin, domainMax, pad.top, height - pad.bottom);
  const tickCount = normalized ? 6 : Math.min(8, domainMax);
  const ticks = Array.from({{length: tickCount}}, (_, index) => Math.round(domainMin + index * (domainMax - domainMin) / Math.max(1, tickCount - 1))).filter((value, index, array) => index === 0 || value !== array[index - 1]);
  const grid = ticks.map(value => `<line x1="${{x(value).toFixed(1)}}" y1="${{pad.top}}" x2="${{x(value).toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#e5e7eb"/><line x1="${{pad.left}}" y1="${{y(value).toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y(value).toFixed(1)}}" stroke="#e5e7eb"/><text x="${{x(value).toFixed(1)}}" y="${{height - 24}}" text-anchor="middle" font-size="11" fill="#6b7280">${{value}}</text><text x="${{pad.left - 10}}" y="${{(y(value) + 4).toFixed(1)}}" text-anchor="end" font-size="11" fill="#6b7280">${{value}}</text>`).join("");
  const dots = predictionViewMode === "scatter" ? populated.map(comparison => comparison.points.map(point => `<circle class="hover-target" cx="${{x(point.predicted).toFixed(1)}}" cy="${{y(point.actual).toFixed(1)}}" r="4" fill="${{comparison.color}}" stroke="white" stroke-width="0.8" data-tooltip="${{escapeAttr(`${{point.name}}\n${{comparison.column.label}}${{point.eventName ? `\n${{point.eventDate}} · ${{point.eventName}}` : ""}}\nIncoming ${{normalized ? "percentile" : "relative rank"}}: ${{Number(point.predicted).toFixed(normalized ? 1 : 0)}}\nActual ${{normalized ? "percentile" : "relative rank"}}: ${{Number(point.actual).toFixed(normalized ? 1 : 0)}}`)}}"></circle>`).join("")).join("") : "";
  const bandSeries = predictionViewMode === "bands" ? populated.map(comparison => ({{comparison, band: rollingPredictionBand(comparison.points, domainMin, domainMax)}})) : [];
  const bands = bandSeries.map(({{comparison, band}}) => {{
    const area = band.map((point, index) => `${{index ? "L" : "M"}} ${{x(point.x).toFixed(1)}} ${{y(point.high).toFixed(1)}}`).join(" ") + " " + [...band].reverse().map(point => `L ${{x(point.x).toFixed(1)}} ${{y(point.low).toFixed(1)}}`).join(" ") + " Z";
    const line = band.map((point, index) => `${{index ? "L" : "M"}} ${{x(point.x).toFixed(1)}} ${{y(point.median).toFixed(1)}}`).join(" ");
    return `<path d="${{area}}" fill="${{comparison.color}}" opacity="0.16"/><path d="${{line}}" fill="none" stroke="${{comparison.color}}" stroke-width="2.5"/>`;
  }}).join("");
  const hoverSlices = predictionViewMode === "bands" ? Array.from({{length: 48}}, (_, index) => {{
    const targetX = domainMin + (index + 0.5) * (domainMax - domainMin) / 48;
    const left = x(domainMin + index * (domainMax - domainMin) / 48);
    const right = x(domainMin + (index + 1) * (domainMax - domainMin) / 48);
    const lines = bandSeries.map(({{comparison, band}}) => {{
      const nearest = band.reduce((best, point) => Math.abs(point.x - targetX) < Math.abs(best.x - targetX) ? point : best, band[0]);
      const digits = normalized ? 1 : 1;
      return `${{comparison.column.label}}: median ${{nearest.median.toFixed(digits)}} · band ${{nearest.low.toFixed(digits)}}–${{nearest.high.toFixed(digits)}}`;
    }});
    const label = `${{normalized ? "Incoming field percentile" : "Incoming relative rank"}}: ${{targetX.toFixed(normalized ? 1 : 0)}}\\n${{lines.join("\\n")}}`;
    return `<rect class="hover-target hover-slice" x="${{left.toFixed(1)}}" y="${{pad.top}}" width="${{Math.max(3, right - left).toFixed(1)}}" height="${{height - pad.bottom - pad.top}}" data-tooltip="${{escapeAttr(label)}}"></rect>`;
  }}).join("") : "";
  const legend = populated.map((comparison, index) => `<g transform="translate(${{pad.left + (index % 3) * 220}}, ${{15 + Math.floor(index / 3) * 16}})"><circle cx="4" cy="0" r="4" fill="${{comparison.color}}"/><text x="12" y="4" font-size="11" fill="#374151">${{comparison.column.label}}</text></g>`).join("");
  const axisTerm = normalized ? "field percentile" : "relative rank";
  const subtitle = predictionViewMode === "bands" ? "Rolling median with ±1 rolling SD" : "Individual entrants";
  container.innerHTML = `<div class="plot-card"><div class="plot-title">Incoming vs Actual ${{normalized ? "Field Percentile" : "Relative Rank"}} · ${{subtitle}}</div><svg viewBox="0 0 ${{width}} ${{height}}">${{grid}}<line x1="${{x(domainMin)}}" y1="${{y(domainMin)}}" x2="${{x(domainMax)}}" y2="${{y(domainMax)}}" stroke="#111827" stroke-width="1.5" stroke-dasharray="6 5"/>${{bands}}${{dots}}${{hoverSlices}}${{legend}}<text x="${{(pad.left + width - pad.right) / 2}}" y="${{height - 5}}" text-anchor="middle" font-size="12">Incoming ${{axisTerm}}</text><text x="16" y="${{(pad.top + height - pad.bottom) / 2}}" text-anchor="middle" font-size="12" transform="rotate(-90 16 ${{(pad.top + height - pad.bottom) / 2}})">Actual ${{axisTerm}}</text></svg></div>`;
}}

function renderPredictionMetrics(comparisons, normalized = false) {{
  const fmt = value => value == null ? "—" : Number(value).toFixed(2);
  const sorted = [...comparisons].sort((a, b) => (a.mae ?? Infinity) - (b.mae ?? Infinity));
  const unit = normalized ? " (pct pts)" : "";
  document.getElementById("predictionMetricsTable").innerHTML = `<thead><tr><th>System</th><th>N</th><th>MAE${{unit}}</th><th>RMSE${{unit}}</th><th>Max Error${{unit}}</th><th>Bias${{unit}}</th></tr></thead><tbody>${{sorted.map(comparison => `<tr><td><span style="color:${{comparison.color}}">●</span> ${{comparison.column.label}}</td><td>${{comparison.points.length}}</td><td>${{fmt(comparison.mae)}}</td><td>${{fmt(comparison.rmse)}}</td><td>${{fmt(comparison.maxError)}}</td><td>${{fmt(comparison.bias)}}</td></tr>`).join("")}}</tbody>`;
}}

function renderPredictionEntrants(result) {{
  const jparComparison = result.comparisons.find(comparison => comparison.column.key === "jpar_rank");
  const jparPoints = new Map((jparComparison?.points || []).map(point => [point.key, point]));
  const comparisonPoints = new Map(result.comparisons.map(comparison => [comparison.column.key, new Map(comparison.points.map(point => [point.key, point]))]));
  const absoluteJparRanks = rankMap([...result.incoming.values()], item => item.jpar);
  const extraColumns = selectedPredictionColumns().filter(column => column.key !== "jpar_rank");
  const rows = result.entrants.map(entrant => {{
    const jparPoint = jparPoints.get(entrant._member_key);
    const record = {{
      entrant,
      jparPoint,
      hasIncomingJpar: Boolean(entrant.incoming && entrant.incoming.jpar != null),
      name: entrant.full_name,
      event_place: entrant.event_place,
      event_time_seconds: entrant.event_time_seconds,
      incoming_jpar_absolute: entrant.incoming?.jpar == null ? null : (absoluteJparRanks.get(entrant._member_key) ?? null),
      incoming_jpar_relative: jparPoint?.predicted ?? null,
    }};
    extraColumns.forEach(column => {{ record[column.key] = comparisonPoints.get(column.key)?.get(entrant._member_key)?.predicted ?? null; }});
    return record;
  }});
  rows.sort((a, b) => {{
    const group = Number(b.hasIncomingJpar) - Number(a.hasIncomingJpar);
    if (group) return group;
    const av = a[predictionEntrantSortKey], bv = b[predictionEntrantSortKey];
    if (av == null && bv == null) return Number(a.event_place) - Number(b.event_place);
    if (av == null) return 1;
    if (bv == null) return -1;
    const comparison = typeof av === "string" ? String(av).localeCompare(String(bv)) : Number(av) - Number(bv);
    return comparison * (predictionEntrantSortDir === "asc" ? 1 : -1);
  }});
  const sortableHeader = (key, label) => `<th data-prediction-sort="${{key}}">${{label}}${{predictionEntrantSortKey === key ? (predictionEntrantSortDir === "asc" ? " ▲" : " ▼") : ""}}</th>`;
  const systemHeaders = extraColumns.map(column => sortableHeader(column.key, `${{column.label}}<br><span class="muted">incoming relative</span>`)).join("");
  let boundaryAdded = false;
  const body = rows.map(row => {{
    let divider = "";
    if (!row.hasIncomingJpar && !boundaryAdded) {{ divider = ` style="border-top:4px solid #64748b"`; boundaryAdded = true; }}
    const systemCells = extraColumns.map(column => `<td>${{row[column.key] ?? "—"}}</td>`).join("");
    return `<tr${{divider}}><td>${{row.hasIncomingJpar ? "Ranked" : "No incoming JPAR"}}</td><td>${{escapeAttr(row.name)}}</td><td>${{row.event_place == null ? "—" : Number(row.event_place).toFixed(Number(row.event_place) % 1 ? 1 : 0)}}</td><td>${{formatTime(row.event_time_seconds)}}</td><td>${{row.incoming_jpar_absolute ?? "—"}}</td><td>${{row.incoming_jpar_relative ?? "—"}}</td>${{systemCells}}</tr>`;
  }}).join("");
  const table = document.getElementById("predictionEntrantsTable");
  table.innerHTML = `<thead><tr><th>Status</th>${{sortableHeader("name", "Name")}}${{sortableHeader("event_place", "Event Place")}}${{sortableHeader("event_time_seconds", "Time")}}${{sortableHeader("incoming_jpar_absolute", "Incoming JPAR Rank")}}${{sortableHeader("incoming_jpar_relative", "Incoming JPAR Relative")}}${{systemHeaders}}</tr></thead><tbody>${{body}}</tbody>`;
  table.querySelectorAll("th[data-prediction-sort]").forEach(header => header.addEventListener("click", () => {{
    const key = header.dataset.predictionSort;
    if (predictionEntrantSortKey === key) predictionEntrantSortDir = predictionEntrantSortDir === "asc" ? "desc" : "asc";
    else {{ predictionEntrantSortKey = key; predictionEntrantSortDir = "asc"; }}
    renderPredictionEntrants(result);
  }}));
}}

function renderPredictions() {{
  const eventIndex = Number(document.getElementById("predictionEventSlider").value);
  const result = buildPredictionComparison(eventIndex);
  if (!result.event) return;
  document.getElementById("predictionEventSelect").value = String(eventIndex);
  document.getElementById("predictionEventLabel").textContent = `${{result.event.date}} · ${{result.event.event_name}} · ${{result.event.event}}`;
  const rankedEntrants = result.entrants.filter(entrant => entrant.incoming?.jpar != null).length;
  const allEvents = predictionScopeMode === "all";
  const plotComparisons = allEvents ? buildAllPredictionComparisons() : result.comparisons;
  const pooledN = allEvents ? Math.max(0, ...plotComparisons.map(comparison => comparison.points.length)) : null;
  document.getElementById("predictionSummary").textContent = allEvents
    ? `All ${{rankingTimeline.length}} events pooled · ranks normalized to 0–100 field percentiles · up to ${{pooledN}} entrant-event predictions per system`
    : `${{result.entrants.length}} entrants · ${{rankedEntrants}} with an incoming JPAR · predictions use only results before this event`;
  renderPredictionPlot(plotComparisons, allEvents);
  renderPredictionMetrics(plotComparisons, allEvents);
  renderPredictionEntrants(result);
  attachPlotTooltips();
}}

function initializePredictions() {{
  const select = document.getElementById("predictionEventSelect");
  if (!select.options.length) {{
    rankingTimeline.forEach((event, index) => {{
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${{event.date}} · ${{event.event_name}} · ${{event.event}}`;
      select.appendChild(option);
    }});
  }}
  const picker = document.getElementById("predictionSystemPicker");
  if (!picker.children.length) {{
    columns.filter(column => column.kind === "rank").forEach(column => {{
      const checked = defaultPredictionSystems.has(column.key);
      if (checked) selectedPredictionSystems.add(column.key);
      picker.insertAdjacentHTML("beforeend", `<label><input type="checkbox" data-prediction-system="${{column.key}}" ${{checked ? "checked" : ""}}> ${{column.label}}</label>`);
    }});
    picker.querySelectorAll("input").forEach(input => input.addEventListener("change", () => {{
      if (input.checked) selectedPredictionSystems.add(input.dataset.predictionSystem);
      else selectedPredictionSystems.delete(input.dataset.predictionSystem);
      renderPredictions();
    }}));
  }}
  renderPredictions();
}}

function attachPlotTooltips() {{
  const tooltip = document.getElementById("tooltip");
  document.querySelectorAll("[data-tooltip]").forEach(el => {{
    if (el.dataset.tooltipBound === "1") return;
    el.dataset.tooltipBound = "1";
    el.addEventListener("mouseenter", () => {{
      tooltip.textContent = el.dataset.tooltip;
      tooltip.style.display = "block";
    }});
    el.addEventListener("mousemove", event => {{
      tooltip.style.left = `${{event.clientX + 14}}px`;
      tooltip.style.top = `${{event.clientY + 14}}px`;
    }});
    el.addEventListener("mouseleave", () => {{
      tooltip.style.display = "none";
    }});
  }});
}}

document.getElementById("search").addEventListener("input", render);
document.getElementById("predictionEventSlider").addEventListener("input", renderPredictions);
document.getElementById("predictionEventSelect").addEventListener("change", event => {{
  document.getElementById("predictionEventSlider").value = event.target.value;
  renderPredictions();
}});
document.getElementById("rankingCutoff").addEventListener("input", event => {{
  const cutoffIndex = Number(event.target.value);
  if (rankingFrame != null) cancelAnimationFrame(rankingFrame);
  rankingFrame = requestAnimationFrame(() => {{
    rebuildRankingRows(cutoffIndex);
    if (selectedMemberKey && !currentRankingRows.some(row => row._member_key === selectedMemberKey)) selectedMemberKey = null;
    render();
    renderCustomRankComparison();
    rankingFrame = null;
  }});
}});
document.getElementById("minEvents").addEventListener("input", render);
document.getElementById("limit").addEventListener("change", render);
document.getElementById("overviewSearch").addEventListener("input", renderCalculationOverview);
document.getElementById("overviewSearchColumn").addEventListener("change", renderCalculationOverview);
document.getElementById("overviewLimit").addEventListener("change", renderCalculationOverview);
document.getElementById("overviewIncludedOnly").addEventListener("change", renderCalculationOverview);
document.getElementById("driftMinEvents").addEventListener("input", () => {{
  renderDriftPlots();
  attachPlotTooltips();
}});
document.getElementById("histMinEvents").addEventListener("input", () => {{
  renderHistPlots();
  attachPlotTooltips();
}});
document.getElementById("scatterMinEvents").addEventListener("input", () => {{
  renderRankScatterPlots();
  attachPlotTooltips();
}});
document.getElementById("customRankX").addEventListener("change", () => {{ renderCustomRankComparison(); attachPlotTooltips(); }});
document.getElementById("customRankY").addEventListener("change", () => {{ renderCustomRankComparison(); attachPlotTooltips(); }});
document.getElementById("customRankMinEvents").addEventListener("input", () => {{ renderCustomRankComparison(); attachPlotTooltips(); }});
document.querySelectorAll(".individual-mode").forEach(button => {{
  button.addEventListener("click", () => {{
    individualView = button.dataset.mode;
    document.querySelectorAll(".individual-mode").forEach(candidate => {{
      candidate.setAttribute("aria-pressed", String(candidate === button));
    }});
    renderDriftPlots();
    attachPlotTooltips();
  }});
}});
document.querySelectorAll(".prediction-view-mode").forEach(button => {{
  button.addEventListener("click", () => {{
    predictionViewMode = button.dataset.mode;
    document.querySelectorAll(".prediction-view-mode").forEach(candidate => candidate.setAttribute("aria-pressed", String(candidate === button)));
    renderPredictions();
  }});
}});
document.querySelectorAll(".prediction-scope-mode").forEach(button => {{
  button.addEventListener("click", () => {{
    predictionScopeMode = button.dataset.mode;
    document.querySelectorAll(".prediction-scope-mode").forEach(candidate => candidate.setAttribute("aria-pressed", String(candidate === button)));
    renderPredictions();
  }});
}});
document.getElementById("cumulativeMinEvents").addEventListener("input", renderCumulativeCalibrationTable);
document.getElementById("cumulativeMaxJpar").addEventListener("input", renderCumulativeCalibrationTable);
document.querySelectorAll(".tab-button").forEach(button => {{
  button.addEventListener("click", () => activateTab(button.dataset.tab));
  button.addEventListener("keydown", event => {{
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const tabs = [...document.querySelectorAll(".tab-button")];
    const index = tabs.indexOf(button);
    const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    event.preventDefault();
    tabs[nextIndex].focus();
    activateTab(tabs[nextIndex].dataset.tab);
  }});
}});
window.addEventListener("hashchange", () => activateTab(window.location.hash.slice(1), false));
document.getElementById("reset").addEventListener("click", () => {{
  document.getElementById("search").value = "";
  document.getElementById("minEvents").value = 1;
  document.getElementById("limit").value = "100";
  document.getElementById("rankingCutoff").value = Math.max(0, rankingTimeline.length - 1);
  document.getElementById("driftMinEvents").value = 1;
  document.getElementById("histMinEvents").value = 1;
  document.getElementById("scatterMinEvents").value = 1;
  document.getElementById("cumulativeMinEvents").value = 1;
  document.getElementById("cumulativeMaxJpar").value = cumulativeJparMax;
  selectedMemberKey = null;
  individualView = "score";
  document.querySelectorAll(".individual-mode").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === "score")));
  sortKey = "jpar_rank";
  sortDir = "asc";
  rebuildRankingRows(Math.max(0, rankingTimeline.length - 1));
  render();
  renderDriftPlots();
  renderHistPlots();
  renderRankScatterPlots();
  renderCustomRankComparison();
  renderSelectedPersonEvents();
  if (renderedTabs.has("misc")) {{
    renderFeedbackSection();
    renderEventImpacts();
    renderCumulativeCalibrationTable();
  }}
  attachPlotTooltips();
}});
activateTab(window.location.hash.slice(1) || "raw", false);
</script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    global EXTERNAL_OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Build interactive rank comparison HTML")
    parser.add_argument("--input", default="outputs/corrected_ranking_diagnostics/master_ranking_systems.csv")
    parser.add_argument("--calculation-df", default="data/data_jpar_v2/source_of_truth_calculation_df.csv")
    parser.add_argument("--output", default="outputs/corrected_ranking_diagnostics/interactive_rank_comparison.html")
    parser.add_argument("--external-systems-dir", default="colleague_ranking_systems/outputs")
    parser.add_argument("--exclude-external-systems", action="store_true", help="Build the original dashboard without the isolated added systems")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    EXTERNAL_OUTPUT_DIR = None if args.exclude_external_systems else Path(args.external_systems_dir)
    external_leaderboard = external_csv("final_leaderboard.csv")
    if not external_leaderboard.empty:
        df["member_key"] = df["resolved_member_id"].apply(keyify)
        added = ["member_key"] + [column for column in external_leaderboard.columns if column.startswith("external_")]
        df = df.merge(external_leaderboard[added], on="member_key", how="left")
    df = augment_master_with_strength_of_field_zscores(df, Path(args.calculation_df))
    build_html(df, Path(args.output), Path(args.calculation_df))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
