#!/usr/bin/env python3
"""Current JPAR pipeline.

This script reads the joined event-member dataset, computes JPAR over all members,
keeps unmatched event rows in the calculation input, and exports both the full
calculation output and an active/recent-only analysis export.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path("jpar_pipeline/config/jpar_pipeline_config.json")
SOURCE_EVENT_RESULTS_DIR = Path("data/data_event_results")


@dataclass
class PipelineConfig:
    event_input_csv: Path
    output_dir: Path
    baseline_event_date: str
    baseline_event_ids: list[str]
    minimum_participants_per_event: int
    calculate_all_members_then_filter_active: bool
    export_active_only: bool
    active_statuses: list[str]
    include_event_ids: list[str]
    exclude_event_ids: list[str]
    include_event_hosts: list[str]
    exclude_event_hosts: list[str]
    include_all_sanctioned_events: bool
    exclude_member_names_from_calculation: list[str]
    deliverables_paid_through_max_date: str
    deliverables_excel_name: str
    calculation_df_name: str
    results_name: str
    export_zscore_outputs: bool
    zscore_deliverables_excel_name: str
    zscore_calculation_df_name: str
    zscore_results_name: str


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return PipelineConfig(
        event_input_csv=Path(raw["event_input_csv"]),
        output_dir=Path(raw["output_dir"]),
        baseline_event_date=str(raw["baseline_event_date"]),
        baseline_event_ids=list(raw.get("baseline_event_ids", [])),
        minimum_participants_per_event=int(raw.get("minimum_participants_per_event", 3)),
        calculate_all_members_then_filter_active=bool(raw.get("calculate_all_members_then_filter_active", True)),
        export_active_only=bool(raw.get("export_active_only", True)),
        active_statuses=list(raw.get("active_statuses", ["ACTIVE", "EXPIRED_RECENT", "RECENTLY EXPIRED", "EXPIRED_RECENT"])),
        include_event_ids=list(raw.get("include_event_ids", [])),
        exclude_event_ids=list(raw.get("exclude_event_ids", [])),
        include_event_hosts=list(raw.get("include_event_hosts", [])),
        exclude_event_hosts=list(raw.get("exclude_event_hosts", [])),
        include_all_sanctioned_events=bool(raw.get("include_all_sanctioned_events", True)),
        exclude_member_names_from_calculation=list(raw.get("exclude_member_names_from_calculation", ["Connor DeLaat", "Conner DeLaat"])),
        deliverables_paid_through_max_date=str(raw.get("deliverables_paid_through_max_date", "")),
        deliverables_excel_name=str(raw.get("deliverables_excel_name", "jpar_deliverables_latest.xlsx")),
        calculation_df_name=str(raw.get("calculation_df_name", "calculation_df_latest.csv")),
        results_name=str(raw.get("results_name", "jpar_results_latest.csv")),
        export_zscore_outputs=bool(raw.get("export_zscore_outputs", True)),
        zscore_deliverables_excel_name=str(raw.get("zscore_deliverables_excel_name", "jpar_deliverables_zscore_latest.xlsx")),
        zscore_calculation_df_name=str(raw.get("zscore_calculation_df_name", "calculation_df_zscore_latest.csv")),
        zscore_results_name=str(raw.get("zscore_results_name", "jpar_results_zscore_latest.csv")),
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalize_member_match_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "member_match_status" not in out.columns:
        out["member_match_status"] = np.where(out["mp_id"].notna(), "matched_member", "unmatched_member")
    out["member_match_status"] = out["member_match_status"].fillna("unmatched_member")
    return out


def add_event_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_date" not in out.columns:
        out["event_date"] = pd.NaT

    if "source_file" in out.columns:
        event_date = out["source_file"].astype(str).str.extract(r"(?P<event_date>\d{6})")
        event_date = event_date["event_date"].fillna("")
        parsed = pd.to_datetime(event_date, format="%y%m%d", errors="coerce")
        out.loc[out["event_date"].isna(), "event_date"] = parsed

    if "event_id" in out.columns:
        out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

        event_id_text = out["event_id"].astype(str).str.strip()
        six_digit = event_id_text.str.extract(r"(?P<event_date>\d{6})")["event_date"]
        parsed_six = pd.to_datetime(six_digit, format="%y%m%d", errors="coerce")
        out["event_date"] = out["event_date"].fillna(parsed_six)

        # Legacy event ids like 52723 / 60323 / 70823 encode MDDYY and need left-padding.
        five_digit = event_id_text.str.extract(r"^(?P<event_date>\d{5})(?:\.\d+)?$")["event_date"]
        parsed_five = pd.to_datetime(five_digit.str.zfill(6), format="%m%d%y", errors="coerce")
        out["event_date"] = out["event_date"].fillna(parsed_five)

    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    return out


def _sheet_round_suffix(sheet_name: object) -> str | None:
    sheet = clean_text(sheet_name).lower()
    if not sheet:
        return None

    if "prelim" in sheet:
        number_match = re.search(r"\bprelim\s*([0-9]+)\b", sheet)
        if number_match:
            value = int(number_match.group(1))
            return f"{value:02d}"

        letter_match = re.search(r"\bprelim\s*([a-z])\b", sheet)
        if letter_match:
            value = ord(letter_match.group(1).upper()) - ord("A") + 1
            if value > 0:
                return f"{value:02d}"

    if "final" in sheet:
        return "99"

    return None


def _extract_event_id_from_sheet_metadata(source_file: object, source_sheet: object) -> str | None:
    source_file_text = str(source_file).strip() if not pd.isna(source_file) else ""
    source_sheet_text = str(source_sheet).strip() if not pd.isna(source_sheet) else ""
    if not source_file_text or not source_sheet_text:
        return None

    xlsx_path = SOURCE_EVENT_RESULTS_DIR / source_file_text
    if not xlsx_path.exists():
        return None

    try:
        raw = pd.read_excel(xlsx_path, sheet_name=source_sheet_text, header=None, nrows=20)
    except Exception:
        return None

    # Template metadata stores Event ID in the first two columns before participant headers.
    for _, row in raw.iterrows():
        label = clean_text(row.iloc[0] if len(row) > 0 else "").lower()
        if "event id" in label:
            candidate = clean_text(row.iloc[1] if len(row) > 1 else "")
            match = re.search(r"\b\d{6}(?:\.\d{1,2})?\b", candidate)
            if match:
                return match.group(0)

    return None


def _extract_competition_name_from_sheet_metadata(source_file: object, source_sheet: object) -> str | None:
    source_file_text = str(source_file).strip() if not pd.isna(source_file) else ""
    source_sheet_text = str(source_sheet).strip() if not pd.isna(source_sheet) else ""
    if not source_file_text or not source_sheet_text:
        return None

    xlsx_path = SOURCE_EVENT_RESULTS_DIR / source_file_text
    if not xlsx_path.exists():
        return None

    try:
        raw = pd.read_excel(xlsx_path, sheet_name=source_sheet_text, header=None, nrows=20)
    except Exception:
        return None

    for _, row in raw.iterrows():
        label = clean_text(row.iloc[0] if len(row) > 0 else "").lower()
        if "competition name" in label:
            candidate = clean_text(row.iloc[1] if len(row) > 1 else "")
            if candidate:
                return candidate

    return None


def _derive_event_name_from_source_file(source_file: object, event_id: object = "") -> str:
    if pd.isna(source_file):
        return ""
    text = str(source_file).strip()
    if not text:
        return ""

    lowered = text.lower()
    event_id_text = clean_text(event_id)
    if "speedpuzzling.com jpar event data" in lowered:
        return f"SPDC {event_id_text}".strip()

    stem = Path(text).stem
    # Remove leading date prefix commonly used in source file names.
    stem = re.sub(r"^\d{6}[_\-\s]*", "", stem)
    stem = stem.replace("_", " ").strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def canonicalize_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_id" not in out.columns:
        return out

    event_id_text = out["event_id"].fillna("").astype(str).str.strip()
    source_sheet = out.get("source_sheet", pd.Series([""] * len(out), index=out.index))
    source_file = out.get("source_file", pd.Series([""] * len(out), index=out.index))

    metadata_event_id_cache: dict[tuple[str, str], str | None] = {}

    def event_id_from_metadata(file_value: object, sheet_value: object) -> str | None:
        key = (clean_text(file_value), clean_text(sheet_value))
        if key not in metadata_event_id_cache:
            metadata_event_id_cache[key] = _extract_event_id_from_sheet_metadata(file_value, sheet_value)
        return metadata_event_id_cache[key]

    normalized: list[str] = []
    for eid, file_value, sheet in zip(event_id_text.tolist(), source_file.tolist(), source_sheet.tolist()):
        metadata_event_id = event_id_from_metadata(file_value, sheet)
        if metadata_event_id:
            normalized.append(metadata_event_id)
            continue

        match = re.match(r"^(\d{6})_(.+)$", eid)
        if match:
            base = match.group(1)
            suffix = _sheet_round_suffix(sheet)
            if suffix is not None:
                normalized.append(f"{base}.{suffix}")
                continue

            tag = clean_text(match.group(2)).lower()
            if "prel" in tag:
                normalized.append(f"{base}.01")
                continue
            if "fina" in tag or "final" in tag:
                normalized.append(f"{base}.99")
                continue

        normalized.append(eid)

    out["event_id"] = pd.Series(normalized, index=out.index)
    return out


def filter_events(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    include_ids = {str(x).strip() for x in config.include_event_ids if str(x).strip()}
    exclude_ids = {str(x).strip() for x in config.exclude_event_ids if str(x).strip()}
    include_hosts = {str(x).strip().lower() for x in config.include_event_hosts if str(x).strip()}
    exclude_hosts = {str(x).strip().lower() for x in config.exclude_event_hosts if str(x).strip()}
    baseline_ids = {str(x).strip() for x in config.baseline_event_ids if str(x).strip()}

    out["event_id_str"] = out["event_id"].astype(str)
    out["event_host_str"] = out.get("event_host", pd.Series([""] * len(out))).astype(str).str.strip().str.lower()
    out["event_date"] = pd.to_datetime(out.get("event_date"), errors="coerce")

    baseline_date = pd.to_datetime(config.baseline_event_date, errors="coerce")
    if pd.notna(baseline_date):
        out = out[out["event_date"].isna() | (out["event_date"] >= baseline_date)]

    if include_ids:
        out = out[out["event_id_str"].isin(include_ids)]

    if exclude_ids:
        out = out[~out["event_id_str"].isin(exclude_ids)]

    if include_hosts:
        out = out[out["event_host_str"].isin(include_hosts)]

    if exclude_hosts:
        out = out[~out["event_host_str"].isin(exclude_hosts)]

    return out.drop(columns=[c for c in ["event_id_str", "event_host_str"] if c in out.columns])


def normalize_person_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _parse_numeric_from_mixed(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except Exception:
        return np.nan


def fill_missing_pieces_assembled_from_sources(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pieces_assembled" not in out.columns:
        return out
    if "source_file" not in out.columns or "source_sheet" not in out.columns:
        return out
    if "full_name" not in out.columns:
        return out

    pieces_num = pd.to_numeric(out["pieces_assembled"], errors="coerce")
    missing_idx = out.index[pieces_num.isna()]
    if len(missing_idx) == 0:
        out["pieces_assembled"] = pieces_num
        return out

    sheet_cache: dict[tuple[str, str], dict[str, float]] = {}

    def get_sheet_piece_lookup(source_file: object, source_sheet: object) -> dict[str, float]:
        key = (clean_text(source_file), clean_text(source_sheet))
        if key in sheet_cache:
            return sheet_cache[key]

        lookup: dict[str, float] = {}
        source_file_text, source_sheet_text = key
        if not source_file_text or not source_sheet_text:
            sheet_cache[key] = lookup
            return lookup

        xlsx_path = SOURCE_EVENT_RESULTS_DIR / source_file_text
        if not xlsx_path.exists():
            sheet_cache[key] = lookup
            return lookup

        try:
            raw = pd.read_excel(xlsx_path, sheet_name=source_sheet_text, header=None)
        except Exception:
            sheet_cache[key] = lookup
            return lookup

        header_row_idx = None
        for row_idx in range(min(len(raw), 30)):
            label = clean_text(raw.iat[row_idx, 0] if raw.shape[1] > 0 else "").lower()
            if "usajpa member id" in label:
                header_row_idx = row_idx
                break

        if header_row_idx is None:
            sheet_cache[key] = lookup
            return lookup

        header = raw.iloc[header_row_idx].astype(str).str.strip()
        data = raw.iloc[header_row_idx + 1 :].copy()
        data.columns = header
        data = data.dropna(how="all")
        col_map = {clean_text(col).lower(): col for col in data.columns}
        full_name_col = col_map.get("full name")
        pieces_col = None
        for norm_col, original_col in col_map.items():
            if "pieces assembled" in norm_col:
                pieces_col = original_col
                break

        if not full_name_col or not pieces_col:
            sheet_cache[key] = lookup
            return lookup

        names = data[full_name_col].apply(normalize_person_name)
        pieces_values = data[pieces_col].apply(_parse_numeric_from_mixed)

        for name, pieces in zip(names.tolist(), pieces_values.tolist()):
            if name and pd.notna(pieces) and name not in lookup:
                lookup[name] = float(pieces)

        sheet_cache[key] = lookup
        return lookup

    for idx in missing_idx:
        row = out.loc[idx]
        lookup = get_sheet_piece_lookup(row.get("source_file"), row.get("source_sheet"))
        name = normalize_person_name(row.get("full_name"))
        if name in lookup:
            pieces_num.at[idx] = lookup[name]

    out["pieces_assembled"] = pieces_num
    return out


def exclude_members_from_calculation_input(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    excluded_names = {
        normalize_person_name(name)
        for name in config.exclude_member_names_from_calculation
        if normalize_person_name(name)
    }
    if not excluded_names:
        return df.copy()

    out = df.copy()
    name_columns = [
        col
        for col in ["full_name", "member_full_name", "source_full_name"]
        if col in out.columns
    ]
    if not name_columns:
        return out

    keep_mask = pd.Series(True, index=out.index)
    for col in name_columns:
        keep_mask &= ~out[col].apply(normalize_person_name).isin(excluded_names)

    return out[keep_mask].copy()


def count_distinct_valid_members(sub: pd.DataFrame) -> int:
    if "member_match_status" in sub.columns:
        sub = sub[sub["member_match_status"].astype(str).str.lower() == "matched_member"]
    elif "mp_id" in sub.columns:
        sub = sub[sub["mp_id"].notna()]
    resolved_ids = resolved_member_id_series(sub)
    return resolved_ids.replace({"nan": ""}).loc[lambda s: s != ""].nunique()


def compute_completion_seconds(df: pd.DataFrame) -> pd.Series:
    completion = pd.to_numeric(df.get("completion_time_seconds"), errors="coerce")

    if "completion_time" in df.columns:
        completion_text = df["completion_time"].fillna("").astype(str).str.strip()

        # Parse legacy minute.second text values like 117.03 as 1:17:03.
        mmss_mask = completion.isna() & completion_text.str.match(r"^\d{1,3}\.\d{1,2}$", na=False)
        if mmss_mask.any():
            minute_second = completion_text.loc[mmss_mask].str.split(".", n=1, expand=True)
            minutes = pd.to_numeric(minute_second[0], errors="coerce")
            seconds = pd.to_numeric(minute_second[1], errors="coerce")
            completion.loc[mmss_mask] = (minutes * 60) + seconds

    if "suggested_time_seconds" in df.columns:
        suggested = pd.to_numeric(df.get("suggested_time_seconds"), errors="coerce")

        # For explicit parser corrections, suggested seconds are the intended value.
        if "suggested_time_reason" in df.columns:
            reason = df["suggested_time_reason"].fillna("").astype(str).str.lower()
            override_mask = reason.eq("interpreted_as_mmss00") & suggested.notna()
            completion.loc[override_mask] = suggested.loc[override_mask]

        completion = completion.fillna(suggested)

    # For DNF rows, extrapolate finish time from pace at event max time:
    # pace = pieces_assembled / max_time_seconds, finish = piece_count / pace.
    if (
        "completion_time" in df.columns
        and "max_time_seconds" in df.columns
        and "pieces_assembled" in df.columns
        and "piece_count" in df.columns
    ):
        completion_text = df["completion_time"].fillna("").astype(str)
        dnf_text_mask = completion_text.str.contains("DNF", case=False, na=False)
        max_seconds = pd.to_numeric(df.get("max_time_seconds"), errors="coerce")
        pieces_assembled = pd.to_numeric(df.get("pieces_assembled"), errors="coerce")
        piece_count = pd.to_numeric(df.get("piece_count"), errors="coerce")

        # Some source sheets encode DNF via pieces assembled < event piece count.
        dnf_by_pieces_mask = (
            pieces_assembled.notna()
            & piece_count.notna()
            & (pieces_assembled > 0)
            & (piece_count > 0)
            & (pieces_assembled < piece_count)
        )
        dnf_mask = dnf_text_mask | dnf_by_pieces_mask

        valid_dnf = (
            dnf_mask
            & max_seconds.notna()
            & pieces_assembled.notna()
            & piece_count.notna()
            & (pieces_assembled > 0)
            & (max_seconds > 0)
            & (piece_count > 0)
        )

        extrapolated = (piece_count / pieces_assembled) * max_seconds
        completion.loc[valid_dnf] = extrapolated.loc[valid_dnf]

    # Legacy fallback when timing is derived from ppm and max piece count.
    if "ppm" in df.columns and "piece_count" in df.columns:
        ppm = pd.to_numeric(df.get("ppm"), errors="coerce")
        piece_count = pd.to_numeric(df.get("piece_count"), errors="coerce")
        derived_seconds = (1 / ppm) * piece_count * 60
        completion = completion.fillna(derived_seconds)

    return completion


def calculate_jpar(df: pd.DataFrame, minimum_participants_per_event: int) -> pd.DataFrame:
    out = add_event_date_column(df)
    out = out.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["event_id"] = out["event_id"].astype(str)
    out = out.sort_values(["event_date", "event_id", "completion_time"], na_position="last").reset_index(drop=True)

    out = fill_missing_pieces_assembled_from_sources(out)

    out["piece_count"] = out.groupby("event_id")["pieces_assembled"].transform("max")
    out["completion_seconds"] = compute_completion_seconds(out)
    out["resolved_member_id"] = resolved_member_id_series(out)

    # Preserve all input columns and add calculated columns.
    new_cols = [
        "event_participant_count",
        "eligible_participant_count",
        "event_mean_completion_seconds",
        "event_jpar",
        "expected_event_average",
        "mean_expected_event_average",
        "adjusted_event_jpar",
        "previous_jpar",
        "jpar_out",
        "latest_jpar",
    ]
    for col in new_cols:
        out[col] = np.nan

    latest_jpar_running: dict[str, float] = {}
    participant_history: dict[str, list[float]] = {}

    unique_events = out[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"])

    for _, ev in unique_events.iterrows():
        mask = (out["event_date"] == ev["event_date"]) & (out["event_id"] == ev["event_id"])
        sub_idx = out.index[mask]
        sub = out.loc[sub_idx]

        out.loc[sub_idx, "event_participant_count"] = len(sub)
        eligible = count_distinct_valid_members(sub)
        out.loc[sub_idx, "eligible_participant_count"] = eligible

        if len(sub) < minimum_participants_per_event:
            continue

        event_mean = sub.loc[sub["completion_seconds"].notna(), "completion_seconds"].mean()
        event_mean = np.round(event_mean) if pd.notna(event_mean) else np.nan
        out.loc[sub_idx, "event_mean_completion_seconds"] = event_mean
        out.loc[sub_idx, "event_jpar"] = sub["completion_seconds"] / event_mean if pd.notna(event_mean) else np.nan

        expected_values = []
        for idx in sub_idx:
            member_key = str(out.at[idx, "resolved_member_id"]) if pd.notna(out.at[idx, "resolved_member_id"]) else ""
            prev = latest_jpar_running.get(member_key)
            out.at[idx, "previous_jpar"] = prev if prev is not None else np.nan
            if prev is not None and pd.notna(prev):
                expected = out.at[idx, "completion_seconds"] * (1 / prev)
                out.at[idx, "expected_event_average"] = expected
                if pd.notna(expected):
                    expected_values.append(expected)

        mean_expected = np.mean(expected_values) if expected_values else np.nan
        out.loc[sub_idx, "mean_expected_event_average"] = mean_expected
        if pd.notna(mean_expected):
            out.loc[sub_idx, "adjusted_event_jpar"] = out.loc[sub_idx, "completion_seconds"] / mean_expected
        else:
            out.loc[sub_idx, "adjusted_event_jpar"] = out.loc[sub_idx, "event_jpar"]

        for idx in sub_idx:
            member_key = str(out.at[idx, "resolved_member_id"]) if pd.notna(out.at[idx, "resolved_member_id"]) else ""
            prev = latest_jpar_running.get(member_key)
            adj = out.at[idx, "adjusted_event_jpar"]
            if prev is None or pd.isna(prev):
                jpar_out = adj
            else:
                jpar_out = (prev + adj) / 2.0
            out.at[idx, "jpar_out"] = jpar_out
            latest_jpar_running[member_key] = jpar_out
            participant_history.setdefault(member_key, []).append(jpar_out)

    last_values = (
        out.sort_values(["event_date", "event_id"]) 
            .groupby("resolved_member_id", sort=False)["jpar_out"]
           .last()
    )
    out["latest_jpar"] = out["resolved_member_id"].map(last_values)
    out["legacy_jpar_out"] = out["jpar_out"]
    out["legacy_latest_jpar"] = out["latest_jpar"]
    return out


def calculate_z_normalized_jpar(df: pd.DataFrame, minimum_participants_per_event: int) -> pd.DataFrame:
    """Compute a z-normalized parallel JPAR track to reduce event-to-event drift.

    z_event_jpar is computed per event from completion_seconds using robust centering
    and scaling (median + MAD), then converted to a skill direction where higher is better.
    """
    out = df.copy()

    z_cols = [
        "z_event_center_seconds",
        "z_event_scale_seconds",
        "z_event_jpar",
        "z_previous_jpar",
        "z_jpar_out",
        "z_latest_jpar",
    ]
    for col in z_cols:
        out[col] = np.nan

    latest_z_running: dict[str, float] = {}

    unique_events = out[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date", "event_id"])
    for _, ev in unique_events.iterrows():
        mask = (out["event_date"] == ev["event_date"]) & (out["event_id"] == ev["event_id"])
        sub_idx = out.index[mask]
        sub = out.loc[sub_idx]

        if len(sub) < minimum_participants_per_event:
            continue

        values = pd.to_numeric(sub["completion_seconds"], errors="coerce").dropna()
        if len(values) < minimum_participants_per_event:
            continue

        center = float(values.median())
        mad = float((values - center).abs().median())
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            continue

        out.loc[sub_idx, "z_event_center_seconds"] = center
        out.loc[sub_idx, "z_event_scale_seconds"] = scale

        completion = pd.to_numeric(sub["completion_seconds"], errors="coerce")
        # Lower completion time means stronger performance, so negate z-score.
        z_event = -((completion - center) / scale)
        out.loc[sub_idx, "z_event_jpar"] = z_event

        for idx in sub_idx:
            member_key = str(out.at[idx, "resolved_member_id"]) if pd.notna(out.at[idx, "resolved_member_id"]) else ""
            prev = latest_z_running.get(member_key)
            out.at[idx, "z_previous_jpar"] = prev if prev is not None else np.nan

            z_value = out.at[idx, "z_event_jpar"]
            if pd.isna(z_value):
                continue

            if prev is None or pd.isna(prev):
                z_out = z_value
            else:
                z_out = (prev + z_value) / 2.0

            out.at[idx, "z_jpar_out"] = z_out
            latest_z_running[member_key] = z_out

    z_last_values = (
        out.sort_values(["event_date", "event_id"]) 
           .groupby("resolved_member_id", sort=False)["z_jpar_out"]
           .last()
    )
    out["z_latest_jpar"] = out["resolved_member_id"].map(z_last_values)
    return out


def export_active_only(calculation_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    out = calculation_df.copy()

    # Legacy switch retained for optional status filtering.
    if config.export_active_only and "membership_status" in out.columns:
        active_statuses = {s.strip().upper() for s in config.active_statuses}
        out["membership_status"] = out["membership_status"].astype(str).str.upper()
        out = out[out["membership_status"].isin(active_statuses)].copy()

    # Deliverables exclusion: drop members paid through after the configured cutoff.
    cutoff_text = str(config.deliverables_paid_through_max_date).strip()
    if cutoff_text and "paid_through_date" in out.columns:
        cutoff = pd.to_datetime(cutoff_text, errors="coerce")
        if pd.notna(cutoff):
            paid_through = pd.to_datetime(out["paid_through_date"], errors="coerce")
            out = out[paid_through.isna() | (paid_through <= cutoff)].copy()

    return out


def format_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().title().split())


def normalize_member_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def resolved_member_id_series(df: pd.DataFrame) -> pd.Series:
    event_member_id = df.get("member_id", pd.Series([pd.NA] * len(df), index=df.index))
    matched_member_id = df.get("mp_id", pd.Series([pd.NA] * len(df), index=df.index))

    event_member_id = event_member_id.apply(normalize_member_id)
    matched_member_id = matched_member_id.apply(normalize_member_id)
    return event_member_id.where(event_member_id != "", matched_member_id)


def with_member_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    name_col = None
    for candidate in ["member_full_name", "full_name", "source_full_name"]:
        if candidate in out.columns:
            name_col = candidate
            break

    out["__member_name"] = out[name_col].apply(format_name) if name_col else ""
    out["__member_id"] = resolved_member_id_series(out)
    return out


def get_member_latest_df(active_df: pd.DataFrame) -> pd.DataFrame:
    d = with_member_identity(active_df)
    d = d[(d["__member_name"] != "") & (d["__member_id"] != "")].copy()
    d["event_date"] = pd.to_datetime(d.get("event_date"), errors="coerce")
    d = d.sort_values(["event_date", "event_id"]).reset_index(drop=True)

    latest = d.groupby("__member_id", as_index=False).tail(1).copy()
    event_counts = (
        d.groupby("__member_id", as_index=False)["event_id"]
        .nunique()
        .rename(columns={"event_id": "# Events"})
    )

    latest = latest.merge(event_counts, on="__member_id", how="left")
    latest["JPAR"] = pd.to_numeric(latest.get("latest_jpar"), errors="coerce")
    latest.loc[latest["JPAR"].isna(), "JPAR"] = pd.to_numeric(latest.get("jpar_out"), errors="coerce")

    out = latest[["__member_name", "__member_id", "JPAR", "# Events"]].copy()
    out = out.rename(columns={"__member_name": "Name", "__member_id": "Member ID"})
    out = out.sort_values(["JPAR", "Member ID"], na_position="last").reset_index(drop=True)
    out.insert(0, "", np.arange(1, len(out) + 1))
    out["JPAR"] = out["JPAR"].round(4)
    return out


def make_jpar_by_name_df(active_df: pd.DataFrame) -> pd.DataFrame:
    by_rank = get_member_latest_df(active_df)
    by_name = by_rank.drop(columns=[""], errors="ignore").copy()
    by_name = by_name.sort_values(["Name", "Member ID"]).reset_index(drop=True)
    return by_name


def _ordinal(day: int) -> str:
    if 10 <= (day % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def format_event_date_from_id(event_id: object) -> object:
    token = str(event_id).split(".")[0]
    if len(token) < 6:
        return pd.NA
    token = token[-6:]
    try:
        yy, mm, dd = int(token[0:2]), int(token[2:4]), int(token[4:6])
        parsed = date(2000 + yy, mm, dd)
        return f"{parsed.strftime('%B')} {_ordinal(parsed.day)}, {parsed.year}"
    except Exception:
        return pd.NA


def parse_event_date_from_id(event_id: object) -> pd.Timestamp:
    token = str(event_id).split(".")[0]
    if len(token) >= 6:
        token = token[-6:]
        try:
            yy, mm, dd = int(token[0:2]), int(token[2:4]), int(token[4:6])
            return pd.Timestamp(date(2000 + yy, mm, dd))
        except Exception:
            pass

    # Legacy compact ids like 52723 represent MDDYY and need zero-padding.
    if len(token) == 5 and token.isdigit():
        try:
            return pd.to_datetime(token.zfill(6), format="%m%d%y", errors="coerce")
        except Exception:
            pass

    return pd.NaT


def make_qualified_events_df(calculation_df: pd.DataFrame) -> pd.DataFrame:
    events = calculation_df[["event_id"]].drop_duplicates(subset=["event_id"]).copy()
    events["Date"] = events["event_id"].apply(format_event_date_from_id)

    if "event_host" in calculation_df.columns:
        host_map = (
            calculation_df[["event_id", "event_host"]]
            .dropna(subset=["event_id"])
            .drop_duplicates(subset=["event_id"])
            .set_index("event_id")["event_host"]
        )
        events["Event Host"] = events["event_id"].map(host_map).fillna("")
    else:
        events["Event Host"] = ""

    events["Event Name"] = ""

    if "event_name" in calculation_df.columns:
        event_name_map = (
            calculation_df[["event_id", "event_name"]]
            .dropna(subset=["event_id"])
            .drop_duplicates(subset=["event_id"])
            .set_index("event_id")["event_name"]
        )
        events["Event Name"] = events["event_id"].map(event_name_map).fillna("")

    if (
        (events["Event Name"].astype(str).str.strip() == "").any()
        and "source_file" in calculation_df.columns
        and "source_sheet" in calculation_df.columns
    ):
        source_rows = (
            calculation_df[["event_id", "source_file", "source_sheet"]]
            .dropna(subset=["event_id", "source_file", "source_sheet"])
            .drop_duplicates(subset=["event_id", "source_file", "source_sheet"])
            .copy()
        )

        name_cache: dict[tuple[str, str], str | None] = {}

        def lookup_name(source_file: object, source_sheet: object) -> str | None:
            key = (clean_text(source_file), clean_text(source_sheet))
            if key not in name_cache:
                name_cache[key] = _extract_competition_name_from_sheet_metadata(source_file, source_sheet)
            return name_cache[key]

        source_rows["__event_name"] = source_rows.apply(
            lambda row: lookup_name(row["source_file"], row["source_sheet"]), axis=1
        )

        source_rows = source_rows[source_rows["__event_name"].fillna("") != ""]
        if not source_rows.empty:
            name_map = (
                source_rows.drop_duplicates(subset=["event_id"], keep="first")
                .set_index("event_id")["__event_name"]
            )
            existing = events["Event Name"].astype(str).str.strip()
            fill_mask = existing == ""
            events.loc[fill_mask, "Event Name"] = events.loc[fill_mask, "event_id"].map(name_map).fillna("")

    if (
        (events["Event Name"].astype(str).str.strip() == "").any()
        and "source_file" in calculation_df.columns
    ):
        source_file_map = (
            calculation_df[["event_id", "source_file"]]
            .dropna(subset=["event_id", "source_file"])
            .drop_duplicates(subset=["event_id"], keep="first")
            .set_index("event_id")["source_file"]
        )
        existing = events["Event Name"].astype(str).str.strip()
        fill_mask = existing == ""
        fallback_names = events.loc[fill_mask, "event_id"].apply(
            lambda eid: _derive_event_name_from_source_file(source_file_map.get(eid), eid)
        )
        events.loc[fill_mask, "Event Name"] = fallback_names.fillna("")

    events["Event ID"] = events["event_id"].astype(str)
    events["Notes"] = ""
    events["__sort_date"] = events["event_id"].apply(parse_event_date_from_id)
    events = events.sort_values(["__sort_date", "Event ID"], na_position="last")
    return events[["Date", "Event Host", "Event Name", "Event ID", "Notes"]].reset_index(drop=True)


def make_event_time_records_df(active_df: pd.DataFrame) -> pd.DataFrame:
    d = with_member_identity(active_df)
    d = d[(d["__member_name"] != "") & (d["__member_id"] != "")].copy()
    d["event_id_num"] = pd.to_numeric(d["event_id"], errors="coerce")
    d["completion_seconds_num"] = pd.to_numeric(d.get("completion_seconds"), errors="coerce")

    def format_seconds_to_hhmmss(value: object) -> str:
        seconds = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(seconds):
            return ""
        total = int(round(float(seconds)))
        if total < 0:
            return ""
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    d["completion_time_display"] = d["completion_seconds_num"].apply(format_seconds_to_hhmmss)
    if "completion_time" in d.columns:
        raw_time = d["completion_time"].fillna("").astype(str)
        d.loc[d["completion_time_display"] == "", "completion_time_display"] = raw_time

    d = d.sort_values(["__member_id", "event_id_num", "completion_seconds_num"], na_position="last")
    best = d.groupby(["__member_id", "event_id_num"], as_index=False).head(1)

    pivot = best.pivot_table(
        index=["__member_name", "__member_id"],
        columns="event_id_num",
        values="completion_time_display",
        aggfunc="first",
    )

    if len(pivot.columns):
        sorted_cols = sorted([col for col in pivot.columns if not pd.isna(col)])
        pivot = pivot.reindex(columns=sorted_cols)

    pivot = pivot.reset_index().rename(columns={"__member_name": "Name", "__member_id": "Member ID"})
    return pivot


def event_id_to_period(event_id: object) -> object:
    token = str(event_id).split(".")[0]
    if len(token) < 6:
        return pd.NA
    token = token[-6:]
    try:
        yy, mm = int(token[0:2]), int(token[2:4])
        year = 2000 + yy
        if 1 <= mm <= 12:
            return pd.Period(year=year, month=mm, freq="M")
    except Exception:
        pass
    return pd.NA


def make_monthly_outputs(active_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = with_member_identity(active_df)
    d = d[(d["__member_name"] != "") & (d["__member_id"] != "")].copy()
    d["event_month"] = d["event_id"].apply(event_id_to_period)
    d = d.dropna(subset=["event_month"]).copy()
    d = d.sort_values(["__member_id", "event_date", "event_id"], na_position="last")

    jpar_col = "jpar_out" if "jpar_out" in d.columns else "latest_jpar"
    d["jpar_value"] = pd.to_numeric(d[jpar_col], errors="coerce")

    competitions = (
        d.groupby(["event_month", "__member_id", "__member_name"], as_index=False)
        .size()
        .rename(columns={"size": "#"})
    )

    month_last = (
        d.groupby(["__member_id", "__member_name", "event_month"], as_index=False)["jpar_value"]
        .last()
    )

    all_months = pd.period_range(d["event_month"].min(), d["event_month"].max(), freq="M")
    all_members = d[["__member_id", "__member_name"]].drop_duplicates().reset_index(drop=True)

    grid = (
        all_members.assign(_key=1)
        .merge(pd.DataFrame({"event_month": all_months, "_key": 1}), on="_key", how="outer")
        .drop(columns=["_key"])
    )

    merged = grid.merge(month_last, on=["__member_id", "__member_name", "event_month"], how="left")
    merged = merged.merge(competitions, on=["event_month", "__member_id", "__member_name"], how="left")
    merged = merged.sort_values(["__member_id", "event_month"]).reset_index(drop=True)
    merged["jpar_value"] = merged.groupby("__member_id")["jpar_value"].ffill()
    merged["#"] = merged["#"].fillna(0).astype(int)

    monthly_list = merged.copy()
    monthly_list["Month"] = monthly_list["event_month"].astype(str)
    monthly_list = monthly_list.rename(columns={"__member_name": "Name", "__member_id": "Member ID", "jpar_value": "JPAR"})
    monthly_list["JPAR"] = monthly_list["JPAR"].round(4)
    monthly_list = monthly_list[["Month", "Name", "Member ID", "JPAR", "#"]]

    monthly_grid = monthly_list.pivot_table(
        index=["Name", "Member ID"],
        columns="Month",
        values="JPAR",
        aggfunc="last",
    )

    try:
        month_order = pd.PeriodIndex(monthly_grid.columns.astype(str), freq="M").sort_values().astype(str)
        monthly_grid = monthly_grid.reindex(columns=month_order)
    except Exception:
        monthly_grid = monthly_grid.reindex(columns=sorted(monthly_grid.columns))

    monthly_grid = monthly_grid.reset_index()
    return monthly_list, monthly_grid


def _write_deliverables_bundle(
    calculation_df: pd.DataFrame,
    active_df: pd.DataFrame,
    output_paths: dict[str, Path],
) -> dict[str, Path]:
    output_paths["calculation_df"].parent.mkdir(parents=True, exist_ok=True)

    calculation_df.to_csv(output_paths["calculation_df"], index=False)
    active_df.to_csv(output_paths["results"], index=False)

    # Build legacy deliverable tabs from current, cleaner pipeline outputs.
    jpar_by_rank = get_member_latest_df(active_df)
    jpar_by_name = make_jpar_by_name_df(active_df)
    qualified_events = make_qualified_events_df(calculation_df)
    event_time_records = make_event_time_records_df(active_df)
    monthly_list, monthly_grid = make_monthly_outputs(active_df)

    with pd.ExcelWriter(output_paths["deliverables_excel"], engine="openpyxl") as writer:
        jpar_by_rank.to_excel(writer, index=False, sheet_name="JPAR by Rank")
        jpar_by_name.to_excel(writer, index=False, sheet_name="JPAR by Name")
        qualified_events.to_excel(writer, index=False, sheet_name="Qualified Events")
        event_time_records.to_excel(writer, index=False, sheet_name="Event Time Records")
        monthly_list.to_excel(writer, index=False, sheet_name="JPAR Monthly List")
        monthly_grid.to_excel(writer, index=False, sheet_name="JPAR Monthly Grid")

    return output_paths


def write_deliverables(calculation_df: pd.DataFrame, active_df: pd.DataFrame, config: PipelineConfig) -> dict[str, Path]:
    output_paths = {
        "calculation_df": config.output_dir / config.calculation_df_name,
        "results": config.output_dir / config.results_name,
        "deliverables_excel": config.output_dir / config.deliverables_excel_name,
    }
    return _write_deliverables_bundle(calculation_df, active_df, output_paths)


def write_zscore_deliverables(calculation_df: pd.DataFrame, active_df: pd.DataFrame, config: PipelineConfig) -> dict[str, Path]:
    z_calc = calculation_df.copy()
    z_active = active_df.copy()

    if "z_jpar_out" in z_calc.columns:
        z_calc["jpar_out"] = z_calc["z_jpar_out"]
    if "z_latest_jpar" in z_calc.columns:
        z_calc["latest_jpar"] = z_calc["z_latest_jpar"]

    if "z_jpar_out" in z_active.columns:
        z_active["jpar_out"] = z_active["z_jpar_out"]
    if "z_latest_jpar" in z_active.columns:
        z_active["latest_jpar"] = z_active["z_latest_jpar"]

    output_paths = {
        "calculation_df": config.output_dir / config.zscore_calculation_df_name,
        "results": config.output_dir / config.zscore_results_name,
        "deliverables_excel": config.output_dir / config.zscore_deliverables_excel_name,
    }
    return _write_deliverables_bundle(z_calc, z_active, output_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Current script-first JPAR pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    joined = pd.read_csv(
        config.event_input_csv,
        dtype={
            "event_id": "string",
            "member_id": "string",
            "mp_id": "string",
            "event_member_id_norm": "string",
            "member_mp_id_norm": "string",
        },
    )
    joined = normalize_member_match_status(joined)
    joined = add_event_date_column(joined)
    joined = canonicalize_event_ids(joined)
    joined = filter_events(joined, config)
    joined = exclude_members_from_calculation_input(joined, config)

    calculation_df = calculate_jpar(joined, config.minimum_participants_per_event)
    calculation_df = calculate_z_normalized_jpar(calculation_df, config.minimum_participants_per_event)
    active_df = export_active_only(calculation_df, config)
    paths = write_deliverables(calculation_df, active_df, config)
    z_paths = None
    if config.export_zscore_outputs:
        z_paths = write_zscore_deliverables(calculation_df, active_df, config)

    print("JPAR pipeline complete.")
    print(f"Calculation DF: {paths['calculation_df']}")
    print(f"Results CSV: {paths['results']}")
    print(f"Deliverables Excel: {paths['deliverables_excel']}")
    if z_paths is not None:
        print(f"Z-Score Calculation DF: {z_paths['calculation_df']}")
        print(f"Z-Score Results CSV: {z_paths['results']}")
        print(f"Z-Score Deliverables Excel: {z_paths['deliverables_excel']}")
    print(f"Rows in calculation_df: {len(calculation_df)}")
    print(f"Rows in active export: {len(active_df)}")


if __name__ == "__main__":
    main()
