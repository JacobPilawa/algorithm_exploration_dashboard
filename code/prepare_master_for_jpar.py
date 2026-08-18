#!/usr/bin/env python3
"""Adapt the master source-of-truth sheet to the JPAR pipeline input shape.

This script intentionally does not modify the raw master sheet. It writes a
derived joined-events CSV under data/data_event_results for the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def format_event_id(value: object) -> str:
    if pd.isna(value):
        return ""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if abs(float(numeric) - int(float(numeric))) < 1e-9:
            return f"{int(float(numeric)):06d}"
        return f"{float(numeric):09.2f}"
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def format_hhmmss(seconds: object, fallback: object = "") -> str:
    numeric = pd.to_numeric(pd.Series([seconds]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return clean_text(fallback)
    total = int(round(float(numeric)))
    if total < 0:
        return clean_text(fallback)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_master(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_pipeline_input(master: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id",
        "event_name",
        "event_date",
        "member_id",
        "first_name",
        "last_name",
        "pieces_assembled",
        "completion_time",
        "completion_time_seconds_extrapolated",
        "clean_source_file",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Master sheet is missing required columns: {missing}")

    out = pd.DataFrame(index=master.index)
    first = master["first_name"].apply(clean_text)
    last = master["last_name"].apply(clean_text)
    full_name = (first + " " + last).str.strip()

    out["full_name"] = full_name
    out["member_id"] = master["member_id"].apply(normalize_id).replace("", pd.NA)
    out["mp_id"] = out["member_id"]
    out["first_name"] = first
    out["last_name"] = last
    out["member_full_name"] = full_name
    out["event_id"] = master["event_id"].apply(format_event_id)
    out["event_name"] = master["event_name"].apply(clean_text)
    out["event_date"] = pd.to_datetime(master["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["event_host"] = ""
    out["src"] = "source_of_truth_master"
    out["source_file"] = master["clean_source_file"].apply(clean_text)
    out["source_sheet"] = master.get("heat", pd.Series([""] * len(master), index=master.index)).apply(clean_text)

    completion_seconds = pd.to_numeric(master["completion_time_seconds_extrapolated"], errors="coerce")
    out["completion_time_seconds"] = completion_seconds
    out["suggested_time_seconds"] = completion_seconds
    out["completion_time"] = [
        format_hhmmss(seconds, fallback)
        for seconds, fallback in zip(completion_seconds.tolist(), master["completion_time"].tolist())
    ]

    pieces = pd.to_numeric(master["pieces_assembled"], errors="coerce")
    event_piece_count = pieces.groupby(out["event_id"]).transform("max")
    out["pieces_assembled"] = pieces.fillna(event_piece_count)
    out["max_time_seconds"] = np.nan
    out["suggested_time_reason"] = "source_of_truth_extrapolated_seconds"
    out["membership_status"] = "ACTIVE"
    out["member_active_any_point_month"] = out["member_id"].notna()
    out["paid_through_date"] = ""
    out["initiated_date"] = ""
    out["match_method"] = np.where(out["member_id"].notna(), "master_member_id", "unmatched_name_only")
    out["member_match_status"] = np.where(out["member_id"].notna(), "matched_member", "unmatched_member")

    out["event_row_id"] = np.arange(len(out))
    out["event_member_id_norm"] = out["member_id"].fillna("")
    out["event_full_name_norm"] = full_name.str.lower()
    out["member_mp_id_norm"] = out["mp_id"].fillna("")
    out["membership_name_notes"] = ""
    out["email"] = ""
    out["name_candidate_count"] = np.where(out["member_id"].notna(), 1, 0)
    out["ambiguous_match_flag"] = False
    out["review_needed_flag"] = out["member_id"].isna()
    out["flag_high_time_outlier"] = False
    out["flag_time_parse_suspect"] = completion_seconds.isna()

    out = out[out["event_id"].astype(str).str.strip() != ""].copy()
    out = out[out["event_date"].astype(str).str.strip() != ""].copy()
    out = out[out["full_name"].astype(str).str.strip() != ""].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare master sheet for JPAR pipeline")
    parser.add_argument("--input", default="source_of_truth_merged.xlsx", help="Raw master sheet path")
    parser.add_argument(
        "--output",
        default="data/data_event_results/source_of_truth_jpar_input.csv",
        help="Derived CSV for the JPAR pipeline",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    master = read_master(input_path)
    derived = build_pipeline_input(master)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(output_path, index=False)

    scored = derived["completion_time_seconds"].notna().sum()
    matched = derived["member_id"].notna().sum()
    print(f"Wrote {output_path}")
    print(f"Rows: {len(derived):,}")
    print(f"Rows with completion seconds: {scored:,}")
    print(f"Rows with member_id: {matched:,}")
    print(f"Events: {derived['event_id'].nunique():,}")
    print(f"Date range: {derived['event_date'].min()} to {derived['event_date'].max()}")


if __name__ == "__main__":
    main()
