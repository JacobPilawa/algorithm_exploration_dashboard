#!/usr/bin/env python3
"""Reconstruct the three submitted ranking systems and emit dashboard adapters.

The files in ``vendor/`` are preserved verbatim.  This module repairs only the
execution-blocking defects and exposes a stable, removable CSV interface to the
existing JPAR dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMS = {
    "external_logtime": "Log-Time Volatility",
    "external_logtime_conservative": "Log-Time Volatility (Conservative)",
    "external_logtime_no_tier": "Log-Time Volatility (No Tier Boost)",
    "external_bayesian": "Bayesian Skill Mu",
    "external_bayesian_conservative": "Bayesian Skill (Conservative)",
    "external_nationals": "Nationals-Constrained Log-Time",
}


def keyify(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def is_usa_nationals(name: str) -> bool:
    return bool(re.search(r"\bnationals?\b", name, re.I) and re.search(r"\busa?\b|\bu\.s\.\b", name, re.I))


@dataclass
class LogTimeModel:
    """Faithful executable reconstruction of the submitted log-time model."""

    obs_var: float = 0.03
    prior_var: float = 0.2
    vol_init: float = 3e-5
    vol_min: float = 3e-5
    vol_max: float = 1e-3
    vol_adapt: float = 0.1
    trend_boost: float = 0.35
    tier_obs_factor: float = 3.0
    tier_usa_only: bool = True
    obs: list[tuple[str, str, float, date]] = field(default_factory=list)
    difficulty: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    mean: dict[str, float] = field(default_factory=dict)
    var: dict[str, float] = field(default_factory=dict)
    vol: dict[str, float] = field(default_factory=dict)
    last: dict[str, date] = field(default_factory=dict)
    trend: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def _is_tier(self, event_name: str) -> bool:
        if self.tier_usa_only:
            return is_usa_nationals(event_name)
        return bool(re.search(r"\bworlds?\b|\bnationals?\b", event_name, re.I))

    def _refit_difficulty(self) -> None:
        # The submitted two iterations are identical because player means are not
        # updated within the loop, so a single pass reproduces their result.
        num: dict[str, float] = defaultdict(float)
        den: dict[str, float] = defaultdict(float)
        for player_id, event_id, log_time, _ in self.obs:
            num[event_id] += log_time - self.mean.get(player_id, 0.0)
            den[event_id] += 1.0
        for event_id in num:
            self.difficulty[event_id] = num[event_id] / den[event_id]

    def update(self, event_id: str, when: date, event_name: str, results: list[tuple[str, float, bool]]) -> None:
        for player_id, seconds, _ in results:
            if seconds > 0:
                self.obs.append((player_id, event_id, math.log(seconds), when))
        self._refit_difficulty()
        event_difficulty = self.difficulty[event_id]
        base_var = self.obs_var
        if self.tier_obs_factor > 1 and self._is_tier(event_name):
            base_var /= self.tier_obs_factor

        for player_id, seconds, finished in results:
            if seconds <= 0:
                continue
            observation_var = base_var if finished else base_var * 4
            residual = math.log(seconds) - event_difficulty
            self.counts[player_id] += 1
            if player_id not in self.mean:
                post_var = self.prior_var * observation_var / (self.prior_var + observation_var)
                gain = post_var / observation_var
                self.mean[player_id] = gain * residual
                self.var[player_id] = post_var
                self.vol[player_id] = self.vol_init
                self.last[player_id] = when
                self.trend[player_id] = 0.0
                continue
            days = max((when - self.last[player_id]).days, 1)
            prior_var = self.var[player_id] + self.vol[player_id] * days
            innovation = residual - self.mean[player_id]
            total_var = prior_var + observation_var
            z_sq = innovation * innovation / total_var
            trend = 0.8 * self.trend.get(player_id, 0.0) + 0.2 * innovation / math.sqrt(total_var)
            self.trend[player_id] = trend
            boost = 1 + self.trend_boost * trend * trend
            self.vol[player_id] = min(
                self.vol_max,
                max(self.vol_min, self.vol[player_id] * math.exp(self.vol_adapt * (z_sq - 1)) * boost),
            )
            if self.trend_boost:
                prior_var *= 1 + 0.5 * trend * trend
                total_var = prior_var + observation_var
            gain = prior_var / total_var
            self.mean[player_id] += gain * innovation
            self.var[player_id] = (1 - gain) * prior_var
            self.last[player_id] = when

    def scores(self, conservative: bool = False) -> dict[str, float]:
        if conservative:
            return {
                player_id: -(ability + 2 * math.sqrt(self.var[player_id]))
                for player_id, ability in self.mean.items()
            }
        return {player_id: -ability for player_id, ability in self.mean.items()}

    def event_difficulty_from_observed(self, results: list[tuple[str, float, bool]]) -> float:
        values = [math.log(seconds) - self.mean.get(player_id, 0.0) for player_id, seconds, _ in results if seconds > 0]
        return float(np.mean(values)) if values else math.nan


@dataclass
class BayesianModel:
    """Executable reconstruction; main output uses the documented conservative view."""

    corrected_probability: bool = False
    players: dict[str, dict[str, float]] = field(default_factory=dict)

    def _player(self, player_id: str) -> dict[str, float]:
        return self.players.setdefault(player_id, {"mu": 1500.0, "sigma": 200.0, "events": 0.0})

    @staticmethod
    def _cdf(value: float) -> float:
        return 0.5 * (1 + math.erf(value / math.sqrt(2)))

    def _p_beat(self, a: dict[str, float], b: dict[str, float]) -> float:
        denom = math.sqrt(a["sigma"] ** 2 + b["sigma"] ** 2)
        z = (a["mu"] - b["mu"]) / denom if denom else 0.0
        # The submitted _p_beat divides by sqrt(2) before passing to a function
        # that already implements the standard-normal CDF. Keep that behavior in
        # the primary reconstruction; a corrected version is emitted for audit.
        return self._cdf(z if self.corrected_probability else z / math.sqrt(2))

    def update(self, finish_order: list[str]) -> None:
        if len(finish_order) < 2:
            return
        state = {pid: self._player(pid).copy() for pid in finish_order}
        n = len(finish_order)
        expected = {
            pid: sum(self._p_beat(state[pid], state[other]) for other in finish_order if other != pid) / (n - 1)
            for pid in finish_order
        }
        for position, pid in enumerate(finish_order):
            actual = 1 - position / (n - 1)
            previous = state[pid]
            k = max(16.0, min(32.0, 24.0 * previous["sigma"] / 200.0))
            player = self._player(pid)
            player["mu"] = previous["mu"] + k * (actual - expected[pid])
            player["sigma"] = max(30.0, previous["sigma"] * (0.95 if n >= 5 else 0.97))
            player["events"] = previous["events"] + 1

    def scores(self, conservative: bool = True) -> dict[str, float]:
        return {
            pid: values["mu"] - (2 * values["sigma"] if conservative else 0)
            for pid, values in self.players.items()
            if values["events"] > 0
        }


def ranks_desc(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))
    return {pid: index for index, pid in enumerate(ordered, 1)}


def constrained_order(measurement_scores: dict[str, float], nationals: dict[str, int]) -> list[str]:
    """Create a total order while enforcing Nationals order among its entrants.

    The submitted prose defines a pairwise rule that can be non-transitive.  This
    stable-slot reconstruction starts with measurement order and replaces only
    the slots occupied by Nationals entrants with those entrants in finish order.
    """
    measured = sorted(measurement_scores, key=lambda pid: (-measurement_scores[pid], pid))
    slots = [index for index, pid in enumerate(measured) if pid in nationals]
    national_order = sorted((pid for pid in measured if pid in nationals), key=lambda pid: nationals[pid])
    for slot, pid in zip(slots, national_order):
        measured[slot] = pid
    return measured


def load_events(path: Path) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, str]]:
    raw = pd.read_csv(path, low_memory=False, dtype={"event_id": "string", "resolved_member_id": "string"})
    raw["event_date"] = pd.to_datetime(raw["event_date"], errors="coerce")
    raw["seconds"] = pd.to_numeric(raw.get("completion_time_seconds", raw.get("completion_seconds")), errors="coerce")
    raw["member_key"] = raw["resolved_member_id"].apply(keyify)
    raw = raw.dropna(subset=["event_date", "event_id", "seconds"])
    raw = raw[raw["member_key"].ne("") & raw["seconds"].gt(0)].copy()
    raw = raw.sort_values(["event_date", "event_id", "seconds", "member_key"]).reset_index(drop=True)
    names = {
        row.member_key: str(row.full_name if pd.notna(row.full_name) else row.member_key)
        for row in raw[["member_key", "full_name"]].drop_duplicates("member_key", keep="last").itertuples()
    }
    events: list[dict[str, object]] = []
    for (event_date, event_id), rows in raw.groupby(["event_date", "event_id"], sort=True):
        event_name = str(rows["event_name"].dropna().iloc[0]) if rows["event_name"].notna().any() else str(event_id)
        results = [(str(row.member_key), float(row.seconds), True) for row in rows.itertuples()]
        events.append({"date": event_date.date(), "event_id": str(event_id), "event_name": event_name, "results": results})
    return raw, events, names


def build_outputs(input_path: Path, output_dir: Path) -> None:
    raw, events, names = load_events(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logtime = LogTimeModel()
    logtime_no_tier = LogTimeModel(tier_obs_factor=1.0)
    combined_measurement = LogTimeModel(tier_obs_factor=0.3, tier_usa_only=False)
    bayesian = BayesianModel()
    bayesian_corrected = BayesianModel(corrected_probability=True)
    nationals: dict[str, int] = {}
    history_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for event in events:
        event_id = str(event["event_id"])
        when = event["date"]
        event_name = str(event["event_name"])
        results = event["results"]
        participants = [pid for pid, _, _ in results]
        actual_times = {pid: seconds for pid, seconds, _ in results}
        actual_ranks = ranks_desc({pid: -seconds for pid, seconds in actual_times.items()})
        incoming = {
            "external_logtime": logtime.scores(),
            "external_logtime_conservative": logtime.scores(conservative=True),
            "external_logtime_no_tier": logtime_no_tier.scores(),
            "external_bayesian": bayesian.scores(conservative=False),
            "external_bayesian_conservative": bayesian.scores(conservative=True),
        }
        incoming_combined_order = constrained_order(combined_measurement.scores(), nationals)
        incoming["external_nationals"] = {pid: -rank for rank, pid in enumerate(incoming_combined_order, 1)}
        log_difficulty = logtime.event_difficulty_from_observed(results)
        log_no_tier_difficulty = logtime_no_tier.event_difficulty_from_observed(results)
        combined_difficulty = combined_measurement.event_difficulty_from_observed(results)

        for system_key, scores in incoming.items():
            eligible = [pid for pid in participants if pid in scores]
            predicted_ranks = ranks_desc({pid: scores[pid] for pid in eligible})
            eligible_actual = ranks_desc({pid: -actual_times[pid] for pid in eligible})
            for pid in eligible:
                predicted_time = math.nan
                if system_key == "external_logtime":
                    predicted_time = math.exp(log_difficulty - scores[pid])
                elif system_key == "external_logtime_conservative":
                    predicted_time = math.exp(log_difficulty - scores[pid])
                elif system_key == "external_logtime_no_tier":
                    predicted_time = math.exp(log_no_tier_difficulty - scores[pid])
                elif system_key == "external_nationals":
                    predicted_time = math.exp(combined_difficulty - combined_measurement.scores()[pid])
                elif eligible:
                    quantile = (predicted_ranks[pid] - 1) / max(len(eligible) - 1, 1)
                    predicted_time = float(np.quantile(list(actual_times.values()), quantile))
                prediction_rows.append({
                    "event_date": when.isoformat(), "event_id": event_id, "event_name": event_name,
                    "member_key": pid, "full_name": names.get(pid, pid), "system": system_key,
                    "incoming_score": scores[pid], "predicted_rank": predicted_ranks[pid],
                    "actual_rank_common_cohort": eligible_actual[pid], "actual_rank_full_event": actual_ranks[pid],
                    "actual_time": actual_times[pid], "predicted_time_diagnostic": predicted_time,
                    "eligible_entrants": len(eligible), "event_entrants": len(participants),
                })

        logtime.update(event_id, when, event_name, results)
        logtime_no_tier.update(event_id, when, event_name, results)
        combined_measurement.update(event_id, when, event_name, results)
        bayesian.update(participants)
        bayesian_corrected.update(participants)
        if is_usa_nationals(event_name):
            nationals = {pid: position for position, pid in enumerate(participants, 1)}
        combined_order = constrained_order(combined_measurement.scores(), nationals)
        post_scores = {
            "external_logtime": logtime.scores(),
            "external_logtime_conservative": logtime.scores(conservative=True),
            "external_logtime_no_tier": logtime_no_tier.scores(),
            "external_bayesian": bayesian.scores(conservative=False),
            "external_bayesian_conservative": bayesian.scores(conservative=True),
            "external_nationals": {pid: -rank for rank, pid in enumerate(combined_order, 1)},
        }
        all_players = sorted(set().union(*(set(scores) for scores in post_scores.values())))
        for pid in all_players:
            history_rows.append({
                "event_date": when.isoformat(), "event_id": event_id, "event_name": event_name,
                "member_key": pid, "full_name": names.get(pid, pid),
                "events": max(logtime.counts.get(pid, 0), int(bayesian.players.get(pid, {}).get("events", 0))),
                **{key: scores.get(pid) for key, scores in post_scores.items()},
            })

    final_scores = {
        "external_logtime": logtime.scores(),
        "external_logtime_conservative": logtime.scores(conservative=True),
        "external_logtime_no_tier": logtime_no_tier.scores(),
        "external_bayesian": bayesian.scores(conservative=False),
        "external_bayesian_conservative": bayesian.scores(conservative=True),
        "external_nationals": {pid: -rank for rank, pid in enumerate(constrained_order(combined_measurement.scores(), nationals), 1)},
    }
    all_players = sorted(set().union(*(set(scores) for scores in final_scores.values())))
    final_ranks = {key: ranks_desc(scores) for key, scores in final_scores.items()}
    leaderboard = pd.DataFrame([
        {
            "member_key": pid, "full_name": names.get(pid, pid), "external_events": logtime.counts.get(pid, 0),
            **{key: final_scores[key].get(pid) for key in SYSTEMS},
            **{f"{key}_rank": final_ranks[key].get(pid) for key in SYSTEMS},
            "external_logtime_sigma": math.sqrt(logtime.var[pid]) if pid in logtime.var else np.nan,
            "external_logtime_volatility": logtime.vol.get(pid),
            "external_logtime_no_tier_sigma": math.sqrt(logtime_no_tier.var[pid]) if pid in logtime_no_tier.var else np.nan,
            "external_logtime_no_tier_volatility": logtime_no_tier.vol.get(pid),
            "external_bayesian_mu": bayesian.players.get(pid, {}).get("mu"),
            "external_bayesian_sigma": bayesian.players.get(pid, {}).get("sigma"),
            "external_bayesian_corrected_score": bayesian_corrected.scores(conservative=False).get(pid),
        }
        for pid in all_players
    ])
    history = pd.DataFrame(history_rows)
    predictions = pd.DataFrame(prediction_rows)
    leaderboard.to_csv(output_dir / "final_leaderboard.csv", index=False)
    history.to_csv(output_dir / "state_history.csv", index=False)
    predictions.to_csv(output_dir / "rolling_predictions.csv", index=False)

    summary_rows = []
    for system_key, group in predictions.groupby("system"):
        usable = group[group["eligible_entrants"].ge(2)].copy()
        usable["predicted_pct"] = (usable["predicted_rank"] - 1) / (usable["eligible_entrants"] - 1)
        usable["actual_pct"] = (usable["actual_rank_common_cohort"] - 1) / (usable["eligible_entrants"] - 1)
        summary_rows.append({
            "system": system_key, "entrant_event_predictions": len(usable), "events": usable["event_id"].nunique(),
            "rank_percentile_mae": float((usable["predicted_pct"] - usable["actual_pct"]).abs().mean()),
            "rank_percentile_spearman": float(usable["predicted_pct"].corr(usable["actual_pct"], method="spearman")),
            "time_mae_seconds_diagnostic": float((usable["predicted_time_diagnostic"] - usable["actual_time"]).abs().mean()),
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "audit_summary_metrics.csv", index=False)
    metadata = {
        "input": str(input_path), "usable_rows": len(raw), "events": len(events), "people": len(all_players),
        "systems": SYSTEMS, "vendor_source_commit": "3141f455db2fec91567c1c0b7eb620f6afba6201",
        "nationals_events_detected": [e["event_name"] for e in events if is_usa_nationals(str(e["event_name"]))],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/data_jpar_v2/source_of_truth_calculation_df.csv")
    parser.add_argument("--output-dir", default="colleague_ranking_systems/outputs")
    args = parser.parse_args()
    build_outputs(Path(args.input), Path(args.output_dir))
    print(f"Wrote reconstructed ranking outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
