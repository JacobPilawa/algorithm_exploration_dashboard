#!/usr/bin/env python3

""" 
Minimal standalone script to do US ranking systems test.

Run any ranking system with CSV input

QUICKSTART:
    python3 rank_from_csv.py 4 results.csv -o output.csv

USAGE:
    python3 rank_from_csv.py <system> <csv_file> [-o output.csv]
    
SYSTEMS:
    4 = LogTime + Volatility (tracks speed + consistency)
    6 = Bayesian Skill Mu (Bayesian strength estimate)
    7 = Combined (Nationals binding + measurement)

INPUT CSV COLUMNS (auto-detected, case-insensitive):
    - event_id, event_name, event_date
    - player_id (or member_id, person_id)
    - completion_time_seconds_extrapolated (or time_seconds, seconds, completion_time)
    - Time format: H:MM:SS or MM:SS or raw seconds
    - Date format: YYY-MM-DD, MM/DD/YYYY, etc.

OUTPUT:
    Console display of top ranknigs
    Optional CSV ouput with: rank, player_id, score, uncertainty

EXAMPLES:
    # System 4 (LogTime + Volatility)
    python3 rank_from_csv.py 4 sample_input.csv -o log_v_ranking.csv
"""

import argparse
import csv
import datetime as dt
from collections import defaultdict

from simple_rating import parse_date

def parse_seconds(text):
    text = (text or "").strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        return parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) > 2 else 0)
    return float(text.replace(",", "")) 

def main():
    parser = argparse.ArgumentParser(description="Run ranking system from CSV input")
    parser.add_argument("system", choices = ["4", "6", "7"], help="System: 4 = LogTime + Volatility, 6 = Bayesian Skill Mu, 7 = Combined Nationals Constraint")
    parser.add_argument("csv", help="Input CSV file with event results")
    parser.add_argument("-o", "--output", help="Output CSV file (optional)")
    args = parser.parse_args()

    # Import the right system
    if args.system == "4":
        from ranking_systems.logtime_volatility_uncertainty_ranking import LogTimeVolatilityUncertaintyRanking as System
    elif args.system == "6":
        from ranking_systems.bayesian_skill_mu_ranking import BayesianSkillMuRanking as System
    elif args.system == "7":
        from ranking_systems.combined_nationals_constraint_ranking import CombinedNationalsConstraintRanking as System  
    else:
        raise ValueError(f"Unknown system: {args.system}")
    
    system = System()

    # Read CSV and process events
    events = defaultdict(lambda: {"date": None, "name": None, "results": []})
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            event_id = row.get("event_id") or row.get("event") 
            event_name = row.get("event_name") or row.get("name")
            event_date = row.get("event_date") or row.get("date")
            player_id = row.get("mid") or row.get("member_id") or row.get("member") or row.get("person_id") 
            time_str = (
                row.get("completion_time_seconds_extrapolated") or row.get("completion_time_seconds") or
                row.get("time_seconds") or row.get("seconds") or row.get("completion_time") or
                row.get("time") or row.get("completion_time")
            )

            if not all([event_id, event_name, event_date, player_id, time_str]):
                print(f"Skipping incomplete row: {row}")
                continue

            date = parse_date(event_date)
            time_sec = parse_seconds(time_str)
            if not date or not time_sec or time_sec <= 0:
               continue

            if events[event_id]["date"] is None:
                events[event_id]["date"] = date
                events[event_id]["name"] = event_name
            events[event_id]["results"].append((player_id, time_sec, True)) 

    # Process events
    print(f"System {args.system}: Processing {len(events)} events from {args.csv}")
    for event_id in sorted(events.keys(), key=lambda x: events[x]["date"]):
        event = events[event_id]
        system.update(event_id, event["date"], event["name"], event["results"])

    # Output
    leaderboard = system.leaderboard()
    print(f"\n{'Rank':<6} {'Player':<20} {'Score':<15} {'Uncertainty':<15}")
    print("-" * 60)
    for row in leaderboard:
        player_id, rank = row[0], row[1]
        score = row[2]
        uncertainty = row[3]
        print(f"{rank:<6} {player_id:<20} {score:<15.3f} {uncertainty:<15.3f}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "player_id", "score", "uncertainty"])
            for row in leaderboard:
                player_id, rank = row[0], row[1]
                score = row[2]
                uncertainty = row[3]
                writer.writerow([rank, player_id, score, uncertainty])
        print(f"\nOutput written to {args.output}")

    if __name__ == "__main__":
        main()


