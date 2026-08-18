#!/usr/bin/env python3

"""
Combined Nationals Constraint Ranking

Preferred current system. Combines two layers:

1. MESUREMENT LAYER (LogTimeKalman): estimates current speed from
difficult-adjusteod log tmes, trackeod with per-player volatility/uncertainty

2. ORDERING LAYER (Nationals Rule): the most recent USA Nationals standings
bind the published order among its participants. Only participants at the
current nationalscan move relative to each other; everyone else is
slotted by measured speed.

The rule: "Nobody passes you by grinding online events; they have to beat you at
an 'authoratative' event (i.e. Nationals for the moment). This balances measurement
accuracy against ordering legitimacy.

Score = combined rank (lower = better). For pairwise comparison:
    - if both attended current nationals -> higher nationals finish wins
    - else -> higher measured speed wins
"""

import math
import datetime as dt
from collections import defaultdict

class CombinedNationalsConstraintRanking:
    """The preferred system: measurement (LogTime+Vol) + nationals binding."""

    # ---- LogTimeKalman parameters
    OBS_VAR = 0.030
    PRIOR_VAR = 0.2
    VOL_INIT = 3e-5
    VOL_MIN = 3e-5
    VOL_MAX = 1e-3
    VOL_ADAPT = 0.1
    TREND_BOOST = 0.35
    TIER_OBS_FACTOR = 0.3
    DIFFICULTY_ITERS = 2

    # ---- Nationals binding window (one Championship weekend)
    BIND_WINDOW_DAYS = 10

    def __init__(self):
        # Measurement layer (LogTimeKalman)
        self.obs = []
        self.difficulty = defaultdict(float)
        self.mean = {}
        self.var = {}
        self.vol = {}
        self.last = {}
        self.trend = {}

        # Top tier authoriative event (Nationals, for now) layer
        self.nationals_events = [] # [date, name, results: [player_ids]]
        self.nationals_standings = {} # player_id -> rank

    def _is_tier2_event(self, event_name):
        """Check if tier-2 (authoritative)."""
        import re
        return bool(
            re.search(r"\bworlds?\b", event_name, re.I) or
            re.search(r"\bnationals?\b", event_name, re.I)
        )
    
    # Note to team: We can change this if we want by maintaining a list of nationals
    # or adding a label or something, esp. if we start labelling events
    # in tiers by sactioned vs in approval etc <- can use this column to 
    # mark tiers of authority
    def _is_usa_nationals(self, event_name):
        """Check if USA Nationals tier-3 (binding)."""
        import re
        nats = re.search(r"\bnationals?\b, event_name, re.I")
        usa = re.search(r"\busa?\b|\bu\.s\.", event_name, re.I)
        return bool(nats and usa)
    
    def _refit_difficulty(self):
        """Refit event difficulties"""
        for _ in range(self.DIFFICULTY_ITERS):
            num, den = defaultdict(float), defaultdict(float)
            for player_id, event_id, log_t, d in self.obs:
                num[event_id] += log_t - self.mean.get(player_id, 0.0)
                den[event_id] += 1.0
        for event_id in num:
            if den[event_id] > 0:
                self.difficulty[event_id] = num[event_id]/den[event_id]

    def update(self, event_id, date, event_name, event_results):

        """Process one event.
        
        Args:
            event_id: unique identifier
            date: datetime.date
            event_name: name (checked for tier detection)
            event_results: list of (player_id, time_sec, finished) 
        """

        # Log observations for measurement layer
        for player_id, time_sec, finished in event_results:
            if time_sec > 0:
                self.obs.append((player_id, event_id, math.log(time_sec), date))

        # Track nationals events separately
        if self._is_usa_nationals(event_name):
            player_ids = [p for p, _, _ in event_results]
            self.nationals_events.append({
                "date": date,
                "name": event_name,
                "results": player_ids
            })
            # Update standings
            self.nationals_standings = {p: i for i, p in enumerate(player_ids)}

        # Refit difficulty
        self._refit_difficulty()
        d_event = self.difficulty[event_id]

        # Observation noise
        base_var = self.OBS_VAR
        if self.TIER_OBS_FACTOR > 1 and self._is_tier2_event(event_name):
            base_var = self.OBS_VAR/ self.TIER_OBS_FACTOR

        # Kalman update
        for player_id, time_sec, finished in event_results:
            if time_sec <=0:
                continue

            obs_var = base_var if finished else base_var * 4.0
            resid = math.log(time_sec) - d_event

            # For a new person
            if player_id not in self.mean:
                post_var = self.PRIOR_VAR*obs_var / (self.PRIOR_VAR + obs_var)
                gain = post_var / obs_var
                self.mean[player_id] = gain * resid
                self.var[player_id] = post_var
                self.vol[player_id] = self.VOL_INIT
                self.last[player_id] = date
                self.trend[player_id] = 0.0
                continue

            # For someone returning:
            # How long since we last saw them? (inactivity > uncertainty grows)
            days_since = max((date - self.last[player_id]).days, 1)
            prior_var = self.var[player_id] + self.vol[player_id]*days_since
            # Suprise: How they do better/wores than expected
            innov = resid - self.mean[player_id]
            # total uncertainty
            s = prior_var + obs_var
            # standalone suprise
            z_sq = innov * innov / s
            tr = 0.8 * self.trend.get(player_id, 0.0) + 0.2 * (innov/math.sqrt(s))
            self.trend[player_id] = tr
            # Volatility adaptation: if surprising result, icrease drift
            boost = 1.0 + self.TREND_BOOST * tr * tr
            self.vol[player_id] = min(
                self.VOL_MAX,
                max(
                    self.VOL_MIN,
                    self.vol[player_id]*math.exp(self.VOL_ADAPT*(z_sq-1.0))
                    *boost,
                ),
            )
            if self.TREND_BOOST:
                    prior_var *= 1.0 + 0.5 * tr * tr
                    s = prior_var + obs_var
                    # Kalman gain: how muc to update the estimate?
                    gain = prior_var / s
                    # Update ability estimate
                    self.mean[player_id] += gain * innov
                    self.var[player_id] = (1.0 - gain)*prior_var
                    self.last[player_id] = date

    def _get_measurement_order(self, players):
        """Sort players by measurement (higher = better)."""
        return sorted(players, key = lambda p: -self.mean.get(p, 0.0))
    
    def leaderboard(self, min_events = 1):
        """ Return combined ranking (nationals binding + measurement).
        
        Returns:
            list of (player_id, combined_rank, mu, sigma, volatility, events)
        """

        all_players = list(self.mean.keys())
        nationals_players = set(self.nationals_standings.keys())

        # Start with nationals order (those who competed)
        ranked = []
        for player_id in sorted(nationals_players, 
                                key=lambda p: self.nationals_standings[p]):
            n_events = sum(1 for x, _, _, _ in self.obs if x == player_id)
            if n_events >= min_events:
                ranked.append(player_id)

        # Add non-nationals players by measurement order
        non_nationals = [p for p in all_players if p not in nationals_players]
        for player_id in self._get_measurement_order(non_nationals):
            n_events = sum(1 for x, _, _, _ in self.obs if x == player_id)
            if n_events >= min_events:
                ranked.append(player_id)

        # Return with metadata
        result = []
        for combined_rank, player_id in enumerate(ranked, 1):
            sigma = math.sqrt(self.var.get(player_id, self.PRIOR_VAR))
            n_events = sum(1 for x, _, _, _ in self.obs if x == player_id)
            if n_events >= min_events:
                ranked.append((
                    player_id, combined_rank, 
                    self.mean.get(player_id, 0.0), 
                    sigma, 
                    self.vol.get(player_id, self.VOL_INIT), 
                    n_events
                ))
        return result
    
# ---- example usage ----

if __name__ == "__main__":
    import datetime as dt

    system = CombinedNationalsConstraintRanking()

    # Early events
    system.update("ev1", dt.date(2024, 1, 15), "Monthly Event", [
        ("alice", 1800.0, True),
        ("bob", 1900.0, True),
        ("carol", 2000.0, True),
    ])

    system.update("ev2", dt.date(2024, 2, 15), "Monthly Event", [
        ("alice", 1750.0, True),
        ("bob", 2000.0, True),
        ("carol", 1900.0, True),
    ])

    print("Before Nationals (measurement only):")
    for pid, rank, mu, sigma, vol, n_events in system.leaderboard():
        print(f"{pid}: rank={rank}, mu={mu:.3f}, sigma={sigma:.3f}, vol={vol:.6f}, events={n_events}")

    system.update("nats_2024", dt.date(2024, 7, 20), "USA Nationals", [
        ("alice", 1900.0, True),
        ("bob", 2050.0, True),
        ("carol", 1850.0, True), # <- carol wins nationals, should move ahead of alice and bob in combined ranking
    ])

    print("\nAfter Nationals (nationals binding + measurement):")
    print(" Nationals standings published order respects nationals results, but non-nationals are ordered by measured speed.")
    for pid, rank, mu, sigma, vol, n_events in system.leaderboard():
        nat_rank = system.nationals_standings.get(pid, None)
        print(f"{pid}: rank={rank}, mu={mu:.3f}, sigma={sigma:.3f}, vol={vol:.6f}, events={n_events}, nationals_rank={nat_rank}")
    print("\nNote: Nationals binding ensures that those who competed at Nationals are ordered according to their Nationals finish, while others are ranked by their measured speed.")




    