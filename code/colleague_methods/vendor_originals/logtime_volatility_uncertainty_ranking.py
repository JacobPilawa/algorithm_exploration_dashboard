#!/usr/bin/env python3

"""LogTime volatility & uncertainty ranking

LogTime model with Kalman filter dynamics (Glicko-2 inspired):

Per-player ability follows a random walk on log-time scale.
Uncertainty is tracked alongside the estimate and adapts to:
    - evidence shrinks uncertainty
    - inactivity grows uncertainty at the player's own volatility rate
    - surprising results increase volatility (detects improvement and decline)
    - sustained trend detection unfreezes hard filters (trend_boost)

Score = -ability_mean (lower log time = better)
Conservative score = -(ability_mean + 2*sigma) for prediction
"""

from datetime import date
import math
from collections import defaultdict

class LogTimeVolatilityUncertaintyRanking:
    """ LogTime with Kalman filter tracking per-player volatility/uncertainty """

    # ---- Observation noise (on log-time scale; sd = 17% of time) ----
    # E.g. we convert variance on logstage to variance on time scale:
    # variance = 0.3
    # std_dev = math.sqrt(variance) # 0.173
    # lower = 100*math.exp(-0.173) # -15.9%
    # upper = 100*math.exp(0.173)  # +18.9%

    OBS_VAR = 0.03


    # --- Prior for debut players (sd = 45%)
    PRIOR_VAR = 0.2

    # --- Volatility (skill drift per day)
    VOL_INIT = 3e-5     # corresponds to approx 10% drift over a year
    VOL_MIN = 3e-5      # floor: nobody is ever assumed perfectly static
    VOL_MAX = 1e-3      # ceiling: prevents wild stretches from blowing up

    # --- Volatility adaptation (surprising results)
    VOL_ADAPT = 0.1     # how much to increase volatility after a surprising - multiplicative rate
    TREND_BOOST = 0.35  # sensitivity to sustained improvement/decline

    # ---- Event tier weight (policy: nationals/worlds/etc? = 3x cleaner readings)
    TIER_OBS_FACTOR = 3.0

    # ---- difficulty model
    DIFFICULTY_ITERS = 2

    def __init__(self):
        self.obs = [] # player_id, event_id, log_time, date
        self.difficulty = defaultdict(float) 
        self.mean = {} # player ability mean
        self.var = {}  # player ability variance (uncertainty)
        self.vol = {}  # player volatility (skill drift)
        self.last = {} # last update date per player
        self.trend = {} # exponential moving average of signed innovation

    def _is_tier2_event(self, event_name):
        """Check if USA Nationals tier-3 (binding)."""
        import re
        nats = re.search(r"\bnationals?\b", event_name, re.I)
        usa = re.search(r"\busa?\b|\bu\.s\.", event_name, re.I)
        return bool(nats and usa)
    
    def _refit_difficulty(self):
        """Refit difficulty estimates for residuals."""
        for _ in range(self.DIFFICULTY_ITERS):
            num, den = defaultdict(float), defaultdict(float)
            for player_id, event_id, log_t, d, in self.obs:
                num[event_id] += log_t - self.mean.get(player_id, 0.0)
                den[event_id] += 1.0
            for event_id in num:
                if den[event_id] > 0:
                    self.difficulty[event_id] = num[event_id] / den[event_id]

    def update(self, event_id, date, event_name, event_results):
        """Process one event (finished results only).
        
        Args:
            event_id: unique event identifier
            date: date of event (datetime.date)
            event_name: name of the event (string)
            event_results: list of tuples (player_id, time_sec, rank)
        """
        # Log observations
        for player_id, time_sec, finished in event_results:
            if time_sec > 0:
                self.obs.append((player_id, event_id, math.log(time_sec), date))


        # refit difficulty
        self._refit_difficulty()
        d_event = self.difficulty[event_id]

        # Observation noise variance
        base_var = self.OBS_VAR
        if self.TIER_OBS_FACTOR > 1.0 and self._is_tier2_event(event_name):
            base_var = self.OBS_VAR / self.TIER_OBS_FACTOR

        # Kalman upadte per player
        for player_id, time_sec, finished in event_results:
            if time_sec <= 0:
                continue

            # DNF times are extrapolated estimates: much noisier 
            obs_var = base_var if finished else base_var * 4.00
            resid = math.log(time_sec) - d_event

            # Initialization: debut player
            if player_id not in self.mean:
                post_var = self.PRIOR_VAR * obs_var / (self.PRIOR_VAR + obs_var)
                gain = post_var / obs_var
                self.mean[player_id] = gain * resid
                self.var[player_id] = post_var
                self.vol[player_id] = self.VOL_INIT
                self.last[player_id] = date
                self.trend[player_id] = 0.0
                continue

            # --- Predict step: inactivity widens uncertainty at player's own rate
            days_since = max((date - self.last[player_id]).days, 1)
            prior_var = self.var[player_id] + self.vol[player_id] * days_since

            # Innovation
            innov = resid - self.mean[player_id]
            s = prior_var + obs_var

            # Adapt volatility from surprise
            z_sq = innov * innov / s
            tr = self.trend.get(player_id, 0.0)
            tr = 0.8 * tr + 0.2 * innov / math.sqrt(s)  # EMA of signed innovation
            self.trend[player_id] = tr
            boost = 1.0 + self.TREND_BOOST * tr * tr
            self.vol[player_id] = min(
                self.VOL_MAX, 
                max(
                    self.VOL_MIN, 
                    self.vol[player_id] * math.exp(self.VOL_ADAPT * (z_sq - 1.0))
                    * boost
                    ),
            )

            # Trend also grows uncertainty
            if self.TREND_BOOST:
                prior_var *= 1.0 + 0.5 * tr * tr
                s = prior_var + obs_var

            # Observe
            gain = prior_var / s
            self.mean[player_id] += gain * innov
            self.var[player_id] = (1.0 - gain) * prior_var
            self.last[player_id] = date

    def get_rating(self, player_id, view = 'mean'):
        """ Get rating (higher = better),
        
        Args:
            view: "mean" or "conservative" (mean + 2*sigma)
        """

        if player_id not in self.mean:
            return None
        if view == 'mean':
            return -self.mean[player_id]
        elif view == 'conservative':
            sd = math.sqrt(self.var[player_id])
            return -(self.mean[player_id] + 2 * sd)
                     
    def leaderboard(self, min_events =1, view = 'mean'):
        """ Return players ranked by rating.

        Args:
            min_events: minimum number of events to appear
            view: "mean" or "conservative"and
        
        Returns:
            list of (player_id, rating, mean, sigma, volatility, event_count)
        
        """
        board = []
        for player_id in self.mean:
            n_events = sum(1 for p, _, _, _ in self.obs if p == player_id)
            if n_events >= min_events:
                sigma = math.sqrt(self.var[player_id])
                rating = self.get_rating(player_id, view)
                board.append(
                    (player_id, rating, self.mean[player_id], sigma,
                     self.vol[player_id], n_events)
                )
        return sorted(board, key=lambda x: x[1], reverse=True)
    
# ----- example usage -----
if __name__ == "__main__":
    import datetime as dt

    system = LogTimeVolatilityUncertaintyRanking()

    # Event 1
    system.update("ev1", dt.date(2024, 1, 15), "Monthly Puzzle", [
        ("alice", 1800, True),
        ("bob", 1900, True),
        ("carol", 2000, True),
        ])
    
    # Event 2
    system.update("ev2", dt.date(2024, 2, 15), "Monthly Puzzle", [
        ("alice", 1750, True),
        ("bob", 2000, True),
        ("carol", 1900, True),
        ])
    
print("LogTime Volatility Rankings (higher = better):\n"
      "  player id | rating | ability | sigma | volatility | events")
for pid, rating, ability, sigma, vol, n in system.leaderboard():
    print(f" {pid:12} | {rating:8.3f} | {ability:7.3f} | {sigma:7.4f} | {vol:.2e} | {n}")