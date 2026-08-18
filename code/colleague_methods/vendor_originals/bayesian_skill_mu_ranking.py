#!/usr/bin/env python3

""" Bayesian Skill Mu Ranking

Simple Bayesian skill rating system inspired byTrueSkill/Glicko.
Each player has a skill estimate (mu) and uncertainty (sigma).

After each event finish:
1. Estimate expected percentile from mu vs other participants
2. Ajust mu based on actual finish vs expected (K-factor scaled by sigma)
3. Reduce sigma (confidence grows with more evidence)

Highe mu = better skill
Conservative ranking uses (mu - 2*sigma) to account for uncertainty

"""

from collections import defaultdict
import math

class BayesianSkillMuRanking:
    """ Bayesian skill rating with mu and sigma """

    # ---- Constraints
    START_MU = 1500
    START_SIGMA = 200
    MIN_SIGMA = 30
    MAX_K = 32.0
    MIN_K = 16.0
    BASE_K = 24.0 # K-factor for typical uncertainty

    # ---- Sigma reduction (shrinkage) based on event size
    SHRINK_LARGE = 0.95 # 5 players: sigma *= 0.95
    SHRINK_SMALL = 0.97 # 2 players: sigma *= 0.97

    def __init__(self):
        self.players = defaultdict(lambda: {
            "mu": self.START_MU,
            "sigma": self.START_SIGMA,
            "event_count": 0,    
        })

    def _normal_cdf(self, x):
        """ Approximate cumulative distribution function of standard normal. """
        # Abramowitz and Stegun approximation
        a = 0.254829592
        b = -0.284496736
        c = 1.421413741
        d = -1.453152027
        e = 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2.0)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((e * t + d) * t + c) * t + b) * t + a) * t) * math.exp(-x * x) 
        return 0.5 * (1.0 + sign * y)
    
    def _p_beat(self, mu_a, sigma_a, mu_b, sigma_b):
        """ Probability that player A beats player B. """
        delta_mu = mu_a - mu_b
        delta_sigma = math.sqrt(sigma_a ** 2 + sigma_b ** 2)
        z = delta_mu / delta_sigma if delta_sigma > 0 else 0
        return self._normal_cdf(z / math.sqrt(2.0))
    
    def update(self, event_results):
        """ Process one event (players in finish order).
        Args:
            event_results: list of player_ids in finish order 
        """
        n = len(event_results)
        if n < 2:
            return
        
        # Compute expected finish precentile per player vs al others
        member_ids = event_results
        pre_mu = {m: self.players[m]["mu"] for m in member_ids}
        pre_sigma = {m: self.players[m]["sigma"] for m in member_ids}

        expected = {}
        for i, m in enumerate(member_ids):
            exp_percentile = sum(
                self._p_beat(pre_mu[m], pre_sigma[m], 
                             pre_mu[o], pre_sigma[o])
                for o in member_ids if o != m
            ) / (n - 1)
            expected[m] = exp_percentile

        # Actual finish percentile (0 = last, 1 = first)
        actual = {}
        for i, m in enumerate(member_ids):
            actual[m] = 1.0 - (i / (n - 1)) if n > 1 else 0.5

        # Update each player
        shrink_factor = self.SHRINK_LARGE if n >= 5 else self.SHRINK_SMALL

        for m in member_ids:
            # K-factor scales with uncertainty
            k = max(
                self.MIN_K, 
                min(self.MAX_K,
                     self.BASE_K * (pre_sigma[m] / self.START_SIGMA))
            )
            
            # Update mu
            delta = actual[m] - expected[m]
            self.players[m]["mu"] = pre_mu[m] + k * delta

            # Reduce sigma (evidence increases confidence)
            self.players[m]["sigma"] = max(
                self.MIN_SIGMA, pre_sigma[m] * shrink_factor
            )

            # Track events
            self.players[m]["event_count"] += 1

            def get_rating(self, player_id, view='mu'):
                """
                Get player rating.

                Args: 
                    view: "mu" or "conservative" (mu - 2*sigma)
                """
                p = self.players[player_id]
                if view == 'mu':
                    return p["mu"]
                elif view == 'conservative':
                    return p["mu"] - 2 * p["sigma"]
                
            def leaderboard(self, min_events=1, view='mu'):
                """
                Return players ranked by skill.

                Args:
                    min_events: minimum number of events to appear
                    view: "mu" or "conservative"

                Returns:

                """ 

                board = [
                    (pid, self.get_rating(pid, view),
                     p["mu"], p["sigma"], p["event_count"])
                    for pid, p in self.players.items()
                    if p["event_count"] >= min_events
                ]
                return sorted(board, key=lambda x: x[1], reverse=True)
            
    # ---- example usage ----
# if __name__ == "__main__":

#     system = BayesianSkillMuRanking()

#     # Event 1: alice > bob > carol
#     system.update(["alice", "bob", "carol"])
#     print("\nAfter Event 1: (alice > bob > carol)")
#     for pid, rating, mu, sigma, n in system.leaderboard():
#         print(f"{pid:12s} | {rating:7.2f} | mu={mu:7.2f} | sigma={sigma:7.2f} | events={n}")
# ... meh