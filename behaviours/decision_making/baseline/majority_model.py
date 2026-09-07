"""
Majority-vote social behaviour (SOCIAL_MAJORITY) for Best-of-N
decision-making, ported from CThymioBestOfTwo::ProcessNeighborMessagesMajority.

ARGoS's range-and-bearing sensor delivers every neighbour's message each
tick, so the C++ controller tallies votes across ALL messages received in
a single tick and switches toward whichever option has the most support
(confidence-weighted, though the baseline control variant always sends
confidence=1.0, so in practice it's a plain vote count).

The real Thymio's prox.comm link only ever exposes ONE message per tick
(the last one received via prox.comm.rx), so there's nothing to tally
within a single tick. This ports the same tally/argmax/switch logic onto
a SLIDING WINDOW of the last `window_ticks` received messages instead:
each tick's one message is pushed into the window, votes are re-tallied
from whatever is currently in the window, and the same probabilistic
switch used by the voter model,

    p_switch = 1 - exp(-k * (q_winner - q_self))

is applied toward the window's current majority winner.

The winning option's quality is the MEAN of the quality values claimed
for it within the window, not the max. Taking the max over a window
that mostly accumulates repeated/independent noisy samples from the
same one or two nearby neighbours (rather than ARGoS's simultaneous
snapshot of however many distinct neighbours are in range this tick)
systematically inflates the reported quality as noise or window size
grows - the max of N noisy draws climbs toward 1.0 with N, so the
switch-probability formula below would see an almost-always-saturated
quality and become MORE confident as noise increases, the opposite of
the intended effect. The mean has no such bias.
"""

import math
import random
from collections import deque

from behaviours.decision_making.baseline.voter_model import noisy_measure  # noqa: F401


class MajorityVoteTally:
    """Rolling window of (opinion, quality) votes, one push per tick."""

    def __init__(self, num_options, window_ticks=20):
        self.num_options = num_options
        self.window_ticks = window_ticks
        self.votes = deque()

    def add(self, opinion, quality):
        self.votes.append((opinion, quality))
        while len(self.votes) > self.window_ticks:
            self.votes.popleft()

    def winner(self, rng=None):
        """
        Returns (winning_option, winning_quality): the option with the
        most votes in the window (ties broken randomly) and the MEAN
        quality claimed for it within the window (see module docstring
        for why mean, not max) - or (None, None) if the window is empty.
        """
        rng = rng or random
        if not self.votes:
            return None, None

        vote_count = [0.0] * self.num_options
        quality_sum = [0.0] * self.num_options
        for opinion, quality in self.votes:
            if opinion < 0 or opinion >= self.num_options:
                continue
            vote_count[opinion] += 1.0
            quality_sum[opinion] += quality

        max_votes = max(vote_count)
        if max_votes <= 0.0:
            return None, None

        ties = [k for k in range(self.num_options) if vote_count[k] == max_votes]
        chosen = rng.choice(ties)
        return chosen, quality_sum[chosen] / vote_count[chosen]


def process_majority_tally(opinion, q_est, tally, k=6.0, rng=None):
    """
    opinion / q_est: this robot's current opinion (-1 if none) and quality
        estimate.
    tally: a MajorityVoteTally already updated with this tick's message.
    k: steepness of the switching-probability curve (matches the C++
        constant 6.0 in ProcessNeighborMessagesMajority).

    Returns (new_opinion, new_q_est).
    """
    rng = rng or random
    chosen, winner_quality = tally.winner(rng=rng)
    if chosen is None:
        return opinion, q_est

    if opinion < 0:
        return chosen, winner_quality

    if winner_quality <= max(0.0, min(1.0, q_est)):
        return opinion, q_est

    p_switch = 1.0 - math.exp(-k * (winner_quality - q_est))
    if rng.uniform(0.0, 1.0) < p_switch:
        return chosen, winner_quality

    return opinion, q_est
