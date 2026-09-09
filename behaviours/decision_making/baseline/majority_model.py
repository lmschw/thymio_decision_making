"""
Majority-vote social behaviour (SOCIAL_MAJORITY) for Best-of-N
decision-making, ported from CThymioBestOfTwo::ProcessNeighborMessagesMajority.

ARGoS's range-and-bearing sensor delivers every neighbour's message each
tick, so the C++ controller tallies votes across ALL messages received in
a single tick and switches toward whichever option has the most support
(confidence-weighted, though the baseline control variant always sends
confidence=1.0, so in practice it's a plain vote count).

Two real-hardware constraints previously stood in for this with a
sliding window of the last N prox.comm messages, which turned out to be
a poor substitute: the real Thymio's prox.comm link only carries ~10
bits (heavily quantising the quality value) and only ever exposes ONE
message per tick with no sender id, so a window accumulated over time
was mostly repeated samples from the same one or two nearby neighbours
rather than distinct simultaneous ones - both taking the max AND taking
many repeated draws from one sender inflated the reported quality far
above what ARGoS's single-tick, multi-neighbour snapshot would show
(confirmed by simulation: at noise_sigma=0.4, max-over-a-20-sample
window averaged 0.998 for a true quality of 0.6).

This module now works over a genuine set of DISTINCT, currently-nearby
neighbour ids instead:
  - `IdBuffer` is fed by a lightweight id-only prox.comm broadcast (see
    MajorityVotingBaselineExperiment._tick) - it is deduplicated by
    construction (one entry per id) and evicts ids that haven't been
    refreshed recently, so a single persistent neighbour occupies
    exactly one "slot" no matter how many ticks it's been in range.
  - the robot's own (opinion, quality) is exchanged with every other
    robot in the swarm via the coordinator (Robot.exchange_swarm_info),
    at full float precision - no quantisation.
  - `majority_from_swarm_info` tallies votes and mean quality only over
    ids currently in the IdBuffer, restoring the "distinct simultaneous
    neighbours" semantics ARGoS's RAB sensor gives for free, and mean
    (not max) so an option's reported quality doesn't inflate merely
    because more neighbours happen to be nearby right now.
"""

import math
import random


class IdBuffer:
    """
    Bounded, deduplicated, recency-ordered set of neighbour ids, refreshed
    by a lightweight id-only prox.comm broadcast (see
    MajorityVotingBaselineExperiment._tick). Acts as the locality filter
    over the coordinator's globally-relayed swarm-info dict: a neighbour's
    rich (opinion, quality) data is only consulted while their id is
    present in this buffer.

    `capacity`: max number of distinct ids remembered at once - the
    least-recently-refreshed id is evicted first once exceeded.
    `max_age_ticks`: an id not refreshed within this many ticks is
    evicted even if capacity hasn't been reached, so "currently nearby"
    stays current rather than an id lingering indefinitely after a
    neighbour has actually moved out of range. ARGoS's range-and-bearing
    sensor has no memory at all (a fresh, instantaneous snapshot every
    tick), so there's no literal reference value to port here - these
    two defaults are a hardware-appropriate stand-in, not a measured
    ARGoS constant, and are worth tuning against real encounter
    durations once this runs on the robots.
    """

    def __init__(self, capacity=10, max_age_ticks=20):
        self.capacity = capacity
        self.max_age_ticks = max_age_ticks
        self._last_seen_tick = {}  # id -> tick last refreshed

    def refresh(self, neighbor_id, tick):
        """Marks `neighbor_id` as seen at `tick`, evicting the
        least-recently-seen id if that pushes the buffer over capacity."""
        self._last_seen_tick[neighbor_id] = tick
        if len(self._last_seen_tick) > self.capacity:
            oldest_id = min(self._last_seen_tick, key=self._last_seen_tick.get)
            del self._last_seen_tick[oldest_id]

    def active_ids(self, tick):
        """Evicts ids older than `max_age_ticks` and returns the
        remaining (still-recent) ids as a set."""
        cutoff = tick - self.max_age_ticks
        expired = [nid for nid, t in self._last_seen_tick.items() if t < cutoff]
        for nid in expired:
            del self._last_seen_tick[nid]
        return set(self._last_seen_tick.keys())


def majority_from_swarm_info(active_ids, swarm_info, num_options):
    """
    active_ids: ids currently in the local IdBuffer (the locality filter -
        ids outside this set are ignored even if present in swarm_info).
    swarm_info: {id: {"opinion": int, "quality": float}}, as returned by
        Robot.exchange_swarm_info - the coordinator's merged dict for the
        whole swarm.

    Returns (winning_option, mean_quality): the option held by the most
    distinct active neighbours (ties broken randomly) and the mean
    quality they reported for it - or (None, None) if no active neighbour
    has usable info.
    """
    vote_count = [0.0] * num_options
    quality_sum = [0.0] * num_options
    for neighbor_id in active_ids:
        info = swarm_info.get(neighbor_id)
        if not info:
            continue
        op = info.get("opinion")
        q = info.get("quality")
        if op is None or q is None or not (0 <= op < num_options):
            continue
        vote_count[op] += 1.0
        quality_sum[op] += q

    max_votes = max(vote_count) if vote_count else 0.0
    if max_votes <= 0.0:
        return None, None

    ties = [k for k in range(num_options) if vote_count[k] == max_votes]
    chosen = random.choice(ties)
    return chosen, quality_sum[chosen] / vote_count[chosen]


def process_majority_vote(opinion, q_est, chosen, winner_quality, k=6.0, rng=None):
    """
    opinion / q_est: this robot's current opinion (-1 if none) and quality
        estimate.
    chosen / winner_quality: from majority_from_swarm_info() this tick.
    k: steepness of the switching-probability curve (matches the C++
        constant 6.0 in ProcessNeighborMessagesMajority).

    Returns (new_opinion, new_q_est).
    """
    rng = rng or random
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
