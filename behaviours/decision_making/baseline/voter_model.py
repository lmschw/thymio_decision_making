"""
Voter model social behaviour (SOCIAL_VOTER_MODEL) for Best-of-N
decision-making, ported from CThymioBestOfTwo::ProcessOneNeighborMessage.

Each tick, a disseminating robot hears at most one neighbour's
(opinion, quality) message and probabilistically adopts it:

  - if the robot has no opinion yet, it adopts the neighbour's opinion
    with probability equal to the neighbour's reported quality
    (higher-quality claims recruit more readily);
  - if the robot already has an opinion, it only considers switching when
    the neighbour's quality is strictly higher than its own estimate, and
    even then only switches with probability

        p_switch = 1 - exp(-k * (q_neighbour - q_self))

    so a marginally-better neighbour rarely causes a switch, while a much
    better one almost always does.
"""

import math
import random


def noisy_measure(q_true, noise_sigma, rng=None):
    """Adds Gaussian noise when noise_sigma > 0 (noiseless otherwise)."""
    rng = rng or random
    if noise_sigma <= 0.0:
        return max(0.0, min(1.0, q_true))
    return max(0.0, min(1.0, q_true + rng.gauss(0.0, noise_sigma)))


def process_one_neighbor_message(opinion, q_est, neighbor_opinion, neighbor_quality,
                                  k=6.0, rng=None):
    """
    opinion / q_est: this robot's current opinion (-1 if none) and quality
        estimate.
    neighbor_opinion / neighbor_quality: decoded from the received message.
    k: steepness of the switching-probability curve (matches the C++
        constant 6.0 in ProcessOneNeighborMessage / ProcessNeighborMessagesMajority).

    Returns (new_opinion, new_q_est).
    """
    rng = rng or random
    other_q = max(0.0, min(1.0, neighbor_quality))

    if opinion < 0:
        if rng.uniform(0.0, 1.0) < other_q:
            return neighbor_opinion, other_q
        return opinion, q_est

    if other_q <= max(0.0, min(1.0, q_est)):
        return opinion, q_est

    p_switch = 1.0 - math.exp(-k * (other_q - q_est))
    if rng.uniform(0.0, 1.0) < p_switch:
        return neighbor_opinion, other_q

    return opinion, q_est
