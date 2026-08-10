"""
Cross-inhibition social behaviour, ported from
CThymioBestOfTwo::ProcessNeighborMessagesCrossInhibition_Standard
(control_variant="cross_inhibition").

Unlike the baseline/voter, majority-vote, and active-inference variants,
cross-inhibition has no quality-estimation model: a robot adopts whatever
option patch it happens to be standing on as its opinion directly (no
noise, no NoisyMeasure), then reacts to one neighbour message per tick:

  - no opinion yet -> RECRUIT toward the neighbour's opinion with
    probability kappa_recruit * qj
  - opinion differs from the neighbour's -> INHIBIT (drop the opinion,
    i.e. go back to undecided) with probability kappa_inhib * qj

qj is the neighbour's broadcast quality byte, floored at 0.05. Faithfully
matching the ARGoS controller: because cross-inhibition never calls
NoisyMeasure, BroadcastOpinionAndEstimate always sends quality=0.0 for
this variant (m_fQEst is never populated), so qj is always exactly the
0.05 floor in practice - kappa_recruit/kappa_inhib end up acting as flat
per-message probabilities (0.05 * kappa_*) rather than genuinely
quality-weighted ones. This is a quirk of the original C++ (an unused
quality channel), preserved here rather than "fixed", to keep behaviour
identical to argos_code.
"""

import random


def process_neighbor_message(opinion, num_options, other_opinion, other_quality,
                              kappa_recruit=0.4, kappa_inhib=1.0, rng=None):
    """
    opinion: this robot's current opinion (-1 if none).
    other_opinion / other_quality: decoded from the received message.

    Returns the new opinion.
    """
    rng = rng or random
    if other_opinion < 0 or other_opinion >= num_options:
        return opinion

    qj = max(0.05, min(1.0, other_quality))

    if opinion < 0:
        p = max(0.0, min(1.0, kappa_recruit * qj))
        if rng.uniform(0.0, 1.0) < p:
            return other_opinion
        return opinion

    if other_opinion != opinion:
        p = max(0.0, min(1.0, kappa_inhib * qj))
        if rng.uniform(0.0, 1.0) < p:
            return -1

    return opinion
