def clamp01(x):
    return max(0.0, min(1.0, x))


def true_best_option(option_qualities):
    """
    Index of the highest-quality option, ties broken toward the lowest
    index. Port of CThymioBestOfTwo::GetTrueBestOption, meant to be
    recomputed every tick (not cached) so it tracks quality-switch
    perturbations as they happen.
    """
    if not option_qualities:
        return -1
    best = 0
    for k in range(1, len(option_qualities)):
        if option_qualities[k] > option_qualities[best]:
            best = k
    return best


