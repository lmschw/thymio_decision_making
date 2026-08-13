def clamp01(x):
    return max(0.0, min(1.0, x))


# Option identity, derived from the ground-patch colours each option is
# recognised by (see OptionGroundSensor / OPINION_COLORS in each
# experiment module) - option k's name is always this fixed colour,
# regardless of which quality value is currently assigned to it (a
# quality-switch swaps qualities between options, not their colours).
OPTION_NAMES = ["red", "green", "blue", "yellow"]


def pad_to_length(values, length):
    """Right-pads a list with "" so per-option CSV columns (e.g.
    option_name_0..3, option_quality_0..3, mu_0..3) always have a value
    to log even when num_options < length."""
    values = list(values)[:length]
    return values + [""] * (length - len(values))


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


