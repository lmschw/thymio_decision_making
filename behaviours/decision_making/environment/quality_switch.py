"""
Quality-reversal ("swap") environment perturbation, ported from the
"Quality reversal" block in
CThymioBestOfTwoLoopFunctions::PostStep (argos_code/for logging and
setting the env/thymio_best_of_two_loop_functions.cpp).

In ARGoS this lives in the loop functions - a single process external to
the per-robot controllers that mutates every robot's option_qualities in
lockstep at a fixed simulation tick. There is no equivalent shared process
here: each robot runs its own experiment instance, so this applies the
same swap_seconds / gradual_reversal_seconds schedule independently on
each robot's own elapsed wall-clock time. Wall-clock rather than tick
count, for the same reason `duration_seconds` is: tick_count is a local,
unsynchronized per-robot counter (variable sensor/comms latency each
iteration), so two robots at the "same tick" can be at meaningfully
different real times - which would mean the environment change doesn't
actually land at the same moment for every robot. As long as every robot
in a run is started at (approximately) the same moment with the same
config, wall-clock elapsed time keeps them aligned.

Swaps option 0 and option 1's qualities - either abruptly at
`swap_seconds` (gradual_reversal_seconds=0, matching
SwapOptionQualities01) or linearly interpolated over
[swap_seconds, swap_seconds + gradual_reversal_seconds) (matching the
SetOptionQuality interpolation loop) - leaving every other option's
quality untouched. `env_state` is derived purely from elapsed time
relative to the schedule, matching PostStep's env_state:

    0 = pre-change
    1 = during a gradual transition
    2 = post-change (abrupt swap, or after a gradual transition completes)

swap_seconds=0 (the default) disables the mechanism entirely, matching
ARGoS's m_unSwapTick(0) default.
"""


class QualitySwitch:

    def __init__(self, swap_seconds=0, gradual_reversal_seconds=0):
        self.swap_seconds = swap_seconds
        self.gradual_reversal_seconds = gradual_reversal_seconds

        self.swap_done = False
        self.pre_swap_qualities = None
        self.post_swap_qualities = None

    def env_state(self, elapsed_seconds):
        """0 = pre-change, 1 = during transition, 2 = post-change."""
        if self.swap_seconds <= 0:
            return 0
        if self.gradual_reversal_seconds > 0:
            if self.swap_seconds <= elapsed_seconds < self.swap_seconds + self.gradual_reversal_seconds:
                return 1
            if elapsed_seconds >= self.swap_seconds + self.gradual_reversal_seconds:
                return 2
            return 0
        return 2 if elapsed_seconds >= self.swap_seconds else 0

    def apply(self, elapsed_seconds, option_qualities):
        """
        Mutates `option_qualities` IN PLACE if the elapsed time calls for
        it. Must be called once per tick, every tick, with a monotonically
        increasing `elapsed_seconds` (seconds since the experiment
        started - see the `wall_time`/`start_time` bookkeeping in each
        experiment's _tick()).
        """
        if self.swap_seconds <= 0 or self.swap_done:
            return

        if self.gradual_reversal_seconds > 0 and self.pre_swap_qualities is None:
            # Snapshot the pre-swap baseline the first time we're called -
            # qualities never change before swap_seconds anyway, so any
            # call before the swap starts gives the correct baseline.
            self.pre_swap_qualities = list(option_qualities)
            self.post_swap_qualities = list(option_qualities)
            if len(self.post_swap_qualities) >= 2:
                self.post_swap_qualities[0], self.post_swap_qualities[1] = (
                    self.post_swap_qualities[1], self.post_swap_qualities[0])

        if elapsed_seconds < self.swap_seconds:
            return

        if self.gradual_reversal_seconds <= 0:
            if len(option_qualities) >= 2:
                option_qualities[0], option_qualities[1] = (
                    option_qualities[1], option_qualities[0])
            self.swap_done = True
            return

        progress = min(
            1.0,
            (elapsed_seconds - self.swap_seconds) / self.gradual_reversal_seconds,
        )
        for k in range(len(option_qualities)):
            option_qualities[k] = (
                self.pre_swap_qualities[k]
                + progress * (self.post_swap_qualities[k] - self.pre_swap_qualities[k])
            )
        if progress >= 1.0:
            self.swap_done = True
