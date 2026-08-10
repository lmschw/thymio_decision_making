"""
Quality-reversal ("swap") environment perturbation, ported from the
"Quality reversal" block in
CThymioBestOfTwoLoopFunctions::PostStep (argos_code/for logging and
setting the env/thymio_best_of_two_loop_functions.cpp).

In ARGoS this lives in the loop functions - a single process external to
the per-robot controllers that mutates every robot's option_qualities in
lockstep at a fixed simulation tick. There is no equivalent shared process
here: each robot runs its own experiment instance, so this applies the
same swap_tick / gradual_reversal_ticks schedule independently on each
robot's own tick counter. As long as every robot in a run is started with
the same config, they all perform the same mutation at (approximately)
the same tick.

Swaps option 0 and option 1's qualities - either abruptly at `swap_tick`
(gradual_reversal_ticks=0, matching SwapOptionQualities01) or linearly
interpolated over [swap_tick, swap_tick + gradual_reversal_ticks)
(matching the SetOptionQuality interpolation loop) - leaving every other
option's quality untouched. `env_state` is derived purely from the tick
count relative to the schedule, matching PostStep's env_state:

    0 = pre-change
    1 = during a gradual transition
    2 = post-change (abrupt swap, or after a gradual transition completes)

swap_tick=0 (the default) disables the mechanism entirely, matching
ARGoS's m_unSwapTick(0) default.
"""


class QualitySwitch:

    def __init__(self, swap_tick=0, gradual_reversal_ticks=0):
        self.swap_tick = swap_tick
        self.gradual_reversal_ticks = gradual_reversal_ticks

        self.swap_done = False
        self.pre_swap_qualities = None
        self.post_swap_qualities = None

    def env_state(self, tick):
        """0 = pre-change, 1 = during transition, 2 = post-change."""
        if self.swap_tick <= 0:
            return 0
        if self.gradual_reversal_ticks > 0:
            if self.swap_tick <= tick < self.swap_tick + self.gradual_reversal_ticks:
                return 1
            if tick >= self.swap_tick + self.gradual_reversal_ticks:
                return 2
            return 0
        return 2 if tick >= self.swap_tick else 0

    def apply(self, tick, option_qualities):
        """
        Mutates `option_qualities` IN PLACE if this tick calls for it.
        Must be called once per tick, every tick, with tick counts
        increasing by 1 each call (mirrors the experiment's tick_count),
        so the tick immediately before swap_tick can be caught for the
        gradual case exactly like PostStep does.
        """
        if self.swap_tick <= 0 or self.swap_done:
            return

        if (self.gradual_reversal_ticks > 0
                and tick == self.swap_tick - 1
                and self.pre_swap_qualities is None):
            self.pre_swap_qualities = list(option_qualities)
            self.post_swap_qualities = list(option_qualities)
            if len(self.post_swap_qualities) >= 2:
                self.post_swap_qualities[0], self.post_swap_qualities[1] = (
                    self.post_swap_qualities[1], self.post_swap_qualities[0])
            return

        if tick == self.swap_tick and self.gradual_reversal_ticks == 0:
            if len(option_qualities) >= 2:
                option_qualities[0], option_qualities[1] = (
                    option_qualities[1], option_qualities[0])
            self.swap_done = True
            return

        if (self.gradual_reversal_ticks > 0
                and self.pre_swap_qualities is not None
                and self.swap_tick <= tick < self.swap_tick + self.gradual_reversal_ticks):
            progress = (tick - self.swap_tick + 1) / self.gradual_reversal_ticks
            for k in range(len(option_qualities)):
                option_qualities[k] = (
                    self.pre_swap_qualities[k]
                    + progress * (self.post_swap_qualities[k] - self.pre_swap_qualities[k])
                )
            if tick == self.swap_tick + self.gradual_reversal_ticks - 1:
                self.swap_done = True
