from experiments.decision_making.cross_inhibition import CrossInhibitionBaselineExperiment


class CrossInhibitionQualitySwitchExperiment(CrossInhibitionBaselineExperiment):
    """
    Same as CrossInhibitionBaselineExperiment, but with the quality-reversal
    environment perturbation enabled by default: option 0 and option 1's
    qualities swap abruptly 60s into the run, matching the swap_tick sweep
    on ARGoS's bestofN_dynamic_template.argos (a 120s/1200-tick run,
    swapping around the midpoint; gradual_reversal_seconds=0, abrupt swap).

    Note this only affects true_best/env_state logging - cross-inhibition's
    own decision loop never reads option_qualities (see the base class
    docstring), so the swap has no effect on robot behaviour itself.
    """

    DEFAULT_SWAP_SECONDS = 60
    DEFAULT_GRADUAL_REVERSAL_SECONDS = 0

    def __init__(self, robot, config=None, logger=None):
        config = dict(config or {})
        config.setdefault("swap_seconds", self.DEFAULT_SWAP_SECONDS)
        config.setdefault("gradual_reversal_seconds", self.DEFAULT_GRADUAL_REVERSAL_SECONDS)
        super().__init__(robot, config=config, logger=logger)
