from experiments.decision_making.majority_voting import MajorityVotingBaselineExperiment


class MajorityVotingQualitySwitchExperiment(MajorityVotingBaselineExperiment):
    """
    Same as MajorityVotingBaselineExperiment, but with the quality-reversal
    environment perturbation enabled by default: option 0 and option 1's
    qualities swap abruptly at tick 1200 (~60s at this platform's ~20-tick/s
    loop rate), matching the swap_tick sweep on ARGoS's
    bestofN_dynamic_template.argos (gradual_reversal_ticks=0, abrupt swap).
    """

    DEFAULT_SWAP_TICK = 600
    DEFAULT_GRADUAL_REVERSAL_TICKS = 0

    def __init__(self, robot, config=None, logger=None):
        config = dict(config or {})
        config.setdefault("swap_tick", self.DEFAULT_SWAP_TICK)
        config.setdefault("gradual_reversal_ticks", self.DEFAULT_GRADUAL_REVERSAL_TICKS)
        super().__init__(robot, config=config, logger=logger)
