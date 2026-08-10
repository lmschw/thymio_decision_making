from experiments.decision_making.majority_voting import MajorityVotingBaselineExperiment


class MajorityVotingNoisyQualitySwitchExperiment(MajorityVotingBaselineExperiment):
    """
    Combines MajorityVotingNoisyExperiment and
    MajorityVotingQualitySwitchExperiment: defaults to both the elevated
    observation noise (noise_sigma=0.2 vs the 0.05 baseline default) and
    the quality-reversal environment perturbation (option 0 and option 1
    swap abruptly at tick 1200), matching the combined "noisy dynamic"
    condition of the ARGoS noise_sigma / swap_tick sweeps.
    """

    DEFAULT_NOISE_SIGMA = 0.2
    DEFAULT_SWAP_TICK = 1200
    DEFAULT_GRADUAL_REVERSAL_TICKS = 0

    def __init__(self, robot, config=None, logger=None):
        config = dict(config or {})
        config.setdefault("noise_sigma", self.DEFAULT_NOISE_SIGMA)
        config.setdefault("swap_tick", self.DEFAULT_SWAP_TICK)
        config.setdefault("gradual_reversal_ticks", self.DEFAULT_GRADUAL_REVERSAL_TICKS)
        super().__init__(robot, config=config, logger=logger)
