from experiments.decision_making.majority_voting import MajorityVotingBaselineExperiment


class MajorityVotingNoisyExperiment(MajorityVotingBaselineExperiment):
    """
    Same as MajorityVotingBaselineExperiment, but defaults to a
    substantially higher observation noise (noise_sigma=0.2 vs the 0.05
    baseline default), matching the "noisy" condition of the ARGoS
    noise_sigma sweep (see thymio_best_of_two.h / bestofN_static_template.argos
    in argos_code/).
    """

    DEFAULT_NOISE_SIGMA = 0.2

    def __init__(self, robot, config=None, logger=None):
        config = dict(config or {})
        config.setdefault("noise_sigma", self.DEFAULT_NOISE_SIGMA)
        super().__init__(robot, config=config, logger=logger)
