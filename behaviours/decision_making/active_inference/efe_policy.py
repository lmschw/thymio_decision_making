import math
import random


class EFEPolicy:
    """
    Expected Free Energy policy selection between "explore" (visit ground
    patches, gather private evidence) and "disseminate" (broadcast opinion,
    gather social evidence).

    Direct port of CThymioBestOfTwo::EFE_SelectDisseminate.

        G(explore) = -IG_explore
        G(dissem)  = -IG_dissem - pragmatic

        P(dissem) = softmax_gamma(-G_dissem, -G_explore)

    gamma controls how deterministic the policy is; it does NOT set the
    crossover point, which is determined by IG_explore, IG_dissem and
    pragmatic (all derived from the belief state).
    """

    def __init__(self, gamma=20.0, c_expected=0.5, noise_sigma=0.05, rng=None):
        self.gamma = gamma
        self.c_expected = c_expected
        self.noise_sigma = noise_sigma
        self.rng = rng or random.Random()

        # last-computed diagnostics, exposed for logging
        self.last = {
            "p_dissem": 0.5,
            "g_explore": 0.0,
            "g_dissem": 0.0,
            "ig_explore": 0.0,
            "ig_dissem": 0.0,
            "pragmatic": 0.0,
        }

    def select_disseminate(self, beliefs):
        """
        beliefs: an ActiveInferenceBeliefs instance (already updated and with
        recompute_belief_best() called this tick).
        Returns True if the robot should (start/keep) disseminating.
        """
        sig = max(0.1, self.noise_sigma)
        tau_soc = 1.0 / (sig * sig)

        # IG(explore): posterior std of the option this robot has personally
        # sampled most confidently (highest tau), scaled by alpha = 20.0.
        # See CThymioBestOfTwo::EFE_SelectDisseminate for the derivation /
        # calibration of alpha against exploration-time ratios.
        tau_map = max(1e-9, max(beliefs.tau_q))
        ig_explore = 20.0 * math.sqrt(1.0 / tau_map)

        # IG(dissem): expected information gain from one incoming message,
        # weighted by current belief over which option is best.
        ig_dissem = 0.0
        for k in range(beliefs.num_options):
            tau_k = max(1e-9, beliefs.tau_q[k])
            ig_dissem += beliefs.belief_best[k] * 0.5 * math.log(
                1.0 + self.c_expected * tau_soc / tau_k)

        # Pragmatic value: confidence above chance that current MAP is best.
        map_k = beliefs.map_best()
        p_map = beliefs.belief_best[map_k]
        pragmatic = max(0.0, p_map - 1.0 / beliefs.num_options)

        g_explore = -ig_explore
        g_dissem = -ig_dissem - pragmatic

        a_e = -self.gamma * g_explore
        a_d = -self.gamma * g_dissem
        mx = max(a_e, a_d)
        w_e = math.exp(a_e - mx)
        w_d = math.exp(a_d - mx)
        z = w_e + w_d
        p_dissem = (w_d / z) if z > 1e-12 else 0.5
        p_dissem = min(0.99, max(0.01, p_dissem))

        self.last = {
            "p_dissem": p_dissem,
            "g_explore": g_explore,
            "g_dissem": g_dissem,
            "ig_explore": ig_explore,
            "ig_dissem": ig_dissem,
            "pragmatic": pragmatic,
        }

        return self.rng.uniform(0.0, 1.0) < p_dissem
