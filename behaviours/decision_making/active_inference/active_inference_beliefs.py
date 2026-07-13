import math
import random

from utils.utils import clamp01

class ActiveInferenceBeliefs:
    """
    Gaussian-conjugate belief tracker over option qualities Q_k.

    Port of the belief machinery in CThymioBestOfTwo:
      - BeliefUpdateFromGround   -> update_from_ground
      - BeliefUpdateFromMessages -> update_from_message
      - RecomputeBeliefBest      -> recompute_belief_best
      - MAPBest / MAPBest_TieBreakRandom
      - EpistemicConfidence

    mu_q[k]  : posterior mean estimate of option k's quality
    tau_q[k] : posterior precision (1 / variance) of that estimate
    belief_best[k]: adaptive-temperature softmax P(option k is the best one)
    """

    def __init__(self,
                 num_options=2,
                 option_qualities=None,
                 noise_sigma=0.05,
                 prior_var=0.25,
                 precision_decay=0.995,
                 rng=None):
        self.num_options = num_options
        self.option_qualities = option_qualities or [0.5] * num_options
        self.noise_sigma = noise_sigma
        self.prior_var = prior_var
        self.precision_decay = precision_decay
        self.rng = rng or random.Random()

        self.has_sampled_any = False
        self.reset()

    def reset(self):
        mu0 = 1.0 / self.num_options
        tau0 = 1.0 / max(1e-4, self.prior_var)
        self.mu_q = [mu0] * self.num_options
        self.tau_q = [tau0] * self.num_options
        self.belief_best = [mu0] * self.num_options
        self.has_sampled_any = False

    # ---- noise model ----
    def noisy_measure(self, q_true):
        """noise_sigma <= 0 -> noiseless (matches ARGoS controller default)."""
        if self.noise_sigma <= 0.0:
            return clamp01(q_true)
        return clamp01(q_true + self.rng.gauss(0.0, self.noise_sigma))

    # ---- private (ground) evidence ----
    def update_from_ground(self, option_idx):
        """
        option_idx: result of OptionGroundSensor.detect_option()[0];
        < 0 means "not currently on an option patch" -> no update.
        Returns True if a belief update was made.
        """
        if option_idx is None or option_idx < 0 or option_idx >= self.num_options:
            return False
        k = option_idx
        y = self.noisy_measure(self.option_qualities[k])
        sigma = self.noise_sigma if self.noise_sigma > 0.0 else 0.01
        tau_y = 1.0 / (sigma * sigma)
        tau1 = self.tau_q[k] + tau_y
        mu1 = clamp01((self.tau_q[k] * self.mu_q[k] + tau_y * y) / tau1)
        self.tau_q[k] = tau1
        self.mu_q[k] = mu1
        self.has_sampled_any = True
        return True

    # ---- social evidence ----
    def update_from_message(self, op, quality01, confidence01):
        """
        Treats a received (option, quality, confidence) message as a noisy
        observation of Q_op, weighted by the sender's confidence.
        No-op until this robot has gathered at least one private sample
        (matches the ARGoS controller: it ignores social evidence before
        it has any private evidence to reconcile it against).
        """
        if not self.has_sampled_any:
            return
        if op is None or op < 0 or op >= self.num_options:
            return
        q_msg = clamp01(quality01)
        c_msg = max(0.05, clamp01(confidence01))
        sigma = self.noise_sigma if self.noise_sigma > 0.0 else 0.01
        tau_base = 1.0 / (sigma * sigma)
        tau_y = c_msg * tau_base
        if tau_y <= 1e-9:
            return
        k = op
        tau1 = self.tau_q[k] + tau_y
        mu1 = clamp01((self.tau_q[k] * self.mu_q[k] + tau_y * q_msg) / tau1)
        self.tau_q[k] = tau1
        self.mu_q[k] = mu1

    def decay_precision(self):
        """
        Precision decays toward tau_floor = 1/prior_var each tick, so
        beliefs can loosen again if option qualities change (enables
        re-exploration). No-op when precision_decay >= 1.0.
        """
        if self.precision_decay >= 1.0:
            return
        tau_floor = 1.0 / max(1e-4, self.prior_var)
        for k in range(self.num_options):
            self.tau_q[k] = max(tau_floor, self.tau_q[k] * self.precision_decay)

    # ---- belief over best option ----
    @staticmethod
    def _entropy(p):
        h = 0.0
        for x in p:
            if x > 1e-12:
                h -= x * math.log(x)
        return h

    def belief_entropy01(self):
        h = self._entropy(self.belief_best)
        d = math.log(self.num_options)
        return clamp01(h / d) if d > 0.0 else 0.0

    def recompute_belief_best(self):
        """
        Adaptive-temperature softmax over posterior means: temperature
        falls as average posterior std falls, so tighter beliefs produce
        sharper P(best).
        """
        avg_std = sum(math.sqrt(1.0 / max(1e-9, t)) for t in self.tau_q) / self.num_options
        beta = max(0.5, 6.0 / (1.0 + 2.0 * avg_std))
        max_mu = max(self.mu_q)
        exps = [math.exp(beta * (mu - max_mu)) for mu in self.mu_q]
        z = sum(exps)
        if z <= 1e-12:
            u = 1.0 / self.num_options
            self.belief_best = [u] * self.num_options
            return
        self.belief_best = [e / z for e in exps]

    def map_best(self):
        best = 0
        for i in range(1, self.num_options):
            if self.belief_best[i] > self.belief_best[best]:
                best = i
        return best

    def map_best_tiebreak_random(self):
        mx = max(self.belief_best)
        ties = [k for k in range(self.num_options)
                if abs(self.belief_best[k] - mx) <= 1e-9]
        return self.rng.choice(ties)

    def epistemic_confidence(self):
        k = self.map_best()
        tau = max(1e-9, self.tau_q[k])
        return clamp01(1.0 / (1.0 + math.sqrt(1.0 / tau)))

    def expected_quality(self):
        return clamp01(sum(b * m for b, m in zip(self.belief_best, self.mu_q)))
