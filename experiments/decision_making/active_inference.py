import asyncio

from behaviours.base_behaviours.obstacle_avoidance import ObstacleAvoidance
from behaviours.base_behaviours.colour_recognition import OptionGroundSensor
from behaviours.decision_making.active_inference.active_inference_beliefs import ActiveInferenceBeliefs
from behaviours.decision_making.active_inference.efe_policy import EFEPolicy
from behaviours.decision_making.environment.quality_switch import QualitySwitch
from utils.communication import encode_message, decode_message
from utils.utils import true_best_option


OPINION_COLORS = [
    (32, 0, 0),   # option 0 -> red
    (0, 32, 0),   # option 1 -> green
    (0, 0, 32),   # option 2 -> blue
    (32, 32, 0),  # option 3 -> yellow
]


def _pad(values, length):
    """Right-pads a list with "" so mu_0..3/tau_0..3/pb_0..3 always have
    a value to log even when num_options < 4."""
    values = list(values)[:length]
    return values + [""] * (length - len(values))


class ActiveInferenceBaselineExperiment:
    """
    Best-of-N collective decision making driven by an Active-Inference
    (Expected Free Energy) explore/disseminate policy.

    Port of CThymioBestOfTwo's "active_inference" control_variant
    (ControlStep_ActiveInference) onto the async Robot API, in the same
    experiment-class shape as ColourRecognitionExperiment.

    Each tick, while EXPLORING:
      - drives around via obstacle avoidance
      - reads the ground sensor; if on an option patch, updates beliefs
        about that option's quality (private evidence)

    Each tick, while DISSEMINATING:
      - broadcasts (opinion, quality, confidence) over prox.comm
      - updates beliefs from anything received (social evidence)

    Every `decide_every` ticks (once at least `min_dwell` ticks have been
    spent in the current phase), the EFE policy is re-evaluated to decide
    whether to switch phase.

    config keys (all optional, defaults mirror the ARGoS controller):
      num_options, option_qualities,
      efe_gamma, noise_sigma, c_expected, precision_decay, prior_var,
      min_dwell, decide_every,
      option_centers, allowed_offset,
      delta, wheel_velocity, turn_steps

    Optionally, `swap_tick` / `gradual_reversal_ticks` config keys enable
    the same quality-reversal environment perturbation as ARGoS's loop
    functions (see behaviours.decision_making.environment.quality_switch)
    - disabled by default (swap_tick=0).
    """

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

        # --- identity, for the shared CSV log ---
        self.robot_id = self.config.get("robot_id", "")
        # recomputed every tick from option_qualities - see _tick()
        self.true_best = -1

        # --- quality-reversal environment perturbation (off by default) ---
        self.quality_switch = QualitySwitch(
            swap_tick=self.config.get("swap_tick", 0),
            gradual_reversal_ticks=self.config.get("gradual_reversal_ticks", 0),
        )
        self.env_state = 0

        # --- motion params ---
        self.delta = self.config.get("delta", 1000)
        self.wheel_velocity = self.config.get("wheel_velocity", 300)
        self.obstacle_avoidance = ObstacleAvoidance(
            wheel_velocity=self.wheel_velocity,
            delta=self.delta,
        )

        # --- decision-making params ---
        self.num_options = self.config.get("num_options", 3)

        option_qualities = self.config.get("option_qualities")
        if option_qualities is None:
            option_qualities = [max(0.1, 1.0 - 0.4 * i) for i in range(self.num_options)]
        if len(option_qualities) != self.num_options:
            raise ValueError("option_qualities length must match num_options")
        self.option_qualities = option_qualities
        self.true_best = true_best_option(self.option_qualities)

        self.min_dwell = self.config.get("min_dwell", 30)
        self.decide_every = self.config.get("decide_every", 5)

        self.ground_sensor = OptionGroundSensor(
            num_options=self.num_options,
            option_centers=self.config.get("option_centers"),
            allowed_offset=self.config.get("allowed_offset", 50),
        )

        self.beliefs = ActiveInferenceBeliefs(
            num_options=self.num_options,
            option_qualities=self.option_qualities,
            noise_sigma=self.config.get("noise_sigma", 0.05),
            prior_var=self.config.get("prior_var", 0.25),
            precision_decay=self.config.get("precision_decay", 0.995),
        )

        self.policy = EFEPolicy(
            gamma=self.config.get("efe_gamma", 20.0),
            c_expected=self.config.get("c_expected", 0.5),
            noise_sigma=self.config.get("noise_sigma", 0.05),
        )

        # --- phase state ---
        self.disseminating = False
        self.phase_ticks = 0
        self.since_decision = 0
        self.opinion = -1

        # --- bookkeeping for the shared CSV log ---
        self.tick_count = 0
        self.msgs_tx_total = 0
        self.msgs_rx_total = 0
        self.explore_total = 0
        self.exploit_total = 0
        self.last_explore_bout = 0
        self.last_exploit_bout = 0

    async def run(self):

        while self.running:

            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            try:
                await self._tick()
            except Exception as exc:
                # Never let a single bad tick (e.g. a comms read failing
                # because no message has arrived yet) kill the loop and
                # leave the last drive() command latched on the motors.
                # Stop, log, and keep going.
                await self.robot.stop()
                print(f"[ActiveInferenceExperiment] tick error, motors stopped: {exc!r}")

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def _tick(self):
            self.tick_count += 1

            # Apply the quality-reversal schedule (no-op unless swap_tick is
            # configured) and refresh env_state/true_best for this tick.
            # Mutates option_qualities in place, so self.beliefs (which
            # holds the same list reference) sees the update too, and
            # true_best tracks it live, matching GetTrueBestOption() being
            # recomputed fresh every tick in the ARGoS controller.
            self.quality_switch.apply(self.tick_count, self.option_qualities)
            self.env_state = self.quality_switch.env_state(self.tick_count)
            self.true_best = true_best_option(self.option_qualities)

            prox = await self.robot.proximity_horizontal()
            reflected = await self.robot.proximity_ground_reflected()

            # Detected patch, for logging, regardless of phase.
            opt_idx, _avg_ground = self.ground_sensor.detect_option(reflected)

            msgs_tx_tick = 0
            msgs_rx_tick = 0

            # --- belief update ---
            if not self.disseminating:
                sampled = self.beliefs.update_from_ground(opt_idx)
                if sampled and self.opinion < 0:
                    self.opinion = opt_idx
            else:
                incoming = None
                try:
                    incoming = await self.robot.receive()
                except (TypeError, ValueError):
                    # No message present yet (prox.comm.rx not populated) -
                    # treat as "nothing received" rather than crashing.
                    incoming = None
                if incoming is not None:
                    msgs_rx_tick = 1
                    self.msgs_rx_total += 1
                    op, q_msg, c_msg = decode_message(incoming)
                    self.beliefs.update_from_message(op, q_msg, c_msg)

            self.beliefs.decay_precision()
            self.beliefs.recompute_belief_best()

            if self.beliefs.has_sampled_any:
                self.opinion = self.beliefs.map_best_tiebreak_random()

            # --- EFE policy: decide whether to switch phase ---
            self.phase_ticks += 1
            self.since_decision += 1
            can_switch = self.phase_ticks >= self.min_dwell
            want_dissem = self.disseminating

            if (can_switch and self.beliefs.has_sampled_any
                    and self.since_decision >= max(1, self.decide_every)):
                want_dissem = self.policy.select_disseminate(self.beliefs)
                self.since_decision = 0

            if not self.beliefs.has_sampled_any:
                want_dissem = False

            if can_switch and want_dissem != self.disseminating:
                if self.disseminating:
                    self.last_exploit_bout = self.phase_ticks
                else:
                    self.last_explore_bout = self.phase_ticks
                self.disseminating = want_dissem
                self.phase_ticks = 0

            if self.disseminating:
                self.exploit_total += 1
            else:
                self.explore_total += 1

            # --- communicate ---
            if self.disseminating and self.opinion >= 0:
                quality = self.beliefs.mu_q[self.opinion]
                confidence = max(0.05, self.beliefs.epistemic_confidence())
                await self.robot.send(
                    encode_message(self.opinion, quality, confidence))
                msgs_tx_tick = 1
                self.msgs_tx_total += 1

            # --- motion ---
            left, right = self.obstacle_avoidance.step_motion(prox)
            await self.robot.drive(left, right)

            # --- LEDs: colour = current opinion ---
            if 0 <= self.opinion < len(OPINION_COLORS):
                r, g, b = OPINION_COLORS[self.opinion]
            else:
                r, g, b = (0, 0, 0)
            await self.robot.top_led(r, g, b)

            if self.logger:
                mu = _pad(self.beliefs.mu_q, 4)
                tau = _pad(self.beliefs.tau_q, 4)
                pb = _pad(self.beliefs.belief_best, 4)
                correct = ("" if self.true_best is None
                           else int(self.opinion == self.true_best))
                try:
                    self.logger.log(
                        state={
                            "tick": self.tick_count,
                            "ctrl_variant": "active_inference",
                            "robot_id": self.robot_id,
                            "patch": opt_idx,
                            "q_est": round(self.beliefs.expected_quality(), 6),
                            "opinion": self.opinion,
                            "flag": 1 if self.disseminating else 0,
                            "explore_total": self.explore_total,
                            "exploit_total": self.exploit_total,
                            "last_explore_bout": self.last_explore_bout,
                            "last_exploit_bout": self.last_exploit_bout,
                            "ticks_in_phase": self.phase_ticks,
                            "msgs_tx_tick": msgs_tx_tick,
                            "msgs_rx_tick": msgs_rx_tick,
                            "msgs_tx_total": self.msgs_tx_total,
                            "msgs_rx_total": self.msgs_rx_total,
                            "env_state": self.env_state,
                            "belief_entropy": round(self.beliefs.belief_entropy01(), 6),
                            "epistemic_confidence": round(self.beliefs.epistemic_confidence(), 6),
                            "p_dissem": self.policy.last["p_dissem"],
                            "g_explore": self.policy.last["g_explore"],
                            "g_dissem": self.policy.last["g_dissem"],
                            "efe_margin": self.policy.last["g_dissem"] - self.policy.last["g_explore"],
                            "ig_explore": self.policy.last["ig_explore"],
                            "ig_dissem": self.policy.last["ig_dissem"],
                            "pragmatic": self.policy.last["pragmatic"],
                            "mu_0": mu[0], "mu_1": mu[1], "mu_2": mu[2], "mu_3": mu[3],
                            "tau_0": tau[0], "tau_1": tau[1], "tau_2": tau[2], "tau_3": tau[3],
                            "pb_0": pb[0], "pb_1": pb[1], "pb_2": pb[2], "pb_3": pb[3],
                            "true_best": self.true_best,
                            "correct": correct,
                        },
                        command={
                            "left_motor": left,
                            "right_motor": right,
                            "led_r": r, "led_g": g, "led_b": b,
                        },
                    )
                except Exception as log_exc:
                    # A logging failure must NEVER stop the robot. Print and
                    # move on - motion for this tick already happened above.
                    print(f"[ActiveInferenceExperiment] logging failed "
                          f"(motors unaffected): {log_exc!r}")

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
