import asyncio

from behaviours.obstacle_avoidance import ObstacleAvoidance
from behaviours.active_inference.option_ground_sensor import OptionGroundSensor
from behaviours.active_inference.active_inference_beliefs import ActiveInferenceBeliefs
from behaviours.active_inference.efe_policy import EFEPolicy
from behaviours.active_inference.comm_protocol import encode_message, decode_message


OPINION_COLORS = [
    (32, 0, 0),   # option 0 -> red
    (0, 32, 0),   # option 1 -> green
    (0, 0, 32),   # option 2 -> blue
    (32, 32, 0),  # option 3 -> yellow
]


class ActiveInferenceExperiment:
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
      ground_max, white_thr, color_eps,
      red_level, green_level, blue_level, yellow_level,
      delta, wheel_velocity, turn_steps
    """

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

        # --- motion params ---
        self.delta = self.config.get("delta", 1000)
        self.wheel_velocity = self.config.get("wheel_velocity", 300)
        self.turn_steps = self.config.get("turn_steps", 8)
        self.obstacle_avoidance = ObstacleAvoidance(
            wheel_velocity=self.wheel_velocity,
            delta=self.delta,
            turn_steps=self.turn_steps,
        )

        # --- decision-making params ---
        self.num_options = self.config.get("num_options", 2)

        option_qualities = self.config.get("option_qualities")
        if option_qualities is None:
            option_qualities = [max(0.1, 1.0 - 0.4 * i) for i in range(self.num_options)]
        if len(option_qualities) != self.num_options:
            raise ValueError("option_qualities length must match num_options")
        self.option_qualities = option_qualities

        self.min_dwell = self.config.get("min_dwell", 30)
        self.decide_every = self.config.get("decide_every", 5)

        self.ground_sensor = OptionGroundSensor(
            num_options=self.num_options,
            ground_max=self.config.get("ground_max", 1023),
            white_thr=self.config.get("white_thr", 0.95),
            color_eps=self.config.get("color_eps", 0.06),
            red_level=self.config.get("red_level", 0.30),
            green_level=self.config.get("green_level", 0.60),
            blue_level=self.config.get("blue_level", 0.12),
            yellow_level=self.config.get("yellow_level", 0.75),
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

    async def run(self):

        while self.running:

            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            prox = await self.robot.proximity_horizontal()
            reflected = await self.robot.proximity_ground_reflected()

            # --- belief update ---
            if not self.disseminating:
                opt_idx, _avg_ground = self.ground_sensor.detect_option(reflected)
                sampled = self.beliefs.update_from_ground(opt_idx)
                if sampled and self.opinion < 0:
                    self.opinion = opt_idx
            else:
                incoming = await self.robot.receive()
                if incoming is not None:
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
                self.disseminating = want_dissem
                self.phase_ticks = 0

            # # --- communicate ---
            # if self.disseminating and self.opinion >= 0:
            #     quality = self.beliefs.mu_q[self.opinion]
            #     confidence = max(0.05, self.beliefs.epistemic_confidence())
            #     await self.robot.send(
            #         encode_message(self.opinion, quality, confidence))

            # --- motion ---
            left, right = self.obstacle_avoidance.step_motion(prox)
            await self.robot.drive(left, right)

            # # --- LEDs: colour = current opinion ---
            # if 0 <= self.opinion < len(OPINION_COLORS):
            #     r, g, b = OPINION_COLORS[self.opinion]
            # else:
            #     r, g, b = (0, 0, 0)
            # await self.robot.top_led(r, g, b)

            # if self.logger:
            #     self.logger.log(
            #         state={
            #             "proximity": prox,
            #             "reflected_0": reflected[0] if len(reflected) > 0 else None,
            #             "reflected_1": reflected[1] if len(reflected) > 1 else None,
            #             "opinion": self.opinion,
            #             "disseminating": self.disseminating,
            #             "mu_q": list(self.beliefs.mu_q),
            #             "tau_q": list(self.beliefs.tau_q),
            #             "belief_best": list(self.beliefs.belief_best),
            #             "expected_quality": self.beliefs.expected_quality(),
            #             "p_dissem": self.policy.last["p_dissem"],
            #             "g_explore": self.policy.last["g_explore"],
            #             "g_dissem": self.policy.last["g_dissem"],
            #             "ig_explore": self.policy.last["ig_explore"],
            #             "ig_dissem": self.policy.last["ig_dissem"],
            #             "pragmatic": self.policy.last["pragmatic"],
            #         },
            #         command={
            #             "left_motor": left,
            #             "right_motor": right,
            #             "led": (r, g, b),
            #         },
            #     )

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
