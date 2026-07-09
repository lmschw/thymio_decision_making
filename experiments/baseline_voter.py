import asyncio

from behaviours.obstacle_avoidance import ObstacleAvoidance
from behaviours.active_inference.option_ground_sensor import OptionGroundSensor
from behaviours.baseline.voter_model import noisy_measure, process_one_neighbor_message
from behaviours.active_inference.comm_protocol import encode_message, decode_message


OPINION_COLORS = [
    (32, 0, 0),   # option 0 -> red
    (0, 32, 0),   # option 1 -> green
    (0, 0, 32),   # option 2 -> blue
    (32, 32, 0),  # option 3 -> yellow
]


class BaselineVoterExperiment:
    """
    Best-of-N collective decision making with the "baseline" timer-driven
    explore/disseminate cycle and voter-model social influence.

    Port of CThymioBestOfTwo::ControlStep_Baseline
    (control_variant="baseline", social_model="voter") onto the async Robot
    API, in the same experiment-class shape as ColourRecognitionExperiment /
    ActiveInferenceExperiment.

    EXPLORE phase (timer-only phase switch):
      - drive around, sample ground patches, and build a quality estimate
        for whichever option patch the robot is currently standing on
      - `expl_timer` counts consecutive ticks the robot has held an
        opinion; once it reaches `expl_max_ticks` the robot switches to
        disseminating. The quality estimate itself never triggers the
        switch - only the timer does (this mirrors the ARGoS controller
        exactly: UpdateEstimateFromGround()'s return value is ignored for
        phase-switch purposes).

    DISSEMINATE phase (timer-driven duration, scaled by quality):
      - broadcast (opinion, quality) every tick
      - listen for one neighbour message per tick and apply the voter
        model (probabilistic switch toward higher-quality opinions)
      - `dissem_timer`, initialised to tau0 + floor(tau_gain * quality),
        counts down to 0, after which the robot returns to exploring - so
        a robot with a higher-quality opinion broadcasts for longer.
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

        self.noise_sigma = self.config.get("noise_sigma", 0.05)
        self.voter_k = self.config.get("voter_k", 6.0)

        # timer params (mirrors expl_max_ticks / tau0 / tau_gain)
        self.expl_max_ticks = self.config.get("expl_max_ticks", 200)
        self.tau0 = self.config.get("tau0", 30)
        self.tau_gain = self.config.get("tau_gain", 100)

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

        # --- opinion state ---
        self.opinion = -1
        self.q_est = 0.0

        # --- phase state ---
        self.disseminating = False
        self.expl_timer = 0
        self.dissem_timer = 0

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
                await self.robot.stop()
                if self.logger:
                    self.logger.log(
                        state={"error": repr(exc)},
                        command={"left_motor": 0, "right_motor": 0},
                    )

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def _tick(self):
        prox = await self.robot.proximity_horizontal()
        reflected = await self.robot.proximity_ground_reflected()

        if not self.disseminating:
            # --- EXPLORE ---
            if self.opinion >= 0:
                self.expl_timer += 1
            else:
                self.expl_timer = 0

            self._update_estimate_from_ground(reflected)

            # Timer-only trigger: the quality estimate above never flips
            # the phase itself - only expl_timer reaching expl_max_ticks does.
            if self.opinion >= 0 and self.expl_timer >= self.expl_max_ticks:
                q = max(0.0, min(1.0, self.q_est))
                self.dissem_timer = self.tau0 + int(self.tau_gain * q)
                self.disseminating = True
                self.expl_timer = 0
        else:
            # --- DISSEMINATE ---
            if self.opinion >= 0:
                # confidence fixed at 1.0: the baseline/voter model doesn't
                # use a confidence-weighted update like the AIF variant does.
                await self.robot.send(
                    encode_message(self.opinion, self.q_est, 1.0))

            incoming = None
            try:
                incoming = await self.robot.receive()
            except (TypeError, ValueError):
                # No message present yet - treat as "nothing received".
                incoming = None
            if incoming is not None:
                other_op, other_q, _other_conf = decode_message(incoming)
                self.opinion, self.q_est = process_one_neighbor_message(
                    self.opinion, self.q_est, other_op, other_q,
                    k=self.voter_k)

            if self.dissem_timer > 0:
                self.dissem_timer -= 1
            if self.dissem_timer == 0:
                self.disseminating = False

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
            self.logger.log(
                state={
                    "proximity": prox,
                    "reflected_0": reflected[0] if len(reflected) > 0 else None,
                    "reflected_1": reflected[1] if len(reflected) > 1 else None,
                    "opinion": self.opinion,
                    "q_est": self.q_est,
                    "disseminating": self.disseminating,
                    "expl_timer": self.expl_timer,
                    "dissem_timer": self.dissem_timer,
                },
                command={
                    "left_motor": left,
                    "right_motor": right,
                    "led": (r, g, b),
                },
            )

    def _update_estimate_from_ground(self, reflected):
        """
        Port of UpdateEstimateFromGround: refreshes q_est whenever the
        robot is on the patch matching its current opinion, or adopts a
        first opinion if it doesn't have one yet. Returns True if q_est
        was updated (this return value is deliberately NOT used to trigger
        the phase switch, matching the ARGoS controller).
        """
        opt_idx, _avg = self.ground_sensor.detect_option(reflected)
        if opt_idx < 0 or opt_idx >= len(self.option_qualities):
            return False
        q = noisy_measure(self.option_qualities[opt_idx], self.noise_sigma)
        if self.opinion < 0:
            self.opinion = opt_idx
            self.q_est = q
            return True
        if opt_idx == self.opinion:
            self.q_est = q
            return True
        return False

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
