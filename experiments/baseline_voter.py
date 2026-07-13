import asyncio

from behaviours.base_behaviours.obstacle_avoidance import ObstacleAvoidance
from behaviours.base_behaviours.option_ground_sensor import OptionGroundSensor
from behaviours.decision_making.baseline.voter_model import noisy_measure, process_one_neighbor_message
from utils.communication import encode_message, decode_message


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

        # --- identity, for the shared CSV log ---
        self.robot_id = self.config.get("robot_id", "")
        self.env_state = self.config.get("env_state")
        self.true_best = self.config.get("true_best")

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
        self.num_options = self.config.get("num_options", 3)

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
            option_centers=self.config.get("option_centers"),
            allowed_offset=self.config.get("allowed_offset", 50),
        )

        # --- opinion state ---
        self.opinion = -1
        self.q_est = 0.0

        # --- phase state ---
        self.disseminating = False
        self.expl_timer = 0
        self.dissem_timer = 0
        self.phase_ticks = 0  # generic ticks-in-current-phase, for logging/bout length

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
                await self.robot.stop()
                print(f"[BaselineVoterExperiment] tick error, motors stopped: {exc!r}")

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def _tick(self):
        self.tick_count += 1

        prox = await self.robot.proximity_horizontal()
        reflected = await self.robot.proximity_ground_reflected()

        # Detected patch, for logging, regardless of phase.
        opt_idx, _avg = self.ground_sensor.detect_option(reflected)

        msgs_tx_tick = 0
        msgs_rx_tick = 0
        self.phase_ticks += 1

        if not self.disseminating:
            # --- EXPLORE ---
            if self.opinion >= 0:
                self.expl_timer += 1
            else:
                self.expl_timer = 0

            self._update_estimate_from_ground(opt_idx)

            # Timer-only trigger: the quality estimate above never flips
            # the phase itself - only expl_timer reaching expl_max_ticks does.
            if self.opinion >= 0 and self.expl_timer >= self.expl_max_ticks:
                q = max(0.0, min(1.0, self.q_est))
                self.dissem_timer = self.tau0 + int(self.tau_gain * q)
                self.disseminating = True
                self.expl_timer = 0
                self.last_explore_bout = self.phase_ticks
                self.phase_ticks = 0
        else:
            # --- DISSEMINATE ---
            if self.opinion >= 0:
                # confidence fixed at 1.0: the baseline/voter model doesn't
                # use a confidence-weighted update like the AIF variant does.
                await self.robot.send(
                    encode_message(self.opinion, self.q_est, 1.0))
                msgs_tx_tick = 1
                self.msgs_tx_total += 1

            incoming = None
            try:
                incoming = await self.robot.receive()
            except (TypeError, ValueError):
                # No message present yet - treat as "nothing received".
                incoming = None
            if incoming is not None:
                msgs_rx_tick = 1
                self.msgs_rx_total += 1
                other_op, other_q, _other_conf = decode_message(incoming)
                self.opinion, self.q_est = process_one_neighbor_message(
                    self.opinion, self.q_est, other_op, other_q,
                    k=self.voter_k)

            if self.dissem_timer > 0:
                self.dissem_timer -= 1
            if self.dissem_timer == 0:
                self.disseminating = False
                self.last_exploit_bout = self.phase_ticks
                self.phase_ticks = 0

        if self.disseminating:
            self.exploit_total += 1
        else:
            self.explore_total += 1

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
            correct = ("" if self.true_best is None
                       else int(self.opinion == self.true_best))
            try:
                self.logger.log(
                    state={
                        "tick": self.tick_count,
                        "ctrl_variant": "baseline",
                        "robot_id": self.robot_id,
                        "patch": opt_idx,
                        "q_est": round(self.q_est, 6),
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
                print(f"[BaselineVoterExperiment] logging failed "
                      f"(motors unaffected): {log_exc!r}")

    def _update_estimate_from_ground(self, opt_idx):
        """
        Port of UpdateEstimateFromGround: refreshes q_est whenever the
        robot is on the patch matching its current opinion, or adopts a
        first opinion if it doesn't have one yet. Returns True if q_est
        was updated (this return value is deliberately NOT used to trigger
        the phase switch, matching the ARGoS controller).
        """
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
