import asyncio
import time
import numpy as np


from behaviours.base_behaviours.obstacle_avoidance import ObstacleAvoidance
from behaviours.base_behaviours.colour_recognition import OptionGroundSensor
from behaviours.decision_making.baseline.voter_model import noisy_measure
from behaviours.decision_making.baseline.majority_model import MajorityVoteTally, process_majority_tally
from utils.communication import encode_opinion_quality, decode_opinion_quality
from utils.utils import true_best_option


OPINION_COLORS = [
    (32, 0, 0),   # option 0 -> red
    (0, 32, 0),   # option 1 -> green
    (0, 0, 32),   # option 2 -> blue
    (32, 32, 0),  # option 3 -> yellow
]


class MajorityVotingBaselineExperiment:
    """
    Best-of-N collective decision making with the "baseline" timer-driven
    explore/disseminate cycle and majority-vote social influence.

    Port of CThymioBestOfTwo::ControlStep_Baseline
    (control_variant="baseline", social_model="majority") onto the async
    Robot API, in the same experiment-class shape as
    BaselineVoterExperiment / ActiveInferenceExperiment.

    EXPLORE phase: identical to BaselineVoterExperiment - drive around,
    sample ground patches, build a quality estimate, and switch to
    disseminating once `expl_timer` reaches `expl_max_ticks`.

    DISSEMINATE phase (timer-driven duration, scaled by quality):
      - broadcast (opinion, quality) every tick
      - the real Thymio's prox.comm link only ever exposes ONE message per
        tick (unlike ARGoS's range-and-bearing sensor, which sees every
        neighbour's message every tick), so there is nothing to tally
        votes over within a single tick. Instead, each tick's message is
        pushed into a rolling `window_ticks`-tick window (MajorityVoteTally)
        and the majority opinion currently held in that window is applied
        with the same probabilistic voter-model switch curve.
      - `dissem_timer`, initialised to tau0 + floor(tau_gain * quality),
        counts down to 0, after which the robot returns to exploring.

    A quality swap is triggered live, at any moment, by calling
    internal_update("swap") - this swaps the qualities of whichever two
    options currently hold the lowest and highest quality. It's driven by
    an external controller (e.g. decision_external_repo.py's
    send_swap_command), not scheduled internally: this class has no
    tick-based swap schedule or duration cutoff of its own - both the
    swap timing and the overall run length are controlled from outside
    (via internal_update and an explicit stop()).
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

        # env_state: 0 = no quality swap yet, 2 = swapped. Set by
        # internal_update("swap") - swaps are instantaneous and
        # externally triggered, there is no scheduled/gradual transition.
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
            #option_qualities = [max(0.1, 1.0 - 0.4 * i) for i in range(self.num_options)]
            option_qualities = [0.8, 0.3, 0.6]
        if len(option_qualities) != self.num_options:
            raise ValueError("option_qualities length must match num_options")
        self.option_qualities = option_qualities
        self.true_best = true_best_option(self.option_qualities)

        self.noise_sigma = self.config.get("noise_sigma", 0.05)
        self.voter_k = self.config.get("voter_k", 6.0)

        # timer params (mirrors expl_max_ticks / tau0 / tau_gain)
        self.expl_max_ticks = self.config.get("expl_max_ticks", 200)
        self.tau0 = self.config.get("tau0", 30)
        self.tau_gain = self.config.get("tau_gain", 100)

        # majority-vote window (real-hardware stand-in for ARGoS's
        # "all messages this tick" tally - see class docstring)
        self.window_ticks = self.config.get("window_ticks", 20)
        self.tally = MajorityVoteTally(
            num_options=self.num_options,
            window_ticks=self.window_ticks,
        )

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
                print(f"[MajorityVotingExperiment] tick error, motors stopped: {exc!r}")

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def _tick(self):
        self.tick_count += 1
        # Wall-clock time, for cross-robot alignment: tick_count is a local,
        # unsynchronized per-robot counter (unlike ARGoS's single shared
        # simulation clock), so post-hoc analysis needs a real timestamp to
        # align ticks across robots rather than trusting raw tick numbers.
        wall_time = time.time()

        # option_qualities can change at any moment via internal_update
        # ("swap"), so true_best must be recomputed live every tick,
        # matching GetTrueBestOption() being recomputed fresh every tick
        # in the ARGoS controller.
        self.true_best = true_best_option(self.option_qualities)

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
                # confidence fixed at 1.0: the baseline social models don't
                # use a confidence-weighted update like the AIF variant does.
                await self.robot.send(
                    encode_opinion_quality(self.opinion, self.q_est))
                msgs_tx_tick = 1
                self.msgs_tx_total += 1

            incoming = None
            try:
                incoming, _, _, _ = await self.robot.receive()
            except (TypeError, ValueError):
                # No message present yet - treat as "nothing received".
                incoming = None
            if incoming is not None:
                msgs_rx_tick = 1
                self.msgs_rx_total += 1
                other_op, other_q = decode_opinion_quality(incoming)
                self.tally.add(other_op, other_q)
                self.opinion, self.q_est = process_majority_tally(
                    self.opinion, self.q_est, self.tally, k=self.voter_k)

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
                        "timestamp": wall_time,
                        "ctrl_variant": "majority",
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
                print(f"[MajorityVotingExperiment] logging failed "
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

    async def internal_update(self, update_type):
        """
        Live, externally-triggered environment update - the sole way a
        quality swap happens now (no internal scheduling or duration
        cutoff exists in this class; both are controlled from outside).

        update_type="swap": swaps the qualities of whichever two options
        currently hold the lowest and highest quality (not fixed to
        options 0/1), and marks env_state as post-change (2). Meant to be
        invoked on demand by an external controller (e.g. via a
        session-level command), independent of this robot's own
        tick_count.
        """
        if update_type == "swap":
            i_min = np.argmin(self.option_qualities)
            i_max = np.argmax(self.option_qualities)
            v_max = self.option_qualities[i_max]
            self.option_qualities[i_max] = self.option_qualities[i_min]
            self.option_qualities[i_min] = v_max
            self.env_state = 2
            print("option_quality", self.option_qualities)

