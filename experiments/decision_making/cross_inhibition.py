import asyncio
import time

from behaviours.base_behaviours.obstacle_avoidance import ObstacleAvoidance
from behaviours.base_behaviours.colour_recognition import OptionGroundSensor
from behaviours.decision_making.cross_inhibition.cross_inhibition_model import process_neighbor_message
from behaviours.decision_making.environment.quality_switch import QualitySwitch
from utils.communication import encode_opinion_quality, decode_opinion_quality
from utils.utils import true_best_option


OPINION_COLORS = [
    (32, 0, 0),   # option 0 -> red
    (0, 32, 0),   # option 1 -> green
    (0, 0, 32),   # option 2 -> blue
    (32, 32, 0),  # option 3 -> yellow
]


class CrossInhibitionBaselineExperiment:
    """
    Best-of-N collective decision making via direct cross-inhibition
    dynamics.

    Port of CThymioBestOfTwo::ControlStep_CrossInhibition /
    ProcessNeighborMessagesCrossInhibition_Standard
    (control_variant="cross_inhibition") onto the async Robot API, in the
    same experiment-class shape as BaselineVoterBaselineExperiment /
    MajorityVotingBaselineExperiment / ActiveInferenceBaselineExperiment.

    Unlike those three, this variant has no explore/disseminate phase
    machine and no quality-estimation model:

      - if the robot has no opinion, it adopts whatever option patch it is
        currently standing on directly (no noise - matches
        DetectOptionFromGround being used raw, not through NoisyMeasure)
      - every tick it broadcasts its opinion (quality is always sent as
        0.0 - q_est is never populated for this variant, matching the
        ARGoS controller) and reacts to one incoming neighbour message:
          - no opinion yet -> RECRUIT toward the neighbour's opinion with
            probability kappa_recruit * qj
          - opinion differs from the neighbour's -> INHIBIT (drop back to
            undecided) with probability kappa_inhib * qj
        where qj is the neighbour's quality byte, floored at 0.05 (see
        behaviours.decision_making.cross_inhibition.cross_inhibition_model
        for why qj is always exactly that floor in practice).

    The explore/exploit flag used for bout-length bookkeeping is simply
    "has an opinion" (opinion >= 0), matching
    GetExploreExploitFlag()'s cross-inhibition branch - there is no
    internal phase timer to derive it from, so bout starts/ends are
    detected from flag transitions directly, the same way the ARGoS loop
    functions do it generically in TrackAndLog().

    Optionally, `swap_tick` / `gradual_reversal_ticks` config keys enable
    the same quality-reversal environment perturbation as ARGoS's loop
    functions (see behaviours.decision_making.environment.quality_switch)
    - disabled by default (swap_tick=0). Note noise_sigma has no effect
    here (there is nothing in this variant that consumes it, matching
    ARGoS - cross-inhibition never calls NoisyMeasure).
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

        self.kappa_recruit = self.config.get("kappa_recruit", 0.4)
        self.kappa_inhib = self.config.get("kappa_inhib", 1.0)

        self.ground_sensor = OptionGroundSensor(
            num_options=self.num_options,
            option_centers=self.config.get("option_centers"),
            allowed_offset=self.config.get("allowed_offset", 50),
        )

        # --- opinion state ---
        self.opinion = -1
        # never updated - matches the ARGoS controller, see class docstring
        self.q_est = 0.0

        # --- bout tracking (flag = has opinion; no internal phase timer,
        # so bout boundaries are detected from flag transitions directly,
        # like TrackAndLog() does generically for every control variant) ---
        self.prev_flag = None
        self.bout_start_tick = 0
        self.phase_ticks = 0

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
                print(f"[CrossInhibitionExperiment] tick error, motors stopped: {exc!r}")

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def _tick(self):
        self.tick_count += 1
        # Wall-clock time, for cross-robot alignment: tick_count is a local,
        # unsynchronized per-robot counter (unlike ARGoS's single shared
        # simulation clock), so post-hoc analysis needs a real timestamp to
        # align ticks across robots rather than trusting raw tick numbers.
        wall_time = time.time()

        # Apply the quality-reversal schedule (no-op unless swap_tick is
        # configured) and refresh env_state/true_best for this tick.
        self.quality_switch.apply(self.tick_count, self.option_qualities)
        self.env_state = self.quality_switch.env_state(self.tick_count)
        self.true_best = true_best_option(self.option_qualities)

        prox = await self.robot.proximity_horizontal()
        reflected = await self.robot.proximity_ground_reflected()

        opt_idx, _avg = self.ground_sensor.detect_option(reflected)

        msgs_tx_tick = 0
        msgs_rx_tick = 0

        # --- adopt an opinion directly from the current ground patch ---
        if self.opinion < 0 and 0 <= opt_idx < self.num_options:
            self.opinion = opt_idx

        # --- broadcast ---
        if self.opinion >= 0:
            await self.robot.send(
                encode_opinion_quality(self.opinion, self.q_est))
            msgs_tx_tick = 1
            self.msgs_tx_total += 1

        # --- react to one incoming neighbour message: recruit / inhibit ---
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
            self.opinion = process_neighbor_message(
                self.opinion, self.num_options, other_op, other_q,
                kappa_recruit=self.kappa_recruit, kappa_inhib=self.kappa_inhib)

        # --- bout tracking, from flag transitions (flag = has opinion) ---
        flag = 1 if self.opinion >= 0 else 0
        if self.prev_flag is None:
            self.prev_flag = flag
            self.bout_start_tick = self.tick_count
        elif flag != self.prev_flag:
            dur = self.tick_count - self.bout_start_tick
            if self.prev_flag == 0:
                self.last_explore_bout = dur
            else:
                self.last_exploit_bout = dur
            self.bout_start_tick = self.tick_count
            self.prev_flag = flag
        self.phase_ticks = self.tick_count - self.bout_start_tick

        if flag:
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
                        "ctrl_variant": "cross_inhibition",
                        "robot_id": self.robot_id,
                        "patch": opt_idx,
                        "q_est": round(self.q_est, 6),
                        "opinion": self.opinion,
                        "flag": flag,
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
                print(f"[CrossInhibitionExperiment] logging failed "
                      f"(motors unaffected): {log_exc!r}")

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
