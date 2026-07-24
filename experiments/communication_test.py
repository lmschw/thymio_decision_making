import asyncio

from behaviours.decision_making.baseline.voter_model import noisy_measure, process_one_neighbor_message
from behaviours.base_behaviours.colour_recognition import OptionGroundSensor
from utils.communication import encode_message, decode_message


OPINION_COLORS = [
    (32, 0, 0),   # option 0 -> red
    (0, 32, 0),   # option 1 -> green
    (0, 0, 32),   # option 2 -> blue
    (32, 32, 0),  # option 3 -> yellow
]


class CommunicationTestExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

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
        self.opinion = 1
        self.q_est = 8.0

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

        self.other_op = -10
        self.other_q = -10
        self._other_conf = -10

    async def run(self):
        print("in run")
        while self.running:
            print("is running")
            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            try:
                await self._tick()
                print("finished tick")
            except Exception as exc:
                # Never let a single bad tick (e.g. a comms read failing
                # because no message has arrived yet) kill the loop and
                # leave the last drive() command latched on the motors.
                await self.robot.stop()
                print(f"[BaselineVoterExperiment] tick error, motors stopped: {exc!r}")

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def _tick(self):

        print("start tick")
        self.tick_count += 1

        # --- DISSEMINATE ---
        if self.opinion >= 0:
            # confidence fixed at 1.0: the baseline/voter model doesn't
            # use a confidence-weighted update like the AIF variant does.
            await self.robot.send(
                encode_message(self.opinion, self.q_est, 1.0))
            self.msgs_tx_total += 1

        incoming = None
        try:
            incoming = await self.robot.receive()
            print("incoming: ", incoming)
        except (TypeError, ValueError):
            # No message present yet - treat as "nothing received".
            incoming = None
        if incoming is not None:
            self.msgs_rx_total += 1
            self.other_op, self.other_q, self._other_conf = decode_message(incoming)
            self.opinion, self.q_est = process_one_neighbor_message(
                self.robot, self.opinion, self.q_est, self.other_op, self.other_q,
                k=self.voter_k)

        print("past dissemination")
        # --- LEDs: colour = current opinion ---
        if 0 <= self.opinion < len(OPINION_COLORS):
            r, g, b = OPINION_COLORS[self.opinion]
        else:
            r, g, b = (0, 0, 0)
        await self.robot.top_led(r, g, b)

        if self.logger:
            try:
                self.logger.log(
                    state={
                        "option": self.opinion,
                        "quality": self.q_est,
                        "received": incoming,
                        "other_op": self.other_op,
                        "other_q": self.other_q,
                        "other_conf": self._other_conf,
                        "msgs_tx_total": self.msgs_tx_total,
                        "msgs_rx_total": self.msgs_rx_total,
                    },
                    command={
                        "led_r": r, "led_g": g, "led_b": b,
                    },
                )
            except Exception as log_exc:
                # A logging failure must NEVER stop the robot. Print and
                # move on - motion for this tick already happened above.
                print(f"[BaselineVoterExperiment] logging failed "
                      f"(motors unaffected): {log_exc!r}")


    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
