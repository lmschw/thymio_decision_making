import asyncio
import math

from behaviours.base_behaviours.colour_recognition_per_robot import OptionGroundSensor
from behaviours.base_behaviours.obstacle_avoidance import ObstacleAvoidance
from utils.geometry import SENSOR_ANGLES

class ColourRecognitionExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

        # Parameters
        self.delta = self.config.get("delta", 1000)
        self.wheel_velocity = self.config.get("wheel_velocity", 300)

        self.turning_left = 0

        self.ground_sensor = OptionGroundSensor()
        self.obstacle_avoidance = ObstacleAvoidance(wheel_velocity=self.wheel_velocity,
                                                    delta=self.delta)

        # Approximate Thymio proximity sensor angles (radians)
        self.sensor_angles = SENSOR_ANGLES

    async def run(self):

        while self.running:

            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            prox = await self.robot.proximity_horizontal()
            reflected = await self.robot.proximity_ground_reflected()

            colour = self.ground_sensor.detect_option(reflected)

            await self.robot.top_led(0, 0, 0)

            if colour == 0:
                await self.robot.top_led(0, 0, 100)
            elif colour == 1:
                await self.robot.top_led(0, 100, 0)
            elif colour == 2:
                await self.robot.top_led(100, 100, 100)
            else:
                await self.robot.top_led(100, 0, 0)

            left, right = self.obstacle_avoidance.step_motion(prox)

            await self.robot.drive(left, right)

            if self.logger:
                self.logger.log(
                    state={"proximity": prox, 
                           "reflected_0": reflected[0], 
                           "reflected_1": reflected[1],
                           "reflected_avg": (reflected[0] + reflected[1])/2,
                           "colour": colour[0]},
                    command={
                        "left_motor": left,
                        "right_motor": right,
                    },
                )

            await asyncio.sleep(0.05)

        await self.robot.stop()


    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False