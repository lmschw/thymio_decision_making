import asyncio
import math


class ObstacleAvoidanceExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

        # Parameters
        self.delta = self.config.get("delta", 1000)
        self.wheel_velocity = self.config.get("wheel_velocity", 100)
        self.turn_steps = self.config.get("turn_steps", 8)

        self.turning_left = 0

        # Approximate Thymio proximity sensor angles (radians)
        self.sensor_angles = [
            math.radians(-70),
            math.radians(-35),
            math.radians(0),
            math.radians(35),
            math.radians(70),
            math.radians(145),
            math.radians(-145),
        ]

    async def run(self):

        while self.running:

            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            prox = await self.robot.proximity_horizontal()

            left, right = self.step_motion(prox)

            await self.robot.drive(left, right)

            if self.logger:
                self.logger.log(
                    state={"proximity": prox},
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