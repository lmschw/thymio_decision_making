import asyncio

class ColourTestExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.config = config or {}
        self.logger = logger

        self.running = True
        self.paused = False

    async def run(self):
        counter = 0
        max_counter = 10
        while self.running:

            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.1)
                continue

            if counter < max_counter/2:
                await self.robot.drive(50, 50)
            else:
                await self.robot.drive(-50, -50)

            counter = counter % max_counter

            reflected = await self.robot.proximity_ground_reflected()

            if self.logger:
                self.logger.log(
                    state={"reflected_0": reflected[0], "reflected_1": reflected[1]},
                    command={}
                )

            await asyncio.sleep(1)

        await self.robot.drive(0, 0)

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False