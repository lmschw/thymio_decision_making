import asyncio
import numpy as np

class CommunicationTestExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

        self.value = np.random.choice([1,2,3,4,5,6,7,8,9,10], 1)

    async def run(self):
        print("in run")
        while self.running:
            print("is running")
            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            try:
                print("value:", self.value)
                await self.robot.send(self.value)

                incoming = await self.robot.receive()
                print("incoming: ", incoming)


                if self.logger:
                    try:
                        self.logger.log(
                            state={
                                "value": self.value,
                                "received": incoming,

                            },
                        )
                    except Exception as log_exc:
                        # A logging failure must NEVER stop the robot. Print and
                        # move on - motion for this tick already happened above.
                        print(f"[CommunicationTestExperiment] logging failed "
                            f"(motors unaffected): {log_exc!r}")


                print("finished tick")
            except Exception as exc:
                # Never let a single bad tick (e.g. a comms read failing
                # because no message has arrived yet) kill the loop and
                # leave the last drive() command latched on the motors.
                await self.robot.stop()
                print(f"[CommunicationTestExperiment] tick error, motors stopped: {exc!r}")

            await asyncio.sleep(0.05)

        await self.robot.stop()

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
