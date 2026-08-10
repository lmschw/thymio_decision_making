import asyncio
import numpy as np

class CommunicationTestExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger
        self.config = config or {}

        self.running = True
        self.paused = False

        self.value = np.random.randint(1,10)
        self.speed = 30

    async def run(self):
        print("in run")
        while self.running:
            print("is running")
            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.05)
                continue

            try:

                self.robot.connection.client.process_waiting_messages()

                rx = self.robot.connection.node.var.get("prox.comm.rx")
                payloads = self.robot.connection.node.var.get("prox.comm.rx._payloads")

                if rx != [0] or any(payloads):
                    print("[COMM RX]", rx, payloads)
                    
                print("value:", self.value)
                await self.robot.send(self.value)

                incoming, intensities, front_intensity, rear_intensity = await self.robot.receive()
                print("incoming: ", incoming)

                if  front_intensity > 0:
                    left = -self.speed
                    right = -self.speed
                elif rear_intensity > 0:
                    left = self.speed
                    right = self.speed
                else:
                    left = 0
                    right = 0

                await self.robot.drive(left,right)

                if self.logger:
                    try:
                        self.logger.log(
                            state={
                                "value": self.value,
                                "received": incoming,
                                "intensities": intensities,
                                "front_intensity": front_intensity,
                                "rear_intensity": rear_intensity
                            },
                            command={
                                "left_motor": left,
                                "right_motor": right,
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

            await asyncio.sleep(1)

        await self.robot.stop()

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
