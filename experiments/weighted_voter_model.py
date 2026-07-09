import asyncio

from behaviours.weighted_voter_model import WeightedVoterModel
from behaviours.obstacle_avoidance import ObstacleAvoidance

class WeightedVoterExperiment:

    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.logger = logger

        self.running = True
        self.paused = False

        self.voter = WeightedVoterModel()
        self.motion = ObstacleAvoidance()

    async def run(self):

        while self.running:

            if self.paused:
                await asyncio.sleep(0.1)
                continue

            # Social behaviour
            choice = await self.voter.perform_weighted_voter_model(
                self.robot
            )

            # Motion behaviour
            prox = await self.robot.proximity_horizontal()
            left, right, _ = self.motion.step_motion(prox)

            await self.robot.drive(left, right)

            if self.logger:
                self.logger.log(
                    state={
                        "choice": choice,
                        "quality": self.voter.q_est,
                    },
                    command={
                        "left_motor": left,
                        "right_motor": right,
                    },
                )

            await asyncio.sleep(0.05)