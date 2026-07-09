import math

from utils.geometry import SENSOR_ANGLES

class ObstacleAvoidance:

    def __init__(
        self,
        wheel_velocity=100,
        delta=1000,
        turn_steps=8,
    ):
        self.wheel_velocity = wheel_velocity
        self.delta = delta
        self.turn_steps = turn_steps

        self.turning_left = 0

        self.sensor_angles = SENSOR_ANGLES

    def step_motion(self, prox):
        """
        Translation of ARGoS StepMotion().
        Returns (left_motor, right_motor).
        """

        f_front = max(prox[:5])
        f_rear = max(prox[5], prox[6])

        f_left = max(prox[0], prox[1], prox[6])
        f_right = max(prox[3], prox[4], prox[5])

        b_front = f_front > self.delta
        b_rear = f_rear > self.delta

        # ---------------------------------------------------------
        # Continue a previously initiated turn
        # ---------------------------------------------------------
        if self.turning_left > 0:
            self.turning_left -= 1

            if f_left > f_right:
                return self.wheel_velocity, 0
            else:
                return 0, self.wheel_velocity

        # ---------------------------------------------------------
        # Obstacle only in front
        # ---------------------------------------------------------
        if b_front and not b_rear:
            self.turning_left = self.turn_steps

            if f_left > f_right:
                return -self.wheel_velocity, 0
            else:
                return 0, -self.wheel_velocity

        # ---------------------------------------------------------
        # Obstacle only in rear
        # ---------------------------------------------------------
        if b_rear and not b_front:
            self.turning_left = self.turn_steps

            if f_left > f_right:
                return self.wheel_velocity, 0
            else:
                return 0, self.wheel_velocity

        # ---------------------------------------------------------
        # Obstacles both front and rear
        # ---------------------------------------------------------
        if b_front and b_rear:
            self.turning_left = self.turn_steps

            if f_left > f_right:
                return -self.wheel_velocity, self.wheel_velocity
            else:
                return self.wheel_velocity, -self.wheel_velocity

        # ---------------------------------------------------------
        # Compute resultant proximity vector
        # ---------------------------------------------------------
        x = 0.0
        y = 0.0

        for value, angle in zip(prox, self.sensor_angles):
            x += value * math.cos(angle)
            y += value * math.sin(angle)

        x /= len(prox)
        y /= len(prox)

        length = math.hypot(x, y)
        angle = math.atan2(y, x)

        # Equivalent of:
        #
        # !(angle in straight_range && length < delta)
        #
        straight_range = math.radians(10)

        if not (-straight_range <= angle <= straight_range and
                length < self.delta):

            if angle < 0:
                return self.wheel_velocity, 0
            else:
                return 0, self.wheel_velocity

        # Drive straight
        return self.wheel_velocity, self.wheel_velocity
