import math
import random

from utils.geometry import SENSOR_ANGLES as SENSOR_ANGLES

class ObstacleAvoidance:

    def __init__(
        self,
        wheel_velocity=100,
        delta=800,
        turn_steps=8,
    ):

        self.wheel_velocity = wheel_velocity
        self.delta = delta
        # Corner/stuck detection: count consecutive front-avoidance triggers.
        # If we keep re-triggering front avoidance without a break, we're
        # probably stuck pivoting in a corner -> back up instead.
        self.stuck_trigger_limit = 3
        self.reverse_steps = 6
        self.reversing = 0

        # Thymio proximity sensor angles (radians), matching the physical
        # layout used by f_left/f_right below: sensors 0, 1, 6 are on the
        # left (positive angle), sensors 3, 4, 5 are on the right (negative
        # angle), sensor 2 points straight ahead.
        self.sensor_angles = SENSOR_ANGLES

        self.filtered_x = 0.0
        self.filtered_y = 0.0

        # tuning
        self.filter_alpha = 0.25
        self.max_turn = self.wheel_velocity
        self.emergency_threshold = self.delta * 1.5

        # keep your existing reverse recovery
        self.reverse_steps = 6
        self.reversing = 0

        self.turn_bias = 1
        self.reversing = 0

    def step_motion(self, prox):
        # --------- Sensor groups ---------
        left = prox[0] + prox[1] + 0.5 * prox[6]
        right = prox[4] + prox[3] + 0.5 * prox[5]
        front = prox[1] + 2.0 * prox[2] + prox[3]

        # --------- Escape behaviour ---------
        if self.reversing > 0:
            self.reversing -= 1

            if self.turn_bias > 0:
                return -self.wheel_velocity, 0
            else:
                return 0, -self.wheel_velocity

        # Robot is completely facing a wall
        if (
            prox[2] > self.delta
            and abs(left - right) < 300
        ):
            self.reversing = 8
            self.turn_bias *= -1          # alternate direction each escape

            if self.turn_bias > 0:
                return -self.wheel_velocity, 0
            else:
                return 0, -self.wheel_velocity

        # --------- Continuous avoidance ---------

        K_SIDE = 0.015
        K_FRONT = 0.010

        left_speed = (
            self.wheel_velocity
            - K_SIDE * left
            - K_FRONT * front
        )

        right_speed = (
            self.wheel_velocity
            - K_SIDE * right
            - K_FRONT * front
        )

        # Never completely stop
        MIN_SPEED = 25

        left_speed = max(MIN_SPEED, min(self.wheel_velocity, left_speed))
        right_speed = max(MIN_SPEED, min(self.wheel_velocity, right_speed))

        return int(left_speed), int(right_speed)