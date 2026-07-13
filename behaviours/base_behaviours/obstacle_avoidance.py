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
        # ---------- Weighted sensor sums ----------
        left = (
            1.8 * prox[0] +
            1.2 * prox[1] +
            0.3 * prox[6]
        )

        right = (
            1.8 * prox[4] +
            1.2 * prox[3] +
            0.3 * prox[5]
        )

        front = (
            0.7 * prox[1] +
            2.5 * prox[2] +
            0.7 * prox[3]
        )

        # ---------- Escape behaviour ----------
        if self.reversing > 0:
            self.reversing -= 1

            if self.turn_bias > 0:
                return -80, 80
            else:
                return 80, -80

        # ---------- Wall directly ahead ----------
        #
        # If the centre sensor is high and both sides are roughly equal,
        # commit to one turn direction.
        #
        if front > 1.8 * self.delta and abs(left - right) < 300:

            self.reversing = 6

            # Alternate direction each time so we don't always choose
            # the wrong side in a corner.
            self.turn_bias *= -1

            if self.turn_bias > 0:
                return -80, 80
            else:
                return 80, -80

        # ---------- Continuous avoidance ----------

        # Cross-coupling:
        #
        # left obstacle -> slow RIGHT wheel
        # right obstacle -> slow LEFT wheel
        #
        K_SIDE = 0.025
        K_FRONT = 0.015

        left_speed = (
            self.wheel_velocity
            - K_SIDE * right
            - K_FRONT * front
        )

        right_speed = (
            self.wheel_velocity
            - K_SIDE * left
            - K_FRONT * front
        )

        # Never reverse during normal avoidance.
        left_speed = max(20, min(self.wheel_velocity, left_speed))
        right_speed = max(20, min(self.wheel_velocity, right_speed))

        return int(left_speed), int(right_speed)