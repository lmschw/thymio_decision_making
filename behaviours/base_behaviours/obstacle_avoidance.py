import random

class ObstacleAvoidance:

    def __init__(
        self,
        wheel_velocity=100,
        delta=800,
    ):
        self.wheel_velocity = wheel_velocity
        self.delta = delta

        # Number of consecutive cycles spent escaping before
        # trying normal driving again.
        self.escape_steps = 8

        # If we've had several failed escape attempts,
        # make the next one longer.
        self.max_escape_level = 3

        self.escape_timer = 0
        self.escape_level = 0

        # None / "left" / "right"
        self.turn_direction = None

    def step_motion(self, prox):

        f_front = max(prox[:5])
        f_rear = max(prox[5:])

        left_strength = prox[0] + prox[1]
        right_strength = prox[3] + prox[4]

        front_clear = f_front < self.delta * 0.6

        # --------------------------------------------------
        # Continue an escape manoeuvre
        # --------------------------------------------------

        if self.escape_timer > 0:

            self.escape_timer -= 1

            # Once the front is genuinely clear,
            # immediately return to normal driving.
            if front_clear:
                self.escape_timer = 0
                self.escape_level = 0
                self.turn_direction = None

            else:

                if self.turn_direction == "left":
                    return self.wheel_velocity, -self.wheel_velocity
                else:
                    return -self.wheel_velocity, self.wheel_velocity

        # --------------------------------------------------
        # Front obstacle
        # --------------------------------------------------

        if f_front > self.delta:

            # Pick a direction ONCE.
            if self.turn_direction is None:

                difference = left_strength - right_strength

                if abs(difference) < 200:
                    # Symmetric wall / another robot.
                    self.turn_direction = random.choice(["left", "right"])

                elif difference > 0:
                    # More obstacle on the left -> turn right.
                    self.turn_direction = "right"

                else:
                    self.turn_direction = "left"

            self.escape_level = min(
                self.escape_level + 1,
                self.max_escape_level
            )

            self.escape_timer = (
                self.escape_steps
                + 2 * self.escape_level
            )

            if self.turn_direction == "left":
                return self.wheel_velocity, -self.wheel_velocity
            else:
                return -self.wheel_velocity, self.wheel_velocity

        # --------------------------------------------------
        # Rear obstacle
        # --------------------------------------------------

        if f_rear > self.delta:

            if left_strength > right_strength:
                return self.wheel_velocity, 0
            else:
                return 0, self.wheel_velocity

        # --------------------------------------------------
        # Gentle steering
        # --------------------------------------------------

        error = left_strength - right_strength

        steer = 0.015 * error

        left = self.wheel_velocity + steer
        right = self.wheel_velocity - steer

        left = max(40, min(self.wheel_velocity, left))
        right = max(40, min(self.wheel_velocity, right))

        return int(left), int(right)