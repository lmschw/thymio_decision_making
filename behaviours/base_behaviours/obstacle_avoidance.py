import math

from utils.geometry import SENSOR_ANGLES_FLIPPED

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

        # Separate, smaller threshold for the *averaged* resultant vector
        # (7 raw values averaged together will basically never reach `delta`,
        # so gating "go straight" on the same threshold made it unreachable).
        self.straight_length_threshold = self.delta / 3
        
        # Wider "go straight" cone than the original 10 degrees, which was
        # tight enough that noise alone kept kicking the robot into the
        # correction branch.
        self.straight_range = math.radians(25)

        # Hysteresis: ignore direction flips smaller than this when the
        # resultant angle is close to zero, to stop side-to-side jitter.
        self.angle_deadzone = math.radians( 5)
        self._last_steer_angle = 0.0

        # Corner/stuck detection: count consecutive front-avoidance triggers.
        # If we keep re-triggering front avoidance without a break, we're
        # probably stuck pivoting in a corner -> back up instead.
        self.stuck_trigger_limit = 3
        self.reverse_steps = 6
        self._front_trigger_streak = 0
        self.reversing = 0

        self.turning_left = 0

        # Thymio proximity sensor angles (radians), matching the physical
        # layout used by f_left/f_right below: sensors 0, 1, 6 are on the
        # left (positive angle), sensors 3, 4, 5 are on the right (negative
        # angle), sensor 2 points straight ahead.
        self.sensor_angles = SENSOR_ANGLES_FLIPPED

    def step_motion(self, prox):
            """
            Translation of ARGoS StepMotion(), with hysteresis on the steering
            fallback and a reverse maneuver for corner/stuck recovery.
            Returns (left_motor, right_motor).
            """

            f_front = max(prox[:5])
            f_rear = max(prox[5], prox[6])

            f_left = max(prox[0], prox[1], prox[6])
            f_right = max(prox[3], prox[4], prox[5])

            b_front = f_front > self.delta
            b_rear = f_rear > self.delta

            # ---------------------------------------------------------
            # Reversing out of a corner
            # ---------------------------------------------------------
            if self.reversing > 0:
                self.reversing -= 1
                if f_left > f_right:
                    return -self.wheel_velocity, -self.wheel_velocity // 2
                else:
                    return -self.wheel_velocity // 2, -self.wheel_velocity

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
                self._front_trigger_streak += 1

                if self._front_trigger_streak >= self.stuck_trigger_limit:
                    # Pivoting hasn't cleared the obstacle after several
                    # consecutive triggers -> back up before trying again.
                    self._front_trigger_streak = 0
                    self.reversing = self.reverse_steps
                    if f_left > f_right:
                        return -self.wheel_velocity, -self.wheel_velocity // 2
                    else:
                        return -self.wheel_velocity // 2, -self.wheel_velocity

                self.turning_left = self.turn_steps

                if f_left > f_right:
                    return -self.wheel_velocity, 0
                else:
                    return 0, -self.wheel_velocity

            # Front is clear this tick -> reset the stuck counter
            self._front_trigger_streak = 0

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

            # Hysteresis: below the deadzone, keep steering the way we were
            # steering last tick instead of letting sensor noise flip it.
            if abs(angle) < self.angle_deadzone:
                angle = self._last_steer_angle
            self._last_steer_angle = angle

            if not (-self.straight_range <= angle <= self.straight_range and
                    length < self.straight_length_threshold):

                if angle < 0:
                    return self.wheel_velocity, 0
                else:
                    return 0, self.wheel_velocity

            # Drive straight
            self._last_steer_angle = 0.0
            return self.wheel_velocity, self.wheel_velocity
