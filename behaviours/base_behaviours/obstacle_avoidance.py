import math

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

    def step_motion(self, prox):
        """
        Continuous obstacle avoidance using a repulsive potential field.

        Returns:
            (left_motor, right_motor)
        """

        f_front = max(prox[:5])
        f_rear = max(prox[5], prox[6])

        # ---------------------------------------------------------
        # Recovery mode
        # ---------------------------------------------------------
        if self.reversing > 0:
            self.reversing -= 1

            x = 0.0
            y = 0.0

            for value, angle in zip(prox, self.sensor_angles):
                x += value * math.cos(angle)
                y += value * math.sin(angle)

            angle = math.atan2(y, x)

            steer = self.max_turn * math.sin(angle)

            left = -self.wheel_velocity + steer
            right = -self.wheel_velocity - steer

            return int(left), int(right)

        # ---------------------------------------------------------
        # Emergency escape
        # ---------------------------------------------------------
        if f_front > self.emergency_threshold:
            self.reversing = self.reverse_steps
            return -self.wheel_velocity, -self.wheel_velocity

        # ---------------------------------------------------------
        # Build repulsive vector
        # ---------------------------------------------------------
        x = 0.0
        y = 0.0

        for value, angle in zip(prox, self.sensor_angles):
            x += value * math.cos(angle)
            y += value * math.sin(angle)

        x /= len(prox)
        y /= len(prox)

        # ---------------------------------------------------------
        # Low-pass filtering
        # ---------------------------------------------------------
        a = self.filter_alpha

        self.filtered_x = a * x + (1.0 - a) * self.filtered_x
        self.filtered_y = a * y + (1.0 - a) * self.filtered_y

        x = self.filtered_x
        y = self.filtered_y

        # ---------------------------------------------------------
        # Resultant obstacle vector
        # ---------------------------------------------------------
        length = math.hypot(x, y)

        if length < 1e-6:
            return self.wheel_velocity, self.wheel_velocity

        angle = math.atan2(y, x)

        # ---------------------------------------------------------
        # Convert obstacle vector into steering
        #
        # Positive y = obstacle on left
        # Negative y = obstacle on right
        # ---------------------------------------------------------
        avoidance = min(1.0, length / self.delta)

        steer = -math.sin(angle) * avoidance * self.max_turn

        # slow down when obstacles are nearby
        speed = self.wheel_velocity * (1.0 - 0.7 * avoidance)

        left = speed + steer
        right = speed - steer

        left = max(-self.wheel_velocity,
                min(self.wheel_velocity, left))

        right = max(-self.wheel_velocity,
                    min(self.wheel_velocity, right))

        return int(left), int(right)