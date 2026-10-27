class ObstacleAvoidance:
    """
    Obstacle avoidance using the Thymio's
    7 horizontal proximity sensors (5 front, 2 back).

    Two states:
      - normal driving: goes straight on a clear path, or turns away from
        the closest obstacle (turn direction held for a few steps to avoid
        oscillations), until either the path clears or the turn gets too
        long / an obstacle gets too close, both of which trigger backward.
      - backward escape: reverse while the rear stays clear, then
        pivot away from the closest obstacle, then go back to normal driving.

    Parameters:
      wheel_velocity (300): base wheel speed
      delta (800): threshold to start turning away from an obstacle
        (stored as min_prox)
      max_prox (2500): threshold to trigger an immediate backward move
      max_side (3500): threshold on left and right together that also
        triggers backward
      max_turn (15): max steps held in a turn before forcing backward
      max_backward (10): steps spent reversing 
    """

    def __init__(
        self,
        wheel_velocity=300,
        delta=800,
    ):
        self.wheel_velocity = wheel_velocity
        self.delta = delta

        self.count_backward = 0
        self.count_turn = 0
        self.max_backward = 10
        self.max_turn = 15
        self.backward = False
        self.max_prox = 2500
        self.max_side = 3500
        self.min_prox = delta
        self.turn_direction = None


    def step_motion(self, prox):
        """
        Compute the next (left, right) wheel velocities from a proximity
        reading.

        prox: list of 7 horizontal proximity sensor values (indices 0-4
        front, 5-6 back). Higher values mean closer obstacle.
        Returns: (left_velocity, right_velocity).
        """
        left = prox[0] + prox[1]
        right = prox[3] + prox[4]
        closer = max(prox[:5])
        factor = self._factor(closer)
        if not self.backward: 
            if closer >= self.max_prox or (left >= self.max_side and right >= self.max_side):
                self.backward = True
                self.turn_direction = None
                self.count_turn = 0
                return -self.wheel_velocity, -self.wheel_velocity
            elif closer >= self.min_prox:
                if self.turn_direction is not None and self.count_turn < self.max_turn:
                    self.count_turn += 1
                    if self.turn_direction == "right" and (right - left) > 400:
                        self.turn_direction = "left"
                    elif self.turn_direction == "left" and (left - right) > 400:
                        self.turn_direction = "right"

                    if self.turn_direction == "right":
                        return self.wheel_velocity, self.wheel_velocity * max(0, 1 - 2 * factor)
                    else:
                        return self.wheel_velocity * max(0, 1 - 2 * factor), self.wheel_velocity
                elif self.turn_direction is not None and self.count_turn >= self.max_turn:
                    self.backward = True
                    self.turn_direction = None
                    self.count_turn = 0
                    return -self.wheel_velocity, -self.wheel_velocity
                else:
                    self.count_turn = 0
                    if left > right:
                        self.turn_direction = "right"
                        return self.wheel_velocity, self.wheel_velocity * max(0, 1 - 2 * factor)
                    else:
                        self.turn_direction = "left"
                        return self.wheel_velocity * max(0, 1 - 2 * factor), self.wheel_velocity
            else:  
                self.turn_direction = None
                self.count_turn = 0
                return self.wheel_velocity, self.wheel_velocity
        else:
            self.count_backward += 1
            if self.count_backward < self.max_backward and prox[5] < self.min_prox and prox[6] < self.min_prox:
                return -self.wheel_velocity, -self.wheel_velocity
            elif self.count_backward < self.max_backward and (prox[5] >= self.min_prox or prox[6] >= self.min_prox):
                self.count_backward = self.max_backward
                if left > right:
                    self.turn_direction = "right"
                    return self.wheel_velocity, -self.wheel_velocity
                else:
                    self.turn_direction = "left"
                    return -self.wheel_velocity, self.wheel_velocity
            elif self.count_backward >= self.max_backward and self.count_backward < (self.max_backward + 5):
                if self.turn_direction is None :
                    if left > right:
                        self.turn_direction = "right"
                        return self.wheel_velocity, -self.wheel_velocity
                    else:
                        self.turn_direction = "left"
                        return -self.wheel_velocity, self.wheel_velocity
                elif self.turn_direction == "right" : 
                    return self.wheel_velocity, -self.wheel_velocity
                else : 
                    return -self.wheel_velocity, self.wheel_velocity
            else:
                self.count_backward = 0
                self.backward = False
                self.turn_direction = None
                return self.wheel_velocity, self.wheel_velocity

    def _factor(self, closer):
        """
        Normalise the closest obstacle reading to [0, 1] between
        min_prox and 4500, used to scale how sharply the robot turns.
        """
        factor = (closer - self.min_prox) / (4500 - self.min_prox)
        return max(0.0, min(1.0, factor))
