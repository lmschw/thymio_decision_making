"""
Ground patch classifier for decision-making options, calibrated to real
Thymio hardware readings.

Same "nearest calibrated centre within an allowed offset" classification
as GroundColourSensor, generalised to N configurable option centres.

Unlike the earlier version, white is NOT treated as a "background /
no-option" sentinel here - it's a real decision option like black or
grey (matching a 3-option best-of-3 over black / grey / white patches).
UNKNOWN only means "this reading doesn't match any calibrated centre
closely enough", not "this is the empty floor".
"""

UNKNOWN = -1    # reading doesn't match any calibrated centre closely enough


class OptionGroundSensor:

    ALLOWED_OFFSET = 50

    # Default centres, in index order: option 0, option 1, option 2
    # (black, grey, white - calibrated hardware values from
    # GroundColourSensor: BLACK_CENTER=51, GREY_CENTER=154, WHITE_CENTER=885).
    DEFAULT_OPTION_CENTERS = [51, 154, 885]

    def __init__(self,
                 num_options=3,
                 option_centers=None,
                 allowed_offset=None):
        self.num_options = num_options
        self.option_centers = (list(option_centers) if option_centers is not None
                                else self.DEFAULT_OPTION_CENTERS[:num_options])
        if len(self.option_centers) != num_options:
            raise ValueError(
                "option_centers length must match num_options "
                f"({len(self.option_centers)} != {num_options})")
        self.allowed_offset = (allowed_offset if allowed_offset is not None
                                else self.ALLOWED_OFFSET)

    def _classify(self, value: int) -> int:
        """
        Returns the option index (0..num_options-1) whose centre is
        nearest `value`, or UNKNOWN if nothing is within allowed_offset.
        """
        best_key = UNKNOWN
        best_distance = float("inf")

        for idx, centre in enumerate(self.option_centers):
            distance = abs(value - centre)
            if distance < best_distance:
                best_distance = distance
                best_key = idx

        if best_distance <= self.allowed_offset:
            return best_key

        return UNKNOWN

    def _choose(self, left: int, right: int) -> int:
        def is_option(c):
            return c >= 0

        if left == right:
            return left

        if is_option(left) and not is_option(right):
            return left

        if is_option(right) and not is_option(left):
            return right

        return UNKNOWN

    def detect_option(self, reflected):
        """
        reflected: [left_reading, right_reading] raw ADC values from
        robot.proximity_ground_reflected().

        Returns (option_index, avg_reading):
          option_index is -1 only if the two sensors disagree on
          different options, or the reading matches no centre at all.
        """
        left = self._classify(reflected[0]) if len(reflected) > 0 else UNKNOWN
        right = self._classify(reflected[1]) if len(reflected) > 1 else UNKNOWN
        avg = 0.5 * ((reflected[0] if len(reflected) > 0 else 0)
                     + (reflected[1] if len(reflected) > 1 else 0))

        return self._choose(left, right), avg
