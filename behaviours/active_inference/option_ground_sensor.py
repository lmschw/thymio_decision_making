from enum import Enum


class GroundOption(Enum):
    OPTION_0 = 0  # red
    OPTION_1 = 1  # green
    OPTION_2 = 2  # blue
    OPTION_3 = 3  # yellow
    WHITE = 100
    UNKNOWN = -1


class OptionGroundSensor:
    """
    Classifies raw ground-reflectance readings into one of up to 4 coloured
    decision-options (red/green/blue/yellow) or 'white' (empty arena floor).

    Port of CThymioBestOfTwo::ClassifyGroundBand / DetectOptionFromGround,
    adapted to the Thymio's two ground sensors
    (prox.ground.reflected[0], prox.ground.reflected[1]).

    `ground_max` normalises the raw ADC reading to [0, 1] before comparing
    against the calibrated bands (the ARGoS controller divides an 8-bit
    reading by 255; the real Thymio's prox.ground.reflected typically tops
    out well above that, so this is configurable).
    """

    def __init__(self,
                 num_options=2,
                 ground_max=1023,
                 white_thr=0.95,
                 color_eps=0.06,
                 red_level=0.30,
                 green_level=0.60,
                 blue_level=0.12,
                 yellow_level=0.75):
        self.num_options = num_options
        self.ground_max = ground_max
        self.white_thr = white_thr
        self.color_eps = color_eps
        self.levels = [red_level, green_level, blue_level, yellow_level]

    def _normalise(self, raw):
        return max(0.0, min(1.0, raw / self.ground_max))

    def classify_band(self, g):
        if g >= self.white_thr:
            return GroundOption.WHITE
        for idx, level in enumerate(self.levels[:self.num_options]):
            if abs(g - level) <= self.color_eps:
                return GroundOption(idx)
        return GroundOption.UNKNOWN

    def detect_option(self, reflected):
        """
        reflected: [left_reading, right_reading] raw ADC values from
        robot.proximity_ground_reflected().

        Returns (option_index, avg_normalised_ground):
          option_index is -1 if no option patch is currently detected.
        """
        g0 = self._normalise(reflected[0]) if len(reflected) > 0 else 1.0
        g1 = self._normalise(reflected[1]) if len(reflected) > 1 else 1.0
        avg = 0.5 * (g0 + g1)

        c0 = self.classify_band(g0)
        c1 = self.classify_band(g1)

        def is_opt(c):
            return c not in (GroundOption.UNKNOWN, GroundOption.WHITE)

        b0, b1 = is_opt(c0), is_opt(c1)
        if b0 and b1 and c0 == c1:
            return c0.value, avg
        if b0 and not b1:
            return c0.value, avg
        if not b0 and b1:
            return c1.value, avg
        return -1, avg
