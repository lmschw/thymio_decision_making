from enum import IntEnum

class GroundColour(IntEnum):
    BLACK = 0
    GREY = 1
    WHITE = 2

    GROUND_NEUTRAL = -1
    GROUND_UNKNOWN = -2


class GroundColourSensor:
    BLACK_CENTER = 51
    GREY_CENTER = 154
    WHITE_CENTER = 885

    ALLOWED_COLOUR_OFFSET = 50

    def sense_ground_colour(self, reflected) -> GroundColour:
        left = self._classify(reflected[0])
        right = self._classify(reflected[1])

        return self._choose(left, right)

    def _classify(self, value: int) -> GroundColour:
        centres = {
            GroundColour.BLACK: self.BLACK_CENTER,
            GroundColour.GREY: self.GREY_CENTER,
            GroundColour.WHITE: self.WHITE_CENTER,
        }

        best_colour = GroundColour.GROUND_UNKNOWN
        best_distance = float("inf")

        for colour, centre in centres.items():
            distance = abs(value - centre)

            if distance < best_distance:
                best_distance = distance
                best_colour = colour

        if best_distance <= self.ALLOWED_COLOUR_OFFSET:
            return best_colour

        return GroundColour.GROUND_UNKNOWN

    @staticmethod
    def _choose(left: GroundColour, right: GroundColour) -> GroundColour:

        if left == right:
            return left

        if left >= 0 and right <= 0:
            return left

        if left <= 0 and right >= 0:
            return right

        return GroundColour.GROUND_UNKNOWN