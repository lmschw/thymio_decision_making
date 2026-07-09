from enum import IntEnum

class GroundColour(IntEnum):
    BLACK = 0
    GREY = 1
    WHITE = 2

    GROUND_NEUTRAL = -1
    GROUND_UNKNOWN = -2


class GroundColourSensor:
    BLACK_CENTER = 30
    GREY_CENTER = 520
    WHITE_CENTER = 900

    ALLOWED_COLOUR_OFFSET = 100

    async def sense_ground_colour(self, robot):
        reflected = await robot.proximity_ground_reflected()

        left = self._classify(reflected[0])
        right = self._classify(reflected[1])

        return self._choose(left, right)

    def _classify(self, value):
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
    def _choose(left, right):

        if left == right:
            return left

        if left >= 0 and right <= 0:
            return left

        if left <= 0 and right >= 0:
            return right

        return GroundColour.GROUND_UNKNOWN