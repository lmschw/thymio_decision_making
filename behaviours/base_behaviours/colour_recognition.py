from utils.colour import Colour

class GroundColourSensor:
    def __init__(self, colour_list=[Colour.BLACK, Colour.GREY, Colour.WHITE], allowed_colour_offset=50):
        self.colour_list = colour_list
        self.allowed_colour_offset = allowed_colour_offset

    def sense_ground_colour(self, reflected) -> Colour:
        left = self._classify(reflected[0])
        right = self._classify(reflected[1])

        return self._choose(left, right)

    def _classify(self, value: int) -> Colour:
        best_colour = Colour.GROUND_UNKNOWN
        best_distance = float("inf")

        for colour in self.colour_list:
            distance = abs(value - colour.default_centre)

            if distance < best_distance:
                best_distance = distance
                best_colour = colour

        if best_distance <= self.allowed_colour_offset:
            return best_colour

        return Colour.GROUND_UNKNOWN

    @staticmethod
    def _choose(left: Colour, right: Colour) -> Colour:

        if left == right:
            return left

        if left >= 0 and right <= 0:
            return left

        if left <= 0 and right >= 0:
            return right

        return Colour.GROUND_UNKNOWN