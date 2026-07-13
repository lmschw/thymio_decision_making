from enum import Enum

class Colour(int, Enum):
    GROUND_NEUTRAL = (-1, "Ground_Neutral", (32, 32, 0), None)
    GROUND_UNKNOWN = (-2, "Ground_Unknown", (32, 32, 0), None)
    BLACK = (0, "Black", (0, 0, 10), 51)
    GREY  = (1, "Grey",  (16, 16, 16), 154)
    WHITE = (2, "White", (32, 32, 32), 885)

    def __new__(cls, id_, label, rgb, default_centre):
        obj = int.__new__(cls, id_)
        obj._value_ = id_          # The enum's value is the integer ID
        obj.label = label
        obj.rgb = rgb
        obj.default_centre = default_centre
        return obj
