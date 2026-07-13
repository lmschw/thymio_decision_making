import math

SENSOR_ANGLES = [
    math.radians(-70),
    math.radians(-35),
    math.radians(0),
    math.radians(35),
    math.radians(70),
    math.radians(145),
    math.radians(-145),
]

SENSOR_ANGLES_FLIPPED = [
            math.radians(70),    # 0: front-left-outer
            math.radians(35),    # 1: front-left-inner
            math.radians(0),     # 2: front-center
            math.radians(-35),   # 3: front-right-inner
            math.radians(-70),   # 4: front-right-outer
            math.radians(-145),  # 5: rear-right
            math.radians(145),   # 6: rear-left
        ]