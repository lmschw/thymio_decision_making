from pathlib import Path
import socket

import yaml

UNKNOWN = -1
WHITE_OPTION = 2

# Ground sensor used for classification (0 = left, 1 = right). The two
# sensors are not calibrated against each other and can differ by several
# hundred, so only one of them is used instead of their average. Can be
# overridden per robot with a "sensor" key in colour_calibration.yaml.
DEFAULT_SENSOR = 0

CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "config"
    / "colour_calibration.yaml"
)

with CALIBRATION_FILE.open() as f:
    ROBOT_CALIBRATION = yaml.safe_load(f)


class OptionGroundSensor:

    #ALLOWED_SENSOR_OFFSET = 40

    def __init__(self, num_options=3, sensor=None):
        hostname = socket.gethostname()

        if hostname not in ROBOT_CALIBRATION:
            raise ValueError(
                f"No ground sensor calibration found for hostname "
                f"'{hostname}'. Known robots: "
                f"{', '.join(ROBOT_CALIBRATION)}"
            )

        calibration = ROBOT_CALIBRATION[hostname]

        self.option_centers = calibration["option_centers"]
        self.allowed_offsets = calibration["allowed_offsets"]
        self.sensor = (sensor if sensor is not None
                       else calibration.get("sensor", DEFAULT_SENSOR))

        if self.sensor not in (0, 1):
            raise ValueError(
                f"Ground sensor index must be 0 (left) or 1 (right), "
                f"got {self.sensor}."
            )

        if len(self.option_centers) != num_options:
            raise ValueError(
                f"Calibration for '{hostname}' has "
                f"{len(self.option_centers)} option centres, "
                f"but {num_options} options were requested."
            )

        if len(self.allowed_offsets) != num_options:
            raise ValueError(
                f"Calibration for '{hostname}' has "
                f"{len(self.allowed_offsets)} allowed offsets, "
                f"but {num_options} options were requested."
            )

    def _classify(self, value: int) -> int:
        best_key = UNKNOWN
        best_distance = float("inf")

        for idx, centre in enumerate(self.option_centers):
            if idx == WHITE_OPTION:
                if value >= centre - self.allowed_offsets[idx]:
                    return idx
                continue

            distance = abs(value - centre)

            if distance < best_distance:
                best_distance = distance
                best_key = idx

        if best_distance <= self.allowed_offsets[best_key]:
            return best_key

        return UNKNOWN
        
    def detect_option(self, reflected):
        value = reflected[self.sensor]

        colour = self._classify(value)

        return colour, value
