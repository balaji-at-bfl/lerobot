from dataclasses import dataclass
from lerobot.teleoperators.config import TeleopConfig

@dataclass
class DobotTeleopConfig(TeleopConfig):
    name: str = "dobot"
    # Add other dobot teleop specific configs here
