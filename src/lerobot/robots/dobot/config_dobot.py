from dataclasses import dataclass
from lerobot.robots.config import RobotConfig

@dataclass
class DobotConfig(RobotConfig):
    name = "dobot"
    ip_address: str = "192.168.1.6"
    # Add other dobot specific configs here
