from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# TODO(vikashplus): Kepp this for backward compatibility for now.
# Deprecated since v0.2
try:
    from .lekiwi import LeKiwi
    from .so101_follower import SO101Follower
    from .sim_101 import Sim101
except (ImportError, SyntaxError):
    # In case of a syntax error, it's likely that an old version of `lerobot` is being used,
    # where the dependencies for the robots are not installed.
    # In this case, we can safely ignore the import error.
    pass
