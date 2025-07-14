from .config import TeleoperatorConfig
from .teleoperator import Teleoperator
from .utils import make_teleoperator_from_config

# TODO(vikashplus): Kepp this for backward compatibility for now.
# Deprecated since v0.2
try:
    from .gamepad import Gamepad
    from .keyboard import Keyboard
    from .so101_leader import SO101Leader
    from .sim_101 import Sim101Teleop
except (ImportError, SyntaxError):
    # In case of a syntax error, it's likely that an old version of `lerobot` is being used,
    # where the dependencies for the teleoperators are not installed.
    # In this case, we can safely ignore the import error.
    pass
