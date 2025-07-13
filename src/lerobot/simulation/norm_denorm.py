import numpy as np

class RobotJointController:
    """
    A controller to normalize and unnormalize robot joint angles based on their
    specific hardware ranges. Supports both dictionary and array-based operations.
    """

    # Joint ranges extracted from the MuJoCo XML file (in radians)
    # The order of this dictionary defines the order for array operations.
    JOINT_RANGES_MAP = {
        'shoulder_pan': (-1.91986, 1.91986),
        'shoulder_lift': (-1.74533, 1.74533),
        'elbow_flex': (-1.69, 1.69),
        'wrist_flex': (-1.65806, 1.65806),
        'wrist_roll': (-2.74385, 2.84121),
        'gripper': (-0.17453, 1.74533),
    }

    def __init__(self, target_min=-100, target_max=100):
        """
        Initializes the controller and pre-computes arrays for vectorized operations.
        """
        self.target_min = target_min
        self.target_max = target_max
        self.target_range = target_max - target_min

        # --- Setup for array-based operations ---
        # 1. Establish a fixed joint order
        self.joint_names = list(self.JOINT_RANGES_MAP.keys())
        
        # 2. Create NumPy arrays of the original ranges based on the fixed order
        ranges = np.array(list(self.JOINT_RANGES_MAP.values()))
        self.min_ranges_rad = ranges[:, 0]
        self.max_ranges_rad = ranges[:, 1]
        self.original_ranges_rad = self.max_ranges_rad - self.min_ranges_rad

    # --- Dictionary-based methods ---

    def _normalize(self, joint_angles: dict[str, float]) -> dict[str, float]:
        """(Dictionary in, Dictionary out) Normalizes angles to the target range."""
        normalized_angles = {}
        for joint_name, angle in joint_angles.items():
            if joint_name not in self.JOINT_RANGES_MAP:
                raise ValueError(f"Unknown joint '{joint_name}' found.")
            
            min_orig, max_orig = self.JOINT_RANGES_MAP[joint_name]
            original_range = max_orig - min_orig
            
            scaled_value = self.target_min + (angle - min_orig) * (self.target_range / original_range)
            normalized_angles[joint_name] = np.clip(scaled_value, self.target_min, self.target_max)
        return normalized_angles

    def _unnormalize(self, normalized_angles: dict[str, float]) -> dict[str, float]:
        """(Dictionary in, Dictionary out) Unnormalizes values to radians."""
        original_angles = {}
        for joint_name, norm_value in normalized_angles.items():
            if joint_name not in self.JOINT_RANGES_MAP:
                raise ValueError(f"Unknown joint '{joint_name}' found.")

            min_orig, max_orig = self.JOINT_RANGES_MAP[joint_name]
            original_range = max_orig - min_orig
            norm_value = np.clip(norm_value, self.target_min, self.target_max)

            original_value = min_orig + (norm_value - self.target_min) * (original_range / self.target_range)
            original_angles[joint_name] = original_value
        return original_angles

    # --- Array-based methods ---

    def _normalize_array(self, joint_angles_array: np.ndarray) -> np.ndarray:
        """
        (Array in, Array out) Normalizes a NumPy array of joint angles.
        The input array order must match self.joint_names.
        """
        if joint_angles_array.shape[0] != len(self.joint_names):
            raise ValueError(f"Input array must have {len(self.joint_names)} elements.")
        
        # Vectorized min-max scaling, no loops needed
        scaled_array = self.target_min + (joint_angles_array - self.min_ranges_rad) * \
                       (self.target_range / self.original_ranges_rad)
        
        return np.clip(scaled_array, self.target_min, self.target_max)

    def _unnormalize_array(self, normalized_angles_array: np.ndarray) -> np.ndarray:
        """
        (Array in, Array out) Unnormalizes a NumPy array to radian values.
        The input array order must match self.joint_names.
        """
        if normalized_angles_array.shape[0] != len(self.joint_names):
            raise ValueError(f"Input array must have {len(self.joint_names)} elements.")
        
        # Clip input to ensure it's within the expected bounds
        clipped_norm_array = np.clip(normalized_angles_array, self.target_min, self.target_max)

        # Vectorized inverse scaling
        original_array = self.min_ranges_rad + (clipped_norm_array - self.target_min) * \
                         (self.original_ranges_rad / self.target_range)
        
        return original_array

# --- Example Usage for Array Methods ---

# # 1. Create an instance of the controller
# controller = RobotJointController()
# print("Fixed joint order for array operations:")
# print(controller.joint_names)
# print("-" * 40)

# # 2. Define a "home" position as a NumPy array.
# #    The order MUST match controller.joint_names.
# #    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# home_position_array = np.array([0.0, -1.75, 1.57, 1.18, 0.0, -0.174])

# # 3. Normalize the array of radian values
# normalized_array = controller._normalize_array(home_position_array)
# print("Original Array (rad):", np.round(home_position_array, 4))
# print("Normalized Array [-100, 100]:", np.round(normalized_array, 2))

# print("\n" + "="*40 + "\n")

# # 4. Unnormalize the array back to radians
# unnormalized_array = controller._unnormalize_array(normalized_array)
# print("Normalized Array [-100, 100]:", np.round(normalized_array, 2))
# print("Unnormalized Array (rad):", np.round(unnormalized_array, 4))

# # Check if the round trip was successful
# assert np.allclose(home_position_array, unnormalized_array, atol=1e-5)
# print("\nRound trip successful!")