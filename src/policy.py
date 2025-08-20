import numpy as np
import config
from scipy.spatial.transform import Rotation as R


class TeleopPolicy:
    """
    Converts raw input device state into robot actions.
    This policy defines how the robot should react to user input.
    """

    def __init__(self, translation_scale=None, rotation_scale=None, 
                 controller_deadzone=None, max_translation_delta=None, max_rotation_delta=None):
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.controller_deadzone = controller_deadzone
        self.max_translation_delta = max_translation_delta
        self.max_rotation_delta = max_rotation_delta

    def apply_deadzone(self, value, deadzone):
        return value if abs(value) > deadzone else 0.0
    
    def get_action(self, controller_state, current_ee_pose):
        """
        Calculates incremental movement commands for real-time control.

        Args:
            controller_state: The state object from the SpaceMouseManager.
            current_ee_pose: The current 4x4 pose matrix of the robot's end-effector.

        Returns:
            A dictionary containing incremental movement deltas and gripper command.
        """
        # Apply deadzone to spacemouse input

        # 1. Calculate Translation Delta (incremental movement) with deadzone and safety limits
        raw_translation = np.array([
            self.apply_deadzone(controller_state.x, self.controller_deadzone),
            self.apply_deadzone(controller_state.y, self.controller_deadzone), 
            self.apply_deadzone(controller_state.z, self.controller_deadzone)
        ]) * self.translation_scale

        # Apply safety limits
        delta_translation = np.clip(raw_translation, -self.max_translation_delta, self.max_translation_delta)

        # 2. Calculate Rotation Delta with deadzone and safety limits
        raw_rotation = np.array([
            self.apply_deadzone(controller_state.roll, self.controller_deadzone),
            self.apply_deadzone(-controller_state.pitch, self.controller_deadzone),
            self.apply_deadzone(-controller_state.yaw, self.controller_deadzone)
        ]) * self.rotation_scale * np.pi / 180.0 # Convert to radians

        # Apply safety limits for rotation
        delta_rotation = np.clip(raw_rotation, -self.max_rotation_delta, self.max_rotation_delta)


        # 3. Handle Gripper State using config values
        gripper_width = config.GRIPPER_CLOSED_WIDTH  # Default closed
        if controller_state.buttons[1]:  # Right button = open (like reference)
            gripper_width = config.GRIPPER_OPEN_WIDTH
        elif controller_state.buttons[0]:  # Left button = close (like reference)
            gripper_width = config.GRIPPER_CLOSED_WIDTH

        # For real-time control, we provide deltas instead of absolute pose
        # But also provide target pose for compatibility
        target_pose = current_ee_pose.copy()
        target_pose[:3, 3] += delta_translation

        if controller_state.buttons[1]:
            delta_rotation_matrix = R.from_euler('xyz', delta_rotation)
            current_rotation = R.from_matrix(target_pose[:3, :3])
            new_rotation = delta_rotation_matrix * current_rotation
            target_pose[:3, :3] = new_rotation.as_matrix()

        return {
            'ee_pose': target_pose,
            'gripper_width': gripper_width,
            'delta_translation': delta_translation,  # For real-time control
            'delta_rotation': delta_rotation,  # For real-time control
            'velocity_cmd': {  # For franky velocity control (not used anymore)
                "x": float(delta_translation[0]),
                "y": float(delta_translation[1]),
                "z": float(delta_translation[2]),
                "R": float(delta_rotation[0]),
                "P": float(delta_rotation[1]),
                "Y": float(delta_rotation[2]),
                "is_async": True,  # No duration - continuous velocity control
            }
        }
