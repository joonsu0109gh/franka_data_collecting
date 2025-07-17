import numpy as np
from scipy.spatial.transform import Rotation as R


class TeleopPolicy:
    """
    Converts raw input device state into robot actions.
    This policy defines how the robot should react to user input.
    """

    def __init__(self, translation_scale=0.005, rotation_scale=0.5):
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale

    def get_action(self, mouse_state, current_ee_pose):
        """
        Calculates incremental movement commands for real-time control.

        Args:
            mouse_state: The state object from the SpaceMouseManager.
            current_ee_pose: The current 4x4 pose matrix of the robot's end-effector.

        Returns:
            A dictionary containing incremental movement deltas and gripper command.
        """
        # 1. Calculate Translation Delta (incremental movement)
        delta_translation = np.array(
            [mouse_state.x, mouse_state.y, mouse_state.z]) * self.translation_scale

        # 2. Calculate Rotation Delta (disabled like reference robot.py)
        # Reference code sets drot_xyz[:] = 0, so we disable rotation for now
        delta_rotation = np.zeros(3)
        # if mouse_state.buttons[1]:  # Right button for rotation (disabled)
        #     delta_rotation = np.array([
        #         mouse_state.roll * self.rotation_scale,
        #         -mouse_state.pitch * self.rotation_scale,
        #         -mouse_state.yaw * self.rotation_scale
        #     ]) * np.pi / 180.0  # Convert to radians

        # 3. Handle Gripper State (match reference: button 0=close, button 1=open)
        # No toggle logic - direct button control like reference
        gripper_width = 0.0  # Default closed
        if mouse_state.buttons[1]:  # Right button = open (like reference)
            gripper_width = 0.08
        elif mouse_state.buttons[0]:  # Left button = close (like reference)
            gripper_width = 0.0

        # For real-time control, we provide deltas instead of absolute pose
        # But also provide target pose for compatibility
        target_pose = current_ee_pose.copy()
        target_pose[:3, 3] += delta_translation

        if mouse_state.buttons[1]:
            delta_rotation_matrix = R.from_euler('xyz', delta_rotation)
            current_rotation = R.from_matrix(target_pose[:3, :3])
            new_rotation = delta_rotation_matrix * current_rotation
            target_pose[:3, :3] = new_rotation.as_matrix()

        return {
            'ee_pose': target_pose,
            'gripper_width': gripper_width,
            'delta_translation': delta_translation,  # For real-time control
            'delta_rotation': delta_rotation  # For real-time control
        }
