import time
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
        self.gripper_open = True
        self.last_gripper_toggle_time = 0

    def get_action(self, mouse_state, current_ee_pose):
        """
        Calculates the target end-effector pose and gripper action.

        Args:
            mouse_state: The state object from the SpaceMouseManager.
            current_ee_pose: The current 4x4 pose matrix of the robot's end-effector.

        Returns:
            A dictionary containing the target pose and gripper width.
        """
        target_pose = current_ee_pose.copy()

        # 1. Calculate Translation
        delta_translation = np.array(
            [mouse_state.x, mouse_state.y, mouse_state.z]) * self.translation_scale
        target_pose[:3, 3] += delta_translation

        # 2. Calculate Rotation (only if the right button is pressed)
        if mouse_state.buttons[1]:
            delta_rotation = R.from_euler(
                'yxz',
                [
                    mouse_state.roll * self.rotation_scale,
                    -mouse_state.pitch * self.rotation_scale,
                    -mouse_state.yaw * self.rotation_scale
                ],
                degrees=True
            )
            current_rotation = R.from_matrix(target_pose[:3, :3])
            # Apply the delta rotation: new_orientation = delta * old_orientation
            new_rotation = delta_rotation * current_rotation
            target_pose[:3, :3] = new_rotation.as_matrix()

        # 3. Handle Gripper State (left button toggles)
        # Debounce the button press to avoid rapid toggling
        if mouse_state.buttons[0] and (time.time() - self.last_gripper_toggle_time > 0.5):
            self.gripper_open = not self.gripper_open
            self.last_gripper_toggle_time = time.time()

        gripper_width = 0.08 if self.gripper_open else 0.0

        return {'ee_pose': target_pose, 'gripper_width': gripper_width}
