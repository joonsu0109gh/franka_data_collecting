import numpy as np
from typing import Dict, Any


class RobotInterface:
    """
    A unified interface for robot control that can work with different robot implementations.
    This adapter allows the data collection system to work with various robot backends.
    """

    def __init__(self, robot_ip="172.16.0.2"):
        self.robot_ip = robot_ip
        self.robot = None
        self._initialize_robot()

    def _initialize_robot(self):
        """Initialize the robot connection. Using panda_py library."""
        try:
            # Use panda_py directly
            import panda_py
            import panda_py.controllers
            from panda_py import libfranka
            import numpy as np

            self.panda = panda_py.Panda(hostname=self.robot_ip)
            self.gripper = libfranka.Gripper(self.robot_ip)
            self._robot_type = "panda_py"
            print(
                f"Using panda_py robot interface connected to {self.robot_ip}")

            # Move to home position
            home_pose = np.array([
                -0.01588696, -0.25534376, 0.18628714, -2.28398158,
                0.0769999, 2.02505396, 0.07858208,
            ])
            self.panda.move_to_joint_position(home_pose)
            self.gripper.move(width=0.09, speed=0.1)  # Open gripper

        except ImportError as e:
            raise ImportError(
                f"panda_py library not found. Please install it with: pip install panda_py\nError: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to robot at {self.robot_ip}. Error: {e}")

    def get_obs(self) -> Dict[str, Any]:
        """Get current robot observation/state."""
        if self._robot_type == "panda_py":
            try:
                # Get robot state
                robot_state = self.panda.get_state()
                gripper_state = self.gripper.read_once()
                ee_pos = self.panda.get_position()
                ee_ori = self.panda.get_orientation()

                return {
                    'panda_joint_positions': robot_state.q,
                    'panda_hand_pose': self._pose_from_position_orientation(ee_pos, ee_ori),
                    'panda_gripper_width': gripper_state.width
                }
            except Exception as e:
                print(f"Warning: Could not get robot state: {e}")
                # Return default values if robot state unavailable
                return {
                    'panda_joint_positions': np.zeros(7),
                    'panda_hand_pose': np.eye(4),
                    'panda_gripper_width': 0.0
                }
        else:
            raise NotImplementedError(
                f"get_obs not implemented for robot type: {self._robot_type}")

    def get_ee_pose(self):
        """Get current end-effector pose as 4x4 matrix."""
        obs = self.get_obs()
        return obs['panda_hand_pose']

    def set_ee_pose(self, target_pose, duration=0.1):
        """Set target end-effector pose."""
        if self._robot_type == "panda_py":
            try:
                # Extract position and orientation from 4x4 matrix
                position = target_pose[:3, 3]
                orientation_matrix = target_pose[:3, :3]

                # Convert rotation matrix to quaternion
                from scipy.spatial.transform import Rotation as R
                rotation = R.from_matrix(orientation_matrix)
                quaternion = rotation.as_quat()  # [x, y, z, w]

                # Use move_to_pose instead of set_pose
                self.panda.move_to_pose(position, quaternion)
            except Exception as e:
                print(f"Warning: Could not set EE pose: {e}")
        else:
            raise NotImplementedError(
                f"set_ee_pose not implemented for robot type: {self._robot_type}")

    def step(self, action: Dict[str, Any]):
        """Execute a single action step."""
        target_pose = action.get('ee_pose')
        gripper_width = action.get('gripper_width', 0.0)

        if target_pose is not None:
            self.set_ee_pose(target_pose)

        # Handle gripper
        if gripper_width > 0.04:  # Open threshold
            self.open_gripper()
        else:
            self.close_gripper()

    def open_gripper(self):
        """Open the robot gripper."""
        if self._robot_type == "panda_py":
            try:
                self.gripper.move(width=0.09, speed=0.1)
            except Exception as e:
                print(f"Warning: Could not open gripper: {e}")

    def close_gripper(self):
        """Close the robot gripper."""
        if self._robot_type == "panda_py":
            try:
                self.gripper.grasp(width=0.01, speed=0.1, force=20)
            except Exception as e:
                print(f"Warning: Could not close gripper: {e}")

    def reset_joints(self):
        """Reset robot to a neutral joint configuration."""
        if self._robot_type == "panda_py":
            try:
                home_pose = np.array([
                    -0.01588696, -0.25534376, 0.18628714, -2.28398158,
                    0.0769999, 2.02505396, 0.07858208,
                ])
                self.panda.move_to_joint_position(home_pose)
            except Exception as e:
                print(f"Warning: Could not reset joints: {e}")

    def end(self):
        """Cleanup and end robot connection."""
        if self._robot_type == "panda_py":
            try:
                # No explicit cleanup needed for panda_py
                print("Robot connection ended.")
            except Exception as e:
                print(f"Warning: Error during robot cleanup: {e}")

    def _pose_from_position_orientation(self, position, orientation):
        """Convert position and orientation to 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, 3] = position

        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        if len(orientation) == 4:  # quaternion
            rotation = R.from_quat(orientation)
            pose[:3, :3] = rotation.as_matrix()

        return pose
