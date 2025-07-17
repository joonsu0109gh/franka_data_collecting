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
        """Initialize the robot connection. Override this method for different robot types."""
        try:
            # Try to use the existing FrankaAPI from robot.py
            from robot import FrankaAPI
            from franka.utils.inputs.spacemouse_shared_memory import Spacemouse
            from multiprocessing.managers import SharedMemoryManager

            # Create spacemouse for robot initialization
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            spacemouse = Spacemouse(shm_manager=shm_manager, deadzone=0.1)

            self.robot = FrankaAPI(robot_ip=self.robot_ip)
            self.robot.startup(spacemouse)
            self._robot_type = "FrankaAPI"
            print("Using FrankaAPI robot interface")

        except ImportError:
            try:
                # Fallback to PandaRealRobot if available
                from src.frankapy.src.realrobot import PandaRealRobot
                self.robot = PandaRealRobot()
                self._robot_type = "PandaRealRobot"
                print("Using PandaRealRobot interface")
            except ImportError:
                raise ImportError(
                    "No compatible robot interface found. Please ensure either robot.py or frankapy is available.")

    def get_obs(self) -> Dict[str, Any]:
        """Get current robot observation/state."""
        if self._robot_type == "FrankaAPI":
            status = self.robot.get_status()
            if status is not None:
                return {
                    'panda_joint_positions': status['q'],
                    'panda_hand_pose': self._pose_from_position_orientation(
                        status['EE_position'],
                        status['EE_orientation']
                    ),
                    'panda_gripper_width': status['gripper_state']
                }
            else:
                # Return default values if no status available
                return {
                    'panda_joint_positions': np.zeros(7),
                    'panda_hand_pose': np.eye(4),
                    'panda_gripper_width': 0.0
                }
        elif self._robot_type == "PandaRealRobot":
            return self.robot.get_obs()
        else:
            raise NotImplementedError(
                f"get_obs not implemented for robot type: {self._robot_type}")

    def get_ee_pose(self):
        """Get current end-effector pose as 4x4 matrix."""
        obs = self.get_obs()
        return obs['panda_hand_pose']

    def set_ee_pose(self, target_pose, duration=0.1):
        """Set target end-effector pose."""
        if self._robot_type == "FrankaAPI":
            # For FrankaAPI, the pose control is handled internally via spacemouse
            # This is a simplified implementation - you might need to modify robot.py
            # to accept direct pose commands
            pass
        elif self._robot_type == "PandaRealRobot":
            self.robot.set_ee_pose(target_pose, duration=duration)
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
        if self._robot_type == "FrankaAPI":
            # FrankaAPI handles gripper through spacemouse buttons
            pass
        elif self._robot_type == "PandaRealRobot":
            self.robot.open_gripper()

    def close_gripper(self):
        """Close the robot gripper."""
        if self._robot_type == "FrankaAPI":
            # FrankaAPI handles gripper through spacemouse buttons
            pass
        elif self._robot_type == "PandaRealRobot":
            self.robot.close_gripper()

    def reset_joints(self):
        """Reset robot to a neutral joint configuration."""
        if self._robot_type == "FrankaAPI":
            # FrankaAPI initializes to home position in startup
            pass
        elif self._robot_type == "PandaRealRobot":
            if hasattr(self.robot, 'reset_joints'):
                self.robot.reset_joints()
            else:
                print("reset_joints not available for this robot type")

    def end(self):
        """Cleanup and end robot connection."""
        if self.robot is not None:
            if self._robot_type == "FrankaAPI":
                self.robot.stop()
            elif self._robot_type == "PandaRealRobot":
                self.robot.end()

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
