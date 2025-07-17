"""
Example usage of the modular data collection components.
This script demonstrates how to use each component independently.
"""

import time
import numpy as np
from src.input_manager import SpaceMouseManager
from src.policy import TeleopPolicy
from src.recorder import DataRecorder


def test_spacemouse():
    """Test SpaceMouse input reading."""
    print("Testing SpaceMouse input...")
    mouse = SpaceMouseManager()
    mouse.start()

    try:
        for i in range(50):  # Read 50 samples
            state = mouse.get_state()
            if state:
                print(f"Position: ({state.x:.3f}, {state.y:.3f}, {state.z:.3f}) "
                      f"Buttons: {state.buttons}")
            time.sleep(0.1)
    finally:
        mouse.stop()


def test_policy():
    """Test the teleoperation policy."""
    print("Testing teleoperation policy...")

    # Create a mock mouse state
    class MockMouseState:
        def __init__(self):
            self.x = 0.5
            self.y = -0.3
            self.z = 0.1
            self.roll = 10
            self.pitch = -5
            self.yaw = 15
            self.buttons = [False, True]  # Right button pressed

    policy = TeleopPolicy()
    current_pose = np.eye(4)
    mock_state = MockMouseState()

    action = policy.get_action(mock_state, current_pose)
    print(f"Generated action: {action}")


def test_recorder():
    """Test data recording functionality."""
    print("Testing data recorder...")

    recorder = DataRecorder("./test_data")

    # Record some mock data
    for i in range(10):
        mock_obs = {
            'panda_joint_positions': np.random.randn(7),
            'panda_hand_pose': np.eye(4),
            'panda_gripper_width': 0.05
        }
        mock_action = {
            'ee_pose': np.eye(4),
            'gripper_width': 0.08
        }
        recorder.record_step(mock_obs, mock_action)
        time.sleep(0.01)

    recorder.save_to_hdf5()
    print("Test data saved successfully!")


def main():
    """Run all tests."""
    print("Running component tests...\n")

    # Uncomment the tests you want to run:

    # test_spacemouse()  # Requires physical SpaceMouse
    test_policy()
    test_recorder()

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
