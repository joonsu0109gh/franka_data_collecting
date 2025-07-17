import os
import h5py
import time
import re
import random
import numpy as np
from pathlib import Path


class DataRecorder:
    """Handles data buffering and saving to HDF5 files."""

    def __init__(self, log_root: str):
        self.save_path = self._create_log_path(log_root)
        self.trajectory_buffer = []
        print(f"Data will be saved to: {self.save_path}")

    def _create_log_path(self, log_root: str):
        """Creates a unique, timestamped directory for the trajectory file."""
        log_folder = Path(log_root) / time.strftime("%Y-%m-%d")
        os.makedirs(log_folder, exist_ok=True)

        # Find the next available demonstration number
        pattern = r"log-(\d{6})-\d{4}"
        files = os.listdir(log_folder)
        existing_numbers = []
        for f in files:
            match = re.match(pattern, f)
            if match:
                existing_numbers.append(int(match.group(1)))
        next_number = max(existing_numbers) + 1 if existing_numbers else 1

        random_id = random.randint(1000, 9999)
        dir_path = log_folder / f"log-{next_number:06d}-{random_id}"
        os.makedirs(dir_path, exist_ok=True)

        return dir_path / "traj.hdf5"

    def record_step(self, observation: dict, action: dict):
        """
        Records a single timestep of data into an in-memory buffer.

        Args:
            observation: A dictionary containing robot state (e.g., from robot.get_obs()).
            action: A dictionary containing the action sent to the robot.
        """
        step_data = {
            'timestamp': time.time(),
            'joint_positions': observation.get('panda_joint_positions', []),
            'end_effector_pose': observation.get('panda_hand_pose', []),
            'gripper_width': observation.get('panda_gripper_width', 0.0),
            'action_ee_pose_target': action.get('ee_pose', []),
            'action_gripper_width_target': action.get('gripper_width', 0.0)
        }
        self.trajectory_buffer.append(step_data)

    def save_to_hdf5(self):
        """Saves the buffered trajectory data to an HDF5 file."""
        if not self.trajectory_buffer:
            print("No data to save.")
            return

        print(f"Saving {len(self.trajectory_buffer)} steps to HDF5 file...")
        with h5py.File(self.save_path, 'w') as f:
            # Convert list of dicts to dict of lists
            keys = self.trajectory_buffer[0].keys()
            data_dict = {key: np.array(
                [step[key] for step in self.trajectory_buffer]) for key in keys}

            for key, data in data_dict.items():
                f.create_dataset(key, data=data)

        print("Data saved successfully.")
