import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
from scipy.spatial.transform import Rotation as R


class IncrementalDataRecorder:
    """
    Handles incremental data saving with proper directory structure and camera integration.
    Saves data at every timestep instead of buffering.
    """

    def __init__(self, log_root: str):
        self.log_root = Path(log_root)
        self.experiment_dir: Optional[Path] = None
        self.episode_dir: Optional[Path] = None
        self.current_timestep = 0
        self.last_action_pose = None  # For delta calculation

        # Camera setup (will be initialized when needed)
        self.primary_camera = None
        self.wrist_camera = None
        self.cameras_initialized = False

        # Create experiment directory
        self._create_experiment_dir()

    def _create_experiment_dir(self):
        """Create root experiment folder (e.g., /0000/)."""
        os.makedirs(self.log_root, exist_ok=True)

        # Find next experiment number
        existing_experiments = [d for d in self.log_root.iterdir()
                                if d.is_dir() and d.name.isdigit()]

        if existing_experiments:
            next_exp_num = max(int(d.name) for d in existing_experiments) + 1
        else:
            next_exp_num = 0

        self.experiment_dir = self.log_root / f"{next_exp_num:04d}"
        self.experiment_dir.mkdir(exist_ok=True)
        print(f"📁 Experiment directory: {self.experiment_dir}")

    def initialize_cameras(self, primary_serial: Optional[str] = None,
                           wrist_serial: Optional[str] = None):
        """
        Initialize RealSense cameras by serial numbers.
        If serials not provided, will use first two available cameras.
        """
        try:
            import pyrealsense2 as rs

            # Get connected devices
            ctx = rs.context()
            devices = ctx.query_devices()

            if len(devices) < 2:
                print(
                    f"⚠️ Only {len(devices)} camera(s) found. Need at least 2.")
                return False

            # Initialize cameras
            self.primary_camera = rs.pipeline()
            self.wrist_camera = rs.pipeline()

            config_primary = rs.config()
            config_wrist = rs.config()

            # Configure cameras with specific serials if provided
            available_serials = [dev.get_info(rs.camera_info.serial_number)
                                 for dev in devices]

            if primary_serial and primary_serial in available_serials:
                config_primary.enable_device(primary_serial)
                primary_used = primary_serial
            else:
                primary_used = available_serials[0]
                config_primary.enable_device(primary_used)

            if wrist_serial and wrist_serial in available_serials and wrist_serial != primary_used:
                config_wrist.enable_device(wrist_serial)
                wrist_used = wrist_serial
            else:
                wrist_used = available_serials[1] if len(
                    available_serials) > 1 else available_serials[0]
                config_wrist.enable_device(wrist_used)

            # Enable color streams
            config_primary.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config_wrist.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.primary_camera.start(config_primary)
            self.wrist_camera.start(config_wrist)

            # Warm up cameras
            for _ in range(10):
                self.primary_camera.wait_for_frames()
                self.wrist_camera.wait_for_frames()

            self.cameras_initialized = True
            print(
                f"📷 Cameras initialized - Primary: {primary_used}, Wrist: {wrist_used}")
            return True

        except ImportError:
            print("⚠️ pyrealsense2 not found. Install with: pip install pyrealsense2")
            return False
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False

    def start_new_episode(self, language_instruction: str) -> str:
        """
        Start a new episode with given language instruction.
        Returns the episode directory path.
        """
        if self.experiment_dir is None:
            raise RuntimeError("Experiment directory not initialized")

        # Find next episode number in current experiment
        existing_episodes = [d for d in self.experiment_dir.iterdir()
                             if d.is_dir() and d.name.isdigit()]

        if existing_episodes:
            next_ep_num = max(int(d.name) for d in existing_episodes) + 1
        else:
            next_ep_num = 0

        self.episode_dir = self.experiment_dir / f"{next_ep_num:06d}"
        self.episode_dir.mkdir(exist_ok=True)

        # Create steps subdirectory
        steps_dir = self.episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        # Save language instruction
        lang_file = self.episode_dir / "language_instruction.txt"
        with open(lang_file, 'w') as f:
            f.write(language_instruction)

        # Reset episode state
        self.current_timestep = 0
        self.last_action_pose = None

        print(f"📝 Started episode {next_ep_num:06d}: '{language_instruction}'")
        return str(self.episode_dir)

    def reset_episode_state(self):
        """Reset recorder state for new episode (clears action history)."""
        self.current_timestep = 0
        self.last_action_pose = None
        print("🔄 Recorder state reset for new episode")

    def capture_images(self):
        """Capture images from both cameras."""
        if not self.cameras_initialized or self.primary_camera is None or self.wrist_camera is None:
            # Return dummy images if cameras not available
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            return dummy_img, dummy_img

        try:
            # Capture frames
            frames_primary = self.primary_camera.wait_for_frames()
            frames_wrist = self.wrist_camera.wait_for_frames()

            color_frame_primary = frames_primary.get_color_frame()
            color_frame_wrist = frames_wrist.get_color_frame()

            if not color_frame_primary or not color_frame_wrist:
                print("⚠️ Failed to capture frames")
                dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                return dummy_img, dummy_img

            # Convert to numpy arrays
            img_primary = np.asanyarray(color_frame_primary.get_data())
            img_wrist = np.asanyarray(color_frame_wrist.get_data())

            return img_primary, img_wrist

        except Exception as e:
            print(f"⚠️ Image capture error: {e}")
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            return dummy_img, dummy_img

    def pose_matrix_to_6d(self, pose_matrix: np.ndarray) -> np.ndarray:
        """Convert 4x4 pose matrix to 6D vector [x, y, z, euler_x, euler_y, euler_z]."""
        # Extract position
        position = pose_matrix[:3, 3]

        # Extract rotation and convert to euler angles
        rotation_matrix = pose_matrix[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz')

        return np.concatenate([position, euler_angles])

    def gripper_width_to_binary(self, gripper_width: float, threshold: float = 0.01) -> float:
        """Convert gripper width to binary state."""
        return 1.0 if gripper_width > threshold else -1.0

    def record_timestep_with_images(self, observation: Dict[str, Any], action: Dict[str, Any],
                                    language_instruction: str, img_primary: np.ndarray, img_wrist: np.ndarray):
        """
        Record a single timestep with pre-captured images (for shared memory cameras).
        This version is optimized for non-blocking operation.
        """
        if self.episode_dir is None:
            raise RuntimeError(
                "No active episode. Call start_new_episode() first.")

        # Create timestep directory
        timestep_dir = self.episode_dir / \
            "steps" / f"{self.current_timestep:06d}"
        timestep_dir.mkdir(exist_ok=True)

        try:
            # 1. Save images (already captured from shared memory)
            primary_path = timestep_dir / "image_primary.jpg"
            wrist_path = timestep_dir / "image_wrist.jpg"

            cv2.imwrite(str(primary_path), img_primary)
            cv2.imwrite(str(wrist_path), img_wrist)

            # 2. Process robot state
            pose_6d = self.pose_matrix_to_6d(observation['panda_hand_pose'])
            gripper_binary = self.gripper_width_to_binary(
                observation['panda_gripper_width'])

            # 3. Process action deltas
            action_deltas = action.get('delta_translation', [
                                       0, 0, 0]) + action.get('delta_rotation', [0, 0, 0])

            # 4. Create data structure
            data = {
                'observation': {
                    'joint_positions': observation['panda_joint_positions'].astype(np.float32),
                    'ee_pose_6d': pose_6d.astype(np.float32),
                    'gripper_state': np.array([gripper_binary], dtype=np.float32)
                },
                'action': {
                    'ee_delta_pose_6d': np.array(action_deltas, dtype=np.float32),
                    # Use same as state for now
                    'gripper_action': np.array([gripper_binary], dtype=np.float32)
                },
                'language_instruction': language_instruction
            }

            # 5. Save data
            data_path = timestep_dir / "other.npz"
            np.savez_compressed(data_path, **{
                key: value for key, value in data.items()
                if key != 'language_instruction'
            })

            # 6. Save language instruction separately
            lang_path = timestep_dir / "language_instruction.txt"
            with open(lang_path, 'w') as f:
                f.write(language_instruction)

            self.current_timestep += 1

        except Exception as e:
            print(f"⚠️ Error recording timestep {self.current_timestep}: {e}")
            raise

    def record_timestep(self, observation: Dict[str, Any], action: Dict[str, Any],
                        language_instruction: str):
        """
        Record a single timestep with all required data.

        Args:
            observation: Current robot state
            action: Action taken at this timestep
            language_instruction: Language instruction for this episode
        """
        if self.episode_dir is None:
            raise RuntimeError(
                "No episode started. Call start_new_episode() first.")

        # Create timestep directory
        timestep_dir = self.episode_dir / \
            "steps" / f"{self.current_timestep:04d}"
        timestep_dir.mkdir(exist_ok=True)

        # 1. Capture images (minimize latency by doing this first after getting robot data)
        img_primary, img_wrist = self.capture_images()

        # Save images as JPG (swap to fix camera assignment)
        cv2.imwrite(str(timestep_dir / "image_primary.jpg"),
                    img_wrist)  # Swap: was img_primary
        cv2.imwrite(str(timestep_dir / "image_wrist.jpg"),
                    img_primary)   # Swap: was img_wrist

        # 2. Transform observation data
        gripper_pose = self.pose_matrix_to_6d(observation['panda_hand_pose'])
        gripper_open_state = self.gripper_width_to_binary(
            observation['panda_gripper_width'])
        joints = observation['panda_joint_positions']

        # 3. Transform action data
        # action_gripper_pose (Absolute Action)
        if 'ee_pose' in action:
            action_pose_6d = self.pose_matrix_to_6d(action['ee_pose'])
        else:
            # If no target pose, use current pose
            action_pose_6d = gripper_pose.copy()

        action_gripper_binary = self.gripper_width_to_binary(
            action.get('gripper_width', observation['panda_gripper_width']))

        action_gripper_pose = np.concatenate(
            [action_pose_6d, [action_gripper_binary]])

        # 4. Calculate delta action
        if self.last_action_pose is not None:
            delta_cur_2_last_action = action_gripper_pose - self.last_action_pose
        else:
            # First timestep - delta is zero
            delta_cur_2_last_action = np.zeros_like(action_gripper_pose)

        # Update last action for next timestep
        self.last_action_pose = action_gripper_pose.copy()

        # 5. Create other.npz data
        npz_data = {
            'gripper_pose': gripper_pose.astype(np.float32),
            'gripper_open_state': np.array([gripper_open_state], dtype=np.float32),
            'joints': joints.astype(np.float32),
            'action_gripper_pose': action_gripper_pose.astype(np.float32),
            'delta_cur_2_last_action': delta_cur_2_last_action.astype(np.float32),
            'language_instruction': language_instruction
        }

        # Save other.npz
        np.savez_compressed(timestep_dir / "other.npz", **npz_data)

        self.current_timestep += 1

    def cleanup(self):
        """Clean up camera resources."""
        if self.cameras_initialized:
            try:
                if self.primary_camera:
                    self.primary_camera.stop()
                if self.wrist_camera:
                    self.wrist_camera.stop()
                print("📷 Cameras stopped")
            except Exception as e:
                print(f"⚠️ Camera cleanup error: {e}")

        self.cameras_initialized = False


# Backward compatibility alias
DataRecorder = IncrementalDataRecorder
