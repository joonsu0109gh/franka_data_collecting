import numpy as np
import threading
import time
from typing import Dict, Any
from multiprocessing.managers import SharedMemoryManager
from franka.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from scipy.spatial.transform import Rotation as R

# Constants for movement (3x faster than working example)
MOVE_INCREMENT = 0.003


class RealTimeRobotInterface:
    """
    Real-time robot interface using panda_py with proper control loop and shared memory.
    Based on the reference implementation in robot.py but adapted for the modular system.
    """

    def __init__(self, robot_ip="172.16.0.2"):
        self.robot_ip = robot_ip

        # Shared memory for real-time communication
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Robot state template
        robot_state = {
            "q": np.zeros(7, dtype=np.float32),
            "dq": np.zeros(7, dtype=np.float32),
            "tau_J": np.zeros(7, dtype=np.float32),
            "EE_position": np.zeros(3, dtype=np.float32),
            "EE_orientation": np.zeros(4, dtype=np.float32),
            "gripper_state": 0.0
        }

        self.robot_state_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=self.shm_manager,
            examples=robot_state,
            get_max_k=32,
            get_time_budget=0.1,
            put_desired_frequency=15
        )

        # Control threading
        self._control_thread = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        # Movement deltas for real-time control (like reference code)
        self._delta_translation = np.zeros(3, dtype=np.float32)
        self._delta_rotation = np.zeros(3, dtype=np.float32)
        self._command_lock = threading.Lock()

        # Gripper control
        self._gripper_thread = None
        self._gripper_stop_event = threading.Event()
        self._gripper_command = None
        self._gripper_lock = threading.Lock()
        # Track previous button states for edge detection
        self._last_button_state = [False, False]

        # Real-time control variables
        self.current_translation = None
        self.current_rotation = None
        self.controller = None
        self.ctx = None

        # Initialize robot connection
        self._initialize_robot()

    def _initialize_robot(self):
        """Initialize the robot connection."""
        try:
            import panda_py
            import panda_py.controllers
            from panda_py import libfranka

            self.panda = panda_py.Panda(hostname=self.robot_ip)
            self.gripper = libfranka.Gripper(self.robot_ip)
            print(f"✅ Connected to robot at {self.robot_ip}")

            # Move to home position (initial setup) - matches reference robot.py
            home_pose = np.array([
                [-0.01588696],
                [-0.25534376],
                [0.18628714],
                [-2.28398158],
                [0.0769999],
                [2.02505396],
                [0.07858208]
            ], dtype=np.float64)

            self.panda.move_to_joint_position(home_pose)  # type: ignore
            self.gripper.move(width=0.09, speed=0.1)
            print("✅ Robot moved to home position")

        except ImportError as e:
            raise ImportError(
                f"panda_py library not found. Please install it with: pip install panda_py\nError: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to robot at {self.robot_ip}. Error: {e}")

    def start_realtime_control(self):
        """Start the real-time control thread."""
        if self._control_thread is None or not self._control_thread.is_alive():
            self._stop_event.clear()
            self._ready_event.clear()
            self._control_thread = threading.Thread(
                target=self._realtime_control_loop, daemon=True)
            self._control_thread.start()

            # Wait for control loop to be ready
            if not self._ready_event.wait(timeout=10.0):
                raise RuntimeError(
                    "Real-time control loop failed to start within 10 seconds")
            print("✅ Real-time control started")

    def _realtime_control_loop(self):
        """Real-time control loop running in separate thread - matches reference code pattern."""
        try:
            import panda_py.controllers

            # Create high-frequency control context (1000Hz like reference)
            self.ctx = self.panda.create_context(frequency=1000)
            self.controller = panda_py.controllers.CartesianImpedance()
            self.panda.start_controller(self.controller)
            time.sleep(1)  # Let controller settle like reference code

            # Get initial pose (like reference code)
            self.current_translation = self.panda.get_position()
            self.current_rotation = self.panda.get_orientation()

            # Start gripper control thread
            self._gripper_stop_event.clear()
            self._gripper_thread = threading.Thread(
                target=self._gripper_control_loop, daemon=True)
            self._gripper_thread.start()

            # Signal that we're ready
            self._ready_event.set()
            print("🔄 Real-time control loop started at 1000Hz")

            # Main real-time loop (matches working example: while ctx.ok() and not self._stop_event.is_set())
            while self.ctx.ok() and not self._stop_event.is_set():
                # Get movement deltas from main thread (like working example gets from spacemouse)
                with self._command_lock:
                    dpos = self._delta_translation * MOVE_INCREMENT
                    drot = self._delta_rotation * MOVE_INCREMENT * 3  # Like working example

                # Update translation (exactly like working example)
                self.current_translation += np.array(
                    [dpos[0], dpos[1], dpos[2]], dtype=np.float32)

                # Update rotation (exactly like working example)
                if np.any(drot != 0):
                    delta_rotation = R.from_euler("xyz", drot, degrees=False)
                    current_rotation_R = R.from_quat(self.current_rotation)
                    self.current_rotation = (
                        delta_rotation * current_rotation_R).as_quat()

                # Send to controller (like working example)
                self.controller.set_control(
                    self.current_translation, self.current_rotation)

                # Update state buffer
                self._update_robot_state()

                # (The panda context handles timing automatically)

            # Stop gripper thread
            self._gripper_stop_event.set()
            if self._gripper_thread:
                self._gripper_thread.join(timeout=1.0)

        except Exception:
            print("❌ Error in real-time control loop")
        finally:
            print("🔄 Real-time control loop stopped")

    def _gripper_control_loop(self):
        """Gripper control loop - execute commands only once per button press."""
        while not self._gripper_stop_event.is_set():
            try:
                command = None
                with self._gripper_lock:
                    command = self._gripper_command
                    # Clear command immediately to prevent re-execution
                    if command:
                        self._gripper_command = None

                if command == "open":
                    print("Opening gripper...")
                    success = self.gripper.move(width=0.08, speed=0.05)
                    if success:
                        print("Release successful")
                    else:
                        print("Release failed")
                elif command == "close":
                    print("Closing gripper...")
                    success = self.gripper.grasp(
                        width=0.01, speed=0.05, force=20,
                        epsilon_inner=0.005, epsilon_outer=0.005
                    )
                    if success:
                        print("Grasp successful")
                    else:
                        print("Grasp failed")

            except Exception as e:
                print(f"⚠️ Gripper control error: {e}")

            time.sleep(0.01)  # 100Hz check rate

    def _update_robot_state(self):
        """Update the shared robot state buffer."""
        try:
            robot_state = self.panda.get_state()
            gripper_state = self.gripper.read_once()

            state_data = {
                "q": robot_state.q,
                "dq": robot_state.dq,
                "tau_J": robot_state.tau_J,
                "EE_position": self.current_translation,
                "EE_orientation": self.current_rotation,
                "gripper_state": gripper_state.width
            }

            self.robot_state_buffer.put(state_data)
        except Exception as e:
            pass  # Don't spam errors in real-time loop

    def set_movement_delta(self, translation_delta, rotation_delta):
        """Set movement deltas for real-time control (replaces direct pose setting)."""
        with self._command_lock:
            self._delta_translation = np.array(
                translation_delta, dtype=np.float32)
            self._delta_rotation = np.array(rotation_delta, dtype=np.float32)

    def set_gripper_button_state(self, button_0_pressed, button_1_pressed):
        """Handle gripper control via button PRESS events (not continuous states)."""
        current_button_state = [button_0_pressed, button_1_pressed]

        # Only trigger on button press (rising edge), not continuous press
        # Left button just pressed
        if button_0_pressed and not self._last_button_state[0]:
            print("🔘 Left button pressed - closing gripper")
            with self._gripper_lock:
                self._gripper_command = "close"
        # Right button just pressed
        elif button_1_pressed and not self._last_button_state[1]:
            print("🔘 Right button pressed - opening gripper")
            with self._gripper_lock:
                self._gripper_command = "open"

        # Update last button state for next comparison
        self._last_button_state = current_button_state

    def get_obs(self) -> Dict[str, Any]:
        """Get current robot observation/state from shared memory buffer."""
        try:
            state_data = self.robot_state_buffer.get()
            if state_data is not None:
                return {
                    'panda_joint_positions': state_data['q'],
                    'panda_hand_pose': self._pose_from_position_orientation(
                        state_data['EE_position'],
                        state_data['EE_orientation']
                    ),
                    'panda_gripper_width': state_data['gripper_state']
                }
        except Exception:
            pass

        # Fallback if no data available
        return {
            'panda_joint_positions': np.zeros(7),
            'panda_hand_pose': np.eye(4),
            'panda_gripper_width': 0.0
        }

    def get_ee_pose(self):
        """Get current end-effector pose as 4x4 matrix."""
        obs = self.get_obs()
        return obs['panda_hand_pose']

    def step(self, action: Dict[str, Any]):
        """Execute a single action step by setting movement deltas."""
        # Use direct deltas if provided (preferred for real-time control)
        if 'delta_translation' in action and 'delta_rotation' in action:
            delta_translation = action['delta_translation']
            delta_rotation = action['delta_rotation']
            self.set_movement_delta(delta_translation, delta_rotation)
        else:
            # Fallback: calculate deltas from pose difference
            target_pose = action.get('ee_pose')
            if target_pose is not None:
                current_pose = self.get_ee_pose()
                delta_pos = target_pose[:3, 3] - current_pose[:3, 3]

                # Calculate rotation delta
                current_rot = R.from_matrix(current_pose[:3, :3])
                target_rot = R.from_matrix(target_pose[:3, :3])
                delta_rot = (target_rot * current_rot.inv()).as_euler("xyz")

                # Set movement deltas for real-time control
                self.set_movement_delta(delta_pos, delta_rot)

        # Gripper control is now handled by button states only, not policy actions
        # This prevents continuous gripper commands from the policy
        # The gripper will only respond to button press events via set_gripper_button_state()

    def open_gripper(self):
        """Open the robot gripper."""
        with self._gripper_lock:
            self._gripper_command = "open"

    def close_gripper(self):
        """Close the robot gripper."""
        with self._gripper_lock:
            self._gripper_command = "close"

    def reset_joints(self):
        """Reset robot to home position."""
        try:
            home_pose = np.array([
                [-0.01588696],
                [-0.25534376],
                [0.18628714],
                [-2.28398158],
                [0.0769999],
                [2.02505396],
                [0.07858208]
            ], dtype=np.float64)
            self.panda.move_to_joint_position(home_pose)  # type: ignore
        except Exception as e:
            print(f"⚠️ Could not reset joints: {e}")

    def reset_to_home_pose_realtime(self):
        """Reset robot to home position within real-time control context."""
        try:
            # Clear any movement deltas first
            with self._command_lock:
                self._delta_translation = np.zeros(3, dtype=np.float32)
                self._delta_rotation = np.zeros(3, dtype=np.float32)

            print("🏠 Reset to home pose (clearing movement commands)")
        except Exception as e:
            print(f"⚠️ Could not reset to home pose: {e}")

    def safe_episode_reset(self):
        """Safely reset robot for new episode by temporarily stopping real-time control."""
        try:
            print("🔄 Stopping real-time control for safe reset...")

            # Stop the real-time control temporarily
            self._stop_event.set()
            if self._control_thread and self._control_thread.is_alive():
                self._control_thread.join(timeout=3.0)

            # Now safely reset to home position
            home_pose = np.array([
                [-0.01588696],
                [-0.25534376],
                [0.18628714],
                [-2.28398158],
                [0.0769999],
                [2.02505396],
                [0.07858208]
            ], dtype=np.float64).flatten()
            self.panda.move_to_joint_position(home_pose)  # type: ignore

            # Clear movement deltas
            with self._command_lock:
                self._delta_translation = np.zeros(3, dtype=np.float32)
                self._delta_rotation = np.zeros(3, dtype=np.float32)

            # Restart real-time control
            print("🔄 Restarting real-time control...")
            self._stop_event.clear()
            self._ready_event.clear()
            self._control_thread = threading.Thread(
                target=self._realtime_control_loop, daemon=True)
            self._control_thread.start()

            # Wait for control loop to be ready
            if not self._ready_event.wait(timeout=10.0):
                raise RuntimeError("Real-time control loop failed to restart")

            print("✅ Episode reset complete with real-time control restarted")

        except Exception as e:
            print(f"⚠️ Could not perform safe episode reset: {e}")
            raise

    def stop(self):
        """Stop the real-time control loop."""
        self._stop_event.set()
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        print("✅ Real-time control stopped")

    def end(self):
        """Cleanup and end robot connection."""
        self.stop()
        if self.shm_manager:
            self.shm_manager.shutdown()
        print("✅ Robot interface cleaned up")

    def _pose_from_position_orientation(self, position, orientation):
        """Convert position and orientation to 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, 3] = position

        # Convert quaternion to rotation matrix
        if len(orientation) == 4:  # quaternion
            rotation = R.from_quat(orientation)
            pose[:3, :3] = rotation.as_matrix()

        return pose


# Backward compatibility alias
RobotInterface = RealTimeRobotInterface
