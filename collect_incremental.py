import time
import config
import numpy as np
from multiprocessing import Process, shared_memory, Event

from src.incremental_recorder import IncrementalDataRecorder
from src.robot_interface import RobotInterface
from src.utils import Rate
from src.camera_reader import camera_process_realsense, dummy_camera_process
import threading

class DataCollectionController:
    """Orchestrates the entire data collection process with incremental saving."""

    def __init__(self,
                 data_log_root=None,
                 robot_ip=None,
                 loop_rate_hz=None,
                 robot_loop_rate_hz=None,
                 primary_camera_serial=None,
                 wrist_camera_serial=None,
                 home_pose=None,
                 gripper_open_width=None,
                 gripper_closed_width=None):

        # Use config file values as defaults
        self.data_log_root = data_log_root
        self.robot_ip = robot_ip
        self.loop_rate_hz = loop_rate_hz
        self.home_pose = home_pose
        self.robot_loop_rate_hz = robot_loop_rate_hz
        self.gripper_open_width = gripper_open_width
        self.gripper_closed_width = gripper_closed_width

        self.first_record_flg = True
        self.is_collecting_flg = False  # Flag to control camera processes

        # Initialize components with config parameters
        self.robot = RobotInterface(robot_ip=self.robot_ip, home_pose=self.home_pose, 
                                    robot_loop_rate_hz=self.robot_loop_rate_hz,
                                    gripper_open_width=self.gripper_open_width,
                                    gripper_closed_width=self.gripper_closed_width)

        self.recorder = IncrementalDataRecorder(self.data_log_root)

        # Initialize precise rate limiter for stable control loop
        self.rate_limiter = Rate(self.loop_rate_hz)

        # Initialize cameras with shared memory for fast access
        print("------Initializing shared memory cameras...------")
        print(f"⏳ Camera initialization delay set: {config.CAMERA_INITIALIZATION_DELAY} seconds.")

        camera_success = self._initialize_shared_memory_cameras(
            primary_camera_serial, wrist_camera_serial)
        
        # Give cameras time to initialize
        time.sleep(config.CAMERA_INITIALIZATION_DELAY)
        if not camera_success:
            print("⚠️ Cameras failed to initialize, continuing without cameras")

    def _initialize_shared_memory_cameras(self, primary_serial=None, wrist_serial=None):
        """Initialize shared memory cameras for fast, non-blocking access."""
        try:
            # Define camera properties
            self.cam_shape = (480, 640, 3)  # Height, Width, Channels
            self.cam_dtype = np.uint8
            frame_size = int(np.prod(self.cam_shape) *
                             np.dtype(self.cam_dtype).itemsize)            # Create shared memory blocks for both cameras
            self.shm_primary = shared_memory.SharedMemory(
                create=True, size=frame_size)
            self.shm_wrist = shared_memory.SharedMemory(
                create=True, size=frame_size)
            
            self.depth_shape = (480, 640)  # Depth images are single channel
            self.depth_dtype = np.uint16  
            depth_frame_size = int(np.prod(self.depth_shape) *
                                   np.dtype(self.depth_dtype).itemsize)  # Same dtype for depth

            self.shm_primary_depth = shared_memory.SharedMemory(
                create=True, size=depth_frame_size)   
            self.shm_wrist_depth = shared_memory.SharedMemory(
                create=True, size=depth_frame_size)

            # Create NumPy array views for easy access
            self.latest_frame_primary = np.ndarray(
                self.cam_shape, dtype=self.cam_dtype, buffer=self.shm_primary.buf)
            self.latest_frame_wrist = np.ndarray(
                self.cam_shape, dtype=self.cam_dtype, buffer=self.shm_wrist.buf)
            self.latest_frame_primary_depth = np.ndarray(
                self.depth_shape, dtype=self.depth_dtype, buffer=self.shm_primary_depth.buf)
            self.latest_frame_wrist_depth = np.ndarray(
                self.depth_shape, dtype=self.depth_dtype, buffer=self.shm_wrist_depth.buf)

            # Initialize with black frames
            self.latest_frame_primary.fill(0)
            self.latest_frame_wrist.fill(0)
            self.latest_frame_primary_depth.fill(0)
            self.latest_frame_wrist_depth.fill(0)

            # Start camera process
            self.camera_stop_event = Event()

            # Try RealSense cameras first, fall back to dummy if not available
            try:
                self.camera_process = Process(
                    target=camera_process_realsense,
                    args=(self.shm_primary.name, self.shm_wrist.name, self.shm_primary_depth.name, self.shm_wrist_depth.name,
                          self.cam_shape, self.cam_dtype, self.depth_shape, self.depth_dtype, self.camera_stop_event, self.is_collecting_flg,
                          primary_serial, wrist_serial)
                )
                print("📸 Starting RealSense camera processes...")
            except Exception:
                self.camera_process = Process(
                    target=dummy_camera_process,
                    args=(self.shm_primary.name, self.shm_wrist.name, self.shm_primary_depth.name, self.shm_wrist_depth.name,
                          self.cam_shape, self.cam_dtype, self.camera_stop_event)
                )
                print("📸 Starting dummy camera processes...")

            self.camera_process.start()

            return True

        except Exception as e:
            print(f"❌ Shared memory camera initialization failed: {e}")
            return False

    def get_camera_images(self):
        """Get the latest camera images from shared memory (non-blocking)."""
        try:
            # Copy to avoid race conditions (this is very fast)
            img_primary = self.latest_frame_primary.copy()
            img_wrist = self.latest_frame_wrist.copy()
            depth_primary = self.latest_frame_primary_depth.copy()
            depth_wrist = self.latest_frame_wrist_depth.copy()
            return img_primary, img_wrist, depth_primary, depth_wrist
        except Exception as e:
            print(f"❌ Error getting camera images: {e}")
            # Return dummy images if something goes wrong
            dummy_img = np.zeros(self.cam_shape, dtype=self.cam_dtype)
            return dummy_img, dummy_img, dummy_img, dummy_img

    def collect_episode(self, language_instruction: str, episode_num: int, total_episodes: int):
        """Collect a single episode of demonstration data."""
        print(f"\n{'='*60}")
        print(f"🎬 EPISODE {episode_num}/{total_episodes}")
        print(f"📝 Instruction: '{language_instruction}'")
        print(f"{'='*60}")

        # Reset robot and clear control signals for new episode
        print("🔄 Preparing for new episode...")
        self._reset_for_new_episode()

        # Start new episode
        episode_dir = self.recorder.start_new_episode(language_instruction)

        print("Controls:") 
        if self.robot.controller_type == "spacemouse":
            print("  • Move SpaceMouse: Control robot position")
            print("  • Left button (first): Close gripper")
            print("  • Right button (last): Open gripper")
            print("  • Hold BOTH buttons for 3 seconds: End episode and continue to next")
        elif self.robot.controller_type == "xbox":
            print("  • Left Joystick: Control robot position")
            print("  • Right Joystick: Control robot orientation")
            print("  • A button: Close gripper")
            print("  • B button: Open gripper")
            print("  • Hold BOTH A and B buttons for 3 seconds: End episode and continue to next")
        print("="*60 + "\n")

        step_count = 0
        both_buttons_start_time = None  # Track when both buttons were pressed
        episode_ending = False  # Track if episode is ending to skip recording

        # Reset the rate limiter for this episode
        self.rate_limiter.reset()
        self.is_collecting_flg = True  # Set flag to indicate we are collecting data
        try:
            while True:
                # 1. Get current robot state (Observation) - minimize latency
                current_obs = self.robot.get_obs()
                current_ee_pose = current_obs['panda_hand_pose']

                # 2. Get user input
                controller_state = self.robot.controller.get_state()

                if controller_state is None:
                    time.sleep(0.1)  # Small sleep if no mouse data
                    continue

                # Parse button states for different SpaceMouse models
                button_left = False
                button_right = False

                if hasattr(controller_state, 'buttons') and controller_state.buttons:
                    if self.robot.controller_type == "spacemouse":
                        if len(controller_state.buttons) >= 15:
                            # SpaceMouse Pro/Enterprise with 15 buttons
                            # First button (left)
                            button_left = controller_state.buttons[0]
                            # Last button (right)
                            button_right = controller_state.buttons[14]
                        elif len(controller_state.buttons) >= 2:
                            # Standard SpaceMouse with 2 buttons
                            button_left = controller_state.buttons[0]
                            button_right = controller_state.buttons[1]
                        elif len(controller_state.buttons) == 1:
                            # Some SpaceMouse models only report one button
                            button_left = controller_state.buttons[0]
                            # Remove print to avoid slowing down control loop
                    elif self.robot.controller_type == "xbox":
                        # Xbox controller buttons are already handled in the state
                        button_left = controller_state.buttons[0]
                        button_right = controller_state.buttons[1]
                else:
                    print("⚠️ No button state available, using defaults")
                    button_left = False
                    button_right = False

                # Check for both buttons held for episode end
                if button_left and button_right:  # Both buttons pressed
                    if both_buttons_start_time is None:
                        both_buttons_start_time = time.time()
                        episode_ending = True  # Start skipping data recording
                        print("⏳ Hold both buttons for 3 seconds to end episode...")
                        print(
                            "🗑️ Skipping data recording during episode transition...")
                    elif time.time() - both_buttons_start_time >= 3.0:
                        recorded_timesteps = step_count # Actual recorded data points
                        print(
                            f"\n🛑 Episode {episode_num} ended by user. Recorded {recorded_timesteps} timesteps ({step_count} control steps).")
                        self.is_collecting_flg = False  # Stop collecting data
                        break
                else:
                    both_buttons_start_time = None  # Reset if buttons released
                    episode_ending = False  # Resume recording if buttons released

                # 3. Get the desired action from the policy
                action = self.robot.policy.get_action(controller_state, current_ee_pose)


                # 5. Send direct button states for gripper control (only if not ending episode)
                if not (button_left and button_right):
                    self.robot.set_gripper_button_state(
                        button_left, button_right)

                # 6. Record the timestep (much less frequently to maintain 10Hz control)
                # Record every 50th iteration to get ~0.2Hz recording rate at 10Hz control
                # This ensures control loop stays fast while still capturing essential data
                if not episode_ending:
                    try:
                        if self.first_record_flg:
                            print("📁 Starting new episode recording...")
                            self.first_record_flg = False
                        # Get camera images from shared memory (fast, non-blocking)
                        img_primary, img_wrist, depth_primary, depth_wrist = self.get_camera_images()

                        # Use threading to make recording non-blocking
                        recording_thread = threading.Thread(
                            target=self.recorder.record_timestep_with_images,
                            args=(current_obs, action,
                                  language_instruction, img_primary, img_wrist, depth_primary, depth_wrist),
                            daemon=True
                        )
                        recording_thread.start()
                    except Exception as e:
                        print(f"⚠️ Recording error (continuing): {e}")

                step_count += 1  # Always increment step count

                # Monitor control loop performance less frequently to reduce overhead
                # Check every 50 steps (~5 seconds at 10Hz) to reduce console spam
                if step_count % 50 == 0 and step_count > 0:
                    actual_hz = self.rate_limiter.get_actual_rate()
                    recorded_timesteps = step_count

                    # Check if rate is within bounds (9.7-10.3Hz)
                    if not self.rate_limiter.is_rate_stable():
                        if actual_hz > 10.3:
                            print(
                                f"⚠️ Rate too high: {actual_hz:.2f}Hz > 10.3Hz")
                        elif actual_hz < 9.7:
                            print(
                                f"⚠️ Rate too low: {actual_hz:.2f}Hz < 9.7Hz")
                    print(
                        f"📊 Episode {episode_num}: {recorded_timesteps} recorded timesteps ({step_count} control steps) | Control loop: {actual_hz:.2f}Hz")

                # Maintain precise 10Hz control loop using Rate class
                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            recorded_timesteps = step_count  # Actual recorded data points
            print(
                f"\n🛑 Episode {episode_num} interrupted by Ctrl+C. Recorded {recorded_timesteps} timesteps ({step_count} control steps).")

        print(f"📁 Saved to: {episode_dir}")
        # Clear any residual control signals after episode
        self._clear_control_signals()
        recorded_timesteps = step_count  # Return actual recorded timesteps

        # Clear robot motion deltas to ensure no residual commands

        return recorded_timesteps

    def _reset_for_new_episode(self):
        """Reset robot state and clear control signals for new episode."""
        # Clear any existing movement commands
        self._clear_control_signals()

        # Reset recorder state (clears action history)
        self.recorder.reset_episode_state()

        # For episode 1, no need to reset (already at home)
        # For subsequent episodes, robot position reset is handled in the inter-episode wait
        print("🏠 Preparing for new episode...")


        # Restart controller process if it died during Ctrl+C
        if not self.robot.controller.is_alive():
            print("🔄 Restarting controller process...")
            self.robot.controller.stop()
            time.sleep(0.5)
            self.robot.controller.start()

        # Clear all movement deltas (real-time safe)
        self.robot.set_movement_delta([0, 0, 0], [0, 0, 0])

        # Open gripper
        self.robot.open_gripper()

        # Brief pause to let robot settle
        time.sleep(0.5)  # Much shorter wait

        print("✅ Episode reset complete")

    def _clear_control_signals(self):
        """Clear all control signals and movement commands."""
        # Clear robot movement deltas
        if self.robot:
            self.robot.set_movement_delta([0, 0, 0], [0, 0, 0])
        print("🧹 Control signals cleared")

    def run(self):
        

        # Reset robot to a neutral position and open gripper
        print("------Resetting robot to initial pose...------")

        # This is OK during initial setup before real-time control starts
        self.robot.reset_joints()
        self.robot.open_gripper()

        # Start real-time control loop
        print("------Starting real-time control loop...------")
        self.robot.start_realtime_control()

        try:
            # Get language instruction and number of episodes
            print("\n" + "="*60)
            print("🤖 MULTI-EPISODE DATA COLLECTION SETUP")
            print("="*60)

            # language_instruction = input(
            #     "📝 Enter language instruction for demonstrations: ")
            # while not language_instruction.strip():
            #     language_instruction = input(
            #         "📝 Please enter a valid instruction: ")
            language_instruction = config.LANGUAGE_INSTRUCTION
            print(f"📜 Using instruction: '{language_instruction}'")
            num_episodes = input("🔢 Enter number of episodes to collect: ")
            while True:
                try:
                    num_episodes = int(num_episodes)
                    if num_episodes > 0:
                        break
                    else:
                        num_episodes = input(
                            "📝 Please enter a positive number: ")
                except ValueError:
                    num_episodes = input("📝 Please enter a valid number: ")

            print(
                f"\n🎯 Will collect {num_episodes} episodes with instruction: '{language_instruction}'")

            total_timesteps = 0

            for episode_num in range(1, num_episodes + 1):
                # Wait between episodes (except for the first one)
                if episode_num > 1:
                    print(
                        f"\n⏳ Waiting 10 seconds before episode {episode_num}...")
                    print("🏠 Moving robot to initial position...")

                    # Actually move robot to initial position using safe reset
                    try:
                        self.robot.safe_episode_reset()
                        self.robot.open_gripper()
                        print("✅ Robot moved to initial position")
                    except Exception as e:
                        print(f"⚠️ Could not reset robot position: {e}")
                        # Fallback to simple movement clearing
                        self.robot.set_movement_delta([0, 0, 0], [0, 0, 0])
                        self.robot.open_gripper()

                    # self.robot._initialize_robot()
                    # self.robot.start_realtime_control()



                    for i in range(10, 0, -1):
                        print(f"   Starting in {i} seconds...", end="\r")
                        time.sleep(1)
                    print("   Starting now!                ")

                # Collect the episode
                timesteps = self.collect_episode(
                    language_instruction, episode_num, num_episodes)
                total_timesteps += timesteps

                # Short pause before next episode
                if episode_num < num_episodes:
                    time.sleep(2)

            print("\n🎉 ALL EPISODES COMPLETED!")
            print(f"📊 Total timesteps collected: {total_timesteps}")
            print(
                f"📁 Data saved to experiment directory: {self.recorder.experiment_dir} 📁")

        except KeyboardInterrupt:
            print("\n🛑 Data collection interrupted.")
        except Exception as e:
            print(f"\n❌ Error during data collection: {e}")
        finally:
            print("🔄 Shutting down...")
            self.robot.controller.stop()
            self.recorder.cleanup()
            self._cleanup_cameras()
            self.robot.end()
            print("✅ Shutdown complete.")

    def _cleanup_cameras(self):
        """Clean up shared memory cameras."""
        try:
            if hasattr(self, 'camera_stop_event'):
                self.camera_stop_event.set()

            if hasattr(self, 'camera_process'):
                self.camera_process.join(timeout=5.0)
                if self.camera_process.is_alive():
                    print("⚠️ Camera process still alive. Terminating forcefully...")
                    self.camera_process.terminate()
                    self.camera_process.join()

            # Clean up shared memory
            for shm_attr in ['shm_primary', 'shm_wrist', 'shm_primary_depth', 'shm_wrist_depth']:
                shm = getattr(self, shm_attr, None)
                if shm:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception as e:
                        print(f"⚠️ Error cleaning {shm_attr}: {e}")

            print("📸 Camera resources cleaned up")

        except Exception as e:
            print(f"⚠️ Camera cleanup error: {e}")

def main():
    # Use config defaults if not provided
    data_collecter = DataCollectionController(
        robot_ip=config.ROBOT_IP,
        data_log_root=config.DATA_ROOT_DIR,
        loop_rate_hz=config.DATA_LOOP_RATE_HZ,
        robot_loop_rate_hz=config.ROBOT_LOOP_RATE_HZ,
        primary_camera_serial=config.CAMERA_PRIMARY_SERIAL,
        wrist_camera_serial=config.CAMERA_WRIST_SERIAL,
        home_pose=config.HOME_POSE,
        gripper_open_width=config.GRIPPER_OPEN_WIDTH,
        gripper_closed_width=config.GRIPPER_CLOSED_WIDTH
    )
    data_collecter.run()


if __name__ == "__main__":
    main()
