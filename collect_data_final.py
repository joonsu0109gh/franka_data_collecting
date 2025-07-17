import time
import config

# Import our new modular classes
from src.input_manager import SpaceMouseManager
from src.policy import TeleopPolicy
from src.recorder import DataRecorder
from src.robot_interface import RobotInterface


class DataCollectionController:
    """Orchestrates the entire data collection process."""

    def __init__(self,
                 data_log_root=None,
                 robot_ip=None,
                 translation_scale=None,
                 rotation_scale=None,
                 loop_rate_hz=None):

        # Use config file values as defaults
        self.data_log_root = data_log_root or config.DATA_ROOT_DIR
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.loop_rate_hz = loop_rate_hz or config.LOOP_RATE_HZ

        print("Initializing controller...")
        print(f"Robot IP: {self.robot_ip}")
        print(f"Data directory: {self.data_log_root}")
        print(f"Loop rate: {self.loop_rate_hz} Hz")

        # Initialize components
        self.robot = RobotInterface(robot_ip=self.robot_ip)
        self.mouse = SpaceMouseManager()
        self.policy = TeleopPolicy(
            translation_scale=translation_scale or config.TRANSLATION_SCALE,
            rotation_scale=rotation_scale or config.ROTATION_SCALE
        )
        self.recorder = DataRecorder(self.data_log_root)

    def run(self):
        """Starts the data collection loop."""
        print("Starting SpaceMouse input process...")
        self.mouse.start()

        # Reset robot to a neutral position and open gripper
        print("Resetting robot to initial pose...")
        self.robot.reset_joints()
        self.robot.open_gripper()

        # Start real-time control loop
        print("Starting real-time control loop...")
        self.robot.start_realtime_control()

        print("\n" + "="*60)
        print("🤖 ROBOT DATA COLLECTION READY")
        print("="*60)
        print("Controls:")
        print("  • Move SpaceMouse: Control robot position")
        print("  • Left button: Close gripper")
        print("  • Right button: Open gripper")
        print("  • Ctrl+C: Stop collection and save data")
        print("="*60 + "\n")

        step_count = 0
        try:
            while True:
                start_time = time.time()

                # 1. Get user input
                mouse_state = self.mouse.get_state()
                if mouse_state is None:
                    time.sleep(1 / self.loop_rate_hz)
                    continue

                # 2. Get current robot state (Observation)
                current_obs = self.robot.get_obs()
                current_ee_pose = current_obs['panda_hand_pose']

                # 3. Get the desired action from the policy
                action = self.policy.get_action(mouse_state, current_ee_pose)

                # 4. Execute the action on the robot (real-time)
                self.robot.step(action)

                # Also send direct button states for gripper control (like reference)
                self.robot.set_gripper_button_state(
                    mouse_state.buttons[0], mouse_state.buttons[1])

                # 5. Record the observation and the action
                # We get a fresh observation after the step to record the *result* of the action
                new_obs = self.robot.get_obs()
                self.recorder.record_step(new_obs, action)

                step_count += 1
                if step_count % 50 == 0:  # Print status every 5 seconds at 10Hz
                    print(f"📊 Recorded {step_count} steps...")

                # Maintain a consistent loop rate
                elapsed_time = time.time() - start_time
                sleep_time = max(0, (1 / self.loop_rate_hz) - elapsed_time)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(
                f"\n🛑 Keyboard interrupt detected. Collected {step_count} steps.")
        except Exception as e:
            print(f"\n❌ Error during data collection: {e}")
        finally:
            print("🔄 Shutting down...")
            self.mouse.stop()
            self.recorder.save_to_hdf5()
            self.robot.end()
            print("✅ Shutdown complete.")


def main():
    """Main entry point with command line argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect robot demonstration data using SpaceMouse')
    parser.add_argument('--robot-ip', type=str, help='Robot IP address')
    parser.add_argument('--data-dir', type=str, help='Data storage directory')
    parser.add_argument('--rate', type=int, help='Control loop rate (Hz)')
    parser.add_argument('--translation-scale', type=float,
                        help='Translation sensitivity')
    parser.add_argument('--rotation-scale', type=float,
                        help='Rotation sensitivity')

    args = parser.parse_args()

    controller = DataCollectionController(
        robot_ip=args.robot_ip,
        data_log_root=args.data_dir,
        loop_rate_hz=args.rate,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale
    )
    controller.run()


if __name__ == "__main__":
    main()
