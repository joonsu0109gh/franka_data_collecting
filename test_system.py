"""
Simple test script to check if panda_py is installed and can connect to the robot.
Run this before using the main data collection script.
"""


def test_panda_py_installation():
    """Test if panda_py is properly installed."""
    print("Testing panda_py installation...")

    try:
        import panda_py
        print("✅ panda_py is installed")
        return True
    except ImportError as e:
        print("❌ panda_py is not installed")
        print(f"Error: {e}")
        print("\nTo install panda_py, run:")
        print("pip install panda_py")
        return False


def test_robot_connection(robot_ip="172.16.0.2"):
    """Test connection to the robot."""
    print(f"\nTesting robot connection to {robot_ip}...")

    if not test_panda_py_installation():
        return False

    try:
        import panda_py
        from panda_py import libfranka

        # Test basic connection
        panda = panda_py.Panda(hostname=robot_ip)
        gripper = libfranka.Gripper(robot_ip)

        # Try to get robot state
        state = panda.get_state()
        gripper_state = gripper.read_once()

        print("✅ Successfully connected to robot")
        print(f"✅ Robot joint positions: {state.q}")
        print(f"✅ Gripper width: {gripper_state.width:.3f}m")
        return True

    except Exception as e:
        print("❌ Failed to connect to robot")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if robot is powered on")
        print(f"2. Check if robot IP ({robot_ip}) is correct")
        print("3. Check network connection")
        print("4. Check if robot is in the correct mode")
        return False


def test_spacemouse():
    """Test SpaceMouse connection."""
    print("\nTesting SpaceMouse connection...")

    try:
        import pyspacemouse
        success = pyspacemouse.open()
        if success:
            print("✅ SpaceMouse connected successfully")

            # Test reading a few samples
            print("Testing SpaceMouse input (move the device)...")
            for i in range(5):
                state = pyspacemouse.read()
                if state:
                    print(
                        f"Sample {i+1}: pos=({state.x:.2f}, {state.y:.2f}, {state.z:.2f}), buttons={state.buttons}")
                import time
                time.sleep(0.2)

            pyspacemouse.close()
            return True
        else:
            print("❌ Failed to connect to SpaceMouse")
            return False

    except ImportError:
        print("❌ pyspacemouse is not installed")
        print("To install pyspacemouse, run:")
        print("pip install pyspacemouse")
        return False
    except Exception as e:
        print(f"❌ SpaceMouse error: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Panda Data Collection - System Test")
    print("=" * 50)

    all_good = True

    # Test panda_py installation
    if not test_panda_py_installation():
        all_good = False

    # Test robot connection (only if panda_py is available)
    if all_good:
        if not test_robot_connection():
            all_good = False

    # Test SpaceMouse
    if not test_spacemouse():
        all_good = False

    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All tests passed! You're ready to collect data.")
        print("Run: python collect_data_final.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")

    return all_good


if __name__ == "__main__":
    main()
