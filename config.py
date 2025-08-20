"""
Configuration file for the data collection system.
Modify these parameters to customize the behavior.
"""

# Robot Configuration
ROBOT_IP = "172.16.0.2"  # IP address of the Franka robot
HOME_POSE = [-0.0026, -0.7855, 0.0011, -2.3576, 0.0038, 1.5738, 0.7780]  # Joint angles for home position
LANGUAGE_INSTRUCTION = "Place the green block on the white plate"  # Default language instruction for demonstrations
# Control direction
CONTROL_INVERSE_MODE = True  # To control the robot face to face the robot
CONTROLLER_TYPE = "xbox"  # Type of controller to use, options: "spacemouse",xbox"

# Update rate
ROBOT_LOOP_RATE_HZ = 100  # Robot control update rate in Hz
DATA_LOOP_RATE_HZ = 5          # Control loop frequency

# Data Storage
DATA_ROOT_DIR = "./my_robot_data"  # Where to save collected data

# SpaceMouse Settings
SPACEMOUSE_DEADZONE = 0.05  # Reduced deadzone for more sensitive input
SPACEMOUSE_TRANSLATION_SCALE = 10.0    # Increased for more responsive SpaceMouse input
SPACEMOUSE_ROTATION_SCALE = 20.0       # Increased for more responsive rotation
SPACEMOUSE_MAX_TRANSLATION_DELTA = 0.05   # Maximum translation per step (meters)
SPACEMOUSE_MAX_ROTATION_DELTA = 0.1       # Maximum rotation per step (radians)

# Xbox Controller Settings
XBOX_DEADZONE = 0.1  # Reduced deadzone for more sensitive input
XBOX_TRANSLATION_SCALE = 3.5    # Increased for more responsive Xbox input
XBOX_ROTATION_SCALE = 100.0       # Increased for more responsive rotation
XBOX_MAX_TRANSLATION_DELTA = 0.05   # Maximum translation per step (meters)
XBOX_MAX_ROTATION_DELTA = 0.8       # Maximum rotation per step (
XBOX_CONTROLLER_LOOP_RATE_HZ = 50.0  # Frequency of Xbox controller updates in Hz

# Gripper Settings
GRIPPER_TOGGLE_DEBOUNCE = 0.25  # Minimum time between gripper toggles (seconds)
GRIPPER_OPEN_WIDTH = 0.08      # Target width when gripper is open
GRIPPER_CLOSED_WIDTH = 0.02     # Target width when gripper is closed

# Camera Configuration
CAMERA_PRIMARY_SERIAL =  "238722070845" # Serial number of the primary camera
CAMERA_WRIST_SERIAL = "238222073927"    # Serial number of the wrist
CAMERA_FPS = 30  # Frames per second for camera capture
CAMERA_INITIALIZATION_DELAY = 20.0  # Delay to allow cameras to initialize (seconds)
