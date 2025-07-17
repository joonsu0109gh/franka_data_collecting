# Refactored Panda Data Collection

This project has been refactored from a monolithic script into a clean, modular, class-based structure. The new architecture separates concerns and makes the code more maintainable and extensible.

## Project Structure

```
panda_datacollect/
├── src/                           # Modular components
│   ├── __init__.py               # Package initialization
│   ├── input_manager.py          # SpaceMouse input handling
│   ├── policy.py                 # Input-to-action conversion
│   ├── recorder.py               # Data recording and saving
│   └── robot_interface.py        # Unified robot interface
├── franka/                       # Your existing robot utilities
├── data_collect.py              # Original monolithic script (for reference)
├── collect_data_final.py        # NEW: Main data collection script
├── robot.py                     # Your existing robot implementation
└── config.py                    # Configuration settings
```

## Key Components

### 1. SpaceMouseManager (`src/input_manager.py`)
- Handles SpaceMouse input in a separate process
- Non-blocking input reading
- Thread-safe state management

### 2. TeleopPolicy (`src/policy.py`)
- Converts raw SpaceMouse input to robot actions
- Configurable translation and rotation scaling
- Gripper control with button debouncing

### 3. DataRecorder (`src/recorder.py`)
- Manages trajectory data buffering
- Creates unique timestamped directories
- Saves data to HDF5 format

### 4. RobotInterface (`src/robot_interface.py`)
- Unified interface for different robot implementations
- Supports both FrankaAPI and PandaRealRobot
- Automatic fallback between robot types

### 5. DataCollectionController (`collect_data_final.py`)
- Orchestrates the entire data collection process
- Clean, readable main loop
- Proper resource cleanup

## Usage

### Running the Main Script

```bash
python collect_data_final.py
```

### Customizing Parameters

You can customize the behavior by modifying the controller initialization:

```python
# Custom parameters
controller = DataCollectionController(
    data_log_root="./custom_data_folder",
    robot_ip="192.168.1.100"
)

# Custom policy parameters
controller.policy = TeleopPolicy(
    translation_scale=0.01,  # Increase sensitivity
    rotation_scale=1.0
)

# Custom loop rate
controller.loop_rate_hz = 20  # 20Hz instead of 10Hz
```

## Benefits of Refactored Architecture

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Components can be used in other projects
3. **Testability**: Each class can be tested independently
4. **Maintainability**: Changes to one component don't affect others
5. **Extensibility**: Easy to add new input devices or robot types
6. **Readability**: Code structure matches the conceptual flow

## Migration from Original Script

The original `data_collect.py` script is preserved. The new refactored version provides the same functionality with these improvements:

- **Cleaner separation of concerns**
- **Better error handling**
- **More configurable parameters**
- **Unified robot interface**
- **Improved data organization**

## Future Extensions

The modular structure makes it easy to:

- Add new input devices (joystick, keyboard, etc.)
- Support different robot types
- Implement different control policies
- Add real-time data visualization
- Integrate with learning algorithms

## Requirements

The refactored code uses the same dependencies as the original:
- pyspacemouse
- h5py
- numpy
- scipy
- multiprocessing (built-in)

Additional robot-specific dependencies depend on which robot interface you use.
