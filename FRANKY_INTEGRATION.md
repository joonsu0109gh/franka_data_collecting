# Franky Velocity Control Integration Guide

## Overview
This guide documents the conversion from `panda_py` to `franky` library with cartesian velocity control support, following the reference implementation pattern.

## Key Changes Made

### 1. Robot Wrapper Class (`FrankaRobotWrapper`)
Created a wrapper class that provides the `cartesian_velocity_control()` method compatible with the reference code:

```python
class FrankaRobotWrapper:
    def cartesian_velocity_control(self, velocity_cmd):
        """
        velocity_cmd format:
        {
            "x": 0.01,      # linear velocity in m/s
            "y": 0.0,
            "z": 0.0,
            "R": 0.0,       # angular velocity in rad/s
            "P": 0.0,
            "Y": 0.0,
            "duration": 100, # motion duration in ms
            "is_async": True # asynchronous execution
        }
        """
```

### 2. Updated Files

#### `robot.py` (FrankaAPI):
- Replaced `panda_py` imports with `franky`
- Added `FrankaRobotWrapper` class
- Updated control loop to use velocity commands
- Fixed gripper state access

#### `src/robot_interface.py` (RealTimeRobotInterface):
- Added `FrankaRobotWrapper` class
- Updated real-time control loop
- Modified `step()` method to handle velocity commands
- Updated robot state access

#### `src/policy.py` (TeleopPolicy):
- Added `velocity_cmd` to action output
- Maintained compatibility with existing delta-based control

### 3. Velocity Command Format
The velocity control system now accepts commands in this format (compatible with reference code):

```python
velocity_cmd = {
    "x": 0.01,      # m/s - forward/backward
    "y": 0.0,       # m/s - left/right  
    "z": 0.0,       # m/s - up/down
    "R": 0.0,       # rad/s - roll
    "P": 0.0,       # rad/s - pitch
    "Y": 0.0,       # rad/s - yaw
    "duration": 100, # ms - motion duration
    "is_async": True # async execution
}
```

### 4. Usage Examples

#### Direct velocity control:
```python
robot = FrankaRobotWrapper("172.16.0.2")
robot.cartesian_velocity_control({
    "x": 0.02, "y": 0.0, "z": 0.0,
    "R": 0.0, "P": 0.0, "Y": 0.0,
    "duration": 1000, "is_async": True
})
```

#### Through robot interface:
```python
action = {
    'velocity_cmd': {
        "x": 0.01, "y": 0.0, "z": 0.0,
        "R": 0.0, "P": 0.0, "Y": 0.0,
        "duration": 100, "is_async": True
    }
}
robot_interface.step(action)
```

#### Through policy:
```python
policy = TeleopPolicy()
action = policy.get_action(mouse_state, current_pose)
# action now contains 'velocity_cmd' key
robot_interface.step(action)
```

## Key Differences from panda_py

### Control Method:
- **panda_py**: High-frequency context (1000Hz) with direct controller access
- **franky**: Motion-based commands with velocity control

### State Access:
- **panda_py**: `panda.get_state()`, `panda.get_position()`
- **franky**: `robot.state.q`, `robot.state.O_T_EE.translation`

### Gripper:
- **panda_py**: `gripper.read_once().width`
- **franky**: `gripper.width`

### Movement:
- **panda_py**: `controller.set_control(translation, rotation)`
- **franky**: `robot.cartesian_velocity_control(velocity_cmd)`

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure `franky` is installed: `pip install franky`

2. **Robot connection**: Ensure robot IP is correct (default: "172.16.0.2")

3. **Velocity scaling**: Adjust velocity values if robot moves too fast/slow:
   - Linear: typically 0.001 - 0.05 m/s
   - Angular: typically 0.1 - 1.0 rad/s

4. **Duration settings**: 
   - Shorter duration (50-100ms) for responsive control
   - Longer duration (500-1000ms) for smooth movements

### Testing:
Use `test_velocity_control.py` to verify the integration works correctly.

## Migration Notes

The new system maintains backward compatibility while adding velocity control support. Existing code using delta-based control will continue to work, but velocity commands provide better performance for real-time teleoperation.
