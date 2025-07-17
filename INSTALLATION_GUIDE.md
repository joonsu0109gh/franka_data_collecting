# Installation Guide for Panda Data Collection

## Prerequisites

1. **Python 3.8+** 
2. **Franka Panda Robot** with network connection
3. **SpaceMouse** device

## Step-by-Step Installation

### 1. Install Required Python Packages

```bash
# Install basic dependencies
pip install numpy scipy h5py pyspacemouse

# Install panda_py (main robot interface)
pip install panda_py
```

### 2. Test Your Installation

```bash
# Test if everything is properly installed
python test_system.py
```

This will check:
- ✅ panda_py installation
- ✅ Robot connection
- ✅ SpaceMouse connection

### 3. Configure Robot Settings

Edit `config.py` to match your robot setup:

```python
# Robot Configuration
ROBOT_IP = "172.16.0.2"  # Change this to your robot's IP

# Control Parameters
TRANSLATION_SCALE = 0.005  # Adjust sensitivity
ROTATION_SCALE = 0.5       # Adjust sensitivity
LOOP_RATE_HZ = 10          # Control frequency
```

### 4. Run Data Collection

```bash
# Start data collection with default settings
python collect_data_final.py

# Or with custom parameters
python collect_data_final.py --robot-ip 192.168.1.100 --rate 20
```

## Troubleshooting

### Robot Connection Issues

**Error: "Failed to connect to robot"**

1. Check robot power and network
2. Verify robot IP address
3. Ensure robot is in the correct mode
4. Check firewall settings

### panda_py Installation Issues

**Error: "No module named 'panda_py'"**

```bash
# Try installing from different sources
pip install --upgrade panda_py

# Or install from conda
conda install -c conda-forge panda_py
```

### SpaceMouse Issues

**Error: "Failed to connect to SpaceMouse"**

1. Check USB connection
2. Try different USB port
3. Check device permissions (Linux):
   ```bash
   sudo chmod 666 /dev/input/event*
   ```

## Robot Setup Requirements

### Network Configuration
- Robot and computer must be on the same network
- Robot must have a static IP address
- Firewall must allow communication on ports 1000-1016

### Robot Mode
- Robot should be in "Programming" mode
- User stop should be released
- Robot should be homed and calibrated

### Safety
- Always keep emergency stop within reach
- Ensure clear workspace around robot
- Start with low sensitivity settings

## File Structure After Installation

```
panda_datacollect/
├── src/                    # Modular components
├── my_robot_data/         # Data storage (created automatically)
├── collect_data_final.py  # Main script
├── config.py              # Configuration
├── test_system.py         # System testing
└── ...
```

## Getting Help

1. **Test your setup**: `python test_system.py`
2. **Check configuration**: Review `config.py`
3. **Verify robot connection**: Ping robot IP
4. **Check logs**: Look for error messages in terminal output

## Quick Commands

```bash
# Test everything
python test_system.py

# Collect data with default settings
python collect_data_final.py

# Collect data with custom robot IP
python collect_data_final.py --robot-ip 192.168.1.100

# Test individual components
python test_components.py
```
