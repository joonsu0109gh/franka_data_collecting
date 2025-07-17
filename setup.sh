#!/bin/bash

# Installation script for the refactored panda data collection system

echo "🚀 Setting up Panda Data Collection Environment"
echo "=============================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ Python 3 and pip found"

# Install required packages
echo "📦 Installing Python dependencies..."

pip3 install --user numpy scipy h5py pyspacemouse

echo "✅ Dependencies installed"

# Create data directory
echo "📁 Creating data directory..."
mkdir -p ./my_robot_data
echo "✅ Data directory created at ./my_robot_data"

# Set up Python path (optional)
echo "🔧 Setting up Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the data collection:"
echo "  python3 collect_data_final.py"
echo ""
echo "To test components:"
echo "  python3 test_components.py"
echo ""
echo "To customize settings, edit config.py"
echo ""
echo "For help:"
echo "  python3 collect_data_final.py --help"
