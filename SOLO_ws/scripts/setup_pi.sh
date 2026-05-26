#!/usr/bin/env bash
# SOLO_ws setup script for Raspberry Pi 5
set -e

echo "=== SOLO Deployment Setup ==="

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Ensure user is in dialout group for serial access
if ! groups | grep -q dialout; then
    echo "Adding user to 'dialout' group for USB serial access..."
    sudo usermod -aG dialout "$USER"
    echo "NOTE: Log out and back in for group change to take effect."
fi

# U2D2 udev rule — gives non-root access to FTDI USB serial
UDEV_RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", MODE="0666", SYMLINK+="ttyU2D2"'
UDEV_FILE="/etc/udev/rules.d/99-u2d2.rules"
if [ ! -f "$UDEV_FILE" ]; then
    echo "Installing U2D2 udev rule..."
    echo "$UDEV_RULE" | sudo tee "$UDEV_FILE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "Udev rule installed: $UDEV_FILE"
else
    echo "Udev rule already exists: $UDEV_FILE"
fi

# Create models directory
mkdir -p models

echo "=== Setup complete ==="
echo "Place ONNX models in ./models/ directory:"
echo "  models/teacher_policy.onnx"
echo "  models/lstm_estimator.onnx"
