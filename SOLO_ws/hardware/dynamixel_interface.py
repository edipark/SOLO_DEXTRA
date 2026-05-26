"""Dynamixel AX-18A interface via U2D2 adapter (Protocol 1.0).

Handles serial communication with 12 Dynamixel AX-18A servos for the
DEXTRA lower-body robot. Provides position reading and writing with
raw ↔ radian conversion and per-servo calibration offsets.

AX-18A specs:
  - Protocol 1.0 (no Sync Read support)
  - Position range: 0–1023 (≈300°)
  - Baudrate: up to 1 Mbps
  - Resolution: ~0.29°/unit
"""

import numpy as np

from dynamixel_sdk import (
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
    COMM_SUCCESS,
    DXL_LOBYTE,
    DXL_HIBYTE,
)

# AX-18A Control Table addresses (Protocol 1.0)
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
ADDR_PRESENT_TEMPERATURE = 43
ADDR_MOVING = 46

LEN_GOAL_POSITION = 2
LEN_PRESENT_POSITION = 2

# AX-18A position conversion
# 300° operational range mapped to 0–1023
TOTAL_RANGE_DEG = 300.0
TOTAL_RANGE_RAD = np.radians(TOTAL_RANGE_DEG)  # 5.23599 rad
MAX_POSITION_RAW = 1023
RAD_PER_UNIT = TOTAL_RANGE_RAD / MAX_POSITION_RAW  # ≈0.005115 rad/unit


class DynamixelInterface:
    """Interface for 12 Dynamixel AX-18A servos via U2D2."""

    def __init__(self, port: str, baudrate: int, servo_ids: list[int],
                 offsets_raw: list[int], lower_rads: list[float],
                 upper_rads: list[float]):
        """
        Args:
            port: Serial port path (e.g., "/dev/ttyUSB0").
            baudrate: Communication baudrate (e.g., 1000000).
            servo_ids: List of 12 servo IDs in joint order.
            offsets_raw: Per-servo raw position at zero radians (calibration).
            lower_rads: Per-joint lower limit in radians.
            upper_rads: Per-joint upper limit in radians.
        """
        self.port_path = port
        self.baudrate = baudrate
        self.servo_ids = servo_ids
        self.num_joints = len(servo_ids)
        self.offsets_raw = np.array(offsets_raw, dtype=np.int32)
        self.lower_rads = np.array(lower_rads, dtype=np.float32)
        self.upper_rads = np.array(upper_rads, dtype=np.float32)

        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(1.0)  # Protocol 1.0
        self.sync_write = None
        self._connected = False

    def connect(self) -> bool:
        """Open port, set baudrate, ping all servos, enable torque."""
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port: {self.port_path}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.baudrate}")

        # Ping all servos
        failed = []
        for sid in self.servo_ids:
            _, result, error = self.packet_handler.ping(self.port_handler, sid)
            if result != COMM_SUCCESS:
                failed.append(sid)

        if failed:
            raise RuntimeError(f"Failed to ping servos: {failed}")

        print(f"[dxl] All {self.num_joints} servos responding on {self.port_path}")

        # Check temperature
        for sid in self.servo_ids:
            temp, _, _ = self.packet_handler.read1ByteTxRx(
                self.port_handler, sid, ADDR_PRESENT_TEMPERATURE)
            if temp > 65:
                print(f"[dxl] WARNING: Servo {sid} temperature = {temp}°C (high!)")

        # Enable torque
        for sid in self.servo_ids:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, sid, ADDR_TORQUE_ENABLE, 1)

        # Setup sync write for goal position
        self.sync_write = GroupSyncWrite(
            self.port_handler, self.packet_handler,
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        self._connected = True
        print(f"[dxl] Torque enabled, ready.")
        return True

    def disconnect(self):
        """Disable torque on all servos and close port."""
        if not self._connected:
            return
        for sid in self.servo_ids:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, sid, ADDR_TORQUE_ENABLE, 0)
        self.port_handler.closePort()
        self._connected = False
        print("[dxl] Torque disabled, port closed.")

    def read_positions(self) -> np.ndarray:
        """Read current position of all joints.

        Returns:
            np.ndarray of shape (12,) in radians.

        Note: AX-18A does not support Sync Read, so we read individually.
        At 1Mbps, 12 reads take ~6ms which fits within 60Hz budget.
        """
        positions = np.zeros(self.num_joints, dtype=np.float32)
        for i, sid in enumerate(self.servo_ids):
            raw, result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, sid, ADDR_PRESENT_POSITION)
            if result != COMM_SUCCESS:
                print(f"[dxl] Read failed for servo {sid}: "
                      f"{self.packet_handler.getTxRxResult(result)}")
                continue
            positions[i] = self._raw_to_rad(raw, i)
        return positions

    def write_position_targets(self, targets_rad: np.ndarray):
        """Write position targets to all joints via Sync Write.

        Args:
            targets_rad: np.ndarray of shape (12,) in radians.
        """
        # Safety clipping
        targets_rad = np.clip(targets_rad, self.lower_rads, self.upper_rads)

        self.sync_write.clearParam()
        for i, sid in enumerate(self.servo_ids):
            raw = self._rad_to_raw(targets_rad[i], i)
            raw = int(np.clip(raw, 0, MAX_POSITION_RAW))
            param = [DXL_LOBYTE(raw), DXL_HIBYTE(raw)]
            self.sync_write.addParam(sid, param)

        result = self.sync_write.txPacket()
        if result != COMM_SUCCESS:
            print(f"[dxl] Sync write failed: "
                  f"{self.packet_handler.getTxRxResult(result)}")

    def go_to_home(self):
        """Move all joints to zero position (calibration offset)."""
        self.write_position_targets(np.zeros(self.num_joints, dtype=np.float32))

    def _raw_to_rad(self, raw: int, joint_idx: int) -> float:
        """Convert raw servo position to radians (centered at offset)."""
        return (raw - self.offsets_raw[joint_idx]) * RAD_PER_UNIT

    def _rad_to_raw(self, rad: float, joint_idx: int) -> int:
        """Convert radians to raw servo position."""
        return int(round(rad / RAD_PER_UNIT + self.offsets_raw[joint_idx]))
