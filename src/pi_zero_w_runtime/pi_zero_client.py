#!/usr/bin/env python3
"""Standalone ONNX-Free Runtime & HIL Client for Raspberry Pi Zero W (v1.1) + ESP32 DevKit V1.

Why this exists:
- The Raspberry Pi Zero W (v1.1) uses an ARMv6 32-bit CPU (BCM2835, armv6l, 512 MB RAM),
  which has no official `onnxruntime` wheels.
- This runtime loads the exported `.npz` policies (`models_npz/*.npz`) using ONLY `numpy`
  (`sudo apt install python3-numpy python3-serial`) and executes the full XGO-Duck
  100 Hz IMU filter + 50 Hz RL control loop (`walk`, `getup`, `kick`, `sit/stand`, `pick`, `roulade`).

Supported Hardware / Emulation Backends:
1. `--hil-udp <PC_IP>:9528`
   Streams 240-byte `DQS1` state packets and 80-byte `DQC1` command packets over Wi-Fi UDP
   to/from `xgoduck_sim_and_hil.py` running MuJoCo on your PC.
2. `--esp32-serial /dev/ttyUSB0` (or `COM3` on Windows)
   Streams framed `[0xA5, 0x5A, type, len, payload, crc8]` packets at 921600 baud to/from
   the ESP32 DevKit V1 running `esp32_xgoduck_bridge.ino` (works with real Feetech 1910
   servos OR with a bare ESP32 in desk-emulation mode!).
"""
from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SUITE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SUITE_DIR.parent
MODELS_NPZ_DIR = ROOT_DIR / "ml_checkpoints" / "models_npz"
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from numpy_policy import NumpyMlpGraph

# ===================== WIRE PROTOCOL (DQS1 / DQC1) =====================
STATE_STRUCT = struct.Struct("<4sBBH10I6h45f")    # 240 bytes
COMMAND_STRUCT = struct.Struct("<4sIIIBBH15f")    # 80 bytes

DISARM = 0
ARM = 1
POLICY = 2
HOLD = 3
ALL_MASK = (1 << 15) - 1

FRAME_SYNC_0 = 0xA5
FRAME_SYNC_1 = 0x5A
MSG_STATE_DQS1 = 0x01
MSG_CMD_DQC1 = 0x10

MOTOR_IDS = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34]
POLICY_IDS = [10, 11, 12, 13, 14, 30, 31, 32, 33, 20, 21, 22, 23, 24]
POLICY_INDEX = np.array([MOTOR_IDS.index(i) for i in POLICY_IDS])

DEFAULT_DEG = [
    0.0, -5.0, -22.0, 0.0, 22.0,
    0.0, 5.0, 22.0, 0.0, -22.0,
    20.0, 20.0, 0.0, 0.0, 0.0,
]
HOME_DEG = np.array(DEFAULT_DEG, dtype=np.float32)[POLICY_INDEX]
MOUTH_INDEX = MOTOR_IDS.index(34)
MOUTH_MAX_DEG = 30.0
HEAD_INDICES = (5, 6, 7, 8)


def crc8_xor(type_byte: int, length_byte: int, payload: bytes) -> int:
    c = (type_byte ^ length_byte) & 0xFF
    for b in payload:
        c ^= b
    return c & 0xFF


# ===================== IMU COMPLEMENTARY FILTER =====================
class PiZeroImuFilter:
    def __init__(self, gyro_alpha: float = 0.5, vel_alpha: float = 0.4, gravity_tau_s: float = 0.3):
        self.gyro_alpha = gyro_alpha
        self.vel_alpha = vel_alpha
        self.gravity_tau_s = gravity_tau_s
        self.grav = [0.0, 0.0, -1.0]
        self.gyro_dps = [0.0, 0.0, 0.0]
        self.vels_deg = [0.0] * 15
        self.initialized = False

    def update(self, acc_ms2: List[float], gyro_dps: List[float], vels_deg: List[float], dt: float = 0.01):
        dt = max(0.001, min(dt, 0.1))
        if not self.initialized:
            self.vels_deg = list(vels_deg)
            norm = math.sqrt(sum(v * v for v in acc_ms2))
            self.grav = [v / norm for v in acc_ms2] if norm > 1e-6 else [0.0, 0.0, -1.0]
            self.gyro_dps = list(gyro_dps)
            self.initialized = True
        else:
            va = self.vel_alpha ** (dt / 0.01)
            self.vels_deg = [va * o + (1.0 - va) * n for o, n in zip(self.vels_deg, vels_deg)]
            gx, gy, gz = self.grav
            wx, wy, wz = [math.radians(v) for v in gyro_dps]
            pred = [
                gx + (wz * gy - wy * gz) * dt,
                gy + (wx * gz - wz * gx) * dt,
                gz + (wy * gx - wx * gy) * dt,
            ]
            pnorm = math.sqrt(sum(v * v for v in pred)) or 1.0
            pred = [v / pnorm for v in pred]
            norm = math.sqrt(sum(v * v for v in acc_ms2))
            if norm > 1e-6:
                acc_unit = [v / norm for v in acc_ms2]
                trust = max(0.0, 1.0 - abs(norm / 9.8 - 1.0) / 0.2)
                gain = (1.0 - math.exp(-dt / self.gravity_tau_s)) * trust
                pred = [(1.0 - gain) * p + gain * a for p, a in zip(pred, acc_unit)]
                pnorm = math.sqrt(sum(v * v for v in pred)) or 1.0
                self.grav = [v / pnorm for v in pred]
            else:
                self.grav = pred
            ga = self.gyro_alpha ** (dt / 0.01)
            self.gyro_dps = [ga * o + (1.0 - ga) * n for o, n in zip(self.gyro_dps, gyro_dps)]

        gyro_rad = [math.radians(v) for v in self.gyro_dps]
        return self.grav, gyro_rad, self.vels_deg


# ===================== PURE-NUMPY POLICY WRAPPER =====================
class PiZeroPolicyWrapper:
    def __init__(self, npz_name: str, action_scale: float = 1.0, leg_alpha: float = 0.35, head_alpha: float = 0.45):
        npz_path = MODELS_NPZ_DIR / npz_name
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ policy: {npz_path}")
        self.name = npz_name
        self.graph = NumpyMlpGraph.from_npz(npz_path)
        self.home_deg = HOME_DEG.copy()
        self.action_scale = action_scale
        self.leg_alpha = leg_alpha
        self.head_alpha = head_alpha
        self.prev_action = np.zeros(14, dtype=np.float32)
        self.act_filt = np.zeros(14, dtype=np.float32)
        self.obs_buf = np.zeros(61, dtype=np.float32)

    def reset(self):
        self.prev_action.fill(0.0)
        self.act_filt.fill(0.0)

    def infer_targets_deg(
        self,
        grav: List[float],
        gyro_rad: List[float],
        angles_deg: List[float],
        vels_deg: List[float],
        twist: List[float],
        head: List[float],
        mouth_deg: float = 0.0,
    ) -> List[float]:
        q_deg = np.asarray(angles_deg, dtype=np.float32)[POLICY_INDEX]
        v_deg = np.asarray(vels_deg, dtype=np.float32)[POLICY_INDEX]
        self.obs_buf[0:3] = gyro_rad
        self.obs_buf[3:6] = grav
        self.obs_buf[6:20] = np.deg2rad(q_deg - self.home_deg)
        self.obs_buf[20:34] = np.deg2rad(v_deg)
        self.obs_buf[34:48] = self.prev_action
        self.obs_buf[48:51] = twist
        self.obs_buf[51:55] = head
        self.obs_buf[55:61] = 0.0

        raw = self.graph.forward(self.obs_buf[: self.graph.input_dim])
        self.prev_action = raw.copy()
        alphas = np.full(14, self.leg_alpha, dtype=np.float32)
        for idx in HEAD_INDICES:
            alphas[idx] = self.head_alpha
        self.act_filt = alphas * self.act_filt + (1.0 - alphas) * raw

        q_out_deg = self.home_deg + np.rad2deg(self.action_scale * self.act_filt)
        targets = np.zeros(15, dtype=np.float32)
        targets[POLICY_INDEX] = q_out_deg
        targets[MOUTH_INDEX] = float(np.clip(mouth_deg, 0.0, MOUTH_MAX_DEG))
        return targets.tolist()


# ===================== PI ZERO W RUNTIME NODE =====================
class PiZeroRuntimeNode:
    def __init__(self, udp_target: Optional[str] = None, serial_port: Optional[str] = None, baud: int = 921600):
        self.policies: Dict[str, PiZeroPolicyWrapper] = {
            "walk": PiZeroPolicyWrapper("xgoduck_walk.npz"),
            "getup": PiZeroPolicyWrapper("xgoduck_getup.npz"),
            "pick": PiZeroPolicyWrapper("xgoduck_pick.npz"),
            "roulade": PiZeroPolicyWrapper("xgoduck_roulade.npz", leg_alpha=0.15, head_alpha=0.15),
            "sitstand": PiZeroPolicyWrapper("xgoduck_sitstand.npz", action_scale=1.1),
            "kick_right": PiZeroPolicyWrapper("xgoduck_ballkick_right.npz", action_scale=0.8, leg_alpha=0.0, head_alpha=0.0),
            "kick_left": PiZeroPolicyWrapper("xgoduck_ballkick_left.npz", action_scale=0.8, leg_alpha=0.0, head_alpha=0.0),
        }
        self.imu_filter = PiZeroImuFilter()
        self.lock = threading.Lock()

        # State from DQS1
        self.rx_count = 0
        self.tx_count = 0
        self.last_sample_us = 0
        self.angles_deg = list(DEFAULT_DEG)
        self.vels_deg = [0.0] * 15
        self.grav = [0.0, 0.0, -1.0]
        self.gyro_rad = [0.0, 0.0, 0.0]
        self.imu_ok = False
        self.mcu_enabled = False

        # Active skill & command
        self.active_skill = "walk"
        self.skill_until = 0.0
        self.pick_phi = 0.0
        self.sit_flag = 0.0
        self.twist = [0.0, 0.0, 0.0]
        self.head = [0.0, 0.0, 0.0, 0.0]
        self.mouth = 0.0
        self.infer_ms_history: List[float] = []

        # Transport setup
        self.udp_sock: Optional[socket.socket] = None
        self.udp_addr: Optional[tuple[str, int]] = None
        self.ser = None

        if udp_target:
            host, port_s = udp_target.split(":")
            self.udp_addr = (host, int(port_s))
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setblocking(False)
            self.udp_sock.sendto(b"HELLO_HIL", self.udp_addr)
        elif serial_port:
            import serial  # type: ignore
            self.ser = serial.Serial(serial_port, baudrate=baud, timeout=0.0)
            self._rx_buf = bytearray()
        else:
            raise ValueError("Specify either --hil-udp <IP:PORT> or --esp32-serial <PORT>")

    def _handle_dqs1(self, payload: bytes) -> None:
        if len(payload) != STATE_STRUCT.size:
            return
        vals = STATE_STRUCT.unpack(payload)
        if vals[0] != b"DQS1":
            return
        flags = vals[2]
        sample_us = vals[5]
        acc_ms2 = [vals[14 + i] * 9.8 / 16384.0 for i in range(3)]
        gyro_dps = [vals[17 + i] / 32.0 for i in range(3)]
        angles = list(vals[20:35])
        vels = list(vals[35:50])

        with self.lock:
            dt = 0.01 if self.last_sample_us == 0 else ((sample_us - self.last_sample_us) & 0xFFFFFFFF) / 1e6
            self.last_sample_us = sample_us
            self.imu_ok = bool(flags & 1)
            self.mcu_enabled = bool(flags & 2)
            self.angles_deg = angles
            self.grav, self.gyro_rad, self.vels_deg = self.imu_filter.update(acc_ms2, gyro_dps, vels, dt)
            self.rx_count += 1

    def poll_transport(self) -> None:
        if self.udp_sock is not None:
            while True:
                try:
                    data, _ = self.udp_sock.recvfrom(512)
                except (BlockingIOError, OSError):
                    break
                if len(data) == STATE_STRUCT.size and data[:4] == b"DQS1":
                    self._handle_dqs1(data)
        elif self.ser is not None:
            chunk = self.ser.read(1024)
            if chunk:
                self._rx_buf.extend(chunk)
            while len(self._rx_buf) >= 5:
                if self._rx_buf[0] != FRAME_SYNC_0 or self._rx_buf[1] != FRAME_SYNC_1:
                    del self._rx_buf[0]
                    continue
                mtype = self._rx_buf[2]
                mlen = self._rx_buf[3]
                if len(self._rx_buf) < 4 + mlen + 1:
                    break
                payload = bytes(self._rx_buf[4 : 4 + mlen])
                crc = self._rx_buf[4 + mlen]
                del self._rx_buf[: 5 + mlen]
                if crc == crc8_xor(mtype, mlen, payload) and mtype == MSG_STATE_DQS1:
                    self._handle_dqs1(payload)

    def send_dqc1(self, mode: int, targets_deg: List[float]) -> None:
        self.tx_count += 1
        pkt = COMMAND_STRUCT.pack(
            b"DQC1",
            self.tx_count & 0xFFFFFFFF,
            self.last_sample_us & 0xFFFFFFFF,
            ALL_MASK,
            mode,
            0,
            0,
            *[float(x) for x in targets_deg],
        )
        if self.udp_sock is not None and self.udp_addr is not None:
            try:
                self.udp_sock.sendto(pkt, self.udp_addr)
            except OSError:
                pass
        elif self.ser is not None:
            crc = crc8_xor(MSG_CMD_DQC1, len(pkt), pkt)
            frame = bytes([FRAME_SYNC_0, FRAME_SYNC_1, MSG_CMD_DQC1, len(pkt)]) + pkt + bytes([crc])
            self.ser.write(frame)

    def trigger_skill(self, name: str) -> None:
        now = time.monotonic()
        with self.lock:
            self.policies[name if name in self.policies else "walk"].reset()
            if name in ("kick_right", "kick_left"):
                self.active_skill = name
                self.skill_until = now + 0.4
            elif name == "roulade":
                self.active_skill = "roulade"
                self.skill_until = now + 1.9
            elif name == "pick":
                self.active_skill = "pick"
                self.pick_phi = 0.0
            elif name == "sit":
                self.active_skill = "sitstand"
                self.sit_flag = 1.0
            elif name == "stand":
                self.active_skill = "sitstand"
                self.sit_flag = 0.0
                self.skill_until = now + 1.0
            else:
                self.active_skill = "walk"
                self.skill_until = 0.0

    def step_50hz(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.rx_count == 0:
                if self.udp_sock is not None and self.udp_addr is not None:
                    self.udp_sock.sendto(b"HELLO_HIL", self.udp_addr)
                return
            grav = list(self.grav)
            gyro_rad = list(self.gyro_rad)
            angles_deg = list(self.angles_deg)
            vels_deg = list(self.vels_deg)
            mcu_enabled = self.mcu_enabled

            # Automatic skill expiration
            if self.active_skill in ("kick_right", "kick_left", "roulade") and now >= self.skill_until:
                self.active_skill = "walk"
                self.policies["walk"].reset()
            elif self.active_skill == "sitstand" and self.sit_flag == 0.0 and self.skill_until > 0 and now >= self.skill_until:
                self.active_skill = "walk"
                self.skill_until = 0.0
                self.policies["walk"].reset()

            # Fall recovery check (unless doing roulade/somersault)
            if self.active_skill != "roulade" and grav[2] > -0.55:
                pol = self.policies["getup"]
                twist = [0.0, 0.0, 0.0]
                head = [0.0, 0.0, 0.0, 0.0]
                mouth = 0.0
            elif self.active_skill == "pick":
                pol = self.policies["pick"]
                angle = 2.0 * math.pi * self.pick_phi
                twist = [math.cos(angle), math.sin(angle), 0.0]
                head = [0.0, 0.0, 0.0, 0.0]
                mouth = 30.0 if self.pick_phi < 0.4 else 0.0
                self.pick_phi += (1.0 / 50.0) / 4.0
                if self.pick_phi >= 1.0:
                    self.active_skill = "walk"
                    self.policies["walk"].reset()
            elif self.active_skill == "sitstand":
                pol = self.policies["sitstand"]
                twist = [self.sit_flag, 0.0, 0.0]
                head = [0.0, 0.0, 0.0, 0.0]
                mouth = 0.0
            elif self.active_skill in ("kick_right", "kick_left", "roulade"):
                pol = self.policies[self.active_skill]
                twist = [0.0, 0.0, 0.0]
                head = [0.0, 0.0, 0.0, 0.0]
                mouth = 0.0
            else:
                pol = self.policies["walk"]
                twist = list(self.twist)
                head = list(self.head)
                mouth = self.mouth

        t0 = time.perf_counter()
        targets_deg = pol.infer_targets_deg(grav, gyro_rad, angles_deg, vels_deg, twist, head, mouth)
        t1 = time.perf_counter()
        self.infer_ms_history.append((t1 - t0) * 1000.0)
        if len(self.infer_ms_history) > 500:
            self.infer_ms_history.pop(0)

        cmd_mode = POLICY if mcu_enabled else ARM
        self.send_dqc1(cmd_mode, targets_deg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi Zero W (v1.1) Pure-NumPy XGO-Duck Runtime & HIL Client")
    parser.add_argument("--hil-udp", type=str, default="127.0.0.1:9528", help="PC MuJoCo HIL UDP <IP:PORT>")
    parser.add_argument("--esp32-serial", type=str, default=None, help="ESP32 Serial Port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--baud", type=int, default=921600, help="ESP32 Serial baudrate")
    parser.add_argument("--duration", type=float, default=0.0, help="Run for N seconds and exit (0 = infinite)")
    parser.add_argument("--vx", type=float, default=0.16, help="Initial forward walk speed (m/s)")
    parser.add_argument(
        "--policy",
        type=str,
        default="xgoduck_walk.npz",
        help="Any .npz policy in models_npz/ (e.g. zoo_moonwalk.npz, zoo_backflip.npz, zoo_dance_120bpm.npz)",
    )
    args = parser.parse_args()

    udp_target = None if args.esp32_serial else args.hil_udp
    node = PiZeroRuntimeNode(udp_target=udp_target, serial_port=args.esp32_serial, baud=args.baud)
    if args.policy != "xgoduck_walk.npz":
        npz_name = args.policy if args.policy.endswith(".npz") else f"{args.policy}.npz"
        node.policies["walk"] = PiZeroPolicyWrapper(
            npz_name, use_model_home=not npz_name.startswith("xgoduck_")
        )
    node.twist = [args.vx, 0.0, 0.0]

    print("=" * 78)
    print("Raspberry Pi Zero W (v1.1) ONNX-Free (.npz) XGO-Duck Runtime Client")
    print(f"Transport: {'ESP32 Serial ' + args.esp32_serial if args.esp32_serial else 'UDP HIL ' + args.hil_udp}")
    print(f"Loaded {len(node.policies)} pure-NumPy policies from {MODELS_NPZ_DIR}")
    print("=" * 78)

    start_t = time.monotonic()
    last_50hz = start_t
    last_print = start_t

    try:
        while True:
            now = time.monotonic()
            if args.duration > 0 and (now - start_t) >= args.duration:
                break
            node.poll_transport()
            if (now - last_50hz) >= 0.020:
                last_50hz = now
                node.step_50hz()
            if (now - last_print) >= 1.0:
                last_print = now
                mean_ms = float(np.mean(node.infer_ms_history)) if node.infer_ms_history else 0.0
                print(
                    f"[PiZeroW HIL] rx={node.rx_count:4d} tx={node.tx_count:4d} "
                    f"skill={node.active_skill:10s} grav=({node.grav[0]:+.2f},{node.grav[1]:+.2f},{node.grav[2]:+.2f}) "
                    f"infer={mean_ms:.3f} ms"
                )
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        if node.udp_sock is not None:
            node.udp_sock.close()
        if node.ser is not None:
            node.ser.close()


if __name__ == "__main__":
    main()
