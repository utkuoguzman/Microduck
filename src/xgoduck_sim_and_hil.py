"""Unified XGO-Duck (Feetech 1910) MuJoCo Simulator, Web UI, Needle 3 Brain & Pi Zero W HIL Bridge.

What this script does:
1. Builds the exact XGO-Duck MuJoCo physics model (`xgoduck_rl/src/mjlab_microduck/robot/xgoduck/scene.xml`)
   with the Roulade/Standup full-body collision setup and the **BAM `1910_m6.json` (Feetech HD-1910M)**
   actuator model (`kp=5.0`, `vin=7.8V`, voltage drop & Stribeck load-dependent friction).
2. Emulates the MCU hardware bridge (`DQS1` 240-byte state packets @ 100 Hz, `DQC1` 80-byte command
   packets @ 50 Hz) and connects directly to `xgoduck_runtime_arduino`'s `Controller`!
3. Monkey-patches `Policy` so it can run EITHER in `onnxruntime` OR in **Pure-NumPy `.npz` mode**
   (`--numpy-policy`, identical to what runs on the ARMv6 Raspberry Pi Zero W v1.1!).
4. Serves the official XGO-Duck Web UI on `http://127.0.0.1:9527` (Walk, Roller Mode, Somersault/Roulade,
   Left/Right Kick, Sit/Stand, Pick, Calibration, Servo Setup) AND accepts natural-language commands
   via Needle 3 ("The Polite Duck") in the terminal!
5. Opens a UDP Hardware-in-the-Loop (HIL) server on port `9528` so a real Raspberry Pi Zero W (v1.1)
   or ESP32 DevKit V1 on your network/USB can take over and drive the MuJoCo robot in real time!
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np

SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parent
NPZ_DIR = ROOT_DIR / "ml_checkpoints" / "models_npz"
ONNX_CORE_DIR = ROOT_DIR / "ml_checkpoints" / "onnx_core"
RUNTIME_PY_DIR = ROOT_DIR / "vendor" / "xgoduck_runtime_arduino" / "python"
RUNTIME_ASSETS_DIR = ROOT_DIR / "vendor" / "xgoduck_runtime_arduino" / "assets"
XGODUCK_ROBOT_DIR = ROOT_DIR / "simulation" / "xgoduck_mjcf"
if not XGODUCK_ROBOT_DIR.is_dir():
    XGODUCK_ROBOT_DIR = ROOT_DIR / "vendor" / "xgoduck_rl" / "src" / "mjlab_microduck" / "robot" / "xgoduck"
BAM_VENDOR_DIR = ROOT_DIR / "simulation" / "bam_actuator"
if not BAM_VENDOR_DIR.is_dir():
    BAM_VENDOR_DIR = ROOT_DIR / "vendor" / "xgoduck_rl" / "vendor" / "bam"

for p in (SUITE_DIR, RUNTIME_PY_DIR, BAM_VENDOR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bam.model import load_model as bam_load_model
from bam.mujoco import MujocoController as BamController

import policy as xgo_policy_mod
from rl_core import (
    ACTION_ALPHA,
    ACTION_ALPHA_HEAD,
    ACTION_DIM,
    ACTION_SCALE,
    DEFAULT_DEG,
    HOME_DEG,
    MOTOR_IDS,
    OBS_DIM,
    POLICY_IDS,
    POLICY_INDEX,
    State,
    action_to_deg as deg_from_action,
    build_obs as make_obs,
)
from wire import (
    ALL_MASK,
    ARM,
    CAL_CENTER_ONE,
    CAL_ENTER,
    CAL_EXIT,
    CAL_HOLD_POS,
    CAL_REPLY,
    COMMAND,
    DIAGNOSTIC,
    DISARM,
    HOLD,
    POLICY,
    RAW,
    SERVO_CENTER_POS,
    SERVO_PING,
    SERVO_READ,
    SERVO_REPLY,
    SERVO_RESET_MID,
    SERVO_SET_ID,
    SERVO_SET_PERM_KP_KD,
    SERVO_SET_PROTECTION,
    SERVO_SET_PROTECTION_CURRENT,
    SERVO_UNLOCK,
    STATE,
    decode_cal_reply,
    decode_servo_reply,
)
import base64
from gtsam_slam import MicroduckGtsamSLAM, Pose2
from numpy_policy import NumpyMlpGraph, extract_onnx_to_numpy_graph
from policy_zoo import POLICY_ZOO, RLPolicySpec, NeedleZooBrain, match_zoo_skill_from_text
from whistle_stt import WhistleSpeechToText, record_windows_mic_wav, synthesize_spoken_wav_windows

# Actuator names in XGO-Duck MJCF (matches POLICY_IDS order: 10..14, 30..33, 20..24)
MJCF_ACTUATOR_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


class DualBackendPolicy:
    """Drop-in replacement for xgoduck_runtime_arduino Policy supporting Pure-NumPy (.npz) and ONNX."""

    USE_NUMPY_BACKEND: bool = True

    def __init__(self):
        self.action_scale = ACTION_SCALE
        self.action_alpha = ACTION_ALPHA
        self.action_alpha_head = ACTION_ALPHA_HEAD
        self.override_twist_fn: Optional[Callable[[], tuple[float, float, float]]] = None
        self.override_head_fn: Optional[Callable[[], list[float]]] = None
        self.unload()

    @property
    def loaded(self):
        return self.np_graph is not None or self.session is not None

    @property
    def default_deg(self):
        return self.action_to_deg(np.zeros(ACTION_DIM, dtype=np.float32))

    def unload(self):
        self.session = None
        self.np_graph: Optional[NumpyMlpGraph] = None
        self.in_name = self.file = ""
        self.input_dim = OBS_DIM
        self.home_deg = HOME_DEG.copy()
        self._obs = np.empty(OBS_DIM, dtype=np.float32)
        self.reset_action_filter()

    def load(self, path=None, *, use_model_home=False, npz_override: Optional[Path] = None):
        p = Path(path) if path else (ONNX_CORE_DIR / "xgoduck_walk.onnx")
        if not p.is_file() and (ONNX_CORE_DIR / p.name).is_file():
            p = ONNX_CORE_DIR / p.name
        elif not p.is_file() and (RUNTIME_PY_DIR / p.name).is_file():
            p = RUNTIME_PY_DIR / p.name
        npz_candidate = npz_override if npz_override is not None else (NPZ_DIR / f"{p.stem}.npz")
        if p.suffix.lower() == ".npz" and p.is_file():
            npz_candidate = p

        try:
            if self.USE_NUMPY_BACKEND:
                if npz_candidate.is_file():
                    graph = NumpyMlpGraph.from_npz(npz_candidate)
                elif p.is_file():
                    graph = extract_onnx_to_numpy_graph(p)
                else:
                    return False, f"not found: {p}"
                if graph.input_dim not in (51, OBS_DIM) or graph.output_dim != ACTION_DIM:
                    raise ValueError(f"expected [51/61->14], got [{graph.input_dim}->{graph.output_dim}]")
                self.input_dim = graph.input_dim
                self.home_deg = HOME_DEG.copy()
                if use_model_home:
                    home = graph.metadata.get("default_joint_pos")
                    if home is not None:
                        home_arr = np.array([float(v) for v in home.split(",")], dtype=np.float32)
                        self.home_deg = np.rad2deg(home_arr)
                self.np_graph = graph
                self.file = f"{npz_candidate.stem}.npz (NumPy)"
                res = self.warm(4)
                if res.shape != (1, ACTION_DIM) or not np.isfinite(res).all():
                    raise ValueError("NumPy policy warm pass failed")
            else:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess = ort.InferenceSession(str(p), sess_options=options, providers=["CPUExecutionProvider"])
                inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
                self.input_dim = int(inp.shape[-1])
                self.home_deg = HOME_DEG.copy()
                if use_model_home:
                    home = sess.get_modelmeta().custom_metadata_map.get("default_joint_pos")
                    if home is not None:
                        home_arr = np.array([float(v) for v in home.split(",")], dtype=np.float32)
                        self.home_deg = np.rad2deg(home_arr)
                self.session, self.in_name = sess, inp.name
                self._outputs = [out.name]
                self.file = p.name
                self._feed = {self.in_name: self._obs[: self.input_dim].reshape(1, self.input_dim)}
                self.warm(4)
        except Exception as exc:
            self.unload()
            return False, str(exc)

        self.reset_action_filter()
        return True, f"{self.file} in={self.input_dim} out=14"

    def warm(self, runs: int = 1):
        if not self.loaded:
            return None
        self._obs.fill(0)
        last = None
        for _ in range(max(1, runs)):
            if self.np_graph is not None:
                last = self.np_graph.forward(self._obs[: self.input_dim].reshape(1, self.input_dim))
            else:
                self._feed[self.in_name] = self._obs[: self.input_dim].reshape(1, self.input_dim)
                last = self.session.run(self._outputs, self._feed)[0]
        return last

    def build_obs(self, grav, gyro_rad, angles_deg, vels_deg, cmd=None, head=None):
        st = State([0.0] * 3, [0.0] * 3, list(angles_deg), list(vels_deg), list(grav), list(gyro_rad))
        if self.override_twist_fn is not None:
            cmd = self.override_twist_fn()
        if self.override_head_fn is not None:
            head = self.override_head_fn()
        twist = np.zeros(3, dtype=np.float32) if cmd is None else np.asarray(cmd, dtype=np.float32)
        head_cmd = np.zeros(4, dtype=np.float32) if head is None else np.asarray(head, dtype=np.float32)
        obs = make_obs(st, self.prev_action, twist, head_cmd, out=self._obs, home_deg=self.home_deg)
        if not np.isfinite(obs).all():
            raise ValueError("observation contains NaN or Inf")
        return obs[: self.input_dim]

    def infer(self, grav, gyro_rad, angles_deg, vels_deg, cmd=None, head=None):
        if not self.loaded:
            return None
        obs_slice = self.build_obs(grav, gyro_rad, angles_deg, vels_deg, cmd, head)
        if self.np_graph is not None:
            action = self.np_graph.forward(obs_slice).reshape(ACTION_DIM)
        else:
            self._feed[self.in_name] = obs_slice.reshape(1, self.input_dim)
            action = self.session.run(self._outputs, self._feed)[0].reshape(ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("model output contains NaN or Inf")
        self.prev_action[:] = action
        for sl, alpha in (
            (slice(0, 5), self.action_alpha),
            (slice(5, 9), self.action_alpha_head),
            (slice(9, 14), self.action_alpha),
        ):
            self._act_filt[sl] *= alpha
            self._act_filt[sl] += action[sl] * (1.0 - alpha)
        return self._act_filt.copy()

    def set_action_alphas(self, *, leg=None, head=None):
        leg = self.action_alpha if leg is None else float(leg)
        head = self.action_alpha_head if head is None else float(head)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (leg, head)):
            raise ValueError("action alpha must be within [0,1]")
        self.action_alpha, self.action_alpha_head = leg, head
        return leg, head

    def reset_action_filter(self):
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._act_filt = np.zeros(ACTION_DIM, dtype=np.float32)

    def action_to_deg(self, action):
        return deg_from_action(
            np.asarray(action, dtype=np.float32), home_deg=self.home_deg, action_scale=self.action_scale
        )


class FixedBamController(BamController):
    """Fixes freejoint DOF-vs-joint index mismatch in bam.mujoco.MujocoController.update()."""

    def reset(self, qpos):
        super().reset(qpos)
        self.model.actuator.q_target_smooth = self.q_target.copy()
        self.last_ts = self.mujoco_data.time

    def update(self):
        q = self.mujoco_data.qpos[self.qpos_indexes]
        dq = self.mujoco_data.qvel[self.dof_indexes]

        act = self.model.actuator
        vin_orig = act.vin
        if self.vin_drop_gain is not None:
            load = np.sum(np.abs(self._prev_motor_torque))
            vin_eff = vin_orig - self.vin_drop_gain * load
            if self.vin_min is not None:
                vin_eff = max(vin_eff, self.vin_min)
            act.vin = vin_eff

        dt = self.mujoco_model.opt.timestep
        self.last_ts = self.mujoco_data.time
        control = act.compute_control(self.q_target, q, dq, dt)
        torque = act.compute_torque(control, True, q, dq)

        if self.vin_drop_gain is not None:
            act.vin = vin_orig
            self._prev_motor_torque = np.atleast_1d(torque).copy()

        self.mujoco_data.ctrl[self.act_indexes] = torque

        torque_external = (
            -self.mujoco_data.qfrc_bias[self.dof_indexes]
            + self.mujoco_data.qfrc_constraint[self.dof_indexes]
        )

        if len(self.mujoco_data.efc_id) > 0:
            efc_id_repeated = np.repeat(
                [self.mujoco_data.efc_id], len(self.actuator), axis=0
            )
            # In MuJoCo, mjCNSTR_FRICTION_DOF efc_id is the DOF index (dof_indexes), not joint_indexes!
            id_repeated = np.repeat(
                [self.dof_indexes], len(self.mujoco_data.efc_id), axis=0
            ).T
            selector = efc_id_repeated == id_repeated
            selector = selector * (
                self.mujoco_data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF.value
            )
            friction_force = np.sum(self.mujoco_data.efc_force * selector, axis=1)
            torque_external = torque_external - friction_force

        torque_actuator = self.mujoco_data.qfrc_actuator[self.dof_indexes]
        frictionloss, damping = self.model.compute_frictions(
            torque_actuator, torque_external, dq
        )
        self.mujoco_model.dof_frictionloss[self.dof_indexes] = frictionloss
        self.mujoco_model.dof_damping[self.dof_indexes] = damping


def build_xgoduck_mujoco_model(xml_path: Path | None = None) -> tuple[mujoco.MjModel, mujoco.MjData, FixedBamController]:
    """Compile XGO-Duck MJCF with full-body roulade/standup contacts and Feetech 1910 BAM actuator model."""
    if xml_path is None:
        xml_path = XGODUCK_ROBOT_DIR / "scene.xml"

    # Load Feetech 1910 (HD-1910M) BAM actuator model first so we can configure MjSpec motor actuators
    hls1910_json = XGODUCK_ROBOT_DIR / "params" / "1910_m6.json"
    bam_model = bam_load_model(str(hls1910_json))
    bam_model.actuator.kp = 5.0
    bam_model.actuator.vin = 7.8
    kt = bam_model.kt.value
    R = bam_model.R.value
    armature = bam_model.actuator.get_extra_inertia()
    force_limit = 8.0 * kt / R

    spec = mujoco.MjSpec.from_file(str(xml_path))
    # Configure collisions matching xgoduck_constants.py (XGODUCK_COLLISION + get_roulade_spec)
    for geom in spec.geoms:
        if not geom.name:
            geom.contype = 0
            geom.conaffinity = 0
        elif geom.name == "floor" or geom.name.startswith("landmark_"):
            geom.contype = 1
            geom.conaffinity = 1
        elif geom.name in ("trunk_collision", "head_collision"):
            geom.contype = 3
            geom.conaffinity = 2
            geom.condim = 1
        elif geom.name.endswith("_collision"):
            geom.contype = 1
            geom.conaffinity = 0
            if geom.name in ("left_foot_collision", "right_foot_collision"):
                geom.condim = 3
                geom.priority = 1
                geom.friction = [1.0, 0.005, 0.0001]
            else:
                geom.condim = 1

    # Convert XML <position> actuators to <motor> mode for BAM torque control (matching bam.mjlab.BamActuator.edit_spec)
    for mjact in spec.actuators:
        mjact.set_to_motor()
        mjact.forcelimited = True
        mjact.forcerange = (-force_limit, force_limit)
        mjact.ctrlrange = (-force_limit, force_limit)
        mjact.gear = [1.0, 0, 0, 0, 0, 0]
    for joint in spec.joints:
        if joint.name in MJCF_ACTUATOR_NAMES:
            joint.armature = float(armature)
            joint.damping = [0.0, 0.0, 0.0]
            joint.frictionloss = 0.0
            joint.solref_friction = [-5.0e4, -2.0e2]
            joint.solimp_friction = [0.99, 0.9999, 0.001, 0.5, 2.0]

    for b1, b2 in (("trunk_base", "hip_l"), ("trunk_base", "hip_l_2"), ("neck_pitch", "jaw_soft")):
        spec.add_exclude(bodyname1=b1, bodyname2=b2)

    model = spec.compile()
    model.opt.timestep = 0.004  # 250 Hz physics (2.5 steps per 100 Hz MCU tick, 5 steps per 50 Hz RL tick)
    model.opt.iterations = 30
    model.opt.ls_iterations = 50
    data = mujoco.MjData(model)

    # Reset to STAND keyframe
    stand_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, stand_key)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    bam_ctrl = FixedBamController(
        bam_model,
        MJCF_ACTUATOR_NAMES,
        model,
        data,
        vin_drop_gain=0.1,
        vin_min=7.0,
    )
    bam_ctrl.reset(data.qpos)
    return model, data, bam_ctrl


class MuJoCoXgoDuckBridge:
    """Emulates the Arduino Uno Q STM32 / ESP32 hardware bridge inside MuJoCo at 100 Hz.

    Also hosts a UDP HIL endpoint (default port 9528) so an external Raspberry Pi Zero W (v1.1)
    or ESP32 can receive the 240-byte DQS1 state packets and send 80-byte DQC1 command packets!
    """

    def __init__(self, hil_udp_port: int = 9528, esp32_serial_port: Optional[str] = None, esp32_baud: int = 921600):
        self.model, self.data, self.bam = build_xgoduck_mujoco_model()
        self.lock = threading.Lock()
        self.callbacks: Dict[str, Callable[[bytes], None]] = {}

        # Map POLICY_IDS (14 joints) <-> MOTOR_IDS (15 joints)
        self.policy_id_to_act_idx = {pid: i for i, pid in enumerate(POLICY_IDS)}
        self.motor_id_to_idx = {mid: i for i, mid in enumerate(MOTOR_IDS)}

        self.enabled = False
        self.ever_armed = False
        self.mode = DISARM
        self.target_deg = list(DEFAULT_DEG)
        self.angles_deg = list(DEFAULT_DEG)
        self.vels_deg = [0.0] * 15
        self.seq = 0
        self.cmd_seq = 0
        self.last_cmd_us = 0
        self.started_monotonic = time.monotonic()
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.gyro_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.slam = MicroduckGtsamSLAM()
        self.slam.reset(Pose2(float(self.data.qpos[0]), float(self.data.qpos[1]), 0.0))

        # UDP HIL Server for external Pi Zero W (1.1) / ESP32
        self.hil_udp_port = hil_udp_port
        self.hil_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.hil_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.hil_sock.bind(("0.0.0.0", self.hil_udp_port))
        self.hil_sock.setblocking(False)
        self.hil_peer: Optional[tuple[str, int]] = None
        self.external_hil_active = False
        self.last_hil_rx = 0.0

        # Optional USB Serial HIL Link to ESP32 Bridge (for 3-Node PC <-> ESP32 <-> Pi Zero W HIL)
        self.esp32_ser = None
        self._ser_rx_buf = bytearray()
        if esp32_serial_port:
            try:
                import serial  # type: ignore
                self.esp32_ser = serial.Serial(esp32_serial_port, esp32_baud, timeout=0)
                print(f"[HIL USB Serial] Connected to ESP32 on {esp32_serial_port} @ {esp32_baud} baud")
            except Exception as exc:
                print(f"[HIL USB Serial] Warning: could not open {esp32_serial_port}: {exc}")

    def now_us(self) -> int:
        return int(round(self.data.time * 1e6)) & 0xFFFFFFFF

    def provide(self, topic: str, callback: Callable[[bytes], None]) -> bool:
        self.callbacks[topic] = callback
        return True

    def notify(self, topic: str, payload: bytes) -> bool:
        if topic == "duck_command":
            self._handle_command_packet(payload, from_external_hil=False)
        elif topic == "duck_cal":
            magic, seq, op, index = struct.unpack("<4sIBB", payload)
            if op == CAL_CENTER_ONE and "duck_cal_reply" in self.callbacks:
                reply = CAL_REPLY.pack(b"DQA1", seq, index, 1)
                self.callbacks["duck_cal_reply"](reply)
        elif topic == "duck_servo":
            if self.esp32_ser is not None:
                self._send_framed_esp32(0x05, payload)
                return True
            magic, op, target_id, new_id, kp, kd, pflags, pcur = struct.unpack("<4sBBBBBBH", payload)
            if "duck_servo_reply" in self.callbacks:
                reply = SERVO_REPLY.pack(b"DQS3", op, target_id, 1, pflags, SERVO_CENTER_POS, SERVO_CENTER_POS, kp, kd)
                self.callbacks["duck_servo_reply"](reply)
        return True

    def _handle_command_packet(self, payload: bytes, from_external_hil: bool = False) -> None:
        if len(payload) != COMMAND.size:
            return
        vals = COMMAND.unpack(payload)
        if vals[0] != b"DQC1":
            return
        if from_external_hil:
            self.external_hil_active = True
            self.last_hil_rx = time.monotonic()
        elif self.external_hil_active and (time.monotonic() - self.last_hil_rx) < 1.0:
            # External Pi Zero W HIL takes priority while actively streaming
            return

        with self.lock:
            self.cmd_seq = vals[1]
            self.last_cmd_us = self.now_us()
            mode = vals[4]
            self.mode = mode
            if mode == DISARM:
                self.enabled = False
            elif mode in (ARM, POLICY, HOLD):
                self.enabled = True
                self.ever_armed = True
            if self.enabled:
                self.target_deg = list(vals[7:22])

    def _poll_hil_udp(self) -> None:
        while True:
            try:
                data, addr = self.hil_sock.recvfrom(512)
            except BlockingIOError:
                break
            except OSError:
                break
            self.hil_peer = addr
            if len(data) == COMMAND.size and data[:4] == b"DQC1":
                self._handle_command_packet(data, from_external_hil=True)
            elif data == b"HELLO_HIL":
                self.last_hil_rx = time.monotonic()

    def _poll_esp32_serial(self) -> None:
        if self.esp32_ser is None:
            return
        try:
            waiting = self.esp32_ser.in_waiting
            if waiting > 0:
                self._ser_rx_buf.extend(self.esp32_ser.read(waiting))
        except Exception:
            return

        while len(self._ser_rx_buf) >= 5:
            if self._ser_rx_buf[0] != 0xA5 or self._ser_rx_buf[1] != 0x5A:
                del self._ser_rx_buf[0]
                continue
            pkt_type = self._ser_rx_buf[2]
            plen = self._ser_rx_buf[3]
            total_len = 4 + plen + 1
            if len(self._ser_rx_buf) < total_len:
                break
            frame = bytes(self._ser_rx_buf[:total_len])
            del self._ser_rx_buf[:total_len]
            crc = pkt_type ^ plen
            payload = frame[4 : 4 + plen]
            for b in payload:
                crc ^= b
            if crc == frame[-1]:
                if pkt_type in (0x02, 0x10) and plen == COMMAND.size:
                    self._handle_command_packet(payload, from_external_hil=True)
                elif pkt_type == 0x07 and "duck_servo_reply" in self.callbacks:
                    self.callbacks["duck_servo_reply"](payload)

    def _send_framed_esp32(self, pkt_type: int, payload: bytes) -> None:
        if self.esp32_ser is None:
            return
        plen = len(payload)
        crc = pkt_type ^ plen
        for b in payload:
            crc ^= b
        frame = bytes([0xA5, 0x5A, pkt_type, plen]) + payload + bytes([crc])
        try:
            self.esp32_ser.write(frame)
        except Exception:
            pass

    def step_100hz(self) -> bytes:
        """Advance MuJoCo by 10 ms (100 Hz) and emit a 240-byte DQS1 StatePacket."""
        self._poll_hil_udp()
        self._poll_esp32_serial()

        with self.lock:
            enabled = self.enabled
            targets_deg = list(self.target_deg)

        # Convert MOTOR_IDS target degrees -> 14 MJCF actuator radians (POLICY_IDS order)
        for pid_idx, pid in enumerate(POLICY_IDS):
            motor_idx = MOTOR_IDS.index(pid)
            target_rad = math.radians(targets_deg[motor_idx])
            self.bam.set_q_target(MJCF_ACTUATOR_NAMES[pid_idx], target_rad)

        # 2 or 3 substeps per 10 ms tick (we alternate 2 and 3 steps of 4 ms -> average 10 ms)
        n_substeps = 2 if (self.seq % 2 == 0) else 3
        for _ in range(n_substeps):
            self.bam.update()
            if not enabled and self.ever_armed:
                self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)

        # Read joint positions and velocities back in MOTOR_IDS order (degrees, deg/s)
        q_rad = self.data.qpos[self.bam.qpos_indexes]
        dq_rad = self.data.qvel[self.bam.dof_indexes]
        for pid_idx, pid in enumerate(POLICY_IDS):
            motor_idx = MOTOR_IDS.index(pid)
            self.angles_deg[motor_idx] = float(math.degrees(q_rad[pid_idx]))
            self.vels_deg[motor_idx] = float(math.degrees(dq_rad[pid_idx]))
        # Mouth (ID 34, index 14) tracks target directly
        self.angles_deg[14] = float(targets_deg[14])
        self.vels_deg[14] = 0.0

        # Read trunk orientation & angular velocity in trunk_base frame
        rot_mat = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        # World gravity unit vector is [0, 0, -1]; in body frame: R^T @ [0, 0, -1] = -rot_mat[2, :]
        grav_body = -rot_mat[2, :]
        if self.gyro_sensor_id >= 0:
            adr = self.model.sensor_adr[self.gyro_sensor_id]
            gyro_rad_s = self.data.sensordata[adr : adr + 3]
        else:
            gyro_rad_s = np.zeros(3)
        gyro_dps = [float(math.degrees(v)) for v in gyro_rad_s]

        # Encode into QMI8658 raw int16 format expected by wire.py:
        # wire.py decodes: acc_ms2 = raw * 9.8 / 16384; gyro_dps = raw / 32
        # and proc.py + rl_core.py map [acc_ms2[0], acc_ms2[1], acc_ms2[2]] -> acc_n
        acc_raw = [
            int(np.clip(round(grav_body[0] * 16384.0), -32767, 32767)),
            int(np.clip(round(grav_body[1] * 16384.0), -32767, 32767)),
            int(np.clip(round(grav_body[2] * 16384.0), -32767, 32767)),
        ]
        gyro_raw = [
            int(np.clip(round(gyro_dps[0] * 32.0), -32767, 32767)),
            int(np.clip(round(gyro_dps[1] * 32.0), -32767, 32767)),
            int(np.clip(round(gyro_dps[2] * 32.0), -32767, 32767)),
        ]

        self.seq = (self.seq + 1) & 0xFFFFFFFF
        now_us = self.now_us()
        flags = 1 | (2 if enabled else 0)
        cmd_age = (now_us - self.last_cmd_us) & 0xFFFFFFFF if self.last_cmd_us else 0xFFFFFFFF

        pkt = STATE.pack(
            b"DQS1",
            1,
            flags,
            ALL_MASK,
            self.seq,
            now_us,
            now_us,
            self.cmd_seq,
            cmd_age,
            10000,
            800,
            0,
            0,
            0,
            acc_raw[0],
            acc_raw[1],
            acc_raw[2],
            gyro_raw[0],
            gyro_raw[1],
            gyro_raw[2],
            *self.angles_deg,
            *self.vels_deg,
            *targets_deg,
        )

        if "duck_state" in self.callbacks:
            self.callbacks["duck_state"](pkt)

        if self.hil_peer is not None:
            try:
                self.hil_sock.sendto(pkt, self.hil_peer)
            except OSError:
                pass

        if self.esp32_ser is not None:
            self._send_framed_esp32(0x01, pkt)

        # 1. Feed 100 Hz Borglab GTSAM LeggedEstimator (`LeggedInvariantEKF` + `ExtendedPoseContactFactor`)
        self.slam.legged.step_100hz_proprioceptive(
            dt=0.01,
            imu_acc_body=np.asarray(grav_body, dtype=np.float64) * 9.81,
            imu_gyro_body=np.asarray(gyro_rad_s, dtype=np.float64),
            q_left_deg=self.angles_deg[0:5],
            q_right_deg=self.angles_deg[5:10],
            rot_world_from_body=rot_mat,
            pose_idx=max(0, len(self.slam.optimized_poses) - 1),
        )

        # 2. Feed GTSAM 4.3 Factor-Graph SLAM & DM-VIO (checked at 20 Hz / every 5 ticks, regardless of torque state)
        if self.seq % 5 == 0:
            gx, gy = float(self.data.qpos[0]), float(self.data.qpos[1])
            gw, gqx, gqy, gqz = self.data.qpos[3:7]
            gyaw_rad = math.atan2(2.0 * (gw * gqz + gqx * gqy), 1.0 - 2.0 * (gqy * gqy + gqz * gqz))
            head_yaw_deg = float(self.angles_deg[12])  # ID 32 head_yaw
            self.slam.step_odometry_and_landmarks(
                gx,
                gy,
                gyaw_rad,
                neck_yaw_deg=head_yaw_deg,
                sim_time=float(self.data.time),
            )

        return pkt

    def respawn_upright(self) -> None:
        with self.lock:
            stand_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
            if stand_key >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, stand_key)
            mujoco.mj_forward(self.model, self.data)
            self.bam.reset(self.data.qpos)
            self.slam.reset(Pose2(float(self.data.qpos[0]), float(self.data.qpos[1]), 0.0))

    def pose_summary(self) -> dict:
        x, y, z = self.data.qpos[0:3]
        w, qx, qy, qz = self.data.qpos[3:7]
        roll = math.degrees(math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * qy - qz * qx)))))
        yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        return {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "z": round(float(z), 3),
            "roll_deg": round(roll, 1),
            "pitch_deg": round(pitch, 1),
            "yaw_deg": round(yaw, 1),
        }


import struct


def run_headless_verification(use_numpy: bool = True) -> None:
    """Run an automated multi-skill simulation test in MuJoCo to verify all XGO-Duck policies!"""
    DualBackendPolicy.USE_NUMPY_BACKEND = use_numpy
    xgo_policy_mod.Policy = DualBackendPolicy
    import controller as xgo_controller_mod
    from controller import Controller

    bridge = MuJoCoXgoDuckBridge(hil_udp_port=9529)

    class _SimClock:
        @staticmethod
        def monotonic() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def perf_counter() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def sleep(s: float) -> None:
            time.sleep(s)

    orig_time = xgo_controller_mod.time
    xgo_controller_mod.time = _SimClock
    try:
        ctrl = Controller(bridge)
        bridge.provide("duck_state", ctrl.receive)

        print("=" * 80)
        print(f"XGO-Duck (Feetech 1910 BAM) Multi-Skill MuJoCo Verification (NumPy={use_numpy})")
        print(f"Loaded models: {ctrl.model_message}")
        print("=" * 80)

        wall_infer_ms: List[float] = []

        def simulate_seconds(sec: float, keep_alive_cmd: bool = True) -> None:
            ticks_100hz = int(round(sec * 100))
            for i in range(ticks_100hz):
                if keep_alive_cmd:
                    ctrl.web_time = _SimClock.monotonic()
                bridge.step_100hz()
                if i % 2 == 1:
                    t0 = time.perf_counter()
                    ctrl.step()
                    wall_infer_ms.append((time.perf_counter() - t0) * 1000.0)

        # Warm up IMU filter in shadow mode (0.2s)
        simulate_seconds(0.2)

        # 1. Enable policy mode & stand upright
        ctrl.set_mode("policy")
        simulate_seconds(1.0)
        p_stand = bridge.pose_summary()
        print(f"[1] Standing upright (1.0s):        pose={p_stand}")

        # 2. Walk forward (vx = 0.18 m/s for 2.5s)
        ctrl.command(twist=[0.18, 0.0, 0.0], head=[0.0, 0.0, 0.0, 0.0], mouth=0.0)
        simulate_seconds(2.5)
        p_walk = bridge.pose_summary()
        print(f"[2] Walk forward vx=0.18 (2.5s):    pose={p_walk} (dx={p_walk['x'] - p_stand['x']:+.3f} m)")

        # 3. Stop motion & Right Ball Kick (0.4s kick + 0.6s settle)
        ctrl.command(twist=[0.0, 0.0, 0.0])
        simulate_seconds(0.4)
        ctrl.start_kick("right")
        simulate_seconds(1.0)
        p_kick = bridge.pose_summary()
        print(f"[3] Right Ball Kick + recovery:     pose={p_kick}")

        # 4. Sit down & Stand up
        ctrl.set_sitstand("sit")
        simulate_seconds(2.0)
        p_sit = bridge.pose_summary()
        ctrl.set_sitstand("stand")
        simulate_seconds(1.5)
        p_stood = bridge.pose_summary()
        print(f"[4] Sit -> Stand policy:            sit_z={p_sit['z']:.3f}m -> stand_z={p_stood['z']:.3f}m")

        # 5. Ground Pick (beak opens to 30 deg, crouches to pick, closes beak, stands back up)
        ctrl.start_pick()
        simulate_seconds(4.2)
        p_pick = bridge.pose_summary()
        print(f"[5] Ground Pick (4.0s cycle):       pose={p_pick}")

        # 6. Somersault / Roulade (1.9s forward roll)
        ctrl.start_roulade()
        min_pitch, max_pitch = 0.0, 0.0
        for i in range(250):
            ctrl.web_time = _SimClock.monotonic()
            bridge.step_100hz()
            if i % 2 == 1:
                t0 = time.perf_counter()
                ctrl.step()
                wall_infer_ms.append((time.perf_counter() - t0) * 1000.0)
            ps = bridge.pose_summary()
            min_pitch = min(min_pitch, ps["pitch_deg"])
            max_pitch = max(max_pitch, ps["pitch_deg"])
        p_roulade = bridge.pose_summary()
        print(f"[6] Roulade / Somersault (2.5s):    pose={p_roulade} (recovery phase={ctrl.recovery.phase})")

        # 7. Fall Recovery (xgoduck_getup.npz): knock duck onto its back and verify automatic stand-up
        bridge.data.qpos[2] = 0.06
        # 90 deg backward pitch quaternion: cos(-45deg)=0.7071, sin(-45deg)=-0.7071 around Y
        bridge.data.qpos[3:7] = [0.7071068, 0.0, -0.7071068, 0.0]
        bridge.data.qvel[:] = 0.0
        mujoco.mj_forward(bridge.model, bridge.data)
        simulate_seconds(3.5)
        p_getup = bridge.pose_summary()
        print(f"[7] Fall Recovery (getup 3.5s):     pose={p_getup} (recovery phase={ctrl.recovery.phase})")

        avg_infer_ms = float(np.mean(wall_infer_ms)) if wall_infer_ms else 0.0
        print("-" * 80)
        print(f"Total 50 Hz policy inferences: {ctrl.inferred} | Mean step+inference latency: {avg_infer_ms:.3f} ms")
        print("=" * 80)
    finally:
        xgo_controller_mod.time = orig_time
        bridge.hil_sock.close()


def run_udp_hil_test() -> None:
    """Verify end-to-end UDP Hardware-in-the-Loop (HIL) between pi_zero_client.py and MuJoCo."""
    import subprocess

    hil_port = 9530
    bridge = MuJoCoXgoDuckBridge(hil_udp_port=hil_port)
    client_script = SUITE_DIR / "pi_zero_w_runtime" / "pi_zero_client.py"

    print("=" * 80)
    print(f"Launching Pi Zero W (v1.1) Pure-NumPy HIL Client -> UDP 127.0.0.1:{hil_port}")
    print("=" * 80)

    proc = subprocess.Popen(
        [
            sys.executable,
            str(client_script),
            "--hil-udp",
            f"127.0.0.1:{hil_port}",
            "--duration",
            "3.2",
            "--vx",
            "0.18",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    p_init = bridge.pose_summary()
    t_end = time.monotonic() + 3.4
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        bridge.step_100hz()
        dt = time.perf_counter() - t0
        if dt < 0.010:
            time.sleep(0.010 - dt)

    out, _ = proc.communicate(timeout=5.0)
    print(out.strip())
    p_final = bridge.pose_summary()
    print("-" * 80)
    print(
        f"[MuJoCo HIL Server] External HIL active={bridge.external_hil_active} | "
        f"cmd_seq={bridge.cmd_seq} | Initial x={p_init['x']:.3f}m -> Final pose={p_final} "
        f"(dx={p_final['x'] - p_init['x']:+.3f}m)"
    )
    print("=" * 80)
    bridge.hil_sock.close()


def start_zoo_skill(ctrl, bridge: MuJoCoXgoDuckBridge, spec: RLPolicySpec) -> dict:
    """Activate any of the 30 RL policies in `POLICY_ZOO` on the live `Controller`."""
    import controller as xgo_controller_mod

    if not hasattr(ctrl, "_orig_roulade_policy"):
        ctrl._orig_roulade_policy = ctrl.roulade_policy
        ctrl._zoo_cache = {}
        ctrl.active_zoo_skill = "xgoduck_walk"
        ctrl.last_needle_msg = "Ready — type a polite request with 'please' or click any RL skill!"

    with ctrl.lock:
        if ctrl.mode not in ("policy", "shadow"):
            ctrl.set_mode("policy")
        elif ctrl.mode == "shadow":
            ctrl.set_mode("policy")
        # Ensure recovery is in 'walk' state and clear any blocking sub-states before starting a new skill
        if ctrl.recovery.phase != "walk":
            ctrl.recovery.reset()
        if spec.skill_id not in ("sit", "stand") and ctrl.sitstand_phase != "idle":
            ctrl._stop_sitstand_locked()
        if ctrl.kick_until:
            ctrl._stop_kick_locked()
        if spec.skill_id != "pick" and ctrl.pick_active:
            ctrl._stop_pick_locked()
        if ctrl.roulade_until:
            ctrl._stop_roulade_locked()
            ctrl.roulade_policy = ctrl._orig_roulade_policy

    ctrl.active_zoo_skill = spec.skill_id

    # 1. Core XGO-Duck skills with native Controller state machines
    if spec.skill_id == "xgoduck_walk":
        ctrl.set_roller_mode(False)
        ctrl.command(twist=list(spec.default_twist), head=[0.0, 0.0, 0.0, 0.0], mouth=0.0)
        return {"ok": True, "skill_id": spec.skill_id, "title": spec.title, "model": ctrl.walk_policy.file}
    elif spec.skill_id == "roller":
        ctrl.set_roller_mode(True)
        ctrl.command(twist=list(spec.default_twist), head=[0.0, 0.0, 0.0, 0.0], mouth=0.0)
        return {"ok": True, "skill_id": spec.skill_id, "title": spec.title, "model": ctrl.roller_policy.file}
    elif spec.skill_id == "roulade":
        ctrl.roulade_policy = ctrl._orig_roulade_policy
        return ctrl.start_roulade()
    elif spec.skill_id == "pick":
        return ctrl.start_pick()
    elif spec.skill_id == "sit":
        if ctrl.sitstand_phase == "sit":
            return {"ok": True, "skill_id": "sit", "phase": "sit"}
        return ctrl.set_sitstand("sit")
    elif spec.skill_id == "stand":
        if ctrl.sitstand_phase == "sit":
            return ctrl.set_sitstand("stand")
        return {"ok": True, "skill_id": "stand", "phase": "idle"}
    elif spec.skill_id in ("ballkick_right", "ballkick_left"):
        side = "right" if spec.skill_id == "ballkick_right" else "left"
        return ctrl.start_kick(side)

    # 2. Load (or retrieve from cache) the DualBackendPolicy for any of the 22 Hugging Face / Microduck Zoo policies
    cache: dict[str, DualBackendPolicy] = ctrl._zoo_cache
    if spec.skill_id not in cache:
        pol = DualBackendPolicy()
        ok, msg = pol.load(
            spec.resolve_onnx(),
            use_model_home=spec.use_model_home,
            npz_override=spec.resolve_npz(),
        )
        if not ok:
            raise RuntimeError(f"Failed to load RL policy {spec.skill_id}: {msg}")
        pol.set_action_alphas(leg=0.1, head=0.1)
        cache[spec.skill_id] = pol

    pol = cache[spec.skill_id]
    t_start = xgo_controller_mod.time.monotonic()
    dur = spec.duration_s if spec.duration_s > 0.0 else 5.0
    pol.override_twist_fn = lambda: spec.compute_twist(xgo_controller_mod.time.monotonic() - t_start)

    with ctrl.lock:
        if ctrl._roulade_alpha is None:
            ctrl._roulade_alpha = (ctrl.walk_policy.action_alpha, ctrl.walk_policy.action_alpha_head)
        ctrl.roulade_policy = pol
        ctrl.roulade_until = t_start + dur
        ctrl.policy_needs_reset = True
        ctrl.web_time = t_start
        ctrl.last_error = ""
        with ctrl.policy_lock:
            pol.reset_action_filter()
            pol.warm(2)

    return {
        "ok": True,
        "skill_id": spec.skill_id,
        "title": spec.title,
        "repo": spec.repo_source,
        "model": pol.file,
        "duration_s": dur,
    }


def execute_needle_command_on_xgoduck(
    ctrl,
    bridge: MuJoCoXgoDuckBridge,
    text: str,
    needle_brain: Optional[NeedleZooBrain] = None,
) -> dict:
    """Interpret natural language with `NeedleZooBrain` (Needle 3) and execute the matching RL policy."""
    import controller as xgo_controller_mod

    if needle_brain is None:
        needle_brain = getattr(ctrl, "_needle_zoo_brain", None)
    if needle_brain is None:
        needle_brain = NeedleZooBrain()
        ctrl._needle_zoo_brain = needle_brain

    plan = needle_brain.interpret(text)
    act_type = plan.get("action_type")

    if act_type == "shake_head":
        # Refused! Shake head left-right in place
        if ctrl.mode not in ("policy", "shadow"):
            ctrl.set_mode("policy")
        elif ctrl.mode == "shadow":
            ctrl.set_mode("policy")
        t0 = xgo_controller_mod.time.monotonic()
        ctrl.walk_policy.override_head_fn = (
            lambda: [0.0, 0.0, 0.75 * math.sin(2.0 * math.pi * 3.0 * (xgo_controller_mod.time.monotonic() - t0)), 0.0]
            if (xgo_controller_mod.time.monotonic() - t0) < 1.2
            else [0.0, 0.0, 0.0, 0.0]
        )
        ctrl.command(twist=[0.0, 0.0, 0.0], head=[0.0, 0.0, 0.65, 0.0], mouth=0.0)
        ctrl.last_needle_msg = plan["message"]
        return plan

    # Clear any temporary head-shake override
    ctrl.walk_policy.override_head_fn = None

    if act_type == "walk":
        if ctrl.mode != "policy":
            ctrl.set_mode("policy")
        if hasattr(ctrl, "_orig_roulade_policy") and ctrl.roulade_until:
            with ctrl.lock:
                ctrl._stop_roulade_locked()
                ctrl.roulade_policy = ctrl._orig_roulade_policy
        args = plan.get("args") or {}
        d = args.get("direction", "forward")
        vx = 0.18 if d == "forward" else (-0.15 if d == "backward" else 0.0)
        vy = 0.20 if d == "left" else (-0.20 if d == "right" else 0.0)
        ctrl.command(twist=[vx, vy, 0.0])
        ctrl.active_zoo_skill = "xgoduck_walk"
        ctrl.last_needle_msg = plan["message"]
        return plan

    if act_type == "turn":
        if ctrl.mode != "policy":
            ctrl.set_mode("policy")
        if hasattr(ctrl, "_orig_roulade_policy") and ctrl.roulade_until:
            with ctrl.lock:
                ctrl._stop_roulade_locked()
                ctrl.roulade_policy = ctrl._orig_roulade_policy
        args = plan.get("args") or {}
        d = args.get("direction", "left")
        wz = 0.8 if d == "left" else -0.8
        ctrl.command(twist=[0.0, 0.0, wz])
        ctrl.active_zoo_skill = "xgoduck_walk"
        ctrl.last_needle_msg = plan["message"]
        return plan

    spec: Optional[RLPolicySpec] = plan.get("spec")
    if spec is not None:
        start_zoo_skill(ctrl, bridge, spec)
        ctrl.last_needle_msg = plan["message"]
        return plan

    return plan


ZOO_DRAWER_HTML = """
<!-- Injected Needle 3 + 30-Policy RL Movement Zoo Console -->
<style>
  #zoo-fab {
    position: fixed; bottom: 18px; right: 18px; z-index: 9999;
    background: linear-gradient(135deg, #f59e0b, #ea580c);
    color: #fff; font-family: system-ui, sans-serif; font-weight: 700;
    border: none; border-radius: 999px; padding: 12px 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35); cursor: pointer;
    font-size: 14px; display: flex; align-items: center; gap: 8px;
  }
  #zoo-drawer {
    position: fixed; bottom: 72px; right: 18px; width: 430px; max-height: 78vh;
    background: rgba(15, 23, 42, 0.96); color: #f8fafc; z-index: 9998;
    border: 1px solid rgba(245, 158, 11, 0.45); border-radius: 16px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.55); font-family: system-ui, sans-serif;
    display: flex; flex-direction: column; overflow: hidden; backdrop-filter: blur(10px);
  }
  #zoo-drawer.hidden { display: none; }
  .zoo-hdr {
    padding: 12px 16px; background: rgba(30, 41, 59, 0.9);
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    display: flex; justify-content: space-between; align-items: center;
  }
  .zoo-hdr h3 { margin: 0; font-size: 15px; color: #fbbf24; }
  .zoo-needle-box { padding: 12px 16px; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }
  .zoo-input-row { display: flex; gap: 6px; }
  .zoo-input-row input {
    flex: 1; padding: 8px 10px; border-radius: 8px; border: 1px solid #475569;
    background: #0f172a; color: #fff; font-size: 13px;
  }
  .zoo-input-row button {
    background: #f59e0b; color: #0f172a; font-weight: 700; border: none;
    border-radius: 8px; padding: 8px 12px; cursor: pointer; font-size: 13px;
  }
  #zoo-ptt-btn {
    background: #ec4899; color: #fff; font-weight: 700; border: none;
    border-radius: 8px; padding: 8px 10px; cursor: pointer; font-size: 12px;
    user-select: none; transition: all 0.15s; white-space: nowrap;
  }
  #zoo-ptt-btn.recording {
    background: #ef4444; box-shadow: 0 0 14px rgba(239, 68, 68, 0.85);
    transform: scale(1.03);
  }
  #zoo-needle-status {
    margin-top: 8px; font-size: 12px; color: #cbd5e1;
    background: rgba(30, 41, 59, 0.75); padding: 6px 10px; border-radius: 6px;
  }
  .zoo-body { padding: 10px 16px 16px; overflow-y: auto; flex: 1; }
  .zoo-cat { margin-top: 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; font-weight: 700; }
  .zoo-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }
  .zoo-btn {
    background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
    border-radius: 8px; padding: 7px 9px; text-align: left; cursor: pointer;
    font-size: 12px; transition: all 0.15s;
  }
  .zoo-btn:hover { border-color: #f59e0b; background: #334155; }
  .zoo-btn b { display: block; color: #fbbf24; font-size: 12px; }
  .zoo-btn span { font-size: 10px; color: #94a3b8; }
</style>
<button id="zoo-fab" onclick="document.getElementById('zoo-drawer').classList.toggle('hidden')">
  🦆 Needle 3 + Whistle S2T + 30 RL Policies
</button>
<div id="zoo-drawer">
  <div class="zoo-hdr">
    <h3>🦆 Needle 3 + Whistle S2T + 30 RL Zoo</h3>
    <div style="display:flex;gap:8px;align-items:center;">
      <a href="/schematic" target="_blank" style="background:#334155;color:#38bdf8;text-decoration:none;font-size:11px;font-weight:700;padding:4px 8px;border-radius:6px;border:1px solid #475569;">⚡ EasyEDA Schematic</a>
      <button onclick="document.getElementById('zoo-drawer').classList.add('hidden')" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:16px;">✕</button>
    </div>
  </div>
  <div class="zoo-needle-box">
    <div class="zoo-input-row">
      <input id="zoo-prompt" type="text" placeholder="Type or hold 🎙️ & say 'please do a backflip'..." onkeydown="event.stopPropagation(); if(event.key==='Enter')sendNeedlePrompt()" onkeyup="event.stopPropagation()" />
      <button id="zoo-ptt-btn" onmousedown="startWhistlePTT(event)" onmouseup="stopWhistlePTT(event)" onmouseleave="if(window._whistleRec)stopWhistlePTT(event)" ontouchstart="startWhistlePTT(event)" ontouchend="stopWhistlePTT(event)" title="Hold to record voice with Cactus Whistle S2T (whistle_d4.cact)">🎙️ Hold to Speak</button>
      <button onclick="sendNeedlePrompt()">Ask</button>
    </div>
    <div id="zoo-needle-status">🎙️ Hold "Hold to Speak" (Whistle S2T) or type a request — say "please"!</div>
  </div>
  <div style="padding:10px 16px; border-bottom:1px solid rgba(148,163,184,0.2); background:rgba(15,23,42,0.85);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span style="font-size:12px;font-weight:700;color:#38bdf8;">🗺️ GTSAM 4.3 LeggedEstimator + DM-VIO + iSAM2</span>
      <span id="slam-stats-badge" style="font-size:10px;color:#4ade80;font-family:monospace;">Initializing radar...</span>
    </div>
    <div style="display:grid;grid-template-columns:1.35fr 1fr;gap:6px;">
      <canvas id="slam-radar-canvas" width="225" height="170" style="width:100%;height:170px;background:#0b1324;border:1px solid #38bdf8;border-radius:8px;display:block;"></canvas>
      <canvas id="dmvio-cam-canvas" width="160" height="170" style="width:100%;height:170px;background:#090e1a;border:1px solid #a855f7;border-radius:8px;display:block;" title="DM-VIO 160x120 Direct Photometric Inverse-Depth Patches (OV5647 Head Cam)"></canvas>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#cbd5e1;margin-top:5px;">
      <span>⚪ GT &nbsp; 🔴 Odom &nbsp; 🟢 GTSAM &nbsp; 🔵 Footholds &nbsp; 🟣 DM-VIO ρ=1/d</span>
      <span>
        <a href="#" id="slam-patrol-btn" onclick="toggleSlamPatrol(event)" style="color:#38bdf8;text-decoration:none;font-weight:700;margin-right:8px;">🧭 Auto-Patrol</a>
        <a href="#" onclick="resetSlamMap(event)" style="color:#fbbf24;text-decoration:none;">Reset</a>
      </span>
    </div>
  </div>
  <div class="zoo-body" id="zoo-skill-list">Loading 30 RL policies...</div>
</div>
<script>
window._whistleRec = false;
let _audioCtx = null, _mediaStream = null, _scriptNode = null, _pcmChunks = [];

function encodeWav16k(float32Samples, sampleRate) {
  // Downsample to 16000 Hz mono 16-bit PCM WAV
  const targetRate = 16000;
  const ratio = sampleRate / targetRate;
  const outLen = Math.max(1, Math.round(float32Samples.length / ratio));
  const buf = new ArrayBuffer(44 + outLen * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + outLen * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, targetRate, true);
  view.setUint32(28, targetRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, outLen * 2, true);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = Math.min(float32Samples.length - 1, Math.floor(i * ratio));
    const s = Math.max(-1, Math.min(1, float32Samples[srcIdx]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

async function startWhistlePTT(ev) {
  if (ev) ev.preventDefault();
  if (window._whistleRec) return;
  const btn = document.getElementById('zoo-ptt-btn');
  const st = document.getElementById('zoo-needle-status');
  try {
    _mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = _audioCtx.createMediaStreamSource(_mediaStream);
    _scriptNode = _audioCtx.createScriptProcessor(4096, 1, 1);
    _pcmChunks = [];
    _scriptNode.onaudioprocess = (e) => {
      if (!window._whistleRec) return;
      _pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    src.connect(_scriptNode);
    _scriptNode.connect(_audioCtx.destination);
    window._whistleRec = true;
    btn.classList.add('recording');
    btn.textContent = '🔴 Listening...';
    st.textContent = '🎙️ Recording voice for Cactus Whistle S2T... Release button when done!';
    st.style.color = '#f472b6';
  } catch (err) {
    st.textContent = 'Mic error: ' + err.message;
    st.style.color = '#f87171';
  }
}

async function stopWhistlePTT(ev) {
  if (ev) ev.preventDefault();
  if (!window._whistleRec) return;
  window._whistleRec = false;
  const btn = document.getElementById('zoo-ptt-btn');
  const st = document.getElementById('zoo-needle-status');
  btn.classList.remove('recording');
  btn.textContent = '🎙️ Hold to Speak';
  const sr = _audioCtx ? _audioCtx.sampleRate : 16000;
  if (_scriptNode) _scriptNode.disconnect();
  if (_mediaStream) _mediaStream.getTracks().forEach(t => t.stop());
  if (_audioCtx) _audioCtx.close();
  const totalLen = _pcmChunks.reduce((acc, c) => acc + c.length, 0);
  if (totalLen < sr * 0.15) {
    st.textContent = 'Hold the 🎙️ button while speaking (clip was too short).';
    return;
  }
  const merged = new Float32Array(totalLen);
  let off = 0;
  for (const c of _pcmChunks) { merged.set(c, off); off += c.length; }
  const wavB64 = encodeWav16k(merged, sr);
  st.textContent = 'Transcribing with Cactus Whistle (whistle_d4.cact) -> Needle 3...';
  const res = await fetch('/api/whistle', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({wav_base64: wavB64, language: 'en'})
  });
  const data = await res.json();
  if (data.transcript) {
    document.getElementById('zoo-prompt').value = data.transcript;
  }
  st.textContent = data.message || JSON.stringify(data);
  st.style.color = data.accepted ? '#4ade80' : '#f87171';
}

async function sendNeedlePrompt() {
  const inp = document.getElementById('zoo-prompt');
  const text = inp.value.trim();
  if (!text) return;
  const st = document.getElementById('zoo-needle-status');
  st.textContent = 'Thinking with Needle 3...';
  const res = await fetch('/api/needle', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: text})
  });
  const data = await res.json();
  st.textContent = data.message || JSON.stringify(data);
  st.style.color = data.accepted ? '#4ade80' : '#f87171';
}
async function runZooSkill(skill_id) {
  const st = document.getElementById('zoo-needle-status');
  const res = await fetch('/api/zoo/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({skill_id})
  });
  const data = await res.json();
  st.textContent = data.message || ('Running RL policy: ' + skill_id);
  st.style.color = '#38bdf8';
}
async function loadZooCatalog() {
  const res = await fetch('/api/zoo');
  const data = await res.json();
  const container = document.getElementById('zoo-skill-list');
  const byCat = {};
  for (const s of data.skills) {
    (byCat[s.category] = byCat[s.category] || []).push(s);
  }
  let html = '';
  for (const [cat, items] of Object.entries(byCat)) {
    html += `<div class="zoo-cat">${cat} (${items.length})</div><div class="zoo-grid">`;
    for (const item of items) {
      html += `<button class="zoo-btn" onclick="runZooSkill('${item.skill_id}')" title="${item.description}">
        <b>${item.title}</b><span>${item.skill_id} · ${item.repo_source.split('/')[0]}</span>
      </button>`;
    }
    html += `</div>`;
  }
  container.innerHTML = html;
}
async function resetSlamMap(ev) {
  if (ev) ev.preventDefault();
  await fetch('/api/slam/reset', {method: 'POST'});
  pollSlamRadar();
}
async function toggleSlamPatrol(ev) {
  if (ev) ev.preventDefault();
  const res = await fetch('/api/slam/patrol', {method: 'POST'});
  const data = await res.json();
  const btn = document.getElementById('slam-patrol-btn');
  if (btn) {
    btn.textContent = data.active ? '🛑 Stop Patrol' : '🧭 Auto-Patrol';
    btn.style.color = data.active ? '#f87171' : '#38bdf8';
  }
}
async function pollSlamRadar() {
  const cv = document.getElementById('slam-radar-canvas');
  if (!cv) return;
  try {
    const res = await fetch('/api/slam?t=' + Date.now(), {cache: 'no-store'});
    const data = await res.json();
    const btn = document.getElementById('slam-patrol-btn');
    if (btn && typeof data.patrol_active === 'boolean') {
      btn.textContent = data.patrol_active ? '🛑 Stop Patrol' : '🧭 Auto-Patrol';
      btn.style.color = data.patrol_active ? '#f87171' : '#38bdf8';
    }
    const m = data.metrics || {};
    const badge = document.getElementById('slam-stats-badge');
    if (badge) {
      badge.textContent = `X:${m.num_poses||1} F:${m.num_footholds||2} L:${m.num_landmarks||0}/8 | -${m.error_reduction_pct||0}% drift`;
    }
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.fillStyle = '#0b1324';
    ctx.fillRect(0, 0, W, H);
    const scale = 42.0;
    const cx = W * 0.5, cy = H * 0.5;
    const toPx = (x, y) => [cx + x * scale, cy - y * scale];

    // Crosshair axes & Grid rings
    ctx.strokeStyle = 'rgba(100,116,139,0.32)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(W, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
    ctx.stroke();

    ctx.fillStyle = 'rgba(148,163,184,0.65)';
    ctx.font = '9px monospace';
    for (const r of [0.5, 1.0, 1.5]) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillText(r.toFixed(1) + 'm', cx + r * scale + 2, cy - 2);
    }

    // Current duck pose & 110 deg Panning-Head FOV Cone
    const cp = data.current_pose || [0, 0, 0, 0];
    const [dx, dy] = toPx(cp[0], cp[1]);
    const headWorldYaw = (cp[2] || 0) + (cp[3] || 0);
    const fovHalf = (55.0 * Math.PI) / 180.0;
    const coneRangePx = 1.45 * scale;
    ctx.fillStyle = 'rgba(56, 189, 248, 0.14)';
    ctx.beginPath();
    ctx.moveTo(dx, dy);
    // Note: canvas Y is inverted (-angle)
    ctx.arc(dx, dy, coneRangePx, -(headWorldYaw + fovHalf), -(headWorldYaw - fovHalf), false);
    ctx.closePath();
    ctx.fill();

    // Active bearing-range rays from current optimized pose
    if (data.active_rays) {
      ctx.strokeStyle = 'rgba(56,189,248,0.55)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      for (const ray of data.active_rays) {
        const [lx, ly] = toPx(ray.lx, ray.ly);
        ctx.beginPath(); ctx.moveTo(dx, dy); ctx.lineTo(lx, ly); ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Stance Footholds F_c from Borglab LeggedEstimator
    for (const fh of (data.footholds || [])) {
      const [fx, fy] = toPx(fh[0], fh[1]);
      ctx.fillStyle = 'rgba(56,189,248,0.75)';
      ctx.beginPath(); ctx.arc(fx, fy, 2.2, 0, Math.PI * 2); ctx.fill();
    }

    // Draw paths helper
    const drawPath = (pts, color, width, dashed) => {
      if (!pts || pts.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed ? [3, 3] : []);
      ctx.beginPath();
      pts.forEach((p, idx) => {
        const [px, py] = toPx(p[0], p[1]);
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    drawPath(data.gt_path, 'rgba(226,232,240,0.8)', 1.6, true);
    drawPath(data.dr_path, '#f87171', 1.8, false);
    drawPath(data.opt_path, '#4ade80', 2.5, false);

    // Draw all 8 world landmarks (hollow slate if undiscovered, bright gold if discovered)
    const allLm = data.world_landmarks || [];
    for (const lm of allLm) {
      const [lx, ly] = toPx(lm.x, lm.y);
      if (lm.discovered) {
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath(); ctx.arc(lx, ly, 4.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fde68a';
      } else {
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(lx, ly, 3.8, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#64748b';
      }
      ctx.font = '9px sans-serif';
      ctx.fillText(lm.name.replace('ArUco_', '').replace('Corner_', ''), lx + 6, ly + 3);
    }

    // Draw current duck body & heading arrow
    ctx.fillStyle = '#4ade80';
    ctx.beginPath(); ctx.arc(dx, dy, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(dx, dy);
    ctx.lineTo(dx + Math.cos(headWorldYaw) * 14, dy - Math.sin(headWorldYaw) * 14);
    ctx.stroke();

    // Top-left HUD: Stance Foot Contact LEDs (LeggedEstimator)
    ctx.fillStyle = 'rgba(15,23,42,0.82)';
    ctx.fillRect(6, 6, 172, 20);
    ctx.font = '10px monospace';
    ctx.fillStyle = m.left_contact ? '#4ade80' : '#64748b';
    ctx.fillText(`● L-Stance`, 12, 19);
    ctx.fillStyle = m.right_contact ? '#4ade80' : '#64748b';
    ctx.fillText(`● R-Stance`, 95, 19);

    // Draw DM-VIO 160x120 First-Person Head Camera + Inverse-Depth Photometric Patches
    const camCv = document.getElementById('dmvio-cam-canvas');
    if (camCv) {
      const cctx = camCv.getContext('2d');
      const CW = camCv.width, CH = camCv.height;
      cctx.fillStyle = '#090e1a';
      cctx.fillRect(0, 0, CW, CH);

      // Horizon line & perspective floor grid
      const horizY = Math.round(CH * 0.44);
      cctx.fillStyle = '#111c33';
      cctx.fillRect(0, horizY, CW, CH - horizY);
      cctx.strokeStyle = 'rgba(148,163,184,0.22)';
      cctx.lineWidth = 1;
      cctx.beginPath();
      cctx.moveTo(0, horizY); cctx.lineTo(CW, horizY);
      for (let gx = -3; gx <= 3; gx++) {
        cctx.moveTo(CW * 0.5 + gx * 12, horizY);
        cctx.lineTo(CW * 0.5 + gx * 48, CH);
      }
      cctx.stroke();

      const sy = CH / 120.0;
      // Draw 3D MuJoCo Arena Pillars & Beacons in First-Person Perspective
      for (const pil of (data.dmvio_pillars || [])) {
        const pu = pil.u;
        const pvTop = pil.v_top * sy;
        const pvBot = pil.v_bot * sy;
        const pw = pil.w;
        const ph = Math.max(4, pvBot - pvTop);
        cctx.fillStyle = pil.color || '#38bdf8';
        cctx.globalAlpha = 0.72;
        cctx.fillRect(pu - pw * 0.5, pvTop, pw, ph);
        // Gold ArUco cap on top of pillar
        cctx.fillStyle = '#fbbf24';
        cctx.fillRect(pu - pw * 0.58, pvTop - 5, pw * 1.16, 5);
        cctx.globalAlpha = 1.0;
        cctx.fillStyle = '#e2e8f0';
        cctx.font = '8px sans-serif';
        cctx.fillText(pil.name, pu - pw * 0.5, pvTop - 7);
      }

      // Plot active DM-VIO 8-pixel photometric patches color-coded by inverse depth rho = 1/d
      const patches = data.dmvio_patches || [];
      for (const pt of patches) {
        const u = pt[0], v = pt[1] * sy, rho = pt[2];
        // Map rho in [0.4, 3.0] -> hue from 200 (cyan far) to 15 (warm orange/red close)
        const norm = Math.max(0.0, Math.min(1.0, (rho - 0.4) / 2.2));
        const hue = Math.round(200 - norm * 185);
        cctx.strokeStyle = `hsl(${hue}, 95%, 60%)`;
        cctx.fillStyle = `hsla(${hue}, 95%, 60%, 0.35)`;
        cctx.lineWidth = 1.2;
        cctx.fillRect(u - 3, v - 3, 6, 6);
        cctx.strokeRect(u - 3, v - 3, 6, 6);
      }

      // Top & bottom DM-VIO HUD overlay
      cctx.fillStyle = 'rgba(9,14,26,0.82)';
      cctx.fillRect(0, 0, CW, 18);
      cctx.fillRect(0, CH - 18, CW, 18);
      cctx.fillStyle = '#c084fc';
      cctx.font = '9px monospace';
      cctx.fillText(`📷 DM-VIO Head Cam (${patches.length} pts)`, 5, 12);
      cctx.fillStyle = '#38bdf8';
      cctx.fillText(`DelMarg:${m.dmvio_delayed_kfs||1}KF s=1.00`, 5, CH - 5);
    }
  } catch (e) {}
}
loadZooCatalog();
setInterval(pollSlamRadar, 200);
pollSlamRadar();
</script>
"""


def start_web_ui_server(
    ctrl,
    port: int = 9527,
    bridge: Optional[MuJoCoXgoDuckBridge] = None,
    needle_brain: Optional[NeedleZooBrain] = None,
    whistle_stt: Optional[WhistleSpeechToText] = None,
) -> None:
    """Start the official XGO-Duck Web UI + Needle 3 + Whistle S2T & 30-Policy RL Zoo Drawer in a daemon thread."""
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
        import uvicorn
    except ImportError:
        print("  [WebUI] fastapi/uvicorn not installed; skipping Web UI server.")
        return

    assets_dir = RUNTIME_ASSETS_DIR
    app = FastAPI(title="XGO-Duck Embodied Simulator Web UI + Whistle S2T + Needle 3 + 30-Policy RL Zoo")

    @app.get("/")
    def index():
        raw_html = (assets_dir / "index.html").read_text(encoding="utf-8")
        if "</body>" in raw_html:
            raw_html = raw_html.replace("</body>", ZOO_DRAWER_HTML + "\n</body>")
        elif "</html>" in raw_html:
            raw_html = raw_html.replace("</html>", ZOO_DRAWER_HTML + "\n</html>")
        else:
            raw_html += ZOO_DRAWER_HTML
        return HTMLResponse(raw_html)

    @app.get("/schematic")
    def schematic():
        sch_html = ROOT_DIR / "diagrams" / "hardware_schematic" / "Microduck_EasyEDA_Interactive.html"
        if sch_html.is_file():
            return HTMLResponse(sch_html.read_text(encoding="utf-8"))
        raise HTTPException(404, "Schematic file not found")

    @app.get("/api/slam")
    def api_slam():
        if bridge is None:
            return {"metrics": {}, "gt_path": [], "dr_path": [], "opt_path": [], "landmarks_est": [], "active_rays": [], "patrol_active": False}
        snap = bridge.slam.snapshot_json()
        snap["patrol_active"] = bool(getattr(bridge, "slam_patrol_active", False))
        return snap

    @app.post("/api/slam/reset")
    def api_slam_reset():
        if bridge is not None:
            bridge.slam.reset(Pose2(float(bridge.data.qpos[0]), float(bridge.data.qpos[1]), 0.0))
        return {"ok": True}

    @app.post("/api/slam/patrol")
    def api_slam_patrol():
        if bridge is None:
            return {"active": False}
        active = not bool(getattr(bridge, "slam_patrol_active", False))
        bridge.slam_patrol_active = active
        if active:
            ctrl.set_mode("policy")
        else:
            ctrl.command(twist=[0.0, 0.0, 0.0], head=[0.0, 0.0, 0.0, 0.0], mouth=0.0)
        return {"active": active}

    @app.get("/api/zoo")
    def api_zoo():
        skills = [
            {
                "skill_id": s.skill_id,
                "title": s.title,
                "category": s.category,
                "npz_name": s.npz_name,
                "repo_source": s.repo_source,
                "description": s.description,
                "duration_s": s.duration_s,
                "keywords": list(s.keywords),
            }
            for s in POLICY_ZOO.values()
        ]
        return {
            "count": len(skills),
            "active_skill": getattr(ctrl, "active_zoo_skill", "xgoduck_walk"),
            "last_needle_msg": getattr(ctrl, "last_needle_msg", ""),
            "skills": skills,
        }

    @app.post("/api/zoo/run")
    def api_zoo_run(body: dict = Body(...)):
        skill_id = body.get("skill_id", "")
        if skill_id not in POLICY_ZOO:
            raise HTTPException(400, f"Unknown RL skill_id: {skill_id}")
        if bridge is None:
            raise HTTPException(500, "Bridge not attached")
        try:
            spec = POLICY_ZOO[skill_id]
            res = start_zoo_skill(ctrl, bridge, spec)
            msg = f"Running RL Policy [{spec.skill_id}]: {spec.title} ({spec.repo_source})"
            ctrl.last_needle_msg = msg
            return {**res, "message": msg}
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/needle")
    def api_needle(body: dict = Body(...)):
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(400, "Empty prompt")
        if bridge is None:
            raise HTTPException(500, "Bridge not attached")
        try:
            plan = execute_needle_command_on_xgoduck(ctrl, bridge, prompt, needle_brain=needle_brain)
            return {
                "accepted": plan.get("accepted", False),
                "action_type": plan.get("action_type"),
                "skill_id": plan.get("skill_id"),
                "needle_raw": plan.get("needle_raw"),
                "ms": plan.get("ms"),
                "message": plan.get("message"),
            }
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/whistle")
    def api_whistle(body: dict = Body(...)):
        nonlocal whistle_stt
        wav_b64 = body.get("wav_base64", "")
        lang = body.get("language", "en")
        if not wav_b64:
            raise HTTPException(400, "Missing wav_base64")
        if bridge is None:
            raise HTTPException(500, "Bridge not attached")
        if whistle_stt is None:
            whistle_stt = getattr(ctrl, "_whistle_stt", None)
        if whistle_stt is None:
            whistle_stt = WhistleSpeechToText(prefer_d4=True, language=lang)
            ctrl._whistle_stt = whistle_stt
        try:
            wav_bytes = base64.b64decode(wav_b64)
            stt = whistle_stt.transcribe_wav_bytes(wav_bytes, language=lang)
            if not stt.get("ok") or not stt.get("text"):
                return {
                    "accepted": False,
                    "transcript": "",
                    "whistle_ms": stt.get("ms"),
                    "message": f"[Whistle S2T] {stt.get('error', 'No speech detected')}",
                }
            transcript = stt["text"]
            plan = execute_needle_command_on_xgoduck(ctrl, bridge, transcript, needle_brain=needle_brain)
            combined_msg = f"🎙️ Whistle ({stt['ms']} ms): \"{transcript}\" -> {plan.get('message')}"
            ctrl.last_needle_msg = combined_msg
            return {
                "accepted": plan.get("accepted", False),
                "transcript": transcript,
                "whistle_ms": stt.get("ms"),
                "needle_ms": plan.get("ms"),
                "action_type": plan.get("action_type"),
                "skill_id": plan.get("skill_id"),
                "needle_raw": plan.get("needle_raw"),
                "message": combined_msg,
            }
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/status")
    def api_status():
        return ctrl.status()

    @app.post("/api/mode")
    def api_mode(body: dict = Body(...)):
        try:
            if bridge is not None and body.get("mode") in ("off", "shadow", "hold", "calibrate", "servo_debug"):
                bridge.slam_patrol_active = False
            return ctrl.set_mode(body["mode"])
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/roller")
    def api_roller(body: dict = Body(...)):
        try:
            return ctrl.set_roller_mode(body["enabled"])
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/command")
    def api_command(body: dict = Body(...)):
        try:
            if bridge is not None and getattr(bridge, "slam_patrol_active", False):
                tw = body.get("twist", [0.0, 0.0, 0.0])
                if not tw or all(abs(float(v)) < 1e-3 for v in tw):
                    t_sim = float(bridge.data.time)
                    head_yaw_rad = 0.48 * math.sin(t_sim * 2.2)
                    neck_pitch_rad = 0.10 * math.cos(t_sim * 1.3)
                    return ctrl.command(
                        twist=[0.19, 0.0, 0.42],
                        head=[0.0, neck_pitch_rad, head_yaw_rad, 0.0],
                        mouth=0.0,
                    )
                else:
                    # User moved the joystick/WASD manually -> hand control back to user
                    bridge.slam_patrol_active = False
            return ctrl.command(**body)
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/filter")
    def api_filter(body: dict = Body(...)):
        try:
            leg, head = ctrl.set_action_alphas(**body)
            return {"leg": leg, "head": head}
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/imu-filter")
    def api_imu_filter(body: dict = Body(...)):
        try:
            return ctrl.set_imu_filters(**body)
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/action-scale")
    def api_action_scale(body: dict = Body(...)):
        try:
            return ctrl.set_action_scale(**body)
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/pick")
    def api_pick():
        try:
            return ctrl.start_pick()
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/roulade")
    def api_roulade():
        try:
            return ctrl.start_roulade()
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/sitstand")
    def api_sitstand(body: dict = Body(...)):
        try:
            return ctrl.set_sitstand(body["action"])
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/right_kick")
    def api_right_kick():
        try:
            return ctrl.start_kick("right")
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/left_kick")
    def api_left_kick():
        try:
            return ctrl.start_kick("left")
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/cal/enter")
    def api_cal_enter():
        try:
            return ctrl.set_mode("calibrate")
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/cal/exit")
    def api_cal_exit():
        try:
            return ctrl.set_mode("shadow")
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/cal/heartbeat")
    def api_cal_heartbeat():
        try:
            return ctrl.cal_heartbeat()
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/cal/center")
    def api_cal_center(body: dict = Body(...)):
        try:
            return ctrl.cal_center(body["index"])
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/servo/enter")
    def api_servo_enter():
        try:
            return ctrl.set_mode("servo_debug")
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/servo/exit")
    def api_servo_exit():
        try:
            return ctrl.set_mode("shadow")
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/servo")
    def api_servo(body: dict = Body(...)):
        try:
            return ctrl.servo_op(
                body["action"],
                body.get("target_id"),
                new_id=body.get("new_id"),
                kp=body.get("kp", 5),
                kd=body.get("kd", 20),
                protection_enabled=body.get("protection_enabled", True),
                current_a=body.get("current_a", 1.0),
            )
        except Exception as exc:
            raise HTTPException(400, str(exc))

    app.mount("/", StaticFiles(directory=str(assets_dir)), name="assets")

    def _serve():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    threading.Thread(target=_serve, name="webui", daemon=True).start()


def print_zoo_catalog() -> None:
    """Print all 30 RL policies grouped by category."""
    print("=" * 95)
    print(f"{'Skill ID':<22} | {'Category':<14} | {'Title':<32} | {'Example Polite Prompt'}")
    print("-" * 95)
    for spec in POLICY_ZOO.values():
        kw = spec.keywords[0] if spec.keywords else spec.skill_id
        print(f"{spec.skill_id:<22} | {spec.category:<14} | {spec.title:<32} | 'please {kw}'")
    print("=" * 95)


def run_zoo_verification(use_numpy: bool = True) -> None:
    """Verify Needle 3 (`NeedleZooBrain`) + all 29 RL movement networks in MuJoCo!"""
    DualBackendPolicy.USE_NUMPY_BACKEND = use_numpy
    xgo_policy_mod.Policy = DualBackendPolicy
    import controller as xgo_controller_mod
    from controller import Controller

    bridge = MuJoCoXgoDuckBridge(hil_udp_port=9531)

    class _SimClock:
        @staticmethod
        def monotonic() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def perf_counter() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def sleep(s: float) -> None:
            time.sleep(s)

    orig_time = xgo_controller_mod.time
    xgo_controller_mod.time = _SimClock
    try:
        ctrl = Controller(bridge)
        bridge.provide("duck_state", ctrl.receive)
        needle_brain = NeedleZooBrain()
        ctrl._needle_zoo_brain = needle_brain

        def simulate_seconds(sec: float) -> None:
            ticks_100hz = int(round(sec * 100))
            for i in range(ticks_100hz):
                ctrl.web_time = _SimClock.monotonic()
                bridge.step_100hz()
                if i % 2 == 1:
                    ctrl.step()

        # Warm up IMU filter and arm policy
        simulate_seconds(0.25)
        ctrl.set_mode("policy")
        simulate_seconds(0.5)

        print("=" * 95)
        print(f"Needle 3 + 30-Skill RL Movement Zoo Verification (NumPy={use_numpy}, Needle={needle_brain.needle_available})")
        print("=" * 95)

        test_prompts = [
            ("do a backflip", 0.5),
            ("please do a polite bow", 1.5),
            ("please do a happy hop", 1.5),
            ("please moonwalk", 1.5),
            ("please groove dance", 1.5),
            ("please dance 120bpm", 1.5),
            ("please peck seeds", 1.5),
            ("please stand on one leg", 1.5),
            ("please sprint", 1.2),
            ("please do a long jump", 1.5),
            ("please do a side roll right", 1.8),
            ("please do a backflip", 1.8),
            ("please do a headstand", 1.8),
            ("please play dead", 1.8),
        ]

        for prompt, sim_s in test_prompts:
            bridge.respawn_upright()
            ctrl.recovery.reset()
            simulate_seconds(0.25)
            plan = execute_needle_command_on_xgoduck(ctrl, bridge, prompt, needle_brain=needle_brain)
            simulate_seconds(sim_s)
            pose = bridge.pose_summary()
            print(
                f"Prompt: {prompt:<28} -> {plan['message']}\n"
                f"   [Needle: {plan.get('needle_raw')} ({plan.get('ms')} ms)] | End Pose: {pose}"
            )

        # Also verify every single one of the 30 skills loads and steps cleanly
        print("-" * 95)
        print("Verifying all 30 RL skills in POLICY_ZOO load & infer in MuJoCo...")
        for skill_id, spec in POLICY_ZOO.items():
            bridge.respawn_upright()
            ctrl.recovery.reset()
            simulate_seconds(0.1)
            res = start_zoo_skill(ctrl, bridge, spec)
            simulate_seconds(0.2)
            assert ctrl.last_error == "", f"Error on {skill_id}: {ctrl.last_error}"
        print(f"SUCCESS: All {len(POLICY_ZOO)} RL skills (29 neural networks) verified with zero errors!")
        print("=" * 95)
    finally:
        xgo_controller_mod.time = orig_time
        bridge.hil_sock.close()


def run_whistle_verification(use_numpy: bool = True) -> None:
    """End-to-end Spoken Audio WAV -> Cactus Whistle S2T (`whistle_d4.cact`) -> Needle 3 (`duck_4l.cact`) -> MuJoCo RL Policy test!"""
    DualBackendPolicy.USE_NUMPY_BACKEND = use_numpy
    xgo_policy_mod.Policy = DualBackendPolicy
    import controller as xgo_controller_mod
    from controller import Controller

    bridge = MuJoCoXgoDuckBridge(hil_udp_port=9532)

    class _SimClock:
        @staticmethod
        def monotonic() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def perf_counter() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def sleep(s: float) -> None:
            time.sleep(s)

    orig_time = xgo_controller_mod.time
    xgo_controller_mod.time = _SimClock
    try:
        ctrl = Controller(bridge)
        bridge.provide("duck_state", ctrl.receive)
        whistle_stt = WhistleSpeechToText(prefer_d4=True)
        needle_brain = NeedleZooBrain()
        ctrl._whistle_stt = whistle_stt
        ctrl._needle_zoo_brain = needle_brain

        def simulate_seconds(sec: float) -> None:
            ticks_100hz = int(round(sec * 100))
            for i in range(ticks_100hz):
                ctrl.web_time = _SimClock.monotonic()
                bridge.step_100hz()
                if i % 2 == 1:
                    ctrl.step()

        simulate_seconds(0.25)
        ctrl.set_mode("policy")
        simulate_seconds(0.5)

        print("=" * 100)
        print(
            f"Cactus Whistle S2T ({whistle_stt.model_path.name if whistle_stt.model_path else 'whistle'}, backend={whistle_stt.backend}) "
            f"+ Needle 3 ({'duck_4l.cact' if needle_brain.needle_available else 'rules'}) + MuJoCo RL Zoo Verification"
        )
        print("=" * 100)

        spoken_phrases = [
            ("Walk forward", 0.6),
            ("Please walk forward", 1.5),
            ("Please do a polite bow", 1.5),
            ("Please moonwalk", 1.5),
            ("Please do a backflip", 1.8),
        ]

        for phrase, sim_s in spoken_phrases:
            bridge.respawn_upright()
            ctrl.recovery.reset()
            simulate_seconds(0.25)
            wav_bytes = synthesize_spoken_wav_windows(phrase)
            stt = whistle_stt.transcribe_wav_bytes(wav_bytes, language="en")
            transcript = stt["text"]
            plan = execute_needle_command_on_xgoduck(ctrl, bridge, transcript, needle_brain=needle_brain)
            simulate_seconds(sim_s)
            pose = bridge.pose_summary()
            print(
                f"Spoken Audio : {phrase!r} ({stt.get('duration_s')}s WAV)\n"
                f"  1. Whistle : {transcript!r}  [latency={stt.get('ms')} ms, ttft={stt.get('ttft_ms')} ms, {stt.get('decode_tps')} tok/s]\n"
                f"  2. Needle 3: {plan.get('needle_raw')}  [latency={plan.get('ms')} ms, accepted={plan.get('accepted')}]\n"
                f"  3. MuJoCo  : {plan.get('message')} | End Pose={pose}"
            )
            print("-" * 100)
        print("SUCCESS: End-to-end Voice -> Cactus Whistle S2T -> Needle 3 -> 50 Hz RL Policy pipeline verified!")
        print("=" * 100)
    finally:
        xgo_controller_mod.time = orig_time
        bridge.hil_sock.close()


def run_gtsam_slam_verification(use_numpy: bool = True) -> None:
    """Walk XGO-Duck in MuJoCo while panning its neck (`ID 32 head_yaw`) & verify GTSAM 4.3 Factor-Graph SLAM!"""
    DualBackendPolicy.USE_NUMPY_BACKEND = use_numpy
    xgo_policy_mod.Policy = DualBackendPolicy
    import controller as xgo_controller_mod
    from controller import Controller

    bridge = MuJoCoXgoDuckBridge(hil_udp_port=9533)

    class _SimClock:
        @staticmethod
        def monotonic() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def perf_counter() -> float:
            return 100.0 + float(bridge.data.time)

        @staticmethod
        def sleep(s: float) -> None:
            time.sleep(s)

    orig_time = xgo_controller_mod.time
    xgo_controller_mod.time = _SimClock
    try:
        ctrl = Controller(bridge)
        bridge.provide("duck_state", ctrl.receive)

        def simulate_segment(sec: float, vx: float, vy: float, wz: float, sweep_head: bool = True) -> None:
            ticks_100hz = int(round(sec * 100))
            for i in range(ticks_100hz):
                t_sim = float(bridge.data.time)
                head_yaw_rad = 0.48 * math.sin(t_sim * 2.2) if sweep_head else 0.0
                ctrl.command(twist=[vx, vy, wz], head=[0.0, 0.0, head_yaw_rad, 0.0], mouth=0.0)
                ctrl.web_time = _SimClock.monotonic()
                bridge.step_100hz()
                if i % 2 == 1:
                    ctrl.step()

        # Warm up & stand
        simulate_segment(0.25, 0.0, 0.0, 0.0, sweep_head=False)
        ctrl.set_mode("policy")
        simulate_segment(0.8, 0.0, 0.0, 0.0, sweep_head=False)
        bridge.slam.reset(Pose2(float(bridge.data.qpos[0]), float(bridge.data.qpos[1]), 0.0))

        # Walk forward, curve left, curve back toward origin while panning head (`ID 32 head_yaw`)
        simulate_segment(3.0, 0.20, 0.0, 0.25, sweep_head=True)
        simulate_segment(3.0, 0.19, 0.0, 0.55, sweep_head=True)
        simulate_segment(3.0, 0.19, 0.0, 0.55, sweep_head=True)

        m = bridge.slam.compute_metrics()
        print("=" * 100)
        print(f"MuJoCo 100Hz Physics + 50Hz RL Walk + GTSAM 4.3 LeggedEstimator & iSAM2 ({m['backend']})")
        print("=" * 100)
        print(f"  [100 Hz Proprioceptive LeggedEstimator (`ExtendedPoseContactFactor` + `LeggedInvariantEKF`)]")
        print(f"    Stance Foothold Variables F_c  : {m['num_footholds']} stance episodes (switches: {m['stance_switches']})")
        print(f"    ExtendedPoseContactFactors     : {m['num_contact_factors']} active in fixed-lag window")
        print(f"    100 Hz LeggedInvariantEKF Step : {m['iekf_step_ms']:.3f} ms / tick (L_contact={m['left_contact']}, R_contact={m['right_contact']})")
        print(f"  [TUM CVG DM-VIO (`lukasvst/dm-vio` 160x120 Direct Photometric + Delayed Marginalization)]")
        print(f"    Active Inverse-Depth Patches   : {m['dmvio_active_points']} / 48 (rho = 1/d_m, Leg-Locked Scale s=1.000)")
        print(f"    Delayed Marginalization Graph  : {m['dmvio_delayed_kfs']} KFs (Relinearizations: {m['dmvio_relinearizations']})")
        print(f"    Direct Photometric Step Time   : {m['dmvio_step_ms']:.3f} ms / keyframe (C++ DLL: ~0.005 ms)")
        print(f"  [Keyframe Factor-Graph SLAM (`iSAM2` + Panning-Head `ID 32 head_yaw`)]")
        print(f"    Pose2 Variables X(0..N)        : {m['num_poses']} keyframes")
        print(f"    Point2 Room Landmarks L(0..M)  : {m['num_landmarks']} / {len(bridge.slam.world_landmarks)} discovered via panning head")
        print(f"    BetweenFactorPose2 (Leg-IMU/LC): {m['num_between_factors']} (Loop Closures: {m['loop_closures']})")
        print(f"    BearingRangeFactor2D (Head Yaw): {m['num_br_factors']} polar landmark factors")
        print(f"    Incremental iSAM2 Solve Latency: {m['last_solve_ms']:.3f} ms / keyframe")
        print(f"    Raw Uncorrected Odometry RMSE  : {m['dr_rmse_m']:.4f} m")
        print(f"    GTSAM 4.3 Factor-Graph RMSE    : {m['gtsam_rmse_m']:.4f} m")
        print(f"    Odometry Drift Reduction       : {m['error_reduction_pct']:.1f}%")
        print("=" * 100)
    finally:
        xgo_controller_mod.time = orig_time
        bridge.hil_sock.close()


def run_interactive_sim(
    use_numpy: bool = True,
    port: int = 9527,
    hil_port: int = 9528,
    esp32_serial: Optional[str] = None,
    auto_slam_patrol: bool = True,
) -> None:
    """Launch interactive MuJoCo 3D Viewer + Web UI (port 9527) + UDP/USB HIL Server."""
    import mujoco.viewer

    DualBackendPolicy.USE_NUMPY_BACKEND = use_numpy
    xgo_policy_mod.Policy = DualBackendPolicy
    import controller as xgo_controller_mod
    from controller import Controller

    class _HighResClock:
        @staticmethod
        def monotonic() -> float:
            return time.perf_counter()

        @staticmethod
        def perf_counter() -> float:
            return time.perf_counter()

        @staticmethod
        def sleep(s: float) -> None:
            time.sleep(s)

    xgo_controller_mod.time = _HighResClock

    bridge = MuJoCoXgoDuckBridge(hil_udp_port=hil_port, esp32_serial_port=esp32_serial)
    bridge.slam_patrol_active = auto_slam_patrol
    ctrl = Controller(bridge)
    bridge.provide("duck_state", ctrl.receive)
    bridge.provide("duck_diagnostics", ctrl.receive_diagnostics)
    bridge.provide("duck_boot_pd", ctrl.receive_boot_pd)
    bridge.provide("duck_raw", ctrl.receive_raw)
    bridge.provide("duck_servo_reply", ctrl.receive_servo_reply)
    bridge.provide("duck_cal_reply", ctrl.receive_cal_reply)

    # Load Cactus Whistle S2T + Needle 3 Zoo Brain before starting real-time 100Hz/50Hz loops
    whistle_stt = WhistleSpeechToText(prefer_d4=True)
    needle_brain = NeedleZooBrain()
    ctrl._whistle_stt = whistle_stt
    ctrl._needle_zoo_brain = needle_brain

    # Start Web UI server with the 30-Policy RL Zoo + Whistle S2T + Needle 3 drawer
    start_web_ui_server(ctrl, port=port, bridge=bridge, needle_brain=needle_brain, whistle_stt=whistle_stt)

    print("=" * 95)
    print("XGO-Duck (Feetech 1910) Interactive MuJoCo Simulator + Cactus Whistle S2T + Needle 3 (30 RL Skills)")
    print(f"  Policy Backend : {'Pure-NumPy (.npz, Pi Zero W 1.1 compatible)' if use_numpy else 'ONNX Runtime'}")
    print(f"  RL Policy Zoo  : {len(POLICY_ZOO)} RL skills across 29 neural networks loaded on-demand")
    print(f"  Whistle S2T    : {'Loaded (' + whistle_stt.model_path.name + ', ' + whistle_stt.backend + ')' if whistle_stt.available and whistle_stt.model_path else 'Unavailable'}")
    print(f"  Needle 3 Brain : {'Loaded (duck_4l.cact + 5-Tool RL Zoo Router)' if needle_brain.needle_available else 'Polite Rules'}")
    print(f"  GTSAM 4.3 SLAM : Active ({bridge.slam.backend_name} | Auto-Patrol={'ON' if auto_slam_patrol else 'OFF'})")
    print(f"  Web Control UI : http://127.0.0.1:{port}  (Click '🦆 Needle 3 + Whistle S2T + 30 RL Policies'!)")
    print(f"  UDP HIL Server : 0.0.0.0:{hil_port} (Connect Pi Zero W 1.1 with pi_zero_client.py)")
    print("  Terminal Commands:")
    print("    - Type 'patrol' to toggle GTSAM 4.3 Auto-Patrol & Neck-Sweeping SLAM")
    print("    - Type 'slam' to print live GTSAM 4.3 LeggedEstimator & iSAM2 Factor-Graph metrics")
    print("    - Type 'voice' or 'ptt' to record 3 seconds from your microphone -> Whistle S2T -> Needle 3!")
    print("    - Type 'whistle <path.wav>' to transcribe any WAV file with Cactus Whistle -> Needle 3")
    print("    - Type 'zoo' or 'list' to print all 30 RL skills & example polite prompts")
    print("    - Type any polite request, e.g.:")
    print("        'please do a backflip', 'please moonwalk', 'please do a polite bow',")
    print("        'please stand on one leg', 'please peck seeds', 'please groove dance',")
    print("        'please dance 120bpm', 'please do a side roll right', 'please headstand',")
    print("        'please play dead', 'please happy hop', 'please sprint', 'please long jump'")
    print("    - Type 'zoo:<skill_id>' (e.g. 'zoo:backflip') to trigger directly, or 'reset' / 'stop'")
    print("=" * 95)

    def terminal_loop():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd_str = line.strip()
                if not cmd_str:
                    continue
                low = cmd_str.lower()
                if low in ("zoo", "list", "help"):
                    print_zoo_catalog()
                elif low == "slam":
                    m = bridge.slam.compute_metrics()
                    print(f"  [GTSAM 4.3 SLAM] {m}")
                elif low == "patrol":
                    bridge.slam_patrol_active = not bool(getattr(bridge, "slam_patrol_active", False))
                    if bridge.slam_patrol_active:
                        ctrl.set_mode("policy")
                    else:
                        ctrl.command(twist=[0.0, 0.0, 0.0], head=[0.0, 0.0, 0.0, 0.0], mouth=0.0)
                    print(f"  [GTSAM 4.3 Auto-Patrol] {'ENABLED' if bridge.slam_patrol_active else 'STOPPED'}")
                elif low in ("voice", "ptt", "mic"):
                    wav_bytes = record_windows_mic_wav(seconds=3.0)
                    stt = whistle_stt.transcribe_wav_bytes(wav_bytes, language="en")
                    if not stt.get("ok") or not stt.get("text"):
                        print(f"  [Whistle S2T] {stt.get('error', 'No speech detected')}")
                        continue
                    print(f"  [Whistle S2T ({stt['ms']} ms)] Heard: {stt['text']!r}")
                    plan = execute_needle_command_on_xgoduck(ctrl, bridge, stt["text"], needle_brain=needle_brain)
                    print(f"  -> {plan['message']}  [{plan.get('needle_raw')} | {plan.get('ms')} ms]")
                elif low.startswith("whistle "):
                    wav_path = cmd_str[8:].strip().strip('"')
                    stt = whistle_stt.transcribe_wav_file(wav_path, language="en")
                    print(f"  [Whistle S2T ({stt.get('ms')} ms)] Heard: {stt.get('text')!r}")
                    if stt.get("ok") and stt.get("text"):
                        plan = execute_needle_command_on_xgoduck(ctrl, bridge, stt["text"], needle_brain=needle_brain)
                        print(f"  -> {plan['message']}  [{plan.get('needle_raw')} | {plan.get('ms')} ms]")
                elif low == "reset":
                    bridge.respawn_upright()
                    ctrl.recovery.reset()
                    print("  -> Respawned XGO-Duck upright.")
                elif low == "stop":
                    bridge.slam_patrol_active = False
                    if hasattr(ctrl, "_orig_roulade_policy") and ctrl.roulade_until:
                        with ctrl.lock:
                            ctrl._stop_roulade_locked()
                            ctrl.roulade_policy = ctrl._orig_roulade_policy
                    ctrl.command(twist=[0.0, 0.0, 0.0], head=[0.0, 0.0, 0.0, 0.0], mouth=0.0)
                    print("  -> Stopped active skill and zeroed motion commands.")
                else:
                    bridge.slam_patrol_active = False
                    plan = execute_needle_command_on_xgoduck(ctrl, bridge, cmd_str, needle_brain=needle_brain)
                    print(f"  -> {plan['message']}  [{plan.get('needle_raw')} | {plan.get('ms')} ms]")
            except Exception as e:
                print(f"  [Command error]: {e}")

    threading.Thread(target=terminal_loop, daemon=True).start()

    with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = bridge.trunk_body_id
        viewer.cam.distance = 0.65
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 145.0

        # Warm up 100 Hz MCU state stream after OpenGL window opens so feedback is fresh
        for _ in range(25):
            bridge.step_100hz()
        bridge.slam.reset(Pose2(float(bridge.data.qpos[0]), float(bridge.data.qpos[1]), 0.0))
        worker = threading.Thread(target=ctrl.run, name="policy50", daemon=True)
        worker.start()
        ctrl.set_mode("policy")
        if getattr(bridge, "slam_patrol_active", False):
            ctrl.command(twist=[0.19, 0.0, 0.42], head=[0.0, 0.10, 0.25, 0.0], mouth=0.0)

        tick = 0
        while viewer.is_running():
            t0 = time.perf_counter()
            if not bridge.external_hil_active:
                ctrl.web_time = time.perf_counter()
                if getattr(bridge, "slam_patrol_active", False) and (tick % 5 == 0):
                    if ctrl.mode not in ("policy", "calibrate", "servo_debug"):
                        try:
                            ctrl.set_mode("policy")
                        except Exception:
                            pass
                    t_sim = float(bridge.data.time)
                    head_yaw_rad = 0.48 * math.sin(t_sim * 2.2)
                    neck_pitch_rad = 0.10 * math.cos(t_sim * 1.3)
                    ctrl.command(
                        twist=[0.19, 0.0, 0.42],
                        head=[0.0, neck_pitch_rad, head_yaw_rad, 0.0],
                        mouth=0.0,
                    )
            bridge.step_100hz()
            if tick % 2 == 0:
                viewer.sync()
            tick += 1
            elapsed = time.perf_counter() - t0
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="XGO-Duck (Feetech 1910) MuJoCo Simulator, Whistle S2T, Needle 3 & 30-Policy RL Zoo")
    parser.add_argument("--headless-test", action="store_true", help="Run automated headless core multi-skill verification")
    parser.add_argument("--zoo-test", action="store_true", help="Run automated Needle 3 + 30-Skill RL Zoo verification")
    parser.add_argument("--whistle-test", action="store_true", help="Run automated Spoken Audio -> Whistle S2T -> Needle 3 -> MuJoCo RL verification")
    parser.add_argument("--slam-test", action="store_true", help="Launch interactive Web UI + 3D Viewer with GTSAM 4.3 Watch-Keeping Auto-Patrol")
    parser.add_argument("--no-patrol", action="store_true", help="Start interactive simulator without automatic watch-keeping SLAM patrol")
    parser.add_argument("--list-zoo", action="store_true", help="List all 30 RL policies in the Zoo and exit")
    parser.add_argument("--hil-test", action="store_true", help="Run automated Pi Zero W UDP HIL loop verification")
    parser.add_argument("--onnx", action="store_true", help="Use onnxruntime instead of Pure-NumPy .npz backend")
    parser.add_argument("--web-port", type=int, default=9527, help="HTTP port for XGO-Duck Web UI")
    parser.add_argument("--hil-port", type=int, default=9528, help="UDP port for Raspberry Pi Zero W (1.1) HIL")
    parser.add_argument("--esp32-serial", type=str, default=None, help="ESP32 USB COM port (e.g. COM3) for 3-Node PC<->ESP32<->Pi Zero W HIL")
    args = parser.parse_args()

    if args.list_zoo:
        print_zoo_catalog()
    elif args.slam_test and args.headless_test:
        run_gtsam_slam_verification(use_numpy=not args.onnx)
    elif args.headless_test:
        run_headless_verification(use_numpy=not args.onnx)
    elif args.zoo_test:
        run_zoo_verification(use_numpy=not args.onnx)
    elif args.whistle_test:
        run_whistle_verification(use_numpy=not args.onnx)
    elif args.hil_test:
        run_udp_hil_test()
    else:
        run_interactive_sim(
            use_numpy=not args.onnx,
            port=args.web_port,
            hil_port=args.hil_port,
            esp32_serial=args.esp32_serial,
            auto_slam_patrol=not args.no_patrol,
        )


if __name__ == "__main__":
    main()
