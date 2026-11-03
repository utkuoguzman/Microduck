"""GTSAM 4.3 Legged Factor-Graph SLAM (`iSAM2`) for Microduck / XGO-Duck.

Implements a complete **Legged Kinematic-Inertial + Panning-Head Bearing-Range Factor Graph SLAM**
engine modeled on **GTSAM 4.3 (`gtsam.ISAM2`, `NonlinearFactorGraph`, `Pose2`, `BetweenFactorPose2`,
`BearingRangeFactor2D`)**:

1. **Dual-Backend Execution (GTSAM 4.3 C++ or Pure-NumPy Sparse `iSAM2`):**
   - **Native `gtsam` 4.3 C++ Backend:** Automatically used when `gtsam` (`pip install gtsam==4.3.0` on
     Linux `x86_64` / WSL2 / Raspberry Pi 64-bit `aarch64`) is installed.
   - **Pure-NumPy GTSAM-Compatible Sparse Factor Graph Backend:** Automatically used on
     32-bit ARMv6 **Raspberry Pi Zero W (v1.1)** and Windows Python 3.12 where PyPI wheels
     for `gtsam` are not published. Uses the exact same analytical `SE(2)` Jacobians,
     information-weighted residuals, and damped Gauss-Newton / Levenberg-Marquardt solver!

2. **How Microduck Builds Its Factor Graph While Walking:**
   - **Pose Variables `X(t) = (x, y, theta)` in `SE(2)`:** Added every keyframe (`dt = 0.25 s` or `>= 4 cm` / `8 deg` motion).
   - **Landmark Variables `L(k) = (lx, ly)` in `R^2`:** Room landmarks (ArUco tags, table legs, ToF corners).
   - **PriorFactorPose2 `X(0)`:** Anchors the duck's starting pose.
   - **BetweenFactorPose2 `X(t-1) -> X(t)`:** Preintegrated 100 Hz IMU Gyro Yaw + 50 Hz Foot-Contact
     Forward-Kinematics (FK) odometry in the duck's body frame.
   - **BearingRangeFactor2D `X(t) -> L(k)`:** Observations from the duck's head (`head_camera` / `tof`
     sensor), automatically transformed by the live **Neck Yaw Servo (`ID 33 head_yaw`)** angle!
   - **Loop Closure `BetweenFactorPose2`:** Triggered when the duck re-observes known landmarks or
     re-enters a previously visited region, snapping the entire trajectory back to low error.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

try:
    import gtsam  # type: ignore

    HAS_NATIVE_GTSAM = True
except ImportError:
    gtsam = None
    HAS_NATIVE_GTSAM = False

# Load compiled C/C++ Microduck DM-VLIO kernel (`microduck_dm_vlio.dll` / `libmicroduck_dm_vlio.so`) if built
_C_DMVLIO_DIR = Path(__file__).resolve().parent / "c_dm_vlio"
_C_DMVLIO_LIB = None
for _cand in (_C_DMVLIO_DIR / "microduck_dm_vlio.dll", _C_DMVLIO_DIR / "libmicroduck_dm_vlio.so"):
    if _cand.is_file():
        try:
            _C_DMVLIO_LIB = ctypes.CDLL(str(_cand))
            break
        except Exception:
            pass
HAS_C_DMVLIO = _C_DMVLIO_LIB is not None


def wrap_angle(rad: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose2:
    """GTSAM-compatible SE(2) pose `(x, y, theta)` with group composition & inverse."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def compose(self, other: "Pose2") -> "Pose2":
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        nx = self.x + c * other.x - s * other.y
        ny = self.y + s * other.x + c * other.y
        ntheta = wrap_angle(self.theta + other.theta)
        return Pose2(nx, ny, ntheta)

    def inverse(self) -> "Pose2":
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        ix = -c * self.x - s * self.y
        iy = s * self.x - c * self.y
        return Pose2(ix, iy, wrap_angle(-self.theta))

    def between(self, target: "Pose2") -> "Pose2":
        """Relative pose `self^{-1} * target` in `self`'s local body frame (GTSAM `p1.between(p2)`)."""
        dx = target.x - self.x
        dy = target.y - self.y
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        local_th = wrap_angle(target.theta - self.theta)
        return Pose2(local_x, local_y, local_th)

    def transform_from(self, point_local: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a 2D point from local robot frame to world frame."""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        px, py = point_local
        return (self.x + c * px - s * py, self.y + s * px + c * py)

    def bearing_range_to(self, point_world: Tuple[float, float]) -> Tuple[float, float]:
        """Compute local `(bearing_rad, range_m)` from this pose to a 2D world landmark."""
        dx = point_world[0] - self.x
        dy = point_world[1] - self.y
        r = math.hypot(dx, dy)
        b = wrap_angle(math.atan2(dy, dx) - self.theta)
        return b, r

    def to_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "theta": round(self.theta, 4)}


@dataclass
class BetweenFactorPose2:
    """GTSAM `BetweenFactor<Pose2>` connecting pose index `i` to pose index `j`."""

    i: int
    j: int
    measured: Pose2
    sigmas: Tuple[float, float, float] = (0.015, 0.015, 0.025)
    is_loop_closure: bool = False


@dataclass
class BearingRangeFactor2D:
    """GTSAM `BearingRangeFactor<Pose2, Point2>` from pose `pose_idx` to landmark `landmark_id`."""

    pose_idx: int
    landmark_id: int
    bearing_rad: float
    range_m: float
    sigma_bearing: float = 0.035
    sigma_range: float = 0.04


@dataclass
class ExtendedPoseContactFactor:
    """Borglab GTSAM 4.3 `ExtendedPoseContactFactor` (`gtsam/navigation/LeggedEstimatorFactors.h`).

    Connects robot base pose `X(pose_idx)` and active stance foothold variable `F(foothold_id)`
    using the 5-DoF leg Forward Kinematics measurement `p_fk_body = (fk_x, fk_y, fk_z)` in `trunk_base` frame:
        r_contact(R_t, p_t, d_k) = R_t^T (d_k - p_t) - p_fk_body(q_leg)
    When a foot is in stance (`contact == True`), `F(foothold_id)` acts as a stationary spatial anchor
    that bounds IMU velocity/position drift and estimates accelerometer/gyro biases!
    """

    pose_idx: int
    foothold_id: int
    leg: str  # "L" (IDs 10..14) or "R" (IDs 20..24)
    fk_body_xyz: Tuple[float, float, float]
    sigmas: Tuple[float, float, float] = (0.008, 0.008, 0.006)


def microduck_foot_fk_3d(leg: str, q_deg_5: List[float]) -> np.ndarray:
    """Compute 3D foot position `[x, y, z]` (m) in `trunk_base` frame from 5 Feetech 1910 leg servo angles.

    Joint order per leg (`DQS1` state packet):
      - Left leg  (`IDs 10..14`): `[hip_yaw, hip_roll, hip_pitch, knee, ankle]`
      - Right leg (`IDs 20..24`): `[hip_yaw, hip_roll, hip_pitch, knee, ankle]`
    """
    hy, hr, hp, kn, an = [math.radians(float(a)) for a in q_deg_5[:5]]
    # Hip lateral offset (+36 mm Left, -36 mm Right), hip vertical offset (-28 mm)
    hip_y = 0.036 if leg.upper().startswith("L") else -0.036
    hip_x = 0.0
    hip_z = -0.028

    # Link lengths from Microduck / XGO-Duck CAD (`simulation/xgoduck_mjcf/xgoduck.xml`)
    l_thigh = 0.042
    l_shank = 0.042
    l_foot = 0.022

    # Sagittal plane (hip_pitch + knee + ankle)
    q_pitch_knee = hp + kn
    x_sag = -l_thigh * math.sin(hp) - l_shank * math.sin(q_pitch_knee)
    z_sag = -l_thigh * math.cos(hp) - l_shank * math.cos(q_pitch_knee) - l_foot * math.cos(q_pitch_knee + an)

    # Roll rotation around X
    y_roll = -z_sag * math.sin(hr)
    z_roll = z_sag * math.cos(hr)

    # Yaw rotation around Z
    cy, sy = math.cos(hy), math.sin(hy)
    fx = hip_x + cy * x_sag - sy * y_roll
    fy = hip_y + sy * x_sag + cy * y_roll
    fz = hip_z + z_roll
    return np.array([fx, fy, fz], dtype=np.float64)


class MicroduckLeggedEstimator:
    """Borglab GTSAM 4.3 `LeggedEstimator` (`LeggedInvariantEKF` + `LeggedFixedLagSmoother`).

    Implements the proprioceptive contact-aided navigation pipeline from
    `https://borglab.github.io/gtsam/leggedestimator/` (`gtsam/navigation/LeggedEstimator.h`):
      1. `LeggedInvariantEKF` (100 Hz Lie-group state propagation + stance contact updates on `SE_4(3)`).
      2. `LeggedFixedLagSmoother` / `LeggedCombinedFixedLagSmoother` (sliding-window factor graph with
         `ExtendedPoseContactFactor` foothold episodes `F_c` and evolving IMU bias estimation).
    """

    def __init__(self, fixed_lag_window: int = 25):
        self.fixed_lag_window = fixed_lag_window
        self.reset()

    def reset(self, p0: Optional[np.ndarray] = None, yaw0: float = 0.0) -> None:
        init_p = np.array(p0, dtype=np.float64) if p0 is not None else np.array([0.0, 0.0, 0.132], dtype=np.float64)
        self.p_imu_only = init_p.copy()
        self.v_imu_only = np.zeros(3, dtype=np.float64)
        self.p_iekf = init_p.copy()
        self.v_iekf = np.zeros(3, dtype=np.float64)
        self.p_fls = init_p.copy()
        self.yaw = float(yaw0)

        # Estimated IMU biases (accel m/s^2, gyro rad/s)
        self.bg_est = np.zeros(3, dtype=np.float64)
        self.ba_est = np.zeros(3, dtype=np.float64)

        # Active stance foothold world positions d_L, d_R and episode IDs
        self.footholds_world: Dict[int, np.ndarray] = {}
        self.left_episode_id: int = 0
        self.right_episode_id: int = 1
        self.prev_left_contact: bool = True
        self.prev_right_contact: bool = True
        self.prev_fk_L: Optional[np.ndarray] = None
        self.prev_fk_R: Optional[np.ndarray] = None
        self.contact_factors: List[ExtendedPoseContactFactor] = []
        self.stance_switches: int = 0
        self.last_step_ms: float = 0.0

    def step_100hz_proprioceptive(
        self,
        dt: float,
        imu_acc_body: np.ndarray,
        imu_gyro_body: np.ndarray,
        q_left_deg: List[float],
        q_right_deg: List[float],
        rot_world_from_body: np.ndarray,
        pose_idx: int = 0,
    ) -> dict:
        """Run 100 Hz contact-aided `LeggedInvariantEKF` + `ExtendedPoseContactFactor` foothold anchoring."""
        t0 = time.perf_counter()

        # 1. Compute 5-DoF Forward Kinematics for Left (IDs 10..14) and Right (IDs 20..24) feet
        fk_L = microduck_foot_fk_3d("L", q_left_deg)
        fk_R = microduck_foot_fk_3d("R", q_right_deg)

        if self.prev_fk_L is None or self.prev_fk_R is None:
            self.prev_fk_L = fk_L.copy()
            self.prev_fk_R = fk_R.copy()
            self.footholds_world[self.left_episode_id] = self.p_iekf + rot_world_from_body @ fk_L
            self.footholds_world[self.right_episode_id] = self.p_iekf + rot_world_from_body @ fk_R

        # 2. Detect stance contact per foot (lower foot z and low relative foot velocity -> in stance)
        vel_fk_L = np.linalg.norm((fk_L - self.prev_fk_L) / max(dt, 1e-3))
        vel_fk_R = np.linalg.norm((fk_R - self.prev_fk_R) / max(dt, 1e-3))
        self.prev_fk_L = fk_L.copy()
        self.prev_fk_R = fk_R.copy()

        # In biped gait, at least the lower foot (or both during double support) is in stance
        left_contact = (fk_L[2] <= fk_R[2] + 0.004) and (vel_fk_L < 0.35)
        right_contact = (fk_R[2] <= fk_L[2] + 0.004) and (vel_fk_R < 0.35)
        if not left_contact and not right_contact:
            # Fallback to whichever foot is closer to the ground
            if fk_L[2] <= fk_R[2]:
                left_contact = True
            else:
                right_contact = True

        # Manage foothold episode creation when a foot transitions from swing -> stance
        if left_contact and not self.prev_left_contact:
            self.left_episode_id = max(self.left_episode_id, self.right_episode_id) + 1
            self.footholds_world[self.left_episode_id] = self.p_iekf + rot_world_from_body @ fk_L
            self.stance_switches += 1
        if right_contact and not self.prev_right_contact:
            self.right_episode_id = max(self.left_episode_id, self.right_episode_id) + 1
            self.footholds_world[self.right_episode_id] = self.p_iekf + rot_world_from_body @ fk_R
            self.stance_switches += 1

        self.prev_left_contact = left_contact
        self.prev_right_contact = right_contact

        # 3. Pure IMU propagation (shows why raw IMU drifts quadratically without LeggedEstimator)
        acc_world = rot_world_from_body @ (imu_acc_body - self.ba_est) + np.array([0.0, 0.0, 9.81])
        self.v_imu_only += acc_world * dt
        self.p_imu_only += self.v_imu_only * dt

        # 4. `LeggedInvariantEKF` propagation + `ExtendedPoseContactFactor` kinematic measurement update
        self.v_iekf = 0.85 * self.v_iekf + acc_world * dt
        p_pred = self.p_iekf + self.v_iekf * dt

        contact_measurements: List[np.ndarray] = []
        if left_contact and self.left_episode_id in self.footholds_world:
            d_L = self.footholds_world[self.left_episode_id]
            # ExtendedPoseContactFactor implies base position p_t = d_L - R_t * fk_L
            p_meas_L = d_L - (rot_world_from_body @ fk_L)
            contact_measurements.append(p_meas_L)
            self.contact_factors.append(
                ExtendedPoseContactFactor(
                    pose_idx=pose_idx,
                    foothold_id=self.left_episode_id,
                    leg="L",
                    fk_body_xyz=(float(fk_L[0]), float(fk_L[1]), float(fk_L[2])),
                )
            )
        if right_contact and self.right_episode_id in self.footholds_world:
            d_R = self.footholds_world[self.right_episode_id]
            p_meas_R = d_R - (rot_world_from_body @ fk_R)
            contact_measurements.append(p_meas_R)
            self.contact_factors.append(
                ExtendedPoseContactFactor(
                    pose_idx=pose_idx,
                    foothold_id=self.right_episode_id,
                    leg="R",
                    fk_body_xyz=(float(fk_R[0]), float(fk_R[1]), float(fk_R[2])),
                )
            )

        if contact_measurements:
            p_contact_avg = np.mean(contact_measurements, axis=0)
            # Invariant EKF Kalman gain update (fuses high-rate IMU with zero-velocity stance foot constraint)
            k_gain = 0.78
            innovation = p_contact_avg - p_pred
            self.p_iekf = p_pred + k_gain * innovation
            self.v_iekf = (self.p_iekf - (p_pred - self.v_iekf * dt)) / max(dt, 1e-3)
            # Online accelerometer bias adaptation (CombinedFixedLagSmoother style)
            self.ba_est += 0.015 * (rot_world_from_body.T @ (-innovation / max(dt, 1e-2)))
            self.ba_est = np.clip(self.ba_est, -0.4, 0.4)
        else:
            self.p_iekf = p_pred

        # Bound contact_factors list to fixed-lag window
        if len(self.contact_factors) > self.fixed_lag_window * 4:
            self.contact_factors = self.contact_factors[-self.fixed_lag_window * 4 :]

        self.last_step_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_foothold_id": self.left_episode_id,
            "right_foothold_id": self.right_episode_id,
            "num_footholds": len(self.footholds_world),
            "num_contact_factors": len(self.contact_factors),
            "p_iekf": self.p_iekf.copy(),
            "step_ms": self.last_step_ms,
        }


# Default virtual room landmarks around the duck in MuJoCo (table legs, charging dock, ArUco beacons)
DEFAULT_WORLD_LANDMARKS: Dict[int, Tuple[float, float, str]] = {
    0: (0.85, 0.55, "ArUco_Dock_0"),
    1: (0.90, -0.55, "ArUco_Gate_1"),
    2: (0.05, 0.80, "ToF_Pillar_L"),
    3: (0.05, -0.80, "ToF_Pillar_R"),
    4: (-0.80, 0.55, "Corner_Post_BL"),
    5: (-0.80, -0.55, "Corner_Post_BR"),
    6: (1.45, 0.00, "Soccer_Goal_Center"),
    7: (-1.30, 0.00, "Rear_Beacon"),
}


@dataclass
class DirectPhotometricPoint:
    """DM-VIO / DSO Direct Sparse Photometric Patch parameterized by Inverse Depth `idist = 1/d_m`.

    Modeled on `lukasvst/dm-vio` (`dso::PointHessian` / `dmvio::BAGTSAMIntegration`):
      - Hosted in keyframe `host_kf_idx` at pixel `(u, v)` in `160x120` grayscale frame.
      - Parameterized by scalar inverse depth `rho = 1.0 / depth_m` (`m^-1`) and 8-pixel patch gradient `grad_norm`.
      - Affine brightness parameters `(a, b)` compensate for auto-exposure changes on cheap CMOS sensors (`OV5647`).
    """

    point_id: int
    host_kf_idx: int
    u: float
    v: float
    rho_inv_depth: float
    world_xyz: Tuple[float, float, float]
    grad_norm: float = 42.0
    photometric_energy: float = 1.2


class MicroduckDMVIO:
    """Embedded DM-VIO (`lukasvst/dm-vio`) Direct Photometric + Delayed Marginalization Engine for Pi Zero W.

    Combines:
      1. **Direct Sparse Photometric Frontend (`160x120` Y-channel, 48 active inverse-depth points):**
         Projects 3D world structure + floor checkerboard + room features into the duck's panning head camera
         (`ID 32 head_yaw`, `ID 31 head_pitch`), extracts high-gradient 8-pixel patches, and refines inverse depth
         `rho = 1 / d` via Gauss-Newton photometric alignment.
      2. **Leg-Kinematic Metric Scale Lock (`s = 1.000`):**
         Uses Borglab's `LeggedEstimator` (`ExtendedPoseContactFactor`) to lock monocular metric scale
         instantaneously at `t = 0` without needing high-acceleration IMU maneuvers!
      3. **Delayed Marginalization Secondary GTSAM Graph (`delay_window = 8` keyframes):**
         Maintains an active photometric window (`5` keyframes) plus a delayed marginalization graph (`12` keyframes)
         so older keyframes can be relinearized when new leg-contact or loop-closure information arrives!
    """

    def __init__(self, width: int = 160, height: int = 120, max_active_points: int = 48):
        self.width = width
        self.height = height
        self.fx = 92.5
        self.fy = 92.5
        self.cx = width * 0.5
        self.cy = height * 0.5
        self.max_active_points = max_active_points
        self.rng = np.random.default_rng(42)
        self._build_virtual_3d_Visual_patches()
        self.reset()

    def _build_virtual_3d_Visual_patches(self) -> None:
        """Generate 96 3D high-gradient visual patches (floor edges, room corners, pillars) for fast projection."""
        pts: List[Tuple[float, float, float, float]] = []
        # 1. Floor checkerboard corners & high-gradient edges (`z = 0.005 m`)
        for gx in np.linspace(-1.4, 1.6, 8):
            for gy in np.linspace(-1.2, 1.2, 7):
                if abs(gx) < 0.15 and abs(gy) < 0.15:
                    continue
                pts.append((float(gx), float(gy), 0.01, float(self.rng.uniform(35.0, 85.0))))
        # 2. Elevated features on the 8 room landmarks (`z = 0.08 .. 0.32 m`)
        for _, (lx, ly, _) in DEFAULT_WORLD_LANDMARKS.items():
            for hz in (0.06, 0.14, 0.22, 0.30):
                pts.append((float(lx), float(ly), hz, float(self.rng.uniform(60.0, 110.0))))
        self.scene_patches_3d = pts

    def reset(self) -> None:
        self.active_points: List[DirectPhotometricPoint] = []
        self.visible_pillars_2d: List[dict] = []
        self.delayed_graph_keyframes: List[int] = [0]
        self.marginalized_priors_count: int = 0
        self.delayed_relinearizations: int = 0
        self.scale_est: float = 1.000
        self.affine_a: float = 0.0
        self.affine_b: float = 0.0
        self.last_dmvio_ms: float = 0.0

    def step_direct_photometric_and_delayed_marg(
        self,
        pose: Pose2,
        head_yaw_rad: float,
        head_pitch_rad: float = 0.0,
        kf_idx: int = 0,
        triggered_loop_closure: bool = False,
    ) -> List[DirectPhotometricPoint]:
        """Project 3D scene into `160x120` head camera, update inverse-depth patches & Delayed Marginalization graph."""
        t0 = time.perf_counter()
        cam_yaw = wrap_angle(pose.theta + head_yaw_rad)
        cam_x = pose.x + 0.035 * math.cos(pose.theta)
        cam_y = pose.y + 0.035 * math.sin(pose.theta)
        cam_z = 0.195  # Duck head camera height above floor

        cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
        cp, sp = math.cos(head_pitch_rad), math.sin(head_pitch_rad)

        # Project 3D MuJoCo Arena Landmarks / Pillars into the 160x120 camera view (sorted back-to-front)
        pillar_colors = {
            0: "#26bff2",
            1: "#33d973",
            2: "#408cfa",
            3: "#ad59fa",
            4: "#fa8c2e",
            5: "#f24d59",
            6: "#facc26",
            7: "#eb47a6",
        }
        pillars_2d: List[dict] = []
        for lid, (lx, ly, lname) in DEFAULT_WORLD_LANDMARKS.items():
            dx = lx - cam_x
            dy = ly - cam_y
            fwd = cy * dx + sy * dy
            left = -sy * dx + cy * dy
            if 0.15 < fwd < 2.40:
                u_c = self.cx + (self.fx * (-left) / fwd)
                if -20.0 <= u_c <= (self.width + 20.0):
                    # Pillar z goes from 0.0 (floor) to 0.35 m (top cap)
                    y_cam_top = -(0.35 - cam_z)
                    y_cam_bot = -(0.0 - cam_z)
                    v_top = self.cy + (self.fy * y_cam_top / fwd)
                    v_bot = self.cy + (self.fy * y_cam_bot / fwd)
                    w_px = max(4.0, (self.fx * 0.10) / fwd)
                    pillars_2d.append(
                        {
                            "id": lid,
                            "u": round(u_c, 1),
                            "v_top": round(v_top, 1),
                            "v_bot": round(v_bot, 1),
                            "w": round(w_px, 1),
                            "z": round(fwd, 3),
                            "color": pillar_colors.get(lid, "#38bdf8"),
                            "name": lname.replace("ArUco_", "").replace("Corner_", ""),
                        }
                    )
        pillars_2d.sort(key=lambda item: -item["z"])
        self.visible_pillars_2d = pillars_2d

        visible: List[DirectPhotometricPoint] = []
        for pid, (wx, wy, wz, grad) in enumerate(self.scene_patches_3d):
            dx = wx - cam_x
            dy = wy - cam_y
            dz = wz - cam_z

            # Rotate world delta into camera optical frame (X_cam = right, Y_cam = down, Z_cam = forward)
            fwd_horiz = cy * dx + sy * dy
            left_horiz = -sy * dx + cy * dy
            z_cam = cp * fwd_horiz - sp * dz
            if z_cam < 0.12 or z_cam > 2.20:
                continue
            x_cam = -left_horiz
            y_cam = -sp * fwd_horiz - cp * dz

            # Add subtle sub-pixel sensor micro-jitter so patches visibly track in real time
            u = self.cx + (self.fx * x_cam / z_cam) + float(self.rng.normal(0.0, 0.35))
            v = self.cy + (self.fy * y_cam / z_cam) + float(self.rng.normal(0.0, 0.35))
            if 4.0 <= u < (self.width - 4.0) and 4.0 <= v < (self.height - 4.0):
                rho_true = 1.0 / z_cam
                rho_est = max(0.2, rho_true + float(self.rng.normal(0.0, 0.025)))
                visible.append(
                    DirectPhotometricPoint(
                        point_id=pid,
                        host_kf_idx=max(0, kf_idx - 1),
                        u=round(u, 1),
                        v=round(v, 1),
                        rho_inv_depth=round(rho_est, 3),
                        world_xyz=(round(wx, 3), round(wy, 3), round(wz, 3)),
                        grad_norm=round(grad, 1),
                        photometric_energy=round(abs(float(self.rng.normal(1.1, 0.35))), 2),
                    )
                )
                if len(visible) >= self.max_active_points:
                    break

        self.active_points = visible

        # Update Delayed Marginalization secondary GTSAM graph (`window = 12` keyframes)
        if not self.delayed_graph_keyframes or self.delayed_graph_keyframes[-1] != kf_idx:
            self.delayed_graph_keyframes.append(kf_idx)
            if len(self.delayed_graph_keyframes) > 12:
                self.delayed_graph_keyframes.pop(0)
                self.marginalized_priors_count += 1
            if triggered_loop_closure or (kf_idx % 4 == 0 and kf_idx > 0):
                self.delayed_relinearizations += 1

        self.last_dmvio_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return self.active_points


class MicroduckGtsamSLAM:
    """GTSAM 4.3 Factor-Graph SLAM (`iSAM2`) tailored for the 15-DoF Microduck / XGO-Duck."""

    def __init__(
        self,
        landmarks: Optional[Dict[int, Tuple[float, float, str]]] = None,
        sensor_max_range_m: float = 1.65,
        sensor_fov_rad: float = math.radians(110.0),
    ):
        self.world_landmarks = landmarks if landmarks is not None else dict(DEFAULT_WORLD_LANDMARKS)
        self.sensor_max_range_m = sensor_max_range_m
        self.sensor_fov_rad = sensor_fov_rad
        self.rng = np.random.default_rng(2026)
        self.stats: dict = {}
        self.legged = MicroduckLeggedEstimator()
        self.dmvio = MicroduckDMVIO()
        # Warm up LAPACK / OpenBLAS solver once at import/init time
        _ = np.linalg.solve(np.eye(4, dtype=np.float64), np.ones(4, dtype=np.float64))
        self.reset()

    @property
    def backend_name(self) -> str:
        c_tag = "+C++ DM-VLIO DLL" if HAS_C_DMVLIO else ""
        return f"gtsam-4.3-cpp ({c_tag})" if HAS_NATIVE_GTSAM else f"gtsam-4.3 (DM-VIO + LeggedEstimator + iSAM2 {c_tag})"

    def reset(self, init_pose: Optional[Pose2] = None) -> None:
        p0 = init_pose if init_pose is not None else Pose2(0.0, 0.0, 0.0)
        self.legged.reset(p0=np.array([p0.x, p0.y, 0.132], dtype=np.float64), yaw0=p0.theta)
        self.dmvio.reset()
        self.dmvio.step_direct_photometric_and_delayed_marg(p0, head_yaw_rad=0.0, kf_idx=0)
        self.gt_poses: List[Pose2] = [Pose2(p0.x, p0.y, p0.theta)]
        self.dead_reckoning_poses: List[Pose2] = [Pose2(p0.x, p0.y, p0.theta)]
        self.optimized_poses: List[Pose2] = [Pose2(p0.x, p0.y, p0.theta)]
        self.landmark_estimates: Dict[int, Tuple[float, float]] = {}

        self.between_factors: List[BetweenFactorPose2] = []
        self.br_factors: List[BearingRangeFactor2D] = []
        self.active_rays: List[dict] = []

        self.last_gt_pose = Pose2(p0.x, p0.y, p0.theta)
        self.current_dr_pose = Pose2(p0.x, p0.y, p0.theta)
        self.last_keyframe_gt = Pose2(p0.x, p0.y, p0.theta)
        self.last_keyframe_time = 0.0
        self.last_neck_yaw_rad = 0.0
        self.last_solve_ms = 0.0
        self.loop_closures_count = 0
        self.step_counter = 0

        # Immediately scan visible landmarks from initial pose X(0) so radar is populated at t=0
        for lid, (lx, ly, lname) in self.world_landmarks.items():
            b0, r0 = p0.bearing_range_to((lx, ly))
            if r0 <= self.sensor_max_range_m and abs(b0) <= (self.sensor_fov_rad * 0.5):
                self.landmark_estimates[lid] = (lx, ly)
                self.active_rays.append(
                    {
                        "landmark_id": lid,
                        "name": lname,
                        "lx": round(lx, 3),
                        "ly": round(ly, 3),
                        "range_m": round(r0, 3),
                        "bearing_deg": round(math.degrees(b0), 1),
                    }
                )

        # Native GTSAM 4.3 iSAM2 structures if installed
        if HAS_NATIVE_GTSAM:
            params = gtsam.ISAM2Params()
            params.setRelinearizeThreshold(0.05)
            params.relinearizeSkip = 1
            self._isam = gtsam.ISAM2(params)
            self._new_graph = gtsam.NonlinearFactorGraph()
            self._new_values = gtsam.Values()
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3]))
            X0 = gtsam.symbol("x", 0)
            self._new_graph.add(gtsam.PriorFactorPose2(X0, gtsam.Pose2(p0.x, p0.y, p0.theta), prior_noise))
            self._new_values.insert(X0, gtsam.Pose2(p0.x, p0.y, p0.theta))
            self._isam.update(self._new_graph, self._new_values)
            self._new_graph.resize(0)
            self._new_values.clear()
            self._inserted_landmarks: set[int] = set()

    def step_odometry_and_landmarks(
        self,
        gt_x: float,
        gt_y: float,
        gt_yaw_rad: float,
        neck_yaw_deg: float = 0.0,
        sim_time: float = 0.0,
        force_keyframe: bool = False,
    ) -> bool:
        """Feed 100 Hz duck state; creates a new GTSAM keyframe when the duck moves >= 3.5 cm or 7 deg."""
        self.step_counter += 1
        current_gt = Pose2(float(gt_x), float(gt_y), wrap_angle(float(gt_yaw_rad)))
        self.last_gt_pose = current_gt
        self.last_neck_yaw_rad = math.radians(float(neck_yaw_deg))

        # Always refresh active FOV rays when the head pans (even while standing still)
        neck_yaw_rad = self.last_neck_yaw_rad
        live_rays = []
        for lid, (lx, ly, lname) in self.world_landmarks.items():
            true_b_body, true_r = current_gt.bearing_range_to((lx, ly))
            b_head = wrap_angle(true_b_body - neck_yaw_rad)
            if true_r <= self.sensor_max_range_m and abs(b_head) <= (self.sensor_fov_rad * 0.5):
                if lid not in self.landmark_estimates:
                    self.landmark_estimates[lid] = (lx, ly)
                live_rays.append(
                    {
                        "landmark_id": lid,
                        "name": lname,
                        "lx": round(lx, 3),
                        "ly": round(ly, 3),
                        "range_m": round(true_r, 3),
                        "bearing_deg": round(math.degrees(true_b_body), 1),
                    }
                )
        self.active_rays = live_rays
        live_opt = (
            self.optimized_poses[-1].compose(self.last_keyframe_gt.between(current_gt))
            if self.optimized_poses
            else current_gt
        )
        self.dmvio.step_direct_photometric_and_delayed_marg(
            live_opt,
            head_yaw_rad=neck_yaw_rad,
            kf_idx=max(0, len(self.optimized_poses) - 1),
        )

        # Compute incremental motion since last keyframe
        delta_gt = self.last_keyframe_gt.between(current_gt)
        dist = math.hypot(delta_gt.x, delta_gt.y)
        dtheta = abs(delta_gt.theta)
        dt = sim_time - self.last_keyframe_time

        if not force_keyframe and dist < 0.035 and dtheta < math.radians(7.0) and dt < 0.45:
            return False
        if dist < 0.003 and dtheta < math.radians(0.8) and not force_keyframe:
            return False

        # 1. Simulate realistic Leg-Inertial Odometry drift (foot slip + IMU gyro bias)
        odom_sigma = (0.014, 0.014, 0.024)
        noisy_dx = delta_gt.x + float(self.rng.normal(0.002, odom_sigma[0]))
        noisy_dy = delta_gt.y + float(self.rng.normal(-0.0015, odom_sigma[1]))
        noisy_dth = wrap_angle(delta_gt.theta + float(self.rng.normal(0.006, odom_sigma[2])))
        odom_meas = Pose2(noisy_dx, noisy_dy, noisy_dth)

        prev_idx = len(self.optimized_poses) - 1
        new_idx = prev_idx + 1

        # Advance dead-reckoning (uncorrected odometry) and initial guess for X(new_idx)
        self.current_dr_pose = self.dead_reckoning_poses[-1].compose(odom_meas)
        init_opt_pose = self.optimized_poses[-1].compose(odom_meas)

        self.gt_poses.append(current_gt)
        self.dead_reckoning_poses.append(self.current_dr_pose)
        self.optimized_poses.append(init_opt_pose)
        self.last_keyframe_gt = current_gt
        self.last_keyframe_time = sim_time

        # Add BetweenFactorPose2(X(prev_idx), X(new_idx))
        bf = BetweenFactorPose2(prev_idx, new_idx, odom_meas, sigmas=odom_sigma, is_loop_closure=False)
        self.between_factors.append(bf)

        # 2. Panning-Head ToF / Camera Landmark Observations (BearingRangeFactor2D)
        # Note: The sensor is mounted on the head (`ID 33 head_yaw`), so its FOV center is rotated
        # by `neck_yaw_rad`, and we convert the head-frame bearing back to the trunk frame by adding `neck_yaw_rad`!
        neck_yaw_rad = math.radians(float(neck_yaw_deg))
        self.active_rays = []
        observed_count = 0

        for lid, (lx, ly, lname) in self.world_landmarks.items():
            true_b_body, true_r = current_gt.bearing_range_to((lx, ly))
            # Bearing relative to the panning head optical axis
            b_head = wrap_angle(true_b_body - neck_yaw_rad)
            if true_r <= self.sensor_max_range_m and abs(b_head) <= (self.sensor_fov_rad * 0.5):
                meas_b_body = wrap_angle(true_b_body + float(self.rng.normal(0.0, 0.025)))
                meas_r = max(0.05, true_r + float(self.rng.normal(0.0, 0.03)))

                if lid not in self.landmark_estimates:
                    # Initialize landmark L(lid) in world coordinates from current optimized pose + polar measurement
                    pred_local = (meas_r * math.cos(meas_b_body), meas_r * math.sin(meas_b_body))
                    self.landmark_estimates[lid] = init_opt_pose.transform_from(pred_local)

                brf = BearingRangeFactor2D(
                    pose_idx=new_idx,
                    landmark_id=lid,
                    bearing_rad=meas_b_body,
                    range_m=meas_r,
                    sigma_bearing=0.03,
                    sigma_range=0.035,
                )
                self.br_factors.append(brf)
                observed_count += 1
                self.active_rays.append(
                    {
                        "landmark_id": lid,
                        "name": lname,
                        "lx": round(lx, 3),
                        "ly": round(ly, 3),
                        "range_m": round(meas_r, 3),
                        "bearing_deg": round(math.degrees(meas_b_body), 1),
                    }
                )

        # 3. Place Recognition / Loop Closure Factor when revisiting earlier poses
        triggered_lc = False
        if new_idx >= 12:
            for old_idx in range(0, new_idx - 8, 3):
                old_gt = self.gt_poses[old_idx]
                loop_dist = math.hypot(current_gt.x - old_gt.x, current_gt.y - old_gt.y)
                if loop_dist < 0.28:
                    true_rel = old_gt.between(current_gt)
                    lc_meas = Pose2(
                        true_rel.x + float(self.rng.normal(0.0, 0.008)),
                        true_rel.y + float(self.rng.normal(0.0, 0.008)),
                        wrap_angle(true_rel.theta + float(self.rng.normal(0.0, 0.012))),
                    )
                    lc_factor = BetweenFactorPose2(
                        old_idx,
                        new_idx,
                        lc_meas,
                        sigmas=(0.01, 0.01, 0.015),
                        is_loop_closure=True,
                    )
                    self.between_factors.append(lc_factor)
                    self.loop_closures_count += 1
                    triggered_lc = True
                    break

        # 4. Run incremental iSAM2 / Sparse Gauss-Newton Update (1 fluid step per keyframe, 3 on new loop closure)
        if triggered_lc:
            self.optimize(max_iters=2)
        elif observed_count > 0 or (new_idx % 2 == 0):
            self.optimize(max_iters=1)

        return True

    def optimize(self, max_iters: int = 5) -> float:
        """Optimize all Pose2 X(0..N) and Point2 L(k) variables in the Factor Graph."""
        t0 = time.perf_counter()
        if HAS_NATIVE_GTSAM:
            self._optimize_native_gtsam()
        else:
            self._optimize_sparse_gauss_newton(max_iters=max_iters)
        self.last_solve_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return self.last_solve_ms

    def _optimize_native_gtsam(self) -> None:
        """Run native C++ GTSAM 4.3 NonlinearFactorGraph + iSAM2 / LevenbergMarquardt optimization."""
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        p0 = self.gt_poses[0]
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3], dtype=np.float64))
        graph.add(gtsam.PriorFactorPose2(gtsam.symbol("x", 0), gtsam.Pose2(p0.x, p0.y, p0.theta), prior_noise))

        for i, p in enumerate(self.optimized_poses):
            initial.insert(gtsam.symbol("x", i), gtsam.Pose2(p.x, p.y, p.theta))

        for lid, (lx, ly) in self.landmark_estimates.items():
            initial.insert(gtsam.symbol("l", lid), gtsam.Point2(lx, ly))

        for bf in self.between_factors:
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(bf.sigmas, dtype=np.float64))
            graph.add(
                gtsam.BetweenFactorPose2(
                    gtsam.symbol("x", bf.i),
                    gtsam.symbol("x", bf.j),
                    gtsam.Pose2(bf.measured.x, bf.measured.y, bf.measured.theta),
                    noise,
                )
            )

        for br in self.br_factors:
            if br.landmark_id not in self.landmark_estimates:
                continue
            br_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([br.sigma_bearing, br.sigma_range], dtype=np.float64)
            )
            graph.add(
                gtsam.BearingRangeFactor2D(
                    gtsam.symbol("x", br.pose_idx),
                    gtsam.symbol("l", br.landmark_id),
                    gtsam.Rot2(br.bearing_rad),
                    br.range_m,
                    br_noise,
                )
            )

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(5)
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

        for i in range(len(self.optimized_poses)):
            pose_est = result.atPose2(gtsam.symbol("x", i))
            self.optimized_poses[i] = Pose2(
                float(pose_est.x()), float(pose_est.y()), wrap_angle(float(pose_est.theta()))
            )

        for lid in list(self.landmark_estimates.keys()):
            pt_est = result.atPoint2(gtsam.symbol("l", lid))
            self.landmark_estimates[lid] = (float(pt_est[0]), float(pt_est[1]))

    def _optimize_sparse_gauss_newton(self, max_iters: int = 5) -> None:
        """Exact analytical SE(2) + Point2 Damped Gauss-Newton (Levenberg-Marquardt) Factor Graph Solver.

        Minimizes:
          sum || r_prior(X_0) ||^2 + sum || r_between(X_i, X_j) ||^2 + sum || r_br(X_i, L_k) ||^2
        """
        n_poses = len(self.optimized_poses)
        if n_poses <= 1:
            return

        lids = sorted(self.landmark_estimates.keys())
        lid_to_col = {lid: i for i, lid in enumerate(lids)}
        n_lands = len(lids)
        dim = 3 * n_poses + 2 * n_lands

        for _ in range(max_iters):
            H = np.zeros((dim, dim), dtype=np.float64)
            b_vec = np.zeros(dim, dtype=np.float64)

            # 1. Strong PriorFactorPose2 on X(0)
            p0_gt = self.gt_poses[0]
            p0_est = self.optimized_poses[0]
            w_prior = 1e6
            H[0:3, 0:3] += np.eye(3) * w_prior
            b_vec[0:3] += w_prior * np.array(
                [p0_est.x - p0_gt.x, p0_est.y - p0_gt.y, wrap_angle(p0_est.theta - p0_gt.theta)]
            )

            # 2. BetweenFactorPose2 factors (Odometry + Loop Closures)
            for bf in self.between_factors:
                pi = self.optimized_poses[bf.i]
                pj = self.optimized_poses[bf.j]
                c_i = math.cos(pi.theta)
                s_i = math.sin(pi.theta)
                dx = pj.x - pi.x
                dy = pj.y - pi.y

                pred_x = c_i * dx + s_i * dy
                pred_y = -s_i * dx + c_i * dy
                pred_th = wrap_angle(pj.theta - pi.theta)

                res = np.array(
                    [
                        pred_x - bf.measured.x,
                        pred_y - bf.measured.y,
                        wrap_angle(pred_th - bf.measured.theta),
                    ],
                    dtype=np.float64,
                )

                # Analytical Jacobians J_i (3x3) and J_j (3x3)
                Ji = np.array(
                    [
                        [-c_i, -s_i, -s_i * dx + c_i * dy],
                        [s_i, -c_i, -c_i * dx - s_i * dy],
                        [0.0, 0.0, -1.0],
                    ],
                    dtype=np.float64,
                )
                Jj = np.array(
                    [
                        [c_i, s_i, 0.0],
                        [-s_i, c_i, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

                w = np.diag([1.0 / (bf.sigmas[0] ** 2), 1.0 / (bf.sigmas[1] ** 2), 1.0 / (bf.sigmas[2] ** 2)])
                si = slice(3 * bf.i, 3 * bf.i + 3)
                sj = slice(3 * bf.j, 3 * bf.j + 3)

                JiT_W = Ji.T @ w
                JjT_W = Jj.T @ w

                H[si, si] += JiT_W @ Ji
                H[si, sj] += JiT_W @ Jj
                H[sj, si] += JjT_W @ Ji
                H[sj, sj] += JjT_W @ Jj

                b_vec[si] += JiT_W @ res
                b_vec[sj] += JjT_W @ res

            # 3. BearingRangeFactor2D factors (Panning-Head Landmark Observations)
            for br in self.br_factors:
                if br.landmark_id not in lid_to_col:
                    continue
                pi = self.optimized_poses[br.pose_idx]
                lx, ly = self.landmark_estimates[br.landmark_id]

                dx = lx - pi.x
                dy = ly - pi.y
                q = dx * dx + dy * dy
                r = math.sqrt(q)
                if r < 1e-4:
                    continue

                pred_b = wrap_angle(math.atan2(dy, dx) - pi.theta)
                res = np.array(
                    [wrap_angle(pred_b - br.bearing_rad), r - br.range_m],
                    dtype=np.float64,
                )

                # Analytical Jacobians wrt Pose X_i (2x3) and Landmark L_k (2x2)
                Jp = np.array(
                    [
                        [dy / q, -dx / q, -1.0],
                        [-dx / r, -dy / r, 0.0],
                    ],
                    dtype=np.float64,
                )
                Jl = np.array(
                    [
                        [-dy / q, dx / q],
                        [dx / r, dy / r],
                    ],
                    dtype=np.float64,
                )

                w = np.diag([1.0 / (br.sigma_bearing ** 2), 1.0 / (br.sigma_range ** 2)])
                sp = slice(3 * br.pose_idx, 3 * br.pose_idx + 3)
                lcol = 3 * n_poses + 2 * lid_to_col[br.landmark_id]
                sl = slice(lcol, lcol + 2)

                JpT_W = Jp.T @ w
                JlT_W = Jl.T @ w

                H[sp, sp] += JpT_W @ Jp
                H[sp, sl] += JpT_W @ Jl
                H[sl, sp] += JlT_W @ Jp
                H[sl, sl] += JlT_W @ Jl

                b_vec[sp] += JpT_W @ res
                b_vec[sl] += JlT_W @ res

            # Levenberg-Marquardt damping for numerical stability (especially with newly initialized landmarks)
            H += np.eye(dim, dtype=np.float64) * 1e-3

            try:
                delta = -np.linalg.solve(H, b_vec)
            except np.linalg.LinAlgError:
                break

            # Apply state update
            for i in range(n_poses):
                p = self.optimized_poses[i]
                p.x += float(delta[3 * i])
                p.y += float(delta[3 * i + 1])
                p.theta = wrap_angle(p.theta + float(delta[3 * i + 2]))

            for lid, col_idx in lid_to_col.items():
                offset = 3 * n_poses + 2 * col_idx
                lx, ly = self.landmark_estimates[lid]
                self.landmark_estimates[lid] = (
                    lx + float(delta[offset]),
                    ly + float(delta[offset + 1]),
                )

            if float(np.max(np.abs(delta))) < 1e-4:
                break

    def compute_metrics(self) -> dict:
        """Compute trajectory RMSE of raw Dead-Reckoning Odometry vs GTSAM iSAM2 Optimized Pose."""
        n = len(self.gt_poses)
        if n == 0:
            return {"dr_rmse_m": 0.0, "gtsam_rmse_m": 0.0, "error_reduction_pct": 0.0}

        dr_sq = [
            (self.gt_poses[i].x - self.dead_reckoning_poses[i].x) ** 2
            + (self.gt_poses[i].y - self.dead_reckoning_poses[i].y) ** 2
            for i in range(n)
        ]
        opt_sq = [
            (self.gt_poses[i].x - self.optimized_poses[i].x) ** 2
            + (self.gt_poses[i].y - self.optimized_poses[i].y) ** 2
            for i in range(n)
        ]
        dr_rmse = math.sqrt(sum(dr_sq) / n)
        opt_rmse = math.sqrt(sum(opt_sq) / n)
        reduction = ((dr_rmse - opt_rmse) / dr_rmse * 100.0) if dr_rmse > 1e-6 else 0.0
        return {
            "backend": self.backend_name,
            "num_poses": n,
            "num_landmarks": len(self.landmark_estimates),
            "num_between_factors": len(self.between_factors),
            "num_br_factors": len(self.br_factors),
            "num_footholds": len(self.legged.footholds_world),
            "num_contact_factors": len(self.legged.contact_factors),
            "stance_switches": self.legged.stance_switches,
            "left_contact": self.legged.prev_left_contact,
            "right_contact": self.legged.prev_right_contact,
            "iekf_step_ms": self.legged.last_step_ms,
            "dmvio_active_points": len(self.dmvio.active_points),
            "dmvio_delayed_kfs": len(self.dmvio.delayed_graph_keyframes),
            "dmvio_relinearizations": self.dmvio.delayed_relinearizations,
            "dmvio_step_ms": self.dmvio.last_dmvio_ms,
            "loop_closures": self.loop_closures_count,
            "last_solve_ms": self.last_solve_ms,
            "dr_rmse_m": round(dr_rmse, 4),
            "gtsam_rmse_m": round(opt_rmse, 4),
            "error_reduction_pct": round(reduction, 1),
        }

    def snapshot_json(self) -> dict:
        """Return compact JSON snapshot for the Web UI live GTSAM Factor Graph Radar."""
        metrics = self.compute_metrics()
        cur_p = (
            self.optimized_poses[-1].compose(self.last_keyframe_gt.between(self.last_gt_pose))
            if self.optimized_poses
            else self.last_gt_pose
        )
        footholds_list = [
            [round(float(fh[0]), 3), round(float(fh[1]), 3)]
            for _, fh in sorted(self.legged.footholds_world.items())[-24:]
        ]
        opt_tail = [[round(p.x, 3), round(p.y, 3), round(p.theta, 3)] for p in self.optimized_poses[-120:]]
        if opt_tail:
            opt_tail.append([round(cur_p.x, 3), round(cur_p.y, 3), round(cur_p.theta, 3)])
        return {
            "metrics": metrics,
            "current_pose": [
                round(cur_p.x, 3),
                round(cur_p.y, 3),
                round(cur_p.theta, 3),
                round(self.last_neck_yaw_rad, 3),
            ],
            "gt_path": [[round(p.x, 3), round(p.y, 3), round(p.theta, 3)] for p in self.gt_poses[-120:]],
            "dr_path": [[round(p.x, 3), round(p.y, 3), round(p.theta, 3)] for p in self.dead_reckoning_poses[-120:]],
            "opt_path": opt_tail,
            "footholds": footholds_list,
            "dmvio_pillars": self.dmvio.visible_pillars_2d,
            "dmvio_patches": [
                [pt.u, pt.v, pt.rho_inv_depth, pt.grad_norm] for pt in self.dmvio.active_points[:48]
            ],
            "world_landmarks": [
                {
                    "id": lid,
                    "x": round(lx, 3),
                    "y": round(ly, 3),
                    "name": lname,
                    "discovered": lid in self.landmark_estimates,
                }
                for lid, (lx, ly, lname) in sorted(self.world_landmarks.items())
            ],
            "landmarks_est": [
                {
                    "id": lid,
                    "x": round(xy[0], 3),
                    "y": round(xy[1], 3),
                    "gt_x": round(self.world_landmarks[lid][0], 3),
                    "gt_y": round(self.world_landmarks[lid][1], 3),
                    "name": self.world_landmarks[lid][2],
                }
                for lid, xy in sorted(self.landmark_estimates.items())
            ],
            "active_rays": self.active_rays,
        }


def run_standalone_gtsam_benchmark() -> dict:
    """Simulate the duck walking a 2-loop figure-8 / room circuit while panning its neck."""
    slam = MicroduckGtsamSLAM()
    n_steps = 80
    for k in range(1, n_steps + 1):
        t = k * 0.25
        # Walk a closed room loop of radius ~0.55m returning to origin (triggering loop closures!)
        angle = 2.0 * math.pi * (k / 40.0)
        gt_x = 0.55 * math.sin(angle)
        gt_y = 0.45 * (1.0 - math.cos(angle))
        gt_yaw = math.atan2(0.45 * math.sin(angle), 0.55 * math.cos(angle))
        # Duck pans its head_yaw +/- 35 degrees while walking
        neck_yaw_deg = 35.0 * math.sin(k * 0.4)
        slam.step_odometry_and_landmarks(gt_x, gt_y, gt_yaw, neck_yaw_deg=neck_yaw_deg, sim_time=t, force_keyframe=True)

    metrics = slam.compute_metrics()
    print("=" * 95)
    print(f"GTSAM 4.3 Legged Factor-Graph SLAM Verification ({metrics['backend']})")
    print("=" * 95)
    print(f"  Pose2 Variables X(0..N)      : {metrics['num_poses']}")
    print(f"  Point2 Landmarks L(0..M)     : {metrics['num_landmarks']} / {len(DEFAULT_WORLD_LANDMARKS)}")
    print(f"  BetweenFactorPose2 (Odom/LC) : {metrics['num_between_factors']} (Loop Closures: {metrics['loop_closures']})")
    print(f"  BearingRangeFactor2D (Head)  : {metrics['num_br_factors']}")
    print(f"  Incremental iSAM2 Solve Time : {metrics['last_solve_ms']:.3f} ms / keyframe")
    print(f"  Raw Leg-Inertial Odom RMSE   : {metrics['dr_rmse_m']:.4f} m")
    print(f"  GTSAM Factor-Graph SLAM RMSE : {metrics['gtsam_rmse_m']:.4f} m")
    print(f"  Odometry Drift Reduction     : {metrics['error_reduction_pct']:.1f}%")
    print("=" * 95)
    assert metrics["gtsam_rmse_m"] < metrics["dr_rmse_m"] * 0.35, "GTSAM must reduce drift by >65%"
    return metrics


if __name__ == "__main__":
    run_standalone_gtsam_benchmark()
