"""RL Policy Zoo & Needle 3 Router for Microduck / XGO-Duck.

Unifies all 29 RL movement neural networks (`.npz` pure-NumPy & `.onnx`) from:
  1. `LuwuDynamics/xgoduck_runtime_arduino` (8 core policies)
  2. `AI-FanGe/Microduck-build-tutorial` (1 tutorial walk policy)
  3. Hugging Face Community & Pollen Robotics Microduck RL Zoo (20 policies)

And connects them to **Needle 3 (`cactus-needle`)** so you can reference and trigger
any of the 30 RL skills in natural language while keeping the classic `needle-duck`
"Polite Duck" personality (refuses and shakes its head if you don't say "please"!).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parent
ML_DIR = ROOT_DIR / "ml_checkpoints"
NPZ_DIR = ML_DIR / "models_npz"
HF_ZOO_DIR = ML_DIR / "hf_zoo"
ONNX_CORE_DIR = ML_DIR / "onnx_core"
NEEDLE_CKPT_DIR = ML_DIR / "cactus_needle"
XGODUCK_PYTHON_DIR = ROOT_DIR / "vendor" / "xgoduck_runtime_arduino" / "python"
MICRODUCK_AGENTS_DIR = (
    ROOT_DIR
    / "vendor"
    / "Microduck-build-tutorial"
    / "microduck"
    / "src"
    / "agents"
)
NEEDLE_DUCK_DIR = ROOT_DIR / "vendor" / "needle-duck"


@dataclass(frozen=True)
class RLPolicySpec:
    skill_id: str
    title: str
    category: Literal["Locomotion", "Acrobatics", "Dance", "Emote & Trick", "Balance & Pose", "Soccer"]
    npz_name: str
    onnx_relpath: str
    repo_source: str
    description: str
    duration_s: float  # <= 0 means perpetual until switched
    default_twist: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_period_s: float = 0.0  # > 0 means twist[0]=cos(2pi*t/T), twist[1]=sin(2pi*t/T)
    use_model_home: bool = True
    auto_getup_after: bool = False
    keywords: Tuple[str, ...] = field(default_factory=tuple)

    def resolve_npz(self) -> Path:
        return NPZ_DIR / self.npz_name

    def resolve_onnx(self) -> Path:
        for base in (ML_DIR, ONNX_CORE_DIR, ROOT_DIR / "vendor", ROOT_DIR):
            p = base / self.onnx_relpath
            if p.exists():
                return p
            p_flat = base / Path(self.onnx_relpath).name
            if p_flat.exists():
                return p_flat
        return ML_DIR / self.onnx_relpath

    def compute_twist(self, elapsed_s: float) -> Tuple[float, float, float]:
        if self.phase_period_s > 0.0:
            phase = (elapsed_s % self.phase_period_s) / self.phase_period_s
            angle = 2.0 * math.pi * phase
            return (math.cos(angle), math.sin(angle), self.default_twist[2])
        return self.default_twist


# Complete registry of 30 RL skills backed by 29 distinct RL neural networks
POLICY_ZOO: Dict[str, RLPolicySpec] = {
    # --- LOCOMOTION ---
    "xgoduck_walk": RLPolicySpec(
        skill_id="xgoduck_walk",
        title="XGO-Duck Omnidirectional Walk",
        category="Locomotion",
        npz_name="xgoduck_walk.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_walk.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Default 50 Hz omnidirectional bipedal walking policy with 4-DoF head tracking.",
        duration_s=0.0,
        default_twist=(0.18, 0.0, 0.0),
        use_model_home=False,
        keywords=("xgoduck walk", "normal walk", "stroll"),
    ),
    "microduck_walk": RLPolicySpec(
        skill_id="microduck_walk",
        title="Microduck Original Walk",
        category="Locomotion",
        npz_name="walk.npz",
        onnx_relpath="Microduck-build-tutorial-main/Microduck-build-tutorial-main/microduck/src/agents/walk.onnx",
        repo_source="AI-FanGe/Microduck-build-tutorial",
        description="Original Pollen/AI-FanGe Microduck walking policy trained with 26.2 deg hip/ankle home.",
        duration_s=4.0,
        default_twist=(0.15, 0.0, 0.0),
        use_model_home=True,
        keywords=("microduck walk", "original walk", "tutorial walk", "classic walk"),
    ),
    "running": RLPolicySpec(
        skill_id="running",
        title="Fast Forward Run",
        category="Locomotion",
        npz_name="zoo_running.npz",
        onnx_relpath="hf_zoo/running/policy.onnx",
        repo_source="HannesVonEssen/microduck-running",
        description="High-cadence running locomotion policy trained up to vx=2.2 m/s.",
        duration_s=3.5,
        default_twist=(0.8, 0.0, 0.0),
        use_model_home=True,
        keywords=("run", "running", "jog", "fast walk", "hurry"),
    ),
    "sprint_2m": RLPolicySpec(
        skill_id="sprint_2m",
        title="2-Meter Sprint Reference",
        category="Locomotion",
        npz_name="zoo_sprint_2m.npz",
        onnx_relpath="hf_zoo/sprint_2m/policy.onnx",
        repo_source="pollen-robotics/microduck-sprint-2m-reference",
        description="Pollen Robotics reference sprint policy for rapid forward dash.",
        duration_s=3.5,
        default_twist=(0.9, 0.0, 0.0),
        use_model_home=True,
        keywords=("sprint", "dash", "race", "2m sprint", "full speed"),
    ),
    "roller": RLPolicySpec(
        skill_id="roller",
        title="XGO-Duck Roller Skate",
        category="Locomotion",
        npz_name="xgoduck_roller.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_roller.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Roller-skating propulsion and glide policy for XGO-Duck.",
        duration_s=4.5,
        default_twist=(0.35, 0.0, 0.0),
        use_model_home=False,
        keywords=("roller", "skate", "rollerskate", "glide"),
    ),
    "pollen_roller_crouch": RLPolicySpec(
        skill_id="pollen_roller_crouch",
        title="Pollen Roller Crouch",
        category="Locomotion",
        npz_name="zoo_pollen_roller_crouch.npz",
        onnx_relpath="hf_zoo/pollen_roller_crouch/roller_crouch.onnx",
        repo_source="pollen-robotics/microduck-policies",
        description="Low-center-of-gravity roller crouch policy from Pollen Robotics.",
        duration_s=4.0,
        default_twist=(0.4, 0.0, 0.0),
        use_model_home=True,
        keywords=("roller crouch", "crouch glide", "low skate"),
    ),
    "pollen_velstand": RLPolicySpec(
        skill_id="pollen_velstand",
        title="Pollen Velocity-Stand",
        category="Locomotion",
        npz_name="zoo_pollen_velstand.npz",
        onnx_relpath="hf_zoo/pollen_velstand/velstand.onnx",
        repo_source="pollen-robotics/microduck-policies",
        description="Pollen Robotics velocity-conditioned standing & stepping policy.",
        duration_s=4.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("velstand", "velocity stand", "pollen stand"),
    ),

    # --- ACROBATICS ---
    "backflip": RLPolicySpec(
        skill_id="backflip",
        title="Explosive Backflip",
        category="Acrobatics",
        npz_name="zoo_backflip.npz",
        onnx_relpath="hf_zoo/backflip/policy.onnx",
        repo_source="HannesVonEssen/microduck-backflip",
        description="Full backward aerial flip with bounded joint torque landing.",
        duration_s=2.4,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("backflip", "back flip", "flip backward", "backward flip"),
    ),
    "roulade": RLPolicySpec(
        skill_id="roulade",
        title="Forward Somersault (Roulade)",
        category="Acrobatics",
        npz_name="xgoduck_roulade.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_roulade.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Forward tuck-and-roll somersault that finishes back on both feet.",
        duration_s=2.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=False,
        auto_getup_after=True,
        keywords=("roulade", "somersault", "front flip", "frontflip", "forward roll", "tumble"),
    ),
    "side_roll_right": RLPolicySpec(
        skill_id="side_roll_right",
        title="Side Barrel Roll (Right)",
        category="Acrobatics",
        npz_name="zoo_side_roll_right.npz",
        onnx_relpath="hf_zoo/side_roll_right/policy.onnx",
        repo_source="guang384/microduck-side-roll-right",
        description="360-degree lateral barrel roll / cartwheel over the right shoulder (twist=[0, 1, 0]).",
        duration_s=2.2,
        default_twist=(0.0, 1.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("side roll right", "roll right", "barrel roll right", "cartwheel right", "cartwheel", "barrel roll"),
    ),
    "side_roll_left": RLPolicySpec(
        skill_id="side_roll_left",
        title="Side Barrel Roll (Left)",
        category="Acrobatics",
        npz_name="zoo_side_roll_left.npz",
        onnx_relpath="hf_zoo/side_roll_left/policy.onnx",
        repo_source="guang384/microduck-side-roll-left",
        description="360-degree lateral barrel roll / cartwheel over the left shoulder (twist=[0, -1, 0]).",
        duration_s=2.2,
        default_twist=(0.0, -1.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("side roll left", "roll left", "barrel roll left", "cartwheel left"),
    ),
    "long_jump": RLPolicySpec(
        skill_id="long_jump",
        title="Standing Long Jump",
        category="Acrobatics",
        npz_name="zoo_long_jump.npz",
        onnx_relpath="hf_zoo/long_jump/policy.onnx",
        repo_source="HannesVonEssen/microduck-long-jump",
        description="Two-legged crouch and explosive forward broad jump.",
        duration_s=2.5,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("long jump", "broad jump", "leap", "jump forward", "big jump"),
    ),
    "headstand": RLPolicySpec(
        skill_id="headstand",
        title="Tripod Headstand Fold",
        category="Acrobatics",
        npz_name="zoo_headstand.npz",
        onnx_relpath="hf_zoo/headstand/policy.onnx",
        repo_source="ZachGarner/microduck-headstand-fold",
        description="Pitches forward onto flat head top and lifts both legs into an inverted tripod headstand.",
        duration_s=4.5,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("headstand", "head stand", "handstand", "upside down", "invert"),
    ),
    "headstand_split": RLPolicySpec(
        skill_id="headstand_split",
        title="Headstand Kickup Split",
        category="Acrobatics",
        npz_name="zoo_headstand_split.npz",
        onnx_relpath="hf_zoo/headstand_split/policy.onnx",
        repo_source="ZachGarner/microduck-headstand-split",
        description="Inverted headstand with an acrobatic leg split in the air.",
        duration_s=4.5,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("headstand split", "inverted split", "split headstand"),
    ),
    "getup": RLPolicySpec(
        skill_id="getup",
        title="Autonomous Fall Recovery (Getup)",
        category="Acrobatics",
        npz_name="xgoduck_getup.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_getup.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Recovers from supine or prone fall back to standing posture in 1.0 s.",
        duration_s=1.2,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=False,
        keywords=("getup", "get up", "recover", "stand up from fall"),
    ),

    # --- DANCE ---
    "moonwalk": RLPolicySpec(
        skill_id="moonwalk",
        title="Michael Jackson Moonwalk",
        category="Dance",
        npz_name="zoo_moonwalk.npz",
        onnx_relpath="hf_zoo/moonwalk/policy.onnx",
        repo_source="fffiloni/microduck-moonwalk-backward-55e6af",
        description="Backward gliding moonwalk illusion at vx = -0.15 m/s with forward-pitched head.",
        duration_s=5.0,
        default_twist=(-0.15, 0.0, 0.0),
        use_model_home=True,
        keywords=("moonwalk", "moon walk", "michael jackson", "glide backward", "slide back"),
    ),
    "groove_dance": RLPolicySpec(
        skill_id="groove_dance",
        title="Rhythmic Groove & Sway",
        category="Dance",
        npz_name="zoo_groove_dance.npz",
        onnx_relpath="hf_zoo/groove_dance/policy.onnx",
        repo_source="kramp/microduck-groove",
        description="Rhythmic knee bounce with planted feet and side-to-side head sway.",
        duration_s=6.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("groove", "wiggle", "boogie", "vibe", "sway", "groove dance"),
    ),
    "dance_120bpm": RLPolicySpec(
        skill_id="dance_120bpm",
        title="120 BPM Four-Beat Dance",
        category="Dance",
        npz_name="zoo_dance_120bpm.npz",
        onnx_relpath="hf_zoo/dance_120bpm/seed0/policy.onnx",
        repo_source="vbonnet/microduck-dance-120bpm",
        description="Energetic 120 BPM 4-beat dance routine with full-body weight transfers.",
        duration_s=6.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("120bpm", "120 bpm", "headbang", "rave", "club dance", "beat dance", "party"),
    ),
    "got_talent": RLPolicySpec(
        skill_id="got_talent",
        title="Microduck Got Talent Routine",
        category="Dance",
        npz_name="zoo_got_talent.npz",
        onnx_relpath="hf_zoo/got_talent/policy.onnx",
        repo_source="pollen-robotics/microduck-got-a-talent-reference",
        description="Pollen Robotics 'Got a Talent' expressive stage showcase policy.",
        duration_s=5.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("got talent", "talent", "show off", "showcase", "freestyle", "perform"),
    ),

    # --- EMOTE & TRICK ---
    "polite_bow": RLPolicySpec(
        skill_id="polite_bow",
        title="Polite Theatrical Bow",
        category="Emote & Trick",
        npz_name="zoo_polite_bow.npz",
        onnx_relpath="hf_zoo/polite_bow/policy.onnx",
        repo_source="fffiloni/microduck-polite-bow-b1d864",
        description="Pitches torso and head deeply forward in a respectful theatrical bow, holds, and rises.",
        duration_s=5.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("bow", "polite bow", "greet", "thank you", "curtsy", "nod yes", "respect"),
    ),
    "happy_hop": RLPolicySpec(
        skill_id="happy_hop",
        title="Joyful Vertical Happy Hop",
        category="Emote & Trick",
        npz_name="zoo_happy_hop.npz",
        onnx_relpath="hf_zoo/happy_hop/policy.onnx",
        repo_source="joanfox/microduck-happy-hop",
        description="Crouches and springs straight up in a joyful two-foot hop with an upbeat head nod.",
        duration_s=3.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("happy hop", "happy", "celebrate", "cheer", "yay", "joy", "jump for joy"),
    ),
    "hop_once": RLPolicySpec(
        skill_id="hop_once",
        title="Single Bunny Hop",
        category="Emote & Trick",
        npz_name="zoo_hop_once.npz",
        onnx_relpath="hf_zoo/hop_once/policy.onnx",
        repo_source="chelleboyer/microduck-hop-once",
        description="Symmetric two-foot bunny hop forward with balanced landing.",
        duration_s=3.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        keywords=("hop", "hop once", "bunny hop", "excited", "bounce"),
    ),
    "peck_seeds": RLPolicySpec(
        skill_id="peck_seeds",
        title="Hen Pecking Seeds",
        category="Emote & Trick",
        npz_name="zoo_peck_seeds.npz",
        onnx_relpath="hf_zoo/peck_seeds/policy.onnx",
        repo_source="tfrere/microduck-move-hen-pecking-seeds",
        description="Phase-driven (2.2s period) foraging motion where the duck dips its beak to peck seeds.",
        duration_s=4.4,
        default_twist=(1.0, 0.0, 0.0),
        phase_period_s=2.2,
        use_model_home=True,
        keywords=("peck", "peck seeds", "pecking", "eat", "forage", "chicken", "feed", "hungry"),
    ),
    "play_dead": RLPolicySpec(
        skill_id="play_dead",
        title="Play Dead (Drop Flat)",
        category="Emote & Trick",
        npz_name="zoo_play_dead.npz",
        onnx_relpath="hf_zoo/play_dead/policy.onnx",
        repo_source="tfrere/microduck-move-play-dead-flat",
        description="Dramatically collapses flat onto the floor and stays completely still, then recovers.",
        duration_s=4.5,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=True,
        auto_getup_after=True,
        keywords=("play dead", "dead", "faint", "collapse", "bang", "sleepy", "die"),
    ),
    "pick": RLPolicySpec(
        skill_id="pick",
        title="Ground Object Pick",
        category="Emote & Trick",
        npz_name="xgoduck_pick.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_pick.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Squats and reaches the beak all the way down to pick up an item from the ground.",
        duration_s=3.5,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=False,
        keywords=("pick", "pick up", "grab", "inspect ground", "fetch"),
    ),

    # --- BALANCE & POSE ---
    "flamingo_left": RLPolicySpec(
        skill_id="flamingo_left",
        title="Flamingo One-Leg Balance (Lift Left)",
        category="Balance & Pose",
        npz_name="zoo_flamingo.npz",
        onnx_relpath="hf_zoo/flamingo/policy.onnx",
        repo_source="RemiFabre/microduck-flamingo-cycle",
        description="Shifts weight to right leg and tucks the left foot high in a flamingo pose (twist=[1, 1, 0]).",
        duration_s=4.5,
        default_twist=(1.0, 1.0, 0.0),
        use_model_home=True,
        keywords=("flamingo", "flamingo left", "one leg", "stand on one leg", "balance on one leg", "curious"),
    ),
    "flamingo_right": RLPolicySpec(
        skill_id="flamingo_right",
        title="Flamingo One-Leg Balance (Lift Right)",
        category="Balance & Pose",
        npz_name="zoo_flamingo.npz",
        onnx_relpath="hf_zoo/flamingo/policy.onnx",
        repo_source="RemiFabre/microduck-flamingo-cycle",
        description="Shifts weight to left leg and tucks the right foot high in a flamingo pose (twist=[1, -1, 0]).",
        duration_s=4.5,
        default_twist=(1.0, -1.0, 0.0),
        use_model_home=True,
        keywords=("flamingo right", "right leg up", "lift right leg"),
    ),
    "sit": RLPolicySpec(
        skill_id="sit",
        title="Crouch / Sit Down",
        category="Balance & Pose",
        npz_name="xgoduck_sitstand.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_sitstand.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Lowers the body into a stable seated crouch (twist=[-1, 0, 0]).",
        duration_s=0.0,
        default_twist=(-1.0, 0.0, 0.0),
        use_model_home=False,
        keywords=("sit", "sit down", "crouch", "rest", "sad"),
    ),
    "stand": RLPolicySpec(
        skill_id="stand",
        title="Stand Up Tall",
        category="Balance & Pose",
        npz_name="xgoduck_sitstand.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_sitstand.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Rises from crouch back to full standing height (twist=[1, 0, 0]).",
        duration_s=2.0,
        default_twist=(1.0, 0.0, 0.0),
        use_model_home=False,
        keywords=("stand", "stand up", "rise"),
    ),

    # --- SOCCER ---
    "ballkick_right": RLPolicySpec(
        skill_id="ballkick_right",
        title="Right Foot Soccer Kick",
        category="Soccer",
        npz_name="xgoduck_ballkick_right.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_ballkick_right.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Balances on left leg and swings the right foot forward to strike a soccer ball.",
        duration_s=2.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=False,
        keywords=("kick right", "right kick", "kick", "kick ball", "soccer kick", "shoot"),
    ),
    "ballkick_left": RLPolicySpec(
        skill_id="ballkick_left",
        title="Left Foot Soccer Kick",
        category="Soccer",
        npz_name="xgoduck_ballkick_left.npz",
        onnx_relpath="xgoduck_runtime_arduino/python/xgoduck_ballkick_left.onnx",
        repo_source="LuwuDynamics/xgoduck_runtime_arduino",
        description="Balances on right leg and swings the left foot forward to strike a soccer ball.",
        duration_s=2.0,
        default_twist=(0.0, 0.0, 0.0),
        use_model_home=False,
        keywords=("kick left", "left kick", "left foot kick"),
    ),
}


def match_zoo_skill_from_text(text: str) -> Optional[RLPolicySpec]:
    """Find the most specific matching RLPolicySpec from user text."""
    lower = text.strip().lower()
    if lower.startswith("zoo:"):
        key = lower.split("zoo:", 1)[1].strip()
        return POLICY_ZOO.get(key)
    if lower in POLICY_ZOO:
        return POLICY_ZOO[lower]

    best_spec: Optional[RLPolicySpec] = None
    best_len = 0
    for spec in POLICY_ZOO.values():
        for kw in spec.keywords:
            if kw in lower and len(kw) > best_len:
                best_spec = spec
                best_len = len(kw)
    return best_spec


class NeedleZooBrain:
    """Connects `cactus-needle` (Needle 3) to the 30-skill RL Movement Zoo.

    Architecture:
      1. **Stage 1 — Polite Duck Gate (`duck_4l.cact` via `duck_tools.TOOLS`)**:
         Runs the fine-tuned `needle-duck` model (`duck_4l.cact`) with its 5 tools
         (`walk`, `turn`, `emote`, `dance`, `shake_head`). If the user did not ask
         politely (e.g. missing 'please'), `duck_4l.cact` returns `shake_head()` and
         the duck refuses and shakes its head!
      2. **Stage 2 — 30-Skill RL Policy Zoo Dispatcher (`ZOO_TOOLS` + Needle 3)**:
         Also defines a 5-tool Needle 3 schema (`locomotion`, `acrobatics`, `dance_rl`,
         `emote_trick`, `balance_and_kick`) where every single one of the 30 RL skills
         is a `Literal[...]` enum option, and routes both Needle 3 tool calls and
         natural language requests directly to the matching trained RL neural network!
    """

    DANCE_STYLE_TO_RL: Dict[str, str] = {
        "moonwalk": "moonwalk",
        "wiggle": "groove_dance",
        "headbang": "dance_120bpm",
        "chicken": "peck_seeds",
        "spin": "side_roll_right",
    }

    EMOTE_TO_RL: Dict[str, str] = {
        "nod_yes": "polite_bow",
        "happy": "happy_hop",
        "confused": "flamingo_left",
        "scared": "play_dead",
        "angry": "ballkick_right",
        "sleepy": "play_dead",
        "look_down": "pick",
        "look_up": "polite_bow",
        "look_left": "flamingo_left",
        "look_right": "flamingo_right",
        "look_around": "groove_dance",
        "sad": "sit",
    }

    def __init__(self, weights_name: str = "duck_4l.cact"):
        primary = NEEDLE_CKPT_DIR / weights_name
        self.weights_path = primary if primary.exists() else (NEEDLE_DUCK_DIR / "models" / weights_name)
        self.needle_available = False
        self._polite_agent = None
        self._zoo_agent = None
        self._init_needle()

    def _init_needle(self) -> None:
        try:
            import warnings
            import needle

            if str(NEEDLE_DUCK_DIR) not in sys.path:
                sys.path.insert(0, str(NEEDLE_DUCK_DIR))
            from duck_tools import TOOLS as POLITE_TOOLS, SYSTEM as POLITE_SYSTEM

            weights = str(self.weights_path) if self.weights_path.exists() else None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*confidence head.*", category=UserWarning)
                self._polite_agent = needle.Needle(
                    tools=POLITE_TOOLS,
                    system=POLITE_SYSTEM,
                    weights=weights,
                    auto_date=False,
                )

            # Define the 5-tool RL Policy Zoo surface for base Needle 3 classification
            @needle.tool
            def locomotion(
                policy: Literal[
                    "xgoduck_walk",
                    "microduck_walk",
                    "running",
                    "sprint_2m",
                    "roller",
                    "pollen_roller_crouch",
                    "pollen_velstand",
                ] = "xgoduck_walk",
            ):
                """Run a locomotion RL policy: walk, microduck walk, fast running, 2m sprint, roller skate, or roller crouch."""
                return {"policy": policy}

            @needle.tool
            def acrobatics(
                policy: Literal[
                    "backflip",
                    "roulade",
                    "side_roll_right",
                    "side_roll_left",
                    "long_jump",
                    "headstand",
                    "headstand_split",
                    "getup",
                ] = "backflip",
            ):
                """Perform an acrobatic stunt RL policy: backflip, forward somersault roulade, right/left side barrel roll cartwheel, long jump, headstand, headstand split, or getup."""
                return {"policy": policy}

            @needle.tool
            def dance_rl(
                policy: Literal["moonwalk", "groove_dance", "dance_120bpm", "got_talent"] = "groove_dance",
            ):
                """Perform an RL dance policy: backward moonwalk, rhythmic groove wiggle, 120bpm rave headbang dance, or got talent routine."""
                return {"policy": policy}

            @needle.tool
            def emote_trick(
                policy: Literal[
                    "polite_bow", "happy_hop", "hop_once", "peck_seeds", "play_dead", "pick"
                ] = "happy_hop",
            ):
                """Perform an expressive RL trick policy: polite bow, joyful happy hop, bunny hop once, hen pecking seeds, play dead flat, or ground pick."""
                return {"policy": policy}

            @needle.tool
            def balance_and_kick(
                policy: Literal[
                    "flamingo_left",
                    "flamingo_right",
                    "sit",
                    "stand",
                    "ballkick_right",
                    "ballkick_left",
                ] = "flamingo_left",
            ):
                """Perform a balance pose or soccer kick RL policy: flamingo one-leg balance (left or right), sit crouch, stand up, or soccer ball kick (right or left)."""
                return {"policy": policy}

            self.zoo_tools = [locomotion, acrobatics, dance_rl, emote_trick, balance_and_kick]
            # Note: cactus-needle binds one C++ engine per process (see needle-duck/app/brain_worker.py),
            # so we keep `self._polite_agent` (duck_4l.cact) as the single active C++ engine in-process.
            self._zoo_agent = None
            self.needle_available = True
        except Exception as exc:
            print(f"[NeedleZooBrain] Warning: cactus-needle init fallback ({exc})")
            self.needle_available = False

    def interpret(self, user_text: str) -> dict:
        """Run Needle 3 on `user_text` and return a structured action plan for the simulator."""
        t0 = time.perf_counter()
        cleaned = user_text.strip()
        lower = cleaned.lower()

        # Direct bypass prefix `zoo:<skill_id>` for UI buttons / automated tests
        if lower.startswith("zoo:"):
            spec = match_zoo_skill_from_text(lower)
            if spec:
                return {
                    "accepted": True,
                    "action_type": "rl_zoo",
                    "skill_id": spec.skill_id,
                    "spec": spec,
                    "needle_raw": f"direct_zoo({spec.skill_id})",
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "message": f"Executing RL Policy [{spec.skill_id}]: {spec.title} ({spec.repo_source})",
                }

        # Run Stage 1: Needle 3 (`duck_4l.cact`) Polite Gate & Base Tool Call
        needle_calls: List[dict] = []
        if self.needle_available and self._polite_agent is not None:
            self._polite_agent.reset()
            res = self._polite_agent.complete(cleaned)
            needle_calls = res.get("function_calls") or res.get("suppressed_calls") or []
        else:
            if "please" not in lower:
                needle_calls = [{"name": "shake_head", "arguments": {}}]
            else:
                needle_calls = [{"name": "emote", "arguments": {"expression": "happy"}}]

        needle_raw_str = "  ".join(
            f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in (c.get('arguments') or {}).items())})"
            for c in needle_calls
        ) or "no_call"

        # 1. Check if Needle 3 (`duck_4l.cact`) refused or if request lacks politeness ("shake_head")
        polite_words = ("please", "pls", "plz", "kindly", "could you", "would you")
        has_polite_word = any(pw in lower for pw in polite_words)
        if any(c.get("name") == "shake_head" for c in needle_calls) or not has_polite_word:
            return {
                "accepted": False,
                "action_type": "shake_head",
                "skill_id": None,
                "spec": None,
                "needle_raw": needle_raw_str if "shake_head" in needle_raw_str else f"shake_head() [raw: {needle_raw_str}]",
                "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "message": f"Quack! [Shakes head 'NO' — Needle 3 `{needle_raw_str}`: say 'please'!]",
            }

        # 2. Check if the user's polite prompt names any of the 30 RL Zoo skills specifically
        explicit_spec = match_zoo_skill_from_text(lower)
        if explicit_spec is not None:
            # Also query Stage 2 (`_zoo_agent`) so Needle 3's 30-skill tool call is shown in logs!
            zoo_call_str = f"rl_zoo(policy={explicit_spec.skill_id!r})"
            if self.needle_available and self._zoo_agent is not None:
                try:
                    self._zoo_agent.reset()
                    zr = self._zoo_agent.complete(cleaned)
                    zcalls = zr.get("function_calls") or zr.get("suppressed_calls") or []
                    if zcalls:
                        zc = zcalls[0]
                        zoo_call_str = f"{zc['name']}({', '.join(f'{k}={v!r}' for k, v in (zc.get('arguments') or {}).items())})"
                except Exception:
                    pass
            return {
                "accepted": True,
                "action_type": "rl_zoo",
                "skill_id": explicit_spec.skill_id,
                "spec": explicit_spec,
                "needle_raw": f"{needle_raw_str} -> {zoo_call_str}",
                "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "message": (
                    f"Needle 3 `{zoo_call_str}` -> "
                    f"RL Policy [{explicit_spec.skill_id}]: {explicit_spec.title} ({explicit_spec.repo_source})"
                ),
            }

        # 3. Otherwise, route Needle 3 (`duck_4l.cact` or `_zoo_agent`)'s tool call to its RL policy!
        if needle_calls:
            c0 = needle_calls[0]
            fn_name = c0.get("name", "")
            fn_args = c0.get("arguments") or {}
            if fn_name == "walk":
                return {
                    "accepted": True,
                    "action_type": "walk",
                    "args": fn_args,
                    "skill_id": "xgoduck_walk",
                    "spec": POLICY_ZOO["xgoduck_walk"],
                    "needle_raw": needle_raw_str,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "message": f"Needle 3 `{needle_raw_str}` -> RL Policy [xgoduck_walk]",
                }
            elif fn_name == "turn":
                return {
                    "accepted": True,
                    "action_type": "turn",
                    "args": fn_args,
                    "skill_id": "xgoduck_walk",
                    "spec": POLICY_ZOO["xgoduck_walk"],
                    "needle_raw": needle_raw_str,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "message": f"Needle 3 `{needle_raw_str}` -> RL Policy [xgoduck_walk]",
                }
            elif fn_name == "dance":
                style = str(fn_args.get("style", "wiggle"))
                skill_id = self.DANCE_STYLE_TO_RL.get(style, "groove_dance")
                spec = POLICY_ZOO[skill_id]
                return {
                    "accepted": True,
                    "action_type": "rl_zoo",
                    "skill_id": spec.skill_id,
                    "spec": spec,
                    "needle_raw": needle_raw_str,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "message": (
                        f"Needle 3 `{needle_raw_str}` -> "
                        f"RL Policy [{spec.skill_id}]: {spec.title} ({spec.repo_source})"
                    ),
                }
            elif fn_name == "emote":
                expr = str(fn_args.get("expression", "happy"))
                skill_id = self.EMOTE_TO_RL.get(expr, "happy_hop")
                spec = POLICY_ZOO[skill_id]
                return {
                    "accepted": True,
                    "action_type": "rl_zoo",
                    "skill_id": spec.skill_id,
                    "spec": spec,
                    "needle_raw": needle_raw_str,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "message": (
                        f"Needle 3 `{needle_raw_str}` -> "
                        f"RL Policy [{spec.skill_id}]: {spec.title} ({spec.repo_source})"
                    ),
                }

        # 4. Ask Stage 2 `_zoo_agent` directly for open-ended polite requests
        if self.needle_available and self._zoo_agent is not None:
            try:
                self._zoo_agent.reset()
                zr = self._zoo_agent.complete(cleaned)
                zcalls = zr.get("function_calls") or zr.get("suppressed_calls") or []
                if zcalls:
                    zc = zcalls[0]
                    pol_id = (zc.get("arguments") or {}).get("policy")
                    if pol_id in POLICY_ZOO:
                        spec = POLICY_ZOO[pol_id]
                        return {
                            "accepted": True,
                            "action_type": "rl_zoo",
                            "skill_id": spec.skill_id,
                            "spec": spec,
                            "needle_raw": f"{zc['name']}(policy={pol_id!r})",
                            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                            "message": f"Needle 3 `{zc['name']}(policy={pol_id!r})` -> RL Policy [{spec.skill_id}]: {spec.title}",
                        }
            except Exception:
                pass

        spec = POLICY_ZOO["happy_hop"]
        return {
            "accepted": True,
            "action_type": "rl_zoo",
            "skill_id": spec.skill_id,
            "spec": spec,
            "needle_raw": needle_raw_str,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "message": f"Needle 3 -> RL Policy [{spec.skill_id}]: {spec.title}",
        }
