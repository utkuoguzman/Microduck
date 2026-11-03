---
tags:
  - microduck
  - reinforcement-learning
  - robotics
  - mujoco
  - onnx
---

# Moonwalk Backward — Microduck policy

Generated with **Microduck Motion Lab** from Pollen Robotics' `microduck_rl` sim-to-real stack.

## Prompt

> Make Microduck perform a moonwalk-style backward gliding gait. It should move backward smoothly while alternating its feet, create the visual illusion of sliding rather than ordinary backward walking, keep its upper body stable and upright, avoid jumps or large vertical motion, maintain balance throughout, and finish in a stable standing pose after a few seconds.

## Reproducibility

- Task: `Mjlab-MotionLabMoonwalkBackward-Flat-MicroDuck`
- Base recipe: `velocity`
- `microduck_rl` revision: `d424a0c899f6b33cbd3daeb279913134349c0b63`
- Parallel environments: **4096**
- PPO iterations: **4000**
- Compiler: `recipe-plan-adapter-v0.10`
- Final repo: `fffiloni/microduck-moonwalk-backward-55e6af`

## Deployment checks

- Official `scripts/export.py`: **passed**
- Shared actor observation contract: **61D**
- Policy action output: **14D**
- CPU MuJoCo/ONNX rehearsal: **passed**
- Automatic motion-quality gate: **needs review**
- Preview recipe: **velocity**
- Preview driver: **velocity**
- Semantic match unverified: **True**
- Maximum joint motion: **14.737863554636114°**
- XY displacement: **0.33397904866706285 m**
- Ball displacement: **0.0 m**

## Files

- `policy.onnx` — deployment policy exported by upstream `scripts/export.py`
- `preview.mp4` — headless CPU MuJoCo deployment rehearsal
- `thumbnail.webp`
- `preview_metrics.json`
- `motion_spec.json`
- `resolved_plan.json`
- `compiled_motion.json`
- `training_result.json`
- `checkpoints/model_final.pt`

> A simulation preview is not a guarantee of safe real-robot behavior. Validate conservatively before deploying to hardware.
