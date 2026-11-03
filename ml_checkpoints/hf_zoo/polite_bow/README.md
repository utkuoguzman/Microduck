---
tags:
  - microduck
  - reinforcement-learning
  - robotics
  - mujoco
  - onnx
---

# Polite Bow — Microduck policy

Generated with **Microduck Motion Lab** from Pollen Robotics' `microduck_rl` sim-to-real stack.

## Prompt

> Recovered from cloud artifacts: Small, gentle bow with head and body lowering, returning to standing

## Reproducibility

- Task: `Mjlab-PoliteBow-Flat-MicroDuck`
- Base recipe: `standup`
- `microduck_rl` revision: `d424a0c899f6b33cbd3daeb279913134349c0b63`
- Parallel environments: **4096**
- PPO iterations: **1000**
- Compiler: `guided-standup-v0.4`
- Final repo: `fffiloni/microduck-polite-bow-b1d864`

## Deployment checks

- Official `scripts/export.py`: **passed**
- Shared actor observation contract: **61D**
- Policy action output: **14D**
- CPU MuJoCo/ONNX rehearsal: **passed**
- Automatic motion-quality gate: **passed**
- Preview recipe: **standup**
- Preview driver: **unknown**
- Semantic match unverified: **False**
- Maximum joint motion: **unknown°**
- XY displacement: **unknown m**
- Ball displacement: **unknown m**

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
