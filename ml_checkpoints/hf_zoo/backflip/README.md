---
license: apache-2.0
pipeline_tag: reinforcement-learning
tags: [microduck, robotics, mujoco, onnx, reinforcement-learning]
thumbnail: https://huggingface.co/HannesVonEssen/microduck-backflip/resolve/main/media/social-preview.png
---

# Microduck — Raised-platform backflip

<video src="https://huggingface.co/HannesVonEssen/microduck-backflip/resolve/main/media/preview.mp4" controls muted playsinline width="720"></video>

**⚠️ Landings may break the robot.** Simulation only; not yet validated on hardware. Successful simulated recovery does not establish a safe physical landing.

A backflip from a raised platform onto a soft-contact simulated mat. This is
not a standing-height backflip and has not been validated on hardware.

## Policy and landing assumptions

- `policy.onnx` and `checkpoint.pt`: retained f21, iteration **12,000**.
- 61 observations, 14 bounded joint-position actions, 50 Hz; normalization
  and action bounds included in ONNX, no action low-pass filter.
- Body-command block: `[(drop_height - 0.7) / 0.2, direction_sign, 0, 0, 0, 0]`.
  The profile sets the backflip direction. Full contract: `config.json`.

The release profile uses a **0.8 m drop from platform top to mat top**. The mat
is 0.44342 m thick, with a 1.90154 m × 0.605 m footprint. Its collision is a
soft-contact box (`solref=(0.03, 1)`, `solimp=(0.9, 0.95, 0.01, 0.5, 2)`), not
a calibrated model of a real crash mat. Servo and structural damage are not
simulated. **Do not extrapolate this result to a rigid-floor landing.**

Preparatory bounces are part of the behavior. The reproduction script starts
on the platform, not mid-air, and does not switch to a separate get-up policy.
The sculpted visual mat in the preview is optional decoration; reproduction
uses the same collision box as the visible mat.

## Reproduce in simulation

[Source and instructions](https://github.com/Vottivott/microduck-playground/tree/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour/backflip)

```bash
uv sync --locked
uv run python scripts/parkour_release.py backflip render \
  --checkpoint /path/to/checkpoint.pt --output eval/backflip.mp4
```

The preview is an excerpt from the selected research montage. Its exact
checkpoint/seed mapping has not been verified; it is not an exact replay or
success-rate claim for this package. Earlier research success figures affected
by mid-air resets are not used as validation. See `eval/` for numerical ONNX
parity and a small fresh diagnostic of this checkpoint, `manifest.json` for
provenance and `SHA256SUMS` for integrity. Load the pickle checkpoint only if
you trust its source.

Related: [long jump](https://huggingface.co/HannesVonEssen/microduck-long-jump).

## Continue training

[TRAINING.md](TRAINING.md) documents the full-checkpoint continuation recipe, distinct from render profiles. A clean locked installation passed a 64-environment/five-update resume, exact learning-state restoration, finite-step checks and normalized ONNX parity. Recipe provenance and reconstruction limits are explicit; this is not a claim of bit-identical historical replay or improved behavior. Original policy, checkpoint and media are unchanged.

## Fresh release check — 2026-09-29

Completes the backward rotation and recovers to standing on the simulated mat, but later drifts off and falls at 7.62 seconds. **This is not a stable idle policy.** One CPU rollout at the documented default seed (28), using the exact packaged checkpoint; not a success-rate estimate or hardware validation. See [source verification](https://github.com/Vottivott/microduck-playground/blob/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour/VERIFICATION.md) and `eval/render-20260929.json`.
