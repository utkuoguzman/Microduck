---
license: apache-2.0
pipeline_tag: reinforcement-learning
tags: [microduck, robotics, mujoco, onnx, reinforcement-learning]
thumbnail: https://huggingface.co/HannesVonEssen/microduck-long-jump/resolve/main/media/social-preview.png
---

# Microduck — Raised-platform long jump

<video src="https://huggingface.co/HannesVonEssen/microduck-long-jump/resolve/main/media/preview.mp4" controls muted playsinline width="720"></video>

A learned jump from one raised platform to a lower platform across a gap.
This uses the **platform-jump** task, not the flat-ground long-jump task.

**Simulation only; not yet validated on hardware.**

## Policy

- `policy.onnx`: 61 observations → 14 bounded joint-position actions, 50 Hz.
- `checkpoint.pt`: retained p1 checkpoint, stored iteration **15,250**.
- `config.json`: joint order, offsets, action bounds and command-slot meaning.
- `eval/`: numerical ONNX parity and a small fresh simulation diagnostic.

Observation normalization and action bounds are baked into ONNX; there is no
action low-pass filter. The actor has no vision: the body-command block is
`[gap / 0.3, drop / 0.3, 0, 0, 0, 0]`, with distances in metres.
Matching the walking policy's dimensions does not make it a drop-in replacement.

## Reproduce in simulation

[Source and instructions](https://github.com/Vottivott/microduck-playground/tree/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour/long-jump)
· [Released jump experiments](https://github.com/Vottivott/microduck-playground/tree/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour)

From that source checkout, after downloading `checkpoint.pt`:

```bash
uv sync --locked
uv run python scripts/parkour_release.py long-jump render \
  --checkpoint /path/to/checkpoint.pt --output eval/long-jump.mp4
```

The supplied profile fixes a 30 cm gap and 25 cm drop, with the target platform
10 cm above the floor. It begins on the starting platform and stops at the first
episode termination. No separate recovery policy is used.

The preview is a selected excerpt from the parkour montage. Its exact
checkpoint/seed mapping has not been verified; it is not presented as a seeded
replay of this package or evidence of a measured success rate. Evaluation files
refer specifically to the packaged checkpoint. `manifest.json` pins the source
revision, profile, checkpoint hash and media provenance; `SHA256SUMS` covers the
package. Load the pickle checkpoint only if you trust its source.

Related: [backflip](https://huggingface.co/HannesVonEssen/microduck-backflip).

## Continue training

[TRAINING.md](TRAINING.md) documents the full-checkpoint continuation recipe, distinct from render profiles. A clean locked installation passed a 64-environment/five-update resume, exact learning-state restoration, finite-step checks and normalized ONNX parity. Recipe provenance and reconstruction limits are explicit; this is not a claim of bit-identical historical replay or improved behavior. Original policy, checkpoint and media are unchanged.

## Fresh release check — 2026-09-29

Crosses the gap, lands and stays standing on the target through the 12-second rollout. One CPU rollout at the documented default seed (28), using the exact packaged checkpoint; not a success-rate estimate or hardware validation. See [source verification](https://github.com/Vottivott/microduck-playground/blob/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour/VERIFICATION.md) and `eval/render-20260929.json`.
