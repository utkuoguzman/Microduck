---
library_name: onnx
pipeline_tag: reinforcement-learning
tags:
- robotics
- mujoco
- microduck
---
# Microduck headstand: fold

Simulation policy for `Mjlab-HeadstandFold-Flat-MicroDuck`. The original checkpoint and an ONNX export with the observation normalizer included are provided. Input: 61 observation values; output: 14 joint actions.

The original checkpoint passed 32 of 32 individual simulated attempts at seed 0 under the documented evaluator. This does not establish hardware performance or success from arbitrary starting poses. The ONNX export is provided separately; the published rollout counts were measured with the original checkpoint.

Files:

- `model_1000.pt`: the unchanged evaluated checkpoint.
- `policy.onnx`: export through the training repository's standard exporter.
- `provenance.json`: source run, hashes, and code provenance.
- `evaluation.json`: individual evaluation report.

The six policies together completed 87 of 96 routine attempts across seeds 0, 1, and 2. See the [evaluation code, setup, and raw results](https://github.com/zachgarner/microduck-headstand/tree/71ad395). The saved checkpoint can be placed in the evaluator's cache at `logs/rsl_rl/microduck_headstand_fold/wandb_checkpoints/vorty4kb/model_1000.pt`; no W&B download is then needed for that checkpoint.

These are simulation research artifacts, not a robot-daemon installation package. No runtime manifest or hardware installation command is supplied. The routine uses contact-based handovers and a separate standing policy, and has not been tested on a physical robot. Head-force samples are taken at 50 Hz and may miss shorter impacts. Exit success checks final standing, not leg shape throughout the exit.
