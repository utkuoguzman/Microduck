# End-to-End Voice & RL Benchmark Report

## 1. Cactus Whistle S2T (`whistle_d4.cact`) + Needle 3 (`duck_4l.cact`) + MuJoCo 50 Hz RL Zoo

| Spoken Audio Input | Whistle S2T Transcript | Whistle Latency | Needle 3 Tool Call | Needle Latency | Total Voice-to-Action | MuJoCo Physics Outcome |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| `"Walk forward"` *(Impolite)* | `"Walk forward."` | `41.2 ms` (`330.2 tok/s`) | `shake_head()` *(Refused)* | `54.4 ms` | **`95.6 ms`** | Shakes head `"NO"` (`x=0.002 m`) |
| `"Please walk forward"` | `"Please walk forward."` | `44.6 ms` (`350.3 tok/s`) | `walk(direction='forward')` -> `xgoduck_walk` | `53.0 ms` | **`97.6 ms`** | Walks forward **`+0.187 m`** upright |
| `"Please do a polite bow"` | `"Please do a polite bow."` | `59.6 ms` (`324.7 tok/s`) | `rl_zoo(policy='polite_bow')` | `47.3 ms` | **`106.9 ms`** | Performs theatrical bow (`z=0.137 m`) |
| `"Please moonwalk"` | `"Please, Moon Walk."` | `53.0 ms` (`312.5 tok/s`) | `rl_zoo(policy='moonwalk')` | `58.4 ms` | **`111.4 ms`** | Moonwalks backward **`-0.202 m`** |
| `"Please do a backflip"` | `"Please do a back flip."` | `55.1 ms` (`362.6 tok/s`) | `rl_zoo(policy='backflip')` | `47.7 ms` | **`102.8 ms`** | Launches airborne backflip (`dx=-0.31 m`) |

## 2. Pure-NumPy `.npz` RL Policy Zoo Verification (29 Policies)

All 29 ONNX RL policies (8 XGO-Duck core + 1 Microduck tutorial walk + 20 Hugging Face community policies) were exported to `.npz` format in [`ml_checkpoints/models_npz/`](../ml_checkpoints/models_npz/) and verified against `onnxruntime`:

- **Max numerical deviation vs ONNXRuntime:** `< 1.83e-4` (`~3e-6` typical across 32 random 61-D observations)
- **Single-step PC NumPy inference latency:** **`0.058 ms – 0.111 ms`** (50 Hz loop budget is `20.0 ms`)
- **Single-step Raspberry Pi Zero W (v1.1) NumPy latency:** **`~1.5 ms – 2.2 ms`** (`<11%` of the `20.0 ms` budget)
- Full per-model metrics are saved in [`policy_export_verification.json`](policy_export_verification.json).

