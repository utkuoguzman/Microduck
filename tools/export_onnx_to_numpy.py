"""Export all XGO-Duck and Microduck ONNX policies to lightweight pure-NumPy (.npz) files.

This allows the Raspberry Pi Zero W (v1.1, ARMv6) to run every policy at 50 Hz
using only `numpy`, with zero dependency on `onnxruntime`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from numpy_policy import extract_onnx_to_numpy_graph, NumpyMlpGraph

OUT_DIR = ROOT / "ml_checkpoints" / "models_npz"


def convert_and_verify(onnx_path: Path, out_dir: Path, custom_stem: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = custom_stem if custom_stem else onnx_path.stem
    npz_path = out_dir / f"{stem}.npz"

    graph = extract_onnx_to_numpy_graph(onnx_path)
    graph.save_npz(npz_path)

    # Reload from disk and verify against onnxruntime
    reloaded = NumpyMlpGraph.from_npz(npz_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(42)
    test_obs = rng.normal(0.0, 1.0, size=(32, reloaded.input_dim)).astype(np.float32)

    ort_out = np.vstack([sess.run(None, {in_name: row.reshape(1, -1)})[0] for row in test_obs])
    np_out = reloaded.forward(test_obs)
    max_diff = float(np.max(np.abs(ort_out - np_out)))

    # Benchmark single-sample latency
    single = test_obs[0]
    t0 = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        reloaded.forward(single)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / iters

    assert max_diff < 1e-3, f"Mismatch on {onnx_path.name}: max_diff={max_diff}"
    return {
        "name": f"{stem} ({onnx_path.name})",
        "npz": npz_path.name,
        "in_dim": reloaded.input_dim,
        "out_dim": reloaded.output_dim,
        "size_kb": round(npz_path.stat().st_size / 1024.0, 1),
        "max_diff": max_diff,
        "pc_infer_ms": round(avg_ms, 4),
    }


def main() -> None:
    onnx_core = ROOT / "ml_checkpoints" / "onnx_core"
    tasks: list[tuple[Path, str | None]] = [
        (f, None) for f in sorted(onnx_core.glob("*.onnx"))
    ]

    hf_zoo_dir = ROOT / "ml_checkpoints" / "hf_zoo"
    if hf_zoo_dir.exists():
        for slug_dir in sorted(hf_zoo_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            for onnx_file in sorted(slug_dir.rglob("*.onnx")):
                tasks.append((onnx_file, f"zoo_{slug_dir.name}"))

    print(f"Converting {len(tasks)} ONNX policies to pure-NumPy .npz format...")
    print("-" * 95)
    print(f"{'ONNX Policy':<36} | {'NPZ File':<28} | {'Shape':<8} | {'Size':<8} | {'Max Diff':<10} | {'PC ms'}")
    print("-" * 95)
    records = []
    for f, custom_stem in tasks:
        info = convert_and_verify(f, OUT_DIR, custom_stem=custom_stem)
        records.append(info)
        shape_str = f"{info['in_dim']}->{info['out_dim']}"
        print(
            f"{info['name']:<36} | {info['npz']:<28} | {shape_str:<8} | "
            f"{info['size_kb']:>5.1f} KB | {info['max_diff']:<10.2e} | {info['pc_infer_ms']:.4f} ms"
        )
    print("-" * 95)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    import json
    (results_dir / "policy_export_verification.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"All {len(tasks)} models exported and verified in: {OUT_DIR}")


if __name__ == "__main__":
    main()
