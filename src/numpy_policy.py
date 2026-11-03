"""Pure-NumPy Policy Inference Engine for Microduck & XGO-Duck.

Designed specifically so that Raspberry Pi Zero W (v1.1, ARMv6 32-bit) can run
all 50 Hz RL locomotion & trick policies (walk, roller, roulade/somersault,
getup, pick, sitstand, ballkick_left, ballkick_right) using ONLY `numpy`
(without requiring `onnxruntime`, which has no ARMv6 wheel).

Also supports seamlessly loading `.onnx` directly if `onnx` or `onnxruntime`
is installed on the host PC.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * (np.exp(np.clip(x, -80.0, 0.0)) - 1.0))


class NumpyMlpGraph:
    """Executes an exported sequential ONNX MLP (including observation normalization) in pure NumPy."""

    def __init__(self, ops: List[dict], metadata: dict[str, str], input_dim: int, output_dim: int):
        self.ops = ops
        self.metadata = metadata
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> "NumpyMlpGraph":
        data = np.load(str(npz_path), allow_pickle=False)
        manifest = json.loads(str(data["manifest_json"]))
        metadata = manifest.get("metadata", {})
        input_dim = manifest["input_dim"]
        output_dim = manifest["output_dim"]
        ops = []
        for op_spec in manifest["ops"]:
            op_copy = dict(op_spec)
            for k, arr_name in op_spec.get("arrays", {}).items():
                op_copy[k] = data[arr_name].astype(np.float32)
            ops.append(op_copy)
        return cls(ops=ops, metadata=metadata, input_dim=input_dim, output_dim=output_dim)

    def save_npz(self, npz_path: str | Path) -> None:
        arrays_to_save: dict[str, Any] = {}
        serializable_ops = []
        arr_idx = 0
        for op in self.ops:
            op_meta = {"type": op["type"], "arrays": {}}
            for k, v in op.items():
                if k == "type":
                    continue
                if isinstance(v, np.ndarray):
                    arr_name = f"arr_{arr_idx}"
                    arr_idx += 1
                    arrays_to_save[arr_name] = v.astype(np.float32)
                    op_meta["arrays"][k] = arr_name
                else:
                    op_meta[k] = v
            serializable_ops.append(op_meta)

        manifest = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "metadata": self.metadata,
            "ops": serializable_ops,
        }
        arrays_to_save["manifest_json"] = np.array(json.dumps(manifest))
        np.savez_compressed(str(npz_path), **arrays_to_save)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        squeeze_out = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_out = True

        for op in self.ops:
            op_type = op["type"]
            if op_type == "Sub":
                x = x - op["const"]
            elif op_type == "Add":
                x = x + op["const"]
            elif op_type == "Mul":
                x = x * op["const"]
            elif op_type == "Div":
                x = x / op["const"]
            elif op_type == "Clip":
                lo = op.get("min", -np.inf)
                hi = op.get("max", np.inf)
                x = np.clip(x, lo, hi)
            elif op_type == "Linear":
                # x @ W + b (where W is already oriented as [in_features, out_features])
                x = x @ op["W"]
                if "b" in op:
                    x = x + op["b"]
            elif op_type == "Elu":
                x = _elu(x, float(op.get("alpha", 1.0)))
            elif op_type == "Relu":
                x = np.maximum(x, 0.0)
            elif op_type == "Tanh":
                x = np.tanh(x)
            elif op_type == "Max":
                x = np.maximum(x, op["const"])
            elif op_type == "Min":
                x = np.minimum(x, op["const"])
            elif op_type in ("Identity", "Cast", "Flatten"):
                pass
            else:
                raise ValueError(f"Unsupported pure-NumPy op: {op_type}")

        return x[0] if squeeze_out else x


DEFAULT_MICRODUCK_JOINT_POS = (
    "0.000,-0.087,-0.458,-0.005,0.453,0.349,0.349,0.000,0.000,0.000,0.087,0.458,0.005,-0.453"
)
DEFAULT_MICRODUCK_JOINT_NAMES = (
    "left_hip_yaw,left_hip_roll,left_hip_pitch,left_knee,left_ankle,"
    "neck_pitch,head_pitch,head_yaw,head_roll,"
    "right_hip_yaw,right_hip_roll,right_hip_pitch,right_knee,right_ankle"
)


def extract_onnx_to_numpy_graph(onnx_path: str | Path) -> NumpyMlpGraph:
    """Parse an ONNX MLP policy file using the `onnx` library and convert to NumpyMlpGraph."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph

    metadata = {prop.key: prop.value for prop in model.metadata_props}
    if "default_joint_pos" not in metadata:
        metadata["default_joint_pos"] = DEFAULT_MICRODUCK_JOINT_POS
    if "joint_names" not in metadata:
        metadata["joint_names"] = DEFAULT_MICRODUCK_JOINT_NAMES
    if "action_scale" not in metadata:
        metadata["action_scale"] = "0.25"

    initializers: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init).astype(np.float32)
        for init in graph.initializer
    }

    # Also capture Constant nodes into initializers map
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    initializers[node.output[0]] = numpy_helper.to_array(attr.t).astype(np.float32)

    inp_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    input_dim = inp_shape[-1]
    output_dim = out_shape[-1]

    ops: List[dict] = []
    # Track constant folding for shape/eps tensors if any
    for node in graph.node:
        op = node.op_type
        if op in ("Constant", "Identity", "Cast", "Flatten"):
            continue
        elif op in ("Sub", "Add", "Mul", "Div", "Max", "Min"):
            # Find which input is the initializer constant
            const_val = None
            for inp_name in node.input:
                if inp_name in initializers:
                    const_val = initializers[inp_name]
                    break
            if const_val is None:
                raise ValueError(f"Node {node.name} ({op}) has no constant initializer among {node.input}")
            # Merge bias Add into preceding Linear if possible
            if op == "Add" and ops and ops[-1]["type"] == "Linear" and "b" not in ops[-1]:
                ops[-1]["b"] = const_val.reshape(-1)
            else:
                ops.append({"type": op, "const": const_val})
        elif op == "Gemm":
            transB = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
            W = initializers[node.input[1]]
            if transB:
                W = W.T
            linear_op: dict[str, Any] = {"type": "Linear", "W": W.astype(np.float32)}
            if len(node.input) > 2 and node.input[2] in initializers:
                linear_op["b"] = initializers[node.input[2]].reshape(-1).astype(np.float32)
            ops.append(linear_op)
        elif op == "MatMul":
            W = initializers[node.input[1]]
            ops.append({"type": "Linear", "W": W.astype(np.float32)})
        elif op == "Elu":
            alpha = 1.0
            for attr in node.attribute:
                if attr.name == "alpha":
                    alpha = float(attr.f)
            ops.append({"type": "Elu", "alpha": alpha})
        elif op in ("Relu", "Tanh"):
            ops.append({"type": op})
        elif op == "Clip":
            lo = float(initializers[node.input[1]]) if len(node.input) > 1 and node.input[1] in initializers else -np.inf
            hi = float(initializers[node.input[2]]) if len(node.input) > 2 and node.input[2] in initializers else np.inf
            ops.append({"type": "Clip", "min": lo, "max": hi})
        else:
            raise ValueError(f"Unsupported ONNX op_type {op!r} in {onnx_path}")

    return NumpyMlpGraph(ops=ops, metadata=metadata, input_dim=input_dim, output_dim=output_dim)

