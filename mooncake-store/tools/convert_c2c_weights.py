#!/usr/bin/env python3
"""
C2C Projector Weight Converter (v2 format)

Convert HuggingFace PyTorch C2C Fuser weights to Mooncake binary format.
Exports the FULL C2CProjector architecture: input proj, MLP1, proj_mlp2,
proj_out, scalar_mlp2, scalar_head, and gate logits.

Binary format v2 (magic 0xC2C20002):
  Header:
    magic: u32 = 0xC2C20002
    num_layers, src_dim, tgt_dim, hidden_dim: i32
    inter_dim, scalar_inter_dim: i32
    num_kv_heads, head_dim: i32
    num_mlp1_blocks, num_proj_mlp2_blocks, num_scalar_mlp2_blocks: i32

  Per layer:
    gate_logits: [key_f32, value_f32]
    key_in: weight[hidden, src] (row-major), bias[hidden]
    key_mlp1_blocks[N]: norm[hidden], w1[inter, hidden], w2[hidden, inter]
    key_proj_mlp2_blocks[N]: norm[hidden], w1[inter, hidden], w2[hidden, inter]
    key_proj_out: weight[tgt, hidden], bias[tgt]
    key_scalar_mlp2_blocks[N]: norm[hidden], w1[scalar_inter, hidden], w2[hidden, scalar_inter]
    key_scalar_head: weight[num_kv_heads, hidden], bias[num_kv_heads]
    (same for value_*)

Usage:
    python convert_c2c_weights.py \
        --checkpoint nics-efc/C2C_Fuser \
        --fuser qwen3_0.6b+qwen3_4b_Fuser \
        --src-hf Qwen/Qwen3-4B \
        --tgt-hf Qwen/Qwen3-0.6B \
        --output projector.bin
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from urllib.request import urlopen, Request

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MAGIC_V2 = 0xC2C20002


# ============================================================================
# HuggingFace 模型参数自动获取
# ============================================================================
def fetch_hf_config(hf_model: str) -> dict:
    """Fetch model params from HuggingFace config.json"""
    url = f"https://huggingface.co/{hf_model}/resolve/main/config.json"
    print(f"  Fetching: {url}")
    req = Request(url, headers={"User-Agent": "mooncake-c2c/1.0"})
    with urlopen(req, timeout=15) as resp:
        cfg = json.loads(resp.read())

    num_layers = cfg["num_hidden_layers"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim",
                       cfg["hidden_size"] // cfg["num_attention_heads"])

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


# ============================================================================
# 从 checkpoint 中提取 MLP block 权重
# ============================================================================
def extract_mlp_blocks(state_dict: dict, prefix: str):
    """
    Extract all MLP blocks for a given prefix (e.g. 'key_mlp1').
    C2C checkpoint format: {prefix}.blocks.{i}.norm.weight,
                           {prefix}.blocks.{i}.w1.weight,
                           {prefix}.blocks.{i}.w2.weight
    Returns list of (norm, w1, w2) tuples, auto-detecting block count.
    """
    blocks = []
    i = 0
    while True:
        norm_key = f"{prefix}.blocks.{i}.norm.weight"
        w1_key = f"{prefix}.blocks.{i}.w1.weight"
        w2_key = f"{prefix}.blocks.{i}.w2.weight"
        if w1_key not in state_dict:
            break
        norm = state_dict[norm_key].float().numpy() if norm_key in state_dict else None
        w1 = state_dict[w1_key].float().numpy()
        w2 = state_dict[w2_key].float().numpy()
        blocks.append((norm, w1, w2))
        i += 1
    return blocks


# ============================================================================
# 单层权重提取 (完整 C2CProjector 架构)
# ============================================================================
def extract_layer_weights(state_dict: dict, src_dim: int, tgt_dim: int,
                          num_kv_heads: int):
    """
    Extract all weights for one projector layer.
    Input projection takes only src_dim slice (paper concatenates src+tgt,
    but we only need the src portion for inference).
    """
    def get(name):
        return state_dict[name].float().numpy() if name in state_dict else None

    result = {}

    # Gate logits (scalar)
    key_gate = get("key_gate_logit")
    value_gate = get("value_gate_logit")
    result["key_gate_logit"] = float(key_gate) if key_gate is not None else 0.0
    result["value_gate_logit"] = float(value_gate) if value_gate is not None else 0.0

    # Input projection: [hidden, src+tgt] -> slice [hidden, src]
    for kv in ("key", "value"):
        in_w = get(f"{kv}_in.weight")  # [hidden, src+tgt]
        in_b = get(f"{kv}_in.bias")    # [hidden]
        if in_w is not None:
            result[f"{kv}_in_weight"] = in_w[:, :src_dim]  # [hidden, src]
        result[f"{kv}_in_bias"] = in_b

        # MLP1 blocks
        result[f"{kv}_mlp1"] = extract_mlp_blocks(state_dict, f"{kv}_mlp1")

        # Projection MLP2 blocks
        result[f"{kv}_proj_mlp2"] = extract_mlp_blocks(state_dict, f"{kv}_proj_mlp2")

        # Projection output
        result[f"{kv}_proj_out_weight"] = get(f"{kv}_proj_out.weight")  # [tgt, hidden]
        result[f"{kv}_proj_out_bias"] = get(f"{kv}_proj_out.bias")      # [tgt]

        # Scalar MLP2 blocks
        result[f"{kv}_scalar_mlp2"] = extract_mlp_blocks(state_dict, f"{kv}_scalar_mlp2")

        # Scalar head
        result[f"{kv}_scalar_head_weight"] = get(f"{kv}_scalar_head.weight")  # [num_heads, hidden]
        result[f"{kv}_scalar_head_bias"] = get(f"{kv}_scalar_head.bias")      # [num_heads]

    return result


# ============================================================================
# 维度自动推断
# ============================================================================
def infer_dims(weights):
    """Infer all dimensions from weight shapes."""
    dims = {}

    # hidden_dim from input projection
    w = weights.get("key_in_weight")
    if w is not None:
        dims["hidden_dim"] = w.shape[0]

    # inter_dim from mlp1 w1: [inter, hidden]
    mlp1 = weights.get("key_mlp1", [])
    if mlp1:
        dims["inter_dim"] = mlp1[0][1].shape[0]  # w1 shape[0]
    else:
        dims["inter_dim"] = dims.get("hidden_dim", 0)

    # scalar_inter_dim from scalar_mlp2 w1
    scalar_mlp2 = weights.get("key_scalar_mlp2", [])
    if scalar_mlp2:
        dims["scalar_inter_dim"] = scalar_mlp2[0][1].shape[0]
    else:
        dims["scalar_inter_dim"] = dims.get("hidden_dim", 0)

    # Block counts
    dims["num_mlp1_blocks"] = len(weights.get("key_mlp1", []))
    dims["num_proj_mlp2_blocks"] = len(weights.get("key_proj_mlp2", []))
    dims["num_scalar_mlp2_blocks"] = len(weights.get("key_scalar_mlp2", []))

    return dims


# ============================================================================
# 写入 MLP block 到二进制文件
# ============================================================================
def write_mlp_block(f, block, hidden_dim, inter_dim):
    """Write one MLP block: norm[hidden], w1[inter, hidden], w2[hidden, inter]"""
    norm, w1, w2 = block

    # RMSNorm weight [hidden_dim]
    if norm is not None:
        f.write(norm.astype(np.float32).tobytes())
    else:
        f.write(np.ones(hidden_dim, dtype=np.float32).tobytes())

    # w1: [inter_dim, hidden_dim] -> transpose to [hidden_dim, inter_dim] for C++ row-major
    f.write(w1.T.astype(np.float32).tobytes())

    # w2: [hidden_dim, inter_dim] -> transpose to [inter_dim, hidden_dim]
    f.write(w2.T.astype(np.float32).tobytes())


# ============================================================================
# v2 二进制格式写入
# ============================================================================
def write_v2_format(weights_list: list, src_dim: int, tgt_dim: int,
                    hidden_dim: int, inter_dim: int, scalar_inter_dim: int,
                    num_kv_heads: int, head_dim: int,
                    num_mlp1: int, num_proj_mlp2: int, num_scalar_mlp2: int,
                    output_path: str):
    """Write Mooncake v2 binary format with full C2CProjector weights."""
    num_layers = len(weights_list)

    with open(output_path, "wb") as f:
        # Header: 12 i32 fields
        f.write(struct.pack("I", MAGIC_V2))  # magic (unsigned)
        f.write(struct.pack("iiii", num_layers, src_dim, tgt_dim, hidden_dim))
        f.write(struct.pack("ii", inter_dim, scalar_inter_dim))
        f.write(struct.pack("ii", num_kv_heads, head_dim))
        f.write(struct.pack("iii", num_mlp1, num_proj_mlp2, num_scalar_mlp2))

        for layer_idx, w in enumerate(weights_list):
            # Gate logits
            f.write(struct.pack("ff", w["key_gate_logit"], w["value_gate_logit"]))

            for kv in ("key", "value"):
                # Input projection: weight[hidden, src] (pre-transposed), bias[hidden]
                f.write(w[f"{kv}_in_weight"].T.astype(np.float32).tobytes())
                f.write(w[f"{kv}_in_bias"].astype(np.float32).tobytes())

                # MLP1 blocks
                for blk in w[f"{kv}_mlp1"]:
                    write_mlp_block(f, blk, hidden_dim, inter_dim)

                # Proj MLP2 blocks
                for blk in w[f"{kv}_proj_mlp2"]:
                    write_mlp_block(f, blk, hidden_dim, inter_dim)

                # Proj output: weight[tgt, hidden] (pre-transposed), bias[tgt]
                f.write(w[f"{kv}_proj_out_weight"].T.astype(np.float32).tobytes())
                f.write(w[f"{kv}_proj_out_bias"].astype(np.float32).tobytes())

                # Scalar MLP2 blocks
                for blk in w[f"{kv}_scalar_mlp2"]:
                    write_mlp_block(f, blk, hidden_dim, scalar_inter_dim)

                # Scalar head: weight[num_heads, hidden] (pre-transposed), bias[num_heads]
                f.write(w[f"{kv}_scalar_head_weight"].T.astype(np.float32).tobytes())
                f.write(w[f"{kv}_scalar_head_bias"].astype(np.float32).tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"\nSaved {num_layers} layers to {output_path} ({file_size / 1024 / 1024:.1f} MB)")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Convert C2C weights to Mooncake v2 format")
    parser.add_argument("--checkpoint", type=str, default="nics-efc/C2C_Fuser",
                        help="HuggingFace checkpoint repo")
    parser.add_argument("--fuser", type=str, default="qwen3_0.6b+qwen3_4b_Fuser",
                        help="Fuser subdirectory name")
    parser.add_argument("--src-hf", type=str, required=True,
                        help="Source HuggingFace model (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--tgt-hf", type=str, required=True,
                        help="Target HuggingFace model (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--output", type=str, default="projector.bin",
                        help="Output file path")
    parser.add_argument("--local", type=str, default=None,
                        help="Local checkpoint directory")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("Error: PyTorch is required. Install with: pip install torch")
        return 1

    # 从 HuggingFace 获取模型维度
    print(f"\nFetching model configs from HuggingFace...")
    src_cfg = fetch_hf_config(args.src_hf)
    tgt_cfg = fetch_hf_config(args.tgt_hf)

    src_dim = src_cfg["num_kv_heads"] * src_cfg["head_dim"]
    tgt_dim = tgt_cfg["num_kv_heads"] * tgt_cfg["head_dim"]
    num_kv_heads = tgt_cfg["num_kv_heads"]
    head_dim = tgt_cfg["head_dim"]

    print(f"\nSource: {args.src_hf}")
    print(f"  layers={src_cfg['num_layers']}, kv_heads={src_cfg['num_kv_heads']}, "
          f"head_dim={src_cfg['head_dim']}, kv_dim={src_dim}")
    print(f"Target: {args.tgt_hf}")
    print(f"  layers={tgt_cfg['num_layers']}, kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}, kv_dim={tgt_dim}")

    # 定位 checkpoint 目录
    if args.local:
        checkpoint_dir = Path(args.local)
    else:
        from huggingface_hub import snapshot_download
        print(f"\nDownloading {args.checkpoint}/{args.fuser}...")
        checkpoint_dir = Path(snapshot_download(
            repo_id=args.checkpoint,
            allow_patterns=[f"{args.fuser}/*"]
        )) / args.fuser / "final"

    # 查找所有 projector_*.pt 文件
    pt_files = sorted(checkpoint_dir.glob("projector_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))

    if not pt_files:
        print(f"Error: No projector_*.pt files found in {checkpoint_dir}")
        return 1

    print(f"\nFound {len(pt_files)} projector files")

    # 提取所有层的权重
    weights_list = []
    for pt_file in pt_files:
        layer_idx = int(pt_file.stem.split("_")[1])
        print(f"  Loading layer {layer_idx}: {pt_file.name}")
        state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
        weights = extract_layer_weights(state_dict, src_dim, tgt_dim, num_kv_heads)
        weights_list.append(weights)

    # 从第一层推断维度
    dims = infer_dims(weights_list[0])
    hidden_dim = dims["hidden_dim"]
    inter_dim = dims["inter_dim"]
    scalar_inter_dim = dims["scalar_inter_dim"]
    num_mlp1 = dims["num_mlp1_blocks"]
    num_proj_mlp2 = dims["num_proj_mlp2_blocks"]
    num_scalar_mlp2 = dims["num_scalar_mlp2_blocks"]

    print(f"\nProjector architecture:")
    print(f"  src_dim={src_dim}, tgt_dim={tgt_dim}, hidden_dim={hidden_dim}")
    print(f"  inter_dim={inter_dim}, scalar_inter_dim={scalar_inter_dim}")
    print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  mlp1_blocks={num_mlp1}, proj_mlp2_blocks={num_proj_mlp2}, "
          f"scalar_mlp2_blocks={num_scalar_mlp2}")

    # 写入 v2 格式
    write_v2_format(weights_list, src_dim, tgt_dim, hidden_dim,
                    inter_dim, scalar_inter_dim, num_kv_heads, head_dim,
                    num_mlp1, num_proj_mlp2, num_scalar_mlp2, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
