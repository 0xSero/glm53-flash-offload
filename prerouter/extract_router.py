#!/usr/bin/env python3
"""N134: dump the 42 MoE routers of GLM-5.3-Flash (gate.weight [288, 4096], e_score_correction_bias [288]) to npz (fp32).
Run inside the serving image (CPU only): python3 extract_router.py /models /out/router.npz"""
import json, os, sys
import numpy as np
from safetensors import safe_open

md, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(md, "config.json")))["text_config"]
first, L = cfg["first_k_dense_replace"], cfg["num_hidden_layers"]
wm = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
W, B = [], []
for li in range(first, L):
    kw = f"model.language_model.layers.{li}.mlp.gate.weight"
    kb = f"model.language_model.layers.{li}.mlp.gate.e_score_correction_bias"
    for k, dst in ((kw, W), (kb, B)):
        with safe_open(os.path.join(md, wm[k]), framework="pt") as f:
            dst.append(f.get_tensor(k).float().numpy())
W, B = np.stack(W), np.stack(B)
np.savez(out, gate=W, bias=B, first=first, routed_scaling=cfg["routed_scaling_factor"], topk=cfg["num_experts_per_tok"])
print(out, W.shape, W.dtype, B.shape, float(np.abs(B).mean()))
