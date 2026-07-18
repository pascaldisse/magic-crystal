#!/usr/bin/env python3
"""Parity: CoreML fp16 prediction vs Rust f32 golden (golden.json)."""
import json, sys
import numpy as np
import coremltools as ct

model_path = sys.argv[1]
golden = json.load(open(sys.argv[2]))
X = np.array(golden["features"], dtype=np.float32)
Y = np.array(golden["outputs"], dtype=np.float32)

m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
out_name = m.get_spec().description.output[0].name
pred = m.predict({"x": X})[out_name].astype(np.float32)

abs_err = np.abs(pred - Y)
rel = abs_err / (np.abs(Y) + 1e-4)
print(f"N={X.shape[0]}  out range golden [{Y.min():.4f},{Y.max():.4f}]")
print(f"max_abs_err={abs_err.max():.6f}  mean_abs_err={abs_err.mean():.6f}")
print(f"max_rel_err={rel.max():.6f}  mean_rel_err={rel.mean():.6f}")
# fp16 has ~3-4 decimal digits; tolerance derived from fp16 eps accumulated
# over 6 layers of width 64: ~ 2^-10 * sqrt(64*6) ~ 2e-2 relative worst-case.
tol = 2e-2
ok = rel.max() < tol or abs_err.max() < 5e-3
print("PARITY", "PASS" if ok else "FAIL", f"(tol rel<{tol} or abs<5e-3)")
sys.exit(0 if ok else 1)
