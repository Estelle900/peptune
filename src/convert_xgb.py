"""CLI: convert PepTune's four XGBoost JSON files to autograd-ready PyTorch.

Usage from repository root:
python src/convert_peptune_xgb.py \
    --repo . \
    --temperature 0.1
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from soft_xgb import SoftXGBoost

NAMES = ("solubility", "permeability", "nonfouling", "hemolysis")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True, help="Root of Estelle900/peptune")
    p.add_argument("--out", type=Path, default=None, help="Output directory for .pt models")
    p.add_argument("--temperature", type=float, default=0.1)
    a = p.parse_args()
    src = a.repo / "src/scoring/functions/classifiers"
    out = a.out or (src / "soft")
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        path = src / f"{name}-xgboost.json"
        booster = xgb.Booster(model_file=str(path))
        model = SoftXGBoost.from_xgboost(booster, temperature=a.temperature)
        # Verify hard-path fidelity against independent random feature rows.
        rng = np.random.default_rng(7)
        data = rng.standard_normal((128, model.n_features)).astype(np.float32)
        reference = booster.predict(xgb.DMatrix(data, feature_names=booster.feature_names))
        with torch.no_grad():
            got = model(torch.from_numpy(data), mode="hard").cpu().numpy()
        error = float(np.max(np.abs(got - reference)))
        if not np.allclose(got, reference, rtol=2e-5, atol=2e-5):
            raise RuntimeError(f"{name}: hard prediction mismatch max error={error}")
        target = out / f"{name}-soft.pt"
        model.export(target)
        print(f"{name:13s} {model.n_trees:5d} trees  {model.n_features:4d} features "
              f" objective={model.objective:17s} max_abs_err={error:.3g} -> {target}")


if __name__ == "__main__":
    main()
