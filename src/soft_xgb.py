"""Differentiable PyTorch surrogate for ordinary numeric-split XGBoost ensembles.

Features:
- soft: smooth routing with sigmoid((threshold - x)/temperature)
- hard: exactly reproduces XGBoost routing (for finite float32 inputs)
- ste: original hard values in the forward pass, soft routing gradients
- sigmoid transformation for binary:logistic and identity for squared error
- no XGBoost dependency when loading a converted .pt checkpoint

Does not support categorical splits or multi-output ensembles. Missing values
are not supported by the differentiable path; pre-process them exactly as the
original PepTune scorer does before calling the module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _Level(nn.Module):
    def __init__(self, leaf_indices, leaf_values, internal_indices,
                 split_features, thresholds, parent_internal, next_is_yes):
        super().__init__()
        self.register_buffer("leaf_indices", torch.as_tensor(leaf_indices, dtype=torch.long))
        self.register_buffer("leaf_values", torch.as_tensor(leaf_values, dtype=torch.float32))
        self.register_buffer("internal_indices", torch.as_tensor(internal_indices, dtype=torch.long))
        self.register_buffer("split_features", torch.as_tensor(split_features, dtype=torch.long))
        self.register_buffer("thresholds", torch.as_tensor(thresholds, dtype=torch.float32))
        self.register_buffer("parent_internal", torch.as_tensor(parent_internal, dtype=torch.long))
        self.register_buffer("next_is_yes", torch.as_tensor(next_is_yes, dtype=torch.bool))


def _feature_index(split: Any, feature_names: list[str] | None) -> int:
    if not isinstance(split, str):
        raise ValueError(f"Unexpected feature name: {split!r}")
    if feature_names and split in feature_names:
        return feature_names.index(split)
    if split.startswith("f") and split[1:].isdigit():
        return int(split[1:])
    raise ValueError(f"Cannot map split feature {split!r} to an integer index")


def _children(node: dict) -> tuple[dict, dict]:
    if isinstance(node.get("split_condition"), list):
        raise NotImplementedError("Categorical splits are not supported")
    children = {int(child["nodeid"]): child for child in node["children"]}
    return children[int(node["yes"])], children[int(node["no"])]


def _at_leaf(node: dict, x: np.ndarray, feature_names: list[str] | None) -> float:
    while "leaf" not in node:
        feat = _feature_index(node["split"], feature_names)
        yes, no = _children(node)
        node = yes if float(x[feat]) < float(node["split_condition"]) else no
    return float(node["leaf"])


class SoftXGBoost(nn.Module):
    """Differentiable conversion of a scalar-output XGBoost model.

    Input:  float32 embeddings [batch, n_features]
    Output: predictions [batch].
    """

    def __init__(self, trees: list[dict], *, n_features: int, bias: float,
                 objective: str, feature_names: list[str] | None = None,
                 temperature: float = 0.10):
        super().__init__()
        if not trees:
            raise ValueError("At least one tree is required")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if objective not in {"reg:squarederror", "reg:logistic", "binary:logistic"}:
            raise NotImplementedError(f"XGBoost objective {objective!r} is not supported")
        self.n_features = int(n_features)
        self.objective = str(objective)
        self.feature_names = feature_names
        self.temperature = float(temperature)
        self.register_buffer("bias", torch.tensor(float(bias), dtype=torch.float32))
        self._trees = trees  # JSON-compatible primitives for checkpoint serialization
        levels = []
        current = [(tree, None, None) for tree in trees]
        while current:
            leaves, values = [], []
            internals, features, thresholds = [], [], []
            next_level = []
            for node_index, (node, _, _) in enumerate(current):
                if "leaf" in node:
                    leaves.append(node_index)
                    values.append(float(node["leaf"]))
                else:
                    split_index = _feature_index(node["split"], feature_names)
                    if split_index < 0 or split_index >= self.n_features:
                        raise ValueError(f"Tree feature {split_index} outside input dimension")
                    local_parent = len(internals)
                    internals.append(node_index)
                    features.append(split_index)
                    thresholds.append(float(node["split_condition"]))
                    yes, no = _children(node)
                    next_level.extend(((yes, local_parent, True), (no, local_parent, False)))
            levels.append(_Level(
                leaves, values, internals, features, thresholds,
                [entry[1] for entry in next_level],
                [entry[2] for entry in next_level],
            ))
            current = next_level
        self.levels = nn.ModuleList(levels)
        self.n_trees = len(trees)

    @classmethod
    def from_xgboost(cls, booster, *, temperature: float = 0.10) -> "SoftXGBoost":
        """Convert an existing xgboost.Booster into a differentiable module."""
        import xgboost as xgb

        config = json.loads(booster.save_config())
        learner = config["learner"]
        objective = learner["objective"]["name"]
        model_params = learner["learner_model_param"]
        if int(model_params.get("num_class", "0")) > 1 or int(model_params.get("num_target", "1")) > 1:
            raise NotImplementedError("Multi-target or multi-class XGBoost models are not supported")
        trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]
        n_features = booster.num_features()
        names = booster.feature_names

        # Calibrate the XGBoost margin intercept directly. This also works when
        # XGBoost stores a probability-scale base_score (versions >= 3.1).
        anchor = np.zeros((1, n_features), dtype=np.float32)
        matrix = xgb.DMatrix(anchor, feature_names=names)
        margin = np.asarray(booster.predict(matrix, output_margin=True)).reshape(-1)
        if len(margin) != 1:
            raise ValueError("Expected one scalar margin per sample")
        leaf_sum = sum(_at_leaf(tree, anchor[0], names) for tree in trees)
        bias = float(margin[0] - leaf_sum)
        return cls(trees, n_features=n_features, bias=bias,
                   objective=objective, feature_names=names, temperature=temperature)

    @classmethod
    def from_json(cls, path: str | Path, *, temperature: float = 0.10) -> "SoftXGBoost":
        import xgboost as xgb
        booster = xgb.Booster(model_file=str(path))
        return cls.from_xgboost(booster, temperature=temperature)

    def forward(self, x: torch.Tensor, *, mode: str = "soft",
                temperature: float | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expected input [B, {self.n_features}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input should be floating point")
        if not torch.isfinite(x).all():
            raise ValueError("Pre-process NaN/Inf inputs before passing them to SoftXGBoost")
        if mode not in ("soft", "hard", "ste"):
            raise ValueError("mode must be 'soft', 'hard', or 'ste'")
        tau = self.temperature if temperature is None else float(temperature)
        if tau <= 0:
            raise ValueError("temperature must be positive")
        # XGBoost uses float32 split inputs. Do not silently cast to float64.
        if x.dtype != torch.float32:
            raise TypeError("Use float32 features to reproduce XGBoost comparisons")

        prob = x.new_ones((x.shape[0], self.n_trees))
        margin = self.bias.to(dtype=x.dtype).expand(x.shape[0]).clone()
        for level in self.levels:
            if level.leaf_indices.numel():
                margin = margin + (
                    prob.index_select(1, level.leaf_indices) * level.leaf_values
                ).sum(dim=1)
            if not level.internal_indices.numel():
                break
            parent_prob = prob.index_select(1, level.internal_indices)
            input_feature = x.index_select(1, level.split_features)
            if mode == "hard":
                yes_prob = (input_feature < level.thresholds).to(dtype=x.dtype)
            else:
                smooth = torch.sigmoid((level.thresholds - input_feature) / tau)
                if mode == "ste":
                    hard = (input_feature < level.thresholds).to(dtype=x.dtype)
                    yes_prob = hard + smooth - smooth.detach()
                else:
                    yes_prob = smooth
            route = torch.where(level.next_is_yes, yes_prob.index_select(1, level.parent_internal),
                                1 - yes_prob.index_select(1, level.parent_internal))
            prob = parent_prob.index_select(1, level.parent_internal) * route
        if self.objective in ("binary:logistic", "reg:logistic"):
            return torch.sigmoid(margin)
        return margin

    def export(self, path: str | Path) -> None:
        """Store pure data, loadable without xgboost or executable pickle classes."""
        torch.save({"trees": self._trees, "n_features": self.n_features,
                    "bias": self.bias.item(), "objective": self.objective,
                    "feature_names": self.feature_names,
                    "temperature": self.temperature}, str(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location="cpu") -> "SoftXGBoost":
        package = torch.load(str(path), map_location=map_location, weights_only=True)
        return cls(**package)
