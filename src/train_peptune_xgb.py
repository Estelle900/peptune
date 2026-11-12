#!/usr/bin/env python3
"""Train PepTune-compatible solubility, hemolysis and nonfouling XGBoost heads.

The PepTune scoring functions expect mean-pooled PeptideCLM RoFormer embeddings
of SMILES.
This script:
  Decodes PeptideBERT/peptide-dashboard .npz files to one-letter sequences,
  Constructs L-peptide SMILES with RDKit
  Embeds with the SAME PepTune tokenizer/checkpoint,
  Trains binary:logistic boosters with stratified 70/20/10 splits,
  Saves results in output-dir.

Run from the root of programmablebio/peptune:
    python src/train_peptune_xgb.py --repo-root . --data-dir PeptideBERT/data --output-dir xgboost_results --n-trials 50 --threshold-mode validation_f1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import rdkit
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    log_loss, mean_squared_error, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("peptune_xgb")

# Original peptide-dashboard convention, retained by PeptideBERT BEFORE
# convert_encodings.py (0 is padding).
DASHBOARD_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
DASHBOARD_IDS = {i + 1: aa for i, aa in enumerate(DASHBOARD_ALPHABET)}

# convert_encodings.py remaps those IDs to PeptideBERT's ProtBERT ID ordering.
PEPTIDEBERT_IDS = {i: aa for i, aa in enumerate(
    ["L", "A", "G", "V", "E", "S", "I", "K", "R", "D", "T", "P",
     "N", "Q", "F", "Y", "M", "H", "C", "W"], start=5)}
VALID_AA = frozenset(DASHBOARD_ALPHABET)

# Paper thresholds on positive (hemolytic/soluble/nonfouling) probabilities.
# PepTune's hemolysis *reward* is 1 - P(hemolytic), with reward cutoff 0.8,
# corresponding to P(hemolytic) cutoff 0.2 (direction is reversed for rewards).
PAPER_RAW_POSITIVE_THRESHOLDS = {
    "solubility": 0.50, "hemolysis": 0.20, "nonfouling": 0.45,
}
TASKS = {
    "solubility": (("sol-positive.npz", "soluble.npz"),
                   ("sol-negative.npz", "insoluble.npz")),
    "hemolysis": (("hemo-positive.npz",), ("hemo-negative.npz",)),
    "nonfouling": (("nf-positive.npz", "human-positive.npz"),
                   ("nf-negative.npz", "human-negative.npz")),
}


def json_ready(value: Any) -> Any:
    """Convert numpy scalars, NaNs and Path objects to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2, allow_nan=False) + "\n")


def choose_file(data_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing source files in {data_dir}. Expected one of {names}. "
        "Use the original *-positive.npz/*-negative.npz files, not the "
        "PeptideBERT pre-generated train/val/test splits."
    )


def read_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        for key in ("arr_0", "seqs", "inputs"):
            if key in bundle:
                arr = np.asarray(bundle[key])
                break
        else:
            raise ValueError(f"{path}: expected array key arr_0, seqs, or inputs; got {bundle.files}")
    if arr.ndim == 2 and arr.shape[1] == 1 and arr.dtype.kind in "US":
        arr = arr[:, 0]
    if arr.ndim not in (1, 2):
        raise ValueError(f"{path}: unsupported dataset shape {arr.shape}")
    return arr


def identify_encoding(arrays: list[np.ndarray], choice: str) -> str:
    if choice != "auto":
        return choice
    int_arrays = [arr for arr in arrays if arr.dtype.kind in "iuf"]
    if not int_arrays:
        return "sequence"
    values = np.unique(np.concatenate([np.unique(a).astype(np.int64) for a in int_arrays]))
    values = values[values != 0]
    if len(values) == 0:
        raise ValueError("All source arrays contain padding only.")
    if np.any(values < 0) or np.any(values > 24):
        raise ValueError(f"Unrecognized encoded amino acid IDs: {values.tolist()}")
    dashboard_marker = np.any((values >= 1) & (values <= 4))
    pb_marker = np.any(values >= 21)
    if dashboard_marker and pb_marker:
        raise ValueError(
            "The two class files appear to use DIFFERENT encodings (raw 1..20 "
            "and converted 5..24). Re-download/re-convert both together."
        )
    if dashboard_marker:
        return "dashboard"
    if pb_marker:
        return "peptidebert"
    raise ValueError(
        "Cannot infer encoding: observed IDs only in the ambiguous 5..20 range. "
        "Use --encoding dashboard (original peptide-dashboard files) or "
        "--encoding peptidebert (after PeptideBERT convert_encodings.py)."
    )


def decode_sequences(arr: np.ndarray, encoding: str) -> list[str]:
    if arr.dtype.kind in "US":
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D text sequences; got {arr.shape}")
        return [str(s).strip().upper() for s in arr]
    if arr.dtype.kind not in "iuf" or arr.ndim != 2:
        raise ValueError(f"Expected (N, length) integer IDs or (N,) text; got {arr.shape}, {arr.dtype}")
    alphabet = DASHBOARD_IDS if encoding == "dashboard" else PEPTIDEBERT_IDS
    if encoding not in ("dashboard", "peptidebert"):
        raise ValueError(f"Cannot decode numeric data using encoding={encoding}")
    sequences = []
    for row in arr:
        ids = np.asarray(row)
        if not np.all(np.equal(ids, np.floor(ids))):
            raise ValueError("Amino-acid token IDs must be integer-valued")
        ids = ids.astype(int)
        ids = ids[ids != 0]  # 0 is padding (the PeptideBERT files are width 200)
        unknown = set(ids) - set(alphabet)
        if unknown:
            raise ValueError(f"Unknown residue IDs {sorted(unknown)} for encoding {encoding}")
        sequences.append("".join(alphabet[i] for i in ids))
    return sequences


def aa_to_smiles(sequence: str) -> str | None:
    """Generate an unmodified, L-amino-acid linear peptide SMILES with RDKit."""
    if not sequence or set(sequence) - VALID_AA:
        return None
    mol = Chem.MolFromSequence(sequence, sanitize=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def prepare_npz_dataset(task: str, data_dir: Path, encoding: str) -> tuple[pd.DataFrame, dict]:
    positive_path = choose_file(data_dir, TASKS[task][0])
    negative_path = choose_file(data_dir, TASKS[task][1])
    positive, negative = read_npz(positive_path), read_npz(negative_path)
    effective_encoding = identify_encoding([positive, negative], encoding)
    LOGGER.info("%s: input=%s + %s; encoding=%s", task, positive_path.name,
                negative_path.name, effective_encoding)
    pos_seq = decode_sequences(positive, effective_encoding)
    neg_seq = decode_sequences(negative, effective_encoding)

    # Deduplicate *before* the split; remove sequences with conflicting labels
    # so no identical peptide can leak into train, validation and test.
    by_seq: dict[str, set[int]] = {}
    for s, y in [(s, 1) for s in pos_seq] + [(s, 0) for s in neg_seq]:
        by_seq.setdefault(s, set()).add(y)
    conflicts = {s for s, ys in by_seq.items() if len(ys) > 1}
    clean = [(seq, next(iter(labels))) for seq, labels in by_seq.items()
             if seq not in conflicts and seq and not (set(seq) - VALID_AA)]
    data = []
    invalid_smiles = 0
    for seq, y in clean:
        smiles = aa_to_smiles(seq)
        if smiles is None:
            invalid_smiles += 1
            continue
        data.append((seq, smiles, y))
    frame = pd.DataFrame(data, columns=["sequence", "smiles", "label"])
    if len(frame) < 20 or frame.label.nunique() != 2 or frame.label.value_counts().min() < 10:
        raise ValueError(f"{task}: need at least 10 unique sequences per class (got {frame.label.value_counts().to_dict()})")
    stats = {
        "positive_file": str(positive_path), "negative_file": str(negative_path),
        "encoding": effective_encoding,
        "positive_rows_original": len(pos_seq), "negative_rows_original": len(neg_seq),
        "distinct_conflicting_sequences_dropped": len(conflicts),
        "invalid_amino_acid_sequences_dropped": len(by_seq) - len(conflicts) - len(clean),
        "rdkit_conversion_failures": invalid_smiles, "unique_sequences_used": len(frame),
        "positive_count": int(frame.label.sum()),
        "negative_count": int((frame.label == 0).sum()),
    }
    LOGGER.info("%s: %s unique labeled sequences (%s positive / %s negative); "
                "%s contradictions removed", task, len(frame), stats["positive_count"],
                stats["negative_count"], len(conflicts))
    return frame, stats

def read_pepland_txt(path: Path) -> pd.DataFrame:
    """Read headerless PepLand sequence,label files."""

    raw = pd.read_csv(
        path,
        header=None,
        names=["sequence", "label"],
        dtype=str,
    )

    raw["sequence"] = raw["sequence"].str.strip().str.upper()

    labels = pd.to_numeric(raw["label"], errors="raise")

    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path}: expected binary 0/1 labels")

    raw["label"] = labels.astype(int)

    # Convert amino acid sequences into PepTune-compatible SMILES
    raw["smiles"] = raw["sequence"].map(aa_to_smiles)

    # Discard invalid sequences
    result = raw.dropna(subset=["smiles"]).copy()

    result["source"] = path.name

    LOGGER.info(
        "%s: retained %d/%d valid peptides",
        path.name, len(result), len(raw)
    )

    return result[["sequence", "smiles", "label", "source"]]

def read_pepland_nc_cpp(path: Path) -> pd.DataFrame:
    """Read non-canonical CPP molecular structures and binary labels."""

    raw = pd.read_csv(path, dtype=str)

    by_name = {
        str(c).strip().lower(): c
        for c in raw.columns
    }

    mol_key = next(
        (by_name[c] for c in (
            "smiles", "smi", "peptide_smiles",
            "canonical_smiles", "molecule",
            "sequence", "seq",
        ) if c in by_name),
        None,
    )

    y_key = next(
        (by_name[c] for c in (
            "label", "class", "target",
            "cpp", "is_cpp", "permeability", "y",
        ) if c in by_name),
        None,
    )

    # Allow headerless two-column SMILES,label files.
    if mol_key is None or y_key is None:
        if raw.shape[1] != 2:
            raise ValueError(
                f"{path}: cannot identify SMILES/label columns. "
                f"Found {list(raw.columns)}"
            )

        raw = pd.read_csv(
            path,
            header=None,
            names=["smiles", "label"],
            dtype=str,
        )

        mol_key, y_key = "smiles", "label"

    values = raw[mol_key].fillna("").astype(str).str.strip()
    labels = pd.to_numeric(raw[y_key], errors="raise")

    if not labels.isin([0, 1]).all():
        raise ValueError(
            f"{path}: expected binary 0/1 CPP labels"
        )

    is_sequence = str(mol_key).strip().lower() in (
        "sequence", "seq"
    )

    rows = []

    for value, label in zip(values, labels):
        if is_sequence:
            sequence = value.upper()
            smiles = aa_to_smiles(sequence)
        else:
            sequence = ""
            mol = Chem.MolFromSmiles(value) if value else None
            smiles = (
                Chem.MolToSmiles(
                    mol,
                    isomericSmiles=True,
                    canonical=True,
                )
                if mol is not None else None
            )

        if smiles is not None:
            rows.append(
                (sequence, smiles, int(label), path.name)
            )

    LOGGER.info(
        "%s: retained %d/%d valid molecules",
        path.name, len(rows), len(raw)
    )

    return pd.DataFrame(
        rows,
        columns=["sequence", "smiles", "label", "source"],
    )

def merge_labeled_frames(
    frames: list[pd.DataFrame],
    task: str,
) -> tuple[pd.DataFrame, int]:

    combined = pd.concat(frames, ignore_index=True)

    # Find molecules with contradictory labels.
    label_counts = combined.groupby("smiles")["label"].nunique()

    conflicts = set(
        label_counts[label_counts > 1].index
    )

    # Drop all examples of molecules with conflicting labels.
    combined = combined[
        ~combined["smiles"].isin(conflicts)
    ]

    # Keep one example per molecule.
    combined = combined.drop_duplicates(
        subset="smiles"
    ).reset_index(drop=True)

    counts = combined["label"].value_counts()

    if (
        len(combined) < 20
        or len(counts) != 2
        or counts.min() < 10
    ):
        raise ValueError(
            f"{task}: insufficient examples per class: "
            f"{counts.to_dict()}"
        )

    return combined, len(conflicts)

def prepare_dataset(
    task: str,
    data_dir: Path,
    encoding: str,
    pepland_data_dir: Path,
) -> tuple[pd.DataFrame, dict]:

    frames = []
    original_stats = {}

    if task == "permeability":

        # Canonical cell-penetrating peptides
        frames.append(
            read_pepland_txt(
                pepland_data_dir / "c-CPP.txt"
            )
        )

        # Non-canonical cell-penetrating peptides
        nc_path = pepland_data_dir / "nc-CPP.csv"

        if nc_path.is_file():
            frames.append(
                read_pepland_nc_cpp(nc_path)
            )

    else:
        # Existing PeptideBERT data loader
        base, original_stats = prepare_npz_dataset(
            task, data_dir, encoding
        )

        base["source"] = "peptidebert_npz"
        frames.append(base)

        if task == "solubility":
            frames.append(
                read_pepland_txt(
                    pepland_data_dir / "c-Sol.txt"
                )
            )

    # Deduplicate before splitting.
    frame, n_conflicts = merge_labeled_frames(
        frames, task
    )

    stats = {
        "original_dataset": original_stats,
        "sources": frame["source"].value_counts().to_dict(),
        "conflicting_smiles_dropped": n_conflicts,
        "unique_molecules_used": len(frame),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int((frame["label"] == 0).sum()),
    }

    LOGGER.info(
        "%s: %d unique molecules, %d positive, %d negative",
        task,
        len(frame),
        stats["positive_count"],
        stats["negative_count"],
    )

    return frame, stats

def stratified(y: np.ndarray, tvt=(.8, .1, .1), seed: int = 42) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    train_val, test = train_test_split(indices, test_size=tvt[2],
                                       random_state=seed, stratify=y)
    train, val = train_test_split(train_val, test_size=(tvt[1] / (tvt[0]+tvt[1])),
                                  random_state=seed, stratify=y[train_val])
    return {"train": np.asarray(train), "val": np.asarray(val), "test": np.asarray(test)}


class PepTuneEmbedder:
    """Frozen PepTune scoring head feature extractor (RoFormer + SMILES SPE)."""

    def __init__(self, repo_root: Path, hf_model: str, device: str, max_tokens: int):
        try:
            import torch
            from transformers import AutoModelForMaskedLM
        except ImportError as exc:
            raise ImportError("Embedding requires torch and transformers; use PepTune's src/environment.yml") from exc
        from importlib.util import module_from_spec, spec_from_file_location

        candidates = [
            repo_root / "src/scoring/tokenizer/my_tokenizers.py",
            repo_root / "src/scoring/functions/tokenizer/my_tokenizers.py",
            repo_root / "src/tokenizer/my_tokenizers.py",
        ]
        tokenizer_file = next((p for p in candidates if p.exists()), None)
        if tokenizer_file is None:
            raise FileNotFoundError(f"Cannot locate PepTune SMILES_SPE_Tokenizer under {repo_root}")
        vocab = tokenizer_file.parent / "new_vocab.txt"
        splits = tokenizer_file.parent / "new_splits.txt"
        for path in (vocab, splits):
            if not path.exists():
                raise FileNotFoundError(f"Missing PepTune tokenizer asset: {path}")
        spec = spec_from_file_location("peptune_scoring_smiles_tokenizer", tokenizer_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import tokenizer from {tokenizer_file}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tokenizer = module.SMILES_SPE_Tokenizer(str(vocab), str(splits))
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                                   ("cpu" if device == "auto" else device))
        LOGGER.info("Loading %s on %s (first run may download weights)", hf_model, self.device)
        self.model = AutoModelForMaskedLM.from_pretrained(hf_model).roformer.to(self.device).eval()
        self.torch = torch
        self.max_tokens = max_tokens
        self.model_name = hf_model
        self.signature = {
            "model": hf_model,
            "max_tokens": max_tokens,
            "tokenizer_sha256": hashlib.sha256(vocab.read_bytes() + splits.read_bytes()).hexdigest(),
            "pooling": "mean over non-padding tokens (includes [CLS]/[SEP])",
            "smiles_generation": "RDKit Chem.MolFromSequence, canonical isomeric SMILES",
        }

    def transform(self, smiles: list[str], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        torch = self.torch
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start:start + batch_size]
            toks = self.tokenizer(batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_tokens)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                result = self.model(**toks)
                hidden = result.last_hidden_state
                # Mean pooling, but exclude padding introduced by batching.
                attn = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
            vectors.append(pooled.float().cpu().numpy())
            if (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(smiles):
                LOGGER.info("Feature extraction: %s/%s sequences", min(start + batch_size, len(smiles)), len(smiles))
        out = np.concatenate(vectors, axis=0).astype(np.float32)
        return np.nan_to_num(out, nan=0, posinf=np.finfo(np.float32).max,
                             neginf=np.finfo(np.float32).min)

class ECFP4Embedder:
    def __init__(
        self,
        n_bits: int = 1024,
        include_chirality: bool = False,
    ):
        self.n_bits = n_bits
        self.include_chirality = include_chirality

        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,                       # ECFP4
            fpSize=n_bits,                  # 1024 dimensions
            includeChirality=include_chirality,
        )

        # Compatible with your existing cache hashing
        self.signature = {
            "name": "ECFP4_count",
            "radius": 2,
            "n_bits": n_bits,
            "include_chirality": include_chirality,
            "rdkit_version": rdkit.__version__,
        }

    def transform(
        self,
        smiles: list[str],
        batch_size: int = 256,
    ) -> np.ndarray:

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        X = np.zeros(
            (len(smiles), self.n_bits),
            dtype=np.float32,
        )

        for start in range(0, len(smiles), batch_size):
            end = min(start + batch_size, len(smiles))

            for i in range(start, end):
                mol = Chem.MolFromSmiles(smiles[i])

                if mol is None:
                    raise ValueError(
                        f"Invalid SMILES at index {i}: {smiles[i]}"
                    )

                # Count-based, NOT binary
                X[i] = self.generator.GetCountFingerprintAsNumPy(mol)

        return X


def get_features(task: str, smiles: list[str], embedder,
                 batch_size: int, cache_dir: Path, cache: bool) -> np.ndarray:
    sig = json.dumps(embedder.signature, sort_keys=True).encode()
    h = hashlib.sha256(sig)
    for s in smiles:
        h.update(s.encode()); h.update(b"\0")
    path = cache_dir / f"{task}_{h.hexdigest()[:20]}.npz"
    if cache and path.exists():
        with np.load(path, allow_pickle=False) as item:
            X = item["X"]
        if len(X) != len(smiles):
            raise ValueError(f"Stale feature cache at {path}")
        LOGGER.info("Loaded feature cache %s: %s", path, X.shape)
        return X
    X = embedder.transform(smiles, batch_size)
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=X)
        LOGGER.info("Saved feature cache: %s", path)
    return X


def metric_scores(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y_true).astype(int)
    pred = (p >= threshold).astype(int)
    with np.errstate(all="ignore"):
        corr = float(spearmanr(y, p).statistic) if len(np.unique(y)) > 1 and np.std(p) > 0 else float("nan")
    return {
        "n": int(len(y)), "positive_count": int(y.sum()),
        "threshold_positive_class": threshold,
        "spearman": corr,
        "mse": float(mean_squared_error(y, p)),  # Brier score for binary labels
        "brier_score": float(mean_squared_error(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_positive": float(f1_score(y, pred, zero_division=0)),
        "precision_positive": float(precision_score(y, pred, zero_division=0)),
        "recall_positive": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def f1_optimal_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    # Tuned ONLY on validation data. Limit range to prevent pathological extremes.
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in grid]
    best = np.flatnonzero(np.array(scores) == max(scores))
    # Among ties, choose the candidate closest to 0.5 for stability.
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


def trial_params(trial: optuna.Trial, seed: int, n_jobs: int) -> dict:
    """Classification search space: PepTune paper, Table 10 (50 trials)."""
    return {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10., log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10., log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "seed": seed, "nthread": n_jobs, "verbosity": 0,
    }


def fit_one(params: dict, dtrain: xgb.DMatrix, dval: xgb.DMatrix,
            max_rounds: int, patience: int) -> tuple[xgb.Booster, dict]:
    history: dict = {}
    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=max_rounds,
        evals=[(dtrain, "train"), (dval, "val")], evals_result=history,
        callbacks=[xgb.callback.EarlyStopping(rounds=patience,
                                              metric_name="logloss", data_name="val",
                                              save_best=True)],
        verbose_eval=False,
    )
    return booster, history


def plot_learning_curve(history: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ("train", "val"):
        ax.plot(history[split]["logloss"], label=f"{split} logloss")
    ax.set(xlabel="Boosting round (0-indexed)", ylabel="Binary log loss",
           title="XGBoost learning curve (best trial)")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(output, dpi=180); plt.close(fig)


def plot_pred_vs_true(y: np.ndarray, p: np.ndarray, output: Path,
                      task: str, spearman: float, mse: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter_x = y + rng.uniform(-0.12, 0.12, len(y))
    for cls, color in ((0, "#4C78A8"), (1, "#F58518")):
        mask = y == cls
        ax.scatter(jitter_x[mask], p[mask], alpha=0.30, s=14,
                   label=f"True {cls} (n={mask.sum()})", color=color)
    for cls in (0, 1):
        ax.hlines(float(np.mean(p[y == cls])), cls - 0.2, cls + 0.2,
                  color="black", linewidth=2)
    ax.set(xticks=[0, 1], xlim=(-0.35, 1.35), ylim=(-0.03, 1.03),
           xlabel=f"True {task} positive-class label (0/1)",
           ylabel="Predicted positive-class probability",
           title=f"{task.title()} — held-out test set\nSpearman={spearman:.3f}   MSE={mse:.4f}")
    ax.grid(alpha=0.15); ax.legend(loc="best", fontsize=9); fig.tight_layout()
    fig.savefig(output, dpi=180); plt.close(fig)


def plot_test_diagnostics(y: np.ndarray, p: np.ndarray, threshold: float,
                          output: Path, task: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, label=f"AP={average_precision_score(y, p):.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Test PR")
    axes[1].legend()
    mat = confusion_matrix(y, p >= threshold, labels=[0, 1])
    axes[2].imshow(mat, cmap="Blues")
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True",
                title=f"Confusion matrix @ {threshold:.3f}")
    for (row, col), n in np.ndenumerate(mat):
        axes[2].text(col, row, str(n), ha="center", va="center")
    fig.suptitle(f"{task.title()} — held-out test diagnostics")
    for ax in axes[:2]: ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def plot_trials(trials: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trials["number"], trials["value"], marker="o", ms=3, lw=0.8,
            label="Trial validation F1")
    ax.plot(trials["number"], trials["value"].cummax(), lw=2,
            label="Best validation F1 so far")
    ax.set(xlabel="Optuna trial", ylabel="Validation F1", title="Hyperparameter search")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(output, dpi=180); plt.close(fig)


def configure_task_log(folder: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    for h in LOGGER.handlers[:]: LOGGER.removeHandler(h); h.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(folder / "training.log", encoding="utf-8", mode="w")):
        handler.setFormatter(formatter); LOGGER.addHandler(handler)


def train_task(task: str, args: argparse.Namespace, embedder: PepTuneEmbedder) -> dict:
    folder = args.output_dir / task
    folder.mkdir(parents=True, exist_ok=True)
    configure_task_log(folder)
    LOGGER.info("Starting %s | seed=%s | 80/10/10 train/val/test", task, args.seed)
    frame, dataset_stats = prepare_dataset(task, args.data_dir, args.encoding, args.data_dir)
    y = frame["label"].to_numpy(dtype=np.int32)
    splits = stratified(y, tvt=(.8, .1, .1), seed=args.seed)
    frame["split"] = ""
    for name, idx in splits.items(): frame.loc[idx, "split"] = name
    frame.to_csv(folder / "split_manifest.csv", index=False)
    counts = {name: {"n": len(idx), "positive": int(y[idx].sum()),
                     "negative": int((y[idx] == 0).sum())} for name, idx in splits.items()}
    write_json(folder / "dataset_summary.json", {"source": dataset_stats, "splits": counts})
    LOGGER.info("Split counts: %s", counts)

    X = get_features(task, frame.smiles.tolist(), embedder, args.batch_size,
                     args.output_dir / "embedding_cache", not args.no_cache)
    if X.shape[0] != len(frame) or not np.all(np.isfinite(X)):
        raise ValueError(f"Bad feature matrix: {X.shape}")
    LOGGER.info("Feature matrix: %s", X.shape)
    write_json(folder / "feature_provenance.json", {
        **embedder.signature, "feature_dim": X.shape[1],
        "input_data": {
            "peptidebert": str(args.data_dir),
            "pepland": str(args.pepland_data_dir),
        },
        "data_split": [0.7, 0.2, 0.1],
    })

    dm = {name: xgb.DMatrix(X[idx], label=y[idx]) for name, idx in splits.items()}
    p_thr = PAPER_RAW_POSITIVE_THRESHOLDS.get(task, 0.5) # permeability default not paper-derived; uninformative fallback
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"{task}_70_20_10")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial, args.seed, args.n_jobs)
        booster, hist = fit_one(params, dm["train"], dm["val"],
                                args.max_rounds, args.early_stopping_rounds)
        p_val = booster.predict(dm["val"])
        threshold = (p_thr if args.threshold_mode == "paper" else
                     f1_optimal_threshold(y[splits["val"]], p_val))
        metrics = metric_scores(y[splits["val"]], p_val, threshold)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("val_spearman", metrics["spearman"])
        trial.set_user_attr("val_mse", metrics["mse"])
        trial.set_user_attr("num_trained_rounds", len(hist["train"]["logloss"]))
        LOGGER.info("Trial %03d / %03d | val F1=%.4f | Spearman=%.4f | "
                    "MSE=%.4f | n_trees=%d | threshold=%.3f",
                    trial.number + 1, args.n_trials, metrics["f1_positive"],
                    metrics["spearman"], metrics["mse"],
                    int(booster.best_iteration) + 1, threshold)
        return metrics["f1_positive"]

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    trial_frame = study.trials_dataframe()
    trial_frame.to_csv(folder / "optuna_trials.csv", index=False)
    plot_trials(trial_frame, folder / "optuna_search.png")

    best_trial = study.best_trial
    params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "tree_method": "hist", "seed": args.seed, "nthread": args.n_jobs,
              "verbosity": 0, **best_trial.params}
    LOGGER.info("Best trial %d | F1 %.4f | params %s", best_trial.number,
                best_trial.value, best_trial.params)
    booster, history = fit_one(params, dm["train"], dm["val"],
                               args.max_rounds, args.early_stopping_rounds)
    threshold = (p_thr if args.threshold_mode == "paper" else
                 f1_optimal_threshold(y[splits["val"]], booster.predict(dm["val"])))
    # IMPORTANT: xgb.callback.EarlyStopping(save_best=True) trims the model to
    # the selected best iteration, so PepTune's Booster.predict uses the right trees.
    model_path = folder / f"{task}-xgboost.json"
    booster.save_model(str(model_path))
    pd.DataFrame({"round": np.arange(len(history["train"]["logloss"])),
                  "train_logloss": history["train"]["logloss"],
                  "val_logloss": history["val"]["logloss"]}).to_csv(
                      folder / "learning_curve.csv", index=False)
    plot_learning_curve(history, folder / "learning_curve.png")
    write_json(folder / "best_params.json", {
        "params": params, "best_trial": best_trial.number,
        "best_round_0_indexed": int(booster.best_iteration),
        "threshold_mode": args.threshold_mode,
        "threshold_positive_class": threshold,
        "paper_threshold_positive_class": (p_thr if task in PAPER_RAW_POSITIVE_THRESHOLDS else None),
        "default_threshold_positive_class": p_thr,
        "paper_hemolysis_reward_cutoff": 0.8 if task == "hemolysis" else None,
        "selection_metric": "validation positive-class F1",
        "validation_early_stopping_metric": "logloss",
        "training_only_70pct": True,
    })

    all_metrics = {}
    for split_name, idx in splits.items():
        prob = booster.predict(dm[split_name])
        stats = metric_scores(y[idx], prob, threshold)
        all_metrics[split_name] = stats
        result = frame.loc[idx, ["sequence", "smiles", "label"]].copy()
        result = result.rename(columns={"label": "y_true_positive"})
        result["predicted_probability_positive"] = prob
        result["predicted_label_positive"] = (prob >= threshold).astype(int)
        result["peptune_reward_score"] = 1.0 - prob if task == "hemolysis" else prob
        if task == "hemolysis":
            result["y_true_nonhemolytic"] = 1 - result["y_true_positive"]
        result.to_csv(folder / f"{split_name}_predictions.csv", index=False)
        LOGGER.info("%s / %s: Spearman=%s MSE=%.5f F1=%.4f AUC=%s", task,
                    split_name, f"{stats['spearman']:.4f}", stats["mse"],
                    stats["f1_positive"], f"{stats['roc_auc']:.4f}")
        if split_name == "test":
            plot_pred_vs_true(y[idx], prob, folder / "test_pred_vs_true.png",
                              task, stats["spearman"], stats["mse"], args.seed)
            plot_test_diagnostics(y[idx], prob, threshold,
                                  folder / "test_roc_pr_confusion.png", task)
    write_json(folder / "metrics.json", all_metrics)
    write_json(folder / "test_metrics.json", all_metrics["test"])
    # Save gain-based importance for interpretability (768 embedding dimensions).
    importance = booster.get_score(importance_type="gain")
    pd.DataFrame(sorted(importance.items(), key=lambda pair: -pair[1]),
                 columns=["embedding_feature", "gain"]).to_csv(
                     folder / "feature_importance.csv", index=False)
    if args.install_classifiers:
        dest = args.repo_root / "src/scoring/functions/classifiers" / model_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            backup = dest.with_name(dest.stem + ".backup." +
                                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + dest.suffix)
            shutil.copy2(dest, backup)
            LOGGER.warning("Backed up existing PepTune weights to %s", backup)
        shutil.copy2(model_path, dest)
        LOGGER.info("Installed booster at %s", dest)
    return all_metrics["test"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path("."),
                   help="Root directory of the programmablebio/peptune checkout")
    p.add_argument("--data-dir", type=Path, default=Path("xgboost_data"),
                   help="Directory containing the six raw +/- .npz files")
    p.add_argument("--pepland-data-dir", type=Path, default=Path("xgboost_data"),
                   help="Directory containing PepLand c-Sol.txt, c-CPP.txt and nc-CPP.csv")
    p.add_argument("--output-dir", type=Path, default=Path("xgboost_results"))
    p.add_argument("--tasks", nargs="+", default=["solubility", "hemolysis", "nonfouling", "permeability"],
                   choices=list(TASKS))
    p.add_argument("--encoding", choices=["auto", "dashboard", "peptidebert"],
                   default="auto", help="Encoding of *numeric* .npz peptide arrays")
    p.add_argument("--model-id", default="aaronfeller/PeptideCLM-23M-all")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=lambda v: int(v) if v else None, default=None)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--max-rounds", type=int, default=1000)
    p.add_argument("--early-stopping-rounds", type=int, default=40)
    p.add_argument("--threshold-mode", choices=["paper", "validation_f1"],
                   default="validation_f1", help="Choose positive-class cutoff on validation (recommended), or use paper cutoff")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4, help="CPU threads for XGBoost")
    p.add_argument("--no-cache", action="store_true", help="Do not save/read embedding cache")
    p.add_argument("--install-classifiers", action="store_true",
                   help="Opt-in copy to src/scoring/functions/classifiers; backs up prior files")
    p.add_argument("--ecfp4", action="store_true", help="Use count-based ECFP4 fingerprint instead of trained embedder. NOT differentiable.")
    args = p.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.data_dir = (args.repo_root / args.data_dir).resolve()
    args.output_dir = (args.repo_root / args.output_dir).resolve()
    for name in ("batch_size", "n_trials", "max_rounds", "early_stopping_rounds", "n_jobs"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    # Load once so the frozen feature encoder is shared among all three heads.
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.setLevel(logging.INFO)
    embedder = ECFP4Embedder(n_bits=1024, include_chirality=False) if args.ecfp4 else PepTuneEmbedder(args.repo_root, args.model_id, args.device, args.max_tokens)
    summary = {}
    for task in args.tasks:
        summary[task] = train_task(task, args, embedder)
    write_json(args.output_dir / "combined_test_metrics.json", summary)
    LOGGER.info("Finished; test results at %s", args.output_dir / "combined_test_metrics.json")


if __name__ == "__main__":
    main()

