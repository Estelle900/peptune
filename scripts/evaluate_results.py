#!/usr/bin/env python3
"""Evaluate PepTune generation outputs, correcting historical MCTS score labels.

From the repository root:
    python scripts/evaluate_results.py results/
    python scripts/evaluate_results.py results_unconditional/
    python scripts/evaluate_results.py results/ --compare-with results_unconditional/

Writes CSV summaries to <directory>/evaluation_summary/ by default.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
import pandas as pd
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.app import PeptideAnalyzer
from rdkit import rdBase

SMILES = "Generated SMILES"
PEPTIDE = "Peptide Sequence"
PERPLEXITY = "Perplexity"

# These are output labels for the original CSVs, NOT necessarily the metrics
# actually present in each MCTS column (see score_order_for_mcts below).
LEGACY_SCORE_COLUMNS = (
    "Binding Affinity", "Binding Affinity 1", "Binding Affinity 2",
    "Solubility", "Hemolysis", "Nonfouling", "Permeability",
)
METRICS = (
    PERPLEXITY, "Binding Affinity", "Binding Affinity 2",
    "Solubility", "Hemolysis", "Nonfouling", "Permeability",
)

# Actual score order configured inside generate_mcts.generate_valid_mcts.
MCTS_SCORE_ORDERS = {
    "perm": ("permeability", "binding_affinity1", "solubility", "hemolysis", "nonfouling"),
    "dual": ("binding_affinity1", "solubility", "hemolysis", "nonfouling", "binding_affinity2"),
    "standard": ("binding_affinity1", "solubility", "hemolysis", "nonfouling"),
    "single-binding": ("binding_affinity1",),
    "single-permeability": ("permeability",),
}
SCORE_NAME_TO_METRIC = {
    "binding_affinity1": "Binding Affinity",
    "binding_affinity2": "Binding Affinity 2",
    "permeability": "Permeability",
    "solubility": "Solubility",
    "hemolysis": "Hemolysis",  # model's NON-hemolysis score (larger is safer)
    "nonfouling": "Nonfouling",
}


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def csv_files(base: Path, output: Path) -> list[Path]:
    files = []
    for path in sorted(base.rglob("*.csv")):
        relative = path.relative_to(base)
        if not path.is_file() or path.name.startswith("log_"):
            continue
        if "evaluation_summary" in relative.parts:
            continue
        if output != base and output in path.parents:
            continue
        # Expected layout: <root>/<protein>/<run>.csv
        if len(relative.parts) < 2:
            continue
        files.append(path)
    return files

def filter_valid_peptides(frame, analyzer, metadata):
    """Keep valid peptides and record every excluded row."""
    valid_flags = []
    omitted = []

    for i, smiles in enumerate(frame[SMILES]):
        reason = None

        if not isinstance(smiles, str) or not smiles.strip():
            is_valid = False
            reason = "missing or empty SMILES"
        else:
            try:
                with rdBase.BlockLogs():
                    is_valid = bool(analyzer.is_peptide(smiles))
                if not is_valid:
                    reason = "PeptideAnalyzer.is_peptide returned False"
            except Exception as exc:
                is_valid = False
                reason = f"PeptideAnalyzer error: {exc}"

        valid_flags.append(is_valid)

        if not is_valid:
            omitted.append({
                **metadata,
                "csv_row": i + 2,
                SMILES: smiles,
                PEPTIDE: (
                    frame[PEPTIDE].iloc[i]
                    if PEPTIDE in frame else None
                ),
                "reason": reason,
            })

    return frame.loc[valid_flags].copy(), omitted


def validity_counts(n_saved, n_valid):
    return {
        "n_saved_rows": n_saved,
        "n_invalid_omitted": n_saved - n_valid,
        "valid_fraction_among_saved": (
            n_valid / n_saved if n_saved else float("nan")
        ),
    }

def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[values.notna() & values.map(math.isfinite)]


def counts(frame: pd.DataFrame) -> dict:
    result = {"n_peptides": len(frame)}  # stored outputs, NOT generation attempts
    for column, suffix in ((SMILES, "smiles"), (PEPTIDE, "peptide_sequence")):
        if column not in frame:
            result[f"n_unique_{suffix}"] = pd.NA
            result[f"{suffix}_unique_fraction"] = float("nan")
            continue
        seqs = frame[column].dropna().astype(str).str.strip()
        seqs = seqs[seqs.ne("")]
        n = int(seqs.nunique())
        result[f"n_unique_{suffix}"] = n
        result[f"{suffix}_unique_fraction"] = n / len(seqs) if len(seqs) else float("nan")
    return result


def metric_rows(frame: pd.DataFrame, metadata: dict) -> list[dict]:
    records = []
    for metric in METRICS:
        if metric not in frame.columns:
            continue
        series = numeric_values(frame, metric)
        records.append({
            **metadata, "metric": metric,
            "count": len(series), "missing_count": len(frame) - len(series),
            "mean": series.mean(),
            "std": series.std(ddof=1) if len(series) > 1 else float("nan"),
            "median": series.median(), "min": series.min(), "max": series.max(),
        })
    return records


def describe_across_targets(values: pd.Series) -> dict:
    """Describe each target's value with one vote per target, excluding NaN/inf."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[numeric.notna() & numeric.map(math.isfinite)]
    return {
        "mean": valid.mean(),
        # Here std measures *between-target* dispersion, not within-target
        # peptide variation. An n=1 macro std is undefined.
        "std": valid.std(ddof=1) if len(valid) > 1 else float("nan"),
        "median": valid.median(),
        "min": valid.min(),
        "max": valid.max(),
    }


def macro_metric_rows(per_target: pd.DataFrame, n_targets_total: int) -> pd.DataFrame:
    """Equal-weight aggregate of per-target means (one mean per target).

    'count' and 'n_targets' count contributing targets, not peptides.
    'n_peptides' records how many scored peptide rows underpin those means.
    """
    rows = []
    for (method, metric), group in per_target.groupby(["method", "metric"], sort=False):
        means = pd.to_numeric(group["mean"], errors="coerce")
        sample_counts = pd.to_numeric(group["count"], errors="coerce")
        eligible = (sample_counts > 0) & means.notna() & means.map(math.isfinite)
        usable = group.loc[eligible]
        n_targets = len(usable)
        rows.append({
            "method": method, "target": "ALL", "aggregation": "macro",
            "metric": metric, "count": n_targets, "n_targets": n_targets,
            "n_targets_total": n_targets_total,
            "missing_target_count": n_targets_total - n_targets,
            "n_peptides": int(sample_counts.loc[eligible].sum()),
            **describe_across_targets(usable["mean"]),
        })
    return pd.DataFrame(rows)


MACRO_COUNT_FIELDS = (
    "n_saved_rows",
    "n_peptides",
    "n_invalid_omitted",
    "valid_fraction_among_saved",
    "n_unique_smiles",
    "smiles_unique_fraction",
    "n_unique_peptide_sequence",
    "peptide_sequence_unique_fraction",
    "estimated_attempts",
    "estimated_valid_fraction",
)


def macro_count_rows(per_target: pd.DataFrame) -> pd.DataFrame:
    """Target-equal descriptive stats for saved counts and uniqueness rates.

    Note that a mean unique-count per target is NOT a global unique count.
    """
    rows = []
    for name in MACRO_COUNT_FIELDS:
        if name not in per_target.columns:
            continue
        series = pd.to_numeric(per_target[name], errors="coerce")
        eligible = series.notna() & series.map(math.isfinite)
        rows.append({
            "method": per_target["method"].iloc[0],
            "target": "ALL", "aggregation": "macro", "metric": name,
            "count": int(eligible.sum()), "n_targets": int(eligible.sum()),
            "n_targets_total": len(per_target),
            "missing_target_count": int((~eligible).sum()),
            **describe_across_targets(series.loc[eligible]),
        })
    return pd.DataFrame(rows)


def read_sampling_attempts(config_path: Path) -> int | None:
    """Read sampling.num_sequences from YAML without needing PyYAML.

    This is ONLY a *current-configuration estimate*, not a recorded run budget.
    """
    try:
        source = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    within_sampling = False
    for line in source.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^sampling\s*:\s*(?:#.*)?$", line):
            within_sampling = True
            continue
        if within_sampling and line[0] not in " \t":
            break
        if within_sampling:
            found = re.match(r"^\s+num_sequences\s*:\s*(\d+)\s*(?:#.*)?$", line)
            if found:
                number = int(found.group(1))
                return number if number > 0 else None
    return None


def configured_attempts(base: Path, override: int | None) -> tuple[int | None, str]:
    if override is not None:
        return override, "--attempts-per-file"
    # The sbatch scripts regenerate config.yaml from config.yaml.template.
    for name in ("src/config.yaml", "src/config.yaml.template"):
        path = base.parent / name
        value = read_sampling_attempts(path)
        if value is not None:
            return value, str(path.relative_to(base.parent)) + " (current, unverified)"
    return None, "not recorded"


def read_log_header(log_path: Path) -> tuple[str, ...] | None:
    try:
        headers = list(pd.read_csv(log_path, nrows=0).columns)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        warn(f"cannot read MCTS log header {log_path}: {exc}")
        return None
    return tuple(name for name in headers if name not in ("Iteration", "Valid Fraction"))


def log_rollout_stats(log_path: Path, metadata: dict) -> dict | None:
    """MCTS rollout validity is conditional on adaptive tree search, not an
    unconditional generation success rate.
    """
    if not log_path.exists():
        return None
    try:
        data = pd.read_csv(log_path, usecols=["Valid Fraction"])
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        warn(f"cannot read validity from {log_path}: {exc}")
        return None
    series = numeric_values(data, "Valid Fraction")
    series = series[series.between(0, 1)]
    return {
        **metadata, "n_iterations": len(series),
        "mean_rollout_valid_fraction": series.mean(),
        "last_rollout_valid_fraction": series.iloc[-1] if len(series) else float("nan"),
        "min_rollout_valid_fraction": series.min(),
        "max_rollout_valid_fraction": series.max(),
    }


def score_order_for_mcts(
    path: Path, score_columns: list[str], override: str
) -> tuple[tuple[str, ...] | None, str]:
    """Return the actual order of the raw *values* in the MCTS CSV.

    For historical files the written CSV headers can be incorrect. The
    corresponding log headers follow self.score_func_names, which is the
    exact order ScoringFunctions returns values in.
    """
    if override == "raw":
        return None, "raw_unverified"
    if override != "auto":
        return MCTS_SCORE_ORDERS[override], f"manual:{override}"
    log_path = path.with_name("log_" + path.name)
    if log_path.exists():
        names = read_log_header(log_path)
        if names and len(names) == len(score_columns) and all(
            name in SCORE_NAME_TO_METRIC for name in names
        ) and len(set(names)) == len(names):
            return names, "matching_log_header"
        warn(f"ambiguous MCTS score order for {path}: log headers={names}; "
             "property metrics omitted. Use --mcts-score-order to override.")
    else:
        warn(f"missing matching MCTS log: {log_path}; property metrics omitted. "
             "Use --mcts-score-order perm/dual/standard if known.")
    return None, "unknown_skipped"


def normalise_scores(frame: pd.DataFrame, path: Path, method: str,
                     override: str, metadata: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Keep raw data untouched; create correctly labeled, in-memory scores."""
    output = frame[[c for c in (SMILES, PEPTIDE, PERPLEXITY) if c in frame]].copy()
    legacy = [c for c in frame.columns if c in LEGACY_SCORE_COLUMNS]
    mappings = []
    if method == "guided":
        order, source = score_order_for_mcts(path, legacy, override)
        if source == "raw_unverified":
            order = tuple(
                {"Binding Affinity": "binding_affinity1", "Binding Affinity 1": "binding_affinity1",
                 "Binding Affinity 2": "binding_affinity2", "Solubility": "solubility",
                 "Hemolysis": "hemolysis", "Nonfouling": "nonfouling",
                 "Permeability": "permeability"}[column]
                for column in legacy
            )
        if order is None or len(order) != len(legacy):
            if order is not None:
                warn(f"score order length mismatch in {path}: {len(order)} vs {len(legacy)}; skipping scores")
            mappings.append({**metadata, "csv_column": "(all property columns)",
                             "actual_metric": "UNKNOWN", "mapping_source": source})
            return output, mappings
        for raw_col, actual_name in zip(legacy, order):
            canonical = SCORE_NAME_TO_METRIC[actual_name]
            if canonical in output.columns:
                warn(f"duplicate resolved metric {canonical} in {path}; skipping duplicate")
                continue
            output[canonical] = frame[raw_col]
            mappings.append({**metadata, "csv_column": raw_col,
                             "actual_metric": canonical, "mapping_source": source})
    else:
        for raw_col in legacy:
            canonical = "Binding Affinity" if raw_col == "Binding Affinity 1" else raw_col
            output[canonical] = frame[raw_col]
            mappings.append({**metadata, "csv_column": raw_col,
                             "actual_metric": canonical, "mapping_source": "unconditional_csv_header"})
    return output, mappings


def infer_method(directory: Path) -> str:
    if directory.name.endswith("results") or directory.name.endswith("results_reliable_scores"):
        return "guided"
    if directory.name.endswith("results_unconditional"):
        return "unconditional"
    raise ValueError("Input directory must be named ...results/ or ...results_unconditional/")


def summarize(base: Path, output: Path, order_override: str,
              attempts_override: int | None) -> dict[str, pd.DataFrame]:
    method = infer_method(base)
    analyzer = PeptideAnalyzer()
    omitted_rows = []
    attempts, source = configured_attempts(base, attempts_override) if method == "unconditional" else (None, "not applicable")
    per_file_counts, per_file_metrics, score_mappings, rollout_validity = [], [], [], []
    grouped: dict[str, list[pd.DataFrame]] = {}
    all_frames: list[pd.DataFrame] = []

    for path in csv_files(base, output):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, OSError) as exc:
            warn(f"skipping unreadable CSV {path}: {exc}")
            continue
        if SMILES not in frame.columns:
            warn(f"skipping non-peptide CSV {path}")
            continue
        target = path.relative_to(base).parts[0]
        meta = {"method": method, "target": target, "file": str(path.relative_to(base))}

        # Validate every generated SMILES using the same peptide rule.
        valid_frame, omitted = filter_valid_peptides(
            frame, analyzer, meta
        )
        omitted_rows.extend(omitted)

        if omitted:
            warn(
                f"Omitting {len(omitted)} invalid peptides "
                f"from {path}"
            )

        # All subsequent metrics use only valid peptides.
        clean, mapping = normalise_scores(
            valid_frame, path, method, order_override, meta
        )
        score_mappings.extend(mapping)
        count_row = {
            **meta,
            **counts(clean),
            **validity_counts(len(frame), len(clean)),
        }
        if method == "unconditional":
            # Assumes this generation file reflects a COMPLETE run. We cannot
            # reconstruct actual attempted generations from valid-only CSVs.
            count_row.update({
                "estimated_attempts": attempts if attempts is not None else pd.NA,
                "attempts_source": source,
                "estimated_valid_fraction": len(clean) / attempts if attempts else float("nan"),
            })
            if attempts and len(frame) > attempts:
                warn(f"saved rows exceed assumed attempts in {path}: {len(frame)} > {attempts}")
        else:
            log_stats = log_rollout_stats(path.with_name("log_" + path.name), meta)
            if log_stats is not None:
                rollout_validity.append(log_stats)
        per_file_counts.append(count_row)
        per_file_metrics.extend(metric_rows(clean, meta))
        grouped.setdefault(target, []).append(clean)
        all_frames.append(clean)

    if not all_frames:
        raise ValueError(f"No generation CSVs found in {base}")

    per_target_counts, per_target_metrics = [], []
    for target, frames in sorted(grouped.items()):
        combined = pd.concat(frames, ignore_index=True, sort=False)
        meta = {"method": method, "target": target, "n_files": len(frames)}
        n_saved = sum(r["n_saved_rows"] for r in per_file_counts if r["target"] == target)

        row = {**meta,  **counts(combined), **validity_counts(n_saved, len(combined)),}
        if method == "unconditional":
            total_attempts = attempts * len(frames) if attempts else None
            row.update({
                "estimated_attempts": total_attempts if total_attempts is not None else pd.NA,
                "attempts_source": source,
                "estimated_valid_fraction": len(combined) / total_attempts if total_attempts else float("nan"),
            })
        per_target_counts.append(row)
        per_target_metrics.extend(metric_rows(combined, meta))

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    overall_meta = {"method": method, "target": "ALL", "n_files": len(all_frames)}
    n_saved_total = sum(r["n_saved_rows"] for r in per_file_counts)

    overall_count = {
        **overall_meta,
        **counts(combined),
        **validity_counts(n_saved_total, len(combined)),
    }
    if method == "unconditional":
        total_attempts = attempts * len(all_frames) if attempts else None
        overall_count.update({
            "estimated_attempts": total_attempts if total_attempts is not None else pd.NA,
            "attempts_source": source,
            "estimated_valid_fraction": len(combined) / total_attempts if total_attempts else float("nan"),
        })

    target_counts_table = pd.DataFrame(per_target_counts)
    target_metrics_table = pd.DataFrame(per_target_metrics)
    micro_counts_table = pd.DataFrame([{**overall_count, "aggregation": "micro"}])
    micro_metrics_table = pd.DataFrame([
        {**row, "aggregation": "micro"}
        for row in metric_rows(combined, overall_meta)
    ])
    macro_counts_table = macro_count_rows(target_counts_table)
    macro_metrics_table = macro_metric_rows(target_metrics_table, len(target_counts_table))

    results = {
        "per_file_counts.csv": pd.DataFrame(per_file_counts),
        "per_file_metrics.csv": pd.DataFrame(per_file_metrics),
        "per_target_counts.csv": target_counts_table,
        "per_target_metrics.csv": target_metrics_table,
        # Keep the old filenames for existing downstream consumers; these
        # are micro (peptide-pooled) statistics, not target-weighted ones.
        "overall_counts.csv": micro_counts_table,
        "overall_metrics.csv": micro_metrics_table,
        "micro_counts.csv": micro_counts_table,
        "micro_metrics.csv": micro_metrics_table,
        "macro_counts.csv": macro_counts_table,
        "macro_metrics.csv": macro_metrics_table,
        "score_mappings.csv": pd.DataFrame(score_mappings),
        "invalid_peptides.csv": pd.DataFrame(omitted_rows, columns=[
                "method","target","file","csv_row", SMILES, PEPTIDE,"reason"
            ]),
    }
    if method == "guided":
        results["mcts_rollout_validity.csv"] = pd.DataFrame(rollout_validity)
    return results


def compare_summaries(main: dict[str, pd.DataFrame],
                      other: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compare both methods. Differences are always guided - unconditional.

    Micro comparison is peptide-pooled across each method's input targets.
    Macro comparison uses only paired target/metric means, so it gives every
    *matched* target equal weight even if peptide counts differ.
    """
    output = {}
    for level in ("per_target", "micro"):
        first = main[f"{level}_metrics.csv"]
        second = other[f"{level}_metrics.csv"]
        both = pd.concat((first, second), ignore_index=True)
        guided = both[both["method"] == "guided"]
        uncontrolled = both[both["method"] == "unconditional"]
        paired = guided.merge(uncontrolled, on=["target", "metric"], suffixes=("_guided", "_unconditional"))
        if paired.empty:
            warn(f"no matched target + metric pairs for {level} comparison")
        else:
            paired["mean_difference_guided_minus_unconditional"] = paired["mean_guided"] - paired["mean_unconditional"]
            paired["comparison_note"] = (
                "valid saved peptides; unmatched sample sizes; no significance claim"
            )
        output[f"comparison_{level}_metrics.csv"] = paired
        if level == "micro":
            # Previous filename was 'overall', which is the micro aggregate.
            output["comparison_overall_metrics.csv"] = paired

    matched = output["comparison_per_target_metrics.csv"]
    macro_rows = []
    for metric, group in matched.groupby("metric", sort=False):
        g = pd.to_numeric(group["mean_guided"], errors="coerce")
        u = pd.to_numeric(group["mean_unconditional"], errors="coerce")
        c_g = pd.to_numeric(group["count_guided"], errors="coerce")
        c_u = pd.to_numeric(group["count_unconditional"], errors="coerce")
        eligible = ((c_g > 0) & (c_u > 0) & g.notna() & u.notna()
                    & g.map(math.isfinite) & u.map(math.isfinite))
        g, u = g.loc[eligible], u.loc[eligible]
        difference = g - u
        n_targets = len(difference)
        macro_rows.append({
            "aggregation": "macro_paired_targets", "metric": metric,
            "n_matched_targets": n_targets,
            "n_guided_peptides": int(c_g.loc[eligible].sum()),
            "n_unconditional_peptides": int(c_u.loc[eligible].sum()),
            "mean_guided": g.mean(),
            "std_guided_across_targets": g.std(ddof=1) if n_targets > 1 else float("nan"),
            "mean_unconditional": u.mean(),
            "std_unconditional_across_targets": u.std(ddof=1) if n_targets > 1 else float("nan"),
            "mean_difference_guided_minus_unconditional": difference.mean(),
            "std_difference_across_targets": difference.std(ddof=1) if n_targets > 1 else float("nan"),
            "median_difference_guided_minus_unconditional": difference.median(),
            "comparison_note": "paired per-target mean differences; not a significance test",
        })
    output["comparison_macro_metrics.csv"] = pd.DataFrame(macro_rows, columns=(
        "aggregation", "metric", "n_matched_targets", "n_guided_peptides",
        "n_unconditional_peptides", "mean_guided", "std_guided_across_targets",
        "mean_unconditional", "std_unconditional_across_targets",
        "mean_difference_guided_minus_unconditional", "std_difference_across_targets",
        "median_difference_guided_minus_unconditional", "comparison_note",
    ))
    if not macro_rows:
        warn("no matched target + metric pairs for macro comparison")
    return output


def save_results(data: dict[str, pd.DataFrame], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, table in data.items():
        # Preserve informative headers for an empty output.
        table.to_csv(output / name, index=False, float_format="%.8g")
        print(f"Saved {output / name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="...results/ OR ...results_unconditional/ OR ...results_reliable_scores")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <directory>/evaluation_summary/)")
    parser.add_argument("--attempts-per-file", type=int, default=None,
                        help="Override attempted unconditional samples per file (e.g. 100). "
                             "Otherwise estimates from current src/config.yaml/template if found.")
    parser.add_argument("--mcts-score-order", choices=("auto", *MCTS_SCORE_ORDERS, "raw"), default="auto",
                        help="MCTS score order; auto (default) reads matching log_<run>.csv headers. "
                             "Without logs, ambiguous scores are omitted. raw trusts old labels (unsafe).")
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="Optionally compare to the other results directory, matching target/metric.")
    args = parser.parse_args()
    if args.attempts_per_file is not None and args.attempts_per_file <= 0:
        parser.error("--attempts-per-file must be positive")
    base = args.directory.expanduser().resolve()
    if not base.is_dir():
        parser.error(f"not a directory: {base}")
    try:
        infer_method(base)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.expanduser().resolve() if args.output else base / "evaluation_summary"
    try:
        results = summarize(base, output, args.mcts_score_order, args.attempts_per_file)
    except ValueError as exc:
        parser.error(str(exc))

    if args.compare_with is not None:
        other = args.compare_with.expanduser().resolve()
        if not other.is_dir():
            parser.error(f"comparison directory not found: {other}")
        if infer_method(other) == infer_method(base):
            parser.error("--compare-with must be the other generation method")
        try:
            compared = summarize(other, output, args.mcts_score_order, args.attempts_per_file)
        except ValueError as exc:
            parser.error(str(exc))
        
        # Include exclusions from both guided and unconditional runs.
        results["invalid_peptides.csv"] = pd.concat(
            [
                results["invalid_peptides.csv"],
                compared["invalid_peptides.csv"],
            ],
            ignore_index=True,
        )

        results.update(compare_summaries(results, compared))

    save_results(results, output)
    print(
        f"\nTotal invalid peptides omitted: "
        f"{len(results['invalid_peptides.csv'])}"
    )
    print(f"Invalid peptide details: {output / 'invalid_peptides.csv'}")
    print(f"\nEvaluated {len(results['per_file_counts.csv'])} generation files from {base}")
    print("\nOverall saved peptide counts (not MCTS generation-attempt counts):")
    print(results["overall_counts.csv"].to_string(index=False))
    print("\nMICRO property metrics (pooled across all saved peptides):")
    table = results["micro_metrics.csv"]
    print(table[["method", "metric", "count", "mean", "std"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMACRO property metrics (equal weight per target; std across target means):")
    table = results["macro_metrics.csv"]
    print(table[["method", "metric", "n_targets", "mean", "std"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMACRO count/diversity statistics (equal weight per target):")
    table = results["macro_counts.csv"]
    show = table[table["metric"].isin((
        "n_peptides", "peptide_sequence_unique_fraction", "estimated_valid_fraction"
    ))]
    print(show[["method", "metric", "n_targets", "mean", "std"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    if "comparison_macro_metrics.csv" in results:
        print("\nPaired-target MACRO differences (guided minus unconditional):")
        table = results["comparison_macro_metrics.csv"]
        print(table[["metric", "n_matched_targets", "mean_difference_guided_minus_unconditional"]].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"))
    if infer_method(base) == "guided":
        n = len(results["mcts_rollout_validity.csv"])
        print(f"\nMCTS rollout-validity logs found: {n}; these are NOT unconditional validity rates.")
    print("\nNOTE: No model rescoring performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

