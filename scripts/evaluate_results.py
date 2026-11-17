#!/usr/bin/env python3
"""Evaluate PepTune unconditional generation outputs.

From the repository root:
    python scripts/evaluate_results.py results_unconditional/
    python scripts/evaluate_results.py results_a/ --compare-with results_b/

Both directories are evaluated using unconditional CSV score headers.
Comparison columns use the directory names (e.g. mean_results_a and
mean_results_b), with differences calculated as first minus second.
Writes CSV summaries to <directory>/evaluation_summary/ by default.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import rdBase

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.app import PeptideAnalyzer

SMILES = "Generated SMILES"
PEPTIDE = "Peptide Sequence"
PERPLEXITY = "Perplexity"
SCORE_COLUMNS = (
    "Binding Affinity", "Binding Affinity 1", "Binding Affinity 2",
    "Solubility", "Hemolysis", "Nonfouling", "Permeability",
)
METRICS = (
    PERPLEXITY, "Binding Affinity", "Binding Affinity 2",
    "Solubility", "Hemolysis", "Nonfouling", "Permeability",
)


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
                PEPTIDE: frame[PEPTIDE].iloc[i] if PEPTIDE in frame else None,
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
    if "metric" not in per_target.columns:
        return pd.DataFrame(columns=(
            "target", "aggregation", "metric", "count", "n_targets",
            "n_targets_total", "missing_target_count", "n_peptides",
            "mean", "std", "median", "min", "max",
        ))
    for metric, group in per_target.groupby("metric", sort=False):
        means = pd.to_numeric(group["mean"], errors="coerce")
        sample_counts = pd.to_numeric(group["count"], errors="coerce")
        eligible = (sample_counts > 0) & means.notna() & means.map(math.isfinite)
        usable = group.loc[eligible]
        n_targets = len(usable)
        rows.append({
            "target": "ALL", "aggregation": "macro",
            "metric": metric, "count": n_targets, "n_targets": n_targets,
            "n_targets_total": n_targets_total,
            "missing_target_count": n_targets_total - n_targets,
            "n_peptides": int(sample_counts.loc[eligible].sum()),
            **describe_across_targets(usable["mean"]),
        })
    return pd.DataFrame(rows, columns=(
        "target", "aggregation", "metric", "count", "n_targets",
        "n_targets_total", "missing_target_count", "n_peptides",
        "mean", "std", "median", "min", "max",
    ))


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

    A mean unique-count per target is NOT a global unique count.
    """
    rows = []
    for name in MACRO_COUNT_FIELDS:
        if name not in per_target.columns:
            continue
        series = pd.to_numeric(per_target[name], errors="coerce")
        eligible = series.notna() & series.map(math.isfinite)
        rows.append({
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


def normalise_scores(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Use unconditional CSV metric headers, preserving the input data."""
    output = frame[[c for c in (SMILES, PEPTIDE, PERPLEXITY) if c in frame]].copy()
    mappings = []
    for raw_col in frame.columns:
        if raw_col not in SCORE_COLUMNS:
            continue
        canonical = "Binding Affinity" if raw_col == "Binding Affinity 1" else raw_col
        output[canonical] = frame[raw_col]
        mappings.append({
            **metadata, "csv_column": raw_col,
            "actual_metric": canonical, "mapping_source": "csv_header",
        })
    return output, mappings


def summarize(base: Path, output: Path,
              attempts_override: int | None) -> dict[str, pd.DataFrame]:
    analyzer = PeptideAnalyzer()
    attempts, source = configured_attempts(base, attempts_override)
    omitted_rows = []
    per_file_counts, per_file_metrics, score_mappings = [], [], []
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
        meta = {"target": target, "file": str(path.relative_to(base))}
        valid_frame, omitted = filter_valid_peptides(frame, analyzer, meta)
        omitted_rows.extend(omitted)

        if omitted:
            warn(f"Omitting {len(omitted)} invalid peptides from {path}")
        clean, mapping = normalise_scores(valid_frame, meta)
        score_mappings.extend(mapping)
        count_row = {
            **meta,
            **counts(clean),
            **validity_counts(len(frame), len(clean)),
        }
        # Assumes each generation file reflects a COMPLETE run. Actual attempts
        # cannot be reconstructed from valid-only CSV outputs.
        count_row.update({
            "estimated_attempts": attempts if attempts is not None else pd.NA,
            "attempts_source": source,
            "estimated_valid_fraction": len(clean) / attempts if attempts else float("nan"),
        })
        if attempts and len(frame) > attempts:
            warn(f"saved rows exceed assumed attempts in {path}: {len(frame)} > {attempts}")
        per_file_counts.append(count_row)
        per_file_metrics.extend(metric_rows(clean, meta))
        grouped.setdefault(target, []).append(clean)
        all_frames.append(clean)

    if not all_frames:
        raise ValueError(f"No generation CSVs found in {base}")

    per_target_counts, per_target_metrics = [], []
    for target, frames in sorted(grouped.items()):
        combined = pd.concat(frames, ignore_index=True, sort=False)
        meta = {"target": target, "n_files": len(frames)}
        n_saved = sum(r["n_saved_rows"] for r in per_file_counts if r["target"] == target)
        row = {**meta, **counts(combined), **validity_counts(n_saved, len(combined))}
        total_attempts = attempts * len(frames) if attempts else None
        row.update({
            "estimated_attempts": total_attempts if total_attempts is not None else pd.NA,
            "attempts_source": source,
            "estimated_valid_fraction": len(combined) / total_attempts if total_attempts else float("nan"),
        })
        per_target_counts.append(row)
        per_target_metrics.extend(metric_rows(combined, meta))

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    overall_meta = {"target": "ALL", "n_files": len(all_frames)}
    n_saved_total = sum(r["n_saved_rows"] for r in per_file_counts)
    overall_count = {
        **overall_meta,
        **counts(combined),
        **validity_counts(n_saved_total, len(combined)),
    }
    total_attempts = attempts * len(all_frames) if attempts else None
    overall_count.update({
        "estimated_attempts": total_attempts if total_attempts is not None else pd.NA,
        "attempts_source": source,
        "estimated_valid_fraction": len(combined) / total_attempts if total_attempts else float("nan"),
    })
    target_counts_table = pd.DataFrame(per_target_counts)
    target_metrics_table = pd.DataFrame(per_target_metrics, columns=(
        "target", "n_files", "metric", "count", "missing_count",
        "mean", "std", "median", "min", "max",
    ))
    micro_counts_table = pd.DataFrame([{**overall_count, "aggregation": "micro"}])
    micro_metrics_table = pd.DataFrame([
        {**row, "aggregation": "micro"}
        for row in metric_rows(combined, overall_meta)
    ], columns=(
        "target", "n_files", "metric", "count", "missing_count",
        "mean", "std", "median", "min", "max", "aggregation",
    ))
    macro_counts_table = macro_count_rows(target_counts_table)
    macro_metrics_table = macro_metric_rows(target_metrics_table, len(target_counts_table))
    return {
        "per_file_counts.csv": pd.DataFrame(per_file_counts),
        "per_file_metrics.csv": pd.DataFrame(per_file_metrics, columns=(
            "target", "file", "metric", "count", "missing_count",
            "mean", "std", "median", "min", "max",
        )),
        "per_target_counts.csv": target_counts_table,
        "per_target_metrics.csv": target_metrics_table,
        # Keep previous filenames for downstream consumers; these are micro
        # (peptide-pooled) statistics, not target-weighted statistics.
        "overall_counts.csv": micro_counts_table,
        "overall_metrics.csv": micro_metrics_table,
        "micro_counts.csv": micro_counts_table,
        "micro_metrics.csv": micro_metrics_table,
        "macro_counts.csv": macro_counts_table,
        "macro_metrics.csv": macro_metrics_table,
        "score_mappings.csv": pd.DataFrame(score_mappings, columns=[
            "target", "file", "csv_column", "actual_metric", "mapping_source",
        ]),
        "invalid_peptides.csv": pd.DataFrame(omitted_rows, columns=[
            "target", "file", "csv_row", SMILES, PEPTIDE, "reason",
        ]),
    }


def compare_summaries(main: dict[str, pd.DataFrame],
                      other: dict[str, pd.DataFrame],
                      main_name: str, other_name: str) -> dict[str, pd.DataFrame]:
    """Compare any two directories using their names as column suffixes.

    Micro means are pooled independently within each directory. Macro
    differences use equal weight per matched target with valid scores in
    *both* directories. Differences are always main minus other.
    """
    left_suffix, right_suffix = f"_{main_name}", f"_{other_name}"
    difference_col = f"mean_difference_{main_name}_minus_{other_name}"
    output = {}
    for level in ("per_target", "micro"):
        first = main[f"{level}_metrics.csv"]
        second = other[f"{level}_metrics.csv"]
        paired = first.merge(second, on=["target", "metric"],
                             suffixes=(left_suffix, right_suffix))
        if paired.empty:
            warn(f"no matched target + metric pairs for {level} comparison")
        paired[difference_col] = (paired[f"mean{left_suffix}"]
                                  - paired[f"mean{right_suffix}"])
        paired["comparison_note"] = (
            "valid saved peptides; unmatched sample sizes; no significance claim"
        )
        output[f"comparison_{level}_metrics.csv"] = paired
        if level == "micro":
            # The previous 'overall' filename is an alias for micro aggregation.
            output["comparison_overall_metrics.csv"] = paired

    macro_rows = []
    matched = output["comparison_per_target_metrics.csv"]
    for metric, group in matched.groupby("metric", sort=False):
        left = pd.to_numeric(group[f"mean{left_suffix}"], errors="coerce")
        right = pd.to_numeric(group[f"mean{right_suffix}"], errors="coerce")
        left_counts = pd.to_numeric(group[f"count{left_suffix}"], errors="coerce")
        right_counts = pd.to_numeric(group[f"count{right_suffix}"], errors="coerce")
        eligible = ((left_counts > 0) & (right_counts > 0)
                    & left.notna() & right.notna()
                    & left.map(math.isfinite) & right.map(math.isfinite))
        left, right = left.loc[eligible], right.loc[eligible]
        differences = left - right
        n_targets = len(differences)
        macro_rows.append({
            "aggregation": "macro_paired_targets", "metric": metric,
            "n_matched_targets": n_targets,
            f"n_{main_name}_peptides": int(left_counts.loc[eligible].sum()),
            f"n_{other_name}_peptides": int(right_counts.loc[eligible].sum()),
            f"mean_{main_name}": left.mean(),
            f"std_{main_name}_across_targets": (
                left.std(ddof=1) if n_targets > 1 else float("nan")
            ),
            f"mean_{other_name}": right.mean(),
            f"std_{other_name}_across_targets": (
                right.std(ddof=1) if n_targets > 1 else float("nan")
            ),
            difference_col: differences.mean(),
            "std_difference_across_targets": (
                differences.std(ddof=1) if n_targets > 1 else float("nan")
            ),
            f"median_difference_{main_name}_minus_{other_name}": differences.median(),
            "comparison_note": "paired per-target mean differences; not a significance test",
        })
    output["comparison_macro_metrics.csv"] = pd.DataFrame(macro_rows, columns=(
        "aggregation", "metric", "n_matched_targets",
        f"n_{main_name}_peptides", f"n_{other_name}_peptides",
        f"mean_{main_name}", f"std_{main_name}_across_targets",
        f"mean_{other_name}", f"std_{other_name}_across_targets",
        difference_col, "std_difference_across_targets",
        f"median_difference_{main_name}_minus_{other_name}", "comparison_note",
    ))
    if not macro_rows:
        warn("no matched target + metric pairs for macro comparison")
    return output


def save_results(data: dict[str, pd.DataFrame], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, table in data.items():
        table.to_csv(output / name, index=False, float_format="%.8g")
        print(f"Saved {output / name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Directory containing unconditional generation CSVs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <directory>/evaluation_summary/)")
    parser.add_argument("--attempts-per-file", type=int, default=None,
                        help="Override attempted samples per file (e.g. 100). "
                             "Otherwise estimates from current src/config.yaml/template if found.")
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="Compare against another results directory, using directory names as labels.")
    args = parser.parse_args()
    if args.attempts_per_file is not None and args.attempts_per_file <= 0:
        parser.error("--attempts-per-file must be positive")
    base = args.directory.expanduser().resolve()
    if not base.is_dir():
        parser.error(f"not a directory: {base}")
    other = args.compare_with.expanduser().resolve() if args.compare_with else None
    if other is not None:
        if not other.is_dir():
            parser.error(f"comparison directory not found: {other}")
        if other == base:
            parser.error("--compare-with must be a different directory")
        if other.name == base.name:
            parser.error("comparison directories must have different names for CSV column labels")
    output = args.output.expanduser().resolve() if args.output else base / "evaluation_summary"
    try:
        results = summarize(base, output, args.attempts_per_file)
        if other is not None:
            compared = summarize(other, output, args.attempts_per_file)
    except ValueError as exc:
        parser.error(str(exc))

    if other is not None:
        # Identify invalid rows by directory, without reintroducing a method.
        results["invalid_peptides.csv"] = pd.concat([
            results["invalid_peptides.csv"].assign(directory=base.name),
            compared["invalid_peptides.csv"].assign(directory=other.name),
        ], ignore_index=True)
        invalid = results["invalid_peptides.csv"]
        results["invalid_peptides.csv"] = invalid[[
            "directory", *[column for column in invalid if column != "directory"]
        ]]
        results.update(compare_summaries(results, compared, base.name, other.name))

    save_results(results, output)
    print(f"\nTotal invalid peptides omitted: {len(results['invalid_peptides.csv'])}")
    print(f"Invalid peptide details: {output / 'invalid_peptides.csv'}")
    print(f"\nEvaluated {len(results['per_file_counts.csv'])} generation files from {base}")
    print("\nOverall saved peptide counts (not generation-attempt counts):")
    print(results["overall_counts.csv"].to_string(index=False))
    print("\nMICRO property metrics (pooled across all saved peptides):")
    table = results["micro_metrics.csv"]
    print(table[["metric", "count", "mean", "std"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMACRO property metrics (equal weight per target; std across target means):")
    table = results["macro_metrics.csv"]
    print(table[["metric", "n_targets", "mean", "std"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMACRO count/diversity statistics (equal weight per target):")
    table = results["macro_counts.csv"]
    show = table[table["metric"].isin((
        "n_peptides", "peptide_sequence_unique_fraction", "estimated_valid_fraction"
    ))]
    print(show[["metric", "n_targets", "mean", "std"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    if other is not None:
        difference_col = f"mean_difference_{base.name}_minus_{other.name}"
        print(f"\nPaired-target MACRO differences ({base.name} minus {other.name}):")
        table = results["comparison_macro_metrics.csv"]
        print(table[["metric", "n_matched_targets", difference_col]].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nNOTE: No model rescoring performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

