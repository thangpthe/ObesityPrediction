"""Audit obesity CSV files without modifying source data.

Run from the repository root:
    python scripts/audit_data.py

The script writes:
    reports/data_audit.md
    reports/data_audit.json
    reports/figures/*.png (when matplotlib is installed)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_FILES = [
    Path("dataset/obesity_prediction.csv"),
    Path("dataset/obesity_prediction_dataset_final.csv"),
    Path("dataset/data_bmi.csv"),
    Path("dataset/data_weight_height.csv"),
    Path("dataset/obesity_with_bmi.csv"),
    Path("dataset/obesity_prediction_final_dataset.csv"),
    Path("data/raw/obesity_prediction.csv"),
    Path("data/raw/obesity_with_bmi.csv"),
    Path("data/raw/obesity_prediction_final_dataset.csv"),
]
TARGET = "Obesity"
BMI_INPUTS = {"Height", "Weight", "BMI"}
EXPECTED_COLUMNS = {
    "Gender", "Age", "Height", "Weight", "BMI",
    "family_history_with_overweight", "CH2O", "FAF", TARGET,
}


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def inspect_csv(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        item["status"] = "missing"
        return item
    if path.stat().st_size == 0:
        item["status"] = "empty"
        return item

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:  # report the failure instead of stopping the audit
        item.update(status="unreadable", error=f"{type(exc).__name__}: {exc}")
        return item

    item.update(
        status="readable",
        rows=int(len(frame)),
        columns=[str(column) for column in frame.columns],
        duplicate_rows=int(frame.duplicated().sum()),
        missing_by_column={
            str(column): int(count)
            for column, count in frame.isna().sum().items()
        },
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
    )
    if TARGET in frame.columns:
        counts = frame[TARGET].value_counts(dropna=False)
        item["target_counts"] = {
            str(label): int(count) for label, count in counts.items()
        }
        item["target_unique_values"] = [
            str(value) for value in frame[TARGET].dropna().unique().tolist()
        ]
    numeric = frame.select_dtypes(include="number")
    item["numeric_summary"] = {
        str(column): {
            "min": safe_float(values.min()),
            "max": safe_float(values.max()),
            "mean": safe_float(values.mean()),
            "std": safe_float(values.std()),
            "unique": int(values.nunique(dropna=True)),
        }
        for column, values in numeric.items()
    }

    if BMI_INPUTS.issubset(frame.columns):
        valid = frame[["Height", "Weight", "BMI"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        valid = valid[(valid["Height"] > 0) & (valid["Weight"] > 0)]
        if len(valid):
            errors = (
                valid["BMI"] - valid["Weight"] / valid["Height"].pow(2)
            ).abs()
            item["bmi_consistency"] = {
                "rows_checked": int(len(errors)),
                "mean_absolute_error": safe_float(errors.mean()),
                "median_absolute_error": safe_float(errors.median()),
                "p90_absolute_error": safe_float(errors.quantile(0.90)),
                "p95_absolute_error": safe_float(errors.quantile(0.95)),
                "p99_absolute_error": safe_float(errors.quantile(0.99)),
                "max_absolute_error": safe_float(errors.max()),
                "count_abs_error_gt_0_02": int((errors > 0.02).sum()),
                "count_abs_error_gt_0_1": int((errors > 0.1).sum()),
            }
    return item


def analyze_final_dataset(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = sorted(EXPECTED_COLUMNS.difference(frame.columns))
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "missing_required_columns": required,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_full_rows": int(frame.duplicated().sum()),
    }
    if TARGET not in frame.columns:
        return result

    result["class_counts"] = {
        str(label): int(count)
        for label, count in frame[TARGET].value_counts(dropna=False).items()
    }
    result["class_proportions"] = {
        str(label): float(count / len(frame))
        for label, count in frame[TARGET].value_counts(dropna=False).items()
    }

    numeric_cols = [
        column for column in ["Age", "Height", "Weight", "BMI", "CH2O", "FAF"]
        if column in frame.columns
    ]
    result["numeric_summary"] = frame[numeric_cols].describe().round(4).to_dict()

    if "BMI" in frame.columns:
        bmi = pd.to_numeric(frame["BMI"], errors="coerce")
        result["bmi_by_target"] = {}
        for label, group in frame.assign(_bmi=bmi).groupby(TARGET, dropna=False):
            values = group["_bmi"].dropna()
            result["bmi_by_target"][str(label)] = {
                "count": int(len(values)),
                "min": safe_float(values.min()),
                "max": safe_float(values.max()),
                "mean": safe_float(values.mean()),
                "below_30": int((values < 30).sum()),
                "at_least_30": int((values >= 30).sum()),
            }
        if set(frame[TARGET].dropna().unique()).issubset({0, 1}):
            result["bmi30_disagreement_rows"] = int(
                ((bmi >= 30) != (frame[TARGET] == 1)).sum()
            )

    feature_cols = [column for column in frame.columns if column != TARGET]
    feature_keys = frame[feature_cols].astype("string").fillna("<NA>")
    group_labels = frame.assign(
        _feature_key=feature_keys.agg("|".join, axis=1)
    ).groupby("_feature_key")[TARGET].nunique()
    result["feature_groups_with_conflicting_labels"] = int((group_labels > 1).sum())
    result["unique_feature_groups"] = int(group_labels.size)
    return result


def create_figures(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    if TARGET in frame.columns:
        counts = frame[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Target class distribution")
        ax.set_xlabel("Obesity label (meaning not yet verified)")
        ax.set_ylabel("Number of records")
        fig.tight_layout()
        path = output_dir / "target_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(str(path.relative_to(ROOT)))

    if TARGET in frame.columns and "BMI" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        frame.boxplot(column="BMI", by=TARGET, ax=ax)
        ax.set_title("BMI distribution by target label")
        ax.set_xlabel("Obesity label (meaning not yet verified)")
        ax.set_ylabel("BMI")
        fig.suptitle("")
        fig.tight_layout()
        path = output_dir / "bmi_by_target.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(str(path.relative_to(ROOT)))

    return created


def build_markdown(audits: list[dict[str, Any]], final: dict[str, Any] | None,
                   figures: list[str]) -> str:
    lines = [
        "# Data Audit Report",
        "",
        "> Generated by `python scripts/audit_data.py`. Source files are read-only.",
        "",
        "## Executive summary",
        "",
    ]
    if final:
        lines.extend([
            f"- Final dataset: **{final['rows']:,} rows**, "
            f"**{len(final['columns'])} columns**.",
            f"- Missing cells: **{final['missing_cells']:,}**.",
            f"- Exact duplicate rows: **{final['duplicate_full_rows']:,}**.",
            f"- Target counts: `{json.dumps(final.get('class_counts', {}), ensure_ascii=False)}`.",
            f"- Feature combinations with conflicting target labels: "
            f"**{final.get('feature_groups_with_conflicting_labels', 0)}**.",
            "",
        ])

    lines.extend([
        "## File inventory",
        "",
        "| File | Status | Bytes | Rows | Exact duplicate rows |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in audits:
        lines.append(
            f"| `{item['path']}` | {item['status']} | "
            f"{item.get('bytes', '')} | {item.get('rows', '')} | "
            f"{item.get('duplicate_rows', '')} |"
        )

    lines.extend(["", "## Detailed findings", ""])
    for item in audits:
        if item["status"] != "readable":
            continue
        lines.extend([
            f"### `{item['path']}`",
            "",
            f"- Shape: {item.get('rows', 0):,} rows × "
            f"{len(item.get('columns', []))} columns.",
            f"- Columns: {', '.join(f'`{c}`' for c in item.get('columns', []))}.",
            f"- Missing values: {sum(item.get('missing_by_column', {}).values()):,}.",
            f"- Exact duplicate rows: {item.get('duplicate_rows', 0):,}.",
        ])
        if "target_counts" in item:
            lines.append(
                f"- Target distribution: `{json.dumps(item['target_counts'], ensure_ascii=False)}`."
            )
        if "bmi_consistency" in item:
            b = item["bmi_consistency"]
            lines.append(
                "- BMI formula check (Height in metres, Weight in kg): "
                f"mean absolute error={b['mean_absolute_error']:.5f}; "
                f"median={b['median_absolute_error']:.5f}; "
                f"p95={b['p95_absolute_error']:.5f}; "
                f"max={b['max_absolute_error']:.5f}; "
                f"rows with error > 0.02={b['count_abs_error_gt_0_02']:,}."
            )
        lines.append("")

    if final:
        lines.extend([
            "### Target and BMI investigation",
            "",
            "The numeric target values are observed data values only. Their semantic mapping "
            "to 'obese'/'not obese' has not been verified from a data dictionary or primary source.",
            "",
        ])
        for label, stats in final.get("bmi_by_target", {}).items():
            lines.append(
                f"- Label {label}: n={stats['count']:,}, BMI range "
                f"{stats['min']}–{stats['max']}, mean={stats['mean']:.3f}; "
                f"BMI < 30: {stats['below_30']:,}, BMI ≥ 30: {stats['at_least_30']:,}."
            )
        lines.extend([
            "",
            f"- Records that disagree with a simple BMI ≥ 30 vs label 1 rule: "
            f"**{final.get('bmi30_disagreement_rows', 'not calculated'):,}**.",
            "- This disagreement means the current binary target is not a clean deterministic "
            "BMI ≥ 30 label. It does not by itself prove or disprove other forms of leakage.",
            "",
            f"- Distinct feature vectors: {final.get('unique_feature_groups', 0):,}.",
            f"- Feature vectors occurring with more than one target value: "
            f"{final.get('feature_groups_with_conflicting_labels', 0):,}.",
            "",
        ])

    lines.extend([
        "## EDA figures",
        "",
    ])
    lines.extend([f"- `{figure}`" for figure in figures] or [
        "- No figures generated. Install matplotlib and rerun the audit."
    ])
    lines.extend([
        "",
        "## Data quality decisions",
        "",
        "1. Do not overwrite or resample files under `dataset/` during auditing.",
        "2. Treat `obesity_prediction_dataset_final.csv` as a processed/balanced candidate, "
        "not as raw data.",
        "3. Recover the pre-integration dataset and the pre-balancing dataset before training "
        "a model intended to estimate performance on a natural population.",
        "4. Recover the original label mapping and the rule used to derive `Obesity`.",
        "5. Verify units and BMI generation. Investigate records with substantial BMI formula "
        "discrepancies instead of silently recalculating them.",
        "6. Preserve a natural-distribution holdout set. Any resampling must happen only "
        "inside training folds.",
        "",
        "## Files that should be recovered",
        "",
        "- The non-empty original `obesity_prediction.csv` referenced by "
        "`scripts/config.py`.",
        "- The `obesity_with_bmi.csv` referenced by root-level `pre-processing.py`.",
        "- The integrated dataset before duplicate removal / before undersampling "
        "(a project note found online mentions 18,891 rows before balancing; this is "
        "a lead to verify, not a substitute for the actual file).",
        "- Data dictionary or source documentation defining the exact binary target mapping.",
        "",
        "## External references and provenance leads",
        "",
        "- UCI candidate dataset: [Estimation of Obesity Levels Based on Eating Habits and "
        "Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition), "
        "DOI [10.24432/C5H31Z](https://doi.org/10.24432/C5H31Z), CC BY 4.0. "
        "It contains 2,111 records and seven obesity-level classes, so it is not a direct "
        "match for the repository's 17,698-row binary table. Confirm lineage before claiming it.",
        "- The UCI paper reports that its own 2,111-row dataset includes 77% synthetic records "
        "generated using Weka/SMOTE and 23% collected from users. This provenance applies to "
        "that UCI dataset, not automatically to the repository files.",
        "- A secondary project report describes integrating two obesity datasets, removing "
        "duplicates to 18,891 records, then balancing the binary target to 17,698 records. "
        "Use this only as a provenance lead; obtain the original data and methodology.",
        "",
        "## Conclusion",
        "",
        "**Phase 1 status: BLOCKED for model training until provenance and target definition "
        "are verified.** The current files are sufficient for a preliminary technical audit, "
        "but not sufficient to support a defensible claim of clinical diagnosis or reliable "
        "population-level performance.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    audits = [inspect_csv(ROOT / relative) for relative in CANDIDATE_FILES]
    final_path = ROOT / "dataset/obesity_prediction_dataset_final.csv"
    final: dict[str, Any] | None = None
    figures: list[str] = []
    if final_path.exists() and final_path.stat().st_size > 0:
        final = analyze_final_dataset(final_path)
        final_frame = pd.read_csv(final_path, encoding="utf-8-sig")
        figures = create_figures(final_frame, reports_dir / "figures")

    report = build_markdown(audits, final, figures)
    (reports_dir / "data_audit.md").write_text(report, encoding="utf-8")
    payload = {"files": audits, "final_dataset_analysis": final, "figures": figures}
    (reports_dir / "data_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[OK] Wrote {reports_dir / 'data_audit.md'}")
    print(f"[OK] Wrote {reports_dir / 'data_audit.json'}")
    for figure in figures:
        print(f"[OK] Wrote {figure}")


if __name__ == "__main__":
    main()
