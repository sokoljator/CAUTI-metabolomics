#!/usr/bin/env python3
"""COWTEA2 standalone metabolomics project.

This is an independent analysis program in the same conceptual domain as Cowtea
(CAUTI metabolomics), but it does not call or import Cowtea_v1.0.0.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Any


REQUIRED_COLUMNS = ["Sample", "Strain"]


@dataclass
class Config:
    input_file: Path
    output_dir: Path
    alpha: float
    min_replicates: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COWTEA2 standalone CAUTI metabolomics analyzer"
    )
    parser.add_argument("input", help="Path to input .xlsx file")
    parser.add_argument(
        "--output-dir",
        default="COWTEA2/output",
        help="Directory where results will be written",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR threshold for significance (default: 0.05)",
    )
    parser.add_argument(
        "--min-replicates",
        type=int,
        default=3,
        help="Minimum replicates per strain required for testing (default: 3)",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Any:
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Input must be an .xlsx file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    import pandas as pd

    df = pd.read_excel(path, sheet_name=0)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def metabolite_columns(df: Any) -> List[str]:
    import pandas as pd

    excluded = set(REQUIRED_COLUMNS)
    candidates = [column for column in df.columns if column not in excluded]
    numeric = [column for column in candidates if pd.api.types.is_numeric_dtype(df[column])]

    if not numeric:
        raise ValueError("No numeric metabolite columns detected.")

    return numeric


def summarize_by_strain(df: Any, metabolites: List[str]) -> Any:
    import pandas as pd

    grouped = df.groupby("Strain")[metabolites]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).add_suffix("_std")
    median_df = grouped.median().add_suffix("_median")
    n_df = grouped.count().add_suffix("_n")

    summary = pd.concat([mean_df, std_df, median_df, n_df], axis=1).reset_index()
    return summary


def kruskal_screen(df: Any, metabolites: List[str], min_replicates: int) -> Any:
    import numpy as np
    import pandas as pd
    from scipy.stats import kruskal
    from statsmodels.stats.multitest import multipletests

    rows = []
    strains = sorted(df["Strain"].dropna().unique())

    for metabolite in metabolites:
        groups = []
        usable_strains = []

        for strain in strains:
            values = df.loc[df["Strain"] == strain, metabolite].dropna().to_numpy()
            if len(values) >= min_replicates:
                groups.append(values)
                usable_strains.append(strain)

        if len(groups) < 2:
            rows.append(
                {
                    "metabolite": metabolite,
                    "test": "kruskal",
                    "n_groups": len(groups),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "Insufficient groups with required replicates",
                }
            )
            continue

        stat, p_value = kruskal(*groups)
        rows.append(
            {
                "metabolite": metabolite,
                "test": "kruskal",
                "n_groups": len(groups),
                "statistic": float(stat),
                "p_value": float(p_value),
                "note": ", ".join(usable_strains),
            }
        )

    results = pd.DataFrame(rows)
    valid = results["p_value"].notna()
    if valid.any():
        reject, p_adj, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results.loc[valid, "p_adj_bh"] = p_adj
        results.loc[valid, "significant"] = reject
    else:
        results["p_adj_bh"] = np.nan
        results["significant"] = False

    results["significant"] = results["significant"].fillna(False).astype(bool)
    return results.sort_values(["p_adj_bh", "p_value"], na_position="last")


def write_outputs(config: Config, summary: Any, tests: Any) -> Path:
    import pandas as pd

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    significant = tests.loc[tests["p_adj_bh"] <= config.alpha].copy()
    significant.to_csv(run_dir / "significant_metabolites.csv", index=False)

    with pd.ExcelWriter(run_dir / "cowtea2_results.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="strain_summary", index=False)
        tests.to_excel(writer, sheet_name="kruskal_screen", index=False)
        significant.to_excel(writer, sheet_name="significant", index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(config.input_file.resolve()),
        "alpha": config.alpha,
        "min_replicates": config.min_replicates,
        "output_dir": str(run_dir.resolve()),
        "n_significant": int(len(significant)),
    }
    pd.Series(metadata).to_json(run_dir / "run_metadata.json", indent=2)

    return run_dir


def main() -> int:
    args = parse_args()
    config = Config(
        input_file=Path(args.input),
        output_dir=Path(args.output_dir),
        alpha=args.alpha,
        min_replicates=args.min_replicates,
    )

    try:
        df = load_dataset(config.input_file)
        metabolites = metabolite_columns(df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    summary = summarize_by_strain(df, metabolites)
    tests = kruskal_screen(df, metabolites, min_replicates=config.min_replicates)
    run_dir = write_outputs(config, summary, tests)

    print(f"COWTEA2 analysis completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
