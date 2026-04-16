# COWTEA2 (Standalone Project)

`COWTEA2` is a **new, separate project** in this repository for CAUTI metabolomics.
It is connected to Cowtea **by concept only** (same biological domain), and does **not** import or execute `Cowtea_v1.0.0.py`.

## What COWTEA2 does

- Loads one Excel dataset (`.xlsx`) with required columns: `Sample`, `Strain`, and numeric metabolite columns.
- Builds strain-level descriptive summaries (mean, std, median, n).
- Performs Kruskal–Wallis screening across strains per metabolite.
- Applies Benjamini–Hochberg FDR correction.
- Exports independent COWTEA2 outputs (Excel + CSV + metadata JSON).

## Usage

```bash
python COWTEA2/cowtea2_runner.py ./your_data.xlsx
```

Optional settings:

```bash
python COWTEA2/cowtea2_runner.py ./your_data.xlsx \
  --output-dir COWTEA2/output \
  --alpha 0.05 \
  --min-replicates 3
```

## Output structure

Each run creates a timestamped directory:

```text
COWTEA2/output/run_YYYYMMDDTHHMMSSZ/
├── cowtea2_results.xlsx
├── significant_metabolites.csv
└── run_metadata.json
```

## Notes

- This project is intentionally standalone.
- It can evolve independently as `COWTEA2` without changing legacy Cowtea code.
