CAUTI Metabolomics Analysis Pipeline

# Overview

This repository contains a consolidated, reproducible analysis pipeline for comprehensive metabolomics analysis of catheter-associated urinary tract infection (CAUTI) bacterial strains across different media (AUM/ISO), cell types (biofilm/planktonic/supernatant), and growth conditions.

The pipeline performs:
- Automated medium detection (AUM/ISO)
- Control addition and normalization
- One-way ANOVA and Tukey HSD post-hoc tests
- Group-level statistical comparisons
- Fold change calculations (log₂)
- Generation of publication-ready heatmaps and statistical summaries
- Comprehensive Excel output with all results

# Requirements

- Python 3.8 or higher
- Required packages:
  - pandas >= 1.3.0
  - numpy >= 1.21.0
  - scipy >= 1.7.0
  - statsmodels >= 0.13.0
  - matplotlib >= 3.4.0
  - seaborn >= 0.11.0
  - openpyxl >= 3.0.0
  - tqdm >= 4.60.0
  - adjustText >= 0.7.3

# Installation

1. Clone the Repository
```bash
git clone https://github.com/yourusername/CAUTI-metabolomics.git
cd CAUTI-metabolomics
```

2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl tqdm adjustText
```

# Data Format Requirements

Your Excel input file should have the following structure:

Sheet Name: `Sheet1`

| Sample | Strain | Metabolite1 | Metabolite2 | ... | MetaboliteN |
|--------|--------|-------------|-------------|-----|-------------|
| AUM20_001 | Tme12 | 1234.5 | 5678.9 | ... | 123.45 |
| AUM20_002 | Tmp04 | 2345.6 | 6789.1 | ... | 234.56 |
| ISO20_001 | K12 | 3456.7 | 7890.2 | ... | 345.67 |
| ISO20_002 | PA01 | 4567.8 | 8901.3 | ... | 456.78 |

# Column Requirements:
- Sample: Must contain "AUM20" or "ISO20" to identify medium type
- Strain: Strain identifier (e.g., Tme12, Tmp04, K12, PA01, Pair412, etc.)
- Metabolites: All remaining columns treated as metabolite data (each column represents a different metabolite) 

# Important Notes:
- Strain names may contain "Mix" (automatically converted to "Pair")
- Missing values (NaN) are handled properly
- Metabolite values should be positive (log-transformed internally if needed)

# Configuration

Edit the `CONFIG` dictionary in `CAUTI_metabolomics_analysis.py` to customize your analysis:

```python
CONFIG = {
    'input_file': 'Analysis_data.xlsx',        # Path to your Excel file
    'medium_type': 'ISO',                      # 'ISO' or 'AUM' (auto-detected)
    'cell_type': 'supernatant',                # 'biofilm', 'planktonic', or 'supernatant'
    'control_strategy': 'unified',             # 'unified' or 'respective'
    'output_directory': './CAUTI_results',     # Output folder location
    'metadata_file': 'strain_metadata.csv',    # Optional strain descriptions
    'transformation_method': 'log',            # Data transformation method
    'fc_threshold': 2,                         # Fold change threshold
    'pval_threshold': 0.05,                    # P-value significance threshold
    'figsize_heatmap': (10, 12),               # Heatmap dimensions
    'dpi_output': 300,                         # Figure resolution (DPI)
}
```

# Configuration Options

- input_file: Path to your Excel data file (must exist)
- medium_type: Medium type - 'ISO' or 'AUM' (auto-detected from sample names)
- cell_type: Biological context - 'biofilm', 'planktonic', or 'supernatant'
- control_strategy:
  - `'unified'`: Control = average of K12 and PA01
  - `'respective'`: Control = group-specific (K12 for E. coli, PA01 for P. aeruginosa, etc.)
- output_directory: Where results will be saved
- transformation_method: Data normalization before analysis
- fc_threshold: Log2 fold change cutoff for significance
- **pval_threshold**: P-value threshold for statistical significance

# Usage

# Quick Start

```bash
# 1. Place your Excel file in the same directory
# 2. Update CONFIG['input_file'] with your filename
# 3. Run the analysis
python CAUTI_metabolomics_analysis.py
```

# Full Workflow Example

```python
Step 1: Create project directory
mkdir CAUTI_analysis
cd CAUTI_analysis

Step 2: Copy script and data
cp CAUTI_metabolomics_analysis.py .
cp Analysis_sup_ISO.xlsx .

Step 3: Edit CONFIG in script
Change: 'input_file': 'Analysis_sup_ISO.xlsx'
Change: 'cell_type': 'supernatant'
Change: 'control_strategy': 'unified'

Step 4: Run analysis
python CAUTI_metabolomics_analysis.py
```

# Running Multiple Analyses

For analyzing different cell types or media, create multiple configuration files:

```bash
Analyze all combinations
python CAUTI_metabolomics_analysis.py  ISO supernatant (default)

Then modify CONFIG and run again for:
- AUM supernatant
- ISO biofilm
- ISO planktonic
- AUM biofilm
- AUM planktonic
```

# Output Description

The pipeline generates the following outputs in timestamped folders:

# Output Directory Structure
```
CAUTI_results/
└── CAUTI_results_ISO_supernatant_2025-12-09_14-30-45/
    ├── Statistical_results_ISO_supernatant.xlsx    # Main results file
    ├── Group_analysis_ISO_supernatant.csv          # Group comparisons
    ├── Group_heatmap_resp_ISO_supernatant.png      # Respective controls heatmap
    ├── Group_heatmap_unified_ISO_supernatant.png   # Unified control heatmap
    └── CAUTI_metabolomics_analysis_copy.py         # Copy of analysis script
```

# Excel Output File Structure

Statistical_results_ISO_supernatant.xlsx contains multiple sheets:

| Sheet Name | Contents | Description |
|-----------|----------|-------------|
| ANOVA | F-statistic, p-values | One-way ANOVA results for all metabolites |
| Tukey HSD | Pairwise comparisons | Post-hoc Tukey HSD test results |
| FoldChange | Log2 fold changes | Fold changes for each strain vs. control |
| Group Analysis | Group-level stats | Clinical group vs. control comparisons |

# CSV Output Files

Group_analysis_ISO_supernatant.csv contains:
- Metabolite name
- Clinical group (E. coli, P. aeruginosa, Co-cultures)
- Control strain used
- t-statistic
- p-value
- Significance (True/False)

# PNG Output Files

Group_heatmap_*.png files show:
- Rows: Metabolites (clustered by similarity)
- Columns: Clinical groups
- Values: Log2 fold changes
- Annotations: Significance markers (*, **, ***)
  - `*`: p < 0.05
  - `**`: p < 0.01
  - `***`: p < 0.001

# Strain Information

# Strain Groups

The analysis organizes strains into the following groups:

| Group | Strains | Description |
|-------|---------|-------------|
| E. coli Clinical | Tme12, Tme13, Tme14, Tme15 | Clinical CAUTI isolates |
| P. aeruginosa Clinical | Tmp04, Tmp05, Tmp06, Tmp07 | Clinical CAUTI isolates |
| Co-cultures | Pair412, Pair513, Pair614, Pair715 | Mixed clinical isolates |
| E. coli Control | K12 | Reference strain K-12 substr. MG1655 |
| P. aeruginosa Control | PA01 | Reference strain PAO1 |
| Co-culture Control | PairEP | Reference co-culture |

For complete strain descriptions, see `strain_metadata.csv`.

# Strain Naming Convention

- Tme: Escherichia coli clinical isolate
- Tmp: Pseudomonas aeruginosa clinical isolate
- Pair: Co-culture of two clinical isolates (Pair + isolate numbers, e.g., Pair412 = Tme12 + Tmp04)
- K12: E. coli K-12 reference strain
- PA01: P. aeruginosa PAO1 reference strain
- PairEP: Co-culture of K12 + PA01 (control pair)

# Statistical Methods

# Tests Performed

1. One-way ANOVA
   - Compares metabolite abundance across all strains
   - Tests null hypothesis: all strain means are equal
   - Output: F-statistic, p-value

2. Tukey HSD (Honestly Significant Difference)
   - Post-hoc test following significant ANOVA
   - Pairwise comparisons between all strains
   - Controls for multiple comparison error (family-wise error rate)
   - Output: mean differences, confidence intervals, adjusted p-values

3. Independent t-tests (Group Level)
   - Compares clinical groups to their respective controls
   - E. coli clinical strains vs. K12
   - P. aeruginosa clinical strains vs. PA01
   - Co-cultures vs. PairEP
   - Output: t-statistic, p-value, significance

4. Benjamini-Hochberg FDR Correction
   - Controls false discovery rate across multiple tests
   - More powerful than Bonferroni for large test numbers
   - Applied to all p-values
   - Default threshold: adjusted p < 0.05

# Fold Change Calculation

```
Log₂FC = log₂(Mean_Group / Mean_Control)

Interpretation:
- Positive values: group has higher abundance
- Negative values: group has lower abundance
- 0: No change
- ±1: 2-fold change
- ±2: 4-fold change
```

# Troubleshooting

# Issue: "Error reading Excel file"
Solution:
- Ensure file exists and path is correct
- File should have sheet named "Sheet1"
- Check Excel file is not corrupted (try opening in Excel)

# Issue: "Cannot determine medium type"
Solution:
- Sample names must contain "AUM20" or "ISO20"
- Check capitalization (must be uppercase)
- Verify sample names in your data

# Issue: Empty heatmap or missing metabolites
Solution:
- Check for missing/NaN values in data
- Ensure metabolite columns contain numeric values
- Verify strain names match expected values (see SAMPLE_ORDER)

# Issue: "ImportError: No module named 'adjustText'"
Solution:
```bash
pip install adjustText
```

# Issue: Memory error with large datasets
Solution:
- Close other applications
- Use a 64-bit Python interpreter
- Consider splitting analysis into multiple runs

# Data and Code Availability Statement

Include this statement in your paper:

> "The code and data supporting this study are available at [https://github.com/sokoljator/CAUTI-metabolomics/CAUTI_metabolomics_analysis.py]. 
> Strain descriptions and metadata are provided in Supplementary Table S1. 
> All statistical analyses were performed using the CAUTI Metabolomics Analysis 
> Pipeline (v1.0), implemented in Python 3.8+ with dependencies listed in 
> requirements.txt. The pipeline employs ANOVA, Tukey HSD post-hoc tests, and 
> Benjamini-Hochberg FDR correction for multiple comparison adjustment (α = 0.05)."

# Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{CAUTI_metabolomics_2025,
  author = {Sokol, Dmytro},
  title = {CAUTI Metabolomics Analysis Pipeline},
  year = {2025},
  url = {https://github.com/sokoljator/CAUTI-metabolomics/CAUTI_metabolomics_analysis.py},
  version = {1.0}
}
```

# Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Contact & Support

For questions, issues, or suggestions:

- GitHub Issues: [Report bugs or request features](https://github.com/sokoljator/CAUTI-metabolomics/issues)
- Email: sokoldima94@gmail.com
- Documentation: see README.md and inline code comments

# Acknowledgments

- Statistical methods based on standard bioinformatics practices
- Visualization approaches adapted from Seaborn and Matplotlib best practices
- Developed for CAUTI metabolomics research at Chemistry department, Umeå University, Umeå, Sweden

# Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-09 | Initial consolidated release from 6 specialized scripts, which were created on 2025-03-21 by Dmytro Sokol|

---

Last Updated: 9th December 2025  
Maintained by: Dmytro Sokol  
Status: Active Development
