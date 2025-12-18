![ChatGPT Image 18 груд  2025 р , 13_15_56](https://github.com/user-attachments/assets/3f411314-3c36-4b00-8129-be0e1a02a98e)


# Cowtea v1.0.0 - CAUTI Metabolomics Analysis Pipeline

## 🚀 Overview

**Cowtea** (CAUTI Omics Workflow for Targeted Extracellular Analysis) is a production-ready, publication-grade metabolomics analysis pipeline for catheter-associated urinary tract infection (CAUTI) research. Version 1.0.0 represents a complete rewrite consolidating 6 specialized scripts into a single, unified, adaptive statistical framework.

## ✨ Key Features (NEW in v1.0.0)

✅ **Adaptive Statistical Testing**: Automatically selects ANOVA/Welch's ANOVA/Kruskal-Wallis based on normality + variance assumptions  
✅ **Automatic Data Transformation**: Evaluates 6 methods (log, sqrt, boxcox, cube, glog, inverse) and selects optimal  
✅ **Multi-Level Analysis**: Strain → Group → Hierarchical significance testing  
✅ **Comprehensive Visualization Suite**: 12+ plot types (volcano, heatmap, bubble, raincloud, bar, violin+box+points, gradient intervals)  
✅ **Metabolite Pathway Classification**: 9 biological pathways with color-coded visualization  
✅ **Hierarchical Group Analysis**: E.coli vs K12, P.aeruginosa vs PA01, Co-cultures vs PairEP (FDR-corrected)  
✅ **Publication-Ready Outputs**: 300 DPI figures + fully formatted Excel results  
✅ **Between-Group Comparisons**: Paired and unpaired tests for clinical group relationships  
✅ **Global Pairwise Post-Hoc**: Tukey HSD, Games-Howell, Dunn's (strain-level)  
✅ **Comprehensive Methodology Documentation**: 6-phase workflow with statistical rationale  

## 🔬 Statistical Workflow (v1.0.0)

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Data Prep → Control Addition → Medium Detection        │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 2: Transformation Optimization (6 methods evaluated)      │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 3: Assumption Testing → Three-Path Decision Tree          │
│          Path A: Normal + Equal Var    → ANOVA                  │
│          Path B: Normal + Unequal Var  → Welch's ANOVA          │
│          Path C: Non-normal            → Kruskal-Wallis         │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 4: Global Statistical Analysis (adaptive test per met)    │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 5: Multi-Level Post-Hoc Analysis                          │
│          Level 1: Global Pairwise (all strain combinations)     │
│          Level 2: Group-Level (3 biological groups)             │
│          Correction: Benjamini-Hochberg FDR                     │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 6: Multi-Criteria Filtering + Visualization               │
│          (Global sig OR Fold-change large) AND Post-hoc          │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

**Python Version:** 3.8 or higher

**Required Packages:**
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl tqdm scikit-posthocs ptitprince adjustText
```

**Quick Install:**
```bash
pip install -r requirements.txt
```

## 📁 Input Data Format

**Excel File Requirements:**

Sheet name: `Sheet1`

Structure:
```
| Sample    | Strain  | Alanine | Glucose | Pyruvate | [... more metabolites] |
|-----------|---------|---------|---------|----------|----------------------|
| AUM20_001 | Tme12   | 1234.5  | 5678.9  | 890.1    | ...                   |
| AUM20_002 | Tmp04   | 2345.6  | 6789.0  | 901.2    | ...                   |
| ISO20_001 | K12     | 3456.7  | 7890.1  | 912.3    | ...                   |
| ISO20_002 | PA01    | 4567.8  | 8901.2  | 923.4    | ...                   |
```

**Required Columns:**
- `Sample`: Must contain `AUM20` or `ISO20` to identify medium type
- `Strain`: Strain identifier (see "Strain Information" section)
- All other columns treated as metabolite data (numeric values)

**Important Notes:**
- Strain names with "Mix" automatically converted to "Pair"
- Missing values (NaN) handled gracefully
- Metabolite values should be positive (log-transformed internally)
- At least 3 replicates recommended per strain

## 🦠 Strain Information

**Strain Groups (Auto-detected):**

| Category | Strains | Type | Control |
|----------|---------|------|---------|
| E. coli Clinical | Tme12, Tme13, Tme14, Tme15 | CAUTI isolates | K12 |
| P. aeruginosa Clinical | Tmp04, Tmp05, Tmp06, Tmp07 | CAUTI isolates | PA01 |
| Co-cultures | Pair412, Pair513, Pair614, Pair715 | Mixed pairs | PairEP |
| **Reference Controls** | K12, PA01, PairEP | Standard strains | - |

**Naming Convention:**
- `Tme`: Escherichia coli clinical isolate
- `Tmp`: Pseudomonas aeruginosa clinical isolate
- `Pair[digits]`: Co-culture pair (e.g., Pair412 = Tme12 + Tmp04)
- `K12`: E. coli K-12 reference strain
- `PA01`: P. aeruginosa PAO1 reference strain
- `PairEP`: Co-culture reference (K12 + PA01)

## 🎯 Usage

**Basic Execution:**
```bash
python Cowtea.py "path/to/your/data.xlsx"
```

**Example:**
```bash
python Cowtea.py "Analysis_sup_ISO.xlsx"
```

**Output Location:**
```
./[MEDIUM]_sup_Analysis_YYYY-MM-DD_HH-MM-SS/
```

**Output Contents:**
```
ISO_sup_Analysis_2025-12-18_13-22-45/
├── Statistical_results_ISO_sup.xlsx          # Main results (12 sheets)
├── Heatmap_ISO_sup.png                       # Hierarchical clustered heatmap
├── Volcano_Plot_ISO_sup.png                  # Significance vs FC
├── Bubble_plot_resp_ISO_sup.png              # Group comparisons (3 subplots)
├── Bubble_plot_unified_ISO_sup.png           # Alternative control reference
├── Group_heatmap_resp_ISO_sup.png            # Group-level fold changes
├── Group_heatmap_unified_ISO_sup.png         # Unified control heatmap
├── Transformation_Comparison_ISO.xlsx        # Transformation evaluation
├── Between_group_bubble_plots_ISO.png        # Between-group comparisons
├── [Metabolite]_hierarchical_groups_ISO_sup.png  (×N significant metabolites)
├── [Metabolite]_violin_box_points_ISO_sup.png    (×N significant metabolites)
└── Cowtea.py                                 # Self-documented copy of analysis script
```

## 📊 Excel Output Sheets

**Statistical_results_ISO_sup.xlsx** contains 12 analysis sheets:

| Sheet Name | Contents | Description |
|-----------|----------|-------------|
| **ANOVA_Results** | F/H/Welch statistic, p-values | Global significance per metabolite |
| **Assumption_Results** | Normality, variance equality tests | Assumption testing details |
| **PostHoc_Results** | All pairwise strain comparisons | Tukey HSD / Dunn's / Mann-Whitney results |
| **Group_Analysis** | Group vs. control comparisons | Hierarchical group statistics (FDR) |
| **Group_PostHoc** | Between-group pair comparisons | E.coli vs P.aero, etc. |
| **FoldChange** | Log₂(FC) matrix | Log₂ fold-changes for all strains |
| **Significant_Metabolites** | Final filtered list | Metabolites meeting multi-criteria filter |
| **Metabolite_Pathway_Key** | Pathway classification | Biological function annotation |
| **Strain_Metadata** | Strain information | Sample counts and strain types |
| **Summary_Statistics** | Mean/SD descriptive stats | Strain-level basic statistics |

## 🔬 Statistical Methods

### Global Statistical Tests (Phase 4)

**Test Selection Criteria:**

| Test | When Used | Test Statistic | Post-hoc Test |
|------|-----------|----------------|---------------|
| **ANOVA** | Normal + Equal Variance (both p > 0.05) | F-statistic | Tukey HSD |
| **Welch's ANOVA** | Normal but Unequal Variance | F-statistic (adjusted) | Games-Howell |
| **Kruskal-Wallis** | Non-normal (p < 0.05) | H-statistic (χ²-distributed) | Dunn's Test |

### Post-Hoc Testing (Phase 5)

**Level 1: Global Pairwise Comparisons**
```
All C(k,2) strain pairs tested
Method: Strain-specific post-hoc
Correction: Benjamini-Hochberg FDR across all comparisons
```

**Level 2: Group-Level Comparisons**
```
E.coli strains vs K12
P.aeruginosa strains vs PA01
Co-cultures vs PairEP

Method: Mann-Whitney U (robust, non-parametric)
Correction: Benjamini-Hochberg FDR (only 3 tests per metabolite)
```

**Level 3: Between-Group Comparisons**
```
E.coli vs P.aeruginosa (unpaired)
E.coli vs Co-cultures (paired by design)
P.aeruginosa vs Co-cultures (paired by design)

Method: Paired t-test / Wilcoxon (if paired design detected)
        Independent t-test / Mann-Whitney U (if unpaired)
Correction: Benjamini-Hochberg FDR
```

### Multiple Testing Correction

**Benjamini-Hochberg FDR:**
```
FDR = E[# False Positives / # Total Discoveries]

Target: FDR ≤ 0.05 (≤5% of "significant" findings expected false)

Advantages over Bonferroni:
  - More powerful
  - Appropriate for hypothesis-generating studies
  - Standard in metabolomics literature
```

## 🎨 Visualization Suite

### 1. **Volcano Plot** (Volcano_Plot_ISO_sup.png)
- X-axis: Log₂(Fold-Change)
- Y-axis: -Log₁₀(p-value)
- Colors: By metabolite pathway
- Markers: Final significant metabolites highlighted

### 2. **Heatmap** (Heatmap_ISO_sup.png)
- Hierarchical clustering (rows = metabolites, columns = strains)
- Color scale: Blue (low) → White (medium) → Red (high)
- Row annotations: Metabolite pathways (color-coded)
- Column annotations: Strain identifiers

### 3. **Group Comparison Bubble Plots** (Bubble_plot_*.png)
- X-axis: Log₂(FC) between groups
- Y-axis: -Log₁₀(p-value)
- Bubble size: Magnitude of fold-change
- Separate panels: Each biological group vs. control
- Color-coded: By group identity

### 4. **Between-Group Bubble Plots** (Between_group_bubble_plots_ISO.png)
- Three subplots: E.coli vs P.aero, E.coli vs Co-cultures, P.aero vs Co-cultures
- Numbered metabolites: Easy reference to top findings
- Quadrant shading: Indicates which group elevated

### 5. **Hierarchical Bar Plots** ([Metabolite]_hierarchical_groups_ISO_sup.png)
- Group-level bars with strain points overlay
- Error bars: SEM across strains
- Significance indicators: *, **, *** between comparisons
- Shows: Both strain variation and group patterns

### 6. **Violin + Box + Points** ([Metabolite]_violin_box_points_ISO_sup.png)
- Violin: Distribution shape
- Box: Quartiles (Q1-median-Q3)
- Points: Individual observations
- Significance marks: Between strain/group comparisons

### 7. **Gradient Interval Plots** ([Metabolite]_gradient_intervals_ISO_sup.png)
- Horizontal intervals showing uncertainty at 3 levels (68%, 95%, 99.7%)
- Darker regions: Higher confidence
- Black dot: Group mean
- Gray dots: Strain means (for clinical groups)

### 8. **Transformation Comparison** (Transformation_Comparison_ISO.xlsx)
- Before/after transformation distributions
- Skewness and kurtosis metrics
- Shapiro-Wilk pass rates
- Selected method highlighted

## 📈 Metabolite Pathway Classification (9 Categories)

**Cowtea automatically classifies metabolites into biological pathways:**

| Pathway | Count | Examples | Color |
|---------|-------|----------|-------|
| Amino Acids | 20 | Alanine, Valine, Leucine, Tryptophan | Blue |
| Carbohydrates | 13 | Glucose, Fructose, Trehalose, Xylose | Green |
| Organic Acids | 15 | Lactic acid, Pyruvate, Citrate, Acetate | Orange |
| Lipids | 9 | Oleic acid, Palmitic acid, Cholesterol | Purple |
| Nucleotides/Nucleosides | 15 | Adenosine, ATP, GTP, Uracil | Red |
| Vitamins/Cofactors | 18 | Vitamin B12, FAD, NAD, Biotin | Brown |
| Polyamines | 5 | Putrescine, Spermidine, Spermine | Pink |
| Phenolic Compounds | 8 | Phenol, Benzoic acid, Ferulic acid | Gray |
| Amino Acid Metabolism | 1 | Urea | Cyan |

## 🔍 Multi-Criteria Metabolite Selection

**Final significant metabolite filtering:**

```
Step 1: Global Statistical Test
        p-value < 0.05 (ANOVA/Welch/Kruskal-Wallis)

Step 2: Fold-Change Magnitude
        |Log₂(FC)| > 0.75 in ≥33% of non-control strains

Step 3: Post-Hoc Significance
        At least one pairwise comparison significant (raw p < 0.05)

Combination Logic:
  IF (Global significant) AND (Fold-change large):
    Metabolite selected (both filters)
  ELSE IF only one filter passed:
    Metabolite selected (either criterion met)
  
Final Count: Typically 20-50 of 45-100 analyzed metabolites
```

## 🛠️ Advanced Configuration

**Edit Global Constants** (top of Cowtea.py):

```python
# Strain Definitions
controls = ['K12', 'PA01', 'PairEP']
e_mono = ['Tme12', 'Tme13', 'Tme14', 'Tme15']
p_mono = ['Tmp04', 'Tmp05', 'Tmp06', 'Tmp07']
pairs = ['Pair412', 'Pair513', 'Pair614', 'Pair715']

# Statistical Thresholds
ALPHA = 0.05                    # Significance threshold
FC_THRESHOLD = 0.75             # Log₂ fold-change cutoff
FOLD_CHANGE_PERCENTAGE = 0.33   # % of strains needing FC
```

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Cannot determine medium type" | Sample names missing AUM20/ISO20 | Verify sample naming convention |
| "Shapiro warning: zero range" | Metabolite has identical values | Normal - handled automatically |
| Empty heatmap | No metabolites pass filtering | Check fold-change and p-value thresholds |
| Excel export error | Column name mismatch | Verify p-value column names (handled in v1.0.0) |
| Plot generation failure | Insufficient disk space | Free space and retry |
| Missing visualizations | Transformation failed | Check data for extreme outliers |

## 📚 Documentation

**Complete methodology documentation included:**
- `Detailed_Methodology_Documentation.md` - 6-phase framework details
- Inline code comments - Function-level documentation
- Sheet explanations in Excel - Statistical test definitions

## 📊 Expected Output Volumes

| File Type | Count | Total Size | Notes |
|-----------|-------|-----------|-------|
| Excel sheets | 12 | 0.5-5 MB | Results vary by metabolite count |
| Heatmaps | 2-3 | 2-8 MB | Includes clustered + pathway-colored versions |
| Bar plots | 20-50 | 20-100 MB | One per significant metabolite |
| Volcano plots | 1 | 1-3 MB | Global significance visualization |
| Bubble plots | 6 | 6-12 MB | Group and between-group comparisons |
| **Total** | ~100 files | ~50-200 MB | Depends on dataset size |

## 🔗 Citation

If you use Cowtea v1.0.0 in your research, please cite:

```bibtex
@software{Cowtea_v1_0_0_2025,
  author = {Sokol, Dmytro},
  title = {Cowtea v1.0.0: CAUTI Metabolomics Analysis Pipeline},
  year = {2025},
  url = {https://github.com/sokoljator/Cowtea},
  version = {1.0.0},
  note = {Adaptive ANOVA/Welch/Kruskal-Wallis with 6-phase statistical framework}
}
```

## 📝 Version History

| Version | Date | Key Changes | Files Consolidated |
|---------|------|-------------|-------------------|
| **1.0.0** | 2025-12-18 | Complete rewrite with adaptive stats, 12 visualizations, hierarchical analysis, pathway classification | 6 specialized scripts → 1 unified pipeline |
| 0.x.x | 2025-03-21 | Original individual analysis scripts for each medium/cell-type combination | CAUTI_code_[AUM/ISO]_[bio/pln/sup].py |

## 📞 Support & Contact

**Developer:** Dmytro Sokol  
**Affiliation:** Chemistry Department, Umeå University, Sweden  
**Email:** sokoldima94@gmail.com  
**Status:** Production Ready (v1.0.0)

---

## 📋 Checklist Before Running Analysis

- ✅ Excel file has `Sheet1` with correct structure
- ✅ Sample names contain `AUM20` or `ISO20`
- ✅ All required packages installed (`pip install -r requirements.txt`)
- ✅ Sufficient disk space (~100-200 MB for outputs)
- ✅ At least 3 replicates per strain
- ✅ Metabolite values are numeric (not text)
- ✅ No special characters in strain names (use underscores if needed)

---

**Last Updated:** December 18, 2025  
**Documentation Version:** 1.0.0  
**Status:** Production Release
