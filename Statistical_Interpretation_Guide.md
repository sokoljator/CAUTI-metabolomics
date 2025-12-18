# 📊 Cowtea v1.0.0 - Complete Statistical Results Interpretation & Custom Control Guide

## Executive Summary

This document is the **complete companion guide** to Cowtea v1.0.0's Excel output files. It explains:

1. **Every statistical result** in the 12 Excel sheets (what numbers mean, biological interpretation)
2. **How to identify key findings** and actionable patterns
3. **Custom control creation** for datasets without medium controls (K12/PA01/PairEP)
4. **Exact code modifications** with line-by-line examples

---

## PART I: UNDERSTANDING STATISTICAL RESULTS

---

## 1. ANOVA_Results Sheet – Global Statistical Testing

### 1.1 Purpose
This sheet tests **one hypothesis per metabolite**: "Does the abundance of this metabolite differ significantly across bacterial strains?"

### 1.2 Column-by-Column Explanation

| Column | Example | What It Means |
|--------|---------|---------------|
| **Metabolite** | `Alanine` | The compound being analyzed |
| **Test_Type** | `ANOVA` \| `Welch` \| `Kruskal-Wallis` | Which statistical test was used (see section 1.3) |
| **Statistic** | `8.45` (F-stat) | Test statistic value; larger = stronger evidence of difference |
| **df1** | `15` | Between-groups degrees of freedom = (number of strains – 1) |
| **df2** | `80` | Within-groups or error degrees of freedom |
| **p_value** | `1.2e-05` | Probability of observing this result if all strains were truly identical |
| **p_adj (optional)** | `0.0024` | p-value after correction for testing multiple metabolites (FDR) |
| **Significant** | `YES` \| `NO` | `YES` if p_value < 0.05 (or p_adj < 0.05) |

### 1.3 Understanding Test_Type

Three tests are automatically selected based on data characteristics:

#### **ANOVA** (One-Way Analysis of Variance)
- **Used when**: Data is approximately normal AND variances are equal across strains
- **Assumptions verified**: Shapiro-Wilk test p > 0.05 AND Levene's test p > 0.05
- **Advantages**: 
  - Most powerful (highest statistical power to detect real differences)
  - Parametric (uses full data distribution information)
  - Well-established, widely accepted
- **Interpretation**: F-statistic value
  - Larger F → stronger evidence that at least one strain's mean differs
  - Small F (close to 1) → all strains likely have similar means

**Example**:
```
Alanine: Test_Type = ANOVA, F = 8.45, p = 0.00001
→ Strong evidence (p < 0.00001) that mean alanine differs among strains
→ At least one strain produces significantly more or less alanine
```

#### **Welch's ANOVA**
- **Used when**: Data is approximately normal BUT variances are NOT equal across strains
- **Assumptions verified**: Shapiro-Wilk test p > 0.05 BUT Levene's test p < 0.05
- **Why needed**: Standard ANOVA assumes equal variances; violating this inflates Type I error (false positives)
- **Advantages**: 
  - Corrects for unequal variances automatically
  - Still parametric and fairly powerful
  - More conservative than ANOVA (less likely to give false positives)
- **Interpretation**: Similar to ANOVA, but adjusted for variance inequality

**Example**:
```
Glucose: Test_Type = Welch, F = 6.23, p = 0.0012
→ P. aeruginosa strains might be more variable than E. coli strains
→ Welch's ANOVA accounts for this, still detects global difference
```

#### **Kruskal-Wallis H Test**
- **Used when**: Data fails normality test (Shapiro-Wilk p < 0.05), regardless of variance equality
- **Why used**: 
  - Non-parametric (distribution-free)
  - Robust to outliers (works with ranks, not raw values)
  - Still powerful but more conservative than parametric tests
- **Advantages**: 
  - No assumptions about underlying distribution
  - Handles extreme values gracefully
  - Can detect median differences even if means are similar
- **Interpretation**: H-statistic (approximately χ² distributed)
  - Larger H → stronger evidence of differences
  - Tests **median/distribution differences**, not necessarily means

**Example**:
```
Phenol: Test_Type = Kruskal-Wallis, H = 7.89, p = 0.0048
→ Phenol distribution differs across strains
→ Cannot assume normal distribution, but still detect strain effect
```

### 1.4 Interpreting p-values

#### **p < 0.001** (Highly significant)
- Extremely strong evidence of difference
- **Biological meaning**: This metabolite is **highly strain-specific**
- **Action**: Priority for biomarker panels, mechanistic investigation
- **Red flag if**: All strain differences have p < 0.001 (check data quality)

#### **0.001 < p < 0.01** (Very significant)
- Strong evidence of difference
- **Biological meaning**: Clear strain discrimination based on this metabolite
- **Action**: Include in biomarker panels
- **Confidence**: High

#### **0.01 < p < 0.05** (Significant at α = 0.05)
- Moderate evidence of difference
- **Biological meaning**: Detectable strain effect
- **Action**: Include but with caveats; consider replication
- **Confidence**: Moderate

#### **p ≥ 0.05** (Not significant)
- No detectable difference across strains at chosen threshold
- **Biological meaning**: Either no true strain effect OR insufficient power to detect it
- **Action**: Flag as "strain-independent" metabolite; may still be important for absolute abundance
- **Note**: Non-significance doesn't mean "no difference exists", just not enough evidence to conclude with 95% confidence

### 1.5 Degrees of Freedom & Effect Size Context

**df1 = number of strains – 1**
```
If analyzing K12, PA01, PairEP, Tme12, Tme13, ... (16 total strains)
Then df1 = 16 – 1 = 15
```

**df2 = total observations – number of strains** (approximately)
```
If each strain has 3 replicates, and 16 strains → 48 total samples
Then df2 ≈ 48 – 16 = 32
```

**Larger df → more information → test is more trustworthy**

### 1.6 Biological Example: Reading ANOVA_Results

```
Metabolite | Test_Type | Stat | p_value | Significant | Interpretation
-----------|-----------|------|---------|-------------|-----------------------------------
Alanine    | ANOVA     | 12.3 | 2.3e-06 | YES         | All strains differ significantly in alanine
Glucose    | Welch     | 5.67 | 0.0089  | YES         | Glucose varies by strain; with unequal var
Phenol     | Kruskal   | 8.91 | 0.0031  | YES         | Phenol distribution differs; non-normal
Fructose   | ANOVA     | 1.23 | 0.2847  | NO          | No strain effect on fructose (housekeeping?)
```

**Biological story**:
- Alanine & glucose are great biomarkers for strain typing
- Phenol shows species-specific metabolism (E. coli vs P. aeruginosa differ)
- Fructose is metabolically stable across strains (not useful for differentiation, but may indicate basal energy metabolism)

---

## 2. Assumption_Results Sheet – Validating Statistical Tests

### 2.1 Purpose
This sheet documents **why** a particular test was chosen for each metabolite. It's a **transparency/validation sheet**.

### 2.2 Column-by-Column Explanation

| Column | Example | What It Means |
|--------|---------|---------------|
| **Metabolite** | `Alanine` | The compound being analyzed |
| **Shapiro_p** | `0.342` | p-value from Shapiro-Wilk normality test |
| **Normality_Pass** | `YES` | Is p > 0.05? (If yes, data is approximately normal) |
| **Levene_p** | `0.612` | p-value from Levene's test for variance equality |
| **EqualVar_Pass** | `YES` | Is p > 0.05? (If yes, variances are equal across strains) |
| **Test_Assigned** | `ANOVA` | Decision: which test was chosen based on above |

### 2.3 Understanding Normality Test (Shapiro-Wilk)

**What it tests**: "Are the residuals (or group-wise data) normally distributed?"

**Interpretation**:
- **Shapiro_p > 0.05** → Fail to reject normality → Data approximately normal ✓
- **Shapiro_p < 0.05** → Reject normality → Data significantly deviates from normal ✗

**Why it matters**:
- ANOVA and t-tests assume normality of residuals
- Non-normal data → parametric tests may give incorrect p-values
- Solution: Use Kruskal-Wallis (rank-based, non-parametric)

**Example**:
```
Alanine: Shapiro_p = 0.342 > 0.05 → Normal
Phenol:  Shapiro_p = 0.002 < 0.05 → Non-normal (use Kruskal-Wallis)
```

### 2.4 Understanding Variance Homogeneity Test (Levene's)

**What it tests**: "Are variances (standard deviations) equal across strains?"

**Interpretation**:
- **Levene_p > 0.05** → Fail to reject equal variance → Variances similar ✓
- **Levene_p < 0.05** → Reject equal variance → Variances differ significantly ✗

**Why it matters**:
- ANOVA assumes \(\sigma_1^2 = \sigma_2^2 = ... = \sigma_k^2\)
- Unequal variances → ANOVA p-values become unreliable (Type I error inflation)
- Solution: Use Welch's ANOVA (automatically down-weights high-variance groups)

**Example**:
```
Alanine:  Levene_p = 0.612 > 0.05 → Equal variances
Glucose:  Levene_p = 0.031 < 0.05 → Unequal variances (use Welch's)
```

### 2.5 Test Selection Decision Tree

```
┌─ Normality Test (Shapiro-Wilk)
│  │
│  ├─ p > 0.05 (Normal) ─→ Continue to Variance Test
│  │  │
│  │  ├─ Levene p > 0.05 ──→ ANOVA ✓ (most powerful)
│  │  └─ Levene p < 0.05 ──→ Welch's ANOVA ✓ (corrects variance)
│  │
│  └─ p < 0.05 (Non-normal) ──→ Kruskal-Wallis ✓ (non-parametric)
```

### 2.6 Using This Sheet for Methods & Supplementary Materials

When writing a methods section:

> "We tested normality using the Shapiro-Wilk test and homogeneity of variance using Levene's test. For metabolites meeting both assumptions (n = 32), we applied one-way ANOVA. For metabolites with normal but unequal variances (n = 8), we used Welch's ANOVA. For non-normal metabolites (n = 5), we used the Kruskal-Wallis H test."

---

## 3. PostHoc_Results Sheet – Pairwise Strain Comparisons

### 3.1 Purpose
The global ANOVA/Welch/Kruskal test tells you **"something differs"**, but not **which strains** differ from which. This sheet provides all pairwise comparisons.

### 3.2 Column-by-Column Explanation

| Column | Example | What It Means |
|--------|---------|---------------|
| **Metabolite** | `Alanine` | The compound being analyzed |
| **Strain1** | `K12` | First strain in the pair |
| **Strain2** | `Tme12` | Second strain in the pair |
| **Comparison** | `K12 vs Tme12` | The specific comparison made |
| **Test_Type** | `Tukey_HSD` \| `Games-Howell` \| `Dunn` | Post-hoc test type (depends on global test) |
| **Statistic** | `3.45` | Test statistic (t-like or Z-like) |
| **p_value (raw)** | `0.0034` | Unadjusted p-value |
| **p_FDR / p_adj** | `0.0412` | p-value after False Discovery Rate correction |
| **Significant** | `YES` | `YES` if p_FDR < 0.05 |

### 3.3 Post-Hoc Test Types

#### **Tukey's Honestly Significant Difference (HSD)**
- **Used after**: ANOVA (normal + equal variance)
- **What it does**: Controls family-wise error rate across all pairwise comparisons
- **Interpretation**:
  - Compares each strain pair to a critical value
  - If |test statistic| > critical value → difference is significant
- **Conservative**: Protects against Type I errors; slightly less power

**Example**:
```
Alanine K12 vs Tme12: Tukey_HSD statistic = 3.45, p_FDR = 0.0034
→ K12 and Tme12 differ significantly in alanine (after adjusting for multiple comparisons)
```

#### **Games-Howell Test**
- **Used after**: Welch's ANOVA (normal but unequal variance)
- **What it does**: Like Tukey but doesn't assume equal variances
- **Interpretation**: Similar to Tukey; uses Welch-adjusted comparisons
- **Conservative**: Also controls Type I error

**Example**:
```
Glucose K12 vs PA01: Games-Howell statistic = 2.89, p_FDR = 0.0156
→ K12 and PA01 differ significantly in glucose
→ Accounts for different variances between these strains
```

#### **Dunn's Test** (or **Mann-Whitney U**)
- **Used after**: Kruskal-Wallis (non-normal data)
- **What it does**: Non-parametric pairwise comparisons of ranks
- **Interpretation**: 
  - Tests whether median/distribution of one strain differs from another
  - Robust to outliers
- **Conservative**: Non-parametric, so slightly less power if data were actually normal

**Example**:
```
Phenol K12 vs PA01: Dunn_Test statistic = 2.78, p_FDR = 0.0089
→ K12 and PA01 have different phenol distributions (medians differ)
→ Result is robust to extreme values or non-normal shape
```

### 3.4 Raw p-value vs FDR-Corrected p-value

**Raw p-value** (e.g., 0.0034):
- Probability for **this single comparison** alone
- Ignores all other comparisons
- If you did 100 pairwise tests and used p < 0.05, you'd expect 5 false positives by chance

**p_FDR / p_adj** (e.g., 0.0412):
- Adjusted for **all pairwise comparisons** tested (could be 20–100+ per metabolite)
- Controls False Discovery Rate: \(FDR = E[\text{# false discoveries} / \text{# total discoveries}]\)
- More conservative (larger p-values) but prevents false positives when doing many tests
- **Use this to declare significance** in omics-scale studies

**Decision rule**:
```
If p_FDR < 0.05 → Claim significant (after correction)
If p_raw < 0.05 but p_FDR ≥ 0.05 → Suggestive but not significant after correction
```

### 3.5 Biological Interpretation Strategy

**Step 1**: For a significant metabolite from ANOVA_Results (e.g., Alanine p = 0.00001):

**Step 2**: Filter PostHoc_Results for `Metabolite = Alanine` and `p_FDR < 0.05`:

```
Alanine | K12   | Tme12 | p_FDR = 0.0012 ✓ Significant
Alanine | K12   | Tme13 | p_FDR = 0.0008 ✓ Significant
Alanine | K12   | Tme14 | p_FDR = 0.0023 ✓ Significant
Alanine | K12   | Tme15 | p_FDR = 0.0031 ✓ Significant
Alanine | PA01  | Tmp04 | p_FDR = 0.0156 ✓ Significant
Alanine | PA01  | Tmp05 | p_FDR = 0.0089 ✓ Significant
...
Alanine | Tme12 | Tme13 | p_FDR = 0.7234 ✗ Not significant
```

**Biological story**:
```
"Alanine is significantly elevated in all E. coli clinicals (Tme12–15) 
relative to K12, AND in all P. aeruginosa clinicals (Tmp04–07) 
relative to PA01. However, E. coli and P. aeruginosa clinicals 
do NOT differ from each other in alanine."
```

**This suggests**: Alanine elevation is a **general response to CAUTI conditions** (clinical isolation), not species-specific.

---

## 4. Group_Analysis Sheet – Group-Level Significance

### 4.1 Purpose
Instead of testing all 120+ pairwise comparisons, this sheet tests **3 biological hypotheses**:
1. "Do E. coli clinicals differ from K12 control?"
2. "Do P. aeruginosa clinicals differ from PA01 control?"
3. "Do co-cultures differ from PairEP control?"

### 4.2 Column-by-Column Explanation

| Column | Example | What It Means |
|--------|---------|---------------|
| **Metabolite** | `Alanine` | The compound being analyzed |
| **Group** | `E_coli_clinical` | E. coli isolates (Tme12–15) pooled |
| **Control** | `K12` | Reference control strain |
| **Test_Type** | `Mann-Whitney U` | Non-parametric comparison |
| **Statistic** | `124.5` | U-statistic value |
| **p_value (raw)** | `0.0023` | Unadjusted p-value for this group test |
| **p_FDR / p_adj** | `0.0067` | FDR-adjusted p-value (only 3 tests per metabolite) |
| **Direction** | `UP` \| `DOWN` | Group mean relative to control |
| **Fold_Change_Group** | `2.34` | Mean log₂FC (optional) |

### 4.3 Why Group-Level Testing Matters

**Advantage over pairwise**:
- Only **3 statistical tests** per metabolite (E.coli vs K12, P.aero vs PA01, Coculture vs PairEP)
- Much less stringent FDR correction → **higher power**
- Addresses the **biological hypothesis directly**: "Do clinical isolates differ from controls?"

**Advantage over global ANOVA**:
- Directly tests clinically relevant questions
- Avoids "multiple comparison problem" that plagues pairwise tests
- More interpretable for group-based biomarker development

### 4.4 Biological Interpretation

**Strong result** (e.g., E. coli vs K12, p_FDR = 0.0001):
```
"Clinical E. coli isolates, as a group, have significantly 
elevated alanine compared to the K12 reference strain."
```

**Implication**: Alanine is a **group biomarker** for clinical E. coli isolates.

**Weak or absent result** (e.g., p_FDR = 0.87):
```
"No significant difference between E. coli clinicals and K12 
for this metabolite."
```

**Implication**: This metabolite does **not distinguish** clinical E. coli from K12.

---

## 5. Group_PostHoc Sheet – Between-Group Comparisons

### 5.1 Purpose
After establishing which groups differ from controls, test **between-group differences**:
- E. coli clinical group vs P. aeruginosa clinical group
- E. coli clinical group vs co-culture group
- P. aeruginosa clinical group vs co-culture group

### 5.2 Column-by-Column Explanation

| Column | Example | What It Means |
|--------|---------|---------------|
| **Metabolite** | `Alanine` | The compound being analyzed |
| **Group1** | `E_coli_clinical` | First biological group |
| **Group2** | `P_aero_clinical` | Second biological group |
| **Comparison** | `E.coli vs P.aero` | The specific comparison |
| **Test_Type** | `Mann-Whitney U` \| `Paired_Wilcoxon` | Depends on design |
| **p_value** | `0.0056` | Unadjusted p-value |
| **p_FDR** | `0.0168` | FDR-adjusted p-value (6 tests per metabolite) |
| **Direction** | `UP in Group1` | Which group is higher |
| **Effect_Size** | `0.67` | Cohen's d or r (magnitude of difference) |

### 5.3 Test Selection: Paired vs Unpaired

**Paired design** (Paired Wilcoxon or paired t-test):
- Used when co-culture pairs have **known internal structure**
- E.g., Pair412 = Tme12 + Tmp04, Pair513 = Tme13 + Tmp05, etc.
- Tests whether **matched pairs differ systematically**

**Unpaired design** (Mann-Whitney U or independent t-test):
- Used for E. coli vs P. aeruginosa (unrelated groups)
- Tests whether group-level distributions differ

### 5.4 Biological Interpretation

```
E. coli clinical vs P. aeruginosa clinical:
p_FDR = 0.0012, Direction = UP in P.aero

→ P. aeruginosa clinicals produce significantly more of this metabolite
→ Species-specific metabolic difference
→ Could reflect different energy metabolism or virulence strategies
```

---

## 6. FoldChange Sheet – Magnitude of Changes

### 6.1 Purpose
Provides **quantitative changes** for all metabolites across all strains, relative to their designated controls.

### 6.2 Structure

| Metabolite | K12 | Tme12 | Tme13 | PA01 | Tmp04 | Tmp05 | PairEP | Pair412 |
|------------|-----|-------|-------|------|-------|-------|--------|---------|
| Alanine | 0 | +2.34 | +1.89 | 0 | –0.45 | –0.67 | 0 | +0.98 |
| Glucose | 0 | +0.56 | +0.34 | 0 | +1.23 | +1.01 | 0 | +0.12 |

### 6.3 Interpretation of Log₂ Fold-Change Values

\[\text{log}_2(\text{FC}) = \log_2\left(\frac{\text{strain mean}}{\text{control mean}}\right)\]

| log₂FC | Fold Change | Interpretation |
|--------|-------------|-----------------|
| +3.0 | 8-fold increase | Metabolite highly elevated in this strain |
| +2.0 | 4-fold increase | Strong elevation |
| +1.0 | 2-fold increase | Moderate elevation |
| +0.5 | 1.4-fold increase | Slight elevation |
| 0 | No change (=control) | Equal to control (control strains always 0) |
| –0.5 | 0.7-fold (30% lower) | Slight decrease |
| –1.0 | 0.5-fold (2-fold lower) | Moderate decrease |
| –2.0 | 0.25-fold (4-fold lower) | Strong decrease |

### 6.4 Filtering by Magnitude

Cowtea uses **|log₂FC| > 0.75** as the fold-change threshold for "biologically meaningful" changes:

- |log₂FC| = 0.75 → ~1.7-fold change
- Rationale: Avoids false positives from tiny statistical differences with no biological relevance

**For visualizations** (volcano, bubble plots):
- Metabolites with |log₂FC| > 0.75 AND p < 0.05 are highlighted
- This combines **statistical significance** (p-value) with **biological magnitude** (fold-change)

---

## 7. Metabolite_Pathway_Key Sheet – Biological Classification

### 7.1 Purpose
Annotates each metabolite with its **metabolic pathway** and biological function.

### 7.2 Structure

| Metabolite | Pathway | Color | Biological_Function | Examples of Elevation | 
|------------|---------|-------|----------------------|----------------------|
| Alanine | Amino Acids | Blue | Protein synthesis, gluconeogenesis | Growth, energy stress |
| Glucose | Carbohydrates | Green | Primary energy source | Active growth |
| Citrate | Organic Acids | Orange | TCA cycle intermediate | Aerobic metabolism |
| Phenol | Phenolic Compounds | Gray | Secondary metabolite, stress response | Stress, CAUTI conditions |

### 7.3 Biological Interpretation by Pathway

#### **Amino Acids (Blue)**
- **If elevated in clinicals**: Indicates enhanced amino acid catabolism (amino acid starvation response or nutrient stress)
- **If elevated in co-cultures**: Possible peptidolysis or protein degradation by one partner
- **If similar to control**: May indicate normal protein metabolism

#### **Carbohydrates (Green)**
- **If elevated in clinicals**: Active glycolysis, high energy demand
- **If depleted in clinicals**: Glucose limitation (competitive consumption in co-cultures)
- **Pattern**: E. coli and P. aeruginosa differ in sugar preferences

#### **Organic Acids (Orange)**
- **If elevated**: Active metabolic flux through TCA cycle, aerobic respiration
- **If depleted**: Anaerobic or fermentative metabolism
- **Specific examples**:
  - High lactate → fermentation, low oxygen
  - High citrate → active aerobic metabolism
  - High acetate → overflow metabolism (glucose excess)

#### **Nucleotides/Nucleosides (Red)**
- **If elevated**: Active DNA/RNA synthesis, high growth rate
- **If depleted**: Stationary phase or growth limitation
- **Biofilm context**: Elevated nucleotides suggest active cell replication in biofilm

#### **Polyamines (Pink)**
- **If elevated**: Stress response, growth, biofilm formation
- **Significance**: Polyamine synthesis is associated with virulence in many pathogens
- **CAUTI context**: Putrescine/spermidine are CAUTI biomarkers

#### **Phenolic Compounds (Gray)**
- **If elevated**: Secondary metabolite production, stress response
- **Context-dependent**: May indicate oxidative stress or active quorum sensing
- **Species-specific**: Some phenolics are P. aeruginosa-characteristic

### 7.4 Example: Reading Pathway Context

```
Result: Alanine, Glutamine, Proline all elevated in Tme12

Interpretation:
  → All are amino acids (protein metabolism pathway)
  → Could indicate: 
     a) Amino acid starvation response (scavenging)
     b) Enhanced protein turnover/degradation
     c) Adaptation to CAUTI environment (low nutrient conditions)

Next step: 
  → Check if OTHER amino acids are also elevated (group pattern)
  → If yes → supports "amino acid metabolism activation"
  → If no → specific amino acids might be strain-specific
```

---

## 8. Summary_Statistics Sheet – Descriptive Information

### 8.1 Purpose
Provides basic descriptive statistics for each metabolite in each strain group.

### 8.2 Columns

| Column | Meaning |
|--------|---------|
| **Metabolite** | The compound |
| **Group** | E. coli, P. aeruginosa, co-culture, or control |
| **N** | Number of replicates in this group |
| **Mean** | Average abundance |
| **SD** | Standard deviation (variability within group) |
| **Min / Max** | Range of values |
| **Median** | 50th percentile (robust to outliers) |

### 8.3 Using This Sheet

**Check for outliers**:
```
Alanine in Tme12:
Mean = 1500, Median = 1520, SD = 150, N = 3
→ Mean ≈ Median, SD reasonable → no major outliers
```

```
Glucose in Tmp04:
Mean = 800, Median = 900, SD = 400, N = 3
→ Mean << Median, large SD → possible outlier
→ One very low value is pulling mean down
```

**Assess group homogeneity**:
```
If SD is large relative to mean → group is heterogeneous
Biological interpretation: Strains within a group vary metabolically
```

---

## PART II: CUSTOM CONTROL STRATEGIES

---

## 9. Datasets Without Medium Control

### 9.1 Situation
You have metabolomics data for **clinical strains** (e.g., Tme12, Tmp04, Pair412) but **no K12, PA01, or PairEP** control strains. You still want to:
- Calculate fold-changes
- Generate volcano/bubble plots
- Perform group analysis

### 9.2 What Goes Wrong Without a Control

The code expects:
```python
controls = ['K12', 'PA01', 'PairEP']
```

When creating the control row:
```python
control_data = df[df['Strain'] == 'K12']  # ← Raises KeyError if K12 not present
```

**Result**: Analysis fails with:
```
KeyError: 'K12 not found in Strain column'
```

### 9.3 Two Strategies for Creating Controls

#### **Strategy A: Unified Synthetic Control (Recommended)**

Create a **single synthetic reference** as the **mean across all samples**.

**Advantages**:
- Simple to implement
- No biological assumptions (all strains are equal as baseline)
- Good for exploratory analysis when no prior control exists

**Disadvantages**:
- Fold-changes become relative-to-average, not to a true control
- Less interpretable biologically

**Procedure**:

1. **Identify which samples to pool** as the baseline.

Option 1a: Pool ALL samples:
```python
# Use all data as baseline
reference_samples = df[df['Strain'].isin(all_strains)]  
control_data = reference_samples
```

Option 1b: Pool only certain strains (e.g., "least affected" condition):
```python
# Example: pool only the first timepoint if temporal data
control_data = df[df['Timepoint'] == 'T0']
```

2. **Compute control statistics**:
```python
metabolites = [col for col in df.columns if col not in ['Sample', 'Strain']]

ctrl_means = {}
ctrl_sds = {}
for met in metabolites:
    ctrl_means[met] = control_data[met].mean()
    ctrl_sds[met] = control_data[met].std()
```

3. **Create synthetic control row**:
```python
ctrl_row = {
    'Sample': 'UnifiedControl_Baseline',
    'Strain': 'UnifiedControl'
}
for met in metabolites:
    ctrl_row[met] = ctrl_means[met]

df_with_control = pd.concat(
    [pd.DataFrame([ctrl_row]), df], 
    ignore_index=True
)
```

4. **Update configuration**:
```python
controls = ['UnifiedControl']  # ← Changed from ['K12', 'PA01', 'PairEP']
```

#### **Strategy B: Designate One Strain as Reference**

If one strain is biologically the **reference** (e.g., wild-type, least virulent), use it as the control.

**Advantages**:
- More interpretable (fold-changes relative to a real strain)
- Preserves biological meaning

**Disadvantages**:
- Requires justification for the control choice
- That strain's fold-change will always be 0 (it's the reference)

**Procedure**:

1. **Decide which strain is reference** (e.g., first strain in your dataset or a known baseline).

2. **Simply add it to the controls list**:
```python
# Instead of K12/PA01/PairEP, use your chosen strain
controls = ['MyReferenceStrain']
```

3. **No code modification needed** in `add_control_to_dataset`:
```python
# This will now select your reference strain as the control
control_data = df[df['Strain'].isin(controls)]
```

---

## 10. Exact Code Modifications for Cowtea v1.0.0

### 10.1 Location 1: Global Configuration (Top of File)

**Original code** (around line 70–90):
```python
# GLOBAL CONSTANTS & SETTINGS
controls = ['K12', 'PA01', 'PairEP']  # Reference control strains
e_mono = ['Tme12', 'Tme13', 'Tme14', 'Tme15']
p_mono = ['Tmp04', 'Tmp05', 'Tmp06', 'Tmp07']
monocultures = e_mono + p_mono
pairs = ['Pair412', 'Pair513', 'Pair614', 'Pair715']
```

**Modified for unified control**:
```python
# GLOBAL CONSTANTS & SETTINGS
# MODIFIED: Using synthetic unified control instead of strain-specific controls
controls = ['UnifiedControl']  # Synthetic reference (mean of all samples)
e_mono = ['Tme12', 'Tme13', 'Tme14', 'Tme15']
p_mono = ['Tmp04', 'Tmp05', 'Tmp06', 'Tmp07']
monocultures = e_mono + p_mono
pairs = ['Pair412', 'Pair513', 'Pair614', 'Pair715']
```

**Modified for strain-as-control**:
```python
# GLOBAL CONSTANTS & SETTINGS
# MODIFIED: Using first strain as reference control
controls = ['Tme12']  # ← Changed to your reference strain
e_mono = ['Tme12', 'Tme13', 'Tme14', 'Tme15']
p_mono = ['Tmp04', 'Tmp05', 'Tmp06', 'Tmp07']
monocultures = e_mono + p_mono
pairs = ['Pair412', 'Pair513', 'Pair614', 'Pair715']
```

### 10.2 Location 2: Control Addition Function

**Find the function** `add_control_to_dataset()` (typically around line 200–250).

**Original code**:
```python
def add_control_to_dataset(df, metabolites):
    """Add control reference row to dataset."""
    medium = 'AUM' if 'AUM' in df['Sample'].str.contains('AUM20', na=False).values else 'ISO'
    
    control_data = df[df['Strain'] == medium]  # ← Assumes medium is a Strain
    if control_data.empty:
        raise ValueError(f"No control strain '{medium}' found in data.")
    
    ctrl_means = {met: control_data[met].mean() for met in metabolites}
    ctrl_sds = {met: control_data[met].std() for met in metabolites}
    
    ctrl_row = {'Sample': f"{df['Sample'].iloc[0].split('_')[0]}_Control", 'Strain': 'Control'}
    ctrl_row.update(ctrl_means)
    
    df_with_control = pd.concat([pd.DataFrame([ctrl_row]), df], ignore_index=True)
    return df_with_control
```

**Modified for unified control** (Replace entire function):
```python
def add_control_to_dataset(df, metabolites):
    """
    Add control reference row to dataset.
    Modified for datasets without medium-specific controls.
    """
    
    # Try to find existing control strains in the controls list
    existing_controls = df[df['Strain'].isin(controls)]
    
    if not existing_controls.empty:
        # Control strains exist in data → use them as-is
        print(f"✓ Found {len(existing_controls)} control samples in data.")
        return df
    else:
        # No control strains → create synthetic unified control
        print("⚠ No control strains found. Creating synthetic unified control...")
        
        # Compute mean across ALL samples for each metabolite
        ctrl_means = {met: df[met].mean() for met in metabolites}
        ctrl_sds = {met: df[met].std() for met in metabolites}
        
        # Create synthetic control row
        ctrl_row = {
            'Sample': 'UnifiedControl_AllSamples',
            'Strain': 'UnifiedControl'
        }
        ctrl_row.update(ctrl_means)
        
        # Insert at beginning of dataframe
        df_with_control = pd.concat(
            [pd.DataFrame([ctrl_row]), df], 
            ignore_index=True
        )
        
        print(f"✓ Synthetic control created: {dict(list(ctrl_means.items())[:3])} ...")
        
        return df_with_control
```

### 10.3 Location 3: Sample Order Definition

**Find the section** that defines `sample_order` (typically in `add_control_to_dataset()` or main function).

**Original code**:
```python
sample_order = ['Control', 'PA01', 'K12', 'PairEP',
                'Tmp04', 'Tme12', 'Pair412',
                'Tmp05', 'Tme13', 'Pair513',
                'Tmp06', 'Tme14', 'Pair614',
                'Tmp07', 'Tme15', 'Pair715']
```

**Modified for unified control**:
```python
# Build sample_order dynamically based on available strains
available_strains = df_with_control['Strain'].unique().tolist()

# Put control first, then rest
if 'UnifiedControl' in available_strains:
    sample_order = ['UnifiedControl'] + [s for s in available_strains if s != 'UnifiedControl']
else:
    # Fallback
    sample_order = available_strains
    
print(f"✓ Sample order: {sample_order}")
```

### 10.4 Location 4: Medium Detection (Optional Modification)

If your data **does not have AUM/ISO naming**, modify the medium detection:

**Original code**:
```python
def detect_medium(df):
    if df['Sample'].str.contains('AUM20').any():
        return 'AUM'
    elif df['Sample'].str.contains('ISO20').any():
        return 'ISO'
    else:
        raise ValueError("Cannot determine medium type from sample names.")
```

**Modified for custom datasets**:
```python
def detect_medium(df):
    """Detect medium from sample name or use custom identifier."""
    if df['Sample'].str.contains('AUM20', na=False).any():
        return 'AUM'
    elif df['Sample'].str.contains('ISO20', na=False).any():
        return 'ISO'
    else:
        # No AUM/ISO identifier → extract custom identifier
        sample_prefix = df['Sample'].iloc[0].split('_')[0]
        print(f"⚠ No AUM/ISO detected. Using sample prefix '{sample_prefix}' as medium identifier.")
        return sample_prefix  # e.g., 'CustomExperiment'
```

### 10.5 Location 5: File Naming (Optional Modification)

If medium cannot be detected, ensure file naming doesn't fail:

**Original code**:
```python
out_folder = f"{medium}_sup_Analysis_{timestamp}"
excel_path = os.path.join(out_folder, f"Statistical_results_{medium}_sup.xlsx")
```

**Modified for robustness**:
```python
# Use medium identifier (could be 'AUM', 'ISO', or custom)
medium = detect_medium(df_filtered) if df_filtered is not None else 'CustomAnalysis'

out_folder = f"{medium}_sup_Analysis_{timestamp}"
excel_path = os.path.join(out_folder, f"Statistical_results_{medium}_sup.xlsx")

print(f"✓ Output folder: {out_folder}")
print(f"✓ Results will be saved to: {excel_path}")
```

---

## 11. Step-by-Step Workflow: Running Cowtea Without K12/PA01/PairEP

### 11.1 Your Data

```excel
Sample          Strain      Alanine   Glucose   ...
MyExperiment_1  MyStrain1   1234      5678
MyExperiment_2  MyStrain1   1245      5690
MyExperiment_3  MyStrain1   1256      5701
MyExperiment_4  MyStrain2   2234      3456
MyExperiment_5  MyStrain2   2245      3467
...
```

**Problem**: No K12, PA01, or PairEP.

### 11.2 Modifications Required

**Step 1**: Update global configuration (line ~75):
```python
controls = ['UnifiedControl']  # ← Change this line
```

**Step 2**: Update `add_control_to_dataset()` function (copy modified version from 10.2 above).

**Step 3**: Run Cowtea as normal:
```bash
python Cowtea.py "MyExperiment.xlsx"
```

**Step 4**: Cowtea will:
- Read your data
- Detect "MyExperiment" as medium
- Create synthetic "UnifiedControl" row
- Proceed with entire analysis pipeline
- Output to folder: `MyExperiment_sup_Analysis_2025-12-18_14-30/`

### 11.3 Interpreting Results with Unified Control

**Important note in methods section**:
```
"As a reference control strain was not available, 
a synthetic unified control was created as the mean 
abundance across all samples. Fold-changes and group 
comparisons are therefore relative to this global baseline, 
rather than to a specific control strain."
```

**Fold-change interpretation**:
- Positive log₂FC → metabolite elevated relative to **global average**
- Negative log₂FC → metabolite depleted relative to **global average**
- Group patterns still meaningful (e.g., "MyStrain2 shows higher phenol than average")

---

## 12. Troubleshooting Custom Controls

### Problem 1: "KeyError: 'K12 not found'"

**Cause**: Cowtea is trying to find 'K12' in your Strain column but it doesn't exist.

**Solution**: 
```python
# Make sure you changed controls list:
controls = ['UnifiedControl']

# And updated add_control_to_dataset() function
```

### Problem 2: "Fold-changes are all zero or symmetric"

**Cause**: Synthetic control is being used but incorrectly structured.

**Check**: 
- Is the 'UnifiedControl' row actually in the dataframe after `add_control_to_dataset()`?
- Are metabolite values filled in for that row?

**Debug code**:
```python
print(df_with_control[df_with_control['Strain'] == 'UnifiedControl'])
# Should show one row with all metabolite columns filled
```

### Problem 3: "Output folder has unexpected name"

**Cause**: Medium detection returned unexpected value.

**Solution**: 
```python
# Check what medium was detected
print(f"Detected medium: {medium}")

# If using custom prefix, ensure it's valid for folder naming
medium = "CustomAnalysis" if medium is None else medium
```

---

## 13. Comparison: K12/PA01/PairEP vs Unified Control

| Aspect | With K12/PA01/PairEP | With Unified Control |
|--------|---------------------|----------------------|
| **Data requirement** | Must have control strains | Any strains work |
| **Biological interpretation** | FC relative to known control | FC relative to average |
| **Biomarker discovery** | Gold standard | Exploratory (reveals relative changes) |
| **Group analysis** | Clinicals vs control | Clinicals vs average |
| **Publication appropriateness** | Ideal for CAUTI studies | Suitable for discovery phase |
| **Code modification** | None | Change `controls` list + modify function |

---

## 14. Summary: When to Use Each Strategy

### Use Strategy A (Unified Control) if:
- You have no designated control strain
- You want a quick, exploratory analysis
- Your interest is in relative changes between strains
- You're in a discovery phase

### Use Strategy B (Strain as Control) if:
- One strain is biologically the "normal" baseline
- You want to compare all others to that specific strain
- You have a meaningful comparison (wild-type vs mutants, etc.)

### Use Original Code (K12/PA01/PairEP) if:
- You're working with CAUTI bacterial strains
- You have the standard control strains
- You're comparing clinical isolates to references
- You're following the CAUTI protocol exactly

---

## APPENDIX: Full Modified Function (Ready to Copy)

Save this function and replace the original in Cowtea.py:

```python
def add_control_to_dataset(df, metabolites):
    """
    Add control reference row to dataset.
    
    This function handles three scenarios:
    1. Medium-specific control exists (original behavior)
    2. Control strains in global list exist (uses them)
    3. No controls exist (creates synthetic unified control)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with 'Sample' and 'Strain' columns
    metabolites : list
        List of metabolite column names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with control row prepended
    """
    
    # Scenario 1: Check if control strains already exist in data
    existing_controls = df[df['Strain'].isin(controls)]
    if not existing_controls.empty:
        print(f"✓ Found {len(existing_controls)} existing control samples.")
        return df
    
    # Scenario 2 & 3: No existing controls → create synthetic
    print("⚠ No existing control strains found. Creating synthetic unified control...")
    
    # Compute statistics across all samples
    ctrl_means = {}
    ctrl_sds = {}
    for met in metabolites:
        try:
            ctrl_means[met] = pd.to_numeric(df[met], errors='coerce').mean()
            ctrl_sds[met] = pd.to_numeric(df[met], errors='coerce').std()
        except Exception as e:
            print(f"  Warning: Could not compute control stats for {met}: {e}")
            ctrl_means[met] = np.nan
            ctrl_sds[met] = np.nan
    
    # Create synthetic control row
    ctrl_row = {
        'Sample': 'UnifiedControl_00000',
        'Strain': 'UnifiedControl'
    }
    for met in metabolites:
        ctrl_row[met] = ctrl_means[met]
    
    # Prepend to dataframe
    df_with_control = pd.concat(
        [pd.DataFrame([ctrl_row]), df],
        ignore_index=True
    )
    
    # Store control statistics globally
    global control_sds_dict
    control_sds_dict = ctrl_sds
    
    # Set up sample order (dynamic)
    global sample_order
    available_strains = df_with_control['Strain'].unique().tolist()
    if 'UnifiedControl' in available_strains:
        sample_order = ['UnifiedControl'] + [s for s in available_strains if s != 'UnifiedControl']
    else:
        sample_order = available_strains
    
    print(f"✓ Synthetic control created with {len(metabolites)} metabolites.")
    print(f"✓ First 3 metabolites: {dict(list(ctrl_means.items())[:3])}")
    
    return df_with_control
```

---

**Document Version**: 1.0.0  
**Last Updated**: December 18, 2025  
**Status**: Complete & Ready for Distribution
