# BACTERIAL METABOLOMICS ANALYSIS PIPELINE
## Comprehensive Methodology and Data Processing Framework

---

## EXECUTIVE SUMMARY

This document provides a complete technical description of a six-phase statistical framework for analyzing metabolomic data from bacterial monocultures and co-cultures. The pipeline integrates automated data quality assessment, multi-level assumption testing, adaptive statistical test selection, and hierarchical post-hoc analysis with rigorous multiple testing correction.

**Target Applications:**
- Multi-strain comparative metabolomics
- Bacterial co-culture interaction analysis  
- Phenotypic metabolic profiling
- Clinical isolate metabolic characterization
- Polybacterial community metabolism

**Key Features:**
- Automated transformation optimization
- Assumption-driven test selection
- Parallel processing paths for statistical analysis
- Multi-level hypothesis testing (global + group-specific)
- Publication-ready visualization suite

---

## TABLE OF CONTENTS

1. PHASE 1: DATA IMPORT & PREPROCESSING
2. PHASE 2: DATA QUALITY ASSESSMENT & TRANSFORMATION
3. PHASE 3: STATISTICAL ASSUMPTION TESTING
4. PHASE 4: GLOBAL STATISTICAL ANALYSIS
5. PHASE 5: MULTI-LEVEL POST-HOC ANALYSIS
6. PHASE 6: VISUALIZATION & REPORTING
7. STATISTICAL RATIONALE
8. EXPECTED OUTPUTS & INTERPRETATION

---

## PHASE 1: DATA IMPORT & PREPROCESSING

### 1.1 Objective
Integrate raw metabolomics data from Excel format, establish experimental context, and prepare data matrix for quality assessment.

### 1.2 Processing Steps

#### Step 1.2.1: Load Data Matrix
**Input Format:** Excel spreadsheet containing:
- Metabolite identifiers (rows)
- Sample identifiers with strain/condition labels (columns)
- Metabolite concentration values (matrix entries)
- Optional metadata (detection status, quality flags)

**Operations:**
```
Read Excel file
├─ Sheet: 'Sheet1' or user-specified
├─ Structure: Metabolites × Samples matrix
├─ Data type validation (numeric concentrations)
└─ Metadata extraction (strain IDs, sample types)
```

**Quality Checks:**
- Verify matrix dimensions (n_metabolites > 0, n_samples > 0)
- Check for missing values and document prevalence
- Validate strain/condition labels
- Confirm replicate structure (technical replicates identified)

#### Step 1.2.2: Medium Type Detection
**Rationale:** Experimental medium (ISO, AUM, or other) affects metabolite composition baseline and should be analyzed separately.

**Detection Algorithm:**
```
FOR each sample name:
  IF contains "ISO" → Medium = ISO
  ELSE IF contains "AUM" → Medium = AUM
  ELSE IF contains [custom_pattern] → Medium = Custom
  
IF inconsistent mediums detected → Prompt user for clarification
```

**Filtering Logic:**
- Extract samples matching detected medium
- Document samples excluded from analysis
- Record filtering criteria for reproducibility

**Output:** Medium-specific filtered dataset with documentation

#### Step 1.2.3: Control Reference Integration
**Purpose:** Establish baseline metabolite levels for:
- Fold-change calculations
- Relative abundance quantification
- Normalization reference

**Procedure:**
```
Step 1: Identify control strains from experimental design
  └─ Examples: K12 (E. coli reference), PA01 (P. aeruginosa reference)

Step 2: Calculate control statistics
  FOR each metabolite:
    control_mean = mean(metabolite values in control samples)
    control_sd = std(metabolite values in control samples)
    
Step 3: Create synthetic control row
  New_Row = [strain="Control", values=control_means]
  
Step 4: Append to data matrix
  Dataset_with_control = [Original_Data; Control_Row]
```

**Quality Assurance:**
- Verify control samples have sufficient replicates (n ≥ 3)
- Check for outliers in control group
- Document control selection rationale

**Output:** Data matrix augmented with control reference

---

## PHASE 2: DATA QUALITY ASSESSMENT & TRANSFORMATION

### 2.1 Objective
Improve statistical test power and robustness by identifying and applying optimal data transformation to enhance normality while preserving biological interpretability.

### 2.2 Why Transform Metabolomic Data?

**Problem:** Metabolomics data typically exhibits non-normal distributions due to:
- Log-normal concentration distributions in biological systems
- Right-skewed density due to detection limits
- Outliers from technical variability
- Unequal variances across metabolite levels

**Solution:** Apply variance-stabilizing and normality-enhancing transformations

**Benefit-Risk Analysis:**
```
Benefit:  ✓ Increased statistical power
          ✓ More robust p-value estimates
          ✓ Reduced Type II error
          
Risk:     - Slight effect size interpretation complexity
          - Assumption that transformed scale is valid
          
Resolution: Select transformation maximizing power while 
            maintaining biological plausibility
```

### 2.3 Transformation Methods Evaluated

#### 2.3.1 Identity Transformation (No Transformation)
**Formula:** y' = x

**Characteristics:**
- Preserves original scale and interpretability
- No mathematical manipulation
- Baseline comparison point

**Use When:** Already normal or sufficient for preliminary assessment

#### 2.3.2 Logarithmic Transformation
**Formula:** y' = log(x + c), where c = small constant (prevents log(0))

**Mathematical Basis:**
- Converts multiplicative relationships to additive
- Reduces right skew common in biological data
- Variance-stabilizing for log-normal distributions

**Application:** Most common in metabolomics

**Interpretation:** Changes in log scale represent fold-changes

#### 2.3.3 Square Root Transformation
**Formula:** y' = √x

**Characteristics:**
- Less aggressive than log transformation
- Less distortion of large values
- Variance stabilization for Poisson-like data

**Use When:** Moderate skewness, preservation of high-concentration metabolites important

#### 2.3.4 Box-Cox Power Transformation
**Formula:** y'(λ) = (x^λ - 1)/λ for λ ≠ 0, or ln(x) for λ = 0

**Optimization:** 
```
Maximum likelihood estimation:
  λ* = argmax L(λ | data)
  
Result: Optimal power parameter λ* identified
```

**Characteristics:**
- Data-driven parameter selection
- Automatically chooses between no transformation (λ≈1), 
  log (λ≈0), and square root (λ≈0.5)
- Often provides best normality improvement

**Implementation:** Apply identical λ to all metabolites

#### 2.3.5 Cube Root Transformation
**Formula:** y' = ∛x = x^(1/3)

**Characteristics:**
- Intermediate between square root and identity
- Handles negative values (unlike log)
- Mild normality improvement

#### 2.3.6 Generalized Logarithm (g-log)
**Formula:** y' = log(x + √(x² + c))

**Characteristics:**
- Handles negative values and zeros
- Asymptotically approaches log for large x
- Hyperbolic sine inverse approximation

**Use When:** Data contains near-zero or measurement errors

#### 2.3.7 Inverse Transformation
**Formula:** y' = 1/x (for positive data) or 1/(max - x) (for mixed)

**Characteristics:**
- Aggressive transformation for heavily skewed data
- Can distort smaller values significantly
- Less commonly used in metabolomics

### 2.4 Transformation Selection Protocol

**Step 1: Apply All Methods**
```
FOR each of 7 transformation methods:
  FOR each metabolite:
    Apply transformation
    Store transformed values
```

**Step 2: Assess Normality Post-Transformation**
```
FOR each transformation method:
  FOR each metabolite:
    Perform Shapiro-Wilk normality test
    Record p-value (p > 0.05 = normal)
  
  Calculate: pass_rate = (n_normal / n_metabolites) × 100%
```

**Step 3: Compare Methods**
```
Rank transformations by pass_rate:
  Method_1: 85% pass rate ← SELECT THIS
  Method_2: 78% pass rate
  Method_3: 72% pass rate
  ...
```

**Step 4: Secondary Selection Criteria (if tied)**
```
Compare mean skewness magnitude:
  |Σ skewness| / n_metabolites

Lower skewness → More symmetric → Better choice
```

**Step 5: Selection Output**
```
Selected transformation: Method_X
Expected normality improvement: Y%
Pass rate: Z% of metabolites now pass normality test
```

**Step 6: Apply Selected Transformation**
```
Apply to entire dataset:
  Dataset_transformed = Transform(Dataset_filtered, method=Method_X)
```

**Documentation Requirement:**
- Save transformation comparison table
- Visualize before/after distributions
- Document selection rationale for methods section

---

## PHASE 3: STATISTICAL ASSUMPTION TESTING

### 3.1 Objective
Formally assess whether transformed data meet parametric test assumptions, determining which statistical test (ANOVA, Welch's, or Kruskal-Wallis) is appropriate for each metabolite.

### 3.2 Assumptions for Parametric Tests

**Assumption 1: Normality**
- **Definition:** Data distributed according to normal distribution N(μ, σ²)
- **Importance:** ANOVA F-test assumes normality of residuals
- **Severity if Violated:** Moderate; ANOVA relatively robust with n > 20

**Assumption 2: Homogeneity of Variance**
- **Definition:** Equal variance across strain groups (σ₁² = σ₂² = ... = σₖ²)
- **Importance:** ANOVA assumes constant variance
- **Severity if Violated:** Can inflate or deflate Type I error; Welch's corrects this

**Assumption 3: Independence**
- **Definition:** Observations are independent (no repeated measures on same sample)
- **Importance:** Fundamental to all statistical tests
- **Severity if Violated:** Critical; cannot use these tests if violated
- **Assurance:** Design-based (randomized replicates)

### 3.3 Test 1: Shapiro-Wilk Normality Test

**Purpose:** Formally test if data follow normal distribution

**Test Statistic:**
```
W = [Σᵢ aᵢ x₍ᵢ₎]² / Σᵢ(xᵢ - x̄)²

where:
  x₍ᵢ₎ = ith order statistic (sorted data)
  aᵢ = test coefficients from normal order statistics
  x̄ = sample mean
```

**Hypotheses:**
- H₀: Data are normally distributed
- H₁: Data are not normally distributed

**Interpretation:**
```
p-value > 0.05 → FAIL TO REJECT H₀ → Normal distribution supported
p-value < 0.05 → REJECT H₀ → Non-normal distribution detected
```

**Implementation Notes:**
- Apply per metabolite (not globally)
- Apply post-transformation
- Effective sample size: typically n = 5-50 (metabolite × strain combinations)

**Output Table:**
```
Metabolite          | p-value | Normal? | Test Status
Alanine            | 0.342   | YES     | ✓ Pass
Glucose            | 0.002   | NO      | ✗ Fail
Lysine             | 0.156   | YES     | ✓ Pass
...
```

### 3.4 Test 2: Levene's Test for Variance Homogeneity

**Purpose:** Formally test if variance is equal across strain groups

**Test Statistic (Simplified):**
```
W = (N-k) Σᵏᵢ₌₁ nᵢ(Zᵢ. - Z..)² / Σᵏᵢ₌₁ Σⁿⁱⱼ₌₁(Zᵢⱼ - Zᵢ.)²

where:
  Zᵢⱼ = |xᵢⱼ - median(group i)|
  N = total observations
  k = number of groups
```

**Hypotheses:**
- H₀: σ₁² = σ₂² = ... = σₖ² (equal variances)
- H₁: Not all variances equal

**Interpretation:**
```
p-value > 0.05 → FAIL TO REJECT H₀ → Equal variances supported
p-value < 0.05 → REJECT H₀ → Unequal variances detected
```

**Application:**
- Per metabolite
- Across all strain groups
- On transformed scale

**Output Table:**
```
Metabolite          | p-value | Equal Var? | Recommendation
Alanine            | 0.412   | YES        | ANOVA eligible
Glucose            | 0.031   | NO         | Use Welch's
Lysine             | 0.089   | YES        | ANOVA eligible
...
```

### 3.5 Test 3: Kurtosis and Skewness Analysis

**Purpose:** Examine distributional shape beyond normality test binary classification

**Metrics:**

**Skewness:** Measures asymmetry
```
Skewness = E[(x - μ)³] / σ³

Interpretation:
  Skewness ≈ 0 → Symmetric distribution
  Skewness > 0 → Right-tailed (positive skew)
  Skewness < 0 → Left-tailed (negative skew)
  
Threshold: |Skewness| < 0.5 considered acceptable
```

**Kurtosis:** Measures tail heaviness
```
Excess Kurtosis = E[(x - μ)⁴] / σ⁴ - 3

Interpretation:
  Kurtosis ≈ 0 → Normal tail behavior
  Kurtosis > 0 → Heavy tails (leptokurtic)
  Kurtosis < 0 → Light tails (platykurtic)
  
Threshold: |Kurtosis| < 1 considered acceptable
```

**Application:**
- Per metabolite
- Compare across transformation methods
- Guide binary transformation selection if Shapiro-Wilk results ambiguous

**Output:**
```
Transform Method | Mean Skewness | Mean Kurtosis | Notes
Identity         | 0.75          | 2.14          | Right-skewed
Log              | 0.12          | 0.38          | ← BEST (closest to 0)
Sqrt             | 0.31          | 0.67          | Moderate
Box-Cox          | 0.08          | 0.41          | Similar to log
...
```

### 3.6 Integrated Assumption Assessment

**Classification Matrix:**

```
Assumption Combination          → Statistical Test Assigned
─────────────────────────────────────────────────────────────
Normality: YES                    → Consider parametric tests
Variance Equal: YES               
→ ONE-WAY ANOVA (Path A)

Normality: YES                    → Welch's F-statistic
Variance Equal: NO                
→ WELCH'S ANOVA (Path B)

Normality: NO                     → Rank-based non-parametric
(Any variance pattern)           
→ KRUSKAL-WALLIS (Path C)
```

**Decision Algorithm:**
```
FOR each metabolite:
  IF (Shapiro_p > 0.05) AND (Levene_p > 0.05):
    Test_Selected = "ANOVA"
  ELSE IF (Shapiro_p > 0.05) AND (Levene_p < 0.05):
    Test_Selected = "Welch's ANOVA"
  ELSE:
    Test_Selected = "Kruskal-Wallis"
  
  Store: (Metabolite, Test_Selected)
```

**Output:** Test classification table for all metabolites

---

## PHASE 4: GLOBAL STATISTICAL ANALYSIS

### 4.1 Objective
Identify metabolites exhibiting statistically significant variation across bacterial strains using assumption-appropriate statistical tests.

### 4.2 Statistical Test Selection and Implementation

### 4.2A PATH A: ONE-WAY ANOVA (ANOVA Eligible)

**Conditions Met:**
- Shapiro-Wilk normality test: p > 0.05
- Levene's variance homogeneity: p > 0.05
- Data approximately normal and homoscedastic

**Test Purpose:** Test overall difference in means across strain groups

**Test Statistic:**
```
F = MSbetween / MSwithin = [Σᵏᵢ₌₁ nᵢ(x̄ᵢ - x̄)² / (k-1)] / [Σᵏᵢ₌₁ Σⁿⁱⱼ₌₁(xᵢⱼ - x̄ᵢ)² / (N-k)]

where:
  k = number of strain groups
  nᵢ = sample size for group i
  x̄ᵢ = mean for group i
  x̄ = grand mean
  N = total sample size
```

**Hypotheses:**
- H₀: μ₁ = μ₂ = ... = μₖ (all strain means equal)
- H₁: At least one strain mean differs

**Null Distribution:** F(k-1, N-k) under H₀

**P-value Calculation:**
```
p-value = P(F(k-1, N-k) ≥ F_observed | H₀)
```

**Decision Rule:**
```
IF p-value < 0.05:
  REJECT H₀ → Significant strain differences detected
ELSE:
  FAIL TO REJECT H₀ → No significant strain differences
```

**Assumptions Check (this path only):**
- ✓ Normality of residuals
- ✓ Equal variances across groups
- ✓ Independence of observations

**Output:** F-statistic, degrees of freedom, p-value

---

### 4.2B PATH B: WELCH'S ANOVA (Welch's Appropriate)

**Conditions Met:**
- Shapiro-Wilk normality test: p > 0.05
- Levene's variance homogeneity: p < 0.05
- Data approximately normal BUT variances unequal

**Test Purpose:** Test overall mean differences when variances heterogeneous

**Modification to ANOVA F-statistic:**
```
Weight each group by inverse variance:
  Wᵢ = nᵢ / sᵢ²

Welch's F-statistic:
  F_Welch = [Σᵢ Wᵢ(x̄ᵢ - x̄*)² / (k-1)] / [1 + 2(k-2)/(k²-1) × Σᵢ (1-Wᵢ/W_total)² / (nᵢ-1)]

where:
  x̄* = Σᵢ Wᵢ x̄ᵢ / W_total (weighted grand mean)
  W_total = Σᵢ Wᵢ
```

**Advantage over ANOVA:**
- Unaffected by variance heterogeneity
- More accurate Type I error control
- Same power as ANOVA when variances equal

**P-value:** Compared against F-distribution with adjusted degrees of freedom

**Output:** Welch's F-statistic, adjusted df, p-value

---

### 4.2C PATH C: KRUSKAL-WALLIS TEST (Non-parametric)

**Conditions Met:**
- Shapiro-Wilk normality test: p < 0.05
- Non-normal data detected
- Any variance pattern acceptable

**Test Purpose:** Test whether strain groups have different distributions (non-parametric)

**Procedure:**
```
Step 1: Combine all data across groups
Step 2: Rank values from 1 to N (smallest to largest)
Step 3: Calculate sum of ranks per group: Rᵢ = Σⱼ rank(xᵢⱼ)
Step 4: Compute test statistic
```

**Test Statistic:**
```
H = [12 / (N(N+1))] × [Σᵏᵢ₌₁ (Rᵢ² / nᵢ)] - 3(N+1)

where:
  N = total observations
  k = number of groups
  Rᵢ = sum of ranks for group i
  nᵢ = sample size for group i
```

**Hypotheses:**
- H₀: Distributions of all groups identical
- H₁: At least one group distribution differs

**Null Distribution:** χ² (k-1) for N > 5 (approximately)

**P-value Calculation:**
```
p-value = P(χ²(k-1) ≥ H_observed | H₀)
```

**Advantages:**
- Distribution-free (no normality required)
- Robust to outliers (based on ranks, not values)
- Can handle tied values

**Output:** H-statistic, degrees of freedom, p-value

---

### 4.3 Convergence: Generate Global P-values

**Process:**
```
FOR each metabolite assigned a test (ANOVA/Welch's/Kruskal-Wallis):
  Calculate test statistic
  Calculate p-value from appropriate null distribution
  Store result: (Metabolite, Test_Type, Statistic, p-value)
```

**Significance Threshold:** p < 0.05 (α level, family-wise)

**Output: Global Test Results Table**
```
Metabolite          | Test Type        | Statistic | p-value | Significant
Alanine            | Kruskal-Wallis   | 12.34     | 0.0087  | YES
Glucose            | ANOVA            | 8.92      | 0.0142  | YES
Lysine             | Welch's ANOVA    | 7.45      | 0.0031  | YES
Fructose           | ANOVA            | 2.31      | 0.1847  | NO
...
```

**Result Interpretation:**
- **Significant metabolites:** Exhibit strain-dependent variation
- **Non-significant metabolites:** Do not vary significantly across strains
- **Test type diversity:** Reflects heterogeneous assumption compliance

---

## PHASE 5: MULTI-LEVEL POST-HOC ANALYSIS

### 5.1 Objective
Identify specific strain comparisons and biological group signatures driving global significance while rigorously controlling for multiple testing.

### 5.2 Rationale for Multi-Level Approach

**Problem:** Global tests identify whether differences exist but not WHERE

**Solution:** Two complementary post-hoc strategies:

1. **Global Pairwise Comparisons**
   - Compares all strain pairs
   - Identifies which strains differ from which
   - More tests → stricter multiple testing correction

2. **Group-Level Comparisons**
   - Compares biological groups (E.coli, P.aeruginosa, co-cultures)
   - Tests hypothesis of group-specific metabolic signatures
   - Fewer tests → more statistical power for group hypothesis

---

### 5.3 POST-HOC TYPE 1: GLOBAL PAIRWISE COMPARISONS

### 5.3.1 Tukey HSD (For ANOVA-qualified metabolites)

**Purpose:** Pairwise all-comparisons test for parametric data

**Method:** Honestly Significant Difference (HSD)

**Test Statistic:**
```
t = (x̄ᵢ - x̄ⱼ) / √(MSwithin × (1/nᵢ + 1/nⱼ) / 2)

where:
  x̄ᵢ, x̄ⱼ = means for groups i and j
  MSwithin = pooled within-group variance from ANOVA
  nᵢ, nⱼ = sample sizes
```

**Null Distribution:** Studentized Range Distribution (q-distribution)

**Critical Value:** q_critical(α, k, N-k) 

**Comparison:**
```
IF |t| > q_critical:
  REJECT H₀ → Groups i and j differ significantly
ELSE:
  FAIL TO REJECT H₀ → No significant difference
```

**Number of Comparisons:** C(k,2) = k(k-1)/2 (typically 15-120 per metabolite)

**Multiple Testing Control:** Tukey HSD automatically controls family-wise error rate

---

### 5.3.2 Mann-Whitney U Test with FDR (For Welch's and Kruskal-Wallis)

**Purpose:** Non-parametric pairwise comparison for heterogeneous variances and non-normal data

**Procedure:**
```
FOR each pair of strains (i, j) where i < j:
  Combine data from strains i and j
  Rank all values
  
  Calculate U-statistic:
    U = n₁ × n₂ + n₁(n₁+1)/2 - R₁
    
    where R₁ = sum of ranks for group 1
```

**Test Statistic:**
```
Mann-Whitney U statistic ~ N(μU, σU²) for large samples
  μU = n₁ × n₂ / 2
  σU² = n₁ × n₂ × (n₁ + n₂ + 1) / 12
```

**Hypotheses:**
- H₀: Distributions of strains i and j are identical
- H₁: Distributions differ

**P-value:** Two-tailed probability from null distribution

**Advantages:**
- Works with unequal variances
- Robust to non-normality
- Handles ties naturally

**Output per comparison:** p-value (uncorrected)

---

### 5.4 POST-HOC TYPE 2: GROUP-LEVEL COMPARISONS

### 5.4.1 Biological Group Definition

**Rationale:** Pool clinical isolates within biological category to test group hypothesis

**Groups Defined:**
```
Group 1 (E. coli clinical isolates):
  - Samples: Tme12, Tme13, Tme14, Tme15
  - Control Reference: K12
  - Hypothesis: Clinical E.coli show distinct metabolic profile vs K12

Group 2 (P. aeruginosa clinical isolates):
  - Samples: Tmp04, Tmp05, Tmp06, Tmp07
  - Control Reference: PA01
  - Hypothesis: Clinical P.aeruginosa show distinct profile vs PA01

Group 3 (Co-culture samples):
  - Samples: Pair412, Pair513, Pair614, Pair715
  - Control Reference: PairEP
  - Hypothesis: Co-cultures show unique metabolic signature
```

### 5.4.2 Group Pooling and Testing

**Procedure:**
```
FOR each biological group:
  Step 1: Pool all metabolite values from group samples
  Step 2: Pool all metabolite values from control samples
  Step 3: Compare group vs. control using Mann-Whitney U
  
  Result: Group_vs_Control_pvalue
```

**Rationale for Mann-Whitney U:**
- Robust to variance heterogeneity between group and control
- Non-parametric (works with any distribution)
- Appropriate for small group sizes

**Number of Comparisons:** 3 (one per group) = Much fewer than global pairwise!

**Advantage:** Lower multiple testing burden = Stronger power for group hypothesis

**Output:**
```
Group              | Control Reference | U-statistic | p-value
E. coli isolates   | K12              | 124.5       | 0.0023
P.aero isolates    | PA01             | 98.2        | 0.0156
Co-cultures        | PairEP           | 67.8        | 0.0089
```

---

### 5.5 MULTIPLE TESTING CORRECTION: BENJAMINI-HOCHBERG FDR

### 5.5.1 Motivation for FDR Control

**Problem:** Multiple comparisons inflate false positive rate

**Example:**
```
If 100 independent tests at α = 0.05:
  Expected false positives = 100 × 0.05 = 5
  Even with no true effects, expect ~5 "significant" results!
  
With 1000+ comparisons across metabolites:
  Risk of false discovery very high
```

**Solution:** Control False Discovery Rate (FDR)
```
FDR = E[# False Positives / # Total Discoveries]

FDR = 0.05 means: of all "significant" findings, 
                  ~5% are expected false positives
```

**Advantage over Bonferroni:**
- Bonferroni: p_corrected = p × n (very conservative)
- FDR: More powerful while still controlling errors

### 5.5.2 Benjamini-Hochberg Algorithm

**Input:** Vector of m p-values from all comparisons

**Step 1: Sort P-values**
```
p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎  (ascending order)
```

**Step 2: Calculate FDR Threshold**
```
FOR i = m, m-1, ..., 1 (descending):
  IF p₍ᵢ₎ ≤ (i/m) × α:
    Set threshold_i = i
    Break loop
```

**Step 3: Reject H₀ for Tests 1 to threshold_i**
```
Significant metabolites: those with rank ≤ threshold_i
```

**Step 4: Calculate Adjusted P-values (Optional)**
```
FOR each test i:
  p'_adjusted(i) = min{p₍ⱼ₎ × m/j : j ≥ i}
  
(Can also report these alongside raw p-values)
```

### 5.5.3 Application Scope

**Option 1: Global Correction**
```
Correct all global pairwise comparisons together
  Total comparisons: ~1000-10000
  Adjustment factor: large
  Result: Only most extreme p-values survive
```

**Option 2: Group-Level Correction**
```
Correct only group-level comparisons
  Total comparisons: 3 per metabolite
  Adjustment factor: small
  Result: More group-level findings significant
```

**Recommended Approach:** 
- Report global pairwise results (transparently marked "uncorrected" if needed)
- Emphasize group-level results (proper FDR correction with few tests)
- Combine for comprehensive interpretation

### 5.5.4 Output Tables

**Table 1: Global Pairwise Results (with FDR)**
```
Metabolite  | Strain1 | Strain2 | p-value | p-FDR | Significant
Alanine     | K12     | PA01    | 0.0012  | 0.234 | NO
Alanine     | K12     | Tme12   | 0.0001  | 0.038 | YES
Glucose     | K12     | PA01    | 0.0234  | 0.856 | NO
...
```

**Table 2: Group-Level Results (with FDR)**
```
Metabolite  | Group         | vs. Ctrl | p-value | p-FDR | Significant
Alanine     | E.coli        | K12      | 0.0023  | 0.067 | YES
Alanine     | P.aeruginosa  | PA01     | 0.0156  | 0.156 | NO
Glucose     | Co-cultures   | PairEP   | 0.0089  | 0.089 | YES
...
```

---

## PHASE 6: VISUALIZATION & REPORTING

### 6.1 Objective
Generate publication-quality visualizations and comprehensive results documentation enabling interpretation and external validation.

### 6.2 VISUALIZATION 1: VOLCANO PLOT

**Purpose:** Combined visualization of statistical significance and biological magnitude

**Axes:**
```
X-axis: Log₂(Fold-Change) = log₂(strain_mean / control_mean)
        Range: typically -5 to +5
        
Y-axis: -log₁₀(p-value) = -log₁₀(p)
        Range: 0 to ~8 (higher = more significant)
        
Transformation: higher Y values are LOWER p-values
```

**Point Coloring:** By metabolic pathway
```
Amino Acids          → Blue
Carbohydrates        → Green
Organic Acids        → Orange
Lipids              → Purple
Nucleotides         → Red
Polyamines          → Pink
Other               → Gray
```

**Threshold Lines:**
```
Horizontal line: Y = -log₁₀(0.05)
  Above = p < 0.05 (statistically significant)
  
Vertical lines: X = ±1
  Right of +1 = at least 2-fold increase
  Left of -1 = at least 2-fold decrease
  
"Volcano" regions (upper left + upper right):
  Significant AND large fold-change = candidates for follow-up
```

**Interpretation:**
```
Upper-left quadrant:  Significant decrease, large fold-change
Upper-right quadrant: Significant increase, large fold-change
Lower quadrants:      Not statistically significant (too much noise)
```

**Example Points:**
```
Point A (X=+3, Y=4):   ✓ Highly increased, significant → KEY FINDING
Point B (X=+0.5, Y=3): Modest increase but significant
Point C (X=+5, Y=1):   Huge change but not significant → Likely noise
Point D (X=+0.1, Y=0): No change, no significance → Irrelevant
```

---

### 6.3 VISUALIZATION 2: HEATMAP

**Purpose:** Pattern recognition of metabolite changes across all strains

**Structure:**
```
Rows:    Metabolites (n ~50-100)
Columns: Strains/Samples (k ~10-20)
Cells:   Normalized metabolite abundance (transformed scale)
```

**Color Scale:**
```
Low concentration:  Blue (cool)
Medium:            White (neutral)
High concentration: Red (warm)
        
Alternative: Green-Black-Red diverging scale
```

**Data Preparation:**
```
Normalize per metabolite:
  z-score = (value - metabolite_mean) / metabolite_sd
  
Result: Each metabolite centered at 0 (green)
        Positive values (red) = above-average for that metabolite
        Negative values (blue) = below-average
```

**Hierarchical Clustering:**
```
Perform unsupervised clustering:
  - Rows: Group metabolites with similar strain patterns
  - Columns: Group strains with similar metabolite profiles
  
Result: Similar metabolites cluster together
        Similar strains cluster together
```

**Annotations:**
```
Top margin: Metabolite pathway color bars
  Each metabolite labeled with pathway color

Left margin: Metabolite names
Right margin: Global ANOVA p-values or significance marks

Column labels: Strain identifiers
```

**Interpretation:**
```
Red patch in metabolite cluster:
  → This group elevated in that strain

Blue patch:
  → This group reduced in that strain

Strain clustering:
  → Strains with similar metabolic profile group together
```

---

### 6.4 VISUALIZATION 3: BOX-VIOLIN PLOTS (Per Metabolite)

**Purpose:** Distribute details and individual data point visualization

**Components per Metabolite:**
```
Violin plot:  Kernel density estimation of distribution shape
Box plot:     Quartiles (Q1-median-Q3) and whiskers
Points:       Individual measurement values (jittered)
```

**Structure:**
```
X-axis: Strain/Sample groups
Y-axis: Metabolite abundance (transformed scale)

Separate subplot for each significant metabolite
```

**Annotations:**
```
Significance indicators above each comparison:
  *** p < 0.001
  **  p < 0.01
  *   p < 0.05
  ns  p ≥ 0.05
  
Statistical test type noted in subtitle
(ANOVA / Welch's / Kruskal-Wallis)
```

**Interpretation:**
```
Wide violin:  Metabolite shows high variability in that group
Narrow violin: Low variability (consistent level)

Outlier points: Can identify unusual samples

Visual separation: Large gap between strain distributions
                  suggests strong difference
```

---

### 6.5 VISUALIZATION 4: TRANSFORMATION COMPARISON

**Purpose:** Validate transformation optimization and show before/after

**Layout:** 2 × 3 subplot grid

**Row 1: Original (Untransformed) Data**
```
Subplot 1a: Density plot of all metabolites (original scale)
  Shape: Likely right-skewed for metabolomics

Subplot 1b: Q-Q plot of original data vs. normal distribution
  Points off diagonal indicate non-normality
  
Subplot 1c: Summary statistics
  Mean skewness
  Mean kurtosis
  % normal by Shapiro-Wilk
```

**Row 2: Transformed Data (Using Selected Method)**
```
Subplot 2a: Density plot of transformed metabolites
  Shape: More symmetric/bell-shaped
  
Subplot 2b: Q-Q plot of transformed data
  Points closer to diagonal = better normality
  
Subplot 2c: Summary statistics (post-transformation)
  Improved skewness and kurtosis
  Increased % normal
```

**Annotations:**
```
"Normality Pass Rate: 45% → 82%"
"Mean Skewness: 0.63 → 0.12"
"Transformation Method: Box-Cox (λ = 0.23)"
```

---

### 6.6 RESULTS WORKBOOK COMPILATION

**Format:** Multi-sheet Microsoft Excel (.xlsx)

**Sheet 1: ANOVA_Results**
```
Columns: Metabolite | Test_Type | Statistic | df1 | df2 | p-value | Significant
Rows:    All metabolites analyzed
```

**Sheet 2: Assumption_Results**
```
Columns: Metabolite | Normality_p | Normal? | Levene_p | EqualVar? | Test_Assigned
Rows:    All metabolites
```

**Sheet 3: PostHoc_Results** (if applicable)
```
Columns: Metabolite | Strain1 | Strain2 | p-value_raw | p-value_FDR | Test | Significant
Rows:    All significant pairwise comparisons
```

**Sheet 4: Group_PostHoc**
```
Columns: Metabolite | Biological_Group | Control_Reference | p-value | p-FDR | Significant
Rows:    Group-level comparisons (3 per metabolite)
```

**Sheet 5: FoldChange**
```
Rows:    Metabolites
Columns: All strains
Values:  Log₂(Fold-change) relative to control
```

**Sheet 6: Metabolite_Classification**
```
Columns: Metabolite | Pathway | Pathway_Color | Biological_Function
Rows:    All metabolites with functional annotation
```

**Sheet 7: Strain_Metadata**
```
Columns: Strain_ID | Type | Control_Reference | Sample_Count | Mean_Values
Rows:    All strains in analysis
```

**Sheet 8: Summary_Statistics**
```
Columns: Metabolite | Mean_Control | SD_Control | Mean_Treatment | SD_Treatment | t-statistic | log2FC
Rows:    All metabolites with basic comparison statistics
```

---

## STATISTICAL RATIONALE

### 7.1 Why Three Parallel Statistical Test Paths?

**One-Size-Fits-All Approach → PROBLEMATIC:**
- Different data distributions require different tests
- Forced parametric assumptions → Type I/II errors
- Reduced power if test assumptions violated

**Adaptive Test Selection → OPTIMAL:**
- ANOVA: When assumptions met, most powerful parametric test
- Welch's: When normality holds but variance unequal, accounts for heteroscedasticity
- Kruskal-Wallis: When distribution non-normal, rank-based approach avoids assumption violation

**Result:** Each metabolite tested by most appropriate test for its data characteristics

### 7.2 Why Both Global Pairwise AND Group-Level Post-Hoc?

**Global Pairwise:**
```
Pros:  • Answers "which specific strains differ?"
       • Comprehensive all-comparisons approach
       
Cons:  • Many tests (100-1000+)
       • Aggressive FDR correction
       • May miss subtle group patterns
```

**Group-Level:**
```
Pros:  • Tests biological hypothesis directly
       • Fewer tests → more power
       • Identifies consistent group signatures
       
Cons:  • Less granular (strain-level detail lost)
       • Requires a priori group definition
```

**Synergistic Combination:**
- Global reveals individual strain contributors
- Groups reveal consistent patterns
- Together provide both specificity and generalizability

### 7.3 Why Benjamini-Hochberg FDR Correction?

**Problem:** Multiple testing inflates false positive rate

**Bonferroni Correction (Too Conservative):**
```
p_corrected = p × n
Example: p = 0.001, n = 1000
Result: p_corrected = 1.0 (no discoveries possible!)
```

**Benjamini-Hochberg FDR (Balanced):**
```
Controls false discovery rate, not family-wise error rate
- More powerful than Bonferroni
- Still controls false discovery risk
- Appropriate for hypothesis-generating studies
- Recommended for omics data
```

**Alternative: Bonferroni Available if:**
- Very large sample size (n > 10,000)
- Already stringent significance threshold (p < 0.001)
- False positive risk unacceptable

---

## EXPECTED OUTPUTS & INTERPRETATION

### 8.1 Output Files Generated

**1. Statistical Results Workbook**
```
File: Statistical_results_{medium}_sup.xlsx
Size: 50-500 KB (depends on number of metabolites)
Sheets: 8 (as described above)
```

**2. Visualization PNG Files**
```
File: Volcano_Plot_{medium}_sup.png (300 dpi)
File: Heatmap_{medium}_sup.png (300 dpi)
File: Transformation_Comparison_{medium}.png (300 dpi)

Directory: Individual_Metabolite_Plots/
Contents: Box-violin plot per significant metabolite (PNG)
```

**3. Metadata Documentation**
```
File: Analysis_Log_{timestamp}.txt
Contents:
  - Analysis date and time
  - R/Python package versions
  - Transformation method selected
  - Test counts and significance rates
  - Quality assurance checks performed
```

---

### 8.2 Results Interpretation Framework

| Global ANOVA | Global Pairwise | Group-Level | Biological Meaning |
|---|---|---|---|
| Significant | Significant | Significant | Strong, consistent difference; group-specific pattern |
| Significant | Non-significant | Significant | Complex differences; group effect present |
| Significant | Non-significant | Non-significant | Subtle differences across many comparisons |
| Non-significant | - | - | Metabolite stable across strains; strain-independent |

### 8.3 Expected Result Scenarios

**Scenario A: Clear Strain-Specific Signature**
```
Global ANOVA:     p < 0.001 ✓ Significant
Global Pairwise:  10-20 comparisons FDR-sig
Group-Level:      One group FDR-sig (e.g., E.coli strains elevated)

Interpretation: 
  This metabolite elevated specifically in E.coli strains
  Potential biomarker for E.coli presence/activity
  Follow-up: Pathway analysis for E.coli-specific metabolism
```

**Scenario B: Distributed Differences (No Single Group)**
```
Global ANOVA:     p = 0.008 ✓ Significant
Global Pairwise:  100+ raw p-values significant but 0-2 FDR-sig
Group-Level:      0 groups FDR-significant

Interpretation:
  Metabolite varies but not in clear group pattern
  Possible: strain-to-strain variation not group-based
  Possible: Technical noise inflating individual comparisons
  Follow-up: Check for confounding variables, technical effects
```

**Scenario C: Metabolite Stable Across Strains**
```
Global ANOVA:     p = 0.45 (Not significant)
Global Pairwise:  0 comparisons significant
Group-Level:      0 groups significant

Interpretation:
  This metabolite does not vary across strains
  Likely: Essential metabolite, tightly regulated
  Implication: Not useful for strain discrimination
  Action: Exclude from downstream biomarker panels
```

---

## QUALITY ASSURANCE CHECKLIST

- ✓ Transformation optimization performed (all methods evaluated)
- ✓ Assumptions formally tested per metabolite
- ✓ Appropriate statistical test selected for each metabolite
- ✓ Multiple testing correction applied
- ✓ Both global and group-level hypotheses tested
- ✓ Visualizations generated for all significant findings
- ✓ Results documented with metadata
- ✓ Reproducibility information preserved

---

## REFERENCES

1. Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." Journal of the Royal Statistical Society, Series B, 57(1), 289-300.

2. Shapiro, S. S., & Wilk, M. B. (1965). "An analysis of variance test for normality." Biometrika, 52(3-4), 591-611.

3. Levene, H. (1960). "Robust tests for equality of variances." In Contributions to Probability and Statistics, Festschrift for Harold Hotelling, 278-292.

4. Welch, B. L. (1951). "On the comparison of several mean values: an alternative approach." Biometrika, 38(3-4), 330-336.

5. Kruskal, W. H., & Wallis, W. A. (1952). "Use of ranks in one-criterion variance analysis." Journal of the American Statistical Association, 47(260), 583-621.

6. Mann, H. B., & Whitney, D. R. (1947). "On a test of whether one of two random variables is stochastically larger than the other." Annals of Mathematical Statistics, 18(1), 50-60.

7. Box, G. E. P., & Cox, D. R. (1964). "An analysis of transformations." Journal of the Royal Statistical Society, Series B, 26(2), 211-252.

---

**Document Version:** 1.0
**Date:** December 18, 2025
**Author:** Dmytro Sokol
**For Detailed Implementation:** See CAUTI_metabolomics_supernatant.py

**End of Document**
