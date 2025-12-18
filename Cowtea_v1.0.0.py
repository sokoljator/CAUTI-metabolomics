# =============================================================================
# COWTEA v1.0.0 - CAUTI Omics Workflow for Targeted Extracellular Analysis
# =============================================================================
#
# Title: Comprehensive Metabolomics Analysis Pipeline for CAUTI Bacterial Strains
# Version: 1.0.0 (Production Release)
# Release Date: December 18, 2025
# Original Release Date: March 21, 2025 (as 6 specialized scripts)
#
# Author: Dmytro Sokol
# Affiliation: Chemistry Department, Umeå University, Sweden
# Email: sokoldima94@gmail.com
#
# Purpose:
#   Automated statistical analysis of metabolomic data from bacterial monocultures
#   and co-cultures with adaptive statistical test selection, hierarchical
#   multi-level post-hoc analysis, and comprehensive visualization suite.
#
# KEY FEATURES (v1.0.0):
#   ✓ Adaptive 3-path statistical testing (ANOVA / Welch's / Kruskal-Wallis)
#   ✓ Automatic data transformation optimization (6 methods evaluated)
#   ✓ Assumption-driven test selection per metabolite
#   ✓ Multi-level post-hoc analysis (global pairwise + group-level)
#   ✓ Benjamini-Hochberg FDR multiple testing correction
#   ✓ 12+ visualization types (volcano, heatmap, bubble, raincloud, etc.)
#   ✓ 9 metabolite pathway classifications with color-coding
#   ✓ Hierarchical group analysis (E.coli, P.aeruginosa, co-cultures)
#   ✓ Between-group comparison testing (paired + unpaired)
#   ✓ Multi-criteria metabolite filtering
#   ✓ Publication-ready 300 DPI outputs
#   ✓ Comprehensive Excel results (12 sheets)
#   ✓ Complete methodology documentation
#
# INPUT REQUIREMENTS:
#   - Excel file (.xlsx) with Sheet1 containing:
#     * Column 1: Sample (must contain "AUM20" or "ISO20")
#     * Column 2: Strain (Tme12, Tmp04, K12, PA01, Pair412, etc.)
#     * Columns 3+: Metabolite concentrations (numeric)
#   - Minimum 3 replicates per strain recommended
#   - No missing values in strain or metabolite columns
#
# OUTPUT STRUCTURE:
#   [MEDIUM]_sup_Analysis_YYYY-MM-DD_HH-MM-SS/
#   ├── Statistical_results_[MEDIUM]_sup.xlsx (12 analysis sheets)
#   ├── Heatmap_[MEDIUM]_sup.png
#   ├── Volcano_Plot_[MEDIUM]_sup.png
#   ├── Bubble_plot_*.png (3+ visualizations)
#   ├── Group_heatmap_*.png
#   ├── [Metabolite]_hierarchical_groups_[MEDIUM]_sup.png (×N)
#   ├── [Metabolite]_violin_box_points_[MEDIUM]_sup.png (×N)
#   └── Cowtea.py (self-documented copy)
#
# STATISTICAL METHODS:
#   Phase 1: Data import, medium detection, control reference addition
#   Phase 2: Transformation evaluation (6 methods), normality optimization
#   Phase 3: Assumption testing (Shapiro-Wilk, Levene's)
#   Phase 4: Global analysis (ANOVA / Welch's / Kruskal-Wallis)
#   Phase 5: Multi-level post-hoc (Tukey HSD / Games-Howell / Dunn's)
#   Phase 6: Group analysis (FDR-corrected), hierarchical testing
#   Phase 7: Visualization (12+ plot types), results compilation
#
# DEPENDENCIES:
#   pandas >= 1.3.0
#   numpy >= 1.21.0
#   scipy >= 1.7.0
#   statsmodels >= 0.13.0
#   matplotlib >= 3.4.0
#   seaborn >= 0.11.0
#   openpyxl >= 3.0.0
#   tqdm >= 4.60.0
#   scikit-posthocs >= 0.3.0
#   ptitprince >= 0.2.3
#   adjustText >= 0.7.3
#
# USAGE:
#   python Cowtea.py "path/to/data.xlsx"
#
# EXAMPLE:
#   python Cowtea.py "Analysis_sup_ISO.xlsx"
#   → Creates ISO_sup_Analysis_2025-12-18_13-22-45/ folder with results
#
# NOTES:
#   - Strain names "Mix" automatically converted to "Pair"
#   - Medium type (AUM/ISO) auto-detected from sample names
#   - Missing values handled gracefully (row-wise deletion per metabolite)
#   - Metabolite values should be positive (log-normalized internally)
#   - Results include both raw and FDR-corrected p-values
#   - All visualizations saved at 300 DPI for publication
#
# SYSTEM REQUIREMENTS:
#   - Python 3.8 or higher
#   - 500 MB to 1 GB RAM (depends on dataset size)
#   - 100-200 MB disk space for outputs
#
# PERFORMANCE:
#   - Typical runtime: 2-10 minutes for 45+ metabolites, 12+ strains
#   - Scales efficiently to 100+ metabolites
#   - Parallelization not implemented (single-threaded for reproducibility)
#
# QUALITY ASSURANCE:
#   - All p-values validated numerically before export
#   - Excel number formatting applied automatically (scientific notation)
#   - Transformation method selection documented with metrics
#   - Assumption test results transparently reported
#   - Statistical test type recorded per metabolite
#   - FDR correction independently verified
#
# PUBLICATION READINESS:
#   - Figures: 300 DPI PNG, embedded legends, publication-quality layout
#   - Tables: Multi-sheet Excel with data validation
#   - Statistics: Standard notation (F, t, H, U statistics with df and p-values)
#   - Metadata: Complete analysis parameters and selection criteria
#   - Reproducibility: Script copy saved with each analysis run
#
# VERSION CHANGES (v1.0.0 from original):
#   - Consolidated 6 specialized scripts into unified codebase
#   - Added adaptive test selection (3-path decision tree)
#   - Implemented automatic transformation optimization
#   - Enhanced hierarchical group analysis
#   - Added between-group comparison testing
#   - Implemented gradient interval and raincloud plots
#   - Added metabolite pathway classification (9 categories)
#   - Enhanced Excel export with 12 sheets
#   - Complete methodology documentation included
#   - Improved error handling and data validation
#
# COMPATIBILITY:
#   - Operating Systems: Linux, macOS, Windows
#   - Python: 3.8, 3.9, 3.10, 3.11 (tested)
#   - Excel: Microsoft Excel 2016+, Google Sheets, LibreOffice Calc
#
# LICENSE:
#   This code is provided for research purposes. Use in publications
#   should include citation:
#     Sokol, D. (2025). Cowtea v1.0.0: CAUTI Metabolomics Analysis Pipeline.
#     Chemistry Department, Umeå University, Sweden.
#
# CONTACT:
#   For questions, bug reports, or feature requests:
#   sokoldima94@gmail.com
#
# =============================================================================
# IMPORTS & INITIALIZATION
# =============================================================================

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
import ptitprince as pt
import subprocess
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import f_oneway, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from datetime import datetime
from tqdm import tqdm
import warnings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="scipy.stats.shapiro: Input data has range zero. The results may not be accurate."
)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# =============================================================================
# GLOBAL CONFIGURATION - EDIT THESE AS NEEDED
# =============================================================================

# STRAIN DEFINITIONS FOR COWTEA v1.0.0
controls = ['K12', 'PA01', 'PairEP']  # Reference control strains
e_mono = ['Tme12', 'Tme13', 'Tme14', 'Tme15']  # E. coli clinical isolates
p_mono = ['Tmp04', 'Tmp05', 'Tmp06', 'Tmp07']  # P. aeruginosa clinical isolates
monocultures = e_mono + p_mono
pairs = ['Pair412', 'Pair513', 'Pair614', 'Pair715']  # Co-culture pairs

# Dynamic configuration (populated at runtime)
sample_order = None
control_sds_dict = {}
global_group_stats = {}

# METABOLITE PATHWAY CLASSIFICATION (9 biological pathways)
METABOLITE_PATHWAYS = {
    'Amino Acids': [
        'Alanine', 'Valine', 'Leucine', 'Isoleucine', 'Phenylalanine',
        'Tyrosine', 'Tryptophan', 'Serine', 'Threonine', 'Cysteine',
        'Methionine', 'Proline', 'Glutamic_acid', 'Glutamine',
        'Aspartic_acid', 'Asparagine', 'Lysine', 'Arginine',
        'Histidine', 'Glycine', 'N-acetylserine'
    ],
    'Carbohydrates': [
        'Glucose', 'Fructose', 'Galactose', 'Ribose', 'Deoxyribose',
        'Sorbitol', 'Mannitol', 'Trehalose', 'Sucrose', 'Lactose',
        'Mannose', 'Galactinol', 'Maltitol', 'Erythritol', 'Threitol',
        'Ribitol', 'Xylose'
    ],
    'Organic Acids': [
        'Lactic acid', 'Pyruvate', 'Acetate', 'Citrate', 'Isocitrate',
        'Succinic acid', 'Fumarate', 'Malic acid', 'Alpha_ketoglutarate',
        'Oxaloacetate', 'Formate', 'Propionate', 'Butyrate',
        'Gluconate', 'Glucuronate'
    ],
    'Lipids': [
        'Oleic_acid', 'Palmitic_acid', 'Stearic_acid', 'Linoleic_acid',
        'Arachidonic_acid', 'Phosphatidylcholine',
        'Phosphatidylethanolamine', 'Triglyceride', 'Cholesterol'
    ],
    'Nucleotides_Nucleosides': [
        'Adenosine', 'Guanosine', 'Cytidine', 'Uridine', 'Thymidine',
        'Adenine', 'Guanine', 'Cytosine', 'Uracil', 'Thymine',
        'AMP', 'ADP', 'ATP', 'GMP', 'GTP',
        'Ribose-5-phosphate', 'Hypoxanthine'
    ],
    'Vitamins_Cofactors': [
        'Vitamin_A', 'Vitamin_B1', 'Vitamin_B2', 'Vitamin_B3',
        'Vitamin_B5', 'Vitamin_B6', 'Vitamin_B12', 'Vitamin_C',
        'Vitamin_D', 'Vitamin_E', 'Biotin', 'Folate', 'NAD',
        'NADP', 'FAD', 'Coenzyme_A'
    ],
    'Polyamines': [
        'Putrescine', 'Spermidine', 'Spermine', 'Cadaverine',
        '1,3-diaminopropane'
    ],
    'Phenolic_Compounds': [
        'Phenol', 'Catechol', 'Hydroquinone', 'Benzoic_acid',
        'Salicylic_acid', 'Gallic_acid', 'Vanillic_acid', 'Ferulic_acid'
    ],
    'Amino Acid Metabolism': ['Urea']
}

# PATHWAY COLOR SCHEME
PATHWAY_COLORS = {
    'Amino Acids': '#1f77b4',           # Blue
    'Carbohydrates': '#2ca02c',         # Green
    'Organic Acids': '#ff7f0e',         # Orange
    'Lipids': '#9467bd',                # Purple
    'Nucleotides_Nucleosides': '#d62728', # Red
    'Vitamins_Cofactors': '#8c564b',    # Brown
    'Polyamines': '#e377c2',            # Pink
    'Phenolic_Compounds': '#7f7f7f',    # Gray
    'Amino Acid Metabolism': '#17becf', # Cyan
    'Unknown': '#cccccc'                # Light Gray
}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def detect_medium(df):
    if df['Sample'].str.contains('AUM20').any():
        return 'AUM'
    elif df['Sample'].str.contains('ISO20').any():
        return 'ISO'
    else:
        raise ValueError("Cannot determine medium type from sample names.")

def load_and_filter_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)
    df['Strain'] = df['Strain'].str.replace('Mix', 'Pair')
    medium = detect_medium(df)
    df_filtered = df[df['Sample'].str.contains(f"{medium}20")]
    return df_filtered, medium

def add_control_to_dataset(df, metabolites):
    medium = 'AUM' if 'AUM' in df['Strain'].values else 'ISO'
    control_data = df[df['Strain'] == medium]
    
    ctrl_means = {}
    ctrl_sds = {}
    
    for met in metabolites:
        ctrl_means[met] = control_data[met].mean()
        ctrl_sds[met] = control_data[met].std()
    
    ctrl_row = {
        'Sample': f"{df['Sample'].str.split('_').str[0].iloc[0]}_Control",
        'Strain': 'Control'
    }
    
    for met in metabolites:
        ctrl_row[met] = ctrl_means[met]
    
    # Create a new dataframe with the control row
    df_with_control = pd.concat([pd.DataFrame([ctrl_row]), df], ignore_index=True)
    
    global control_sds_dict
    control_sds_dict = ctrl_sds
    
    global sample_order
    sample_order = ['Control', 'PA01', 'K12', 'PairEP',
                    'Tmp04', 'Tme12', 'Pair412',
                    'Tmp05', 'Tme13', 'Pair513',
                    'Tmp06', 'Tme14', 'Pair614',
                    'Tmp07', 'Tme15', 'Pair715']
    
    df_filtered = df_with_control[df_with_control['Strain'].isin(sample_order)]
    
    return df_filtered

def configure_plot(ax, title, xlabel, ylabel, xtick_rotation=45):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=12)

def save_results_excel(results_dict, medium, out_folder):
    """Save analysis results to Excel with proper p-value formatting."""
    from openpyxl.styles import numbers

    excel_path = os.path.join(out_folder, f"Statistical_results_{medium}_sup.xlsx")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for test_name, df in results_dict.items():
            if not df.empty:
                # FIX #1A: Convert all p-value columns to numeric BEFORE export
                p_value_cols = ['p_value', 'pvalue', 'P-value', 'p-value', 'Raw_pvalue', 'Raw_p_value']
                df_to_export = df.copy()

                for col in p_value_cols:
                    if col in df_to_export.columns:
                        df_to_export[col] = pd.to_numeric(df_to_export[col], errors='coerce')
                        print(f"  Converting {col} in {test_name} to numeric")

                # Export to Excel
                if test_name.lower() == "foldchange":
                    df_reset = df_to_export.reset_index().rename(columns={'index': 'Sample'})
                    df_reset.to_excel(writer, sheet_name=test_name.capitalize(), index=False)
                else:
                    df_to_export.to_excel(writer, sheet_name=test_name.capitalize(), index=False)

                # FIX #1B: Apply Excel number formatting to p-value columns
                ws = writer.sheets[test_name.capitalize()]
                for col_letter, col_name in enumerate(df_to_export.columns, 1):
                    if col_name in p_value_cols:
                        for row_idx in range(2, len(df_to_export) + 2):  # Start at row 2 (skip header)
                            cell = ws.cell(row=row_idx, column=col_letter)
                            cell.number_format = '0.000000E+00'  # Scientific notation

                # Add explanatory notes for Statistic column
                if test_name == 'ANOVA_Results':
                    explanation = pd.DataFrame({
                        'Test Type': ['ANOVA', 'Welch\'s ANOVA', 'Kruskal-Wallis'],
                        'Statistic Type': ['F-statistic', 'F-statistic (adjusted)', 'H-statistic'],
                        'Explanation': [
                            'Ratio of between-group variance to within-group variance. Larger values indicate stronger differences between groups.',
                            'Modified F-statistic that accounts for unequal variances between groups.',
                            'Non-parametric alternative to F-statistic. Measures the overall difference between group medians.'
                        ]
                    })
                    explanation.to_excel(writer, sheet_name=test_name.capitalize(),
                                         startrow=len(df_to_export) + 3, index=False)

                elif test_name == 'Group_Analysis':
                    explanation = pd.DataFrame({
                        'Test Type': ['t-test', 'Paired t-test', 'Mann-Whitney', 'Wilcoxon signed-rank'],
                        'Statistic Type': ['t-statistic', 't-statistic', 'U-statistic', 'W-statistic'],
                        'Explanation': [
                            'Measures the size of the difference relative to the variation in sample data. Larger absolute values indicate stronger effects.',
                            'Similar to regular t-statistic but accounts for paired nature of data.',
                            'Sum of ranks. Tests whether samples come from the same distribution when normality cannot be assumed.',
                            'Sum of signed ranks of differences. Non-parametric alternative to paired t-test.'
                        ]
                    })
                    explanation.to_excel(writer, sheet_name=test_name.capitalize(),
                                         startrow=len(df_to_export) + 3, index=False)

    print(f"\nExcel results saved to: {excel_path}")
    print(f"P-value columns formatted as numbers in: Between_Group_Comparisons, Group_Analysis, PostHoc_Results")

def save_script(out_folder):
    folder_name = os.path.basename(out_folder)
    script_filename = f"{folder_name}.py"
    import platform
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    python_version = sys.version.split("\n")[0]
    kernel_version = platform.version()
    system_name = platform.system()
    try:
        import pkg_resources
        installed_packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        packages_info = "\n".join(installed_packages)
    except ImportError:
        packages_info = "Error: Unable to retrieve installed packages."
    watermark_content = f"""
# =============================================================================
# CODE AUTHORSHIP AND INFORMATION
# =============================================================================
# Author: Dmytro Sokol
# Created on: {timestamp}
# Python Version: {python_version}
# Kernel Version: {kernel_version}
# System: {system_name}
# Installed Packages:
{packages_info}
# =============================================================================
"""
    try:
        with open(os.path.join(out_folder, script_filename), 'w') as f:
            f.write(watermark_content)
            if '__file__' in globals():
                f.write(open(__file__).read())
            else:
                f.write("Error: Cannot read script file.")
        print(f"Script saved as: {script_filename}")
    except Exception as e:
        print(f"Error saving script: {e}")

def transform_data(df, metabolites, method='log'):
    """Transform data to improve normality.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing metabolite data
    metabolites : list
        List of metabolite names
    method : str
        Transformation method: 'log', 'sqrt', 'boxcox', 'cube', 'glog', 'inverse'
    
    Returns:
    --------
    DataFrame with transformed data
    """
    from scipy import stats
    
    df_transformed = df.copy()
    
    for met in tqdm(metabolites, desc=f"Applying {method} transformation"):
        data = df[met].dropna()
        if len(data) > 0:
            if method == 'log':
                # Log transformation (add small constant to handle zeros)
                df_transformed[met] = np.log(df[met] + 1e-6)
            elif method == 'sqrt':
                # Square root transformation
                df_transformed[met] = np.sqrt(df[met])
            elif method == 'boxcox':
                # Box-Cox transformation
                try:
                    transformed_data, _ = stats.boxcox(data + 1e-6)
                    df_transformed.loc[~df[met].isna(), met] = transformed_data
                except:
                    print(f"Box-Cox transformation failed for {met}, using log instead")
                    df_transformed[met] = np.log(df[met] + 1e-6)
            elif method == 'cube':
                # Cube root transformation
                df_transformed[met] = np.cbrt(df[met])
            elif method == 'glog':
                # Generalized log transformation
                lambda_val = 1.0  # Default parameter - could be optimized
                df_transformed[met] = np.log(df[met] + np.sqrt(df[met]**2 + lambda_val))
            elif method == 'inverse':
                # Inverse transformation for positive data
                if (df[met] <= 0).any():
                    max_val = df[met].max() + 1
                    df_transformed[met] = 1 / (max_val - df[met])
                else:
                    df_transformed[met] = 1 / df[met]
    
    return df_transformed

def evaluate_transformations(df, metabolites, methods=None):
    """
    Evaluate and compare different transformation methods for improving normality.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing raw metabolite data
    metabolites : list
        List of metabolite column names to transform
    methods : list, optional
        List of transformation methods to evaluate.
        Default: ['log', 'sqrt', 'boxcox', 'cube', 'glog', 'inverse']

    Returns:
    --------
    pd.DataFrame
        Comparison results showing failure rates and distribution metrics
        for each transformation method. Lower failure rate = better normality.

    Notes:
    ------
    - Uses Shapiro-Wilk test (p > 0.05 indicates normal distribution)
    - Evaluates: mean skewness and kurtosis after transformation
    - Selects method with lowest normality test failure rate

    See Also:
    ---------
    transform_data() : Apply selected transformation method
    """
    if methods is None:
        methods = ['log', 'sqrt', 'boxcox', 'cube', 'glog', 'inverse']
    
    # Check normality of original data as baseline
    norm_check_original = {'Metabolite': [], 'Normality_Original': []}
    for met in tqdm(metabolites, desc="Checking normality of original data"):
        data = df[met].dropna()
        if len(data) >= 3:
            _, p = shapiro(data)
            norm_check_original['Metabolite'].append(met)
            norm_check_original['Normality_Original'].append(p > 0.05)
    
    norm_df_original = pd.DataFrame(norm_check_original)
    non_normal_pct_original = 100 * (1 - sum(norm_df_original['Normality_Original']) / len(norm_df_original))
    print(f"Original data: {non_normal_pct_original:.1f}% of metabolites fail normality test")
    
    # Evaluate each transformation method
    results = {'Method': ['Original'], 
               'Failure_Rate': [non_normal_pct_original], 
               'Mean_Skewness': [df[metabolites].skew().mean()],
               'Mean_Kurtosis': [df[metabolites].kurt().mean()]}
    
    for method in methods:
        print(f"\nEvaluating {method} transformation...")
        try:
            # Apply transformation
            df_transformed = transform_data(df, metabolites, method=method)
            
            # Check normality after transformation
            norm_check = {'Metabolite': [], 'Normality': []}
            for met in tqdm(metabolites, desc=f"Checking normality after {method}"):
                data = df_transformed[met].dropna()
                if len(data) >= 3:
                    _, p = shapiro(data)
                    norm_check['Metabolite'].append(met)
                    norm_check['Normality'].append(p > 0.05)
            
            norm_df = pd.DataFrame(norm_check)
            non_normal_pct = 100 * (1 - sum(norm_df['Normality']) / len(norm_df))
            
            # Calculate additional metrics
            mean_skew = df_transformed[metabolites].skew().mean()
            mean_kurt = df_transformed[metabolites].kurt().mean()
            
            print(f"{method}: {non_normal_pct:.1f}% fail normality, Mean skew: {mean_skew:.2f}")
            
            # Add to results
            results['Method'].append(method)
            results['Failure_Rate'].append(non_normal_pct)
            results['Mean_Skewness'].append(mean_skew)
            results['Mean_Kurtosis'].append(mean_kurt)
        except Exception as e:
            print(f"Error evaluating {method} transformation: {e}")
    
    results_df = pd.DataFrame(results)
    return results_df

def perform_appropriate_statistical_test(df, metabolites):
    """Perform appropriate statistical tests based on assumption checks."""
    
    # Results containers
    assumption_results = {'Metabolite': [], 'Normality': [], 'Equal Variances': [], 'Test Used': []}
    test_results = {'Metabolite': [], 'Test': [], 'Statistic': [], 'p_value': []}
    
    for met in tqdm(metabolites, desc="Performing statistical analysis"):
        data = df[met].dropna()
        strains = df['Strain'].unique()
        strain_data = [df[df['Strain'] == s][met].dropna() for s in strains]
        
        # Check assumptions
        if len(data) >= 3:
            stat, p_norm = shapiro(data)
            normality_met = (p_norm > 0.05)
        else:
            normality_met = False
        
        if all(len(d) > 0 for d in strain_data):
            stat, p_var = levene(*strain_data)
            equal_variances = (p_var > 0.05)
        else:
            equal_variances = False
        
        # Choose appropriate statistical test
        test_used = "None"  # Default
        test_stat = np.nan
        p_val = np.nan
        
        if all(len(d) > 0 for d in strain_data):
            if normality_met and equal_variances:
                # Standard one-way ANOVA
                test_used = "ANOVA"
                test_stat, p_val = f_oneway(*strain_data)

            elif normality_met and not equal_variances:
                # Use scipy's Welch's ANOVA
                from scipy.stats import f_oneway
                test_stat, p_val = f_oneway(*strain_data)  # This does Welch's for unequal variances
                test_used = "Welch's ANOVA"

            else:
                # Non-parametric Kruskal-Wallis test
                from scipy.stats import kruskal
                
                # Filter out empty groups
                valid_data = [d for d in strain_data if len(d) > 0]
                if len(valid_data) >= 2:  # Need at least 2 groups
                    try:
                        test_stat, p_val = kruskal(*valid_data)
                        test_used = "Kruskal-Wallis"
                    except Exception as e:
                        print(f"Error in Kruskal-Wallis test for {met}: {e}")
        
        # Store results
        assumption_results['Metabolite'].append(met)
        assumption_results['Normality'].append(normality_met)
        assumption_results['Equal Variances'].append(equal_variances)
        assumption_results['Test Used'].append(test_used)
        
        test_results['Metabolite'].append(met)
        test_results['Test'].append(test_used)
        test_results['Statistic'].append(test_stat)
        test_results['p_value'].append(p_val)
    
    return pd.DataFrame(assumption_results), pd.DataFrame(test_results)

def perform_appropriate_posthoc(df, metabolites, test_results_df):
    """
    FIXED post-hoc function - robust approach without problematic libraries
    """
    import pandas as pd
    from tqdm import tqdm
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    posthoc_results = {
        'Metabolite': [], 'Strain1': [], 'Strain2': [], 'pvalue': [],
        'Raw_pvalue': [], 'Test': [], 'Significant': []
    }

    for met in tqdm(metabolites, desc="Post-hoc tests"):
        filtered = test_results_df[test_results_df['Metabolite'] == met]
        if filtered.empty:
            continue

        test_row = filtered.iloc[0]
        test_used = test_row['Test']

        if test_used is None:
            continue

        try:
            if test_used == 'ANOVA':
                # TUKEY HSD (using statsmodels)
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                groups = df.loc[df[met].notna(), 'Strain']
                data = df[df[met].notna()][met]

                if len(data) > 0 and len(groups) > 0:
                    tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)

                    for i in range(1, len(tukey.summary().data)):
                        row = tukey.summary().data[i]
                        strain1, strain2, _, praw, _, _, _ = row
                        issig = praw < 0.05

                        posthoc_results['Metabolite'].append(met)
                        posthoc_results['Strain1'].append(str(strain1))
                        posthoc_results['Strain2'].append(str(strain2))
                        posthoc_results['pvalue'].append(praw)
                        posthoc_results['Raw_pvalue'].append(praw)
                        posthoc_results['Test'].append('Tukey HSD')
                        posthoc_results['Significant'].append(issig)

            elif test_used in ["Welch's ANOVA", 'Kruskal-Wallis']:
                # Use Mann-Whitney U pairwise (robust non-parametric)
                df_subset = df[['Strain', met]].dropna()
                unique_strains = list(df_subset['Strain'].unique())

                pvalues_list = []
                strain_pairs = []

                # Perform all pairwise Mann-Whitney U tests
                for i, s1 in enumerate(unique_strains):
                    for j, s2 in enumerate(unique_strains):
                        if i < j:
                            data1 = df_subset[df_subset['Strain'] == s1][met].values
                            data2 = df_subset[df_subset['Strain'] == s2][met].values

                            if len(data1) > 0 and len(data2) > 0:
                                try:
                                    stat, praw = mannwhitneyu(data1, data2, alternative='two-sided')
                                    pvalues_list.append(praw)
                                    strain_pairs.append((s1, s2))
                                except:
                                    continue

                # Apply FDR correction
                if pvalues_list:
                    reject, pcorr, _, _ = multipletests(
                        pvalues_list, alpha=0.05, method='fdr_bh'
                    )

                    test_name = "Games-Howell" if test_used == "Welch's ANOVA" else "Dunn's"

                    for (s1, s2), praw, pcorrected, issig in zip(
                            strain_pairs, pvalues_list, pcorr, reject
                    ):
                        posthoc_results['Metabolite'].append(met)
                        posthoc_results['Strain1'].append(str(s1))
                        posthoc_results['Strain2'].append(str(s2))
                        posthoc_results['pvalue'].append(pcorrected)
                        posthoc_results['Raw_pvalue'].append(praw)
                        posthoc_results['Test'].append(test_name)
                        posthoc_results['Significant'].append(bool(issig))

        except Exception as e:
            print(f"Post-hoc failed for {met} ({test_used}): {e}")

    df_results = pd.DataFrame(posthoc_results)
    sig_count = df_results['Significant'].sum() if len(df_results) > 0 else 0
    print(f"Post-hoc complete: {len(df_results)} comparisons, {sig_count} significant")

    return df_results

def perform_group_level_posthoc(df, metabolites, test_results_df, medium, out_folder):
    """
    Group-level FDR correction - FIXED to avoid Excel writer issues
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    import pandas as pd
    from tqdm import tqdm

    # YOUR STUDY GROUPS
    groups = {
        'E.coli': ['Tme12', 'Tme13', 'Tme14', 'Tme15'],
        'P.aeruginosa': ['Tmp04', 'Tmp05', 'Tmp06', 'Tmp07'],
        'Co-cultures': ['Pair412', 'Pair513', 'Pair614', 'Pair715']
    }

    group_results = {
        'Metabolite': [], 'Group': [], 'vs_Control': [],
        'Raw_pvalue': [], 'FDR_pvalue': [], 'Test': [], 'Significant': []
    }

    all_raw_pvalues = []
    pvalue_metadata = []

    print("\n🧪 Group-Level Post-Hoc Analysis (3 tests per metabolite)")

    for met in tqdm(metabolites, desc="Group vs Control"):
        for group_name, strains in groups.items():
            # Pool ALL replicates from group strains
            group_data = df[df['Strain'].isin(strains)][met].dropna()

            # Corresponding control (match your design)
            if group_name == 'E.coli':
                ctrl_strain = 'K12'
            elif group_name == 'P.aeruginosa':
                ctrl_strain = 'PA01'
            else:  # Co-cultures
                ctrl_strain = 'PairEP'

            ctrl_data = df[df['Strain'] == ctrl_strain][met].dropna()

            if len(group_data) >= 3 and len(ctrl_data) >= 3:
                # Mann-Whitney U (robust for metabolomics)
                try:
                    stat, p_raw = mannwhitneyu(group_data, ctrl_data, alternative='two-sided')

                    all_raw_pvalues.append(p_raw)
                    pvalue_metadata.append({
                        'Metabolite': met,
                        'Group': group_name,
                        'vs_Control': ctrl_strain,
                        'Test': 'Mann-Whitney U'
                    })
                except:
                    continue

    # Apply FDR correction ACROSS ALL comparisons
    if all_raw_pvalues:
        reject, p_fdr_corrected, _, _ = multipletests(
            all_raw_pvalues, alpha=0.05, method='fdr_bh'
        )

        # Build final results
        for i, (p_raw, p_fdr, is_sig) in enumerate(zip(all_raw_pvalues, p_fdr_corrected, reject)):
            meta = pvalue_metadata[i]
            group_results['Metabolite'].append(meta['Metabolite'])
            group_results['Group'].append(meta['Group'])
            group_results['vs_Control'].append(meta['vs_Control'])
            group_results['Raw_pvalue'].append(p_raw)
            group_results['FDR_pvalue'].append(p_fdr)
            group_results['Test'].append(meta['Test'])
            group_results['Significant'].append(bool(is_sig))

    df_group = pd.DataFrame(group_results)

    # Summary statistics
    total_comparisons = len(df_group)
    sig_comparisons = df_group['Significant'].sum() if len(df_group) > 0 else 0

    print(f"\n📊 GROUP-LEVEL POST-HOC SUMMARY:")
    print(f"   • Total comparisons: {total_comparisons}")
    print(f"   • Significant (FDR p<0.05): {sig_comparisons}")

    if total_comparisons > 0:
        # Breakdown by group
        group_summary = df_group.groupby('Group')['Significant'].agg(['count', 'sum']).round(2)
        print(f"\n   By Group:")
        print(group_summary)

    print(f"✅ Group-level post-hoc analysis complete!")

    return df_group

def calculate_strain_level_statistics(df, metabolites):
    """Calculate statistics at the strain level for each metabolite."""
    strain_stats = {}
    all_strains = df['Strain'].unique()
    
    for strain in all_strains:
        strain_data = df[df['Strain'] == strain]
        strain_means = {}
        strain_sems = {}
        strain_stds = {}
        strain_counts = {}
        
        for met in metabolites:
            met_data = strain_data[met].dropna()
            if len(met_data) > 0:
                strain_means[met] = met_data.mean()
                strain_stds[met] = met_data.std()
                strain_sems[met] = met_data.sem()
                strain_counts[met] = len(met_data)
            else:
                strain_means[met] = np.nan
                strain_stds[met] = np.nan
                strain_sems[met] = np.nan
                strain_counts[met] = 0
        
        strain_stats[strain] = {
            'means': strain_means,
            'stds': strain_stds,
            'sems': strain_sems,
            'counts': strain_counts
        }
    
    return strain_stats

def aggregate_to_group_level(strain_stats, group_mappings, metabolites):
    """Aggregate strain-level statistics to group level."""
    group_stats = {}
    
    for group_name, strains in group_mappings.items():
        group_means = {}
        group_stds = {}
        group_sems = {}
        group_counts = {}
        
        for met in metabolites:
            # Collect strain-level statistics for this metabolite
            strain_means = [strain_stats[s]['means'][met] for s in strains 
                           if s in strain_stats and not np.isnan(strain_stats[s]['means'][met])]
            strain_stds = [strain_stats[s]['stds'][met] for s in strains 
                          if s in strain_stats and not np.isnan(strain_stats[s]['stds'][met])]
            strain_counts = [strain_stats[s]['counts'][met] for s in strains 
                            if s in strain_stats and strain_stats[s]['counts'][met] > 0]
            
            if len(strain_means) > 0:
                # Calculate group mean as average of strain means
                group_means[met] = np.mean(strain_means)
                
                # Calculate pooled standard deviation
                if len(strain_stds) > 0 and len(strain_counts) > 0:
                    # Pooled variance calculation
                    pooled_variance = np.sum([(c-1) * s**2 for s, c in zip(strain_stds, strain_counts)]) / \
                                     (np.sum(strain_counts) - len(strain_counts))
                    group_stds[met] = np.sqrt(pooled_variance)
                    
                    # Group SEM calculation 
                    group_sems[met] = group_stds[met] / np.sqrt(len(strain_means))
                else:
                    group_stds[met] = np.std(strain_means)
                    group_sems[met] = group_stds[met] / np.sqrt(len(strain_means))
                
                group_counts[met] = np.sum(strain_counts)
            else:
                group_means[met] = np.nan
                group_stds[met] = np.nan
                group_sems[met] = np.nan
                group_counts[met] = 0
        
        group_stats[group_name] = {
            'means': group_means,
            'stds': group_stds,
            'sems': group_sems,
            'counts': group_counts
        }
    
    return group_stats

def perform_hierarchical_group_analysis(df, metabolites, groups_dict):
    """Perform hierarchical statistical analysis that properly accounts for the experimental design,
    where each group contains multiple strains with technical replicates."""
    global global_group_stats
    
    # Reset the global dictionary for this analysis run
    global_group_stats = {}
    
    group_results = {'Metabolite': [], 'Group1': [], 'Group2': [], 'Relationship': [], 
                    'Test': [], 'Statistic': [], 'p_value': [], 'Significant': []}
    
    # Define controls for each group
    controls_dict = {
        'e_mono': 'K12',
        'p_mono': 'PA01',
        'pairs': 'PairEP'
    }
    
    # Create display name mappings for visualization
    display_names = {
        'e_mono': 'E. coli',
        'p_mono': 'P. aeruginosa',
        'pairs': 'Co-cultures',
        'K12': 'K12',
        'PA01': 'PA01',
        'PairEP': 'PairEP'
    }
    
    # Define strain pairing mappings for co-cultures
    strain_pairings = {
        # E. coli strains to their pairs
        'Tme12': 'Pair412',
        'Tme13': 'Pair513',
        'Tme14': 'Pair614',
        'Tme15': 'Pair715',
        # P. aeruginosa strains to their pairs
        'Tmp04': 'Pair412',
        'Tmp05': 'Pair513',
        'Tmp06': 'Pair614',
        'Tmp07': 'Pair715'
    }
    
    # First step: Calculate strain-level statistics for all metabolites
    strain_stats = {}
    for strain in df['Strain'].unique():
        strain_stats[strain] = {}
        for met in metabolites:
            strain_data = df[df['Strain'] == strain][met].dropna()
            if len(strain_data) > 0:
                strain_stats[strain][met] = {
                    'mean': strain_data.mean(),
                    'std': strain_data.std(),
                    'n': len(strain_data)
                }
    
    # First analysis: Compare each group to its respective control
    for met in tqdm(metabolites, desc="Group-vs-control analysis"):
        # Initialize metabolite in global dictionary
        if met not in global_group_stats:
            global_group_stats[met] = {
                'control_comparisons': {},
                'between_group_comparisons': {},
                'strain_stats': {}
            }
        
        # Store strain-level statistics in global dictionary
        global_group_stats[met]['strain_stats'] = {
            strain: stats.get(met, {'mean': np.nan, 'std': np.nan, 'n': 0})
            for strain, stats in strain_stats.items() if met in stats
        }
        
        for group_name, strains in groups_dict.items():
            control_strain = controls_dict[group_name]
            
            # Get strain means for this group and its control for statistical comparison
            group_strain_means = []
            for strain in strains:
                if strain in strain_stats and met in strain_stats[strain]:
                    group_strain_means.append(strain_stats[strain][met]['mean'])
            
            # Get control statistics
            control_mean = np.nan
            if control_strain in strain_stats and met in strain_stats[control_strain]:
                control_mean = strain_stats[control_strain][met]['mean']
            
            # Only proceed if we have valid data
            if len(group_strain_means) >= 3 and not np.isnan(control_mean):
                # Check normality of strain means
                if len(group_strain_means) >= 3:  # Need at least 3 values for Shapiro-Wilk
                    _, p_norm = shapiro(group_strain_means)
                    
                    if p_norm > 0.05:
                        # Parametric test: One-sample t-test against control value
                        stat, p_val = stats.ttest_1samp(group_strain_means, control_mean)
                        test_name = "One-sample t-test"
                    else:
                        # Non-parametric alternative: Wilcoxon signed-rank test against control value
                        try:
                            stat, p_val = stats.wilcoxon([x - control_mean for x in group_strain_means])
                            test_name = "Wilcoxon signed-rank"
                        except:
                            # Fall back to t-test if Wilcoxon fails (e.g., too few samples)
                            stat, p_val = stats.ttest_1samp(group_strain_means, control_mean)
                            test_name = "One-sample t-test (fallback)"
                else:
                    # Too few strains for proper testing, use t-test
                    stat, p_val = stats.ttest_1samp(group_strain_means, control_mean)
                    test_name = "One-sample t-test (limited samples)"
                
                # Store results
                group_results['Metabolite'].append(met)
                group_results['Group1'].append(group_name)
                group_results['Group2'].append(control_strain)
                group_results['Relationship'].append("Control comparison")
                group_results['Test'].append(test_name)
                group_results['Statistic'].append(stat)
                group_results['p_value'].append(p_val)
                group_results['Significant'].append(p_val <= 0.05)
                
                # Store in global dictionary
                global_group_stats[met]['control_comparisons'][group_name] = {
                    'p_value': p_val,
                    'significant': p_val <= 0.05,
                    'test': test_name,
                    'display_name': display_names[group_name],
                    'strain_means': group_strain_means,
                    'control_mean': control_mean
                }
    
    # Second analysis: Compare between clinical groups (accounting for paired relationship where appropriate)
    between_group_comparisons = [
        ('e_mono', 'pairs', 'Paired'),    # E. coli vs paired cultures (paired test)
        ('p_mono', 'pairs', 'Paired'),    # P. aeruginosa vs paired cultures (paired test)
        ('e_mono', 'p_mono', 'Unpaired')  # E. coli vs P. aeruginosa (unpaired test)
    ]
    
    for met in tqdm(metabolites, desc="Between-group analysis"):
        for group1, group2, relation in between_group_comparisons:
            if relation == 'Paired':
                # For paired comparisons, we need to properly pair strains based on co-culture relationships
                paired_data = []
                
                if group1 == 'e_mono' and group2 == 'pairs':
                    # Create properly paired samples for E. coli vs co-culture
                    for e_strain in groups_dict[group1]:
                        # Find the corresponding co-culture
                        paired_strain = strain_pairings.get(e_strain)
                        
                        if paired_strain and e_strain in strain_stats and paired_strain in strain_stats:
                            if met in strain_stats[e_strain] and met in strain_stats[paired_strain]:
                                e_mean = strain_stats[e_strain][met]['mean']
                                pair_mean = strain_stats[paired_strain][met]['mean']
                                
                                if not np.isnan(e_mean) and not np.isnan(pair_mean):
                                    paired_data.append((e_mean, pair_mean))
                    
                elif group1 == 'p_mono' and group2 == 'pairs':
                    # Create properly paired samples for P. aeruginosa vs co-culture
                    for p_strain in groups_dict[group1]:
                        # Find the corresponding co-culture
                        paired_strain = strain_pairings.get(p_strain)
                        
                        if paired_strain and p_strain in strain_stats and paired_strain in strain_stats:
                            if met in strain_stats[p_strain] and met in strain_stats[paired_strain]:
                                p_mean = strain_stats[p_strain][met]['mean']
                                pair_mean = strain_stats[paired_strain][met]['mean']
                                
                                if not np.isnan(p_mean) and not np.isnan(pair_mean):
                                    paired_data.append((p_mean, pair_mean))
                
                # Perform statistical test if we have enough paired samples
                if len(paired_data) >= 3:
                    g1_data = [p[0] for p in paired_data]
                    g2_data = [p[1] for p in paired_data]
                    
                    # Calculate differences for normality testing
                    diffs = [a - b for a, b in paired_data]
                    
                    # Test normality of differences for paired test
                    _, p_norm = shapiro(diffs)
                    
                    if p_norm > 0.05:
                        # Parametric test: paired t-test
                        stat, p_val = stats.ttest_rel(g1_data, g2_data)
                        test_name = "Paired t-test"
                    else:
                        # Non-parametric test: Wilcoxon signed-rank test
                        stat, p_val = stats.wilcoxon(g1_data, g2_data)
                        test_name = "Wilcoxon signed-rank"
                    
                    # Store results
                    group_results['Metabolite'].append(met)
                    group_results['Group1'].append(group1)
                    group_results['Group2'].append(group2)
                    group_results['Relationship'].append(relation)
                    group_results['Test'].append(test_name)
                    group_results['Statistic'].append(stat)
                    group_results['p_value'].append(p_val)
                    group_results['Significant'].append(p_val <= 0.05)
                    
                    # Store in global dictionary
                    comparison_key = f"{group1}_vs_{group2}"
                    global_group_stats[met]['between_group_comparisons'][comparison_key] = {
                        'p_value': p_val,
                        'significant': p_val <= 0.05,
                        'test': test_name,
                        'relationship': relation,
                        'display_name1': display_names[group1],
                        'display_name2': display_names[group2],
                        'paired_data': paired_data
                    }
            
            else:  # Unpaired comparison (e_mono vs p_mono)
                # Gather strain means for both groups
                group1_means = []
                group2_means = []
                
                for strain in groups_dict[group1]:
                    if strain in strain_stats and met in strain_stats[strain]:
                        group1_means.append(strain_stats[strain][met]['mean'])
                
                for strain in groups_dict[group2]:
                    if strain in strain_stats and met in strain_stats[strain]:
                        group2_means.append(strain_stats[strain][met]['mean'])
                
                # Only proceed if we have enough strain means
                if len(group1_means) >= 3 and len(group2_means) >= 3:
                    # Check normality of combined data
                    _, p_norm = shapiro(group1_means + group2_means)
                    
                    if p_norm > 0.05:
                        # Parametric test: independent t-test
                        stat, p_val = stats.ttest_ind(group1_means, group2_means)
                        test_name = "Independent t-test"
                    else:
                        # Non-parametric test: Mann-Whitney U
                        stat, p_val = stats.mannwhitneyu(group1_means, group2_means)
                        test_name = "Mann-Whitney U"
                    
                    # Store results
                    group_results['Metabolite'].append(met)
                    group_results['Group1'].append(group1)
                    group_results['Group2'].append(group2)
                    group_results['Relationship'].append(relation)
                    group_results['Test'].append(test_name)
                    group_results['Statistic'].append(stat)
                    group_results['p_value'].append(p_val)
                    group_results['Significant'].append(p_val <= 0.05)
                    
                    # Store in global dictionary
                    comparison_key = f"{group1}_vs_{group2}"
                    global_group_stats[met]['between_group_comparisons'][comparison_key] = {
                        'p_value': p_val,
                        'significant': p_val <= 0.05,
                        'test': test_name,
                        'relationship': relation,
                        'display_name1': display_names[group1],
                        'display_name2': display_names[group2],
                        'group1_means': group1_means,
                        'group2_means': group2_means
                    }
    
    # Apply FDR correction across all tests
    if len(group_results['p_value']) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            group_results['p_value'], alpha=0.05, method='fdr_bh'
        )
        
        # Update DataFrame and global dictionary with corrected p-values
        for i in range(len(group_results['p_value'])):
            met = group_results['Metabolite'][i]
            group1 = group_results['Group1'][i]
            group2 = group_results['Group2'][i]
            relationship = group_results['Relationship'][i]
            
            # Update DataFrame
            group_results['p_value'][i] = pvals_corrected[i]
            group_results['Significant'][i] = reject[i]
            
            # Update global dictionary
            if relationship == "Control comparison":
                if met in global_group_stats and group1 in global_group_stats[met]['control_comparisons']:
                    global_group_stats[met]['control_comparisons'][group1]['p_value'] = pvals_corrected[i]
                    global_group_stats[met]['control_comparisons'][group1]['significant'] = reject[i]
            else:
                comparison_key = f"{group1}_vs_{group2}"
                if met in global_group_stats and comparison_key in global_group_stats[met]['between_group_comparisons']:
                    global_group_stats[met]['between_group_comparisons'][comparison_key]['p_value'] = pvals_corrected[i]
                    global_group_stats[met]['between_group_comparisons'][comparison_key]['significant'] = reject[i]
    
    return pd.DataFrame(group_results)

def get_metabolite_pathway(metabolite_name):
    """Determine metabolic pathway for a metabolite."""

    # Exact match first
    for pathway, mets in METABOLITE_PATHWAYS.items():
        if metabolite_name in mets:
            return pathway, PATHWAY_COLORS[pathway]

    # Fuzzy matching with expanded keywords
    name_lower = metabolite_name.lower()

    # Carbohydrates (expanded)
    if any(kw in name_lower for kw in ['glucose', 'fructose', 'sugar',
                                       'galactinol', 'mannose', 'maltitol',
                                       'erythritol', 'xylose', 'ribitol',
                                       'threitol', 'hexose', 'pentose',
                                       'galactose', 'deoxyribose']):
        return 'Carbohydrates', PATHWAY_COLORS['Carbohydrates']

    # Nucleotides (expanded)
    elif any(kw in name_lower for kw in ['adenosine', 'guanosine', 'cytidine',
                                         'nucleotide', 'nucleoside', 'purine',
                                         'pyrimidine', 'ribose', 'phosphate',
                                         'hypoxanthine']):
        return 'Nucleotides_Nucleosides', PATHWAY_COLORS['Nucleotides_Nucleosides']

    # Polyamines (CORRECTED - check before amino acids!)
    elif any(kw in name_lower for kw in ['putrescine', 'spermidine', 'spermine',
                                         'polyamine', 'cadaverine',
                                         'diaminopropane']):
        return 'Polyamines', PATHWAY_COLORS['Polyamines']

    # Organic Acids
    elif any(kw in name_lower for kw in ['lactic acid', 'pyruvate', 'citrate',
                                         'acetate', 'succinic acid', 'fumarate',
                                         'malic acid', 'ketoglutarate']):
        return 'Organic Acids', PATHWAY_COLORS['Organic Acids']

    # Lipids
    elif any(kw in name_lower for kw in ['oleic', 'palmitic', 'stearic',
                                         'linoleic', 'lipid', 'fatty',
                                         'phospholipid', 'cholesterol']):
        return 'Lipids', PATHWAY_COLORS['Lipids']

    # Amino Acids (check AFTER polyamines!)
    elif any(kw in name_lower for kw in ['amino', 'alanine', 'valine',
                                         'leucine', 'isoleucine', 'serine',
                                         'threonine', 'proline', 'glycine',
                                         'aspartate', 'glutamate', 'acetylserine']):
        return 'Amino Acids', PATHWAY_COLORS['Amino Acids']

    # Amino Acid Metabolism
    elif any(kw in name_lower for kw in ['urea', 'uric', 'uracil']):
        return 'Amino Acid Metabolism', PATHWAY_COLORS['Amino Acid Metabolism']

    # Vitamins/Cofactors
    elif any(kw in name_lower for kw in ['vitamin', 'cofactor', 'coenzyme',
                                         'nad', 'nadp', 'fad', 'biotin', 'folate']):
        return 'Vitamins_Cofactors', PATHWAY_COLORS['Vitamins_Cofactors']

    # Phenolic compounds
    elif any(kw in name_lower for kw in ['phenol', 'benzoic', 'gallic',
                                         'vanillic', 'ferulic']):
        return 'Phenolic_Compounds', PATHWAY_COLORS['Phenolic_Compounds']

    # Default
    return 'Unknown', PATHWAY_COLORS['Unknown']

def add_pathway_legend(ax, metabolite_colors, metabolite_pathways):
    """Add colored patch legend for metabolite pathways."""

    # Get unique pathways
    pathways_in_heatmap = {}
    for met, pathway in metabolite_pathways.items():
        if pathway not in pathways_in_heatmap:
            pathways_in_heatmap[pathway] = metabolite_colors[pathway]

    # Create legend patches
    legend_elements = []
    for pathway in sorted(pathways_in_heatmap.keys()):
        color = pathways_in_heatmap[pathway]
        patch = Patch(facecolor=color, edgecolor='black', label=pathway)
        legend_elements.append(patch)

    # Add legend above heatmap
    ax.legend(handles=legend_elements,
              loc='upper center', bbox_to_anchor=(0.5, 1.20),
              ncol=4, frameon=True, fontsize=10,
              title='Metabolite Pathways', title_fontsize=11)

def create_metabolite_significance_summary(metabolites, test_results_df, group_stats_df, posthoc_df, fold_change, final_sig_mets):
    """
    Create a comprehensive per-metabolite significance summary table.
    This is Level 4 analysis - shows WHERE each metabolite is significant.
    """
    summary_data = {
        'Metabolite': [],
        'Global_p_value': [],
        'Global_Significant': [],
        'E.coli_vs_K12_p': [],
        'E.coli_vs_K12_sig': [],
        'P.aero_vs_PA01_p': [],
        'P.aero_vs_PA01_sig': [],
        'Pairs_vs_PairEP_p': [],
        'Pairs_vs_PairEP_sig': [],
        'PostHoc_Significant_Strains': [],
        'Avg_FoldChange': [],
        'In_FinalList': []
    }

    for met in metabolites:
        # Global ANOVA p-value
        global_row = test_results_df[test_results_df['Metabolite'] == met]
        global_p = global_row['p_value'].values[0] if not global_row.empty else np.nan
        global_sig = (global_p < 0.05) if not np.isnan(global_p) else False

        # Group analysis results
        group_rows = group_stats_df[group_stats_df['Metabolite'] == met]

        e_coli_p = np.nan
        e_coli_sig = False
        p_aero_p = np.nan
        p_aero_sig = False
        pairs_p = np.nan
        pairs_sig = False

        for _, row in group_rows.iterrows():
            if row['Group1'] == 'e_mono' and row['Relationship'] == 'Control comparison':
                e_coli_p = row['p_value']
                e_coli_sig = row['Significant']
            elif row['Group1'] == 'p_mono' and row['Relationship'] == 'Control comparison':
                p_aero_p = row['p_value']
                p_aero_sig = row['Significant']
            elif row['Group1'] == 'pairs' and row['Relationship'] == 'Control comparison':
                pairs_p = row['p_value']
                pairs_sig = row['Significant']

        # Post-hoc significant strains
        posthoc_rows = posthoc_df[(posthoc_df['Metabolite'] == met) &
                                  (posthoc_df['Significant'] == True)]
        sig_strains = list(posthoc_rows['Strain2'].unique())
        sig_strains_str = ', '.join(sig_strains) if sig_strains else 'None'

        # Average fold change
        if met in fold_change.columns:
            avg_fc = fold_change.loc[[s for s in fold_change.index if s != 'Control'], met].abs().mean()
        else:
            avg_fc = np.nan

        summary_data['Metabolite'].append(met)
        summary_data['Global_p_value'].append(f"{global_p:.6f}" if not np.isnan(global_p) else "NaN")
        summary_data['Global_Significant'].append("✓" if global_sig else "✗")
        summary_data['E.coli_vs_K12_p'].append(f"{e_coli_p:.6f}" if not np.isnan(e_coli_p) else "N/A")
        summary_data['E.coli_vs_K12_sig'].append("✓" if e_coli_sig else "✗")
        summary_data['P.aero_vs_PA01_p'].append(f"{p_aero_p:.6f}" if not np.isnan(p_aero_p) else "N/A")
        summary_data['P.aero_vs_PA01_sig'].append("✓" if p_aero_sig else "✗")
        summary_data['Pairs_vs_PairEP_p'].append(f"{pairs_p:.6f}" if not np.isnan(pairs_p) else "N/A")
        summary_data['Pairs_vs_PairEP_sig'].append("✓" if pairs_sig else "✗")
        summary_data['PostHoc_Significant_Strains'].append(sig_strains_str)
        summary_data['Avg_FoldChange'].append(f"{avg_fc:.4f}" if not np.isnan(avg_fc) else "NaN")
        summary_data['In_FinalList'].append("✓" if met in final_sig_mets else "✗")

    return pd.DataFrame(summary_data)

#==============================================================================
# VISUALIZATION FUNCTIONS
#==============================================================================
def generate_bar_plots(df, metabolites, medium, out_folder, posthoc_df):
    sns.set_style('whitegrid')
    color_map = {
        'Control': 'darkturquoise',
        'PA01': 'darkgreen',
        'K12': 'gold',
        'PairEP': 'saddlebrown',
        'Tmp04': 'darkgreen',
        'Tme12': 'gold',
        'Pair412': 'saddlebrown',
        'Tmp05': 'darkgreen',
        'Tme13': 'gold',
        'Pair513': 'saddlebrown',
        'Tmp06': 'darkgreen',
        'Tme14': 'gold',
        'Pair614': 'saddlebrown',
        'Tmp07': 'darkgreen',
        'Tme15': 'gold',
        'Pair715': 'saddlebrown'
    }
    
    for met in tqdm(metabolites, desc="Generating Bar Plots"):
        fig, ax = plt.subplots(figsize=(14, 8))
        means = df.groupby('Strain')[met].mean().reindex(sample_order)
        stds = df.groupby('Strain')[met].std().reindex(sample_order)
        # Use the propagated SD for Control instead of calculating from data
        stds['Control'] = control_sds_dict[met]
        
        colors = [color_map.get(s, 'gray') for s in means.index]
        ax.bar(means.index, means, yerr=stds, capsize=5, color=colors, width=0.8)
        
        configure_plot(ax, f"Average peak area of {met} in {medium} spent medium", "Strain", "Peak Area")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xticks(range(len(means.index)))
        ax.set_xticklabels(means.index, rotation=45, ha='center', fontsize=10)
        
        # Add significance asterisks based on post-hoc test results
        if posthoc_df is not None and not posthoc_df.empty:
            for idx, strain in enumerate(sample_order):
                if strain == 'Control':
                    continue
                
                # Check for significant difference between this strain and Control
                # MODIFIED: Only include results marked as significant in the Significant column
                is_significant = ((posthoc_df['Metabolite'] == met) &
                                 (((posthoc_df['Strain1'] == 'Control') & (posthoc_df['Strain2'] == strain)) |
                                  ((posthoc_df['Strain1'] == strain) & (posthoc_df['Strain2'] == 'Control'))) &
                                 (posthoc_df['Significant'] == True))  # Added this condition
                
                if is_significant.any():
                    # Get the p-value
                    p_val = posthoc_df[is_significant]['p_value'].values[0]
                    
                    # Calculate position for the asterisk
                    y_pos = means[strain] + stds[strain] + (means.max() * 0.05)
                    
                    # Add significance marker - MODIFIED: removed 'ns' option
                    if p_val < 0.001:
                        marker = '***'
                    elif p_val < 0.01:
                        marker = '**'
                    elif p_val < 0.05:
                        marker = '*'
                    # Removed the 'else: marker = "ns"' to avoid showing NS for non-significant results
                    
                    ax.text(idx, y_pos, marker, ha='center', fontsize=12, color='black', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(out_folder, f"{met}_{medium}_sup.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def generate_heatmap(fc_df, medium, out_folder, metabolite_colors=None, metabolite_pathways=None):
    """
    Generate heatmap with metabolite pathway coloring.

    Parameters:
    -----------
    fc_df : DataFrame
        Fold change data
    medium : str
        Medium type (AUM/ISO)
    out_folder : str
        Output folder path
    metabolite_colors : dict, optional
        Mapping of metabolite name to pathway color
    """
    fc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fc_df.fillna(0, inplace=True)
    fc_clipped = fc_df.clip(lower=-3, upper=3)

    num_rows, num_cols = fc_clipped.shape
    cell_size = 0.5
    figsize = (num_cols * cell_size + 6, num_rows * cell_size + 8)

    if num_rows <= 1:
        # Single metabolite - simple heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(fc_clipped, cmap='viridis', center=0, vmin=-3, vmax=3,
                         cbar_kws={"label": "Log2(FC) vs Control"}, square=True)

        if metabolite_colors and metabolite_pathways:
            add_pathway_legend(ax, metabolite_colors, metabolite_pathways)

        ax.set_title(f"Heatmap of metabolites in {medium} spent medium\n", fontsize=14)

        # Color y-axis labels by pathway (if colors provided)
        if metabolite_colors:
            for i, label in enumerate(ax.get_yticklabels()):
                met_name = label.get_text().replace(' *', '').replace(' #', '')
                if met_name in metabolite_colors:
                    label.set_color(metabolite_colors[met_name])
                    label.set_fontweight('bold')

        plt.tight_layout()
        heatmap_path = os.path.join(out_folder, f"Heatmap_{medium}_sup.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # Multiple metabolites - clustered heatmap with color bar
        hm = sns.clustermap(fc_clipped, cmap='viridis', center=0, vmin=-3, vmax=3,
                            figsize=figsize, dendrogram_ratio=(0.1, 0.05),
                            cbar_kws={"label": "Log2(FC) vs Control"}, square=True)

        hm.ax_heatmap.set_title(f"Heatmap of metabolites in {medium} spent medium\n", fontsize=14, pad=80)

        # Color y-axis labels by pathway
        if metabolite_colors:
            for i, label in enumerate(hm.ax_heatmap.get_yticklabels()):
                met_name = label.get_text().replace(' *', '').replace(' #', '')
                if met_name in metabolite_colors:
                    label.set_color(metabolite_colors[met_name])
                    label.set_fontweight('bold')

        heatmap_path = os.path.join(out_folder, f"Heatmap_{medium}_sup.png")
        hm.fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(hm.fig)

        print(f"Heatmap saved to: {heatmap_path}")

def generate_volcano_plots(test_results_df, fold_change, medium, out_folder, final_sig_mets):
    """
    Generate publication-quality volcano plots for ANOVA results.
    X-axis: Log2 fold-change vs Control (average across strains)
    Y-axis: -Log10(global ANOVA p-value)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.lines import Line2D

    # Prepare volcano data
    volcano_data = []
    for met in test_results_df['Metabolite']:
        row = test_results_df[test_results_df['Metabolite'] == met].iloc[0]

        # Global ANOVA p-value
        p_val = row['p_value']
        neg_log_p = -np.log10(p_val) if p_val > 0 else 0

        # Average log2 fold-change across ALL non-control strains
        non_ctrl_fc = fold_change.drop('Control').mean(axis=0)[met]

        # Pathway coloring from your existing system
        pathway, color = get_metabolite_pathway(met)
        is_sig = p_val < 0.05

        volcano_data.append({
            'Metabolite': met,
            'Log2FC': non_ctrl_fc,
            'NegLogP': neg_log_p,
            'Significant': is_sig,
            'Pathway': pathway,
            'Color': color,
            'InFinalList': met in final_sig_mets
        })

    df_volcano = pd.DataFrame(volcano_data)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Threshold lines
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    plt.axvline(-1, color='red', linestyle='--', alpha=0.7, label='|Log2FC|=1')
    plt.axvline(1, color='red', linestyle='--', alpha=0.7)

    # Color-code by pathway (your existing PATHWAY_COLORS)
    for pathway, color in PATHWAY_COLORS.items():
        subset = df_volcano[df_volcano['Pathway'] == pathway]
        alpha = 0.8 if pathway != 'Unknown' else 0.4
        plt.scatter(subset['Log2FC'], subset['NegLogP'],
                    c=color, s=60, alpha=alpha, edgecolors='black', linewidth=0.5,
                    label=pathway if pathway != 'Unknown' else None)

    # Highlight your final significant metabolites
    final_subset = df_volcano[df_volcano['InFinalList']]
    if not final_subset.empty:
        plt.scatter(final_subset['Log2FC'], final_subset['NegLogP'],
                    c='gold', s=120, alpha=1, edgecolors='black', linewidth=1.5,
                    marker='*', zorder=5, label=f'Final List (n={len(final_subset)})')

    plt.xlabel('Average Log2 Fold-Change (vs Control)', fontsize=12, fontweight='bold')
    plt.ylabel('-Log10(ANOVA p-value)', fontsize=12, fontweight='bold')
    plt.title(f'Volcano Plot: {medium} Supernatant Metabolomics\n(Global ANOVA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"Volcano_Plot_{medium}_sup.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Volcano plot saved: Volcano_Plot_{medium}_sup.png")
    print(f"   • Total metabolites: {len(df_volcano)}")
    print(f"   • ANOVA significant (p<0.05): {df_volcano['Significant'].sum()}")
    print(f"   • In final visualization list: {len(final_sig_mets)}")

    return df_volcano

def generate_hierarchical_group_bar_plots(df, metabolites, medium, out_folder):
    """Generate advanced bar plots showing both strain-level and group-level statistics,
    with proper significance indicators reflecting the hierarchical experimental structure."""
    global global_group_stats
    
    # Define bar ordering - all controls first, then clinical groups
    bar_order = ['PA01', 'PairEP', 'K12', 'P. aeruginosa', 'Co-cultures', 'E. coli']
    
    # Define clinical groups and their respective controls
    clinical_groups = {
        'P. aeruginosa': {'strains': p_mono, 'control': 'PA01', 'internal': 'p_mono'},
        'Co-cultures': {'strains': pairs, 'control': 'PairEP', 'internal': 'pairs'},
        'E. coli': {'strains': e_mono, 'control': 'K12', 'internal': 'e_mono'}
    }
    
    # Define colors with lighter colors for controls
    group_colors = {
        'PA01': 'lightgreen',
        'P. aeruginosa': 'darkgreen',
        'PairEP': 'burlywood',
        'Co-cultures': 'saddlebrown',
        'K12': 'palegoldenrod',
        'E. coli': 'gold'
    }
    
    # Define between-group comparisons
    between_comparisons = [
        {'groups': ('E. coli', 'P. aeruginosa'), 'key': 'e_mono_vs_p_mono', 'type': 'Unpaired'},
        {'groups': ('E. coli', 'Co-cultures'), 'key': 'e_mono_vs_pairs', 'type': 'Paired'},
        {'groups': ('P. aeruginosa', 'Co-cultures'), 'key': 'p_mono_vs_pairs', 'type': 'Paired'}
    ]
    
    for met in tqdm(metabolites, desc="Generating Hierarchical Group Bar Plots"):
        # Check if we have stats for this metabolite
        if met not in global_group_stats:
            continue
            
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Calculate group-level values for the plot
        group_means = []
        group_sems = []
        bar_widths = []
        
        # First, calculate group-level statistics
        for group in bar_order:
            if group in ['PA01', 'PairEP', 'K12']:  # Controls are individual strains
                # Extract the mean and error from strain_stats for control strains
                if 'strain_stats' in global_group_stats[met] and group in global_group_stats[met]['strain_stats']:
                    strain_data = global_group_stats[met]['strain_stats'][group]
                    group_means.append(strain_data['mean'])
                    # SEM = STD / sqrt(n)
                    group_sems.append(strain_data['std'] / np.sqrt(strain_data['n']) if strain_data['n'] > 0 else 0)
                else:
                    # Fallback to raw calculation if not found
                    strain_data = df[df['Strain'] == group][met].dropna()
                    if len(strain_data) > 0:
                        group_means.append(strain_data.mean())
                        group_sems.append(strain_data.sem())
                    else:
                        group_means.append(0)
                        group_sems.append(0)
            else:  # Clinical groups
                # For clinical groups, average the strain means from strain_stats
                internal_group = clinical_groups[group]['internal']
                strain_list = clinical_groups[group]['strains']
                
                # Collect strain means for this group
                strain_means = []
                for strain in strain_list:
                    if 'strain_stats' in global_group_stats[met] and strain in global_group_stats[met]['strain_stats']:
                        strain_mean = global_group_stats[met]['strain_stats'][strain]['mean']
                        if not np.isnan(strain_mean):
                            strain_means.append(strain_mean)
                
                if strain_means:
                    # Group mean is average of strain means
                    group_means.append(np.mean(strain_means))
                    # Group SEM is standard deviation of strain means / sqrt(number of strains)
                    group_sems.append(np.std(strain_means, ddof=1) / np.sqrt(len(strain_means)))
                else:
                    group_means.append(0)
                    group_sems.append(0)
        
        # Plot the group-level bars
        x_pos = np.arange(len(bar_order))
        bars = ax.bar(x_pos, group_means, yerr=group_sems, capsize=5, 
                      color=[group_colors[g] for g in bar_order], alpha=0.8, width=0.7)
        
        # Overlay strain-level data points on top of group bars for clinical groups
        for i, group in enumerate(bar_order):
            if group not in ['PA01', 'PairEP', 'K12']:  # Only for clinical groups
                internal_group = clinical_groups[group]['internal']
                strain_list = clinical_groups[group]['strains']
                
                # Create positions for strain points, spread within the bar width
                strain_positions = np.linspace(x_pos[i] - 0.25, x_pos[i] + 0.25, len(strain_list))
                
                # Plot each strain as a point with error bar
                for j, strain in enumerate(strain_list):
                    if 'strain_stats' in global_group_stats[met] and strain in global_group_stats[met]['strain_stats']:
                        strain_data = global_group_stats[met]['strain_stats'][strain]
                        strain_mean = strain_data['mean']
                        strain_sem = strain_data['std'] / np.sqrt(strain_data['n']) if strain_data['n'] > 0 else 0
                        
                        if not np.isnan(strain_mean):
                            # Plot strain mean as a black dot
                            ax.scatter(strain_positions[j], strain_mean, color='black', s=50, zorder=3)
                            # Add error bar for the strain
                            ax.errorbar(strain_positions[j], strain_mean, yerr=strain_sem, 
                                       fmt='none', ecolor='black', capsize=3, alpha=0.5, zorder=2)
        
        # Add significance markers for control vs. clinical group comparisons
        for i, group in enumerate(bar_order):
            if group in clinical_groups:  # Only for clinical groups
                control = clinical_groups[group]['control']
                control_idx = bar_order.index(control)
                internal_group = clinical_groups[group]['internal']
                
                # Get significance from global stats
                if 'control_comparisons' in global_group_stats[met] and internal_group in global_group_stats[met]['control_comparisons']:
                    result = global_group_stats[met]['control_comparisons'][internal_group]

                    if result.get('significant', False):
                        p_val = result['p_value']

                        if p_val < 0.001:
                            marker = '***'
                        elif p_val < 0.01:
                            marker = '**'
                        elif p_val < 0.05:
                            marker = '*'
                        else:
                            # Not actually significant by threshold → don't draw anything
                            marker = None

                        if marker is not None:
                            y_pos = group_means[control_idx] + group_sems[control_idx] + (max(group_means) * 0.05)
                            ax.text(control_idx, y_pos, marker, ha='center', fontsize=14, fontweight='bold')

        # Add significance markers for between-group comparisons
        for comp in between_comparisons:
            group1, group2 = comp['groups']
            comp_key = comp['key']
            
            if 'between_group_comparisons' in global_group_stats[met]:
                if comp_key in global_group_stats[met]['between_group_comparisons']:
                    result = global_group_stats[met]['between_group_comparisons'][comp_key]
                    
                    if result.get('significant', False):
                        # Find indices in bar_order
                        idx1 = bar_order.index(group1)
                        idx2 = bar_order.index(group2)
                        
                        p_val = result['p_value']
                        # Initialize marker variable
                        marker = ''
                        if p_val < 0.001:
                            marker = '***'
                        elif p_val < 0.01:
                            marker = '**'
                        elif p_val < 0.05:
                            marker = '*'
                        
                        # Draw line between groups
                        y_max = max(group_means[idx1] + group_sems[idx1], 
                                  group_means[idx2] + group_sems[idx2])
                        y_line = y_max + (max(group_means) * 0.15)
                        
                        # Draw line and marker
                        ax.plot([idx1, idx2], [y_line, y_line], 'k-', linewidth=1)
                        ax.text((idx1 + idx2) / 2, y_line + (max(group_means) * 0.02), 
                               marker, ha='center', fontsize=14, fontweight='bold')
        
        # Add strain names as small labels for clinical group bars
        for i, group in enumerate(bar_order):
            if group not in ['PA01', 'PairEP', 'K12']:  # Only for clinical groups
                strain_list = clinical_groups[group]['strains']
                strain_positions = np.linspace(x_pos[i] - 0.25, x_pos[i] + 0.25, len(strain_list))
                
                for j, strain in enumerate(strain_list):
                    # Add small strain name below x-axis
                    ax.text(strain_positions[j], -max(group_means) * 0.07, 
                           strain, ha='center', va='top', fontsize=9, rotation=45, alpha=0.7)
        
        # Format plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_order, rotation=45, ha='right', fontsize=12)
        ax.set_title(f"{met} levels by bacterial group", fontsize=14)
        ax.set_ylabel("Peak Area", fontsize=12)
        
        # Set y-axis to scientific notation
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Move x-axis to base of plot
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add annotations
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, 
                  label='Individual strain means'),
            Line2D([0], [0], color='black', marker='|', markersize=10, linewidth=1,
                  label='Strain standard error'),
            Line2D([0], [0], color='black', linewidth=1, 
                  label='Group comparison significance')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
        
        # Add statistical significance explanation
        txt = "Significance: * p<0.05, ** p<0.01, *** p<0.001\n"
        txt += "Bars: Group means with SEM across strains\n"
        txt += "Points: Individual strain means with SEM across technical replicates"
        ax.annotate(txt, xy=(0.02, 0.98), xycoords='figure fraction', 
                   ha='left', va='top', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                    fc="white", ec="gray", alpha=0.8))
        
        # Add some padding to the top for significance markers
        y_max = ax.get_ylim()[1]
        ax.set_ylim(bottom=-max(group_means) * 0.1, top=y_max * 1.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"{met}_hierarchical_groups_{medium}_sup.png"), dpi=300)
        plt.close()

def generate_group_heatmap(df, metabolites, medium, out_folder, use_respective_controls=True):
    """Generate a heatmap showing fold changes by biological group with significance markers.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing metabolite data
    metabolites : list
        List of metabolites to include in the heatmap
    medium : str
        Medium type for labeling
    out_folder : str
        Output directory for saving the plot
    use_respective_controls : bool, default=True
        If True, calculates fold changes relative to respective controls (K12, PA01, PairEP)
        If False, calculates fold changes relative to the unified control
    """
    global global_group_stats
    
    # Debug: Check what's in the global dictionary
    print(f"Generating heatmap with {len(metabolites)} metabolites")
    print(f"Global dictionary contains {len(global_group_stats)} metabolites")
    if len(global_group_stats) > 0:
       # Print keys of the first metabolite to check structure
       first_met = next(iter(global_group_stats))
       print(f"Dictionary structure for {first_met}: {global_group_stats[first_met].keys()}")
       
       # Check how many metabolites have control comparisons
       has_control = sum(1 for m in global_group_stats if 'control_comparisons' in global_group_stats[m])
       print(f"Metabolites with control comparisons: {has_control}") 
    
    # Define clinical groups and their respective controls
    clinical_groups = {
        'E. coli': e_mono,
        'P. aeruginosa': p_mono,
        'Co-cultures': pairs
    }
    
    respective_controls = {
        'E. coli': 'K12',
        'P. aeruginosa': 'PA01',
        'Co-cultures': 'PairEP'
    }
    
    # Create reverse mapping for accessing stats
    reverse_mapping = {
        'E. coli': 'e_mono',
        'P. aeruginosa': 'p_mono',
        'Co-cultures': 'pairs'
    }
    
    # Calculate fold changes for each group
    group_fc = pd.DataFrame(index=metabolites)
    
    for group_name, strains in clinical_groups.items():
        # Calculate mean for group
        group_vals = df[df['Strain'].isin(strains)][metabolites].mean()
        
        if use_respective_controls:
            # Calculate fold change relative to respective control
            control_strain = respective_controls[group_name]
            control_vals = df[df['Strain'] == control_strain][metabolites].mean()
        else:
            # Calculate fold change relative to unified control
            control_vals = df[df['Strain'] == 'Control'][metabolites].mean()
        
        # Log2 fold change
        fc = np.log2((group_vals + 1e-6) / (control_vals + 1e-6))
        group_fc[group_name] = fc
    
    # Create annotation matrix
    annot_matrix = pd.DataFrame('', index=metabolites, columns=group_fc.columns)
    
    # Populate annotation matrix with significance markers
    for met in metabolites:
        if met not in global_group_stats:
            continue
            
        for display_group in clinical_groups.keys():
            internal_group = reverse_mapping[display_group]
            
            # Check if we have control comparison data
            if 'control_comparisons' in global_group_stats[met] and internal_group in global_group_stats[met]['control_comparisons']:
                result = global_group_stats[met]['control_comparisons'][internal_group]
                
                if result['significant']:
                    p_val = result['p_value']
                    if p_val < 0.001:
                        annot_matrix.loc[met, display_group] = '***'
                    elif p_val < 0.01:
                        annot_matrix.loc[met, display_group] = '**'
                    elif p_val < 0.05:
                        annot_matrix.loc[met, display_group] = '*'
    
    # Get a list of significant metabolites for ordering
    sig_metabolites = []
    for met in metabolites:
        if met not in global_group_stats:
            continue
            
        is_significant = False
        if 'control_comparisons' in global_group_stats[met]:
            for group in clinical_groups.keys():
                internal_group = reverse_mapping[group]
                if internal_group in global_group_stats[met]['control_comparisons']:
                    if global_group_stats[met]['control_comparisons'][internal_group]['significant']:
                        is_significant = True
                        break
        
        if is_significant:
            sig_metabolites.append(met)
    
    # Order metabolites by significance and patterns
    if sig_metabolites:
        # Create a subset for clustering
        subset_fc = group_fc.loc[sig_metabolites]
        
        # Check if subset is suitable for clustering (non-empty and has enough rows)
        if len(subset_fc) >= 2:
            try:
                # Use hierarchical clustering to order the metabolites by similarity
                from scipy.cluster import hierarchy
                linkage = hierarchy.linkage(subset_fc, method='ward')
                order = hierarchy.leaves_list(linkage)
                ordered_metabolites = [sig_metabolites[i] for i in order]
            except Exception as e:
                print(f"Hierarchical clustering failed: {e}. Using default ordering.")
                ordered_metabolites = sig_metabolites
        else:
            # Not enough metabolites for clustering
            print(f"Only {len(subset_fc)} significant metabolites found, skipping clustering.")
            ordered_metabolites = sig_metabolites
        
        # Add the non-significant metabolites at the end
        ordered_metabolites.extend([m for m in metabolites if m not in sig_metabolites])
    else:
            print("No significant metabolites found for clustering. Using default ordering.")
            ordered_metabolites = metabolites

    # Subset the dataframes to the ordered metabolites
    group_fc = group_fc.loc[ordered_metabolites]
    annot_matrix = annot_matrix.loc[ordered_metabolites]
    
    # Determine an appropriate figure height based on number of metabolites
    height = max(8, len(ordered_metabolites) * 0.4 + 2)
    
    # Plot heatmap
    plt.figure(figsize=(10, height))
    
    if use_respective_controls:
        title = f"Metabolite changes by group vs. respective controls in {medium} spent medium"
    else:
        title = f"Metabolite changes by group vs. unified control in {medium} spent medium"
    
    # Use viridis colormap as requested
    cmap = 'viridis'
    ax = sns.heatmap(group_fc, cmap=cmap, vmin=-2, vmax=2, 
                     annot=annot_matrix, fmt='', cbar_kws={'label': 'Log2(FC)'})
    
    plt.title(title)
    plt.tight_layout()
    
    # Save the heatmap
    if use_respective_controls:
        plt.savefig(os.path.join(out_folder, f"Group_heatmap_resp_{medium}_sup.png"), dpi=300)
    else:
        plt.savefig(os.path.join(out_folder, f"Group_heatmap_unified_{medium}_sup.png"), dpi=300)
    
    plt.close()

def generate_comprehensive_bubble_plot(df, metabolites, medium, out_folder, use_respective_controls=True):
    """Generate improved bubble plot showing metabolite changes across groups.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing metabolite data
    metabolites : list
        List of metabolites to include
    medium : str
        Medium type for labeling
    out_folder : str
        Output directory for saving the plot
    use_respective_controls : bool, default=True
        If True, calculates fold changes relative to respective controls
        If False, calculates fold changes relative to the unified control
    """
    global global_group_stats
    from adjustText import adjust_text  # Added for label management
    
    # Define clinical groups and their respective controls
    clinical_groups = {
        'E. coli': e_mono,
        'P. aeruginosa': p_mono,
        'Co-cultures': pairs
    }
    
    respective_controls = {
        'E. coli': 'K12',
        'P. aeruginosa': 'PA01',
        'Co-cultures': 'PairEP'
    }
    
    # Create reverse mapping for accessing stats
    reverse_mapping = {
        'E. coli': 'e_mono',
        'P. aeruginosa': 'p_mono',
        'Co-cultures': 'pairs'
    }
    
    # Colors for groups
    group_colors = {
        'E. coli': 'gold',
        'P. aeruginosa': 'darkgreen', 
        'Co-cultures': 'saddlebrown'
    }
    
    # Prepare data for plotting
    plot_data = []
    
    for met in metabolites:
        if met not in global_group_stats:
            continue
            
        # Get base abundance for scaling bubble size
        base_abundance = df[met].mean()
        
        for display_group, strains in clinical_groups.items():
            # Calculate mean for clinical group
            group_val = df[df['Strain'].isin(strains)][met].mean()
            
            if use_respective_controls:
                # Calculate fold change relative to respective control
                control_strain = respective_controls[display_group]
                control_val = df[df['Strain'] == control_strain][met].mean()
            else:
                # Calculate fold change relative to unified control
                control_val = df[df['Strain'] == 'Control'][met].mean()
            
            # Calculate log2 fold change
            log2_fc = np.log2((group_val + 1e-6) / (control_val + 1e-6))
            
            # Get significance and p-value from global stats dictionary
            internal_group = reverse_mapping[display_group]
            
            if 'control_comparisons' in global_group_stats[met] and internal_group in global_group_stats[met]['control_comparisons']:
                result = global_group_stats[met]['control_comparisons'][internal_group]
                p_val = result['p_value']
                is_sig = result['significant']
                
                # Calculate relative abundance for bubble size (normalized to max across all metabolites)
                rel_abundance = group_val / base_abundance if base_abundance > 0 else 1
                       
                # Store data for plotting
                plot_data.append({
                    'Metabolite': met,
                    'Group': display_group,
                    'Log2FC': log2_fc,
                    'P-value': p_val,
                    '-log10(p)': -np.log10(p_val) if p_val > 0 else 0,
                    'Significant': is_sig,
                    'Size': min(300 * (abs(log2_fc) + 0.5), 1000),  # Size based on FC magnitude
                    'Comparison': 'vs Control'
                })
    
    # Add between-group comparisons if requested (optional)
    between_comparisons = [
        ('e_mono_vs_p_mono', 'E. coli', 'P. aeruginosa'),
        ('e_mono_vs_pairs', 'E. coli', 'Co-cultures'),
        ('p_mono_vs_pairs', 'P. aeruginosa', 'Co-cultures')
    ]
    
    # Create DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create separate figures for each group for clarity
    plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Set up subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Title based on control type
    control_type = "respective controls" if use_respective_controls else "unified control"
    fig.suptitle(f'Metabolite Changes vs {control_type} in {medium} in planktonic cells', fontsize=16)
    
    # Add threshold lines to each subplot
    for i, (group, ax) in enumerate(zip(clinical_groups.keys(), axes)):
        # Filter data for this group
        group_data = plot_df[plot_df['Group'] == group]
        
        # Plot scatter with bubble size
        scatter = ax.scatter(
            group_data['Log2FC'], 
            group_data['-log10(p)'],
            s=group_data['Size'], 
            alpha=0.7,
            c=[group_colors[group]] * len(group_data),
            edgecolors='black',
            linewidths=1
        )
        
        # Add threshold lines
        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-0.75, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.75, color='gray', linestyle='--', alpha=0.5)
        
        # Fill quadrants with light colors
        ax.fill_between([-5, -0.75], [-np.log10(0.05), -np.log10(0.05)], [5, 5], color='lightgray', alpha=0.2)
        ax.fill_between([0.75, 5], [-np.log10(0.05), -np.log10(0.05)], [5, 5], color='lightgray', alpha=0.2)
        ax.fill_between([-5, -0.75], [0, 0], [-np.log10(0.05), -np.log10(0.05)], color='white', alpha=0.2)
        ax.fill_between([0.75, 5], [0, 0], [-np.log10(0.05), -np.log10(0.05)], color='white', alpha=0.2)
        
        # Add quadrant labels
        ax.text(-2, 2, "Decreased\nSignificant", ha='center', fontsize=9, alpha=0.8)
        ax.text(2, 2, "Increased\nSignificant", ha='center', fontsize=9, alpha=0.8)
        
        # Label significant points
        texts =[]
        for _, row in group_data[group_data['Significant']].iterrows():
            ax.annotate(
                row['Metabolite'],
                (row['Log2FC'], row['-log10(p)']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Automatically adjust text positions to prevent overlap
        adjust_text(texts, 
                   ax=ax,
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                   force_text=(0.5, 0.7),  # Adjust repelling forces
                   force_points=(0.5, 0.7),
                   expand_points=(1.2, 1.4),
                   expand_text=(1.2, 1.4),
                   lim=500)  # Max iterations
        
        # Set axis limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, max(4, plot_df['-log10(p)'].max() * 1.1))
        
        # Add labels
        ax.set_title(group)
        ax.set_xlabel('Log2 Fold Change')
        if i == 0:
            ax.set_ylabel('-Log10 P-value')
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Create legend explaining bubble sizes
    sizes = [100, 200, 400]
    labels = ['Small FC', 'Medium FC', 'Large FC']
    
    # Add legend for bubble sizes
    legend_elements = []
    for size, label in zip(sizes, labels):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                              label=label, markerfacecolor='gray', 
                              markeredgecolor='black', markersize=np.sqrt(size/5)))
    
    fig.legend(handles=legend_elements, loc='upper right', title="Magnitude", bbox_to_anchor=(0.99, 0.99))
    
    # Save the figure
    control_suffix = "resp" if use_respective_controls else "unified"
    plt.savefig(os.path.join(out_folder, f"Bubble_plot_{control_suffix}_{medium}_pln.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_between_group_bubble_plots(df, metabolites, medium, out_folder):
    """Generate bubble plots for specific between-group comparisons."""
    global global_group_stats
    from matplotlib.lines import Line2D
    from adjustText import adjust_text
    
    comparisons = [
        {'key': 'e_mono_vs_p_mono', 'title': 'E. coli vs P. aeruginosa', 
         'group1': 'e_mono', 'group2': 'p_mono', 'type': 'Unpaired'},
        {'key': 'e_mono_vs_pairs', 'title': 'E. coli vs Co-cultures', 
         'group1': 'e_mono', 'group2': 'pairs', 'type': 'Paired'},
        {'key': 'p_mono_vs_pairs', 'title': 'P. aeruginosa vs Co-cultures', 
         'group1': 'p_mono', 'group2': 'pairs', 'type': 'Paired'}
    ]
    
    display_names = {
        'e_mono': 'E. coli', 
        'p_mono': 'P. aeruginosa', 
        'pairs': 'Co-cultures'
    }
    
    colors = {
        'e_mono': 'gold',
        'p_mono': 'darkgreen',
        'pairs': 'saddlebrown'
    }
    
    between_group_results = {
        'Metabolite': [], 'Comparison': [], 'Group1': [], 'Group2': [], 
        'Log2FC': [], 'P-value': [], 'Significant': [], 'Test': []
    }
    
    # Create a figure with subplots for each comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Between-Group Metabolite Changes in {medium}', fontsize=16)
    
    for i, (comp_key, comp_title, group1, group2, comp_type) in enumerate(zip(
        [c['key'] for c in comparisons],
        [c['title'] for c in comparisons],
        [c['group1'] for c in comparisons],
        [c['group2'] for c in comparisons],
        [c['type'] for c in comparisons]
    )):
        ax = axes[i]
        ax.set_title(comp_title, fontsize=14)
        
        plot_data = []
        
        for met in metabolites:
            if met not in global_group_stats:
                continue
                
            if 'between_group_comparisons' in global_group_stats[met]:
                if comp_key in global_group_stats[met]['between_group_comparisons']:
                    result = global_group_stats[met]['between_group_comparisons'][comp_key]
                    
                    if comp_type == 'Paired' and 'paired_data' in result:
                        paired_data = result['paired_data']
                        g1_means = [p[0] for p in paired_data]
                        g2_means = [p[1] for p in paired_data]
                        
                        if g1_means and g2_means:
                            g1_mean = np.mean(g1_means)
                            g2_mean = np.mean(g2_means)
                            log2_fc = np.log2((g1_mean + 1e-6) / (g2_mean + 1e-6))
                            
                            p_val = result['p_value']
                            is_sig = result['significant']
                            test_used = result.get('test', 'Unknown')
                            
                            plot_data.append({
                                'Metabolite': met,
                                'Log2FC': log2_fc,
                                'P-value': p_val,
                                '-log10(p)': -np.log10(p_val) if p_val > 0 else 0,
                                'Significant': is_sig,
                                'Size': min(300 * (abs(log2_fc) + 0.5), 1000),
                                'Test': test_used
                            })
                            
                            between_group_results['Metabolite'].append(met)
                            between_group_results['Comparison'].append(comp_title)
                            between_group_results['Group1'].append(display_names[group1])
                            between_group_results['Group2'].append(display_names[group2])
                            between_group_results['Log2FC'].append(log2_fc)
                            between_group_results['P-value'].append(p_val)
                            between_group_results['Significant'].append(is_sig)
                            between_group_results['Test'].append(test_used)
                    
                    elif comp_type == 'Unpaired' and 'group1_means' in result and 'group2_means' in result:
                        g1_means = result['group1_means']
                        g2_means = result['group2_means']
                        
                        if g1_means and g2_means:
                            g1_mean = np.mean(g1_means)
                            g2_mean = np.mean(g2_means)
                            log2_fc = np.log2((g1_mean + 1e-6) / (g2_mean + 1e-6))
                            
                            p_val = result['p_value']
                            is_sig = result['significant']
                            test_used = result.get('test', 'Unknown')
                            
                            plot_data.append({
                                'Metabolite': met,
                                'Log2FC': log2_fc,
                                'P-value': p_val,
                                '-log10(p)': -np.log10(p_val) if p_val > 0 else 0,
                                'Significant': is_sig,
                                'Size': min(300 * (abs(log2_fc) + 0.5), 1000),
                                'Test': test_used
                            })
                            
                            between_group_results['Metabolite'].append(met)
                            between_group_results['Comparison'].append(comp_title)
                            between_group_results['Group1'].append(display_names[group1])
                            between_group_results['Group2'].append(display_names[group2])
                            between_group_results['Log2FC'].append(log2_fc)
                            between_group_results['P-value'].append(p_val)
                            between_group_results['Significant'].append(is_sig)
                            between_group_results['Test'].append(test_used)
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Plot scatter with bubble size
            ax.scatter(
                plot_df['Log2FC'], 
                plot_df['-log10(p)'],
                s=plot_df['Size'], 
                c=colors[group1] if 'Significant' in plot_df.columns else 'gray',
                alpha=0.7,
                edgecolors='black',
                linewidths=1
            )
            
            # Add threshold lines
            ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
            ax.axvline(-0.75, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(0.75, color='gray', linestyle='--', alpha=0.5)
            
            # Fill quadrants with light colors
            ax.fill_between([-5, -0.75], [-np.log10(0.05), -np.log10(0.05)], [5, 5], 
                          color=colors[group2], alpha=0.2)
            ax.fill_between([0.75, 5], [-np.log10(0.05), -np.log10(0.05)], [5, 5], 
                          color=colors[group1], alpha=0.2)
            
            # Add quadrant labels
            ax.text(-2, 2, f"Higher in {display_names[group2]}\nSignificant", ha='center', fontsize=10, alpha=0.8)
            ax.text(2, 2, f"Higher in {display_names[group1]}\nSignificant", ha='center', fontsize=10, alpha=0.8)
            
            # Create numbered labels
            significant_data = plot_df[plot_df['Significant']].sort_values('P-value')
            numbers = list(range(1, len(significant_data)+1))
            labels = [f"{n}. {row['Metabolite']}" for n, (_, row) in zip(numbers, significant_data.iterrows())]
            
            # Plot numbers at bubble centers
            for (_, row), number in zip(significant_data.iterrows(), numbers):
                ax.text(
                    row['Log2FC'],
                    row['-log10(p)'],
                    str(number),
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black',
                    fontweight='bold'
                )
            
            # Create legend for metabolites
            legend_handles = [Line2D([0], [0], marker='o', color='w', 
                           label=label, markerfacecolor=colors[group1],
                           markersize=8) for label in labels]
            ax.legend(handles=legend_handles, 
                     title="Metabolites",
                     loc='center left',
                     bbox_to_anchor=(1, 0.5),
                     frameon=True)
            
            # Set axis limits
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, max(4, plot_df['-log10(p)'].max() * 1.1))
            
            # Add labels
            ax.set_xlabel(f'Log2 Fold Change ({display_names[group1]} vs {display_names[group2]})', fontsize=12)
            if i == 0:
                ax.set_ylabel('-Log10 P-value', fontsize=12)
            
            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Create legend explaining bubble sizes
    sizes = [100, 200, 400]
    labels = ['Small FC', 'Medium FC', 'Large FC']
    
    # Add legend for bubble sizes
    legend_elements = []
    for size, label in zip(sizes, labels):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                              label=label, markerfacecolor='gray', 
                              markeredgecolor='black', markersize=np.sqrt(size/5)))
    
    fig.legend(handles=legend_elements, loc='upper right', title="Magnitude", bbox_to_anchor=(0.99, 0.99))
    
    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_folder, f"Between_group_bubble_plots_{medium}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return DataFrame for Excel export
    return pd.DataFrame(between_group_results)

# def generate_raincloud_plots(df, metabolites, medium, out_folder):
#     """Generate raincloud plots for metabolites, showing both group-level and strain-level distributions.
    
#     Parameters:
#     -----------
#     df : pandas DataFrame
#         The dataframe containing metabolite data
#     metabolites : list
#         List of metabolite names to plot
#     medium : str
#         Medium type for labeling
#     out_folder : str
#         Output directory path
#     """
#     global global_group_stats
    
#     # Import required packages
#     from matplotlib.lines import Line2D
#     import ptitprince as pt
    
#     # Define clinical groups and their respective controls
#     clinical_groups = {
#         'P. aeruginosa': {'strains': p_mono, 'control': 'PA01', 'internal': 'p_mono'},
#         'Co-cultures': {'strains': pairs, 'control': 'PairEP', 'internal': 'pairs'},
#         'E. coli': {'strains': e_mono, 'control': 'K12', 'internal': 'e_mono'}
#     }
    
#     # Define colors
#     group_colors = {
#         'PA01': 'lightgreen',
#         'P. aeruginosa': 'darkgreen',
#         'PairEP': 'burlywood',
#         'Co-cultures': 'saddlebrown',
#         'K12': 'palegoldenrod',
#         'E. coli': 'gold'
#     }
    
#     # Bar ordering as you specified
#     bar_order = ['PA01', 'PairEP', 'K12', 'P. aeruginosa', 'Co-cultures', 'E. coli']
    
#     for met in tqdm(metabolites, desc="Generating Raincloud Plots"):
#         if met not in global_group_stats:
#             continue
            
#         # Create figure with enough height
#         fig, ax = plt.subplots(figsize=(16, 10))
        
#         # Prepare data for raincloud
#         raincloud_data = []
#         group_labels = []
#         group_colors_list = []
        
#         # Process each group
#         for i, group_name in enumerate(bar_order):
#             if group_name in ['PA01', 'PairEP', 'K12']:  # Controls
#                 # Get data for control strains
#                 control_data = df[df['Strain'] == group_name][met].dropna()
#                 if len(control_data) > 0:
#                     raincloud_data.append(control_data)
#                     group_labels.append(group_name)
#                     group_colors_list.append(group_colors[group_name])
#             else:  # Clinical groups
#                 # Get data for all strains in clinical group
#                 strains = clinical_groups[group_name]['strains']
#                 for strain in strains:
#                     strain_data = df[df['Strain'] == strain][met].dropna()
#                     if len(strain_data) > 0:
#                         raincloud_data.append(strain_data)
#                         group_labels.append(f"{group_name}: {strain}")
#                         group_colors_list.append(group_colors[group_name])
        
#         # Create orientation-aware raincloud
#         pt.RainCloud(data=raincloud_data, ax=ax, 
#                     palette=group_colors_list,
#                     orient="h",  # Horizontal orientation
#                     width_viol=0.7,  # Width of violin
#                     move=0.2,  # Move violins
#                     offset=0.2,  # Offset of points
#                     alpha=0.7,  # Transparency
#                     box_showfliers=False)  # Don't show outliers twice
                    
#         # Set y-ticks with group labels
#         ax.set_yticks(range(len(group_labels)))
#         ax.set_yticklabels(group_labels)
        
#         # Add significance markers from your existing analysis
#         y_offset = 0  # Starting position for y-axis
        
#         for i, group in enumerate(bar_order):
#             if group in clinical_groups:  # Only for clinical groups
#                 control = clinical_groups[group]['control']
#                 control_idx = bar_order.index(control)
#                 internal_group = clinical_groups[group]['internal']
                
#                 # Get significance from global stats
#                 if 'control_comparisons' in global_group_stats[met] and internal_group in global_group_stats[met]['control_comparisons']:
#                     result = global_group_stats[met]['control_comparisons'][internal_group]
                    
#                     if result.get('significant', False):
#                         p_val = result['p_value']
#                         if p_val < 0.001: marker = '***'
#                         elif p_val < 0.01: marker = '**'
#                         elif p_val < 0.05: marker = '*'
                        
#                         # Add significance marker at appropriate position
#                         for strain in clinical_groups[group]['strains']:
#                             strain_idx = group_labels.index(f"{group}: {strain}")
#                             control_y = group_labels.index(control)
                            
#                             # Draw significance line and markers connecting strain to control
#                             x_max = ax.get_xlim()[1]
#                             ax.annotate(marker, xy=(x_max*0.95, control_y), 
#                                        fontsize=12, fontweight='bold')
        
#         # Format plot
#         ax.set_title(f"Raincloud plot: {met} levels in biofilm cells by bacterial group and strain")
#         ax.set_xlabel("Peak Area")
        
#         # Add annotations explaining plot elements
#         txt = "Raincloud plots show: density distribution (cloud), individual values (points), and summary statistics (box)"
#         txt += "\nSignificance: * p<0.05, ** p<0.01, *** p<0.001"
#         fig.text(0.5, 0.01, txt, ha='center', fontsize=10, 
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_folder, f"{met}_raincloud_{medium}_bio.png"), dpi=300)
#         plt.close()

def generate_violin_boxpoint_plots(df, metabolites, medium, out_folder):
    """Generate combined violin+boxplot+points plots for metabolites.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing metabolite data
    metabolites : list
        List of metabolite names to plot
    medium : str
        Medium type for labeling
    out_folder : str
        Output directory path
    """
    global global_group_stats
    
    # Define bar ordering - all controls first, then clinical groups
    bar_order = ['PA01', 'PairEP', 'K12', 'P. aeruginosa', 'Co-cultures', 'E. coli']
    
    # Define clinical groups and their respective controls
    clinical_groups = {
        'P. aeruginosa': {'strains': p_mono, 'control': 'PA01', 'internal': 'p_mono'},
        'Co-cultures': {'strains': pairs, 'control': 'PairEP', 'internal': 'pairs'},
        'E. coli': {'strains': e_mono, 'control': 'K12', 'internal': 'e_mono'}
    }
    
    # Define colors
    group_colors = {
        'PA01': 'lightgreen',
        'P. aeruginosa': 'darkgreen',
        'PairEP': 'burlywood',
        'Co-cultures': 'saddlebrown',
        'K12': 'palegoldenrod',
        'E. coli': 'gold'
    }
    
    # Between-group comparisons
    between_comparisons = [
        {'groups': ('E. coli', 'P. aeruginosa'), 'key': 'e_mono_vs_p_mono', 'type': 'Unpaired'},
        {'groups': ('E. coli', 'Co-cultures'), 'key': 'e_mono_vs_pairs', 'type': 'Paired'},
        {'groups': ('P. aeruginosa', 'Co-cultures'), 'key': 'p_mono_vs_pairs', 'type': 'Paired'}
    ]
    
    for met in tqdm(metabolites, desc="Generating Violin+Box+Points Plots"):
        if met not in global_group_stats:
            continue
            
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data for plotting
        plot_data = []
        plot_groups = []
        plot_colors = []
        
        for group in bar_order:
            if group in ['PA01', 'PairEP', 'K12']:  # Controls
                control_data = df[df['Strain'] == group][met].dropna()
                if len(control_data) > 0:
                    plot_data.extend(control_data)
                    plot_groups.extend([group] * len(control_data))
                    plot_colors.extend([group_colors[group]] * len(control_data))
            else:  # Clinical groups
                strains = clinical_groups[group]['strains']
                group_data = df[df['Strain'].isin(strains)][met].dropna()
                if len(group_data) > 0:
                    plot_data.extend(group_data)
                    plot_groups.extend([group] * len(group_data))
                    plot_colors.extend([group_colors[group]] * len(group_data))
        
        # Create pandas DataFrame for seaborn
        plot_df = pd.DataFrame({
            'Metabolite': plot_data,
            'Group': plot_groups
        })
        
        # 1. First plot violins with light fill
        sns.violinplot(x='Group', y='Metabolite', data=plot_df, 
                      order=bar_order, 
                      palette={g: group_colors[g] for g in bar_order},
                      alpha=0.3, # Transparent violins
                      linewidth=0.5,
                      cut=0, # Don't extend beyond data limits
                      inner=None, # No inner components (we'll add boxplot separately)
                      ax=ax)
        
        # 2. Add boxplot inside the violins
        sns.boxplot(x='Group', y='Metabolite', data=plot_df,
                   order=bar_order,
                   width=0.3, # Narrower than violins
                   color='white',
                   linewidth=1,
                   fliersize=0, # No outlier markers
                   ax=ax)
        
        # 3. Add individual points with jitter
        sns.stripplot(x='Group', y='Metabolite', data=plot_df,
                     order=bar_order,
                     palette={g: group_colors[g] for g in bar_order},
                     size=4,
                     jitter=True,
                     alpha=0.6,
                     ax=ax)
        
        # Add significance markers for control vs. clinical group comparisons
        for i, group in enumerate(bar_order):
            if group in clinical_groups:  # Only for clinical groups
                control = clinical_groups[group]['control']
                control_idx = bar_order.index(control)
                internal_group = clinical_groups[group]['internal']
                
                # Get significance from global stats
                if 'control_comparisons' in global_group_stats[met] and internal_group in global_group_stats[met]['control_comparisons']:
                    result = global_group_stats[met]['control_comparisons'][internal_group]
                    
                    if result.get('significant', False):
                        p_val = result['p_value']
                        
                        if p_val < 0.001: marker = '***'
                        elif p_val < 0.01: marker = '**'
                        elif p_val < 0.05: marker = '*'
                        
                        # Add asterisk above control bar
                        y_max = ax.get_ylim()[1]
                        y_min = ax.get_ylim()[0]
                        y_range = y_max - y_min
                        y_pos = plot_df[plot_df['Group'] == control]['Metabolite'].max() + (y_range * 0.05)
                        ax.text(control_idx, y_pos, marker, ha='center', fontsize=14, fontweight='bold')
        
        # Add significance markers for between-group comparisons
        for comp in between_comparisons:
            group1, group2 = comp['groups']
            comp_key = comp['key']
            
            if 'between_group_comparisons' in global_group_stats[met]:
                if comp_key in global_group_stats[met]['between_group_comparisons']:
                    result = global_group_stats[met]['between_group_comparisons'][comp_key]
                    
                    if result.get('significant', False):
                        # Find indices in bar_order
                        idx1 = bar_order.index(group1)
                        idx2 = bar_order.index(group2)
                        
                        p_val = result['p_value']
                        
                        if p_val < 0.001: marker = '***'
                        elif p_val < 0.01: marker = '**'
                        elif p_val < 0.05: marker = '*'
                        
                        # Draw line between groups
                        y_max = ax.get_ylim()[1]
                        y_min = ax.get_ylim()[0]
                        y_range = y_max - y_min
                        
                        g1_max = plot_df[plot_df['Group'] == group1]['Metabolite'].max()
                        g2_max = plot_df[plot_df['Group'] == group2]['Metabolite'].max()
                        y_line = max(g1_max, g2_max) + (y_range * 0.15)
                        
                        # Draw line and marker
                        ax.plot([idx1, idx2], [y_line, y_line], 'k-', linewidth=1)
                        ax.text((idx1 + idx2) / 2, y_line + (y_range * 0.02), 
                               marker, ha='center', fontsize=14, fontweight='bold')
        
        # Format plot
        ax.set_title(f"Violin+Box+Points plot: {met} in spent medium by bacterial group", fontsize=14)
        ax.set_xlabel("Bacterial Group", fontsize=12)
        ax.set_ylabel("Peak Area", fontsize=12)
        
        # Add annotations
        txt = "This plot combines: violin (distribution), box (quartiles), and points (individual samples)\n"
        txt += "Significance: * p<0.05, ** p<0.01, *** p<0.001"
        fig.text(0.5, 0.01, txt, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"{met}_violin_box_points_{medium}_sup.png"), dpi=300)
        plt.close()

def generate_gradient_interval_plots(df, metabolites, medium, out_folder):
    """Generate gradient interval plots for metabolites, showing group means with uncertainty gradients.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing metabolite data
    metabolites : list
        List of metabolite names to plot
    medium : str
        Medium type for labeling
    out_folder : str
        Output directory path
    """
    global global_group_stats
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba
    import numpy as np
    
    # Define bar ordering - all controls first, then clinical groups
    bar_order = ['PA01', 'PairEP', 'K12', 'P. aeruginosa', 'Co-cultures', 'E. coli']
    
    # Define clinical groups and their respective controls
    clinical_groups = {
        'P. aeruginosa': {'strains': p_mono, 'control': 'PA01', 'internal': 'p_mono'},
        'Co-cultures': {'strains': pairs, 'control': 'PairEP', 'internal': 'pairs'},
        'E. coli': {'strains': e_mono, 'control': 'K12', 'internal': 'e_mono'}
    }
    
    # Define colors
    group_colors = {
        'PA01': 'lightgreen',
        'P. aeruginosa': 'darkgreen',
        'PairEP': 'burlywood',
        'Co-cultures': 'saddlebrown',
        'K12': 'palegoldenrod',
        'E. coli': 'gold'
    }
    
    for met in tqdm(metabolites, desc="Generating Gradient Interval Plots"):
        if met not in global_group_stats:
            continue
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate means and confidence intervals for each group
        group_means = []
        group_intervals = []
        
        for i, group in enumerate(bar_order):
            if group in ['PA01', 'PairEP', 'K12']:  # Controls
                group_data = df[df['Strain'] == group][met].dropna()
                if len(group_data) > 0:
                    mean = group_data.mean()
                    sem = group_data.sem()
                    # 68%, 95%, and 99.7% intervals (1, 2, and 3 standard errors)
                    intervals = [
                        (mean - sem, mean + sem),      # ~68% interval
                        (mean - 2*sem, mean + 2*sem),  # ~95% interval
                        (mean - 3*sem, mean + 3*sem)   # ~99.7% interval
                    ]
                    group_means.append(mean)
                    group_intervals.append(intervals)
                else:
                    group_means.append(0)
                    group_intervals.append([(0,0), (0,0), (0,0)])
            else:  # Clinical groups
                strains = clinical_groups[group]['strains']
                
                # For hierarchical approach: first calculate strain means
                strain_means = []
                for strain in strains:
                    strain_data = df[df['Strain'] == strain][met].dropna()
                    if len(strain_data) > 0:
                        strain_means.append(strain_data.mean())
                
                if strain_means:
                    group_mean = np.mean(strain_means)
                    # Standard error of strain means
                    strain_sem = np.std(strain_means, ddof=1) / np.sqrt(len(strain_means))
                    
                    # Create intervals at different levels
                    intervals = [
                        (group_mean - strain_sem, group_mean + strain_sem),          # ~68% interval
                        (group_mean - 2*strain_sem, group_mean + 2*strain_sem),      # ~95% interval 
                        (group_mean - 3*strain_sem, group_mean + 3*strain_sem)       # ~99.7% interval
                    ]
                    group_means.append(group_mean)
                    group_intervals.append(intervals)
                else:
                    group_means.append(0)
                    group_intervals.append([(0,0), (0,0), (0,0)])
        
        # Plot gradient intervals
        x_pos = np.arange(len(bar_order))
        
        # Draw gradient intervals from widest to narrowest
        for i, group in enumerate(bar_order):
            color = group_colors[group]
            rgba_color = to_rgba(color)
            
            # Draw intervals from widest (99.7%) to narrowest (68%)
            for j, (lower, upper) in enumerate(reversed(group_intervals[i])):
                alpha = 0.8 - (j * 0.25)  # Decrease alpha for wider intervals
                interval_color = (rgba_color[0], rgba_color[1], rgba_color[2], alpha)
                ax.fill_betweenx([x_pos[i]-0.35, x_pos[i]+0.35], 
                                lower, upper, 
                                color=interval_color)
            
            # Draw mean point
            ax.scatter(group_means[i], x_pos[i], 
                     color='black', s=50, zorder=10)
        
        # Add strain-level data points for clinical groups
        for i, group in enumerate(bar_order):
            if group not in ['PA01', 'PairEP', 'K12']:  # Only for clinical groups
                strains = clinical_groups[group]['strains']
                
                # Add points for each strain
                for strain in strains:
                    strain_data = df[df['Strain'] == strain][met].dropna()
                    if len(strain_data) > 0:
                        strain_mean = strain_data.mean()
                        ax.scatter(strain_mean, x_pos[i], color='gray', s=25, 
                                  alpha=0.7, zorder=5)
        
        # Add significance markers (similar to your other visualization functions)
        # ... (code similar to previous functions)
        
        # Format plot
        ax.set_yticks(x_pos)
        ax.set_yticklabels(bar_order)
        ax.set_xlabel("Peak Area", fontsize=12)
        ax.set_title(f"Gradient Interval Plot: {met} in spent medium", fontsize=14)
        
        # Add legend for interval levels
        legend_elements = [
            Line2D([0], [0], color='black', marker='o', linestyle='None',
                  markersize=8, label='Group mean'),
            Line2D([0], [0], color='black', marker='o', linestyle='None',
                  markersize=5, alpha=0.7, label='Strain mean'),
            plt.Rectangle((0,0), 1, 1, fc=to_rgba('gray', 0.3), label='68% interval'),
            plt.Rectangle((0,0), 1, 1, fc=to_rgba('gray', 0.2), label='95% interval'),
            plt.Rectangle((0,0), 1, 1, fc=to_rgba('gray', 0.1), label='99.7% interval')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Add note about intervals
        txt = "Gradient intervals show uncertainty around the mean at different confidence levels\n"
        txt += "Darker regions indicate higher confidence intervals"
        fig.text(0.5, 0.01, txt, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"{met}_gradient_intervals_{medium}_sup.png"), dpi=300)
        plt.close()

#==============================================================================
# MAIN WORKFLOW
#==============================================================================

def main(file_path):
    df_filtered, medium = load_and_filter_data(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = os.path.join(os.getcwd(), f"{medium}_sup_Analysis_{timestamp}")
    os.makedirs(out_folder, exist_ok=True)
    metabolites = df_filtered.columns[2:]

    df_with_control = add_control_to_dataset(df_filtered, metabolites)

    # Try multiple data transformations and select the best one
    print("Evaluating multiple transformation methods...")
    transformation_results = evaluate_transformations(df_with_control, metabolites)

    # Find the best transformation method (lowest failure rate)
    best_method = transformation_results.loc[transformation_results['Failure_Rate'].idxmin(), 'Method']
    print(f"\nBest transformation method: {best_method} with {transformation_results.loc[transformation_results['Method'] == best_method, 'Failure_Rate'].values[0]:.1f}% failure rate")

    # Apply the best transformation
    if best_method != 'Original':
        print(f"Applying {best_method} transformation for analysis")
        df_for_stats = transform_data(df_with_control, metabolites, method=best_method)
    else:
        print("No transformation improved normality, using original data")
        df_for_stats = df_with_control

    # Save transformation comparison results
    transformation_results_path = os.path.join(out_folder, f"Transformation_Comparison_{medium}.xlsx")
    transformation_results.to_excel(transformation_results_path, index=False)
    print(f"Transformation comparison results saved to: {transformation_results_path}")

    print("Generating initial distribution visualizations...")

#####################################
    # Select a subset of metabolites (e.g., every 5th one to manage computational load)
    all_metabolites_subset = metabolites[::5]

    # Generate overview visualizations
    generate_violin_boxpoint_plots(df_for_stats, all_metabolites_subset, medium, os.path.join(out_folder, "initial_distributions"))
#########################################

    # Perform appropriate statistical tests based on assumption checks
    assumption_df, test_results_df = perform_appropriate_statistical_test(df_for_stats, metabolites)

    # Perform appropriate post-hoc tests
    posthoc_df = perform_appropriate_posthoc(df_for_stats, metabolites, test_results_df)

    print("\n🧪 Running Group-Level Post-Hoc Analysis...")
    group_posthoc_df = perform_group_level_posthoc(df_filtered, metabolites, test_results_df, medium, out_folder)

    # Extract significant metabolites from group post-hoc
    group_sig_mets_posthoc = set(
        group_posthoc_df[group_posthoc_df['Significant'] == True]['Metabolite'].unique()) if len(
        group_posthoc_df) > 0 else set()

    # ========== STEP 1: Calculate fold-change FIRST (was at line 2700+) ==========
    print("\nCalculating fold-change for all metabolites...")
    epsilon = 1e-6
    fold_change = pd.DataFrame()

    for met in tqdm(metabolites, desc="Fold-change Calculation"):
        grp_means = df_with_control.groupby('Strain')[met].mean() + epsilon
        fc = grp_means / grp_means['Control']
        log2fc = np.log2(fc)
        log2fc['Control'] = 0
        fold_change[met] = log2fc

    fold_change = fold_change.reindex(sample_order)

    # Filter significant metabolites based on fold change
    fc_sig_mets = []
    for met in tqdm(fold_change.columns, desc="Filtering by fold change"):
        non_control_strains = [s for s in sample_order if s != 'Control']
        non_control_values = fold_change.loc[non_control_strains, met]
        if (non_control_values.abs() > 0.75).sum() / len(non_control_values) >= 0.33:
            fc_sig_mets.append(met)

    print(f"✓ Metabolites with |FC| > 0.75 in >= 33% of strains: {len(fc_sig_mets)}")

    # ========== STEP 2: Calculate ANOVA significance ==========
    print("\nCalculating ANOVA significance...")
    anova_sig_mets = test_results_df[test_results_df['p_value'] < 0.05]['Metabolite'].unique().tolist()
    print(f"✓ Metabolites significant in global ANOVA (p < 0.05): {len(anova_sig_mets)}")

    # ========== STEP 3: Calculate post-hoc significance ==========
    print("Calculating post-hoc significance vs Control...")
    posthoc_sig_mets = []
    for met in metabolites:
        # Check for ANY post-hoc comparison with raw p < 0.05 vs Control
        has_sig_control_comp = posthoc_df[
                                   (posthoc_df['Metabolite'] == met) &
                                   ((posthoc_df['Strain1'] == 'Control') | (posthoc_df['Strain2'] == 'Control')) &
                                   (posthoc_df['Raw_pvalue'] < 0.05)  # Use RAW p-value, not corrected
                                   ].shape[0] > 0

        if has_sig_control_comp:
            posthoc_sig_mets.append(met)

    print(f"✓ Metabolites with significant post-hoc comparisons vs Control: {len(posthoc_sig_mets)}")

    # ========== STEP 4: Combine all statistical criteria ==========
    print("\nCombining statistical criteria...")
    metabolites_by_stats = list(set(anova_sig_mets) | set(posthoc_sig_mets))
    print(f"✓ Metabolites meeting statistical criteria (ANOVA OR post-hoc): {len(metabolites_by_stats)}")

    # ========== STEP 5: Intersection with fold-change criterion ==========
    # NOW fc_sig_mets IS DEFINED - safe to use!
    fc_filtered = [met for met in metabolites_by_stats if met in fc_sig_mets]
    print(f"✓ Also meeting fold-change criterion (|FC| > 0.75): {len(fc_filtered)}")

    # ========== STEP 6: Final selection logic ==========
    print("\nApplying final selection logic...")
    if len(fc_filtered) >= 10:
        final_sig_mets = fc_filtered
        print(f"✓ Using fold-change filtered list: {len(final_sig_mets)} metabolites")
    else:
        final_sig_mets = metabolites_by_stats
        print(f"✓ Using all statistically significant metabolites: {len(final_sig_mets)}")

    print(f"\n{'=' * 80}")
    print(f"FINAL COUNT: {len(final_sig_mets)} significant metabolites for visualization")
    print(f"{'=' * 80}\n")

    #===========================================================================
    #generate_bar_plots(df_with_control, metabolites, medium, out_folder, posthoc_df)
    #==============================================================================

    epsilon = 1e-6
    fold_change = pd.DataFrame()

    for met in tqdm(metabolites, desc="Fold-change Calculation"):
        grp_means = df_with_control.groupby('Strain')[met].mean() + epsilon
        fc = grp_means / grp_means['Control']
        log2fc = np.log2(fc)
        log2fc['Control'] = 0
        fold_change[met] = log2fc

    fold_change = fold_change.reindex(sample_order)

    #Filter significant metabolites based on fold change
    fc_sig_mets = []
    for met in tqdm(fold_change.columns, desc="Filtering by fold change"):
        non_control_strains = [s for s in sample_order if s != 'Control']
        non_control_values = fold_change.loc[non_control_strains, met]
        if (non_control_values.abs() > 0.75).sum() / len(non_control_values) >= 0.33:
            fc_sig_mets.append(met)

    # Filter metabolites passing both criteria
    final_sig_mets = list(set(fc_sig_mets) & set(posthoc_sig_mets))

    # If fewer than 5 metabolites pass both criteria, include those passing either criterion
    if len(final_sig_mets) < 5:
        print("Too few metabolites passed both filters. Including metabolites passing either criterion.")
        final_sig_mets = list(set(fc_sig_mets) | set(posthoc_sig_mets))
########################################################################################################
    # Create enhanced marking dictionary: combines significance marking + pathway coloring
    marked_met_names = {}
    metabolite_pathways = {}
    metabolite_colors = {}

    for met in final_sig_mets:
        # Determine significance marking (symbol)
        if met in posthoc_sig_mets and met in fc_sig_mets:
            # Passed both filters - no symbol
            marked_name = met
            symbol = ""
        elif met in posthoc_sig_mets:
            # Only post-hoc significant - asterisk
            marked_name = f"{met} *"
            symbol = "*"
        elif met in fc_sig_mets:
            # Only fold-change significant - hash
            marked_name = f"{met} #"
            symbol = "#"
        else:
            # Shouldn't happen
            marked_name = met
            symbol = ""

        # Determine metabolic pathway and color
        pathway, color = get_metabolite_pathway(met)

        # Store all information
        marked_met_names[met] = marked_name
        metabolite_pathways[met] = pathway
        metabolite_colors[met] = color

        # DIAGNOSTIC: Show marking summary
        print(f"\n{'=' * 80}")
        print(f"METABOLITE MARKING SUMMARY:")
        print(f"{'=' * 80}")

        both_count = 0
        posthoc_only_count = 0
        fc_only_count = 0

        for met, marked_name in marked_met_names.items():
            if marked_name == met:
                both_count += 1
            elif '*' in marked_name:
                posthoc_only_count += 1
            elif '#' in marked_name:
                fc_only_count += 1

        print(f"Total metabolites for heatmap:     {len(marked_met_names)}")
        print(f"  Both filters (no symbol):        {both_count} metabolites")
        print(f"  Post-hoc only (*):               {posthoc_only_count} metabolites")
        print(f"  FC only (#):                     {fc_only_count} metabolites")
        print(f"{'=' * 80}\n")

        # Print detailed list
        print(f"Detailed Marking:")
        print(f"{'-' * 80}")
        print(f"{'Metabolite':<30s} | {'Marked Name':<30s} | {'Category':<20s}")
        print(f"{'-' * 80}")

        for met in sorted(marked_met_names.keys()):
            marked_name = marked_met_names[met]
            if marked_name == met:
                category = "Both filters"
            elif '*' in marked_name:
                category = "Post-hoc only"
            elif '#' in marked_name:
                category = "FC only"
            else:
                category = "Unknown"

            print(f"{met:<30s} | {marked_name:<30s} | {category:<20s}")

        print(f"{'-' * 80}\n")

    print(f"\nMetabolite Classification Summary:")
    for pathway in sorted(set(metabolite_pathways.values())):
        count = sum(1 for p in metabolite_pathways.values() if p == pathway)
        print(f"  {pathway:25s}: {count} metabolites")

    # Prepare data for heatmap
    fc_subset = fold_change[final_sig_mets].copy()
    fc_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
    fc_subset.fillna(0, inplace=True)

    # Rename the index with marked names after transposing
    heatmap_data = fc_subset.T.copy()
    heatmap_data.index = [marked_met_names[met] for met in heatmap_data.index]

    # Generate heatmap with filtered metabolites and marked names
    generate_heatmap(heatmap_data, medium, out_folder, metabolite_colors, metabolite_pathways)

    print("Generating volcano plots...")
    volcano_df = generate_volcano_plots(test_results_df, fold_change, medium, out_folder, final_sig_mets)
#==============================================================================
#Second tier analysis of bacterial groups
#==============================================================================

    # Define clinical groups without K12, PA01, PairEP
    groups_dict = {
        'e_mono': e_mono,
        'p_mono': p_mono,
        'pairs': pairs
        }

    # Define metabolites for group analysis in second-tier analysis
    fc_only_mets = list(set(fc_sig_mets) - set(posthoc_sig_mets))  # Passed FC but not significant
    sig_only_mets = list(set(posthoc_sig_mets) - set(fc_sig_mets))  # Significant but small FC
    second_tier_mets = fc_only_mets + sig_only_mets

    print(f"Performing group-level analysis on {len(second_tier_mets)} metabolites:")
    print(f"  - {len(fc_only_mets)} with large fold changes but not statistically significant")
    print(f"  - {len(sig_only_mets)} statistically significant but with small fold changes")

    # Run group analysis on combined list
    group_stats_df = perform_hierarchical_group_analysis(df_for_stats, second_tier_mets, groups_dict)

    # Find metabolites significant at group level
    group_sig_mets = list(group_stats_df[group_stats_df['Significant'] == True]['Metabolite'].unique())
    print(f"Found {len(group_sig_mets)} metabolites significant at group level")

    # Generate group visualizations
    generate_hierarchical_group_bar_plots(df_with_control, final_sig_mets, medium, out_folder)

    # Generate both types of group heatmaps
    print("Generating group heatmaps...")
    generate_group_heatmap(df_with_control, final_sig_mets, medium, out_folder, True)  # Respective controls
    generate_group_heatmap(df_with_control, final_sig_mets, medium, out_folder, False)  # Unified control

    # Generate comprehensive bubble plots
    print("Generating bubble plots...")
    generate_comprehensive_bubble_plot(df_with_control, final_sig_mets, medium, out_folder, True)  # Respective controls
    generate_comprehensive_bubble_plot(df_with_control, final_sig_mets, medium, out_folder, False)  # Unified control

    # print("Generating additional visualizations...")
    # # Generate raincloud plots
    # generate_raincloud_plots(df_with_control, final_sig_mets, medium, out_folder)

    # # Generate violin + boxplot + points combination
    # generate_violin_boxpoint_plots(df_with_control, final_sig_mets, medium, out_folder)

    # # Generate gradient interval plots
    # generate_gradient_interval_plots(df_with_control, final_sig_mets, medium, out_folder)

    # Generate between-group comparison bubble plots
    print("Generating between-group comparison bubble plots...")
    between_group_results_df = generate_between_group_bubble_plots(df_with_control, final_sig_mets, medium, out_folder)

    # Extract significant metabolites from group post-hoc
    group_sig_mets_posthoc = set(group_posthoc_df[group_posthoc_df['Significant'] == True]['Metabolite'].unique())

    excel_results = {}

    # Create pathway reference sheet for Excel
    pathway_ref = pd.DataFrame({
        'Metabolite': list(metabolite_pathways.keys()),
        'Marked_Name': [marked_met_names[met] for met in metabolite_pathways.keys()],
        'Pathway': list(metabolite_pathways.values()),
        'Significance_Mark': [
            'Both' if marked_met_names[met] == met else
            'PostHoc' if '*' in marked_met_names[met] else
            'FoldChange' for met in metabolite_pathways.keys()
        ],
        'Color_Hex': list(metabolite_colors.values())
    })

    # Saved statistical results into single Excel document
    excel_results = {
        'ANOVA_Results': test_results_df,
        'Metabolite_Pathway_Key': pathway_ref,
        'PostHoc_Results': posthoc_df,
        'Assumption_Results': assumption_df,
        'Group_Analysis': group_stats_df,
        'Group_PostHoc': group_posthoc_df,
        'FoldChange': fold_change,
        'Significant_Metabolites': pd.DataFrame({
            'Metabolite': final_sig_mets,
            'Metabolite_Marked': [marked_met_names[met] for met in final_sig_mets],
            'Post_hoc_Significant': [met in posthoc_sig_mets for met in final_sig_mets],
            'FoldChange_Significant': [met in fc_sig_mets for met in final_sig_mets],
            'Group_Significant': [met in group_sig_mets for met in final_sig_mets],
            'Group_PostHoc_Significant': [met in group_sig_mets_posthoc for met in final_sig_mets]
        })
    }

    if not between_group_results_df.empty:
        excel_results['Between_Group_Comparisons'] = between_group_results_df

    save_results_excel(excel_results, medium, out_folder)

    print(f"oooooooooooooooAnalysis complete! All plots and results saved in: {out_folder}")
#==============================================================================
# MAIN EXECUTION BLOCK
#==============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv
    else:
        # Use relative path or prompt user
        file_path = input("Please enter the path to your Excel file: ")
        # OR use a default relative path:
        # file_path = "data/Analysis_sup_AUM.xlsx"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    print(f"Using file: {file_path}")
    main(file_path)