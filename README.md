# Contact
- Author: Serhat Karaman
- Mail: serhatkaramanworkmail@gmail.com
- [Technical Report] (https://docs.google.com/document/d/10a_I-SKVFP0zOk57_kVwWuTGvCxQkYCWTDc_MiDvKdc/edit?usp=sharing)

# Pusula Data Science Case - Medical Treatment Duration Prediction

A comprehensive data science pipeline for predicting medical treatment duration based on patient characteristics and medical history. This project implements advanced feature engineering, exploratory data analysis, and machine learning-ready data preparation for healthcare analytics.

## 🎯 Project Overview

This pipeline processes medical patient data to predict **treatment duration in sessions** (`TEDAVISURESI_SEANS_SAYI`) using patient demographics, medical conditions, treatment types, and historical patterns. The system implements sophisticated data preprocessing, feature engineering, and comprehensive exploratory data analysis.

### Key Features

- **Patient-Specific Data Processing**: Advanced missing value imputation using individual patient history
- **Comprehensive Feature Engineering**: Creates 200+ features from raw medical data
- **Multiple Encoding Strategies**: Intelligent categorical encoding based on cardinality
- **Advanced EDA**: 100+ visualizations and statistical analyses
- **Production-Ready**: Robust error handling, comprehensive logging, and automated workflows

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Virtual environment** support
- **Excel file**: `data/Talent_Academy_Case_DT_2025.xlsx`

### Installation & Setup

#### Unix/Linux/macOS
```bash
# Complete setup and run
make setup
make run-all

# View results
make show-results
```

#### Windows (Command Prompt)
```cmd
# Complete setup and run
run_pipeline.bat setup
run_pipeline.bat run-all

# View results
run_pipeline.bat show-results
```

#### Windows (PowerShell)
```powershell
# Complete setup and run
.\run_pipeline.ps1 setup
.\run_pipeline.ps1 run-all

# View results
.\run_pipeline.ps1 show-results
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Unix
# OR
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt
pip install category_encoders

# Run complete pipeline
python main.py
```

## 📊 Pipeline Architecture

The project consists of three main processing pipelines that work sequentially to transform raw medical data into machine learning-ready datasets.

### 1. Preprocessing Pipeline (`preprocessing_pipeline.py`)

**Purpose**: Cleans and standardizes raw medical data with intelligent missing value handling.

#### Key Functions:
- **`load_data()`**: Loads Excel data with proper encoding and validation
- **`standardize_column_names()`**: Converts column names to consistent uppercase format
- **`handle_missing_values()`**: Patient-specific imputation strategy
  - Uses individual patient history when available
  - Falls back to population statistics when necessary
  - Handles categorical and numerical data differently
- **`lowercase_string_data()`**: Standardizes text data for consistency
- **`parse_duration_columns()`**: Extracts numerical values from duration strings

#### Output:
- **Shape**: 2,235 rows × 16 columns
- **Quality**: Zero missing values after intelligent imputation
- **Location**: `data/preprocessing/preprocessed_data.csv`

### 2. Feature Engineering Pipeline (`feature_engineering_pipeline.py`)

**Purpose**: Creates advanced features and prepares data for machine learning through comprehensive encoding and scaling.

#### Key Functions:

##### Feature Creation:
- **`create_list_features()`**: Converts comma-separated medical data to structured lists
- **`create_count_features()`**: Counts diagnoses, allergies, treatments, and conditions
- **`create_chronic_disease_features()`**: Groups diseases and calculates risk scores
- **`create_treatment_features()`**: Categorizes treatments and creates treatment flags
- **`create_age_features()`**: Creates age bins and demographic categories
- **`process_kangrubu_features()`**: Extracts blood type and Rh factor information
- **`create_repetitive_data_feature()`**: Identifies duplicate records in dataset
- **`create_hastano_count_feature()`**: Counts patient visit frequency

##### Data Transformation:
- **`encode_categorical_features()`**: Intelligent multi-strategy encoding
  - **Binary columns** (≤2 unique): Label encoding
  - **Low cardinality** (≤10 unique): One-hot encoding
  - **Medium cardinality** (11-50 unique): Target + binary encoding
  - **High cardinality** (>50 unique): Hashing + count encoding
  - **Text columns**: TF-IDF encoding for medical terms

- **`scale_numerical_features()`**: Standardizes numerical features
  - Excludes ID columns and binary flags
  - Creates scaled versions alongside originals
  - Supports multiple scaling methods (Standard, MinMax, Robust)

- **`remove_useless_columns()`**: Removes redundant and low-value features
  - Original categorical columns with encoded versions
  - Zero/low variance features
  - Highly correlated features (>0.95)
  - Non-ML compatible columns

- **`ensure_all_numeric()`**: Converts all remaining columns to numerical format

#### Output:
- **Feature Engineering Data**: 2,235 rows × 258 columns (all features)
- **Final ML Dataset**: 2,235 rows × 140 columns (cleaned, all numerical)
- **Location**: `data/feature_engineering/` and `data/data_final_version/`

### 3. EDA Pipeline (`eda_pipeline.py`)

**Purpose**: Performs comprehensive exploratory data analysis with advanced visualizations and statistical insights.

#### Key Functions:

##### Statistical Analysis:
- **`basic_statistical_analysis()`**: Dataset profiling and descriptive statistics
- **`distribution_analysis()`**: Distribution plots for all variable types
- **`correlation_analysis()`**: Pearson and Spearman correlation matrices
- **`outlier_analysis()`**: IQR and Z-score outlier detection

##### Advanced Analysis:
- **`target_variable_analysis()`**: Deep dive into treatment duration patterns
- **`categorical_analysis()`**: Chi-square tests and cross-tabulations
- **`feature_importance_analysis()`**: Random Forest and Mutual Information ranking
- **`log_transformation_analysis()`**: Skewness analysis and transformation recommendations
- **`outlier_investigation()`**: Detailed outlier characterization
- **`patient_pattern_analysis()`**: Patient visit frequency and behavior patterns
- **`correlation_based_feature_engineering()`**: Data-driven feature suggestions

#### Output:
- **106 analysis files** across 9 organized directories
- **29 high-quality visualizations** (300 DPI PNG files)
- **11 CSV data files** with statistical results
- **3 comprehensive reports** with insights and recommendations
- **Location**: `data/EDA_results/` with organized subdirectories

## 🛡️ Exception Handling & Logging

The project implements a robust, multi-layered approach to error handling and logging that ensures reliability and maintainability in production environments.

### Exception Handling Architecture

#### Custom Exception Classes (`case_src/exception/exception.py`)

The system defines specialized exceptions for different pipeline stages:

- **`DataLoadingException`**: File loading and data access errors
- **`DataSavingException`**: File writing and storage errors  
- **`DataValidationException`**: Data quality and schema validation errors
- **`PreprocessingException`**: Data cleaning and preprocessing errors
- **`FeatureEngineeringException`**: Feature creation and transformation errors

#### Pipeline Error Handler Decorator

Each pipeline function is wrapped with `@pipeline_error_handler()` that:

- **Captures Context**: Automatically logs the pipeline stage, function, and parameters
- **Enriches Errors**: Adds file paths, data shapes, and processing context
- **Maintains Stack Traces**: Preserves original error information for debugging
- **Enables Recovery**: Allows graceful degradation and continuation where possible

```python
@pipeline_error_handler("feature_engineering")
def create_chronic_disease_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Function automatically wrapped with comprehensive error handling
    # Context: stage, file paths, data characteristics logged on failure
```

#### Validation Framework

The system includes extensive validation functions:

- **`validate_file_path()`**: Ensures file paths exist and are accessible
- **`validate_dataframe()`**: Checks data quality, shape, and content requirements
- **`handle_exception()`**: Centralized exception processing and logging

### Logging System

#### Multi-Level Logging (`case_src/logging/logger.py`)

The logging system provides detailed traceability:

- **File-Based Logging**: All pipeline executions logged to timestamped files
- **Console Output**: Real-time progress and status updates
- **Structured Format**: Consistent timestamp, level, module, and message format
- **Automatic Rotation**: New log file for each pipeline execution

#### Log Levels and Content:

- **INFO**: Pipeline stages, successful operations, data shapes, feature counts
- **WARNING**: Non-critical issues, fallback operations, data quality notices
- **ERROR**: Pipeline failures, exception details, context information
- **DEBUG**: Detailed execution flow, intermediate results, performance metrics

#### Example Log Entry:
```
[2025-09-06 15:30:45,123] 156 case_src.pipeline.feature_engineering_pipeline - INFO - Creating count features...
[2025-09-06 15:30:45,130] 196 case_src.pipeline.feature_engineering_pipeline - INFO - Count features created: ['TANI_SAYI', 'ALERJI_SAYI', 'UYGULAMAYERLERI_SAYI', 'KRONIKHASTALIK_SAYI', 'TEDAVIADI_SAYI']
```

### Error Recovery Strategies

#### Graceful Degradation:
- **Column-by-Column Processing**: Encoding failures don't stop entire pipeline
- **Feature-by-Feature Creation**: Individual feature failures are isolated
- **Fallback Mechanisms**: Alternative encoding methods when primary fails
- **Continue on Warning**: Non-critical issues logged but don't halt processing

#### Data Integrity Protection:
- **Input Validation**: All data validated before processing
- **Output Verification**: Results checked after each major operation
- **Backup Strategies**: Original data preserved through pipeline stages
- **Rollback Capability**: Failed operations don't corrupt existing data

## 📁 File Structure

```
ds_case_pusula/
├── README.md                     # This documentation
├── WINDOWS_SETUP.md             # Windows-specific setup guide
├── Makefile                     # Unix/Linux/macOS automation
├── run_pipeline.bat             # Windows batch script
├── run_pipeline.ps1             # PowerShell script
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── setup.py                     # Package configuration
├── params.yaml                  # Pipeline parameters
├── schema.yaml                  # Data schema definition
│
├── case_src/                    # Source code package
│   ├── __init__.py
│   ├── pipeline/                # Pipeline modules
│   │   ├── __init__.py
│   │   ├── main_pipeline.py         # Main orchestrator pipeline
│   │   ├── preprocessing_pipeline.py # Data cleaning and standardization
│   │   ├── feature_engineering_pipeline.py # Feature creation and encoding
│   │   └── eda_pipeline.py          # Exploratory data analysis
│   │
│   ├── exception/               # Error handling framework
│   │   ├── __init__.py
│   │   └── exception.py             # Custom exceptions and handlers
│   │
│   ├── logging/                 # Logging configuration
│   │   ├── __init__.py
│   │   └── logger.py                # Logging setup and configuration
│   │
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── common.py                # Shared utilities
│       ├── analyze_utils/           # Data analysis utilities
│       ├── preprocess_utils/        # Preprocessing utilities
│       ├── visualize_utils/         # Visualization utilities
│       ├── eval_utils/              # Evaluation utilities
│       ├── main_utils/              # Main pipeline utilities
│       ├── setup_utils/             # Setup and configuration utilities
│       └── ml_utils/                # Machine learning utilities
│           ├── metric/              # Evaluation metrics
│           └── model/               # Model definitions
│
├── data/                        # Data directory (auto-created)
│   ├── Talent_Academy_Case_DT_2025.xlsx # Source data file
│   ├── preprocessing/               # Preprocessed data outputs
│   │   └── preprocessed_data.csv        # Clean, standardized data
│   ├── feature_engineering/         # Feature engineering outputs  
│   │   └── feature_engineering_data.csv # All engineered features (258 cols)
│   ├── data_final_version/          # Final ML-ready datasets
│   │   └── final_cleaned_data.csv       # Clean numerical data (140 cols)
│   └── EDA_results/                 # EDA analysis results
│       ├── 01_basic_statistics/         # Dataset profiling
│       ├── 02_distributions/            # Distribution analysis
│       ├── 03_correlations/             # Correlation matrices
│       ├── 04_target_analysis/          # Target variable analysis
│       ├── 05_categorical_analysis/     # Categorical variable analysis
│       ├── 06_outlier_analysis/         # Outlier detection and investigation
│       ├── 07_feature_importance/       # Feature ranking and suggestions
│       ├── 08_advanced_visualizations/  # Advanced plots and patterns
│       └── 09_summary_reports/          # Comprehensive analysis reports
│
├── notebooks/                   # Jupyter notebooks
│   ├── eda.ipynb                    # Exploratory data analysis notebook
│   └── preprocessing.ipynb          # Data preprocessing notebook
│
├── logs/                        # Pipeline execution logs (auto-created)
│   └── YYYY_MM_DD_HH_MM_SS.log     # Timestamped execution logs
│
└── templates/                   # Template files
    └── index.html                   # Web interface template
```

## 🔧 Pipeline Details

### Data Flow Architecture

The pipeline follows a sequential data transformation approach:

```
Raw Excel Data (13 columns)
    ↓ [Preprocessing Pipeline]
Cleaned Data (16 columns)
    ↓ [Feature Engineering Pipeline]
Feature-Rich Data (258 columns)
    ↓ [Column Removal & Cleaning]
ML-Ready Data (140 columns, all numerical)
    ↓ [EDA Pipeline]
Comprehensive Analysis & Visualizations
```

### Pipeline Stages

#### Stage 1: Data Preprocessing
**Input**: Raw Excel file with medical records  
**Processing**: 
- Column standardization and type conversion
- Patient-specific missing value imputation
- Text data standardization and cleaning
- Duration parsing and numerical extraction

**Output**: Clean, standardized dataset ready for feature engineering

#### Stage 2: Feature Engineering
**Input**: Preprocessed medical data  
**Processing**:
- **Medical Feature Creation**: Disease groupings, risk scores, treatment categories
- **Patient Analytics**: Visit counts, repetitive data analysis, blood group parsing
- **Categorical Encoding**: Multi-strategy encoding based on data characteristics
- **Numerical Scaling**: StandardScaler for ML compatibility
- **Data Cleaning**: Removal of redundant and low-value features

**Output**: Two datasets - comprehensive feature set and clean ML-ready data

#### Stage 3: Exploratory Data Analysis
**Input**: Feature-engineered dataset  
**Processing**:
- Statistical profiling and distribution analysis
- Correlation analysis and feature relationships
- Target variable deep-dive analysis
- Outlier detection and investigation
- Patient pattern analysis
- Feature importance ranking
- Advanced visualizations and insights

**Output**: Comprehensive EDA results with 100+ analysis files

## 🧠 Key Functions & Algorithms

### Advanced Feature Engineering

#### Patient-Specific Imputation Algorithm
```python
def _patient_specific_imputation(df, column):
    # For each missing value:
    # 1. Find other records for the same patient (HASTANO)
    # 2. Use patient's historical data (mode/mean) if available
    # 3. Fall back to population statistics if no patient history
    # 4. Preserves individual patient characteristics
```

#### Intelligent Categorical Encoding
The system automatically selects optimal encoding strategies:

- **Binary Variables**: Simple label encoding (0/1)
- **Low Cardinality** (≤10): One-hot encoding for full representation
- **Medium Cardinality** (11-50): Target encoding + binary encoding for efficiency
- **High Cardinality** (>50): Hashing + count encoding to prevent explosion
- **Text Data**: TF-IDF encoding for medical terminology

#### Blood Group Processing
```python
def parse_blood_group(blood_group_str):
    # Extracts: "A Rh+" → type="A", positive=1
    # Handles: "0 Rh-", "AB Rh+", missing values, format variations
    # Creates: KANGRUBU_TYPE and KANGRUBU_POSITIVE features
```

### Advanced EDA Algorithms

#### Skewness Analysis & Log Transformation
- Automatically identifies highly skewed variables (|skewness| > 1)
- Tests log transformation effectiveness
- Provides before/after normality comparisons
- Generates transformation recommendations

#### Outlier Investigation
- Multi-method outlier detection (IQR, Z-score)
- Patient characteristic analysis for outliers
- Treatment pattern analysis for extreme values
- Clinical significance assessment

#### Patient Pattern Analysis
- Visit frequency clustering and analysis
- High-frequency patient identification (top 10%)
- Chronic disease burden correlation with visit patterns
- Treatment duration patterns by patient groups

## 🛡️ Robust Error Handling

### Exception Handling Philosophy

The project implements a **"fail-safe, continue-when-possible"** approach to error handling that prioritizes data pipeline completion while maintaining data integrity.

#### Multi-Layer Error Protection:

1. **Input Validation Layer**
   ```python
   validate_file_path(file_path, must_exist=True)
   validate_dataframe(df, min_rows=1, min_cols=1)
   ```

2. **Pipeline Stage Protection**
   ```python
   @pipeline_error_handler("feature_engineering")
   def create_features(self, df):
       # Automatic error context capture
       # Graceful failure handling
       # Detailed error logging
   ```

3. **Individual Operation Safety**
   ```python
   try:
       # Process individual column/feature
   except Exception as col_error:
       logger.error(f"Failed to process {column}: {str(col_error)}")
       # Continue with other columns
   ```

#### Error Context Enrichment

When errors occur, the system automatically captures:
- **Pipeline stage** and specific function
- **Input data characteristics** (shape, types, sample values)
- **File paths** and processing context
- **Parameter values** and configuration
- **Stack trace** and root cause analysis

### Exception Recovery Strategies

#### Graceful Degradation:
- **Feature Engineering**: Individual feature failures don't stop pipeline
- **Encoding**: Failed encoding attempts use fallback methods
- **EDA**: Visualization failures don't prevent other analyses
- **Data Processing**: Column-by-column processing isolates failures

#### Intelligent Fallbacks:
- **Encoding Failures**: Switch to simpler encoding methods
- **Missing Data**: Use global statistics when patient data unavailable
- **Correlation Issues**: Handle edge cases in correlation calculations
- **Visualization Errors**: Skip problematic plots, continue with others

## 📝 Comprehensive Logging

### Logging Architecture

The logging system provides complete traceability and debugging capabilities for production environments.

#### Log File Organization:
- **Timestamped Files**: Each pipeline run creates a new log file
- **Centralized Location**: All logs stored in `logs/` directory
- **Structured Format**: Consistent formatting across all pipeline stages
- **Automatic Cleanup**: Old logs can be managed through utility commands

#### Log Content Strategy:

##### Informational Logging:
- **Pipeline Initialization**: Component startup and configuration
- **Data Loading**: File paths, data shapes, basic statistics
- **Processing Steps**: Feature creation counts, encoding summaries
- **Progress Tracking**: Stage completion, timing information
- **Results Summary**: Output shapes, feature counts, quality metrics

##### Warning Logging:
- **Data Quality Issues**: Missing values, unexpected formats
- **Fallback Operations**: When primary methods fail but alternatives succeed
- **Performance Notices**: Large datasets, memory usage warnings
- **Configuration Issues**: Missing files, parameter problems

##### Error Logging:
- **Complete Context**: Full error details with pipeline context
- **Data State**: What data was being processed when error occurred
- **Recovery Actions**: What the system attempted to do
- **Impact Assessment**: Whether pipeline can continue or must stop

#### Example Log Sequence:
```
[2025-09-06 15:30:45,123] INFO - FeatureEngineeringPipeline initialized
[2025-09-06 15:30:45,124] INFO - Created directory: data/feature_engineering
[2025-09-06 15:30:45,125] INFO - Loading preprocessed data from: data/preprocessing/preprocessed_data.csv
[2025-09-06 15:30:45,140] INFO - Preprocessed data loaded successfully. Shape: (2235, 16)
[2025-09-06 15:30:45,141] INFO - Creating list-based features...
[2025-09-06 15:30:45,145] INFO - TANILAR_LIST feature created
[2025-09-06 15:30:45,146] INFO - Creating count features...
[2025-09-06 15:30:45,152] INFO - Count features created: ['TANI_SAYI', 'ALERJI_SAYI', 'UYGULAMAYERLERI_SAYI']
```

### Production Monitoring

#### Key Metrics Logged:
- **Data Quality**: Missing values, duplicates, data types
- **Processing Performance**: Feature creation counts, encoding success rates
- **Memory Usage**: Dataset sizes, memory consumption
- **Execution Time**: Stage durations, bottleneck identification
- **Error Rates**: Failure frequencies, recovery success rates

#### Debugging Support:
- **Parameter Tracking**: All function parameters logged
- **State Snapshots**: Data characteristics at each stage
- **Decision Logging**: Why certain encoding/processing choices were made
- **Performance Profiling**: Time spent in each pipeline component

## 🎯 Target Variable & Use Case

### Medical Context

**Target Variable**: `TEDAVISURESI_SEANS_SAYI` (Treatment Duration in Sessions)

This represents the number of treatment sessions required for each patient's medical condition. Understanding treatment duration is crucial for:

- **Resource Planning**: Hospital capacity and staffing
- **Patient Care**: Setting realistic treatment expectations  
- **Cost Management**: Treatment cost estimation and budgeting
- **Quality Metrics**: Identifying optimal treatment protocols

### Predictive Features

The pipeline creates features across multiple medical domains:

#### Patient Demographics:
- Age categories and numerical age
- Gender encoding
- Blood group analysis (type and Rh factor)
- Nationality patterns

#### Medical History:
- Chronic disease counts and risk scoring
- Disease category groupings
- Multi-morbidity flags
- Medical complexity indicators

#### Treatment Characteristics:
- Treatment type categorization
- Application area analysis
- Treatment frequency patterns
- Department specializations

#### Patient Behavior:
- Visit frequency analysis
- Treatment adherence patterns
- Hospital utilization metrics
- Care continuity indicators

## 🚀 Getting Started

### Environment Setup

1. **Clone or download** the project
2. **Place source data** in `data/Talent_Academy_Case_DT_2025.xlsx`
3. **Run setup** using platform-specific commands
4. **Execute pipeline** and review results

### Development Workflow

#### For Data Scientists:
```bash
make setup                    # Initial setup
make run-all                  # Full pipeline execution
make show-results            # Review comprehensive results
make eda                     # Re-run analysis with new insights
```

#### For ML Engineers:
```bash
make feature-engineering     # Generate ML-ready features
make check-final            # Validate numerical dataset
make backup                 # Backup current results
# Use data/data_final_version/final_cleaned_data.csv for modeling
```

#### For Analysts:
```bash
make preprocessing          # Clean data preparation
make eda                   # Comprehensive analysis
# Review data/EDA_results/ for insights and visualizations
```

### Customization

#### Pipeline Parameters:
- **Encoding strategies**: Modify cardinality thresholds in `encode_categorical_features()`
- **Scaling methods**: Change scaler type in `scale_numerical_features()`
- **Feature selection**: Adjust removal criteria in `remove_useless_columns()`
- **EDA depth**: Modify analysis scope in individual EDA functions

#### Output Locations:
- **Data paths**: Update directory variables in pipeline `__init__` methods
- **Log locations**: Modify `LOG_FILE_PATH` in logging configuration
- **EDA organization**: Customize directory structure in `create_output_directories()`

## 📊 Expected Outputs

After running the complete pipeline:

### Data Outputs:
- **Preprocessed Data**: 2,235 rows × 16 columns (clean, standardized)
- **Feature Engineering Data**: 2,235 rows × 258 columns (comprehensive features)
- **Final ML Dataset**: 2,235 rows × 140 columns (all numerical, optimized)

### Analysis Outputs:
- **100+ EDA files**: Statistical analyses, visualizations, reports
- **Feature Importance Rankings**: Multiple algorithms and approaches
- **Data Quality Reports**: Comprehensive quality assessment
- **Transformation Recommendations**: Evidence-based improvement suggestions

### Quality Metrics:
- **Zero missing values** in final dataset
- **100% numerical features** for ML compatibility
- **Comprehensive encoding** of all categorical variables
- **Optimized feature set** with redundancy removal
