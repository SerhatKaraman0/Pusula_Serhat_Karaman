# Windows Setup Guide

This guide explains how to run the Pusula Data Science Case pipeline on Windows systems.

## Quick Start

### Option 1: Batch Script (Command Prompt)
```cmd
# Complete setup and run
run_pipeline.bat setup
run_pipeline.bat run-all

# Individual stages
run_pipeline.bat preprocessing
run_pipeline.bat feature-engineering
run_pipeline.bat eda

# Validation
run_pipeline.bat validate
run_pipeline.bat show-results
```

### Option 2: PowerShell Script (PowerShell)
```powershell
# Complete setup and run
.\run_pipeline.ps1 setup
.\run_pipeline.ps1 run-all

# Individual stages
.\run_pipeline.ps1 preprocessing
.\run_pipeline.ps1 feature-engineering
.\run_pipeline.ps1 eda

# Validation
.\run_pipeline.ps1 validate
.\run_pipeline.ps1 show-results
```

## Prerequisites

1. **Python 3.11+** installed and accessible via `python` command
2. **pip** package manager
3. **Excel file** placed in `data/Talent_Academy_Case_DT_2025.xlsx`

## Available Commands

### Setup Commands
- `setup` - Complete environment setup (creates venv + installs dependencies)
- `install` - Install Python dependencies only
- `clean` - Clean all generated data and outputs
- `dirs` - Create all required directories

### Pipeline Commands
- `run-all` - Run complete pipeline (preprocessing → feature engineering → EDA)
- `preprocessing` - Run only preprocessing pipeline
- `feature-engineering` - Run only feature engineering pipeline
- `eda` - Run only EDA pipeline
- `main-pipeline` - Run main pipeline (alternative approach)

### Data Commands
- `check-data` - Verify data files and structure
- `data-info` - Show detailed data statistics
- `validate` - Validate all pipeline outputs
- `show-results` - Show comprehensive results summary

### Development Commands
- `test` - Run pipeline tests and initialization checks
- `logs` - Show recent pipeline logs
- `debug` - Show debug information and environment status
- `docs` - Show pipeline documentation

### Utility Commands
- `backup` - Backup current data outputs with timestamp
- `restore` - Show available backups for restoration

## Pipeline Outputs

After running the complete pipeline, you'll have:

1. **Preprocessed Data**: `data/preprocessing/preprocessed_data.csv`
   - Clean, standardized data with missing values handled

2. **Feature Engineering Data**: `data/feature_engineering/feature_engineering_data.csv`
   - 258 columns with all engineered features, encoding, and scaling

3. **Final ML Dataset**: `data/data_final_version/final_cleaned_data.csv`
   - 140 columns, all numerical, ready for machine learning

4. **EDA Results**: `data/EDA_results/`
   - Comprehensive exploratory data analysis with visualizations

## Troubleshooting

### Common Issues

1. **Python not found**
   - Ensure Python 3.11+ is installed
   - Add Python to system PATH
   - Try using `python3` instead of `python`

2. **Virtual environment issues**
   - Run `run_pipeline.bat clean` to remove old venv
   - Run `run_pipeline.bat setup` to recreate environment

3. **Permission errors**
   - Run Command Prompt or PowerShell as Administrator
   - Check file/folder permissions

4. **Missing source data**
   - Ensure `data/Talent_Academy_Case_DT_2025.xlsx` exists
   - Check file name spelling and location

### Getting Help

- Run `run_pipeline.bat help` or `.\run_pipeline.ps1 help` for command list
- Check `logs/` directory for detailed pipeline logs
- Run `run_pipeline.bat debug` for environment information

## File Structure

```
ds_case_pusula/
├── run_pipeline.bat          # Windows batch script
├── run_pipeline.ps1          # PowerShell script
├── Makefile                  # Unix/Linux makefile
├── data/
│   ├── Talent_Academy_Case_DT_2025.xlsx  # Source data
│   ├── preprocessing/        # Preprocessed outputs
│   ├── feature_engineering/  # Feature engineering outputs
│   ├── data_final_version/   # Final ML-ready dataset
│   └── EDA_results/         # EDA analysis results
├── logs/                    # Pipeline execution logs
└── case_src/               # Source code
    └── pipeline/           # Pipeline modules
```

## Target Variable

The pipeline predicts **TEDAVISURESI_SEANS_SAYI** (Treatment Duration in Sessions) based on patient medical data and characteristics.
