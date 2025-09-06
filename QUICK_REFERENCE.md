# Quick Reference Guide

## Instant Setup & Run

### macOS/Linux
```bash
make setup && make run-all
```

### Windows (Command Prompt)
```cmd
run_pipeline.bat setup && run_pipeline.bat run-all
```

### Windows (PowerShell)
```powershell
.\run_pipeline.ps1 setup; .\run_pipeline.ps1 run-all
```

## Common Commands

| Task | macOS/Linux | Windows CMD | Windows PS |
|------|-------------|-------------|------------|
| Complete Setup | `make setup` | `run_pipeline.bat setup` | `.\run_pipeline.ps1 setup` |
| Run All Pipelines | `make run-all` | `run_pipeline.bat run-all` | `.\run_pipeline.ps1 run-all` |
| Check Results | `make show-results` | `run_pipeline.bat show-results` | `.\run_pipeline.ps1 show-results` |
| Validate Data | `make validate` | `run_pipeline.bat validate` | `.\run_pipeline.ps1 validate` |
| View Help | `make help` | `run_pipeline.bat help` | `.\run_pipeline.ps1 help` |

## Individual Pipeline Stages

| Stage | Command | Output |
|-------|---------|---------|
| 1. Preprocessing | `make preprocessing` | `data/preprocessing/preprocessed_data.csv` |
| 2. Feature Engineering | `make feature-engineering` | `data/feature_engineering/feature_engineering_data.csv` |
| 3. Final Dataset | (included in stage 2) | `data/data_final_version/final_cleaned_data.csv` |
| 4. EDA Analysis | `make eda` | `data/EDA_results/` (100+ files) |

## Key Outputs

- **Source**: `data/Talent_Academy_Case_DT_2025.xlsx` (2,235 × 13)
- **Preprocessed**: `data/preprocessing/preprocessed_data.csv` (2,235 × 16)
- **Feature Engineered**: `data/feature_engineering/feature_engineering_data.csv` (2,235 × 258)
- **ML-Ready**: `data/data_final_version/final_cleaned_data.csv` (2,235 × 140, all numerical)
- **EDA Results**: `data/EDA_results/` (comprehensive analysis)

## Target Variable

**TEDAVISURESI_SEANS_SAYI** - Treatment Duration in Sessions (1-37 sessions, mean: 14.6)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Python not found | Install Python 3.11+ and add to PATH |
| Permission errors | Run as administrator/sudo |
| Missing dependencies | Run setup command again |
| Source file missing | Place Excel file in `data/` directory |
| Pipeline fails | Check logs in `logs/` directory |

## Quick Data Check

```bash
# Check if everything worked
make check-data
make data-info

# Windows
run_pipeline.bat check-data
run_pipeline.bat data-info
```

Expected final result: **140 numerical columns, 0 missing values, ready for ML**
