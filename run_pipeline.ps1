# Pusula Data Science Case - PowerShell Script
# =============================================
# 
# This PowerShell script provides commands to run the complete data science pipeline
# for medical treatment duration prediction on Windows systems.
#
# Requirements: Python 3.11+, pip
# Target Variable: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)
#
# Usage: .\run_pipeline.ps1 [command]
# Example: .\run_pipeline.ps1 setup
# Example: .\run_pipeline.ps1 run-all

param(
    [Parameter(Mandatory=$false)]
    [string]$Command = "help"
)

# Variables
$PYTHON = "python"
$VENV_PATH = "venv"
$DATA_DIR = "data"
$SOURCE_DATA = "$DATA_DIR\Talent_Academy_Case_DT_2025.xlsx"
$PREPROCESSED_DATA = "$DATA_DIR\preprocessing\preprocessed_data.csv"
$FEATURE_ENG_DATA = "$DATA_DIR\feature_engineering\feature_engineering_data.csv"
$FINAL_DATA = "$DATA_DIR\data_final_version\final_cleaned_data.csv"
$EDA_RESULTS = "$DATA_DIR\EDA_results"

function Show-Help {
    Write-Host ""
    Write-Host "Pusula Data Science Case - Windows PowerShell Commands" -ForegroundColor Cyan
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Setup Commands:" -ForegroundColor Yellow
    Write-Host "  setup              - Complete environment setup (venv + dependencies)"
    Write-Host "  install            - Install Python dependencies"
    Write-Host "  clean              - Clean all generated data and outputs"
    Write-Host ""
    Write-Host "Pipeline Commands:" -ForegroundColor Yellow
    Write-Host "  run-all            - Run complete pipeline (preprocessing → feature engineering → EDA)"
    Write-Host "  preprocessing      - Run only preprocessing pipeline"
    Write-Host "  feature-engineering - Run only feature engineering pipeline"
    Write-Host "  eda                - Run only EDA pipeline"
    Write-Host "  main-pipeline      - Run main pipeline (alternative to run-all)"
    Write-Host ""
    Write-Host "Data Commands:" -ForegroundColor Yellow
    Write-Host "  check-data         - Verify data files and structure"
    Write-Host "  data-info          - Show data statistics and information"
    Write-Host "  validate           - Validate all pipeline outputs"
    Write-Host "  show-results       - Show comprehensive results summary"
    Write-Host ""
    Write-Host "Development Commands:" -ForegroundColor Yellow
    Write-Host "  test               - Run pipeline tests"
    Write-Host "  logs               - Show recent pipeline logs"
    Write-Host "  debug              - Show debug information"
    Write-Host ""
    Write-Host "Utility Commands:" -ForegroundColor Yellow
    Write-Host "  dirs               - Create all required directories"
    Write-Host "  backup             - Backup current data outputs"
    Write-Host "  restore            - Show available backups"
    Write-Host "  docs               - Show pipeline documentation"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\run_pipeline.ps1 setup"
    Write-Host "  .\run_pipeline.ps1 run-all"
    Write-Host "  .\run_pipeline.ps1 validate"
    Write-Host ""
}

function Test-SourceData {
    if (-not (Test-Path $SOURCE_DATA)) {
        Write-Host "ERROR: Source data file not found: $SOURCE_DATA" -ForegroundColor Red
        Write-Host "Please ensure the Excel file exists in the data directory" -ForegroundColor Red
        return $false
    }
    Write-Host "Source data file found: $SOURCE_DATA" -ForegroundColor Green
    return $true
}

function Create-Directories {
    Write-Host "Creating required directories..."
    $dirs = @(
        "$DATA_DIR\preprocessing",
        "$DATA_DIR\feature_engineering", 
        "$DATA_DIR\data_final_version",
        "$DATA_DIR\EDA_results",
        "logs"
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created directory: $dir" -ForegroundColor Green
        }
    }
    Write-Host "All directories created"
}

function Install-Dependencies {
    Write-Host "Installing dependencies..."
    & "$VENV_PATH\Scripts\Activate.ps1"
    & $PYTHON -m pip install --upgrade pip
    & $PYTHON -m pip install -r requirements.txt
    & $PYTHON -m pip install category_encoders
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        return $false
    }
    Write-Host "Dependencies installed successfully" -ForegroundColor Green
    return $true
}

function Setup-Environment {
    Write-Host "Creating complete environment setup..." -ForegroundColor Cyan
    
    # Remove existing venv if it exists
    if (Test-Path $VENV_PATH) {
        Remove-Item -Recurse -Force $VENV_PATH
        Write-Host "Removed existing virtual environment"
    }
    
    # Create virtual environment
    Write-Host "Creating virtual environment..."
    & $PYTHON -m venv $VENV_PATH
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Write-Host "Make sure Python 3.11+ is installed and accessible as 'python'" -ForegroundColor Red
        return
    }
    
    Install-Dependencies
    Create-Directories
    
    Write-Host ""
    Write-Host "Complete setup finished successfully!" -ForegroundColor Green
    Write-Host "Run '.\run_pipeline.ps1 run-all' to execute the full pipeline." -ForegroundColor Yellow
}

function Run-Preprocessing {
    if (-not (Test-SourceData)) { return }
    
    Write-Host "Running preprocessing pipeline..." -ForegroundColor Cyan
    & "$VENV_PATH\Scripts\Activate.ps1"
    & $PYTHON -m case_src.pipeline.preprocessing_pipeline
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Preprocessing pipeline failed" -ForegroundColor Red
        return
    }
    Write-Host "Preprocessing completed: $PREPROCESSED_DATA" -ForegroundColor Green
}

function Run-FeatureEngineering {
    if (-not (Test-Path $PREPROCESSED_DATA)) {
        Write-Host "ERROR: Preprocessed data not found. Run preprocessing first." -ForegroundColor Red
        return
    }
    
    Write-Host "Running feature engineering pipeline..." -ForegroundColor Cyan
    & "$VENV_PATH\Scripts\Activate.ps1"
    & $PYTHON -m case_src.pipeline.feature_engineering_pipeline
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Feature engineering pipeline failed" -ForegroundColor Red
        return
    }
    Write-Host "Feature engineering completed: $FEATURE_ENG_DATA" -ForegroundColor Green
    Write-Host "Final ML dataset created: $FINAL_DATA" -ForegroundColor Green
}

function Run-EDA {
    if (-not (Test-Path $FEATURE_ENG_DATA)) {
        Write-Host "ERROR: Feature engineering data not found. Run feature-engineering first." -ForegroundColor Red
        return
    }
    
    Write-Host "Running EDA pipeline..." -ForegroundColor Cyan
    & "$VENV_PATH\Scripts\Activate.ps1"
    & $PYTHON -m case_src.pipeline.eda_pipeline
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: EDA pipeline failed" -ForegroundColor Red
        return
    }
    Write-Host "EDA analysis completed: $EDA_RESULTS\" -ForegroundColor Green
}

function Run-MainPipeline {
    if (-not (Test-SourceData)) { return }
    
    Write-Host "Running complete main pipeline..." -ForegroundColor Cyan
    & "$VENV_PATH\Scripts\Activate.ps1"
    & $PYTHON main.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Main pipeline failed" -ForegroundColor Red
        return
    }
    Write-Host "Main pipeline completed successfully" -ForegroundColor Green
}

function Run-AllPipelines {
    Write-Host "Running complete pipeline..." -ForegroundColor Cyan
    
    if (-not (Test-SourceData)) { return }
    
    Run-Preprocessing
    if ($LASTEXITCODE -ne 0) { return }
    
    Run-FeatureEngineering
    if ($LASTEXITCODE -ne 0) { return }
    
    Run-EDA
    if ($LASTEXITCODE -ne 0) { return }
    
    Write-Host ""
    Write-Host "COMPLETE PIPELINE EXECUTION FINISHED!" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "All stages completed successfully:" -ForegroundColor Green
    Write-Host "  1. Preprocessing: $PREPROCESSED_DATA"
    Write-Host "  2. Feature Engineering: $FEATURE_ENG_DATA"
    Write-Host "  3. Final ML Dataset: $FINAL_DATA"
    Write-Host "  4. EDA Results: $EDA_RESULTS\"
    Write-Host ""
    Write-Host "Your data is now ready for machine learning!" -ForegroundColor Yellow
}

function Check-Data {
    Write-Host "Checking pipeline outputs..." -ForegroundColor Cyan
    
    $files = @(
        @{Name="Preprocessed data"; Path=$PREPROCESSED_DATA},
        @{Name="Feature engineering data"; Path=$FEATURE_ENG_DATA},
        @{Name="Final ML dataset"; Path=$FINAL_DATA},
        @{Name="EDA results"; Path=$EDA_RESULTS}
    )
    
    foreach ($file in $files) {
        if (Test-Path $file.Path) {
            Write-Host "$($file.Name): EXISTS" -ForegroundColor Green
        } else {
            Write-Host "$($file.Name): MISSING" -ForegroundColor Red
        }
    }
}

function Show-DataInfo {
    Write-Host "Data Pipeline Information" -ForegroundColor Cyan
    Write-Host "========================" -ForegroundColor Cyan
    
    & "$VENV_PATH\Scripts\Activate.ps1"
    
    if (Test-Path $SOURCE_DATA) {
        Write-Host "Source Data:"
        & $PYTHON -c "import pandas as pd; df=pd.read_excel('$SOURCE_DATA'); print(f'  Shape: {df.shape}'); print(f'  Columns: {len(df.columns)} columns')"
        Write-Host ""
    }
    
    if (Test-Path $PREPROCESSED_DATA) {
        Write-Host "Preprocessed Data:"
        & $PYTHON -c "import pandas as pd; df=pd.read_csv('$PREPROCESSED_DATA'); print(f'  Shape: {df.shape}'); print(f'  Missing values: {df.isnull().sum().sum()}')"
        Write-Host ""
    }
    
    if (Test-Path $FEATURE_ENG_DATA) {
        Write-Host "Feature Engineering Data:"
        & $PYTHON -c "import pandas as pd; df=pd.read_csv('$FEATURE_ENG_DATA'); print(f'  Shape: {df.shape}'); print(f'  Data types: {dict(df.dtypes.value_counts())}')"
        Write-Host ""
    }
    
    if (Test-Path $FINAL_DATA) {
        Write-Host "Final ML Dataset:"
        & $PYTHON -c "import pandas as pd; import numpy as np; df=pd.read_csv('$FINAL_DATA'); print(f'  Shape: {df.shape}'); print(f'  All numerical: {len(df.select_dtypes(include=[np.number]).columns) == df.shape[1]}'); print(f'  Missing values: {df.isnull().sum().sum()}')"
        Write-Host ""
    }
}

function Show-Logs {
    Write-Host "Recent pipeline logs:" -ForegroundColor Cyan
    Write-Host "====================" -ForegroundColor Cyan
    
    if (Test-Path "logs") {
        $latestLog = Get-ChildItem "logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestLog) {
            Write-Host "Latest log file: $($latestLog.Name)"
            Write-Host ""
            Get-Content $latestLog.FullName | Select-Object -Last 20
        } else {
            Write-Host "No log files found"
        }
    } else {
        Write-Host "Logs directory not found"
    }
}

function Create-Backup {
    Write-Host "Creating backup of current outputs..." -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "backups\$timestamp"
    
    New-Item -ItemType Directory -Path "backups" -Force | Out-Null
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    if (Test-Path $DATA_DIR) {
        Copy-Item -Recurse $DATA_DIR "$backupDir\data"
    }
    
    if (Test-Path "logs") {
        Copy-Item -Recurse "logs" "$backupDir\logs"
    }
    
    Write-Host "Backup created: $backupDir" -ForegroundColor Green
}

# Main script logic
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "setup" { Setup-Environment }
    "install" { Install-Dependencies }
    "dirs" { Create-Directories }
    "clean" { 
        Write-Host "Cleaning all generated files..." -ForegroundColor Yellow
        if (Test-Path "$DATA_DIR\preprocessing") { Remove-Item -Recurse -Force "$DATA_DIR\preprocessing" }
        if (Test-Path "$DATA_DIR\feature_engineering") { Remove-Item -Recurse -Force "$DATA_DIR\feature_engineering" }
        if (Test-Path "$DATA_DIR\data_final_version") { Remove-Item -Recurse -Force "$DATA_DIR\data_final_version" }
        if (Test-Path "$DATA_DIR\EDA_results") { Remove-Item -Recurse -Force "$DATA_DIR\EDA_results" }
        if (Test-Path "logs") { Remove-Item -Recurse -Force "logs" }
        Write-Host "Cleanup completed" -ForegroundColor Green
    }
    "run-all" { Run-AllPipelines }
    "preprocessing" { Run-Preprocessing }
    "feature-engineering" { Run-FeatureEngineering }
    "eda" { Run-EDA }
    "main-pipeline" { Run-MainPipeline }
    "check-data" { 
        Test-SourceData | Out-Null
        Check-Data 
    }
    "data-info" { Show-DataInfo }
    "validate" { 
        Write-Host "Validating pipeline outputs..." -ForegroundColor Cyan
        Check-Data
    }
    "show-results" { 
        Write-Host "Pipeline Results Summary:" -ForegroundColor Cyan
        Write-Host "========================" -ForegroundColor Cyan
        Check-Data
        Write-Host ""
        Show-DataInfo
    }
    "test" { 
        Write-Host "Running pipeline tests..." -ForegroundColor Cyan
        & "$VENV_PATH\Scripts\Activate.ps1"
        & $PYTHON -c "import sys; sys.path.append('.'); from case_src.pipeline.main_pipeline import MainPipeline; from case_src.pipeline.preprocessing_pipeline import PreprocessingPipeline; from case_src.pipeline.feature_engineering_pipeline import FeatureEngineeringPipeline; from case_src.pipeline.eda_pipeline import EDAPipeline; print('Testing pipeline initialization...'); main = MainPipeline(); prep = PreprocessingPipeline(); fe = FeatureEngineeringPipeline(); eda = EDAPipeline(); print('All pipelines initialized successfully')"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Pipeline tests completed successfully" -ForegroundColor Green
        } else {
            Write-Host "Pipeline initialization failed" -ForegroundColor Red
        }
    }
    "logs" { Show-Logs }
    "backup" { Create-Backup }
    "restore" { 
        Write-Host "Available backups:" -ForegroundColor Cyan
        if (Test-Path "backups") {
            Get-ChildItem "backups" | Format-Table Name, LastWriteTime
            Write-Host "To restore, copy the desired backup:" -ForegroundColor Yellow
            Write-Host "  Copy-Item -Recurse backups\TIMESTAMP\data ."
            Write-Host "  Copy-Item -Recurse backups\TIMESTAMP\logs ."
        } else {
            Write-Host "No backups found"
        }
    }
    "debug" { 
        Write-Host "Debug information:" -ForegroundColor Cyan
        Write-Host "=================" -ForegroundColor Cyan
        Write-Host "Python version:"
        & $PYTHON --version
        Write-Host "Virtual environment: $(if (Test-Path $VENV_PATH) { 'EXISTS' } else { 'MISSING' })"
        Write-Host "Source data: $(if (Test-Path $SOURCE_DATA) { 'EXISTS' } else { 'MISSING' })"
        Write-Host "Working directory: $(Get-Location)"
        Write-Host ""
        Write-Host "Available Python packages:"
        & "$VENV_PATH\Scripts\Activate.ps1"
        & $PYTHON -c "import pkg_resources; [print(f'  {pkg.key}=={pkg.version}') for pkg in sorted(pkg_resources.working_set, key=lambda x: x.key)]" 2>$null | Select-Object -First 10
    }
    "docs" { 
        Write-Host "Pipeline Documentation" -ForegroundColor Cyan
        Write-Host "=====================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "1. PREPROCESSING PIPELINE:" -ForegroundColor Yellow
        Write-Host "   - Loads raw Excel data"
        Write-Host "   - Handles missing values with patient-specific imputation"
        Write-Host "   - Standardizes column names and data types"
        Write-Host "   - Outputs: $PREPROCESSED_DATA"
        Write-Host ""
        Write-Host "2. FEATURE ENGINEERING PIPELINE:" -ForegroundColor Yellow
        Write-Host "   - Creates advanced medical features"
        Write-Host "   - Applies comprehensive categorical encoding"
        Write-Host "   - Scales numerical features"
        Write-Host "   - Removes useless columns"
        Write-Host "   - Outputs: $FEATURE_ENG_DATA and $FINAL_DATA"
        Write-Host ""
        Write-Host "3. EDA PIPELINE:" -ForegroundColor Yellow
        Write-Host "   - Comprehensive exploratory data analysis"
        Write-Host "   - Statistical analysis and visualizations"
        Write-Host "   - Feature importance analysis"
        Write-Host "   - Outputs: $EDA_RESULTS\ directory"
        Write-Host ""
        Write-Host "TARGET VARIABLE: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)" -ForegroundColor Green
        Write-Host ""
        Write-Host "For detailed help: .\run_pipeline.ps1 help" -ForegroundColor Yellow
    }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run '.\run_pipeline.ps1 help' for available commands."
    }
}
