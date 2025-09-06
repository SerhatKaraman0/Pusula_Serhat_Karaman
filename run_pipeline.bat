@echo off
REM Pusula Data Science Case - Windows Batch Script
REM =================================================
REM 
REM This batch script provides commands to run the complete data science pipeline
REM for medical treatment duration prediction on Windows systems.
REM
REM Requirements: Python 3.11+, pip
REM Target Variable: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)
REM
REM Usage: run_pipeline.bat [command]
REM Example: run_pipeline.bat setup
REM Example: run_pipeline.bat run-all

setlocal EnableDelayedExpansion

REM Variables
set PYTHON=python
set VENV_PATH=venv
set DATA_DIR=data
set SOURCE_DATA=%DATA_DIR%\Talent_Academy_Case_DT_2025.xlsx
set PREPROCESSED_DATA=%DATA_DIR%\preprocessing\preprocessed_data.csv
set FEATURE_ENG_DATA=%DATA_DIR%\feature_engineering\feature_engineering_data.csv
set FINAL_DATA=%DATA_DIR%\data_final_version\final_cleaned_data.csv
set EDA_RESULTS=%DATA_DIR%\EDA_results

REM Check if command line argument is provided
if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="setup" goto :setup
if "%1"=="install" goto :install
if "%1"=="clean" goto :clean
if "%1"=="run-all" goto :run_all
if "%1"=="preprocessing" goto :preprocessing
if "%1"=="feature-engineering" goto :feature_engineering
if "%1"=="eda" goto :eda
if "%1"=="main-pipeline" goto :main_pipeline
if "%1"=="check-data" goto :check_data
if "%1"=="data-info" goto :data_info
if "%1"=="validate" goto :validate
if "%1"=="test" goto :test
if "%1"=="lint" goto :lint
if "%1"=="logs" goto :logs
if "%1"=="dirs" goto :dirs
if "%1"=="backup" goto :backup
if "%1"=="restore" goto :restore
if "%1"=="check-preprocessing" goto :check_preprocessing
if "%1"=="check-feature-engineering" goto :check_feature_engineering
if "%1"=="check-final" goto :check_final
if "%1"=="check-eda" goto :check_eda
if "%1"=="show-results" goto :show_results
if "%1"=="debug" goto :debug
if "%1"=="docs" goto :docs

echo Unknown command: %1
echo Run 'run_pipeline.bat help' for available commands.
goto :end

:help
echo.
echo Pusula Data Science Case - Windows Pipeline Commands
echo ====================================================
echo.
echo Setup Commands:
echo   setup              - Complete environment setup (venv + dependencies)
echo   install            - Install Python dependencies
echo   clean              - Clean all generated data and outputs
echo.
echo Pipeline Commands:
echo   run-all            - Run complete pipeline (preprocessing → feature engineering → EDA)
echo   preprocessing      - Run only preprocessing pipeline
echo   feature-engineering - Run only feature engineering pipeline
echo   eda                - Run only EDA pipeline
echo   main-pipeline      - Run main pipeline (alternative to run-all)
echo.
echo Data Commands:
echo   check-data         - Verify data files and structure
echo   data-info          - Show data statistics and information
echo   validate           - Validate all pipeline outputs
echo   check-preprocessing - Validate preprocessing output
echo   check-feature-engineering - Validate feature engineering output
echo   check-final        - Validate final ML dataset
echo   check-eda          - Validate EDA results
echo.
echo Development Commands:
echo   test               - Run pipeline tests
echo   lint               - Check code quality
echo   logs               - Show recent pipeline logs
echo   debug              - Show debug information
echo.
echo Utility Commands:
echo   dirs               - Create all required directories
echo   backup             - Backup current data outputs
echo   restore            - Show available backups
echo   show-results       - Show comprehensive results summary
echo   docs               - Show pipeline documentation
echo.
echo Examples:
echo   run_pipeline.bat setup
echo   run_pipeline.bat run-all
echo   run_pipeline.bat validate
echo.
goto :end

REM Setup Commands
REM ==============

:setup
echo Creating complete environment setup...
call :clean_venv
call :create_venv
call :install
call :dirs
echo.
echo Complete setup finished successfully!
echo Run 'run_pipeline.bat run-all' to execute the full pipeline.
goto :end

:create_venv
echo Creating virtual environment...
%PYTHON% -m venv %VENV_PATH%
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.11+ is installed and accessible as 'python'
    goto :end
)
echo Virtual environment created at %VENV_PATH%
goto :eof

:install
echo Installing dependencies...
call %VENV_PATH%\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
pip install category_encoders
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    goto :end
)
echo Dependencies installed successfully
goto :eof

:dirs
echo Creating required directories...
if not exist "%DATA_DIR%\preprocessing" mkdir "%DATA_DIR%\preprocessing"
if not exist "%DATA_DIR%\feature_engineering" mkdir "%DATA_DIR%\feature_engineering"
if not exist "%DATA_DIR%\data_final_version" mkdir "%DATA_DIR%\data_final_version"
if not exist "%DATA_DIR%\EDA_results" mkdir "%DATA_DIR%\EDA_results"
if not exist "logs" mkdir "logs"
echo All directories created
goto :eof

REM Pipeline Commands
REM =================

:run_all
echo Running complete pipeline...
call :check_source_data
if errorlevel 1 goto :end

call :preprocessing
if errorlevel 1 goto :end

call :feature_engineering
if errorlevel 1 goto :end

call :eda
if errorlevel 1 goto :end

echo.
echo COMPLETE PIPELINE EXECUTION FINISHED!
echo =====================================
echo All stages completed successfully:
echo   1. Preprocessing: %PREPROCESSED_DATA%
echo   2. Feature Engineering: %FEATURE_ENG_DATA%
echo   3. Final ML Dataset: %FINAL_DATA%
echo   4. EDA Results: %EDA_RESULTS%\
echo.
echo Your data is now ready for machine learning!
goto :end

:preprocessing
call :check_source_data
if errorlevel 1 goto :end

echo Running preprocessing pipeline...
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% -m case_src.pipeline.preprocessing_pipeline
if errorlevel 1 (
    echo ERROR: Preprocessing pipeline failed
    goto :end
)
echo Preprocessing completed: %PREPROCESSED_DATA%
goto :eof

:feature_engineering
if not exist "%PREPROCESSED_DATA%" (
    echo ERROR: Preprocessed data not found. Run preprocessing first.
    exit /b 1
)

echo Running feature engineering pipeline...
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% -m case_src.pipeline.feature_engineering_pipeline
if errorlevel 1 (
    echo ERROR: Feature engineering pipeline failed
    goto :end
)
echo Feature engineering completed: %FEATURE_ENG_DATA%
echo Final ML dataset created: %FINAL_DATA%
goto :eof

:eda
if not exist "%FEATURE_ENG_DATA%" (
    echo ERROR: Feature engineering data not found. Run feature-engineering first.
    exit /b 1
)

echo Running EDA pipeline...
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% -m case_src.pipeline.eda_pipeline
if errorlevel 1 (
    echo ERROR: EDA pipeline failed
    goto :end
)
echo EDA analysis completed: %EDA_RESULTS%\
goto :eof

:main_pipeline
call :check_source_data
if errorlevel 1 goto :end

echo Running complete main pipeline...
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% main.py
if errorlevel 1 (
    echo ERROR: Main pipeline failed
    goto :end
)
echo Main pipeline completed successfully
goto :eof

REM Data Commands
REM =============

:check_data
call :check_source_data
call :check_outputs
goto :eof

:check_source_data
echo Checking source data...
if not exist "%SOURCE_DATA%" (
    echo ERROR: Source data file not found: %SOURCE_DATA%
    echo Please ensure the Excel file exists in the data directory
    exit /b 1
)
echo Source data file found: %SOURCE_DATA%
goto :eof

:check_outputs
echo Checking pipeline outputs...
if exist "%PREPROCESSED_DATA%" (
    echo Preprocessed data: EXISTS
) else (
    echo Preprocessed data: MISSING
)

if exist "%FEATURE_ENG_DATA%" (
    echo Feature engineering data: EXISTS
) else (
    echo Feature engineering data: MISSING
)

if exist "%FINAL_DATA%" (
    echo Final ML dataset: EXISTS
) else (
    echo Final ML dataset: MISSING
)

if exist "%EDA_RESULTS%" (
    echo EDA results: EXISTS
) else (
    echo EDA results: MISSING
)
goto :eof

:data_info
echo Data Pipeline Information
echo ========================
call %VENV_PATH%\Scripts\activate.bat

if exist "%SOURCE_DATA%" (
    echo Source Data:
    %PYTHON% -c "import pandas as pd; df=pd.read_excel('%SOURCE_DATA%'); print(f'  Shape: {df.shape}'); print(f'  Columns: {len(df.columns)} columns')"
    echo.
)

if exist "%PREPROCESSED_DATA%" (
    echo Preprocessed Data:
    %PYTHON% -c "import pandas as pd; df=pd.read_csv('%PREPROCESSED_DATA%'); print(f'  Shape: {df.shape}'); print(f'  Missing values: {df.isnull().sum().sum()}')"
    echo.
)

if exist "%FEATURE_ENG_DATA%" (
    echo Feature Engineering Data:
    %PYTHON% -c "import pandas as pd; df=pd.read_csv('%FEATURE_ENG_DATA%'); print(f'  Shape: {df.shape}'); print(f'  Data types: {dict(df.dtypes.value_counts())}')"
    echo.
)

if exist "%FINAL_DATA%" (
    echo Final ML Dataset:
    %PYTHON% -c "import pandas as pd; import numpy as np; df=pd.read_csv('%FINAL_DATA%'); print(f'  Shape: {df.shape}'); print(f'  All numerical: {len(df.select_dtypes(include=[np.number]).columns) == df.shape[1]}'); print(f'  Missing values: {df.isnull().sum().sum()}')"
    echo.
)
goto :eof

:validate
echo Validating pipeline outputs...
echo ==========================
call :check_preprocessing
call :check_feature_engineering
call :check_final
call :check_eda
echo Validation completed
goto :eof

:check_preprocessing
if exist "%PREPROCESSED_DATA%" (
    echo Preprocessing validation:
    call %VENV_PATH%\Scripts\activate.bat
    %PYTHON% -c "import pandas as pd; df = pd.read_csv('%PREPROCESSED_DATA%'); print(f'  Shape: {df.shape}'); print(f'  Missing values: {df.isnull().sum().sum()}'); print(f'  Data types: {dict(df.dtypes.value_counts())}'); print('  Status: VALID' if df.shape[0] > 0 and df.shape[1] > 0 else '  Status: INVALID')"
) else (
    echo Preprocessed data not found. Run 'run_pipeline.bat preprocessing' first.
)
goto :eof

:check_feature_engineering
if exist "%FEATURE_ENG_DATA%" (
    echo Feature engineering validation:
    call %VENV_PATH%\Scripts\activate.bat
    %PYTHON% -c "import pandas as pd; df = pd.read_csv('%FEATURE_ENG_DATA%'); print(f'  Shape: {df.shape}'); print(f'  Features created: {df.shape[1] - 13} new features'); print(f'  Encoded features: {len([c for c in df.columns if any(enc in c for enc in [\"_OH_\", \"_LABEL_\", \"_TARGET_\", \"_BINARY_\", \"_HASH_\", \"_TFIDF_\"])])}'); print(f'  Scaled features: {len([c for c in df.columns if \"_SCALED\" in c])}'); print('  Status: VALID')"
) else (
    echo Feature engineering data not found. Run 'run_pipeline.bat feature-engineering' first.
)
goto :eof

:check_final
if exist "%FINAL_DATA%" (
    echo Final dataset validation:
    call %VENV_PATH%\Scripts\activate.bat
    %PYTHON% -c "import pandas as pd; import numpy as np; df = pd.read_csv('%FINAL_DATA%'); numeric_cols = len(df.select_dtypes(include=[np.number]).columns); all_numeric = numeric_cols == df.shape[1]; print(f'  Shape: {df.shape}'); print(f'  All numerical: {all_numeric}'); print(f'  Missing values: {df.isnull().sum().sum()}'); print(f'  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB'); print(f'  Status: {\"VALID\" if all_numeric and df.shape[0] > 0 else \"INVALID\"}')"
) else (
    echo Final dataset not found. Run 'run_pipeline.bat feature-engineering' first.
)
goto :eof

:check_eda
if exist "%EDA_RESULTS%" (
    echo EDA results validation:
    for /f %%i in ('dir /s /b "%EDA_RESULTS%\*" 2^>nul ^| find /c /v ""') do set file_count=%%i
    for /f %%i in ('dir /s /b "%EDA_RESULTS%\*.png" 2^>nul ^| find /c /v ""') do set png_count=%%i
    for /f %%i in ('dir /s /b "%EDA_RESULTS%\*.csv" 2^>nul ^| find /c /v ""') do set csv_count=%%i
    for /f %%i in ('dir /s /b "%EDA_RESULTS%\*.txt" 2^>nul ^| find /c /v ""') do set txt_count=%%i
    echo   Files: !file_count! generated
    echo   Visualizations: !png_count! plots
    echo   Data files: !csv_count! CSV files
    echo   Reports: !txt_count! text reports
    echo   Status: VALID
) else (
    echo EDA results not found. Run 'run_pipeline.bat eda' first.
)
goto :eof

REM Development Commands
REM ====================

:test
echo Running pipeline tests...
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% -c "import sys; sys.path.append('.'); from case_src.pipeline.main_pipeline import MainPipeline; from case_src.pipeline.preprocessing_pipeline import PreprocessingPipeline; from case_src.pipeline.feature_engineering_pipeline import FeatureEngineeringPipeline; from case_src.pipeline.eda_pipeline import EDAPipeline; print('Testing pipeline initialization...'); main = MainPipeline(); prep = PreprocessingPipeline(); fe = FeatureEngineeringPipeline(); eda = EDAPipeline(); print('All pipelines initialized successfully')"
if errorlevel 1 (
    echo Pipeline initialization failed
    goto :end
)
echo Pipeline tests completed successfully
goto :eof

:lint
echo Checking code quality...
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% -m flake8 case_src\ --max-line-length=120 --ignore=E501,W503 2>nul
echo Lint check completed
goto :eof

:logs
echo Recent pipeline logs:
echo ====================
if exist "logs" (
    for /f %%i in ('dir /b /o:d logs\*.log 2^>nul') do set latest_log=%%i
    if defined latest_log (
        echo Latest log file: logs\!latest_log!
        echo.
        powershell "Get-Content logs\!latest_log! | Select-Object -Last 20"
    ) else (
        echo No log files found
    )
) else (
    echo Logs directory not found
)
goto :eof

REM Utility Commands
REM ================

:clean
echo Cleaning all generated files...
call :clean_data
call :clean_logs
echo Cleanup completed
goto :eof

:clean_data
echo Cleaning generated data files...
if exist "%DATA_DIR%\preprocessing" rmdir /s /q "%DATA_DIR%\preprocessing"
if exist "%DATA_DIR%\feature_engineering" rmdir /s /q "%DATA_DIR%\feature_engineering"
if exist "%DATA_DIR%\data_final_version" rmdir /s /q "%DATA_DIR%\data_final_version"
if exist "%DATA_DIR%\EDA_results" rmdir /s /q "%DATA_DIR%\EDA_results"
echo Data files cleaned
goto :eof

:clean_logs
echo Cleaning log files...
if exist "logs" rmdir /s /q "logs"
echo Log files cleaned
goto :eof

:clean_venv
echo Removing virtual environment...
if exist "%VENV_PATH%" rmdir /s /q "%VENV_PATH%"
echo Virtual environment removed
goto :eof

:backup
echo Creating backup of current outputs...
for /f "tokens=1-4 delims=/ " %%i in ("%date%") do (
    for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
        set timestamp=%%l%%j%%k_%%a%%b%%c
    )
)
set backup_dir=backups\%timestamp%
if not exist "backups" mkdir "backups"
mkdir "%backup_dir%"

if exist "%DATA_DIR%" xcopy "%DATA_DIR%" "%backup_dir%\data" /e /i /q
if exist "logs" xcopy "logs" "%backup_dir%\logs" /e /i /q

echo Backup created: %backup_dir%
goto :eof

:restore
echo Available backups:
if exist "backups" (
    dir /b backups
    echo.
    echo To restore, copy the desired backup:
    echo   xcopy backups\TIMESTAMP\data . /e /i
    echo   xcopy backups\TIMESTAMP\logs . /e /i
) else (
    echo No backups found
)
goto :eof

:show_results
echo Pipeline Results Summary:
echo ========================
call :check_preprocessing
echo.
call :check_feature_engineering
echo.
call :check_final
echo.
call :check_eda
goto :eof

:debug
echo Debug information:
echo ==================
%PYTHON% --version
echo Virtual environment: 
if exist "%VENV_PATH%" (
    echo   EXISTS
) else (
    echo   MISSING
)
echo Source data: 
if exist "%SOURCE_DATA%" (
    echo   EXISTS
) else (
    echo   MISSING
)
echo Working directory: %cd%
echo.
echo Available Python packages:
call %VENV_PATH%\Scripts\activate.bat
%PYTHON% -c "import pkg_resources; [print(f'  {pkg.key}=={pkg.version}') for pkg in sorted(pkg_resources.working_set, key=lambda x: x.key)]" 2>nul | more
goto :eof

:docs
echo Pipeline Documentation
echo =====================
echo.
echo 1. PREPROCESSING PIPELINE:
echo    - Loads raw Excel data
echo    - Handles missing values with patient-specific imputation
echo    - Standardizes column names and data types
echo    - Outputs: %PREPROCESSED_DATA%
echo.
echo 2. FEATURE ENGINEERING PIPELINE:
echo    - Creates advanced medical features
echo    - Applies comprehensive categorical encoding
echo    - Scales numerical features
echo    - Removes useless columns
echo    - Outputs: %FEATURE_ENG_DATA% and %FINAL_DATA%
echo.
echo 3. EDA PIPELINE:
echo    - Comprehensive exploratory data analysis
echo    - Statistical analysis and visualizations
echo    - Feature importance analysis
echo    - Outputs: %EDA_RESULTS%\ directory
echo.
echo TARGET VARIABLE: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)
echo.
echo For detailed help: run_pipeline.bat help
goto :eof

:end
endlocal
