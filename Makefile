# Pusula Data Science Case - Makefile
# ====================================
# 
# This Makefile provides commands to run the complete data science pipeline
# for medical treatment duration prediction.
#
# Requirements: Python 3.11+, Virtual Environment
# Target Variable: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)

# Variables
PYTHON = python3
VENV_PATH = venv
VENV_ACTIVATE = source $(VENV_PATH)/bin/activate
DATA_DIR = data
SOURCE_DATA = $(DATA_DIR)/Talent_Academy_Case_DT_2025.xlsx
PREPROCESSED_DATA = $(DATA_DIR)/preprocessing/preprocessed_data.csv
FEATURE_ENG_DATA = $(DATA_DIR)/feature_engineering/feature_engineering_data.csv
FINAL_DATA = $(DATA_DIR)/data_final_version/final_cleaned_data.csv
EDA_RESULTS = $(DATA_DIR)/EDA_results

# Default target
.DEFAULT_GOAL := help

# Help command
.PHONY: help
help:
	@echo "Pusula Data Science Case - Pipeline Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup              - Complete environment setup (venv + dependencies)"
	@echo "  install            - Install Python dependencies"
	@echo "  clean              - Clean all generated data and outputs"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  run-all            - Run complete pipeline (preprocessing → feature engineering → EDA)"
	@echo "  preprocessing      - Run only preprocessing pipeline"
	@echo "  feature-engineering - Run only feature engineering pipeline"
	@echo "  eda                - Run only EDA pipeline"
	@echo ""
	@echo "Data Commands:"
	@echo "  check-data         - Verify data files and structure"
	@echo "  data-info          - Show data statistics and information"
	@echo "  validate           - Validate all pipeline outputs"
	@echo ""
	@echo "Development Commands:"
	@echo "  test               - Run pipeline tests"
	@echo "  lint               - Check code quality"
	@echo "  logs               - Show recent pipeline logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  dirs               - Create all required directories"
	@echo "  backup             - Backup current data outputs"
	@echo "  restore            - Restore from backup"

# Setup Commands
# ==============

.PHONY: setup
setup: clean-venv create-venv install dirs
	@echo "Complete setup finished successfully!"
	@echo "Run 'make run-all' to execute the full pipeline."

.PHONY: create-venv
create-venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_PATH)
	@echo "Virtual environment created at $(VENV_PATH)"

.PHONY: install
install:
	@echo "Installing dependencies..."
	$(VENV_ACTIVATE) && pip install --upgrade pip
	$(VENV_ACTIVATE) && pip install -r requirements.txt
	$(VENV_ACTIVATE) && pip install category_encoders
	@echo "Dependencies installed successfully"

.PHONY: dirs
dirs:
	@echo "Creating required directories..."
	mkdir -p $(DATA_DIR)/preprocessing
	mkdir -p $(DATA_DIR)/feature_engineering
	mkdir -p $(DATA_DIR)/data_final_version
	mkdir -p $(DATA_DIR)/EDA_results
	mkdir -p logs
	@echo "All directories created"

# Pipeline Commands
# =================

.PHONY: run-all
run-all: check-source-data preprocessing feature-engineering eda
	@echo ""
	@echo "COMPLETE PIPELINE EXECUTION FINISHED!"
	@echo "====================================="
	@echo "All stages completed successfully:"
	@echo "  1. Preprocessing: $(PREPROCESSED_DATA)"
	@echo "  2. Feature Engineering: $(FEATURE_ENG_DATA)"
	@echo "  3. Final ML Dataset: $(FINAL_DATA)"
	@echo "  4. EDA Results: $(EDA_RESULTS)/"
	@echo ""
	@echo "Your data is now ready for machine learning!"

.PHONY: preprocessing
preprocessing: check-source-data
	@echo "Running preprocessing pipeline..."
	$(VENV_ACTIVATE) && $(PYTHON) -m case_src.pipeline.preprocessing_pipeline
	@echo "Preprocessing completed: $(PREPROCESSED_DATA)"

.PHONY: feature-engineering
feature-engineering: $(PREPROCESSED_DATA)
	@echo "Running feature engineering pipeline..."
	$(VENV_ACTIVATE) && $(PYTHON) -m case_src.pipeline.feature_engineering_pipeline
	@echo "Feature engineering completed: $(FEATURE_ENG_DATA)"
	@echo "Final ML dataset created: $(FINAL_DATA)"

.PHONY: eda
eda: $(FEATURE_ENG_DATA)
	@echo "Running EDA pipeline..."
	$(VENV_ACTIVATE) && $(PYTHON) -m case_src.pipeline.eda_pipeline
	@echo "EDA analysis completed: $(EDA_RESULTS)/"

.PHONY: main-pipeline
main-pipeline: check-source-data
	@echo "Running complete main pipeline..."
	$(VENV_ACTIVATE) && $(PYTHON) main.py
	@echo "Main pipeline completed successfully"

# Data Commands
# =============

.PHONY: check-data
check-data: check-source-data check-outputs

.PHONY: check-source-data
check-source-data:
	@echo "Checking source data..."
	@if [ ! -f "$(SOURCE_DATA)" ]; then \
		echo "ERROR: Source data file not found: $(SOURCE_DATA)"; \
		echo "Please ensure the Excel file exists in the data directory"; \
		exit 1; \
	fi
	@echo "Source data file found: $(SOURCE_DATA)"

.PHONY: check-outputs
check-outputs:
	@echo "Checking pipeline outputs..."
	@echo "Preprocessed data: $$(if [ -f '$(PREPROCESSED_DATA)' ]; then echo 'EXISTS'; else echo 'MISSING'; fi)"
	@echo "Feature engineering data: $$(if [ -f '$(FEATURE_ENG_DATA)' ]; then echo 'EXISTS'; else echo 'MISSING'; fi)"
	@echo "Final ML dataset: $$(if [ -f '$(FINAL_DATA)' ]; then echo 'EXISTS'; else echo 'MISSING'; fi)"
	@echo "EDA results: $$(if [ -d '$(EDA_RESULTS)' ]; then echo 'EXISTS'; else echo 'MISSING'; fi)"

.PHONY: data-info
data-info:
	@echo "Data Pipeline Information"
	@echo "========================"
	@if [ -f "$(SOURCE_DATA)" ]; then \
		echo "Source Data:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "import pandas as pd; df=pd.read_excel('$(SOURCE_DATA)'); print(f'  Shape: {df.shape}'); print(f'  Columns: {list(df.columns)}')"; \
	fi
	@echo ""
	@if [ -f "$(PREPROCESSED_DATA)" ]; then \
		echo "Preprocessed Data:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "import pandas as pd; df=pd.read_csv('$(PREPROCESSED_DATA)'); print(f'  Shape: {df.shape}'); print(f'  Missing values: {df.isnull().sum().sum()}')"; \
	fi
	@echo ""
	@if [ -f "$(FEATURE_ENG_DATA)" ]; then \
		echo "Feature Engineering Data:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "import pandas as pd; df=pd.read_csv('$(FEATURE_ENG_DATA)'); print(f'  Shape: {df.shape}'); print(f'  Data types: {dict(df.dtypes.value_counts())}')"; \
	fi
	@echo ""
	@if [ -f "$(FINAL_DATA)" ]; then \
		echo "Final ML Dataset:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "import pandas as pd; import numpy as np; df=pd.read_csv('$(FINAL_DATA)'); print(f'  Shape: {df.shape}'); print(f'  All numerical: {len(df.select_dtypes(include=[np.number]).columns) == df.shape[1]}'); print(f'  Missing values: {df.isnull().sum().sum()}')"; \
	fi

.PHONY: validate
validate: check-outputs
	@echo "Validating pipeline outputs..."
	@echo "=========================="
	@make --no-print-directory check-preprocessing
	@make --no-print-directory check-feature-engineering  
	@make --no-print-directory check-final
	@make --no-print-directory check-eda
	@echo "Validation completed"

# Development Commands
# ====================

.PHONY: test
test:
	@echo "Running pipeline tests..."
	$(VENV_ACTIVATE) && $(PYTHON) -c "\
import sys; \
sys.path.append('.'); \
from case_src.pipeline.main_pipeline import MainPipeline; \
from case_src.pipeline.preprocessing_pipeline import PreprocessingPipeline; \
from case_src.pipeline.feature_engineering_pipeline import FeatureEngineeringPipeline; \
from case_src.pipeline.eda_pipeline import EDAPipeline; \
print('Testing pipeline initialization...'); \
try: \
    main = MainPipeline(); \
    prep = PreprocessingPipeline(); \
    fe = FeatureEngineeringPipeline(); \
    eda = EDAPipeline(); \
    print('All pipelines initialized successfully'); \
except Exception as e: \
    print(f'Pipeline initialization failed: {e}'); \
    exit(1);"

.PHONY: lint
lint:
	@echo "Checking code quality..."
	$(VENV_ACTIVATE) && $(PYTHON) -m flake8 case_src/ --max-line-length=120 --ignore=E501,W503 || true
	@echo "Lint check completed"

.PHONY: logs
logs:
	@echo "Recent pipeline logs:"
	@echo "===================="
	@if [ -d "logs" ]; then \
		tail -20 logs/*.log | head -50; \
	else \
		echo "No log files found"; \
	fi

# Utility Commands
# ================

.PHONY: clean
clean: clean-data clean-logs
	@echo "Cleanup completed"

.PHONY: clean-data
clean-data:
	@echo "Cleaning generated data files..."
	rm -rf $(DATA_DIR)/preprocessing
	rm -rf $(DATA_DIR)/feature_engineering
	rm -rf $(DATA_DIR)/data_final_version
	rm -rf $(DATA_DIR)/EDA_results
	@echo "Data files cleaned"

.PHONY: clean-logs
clean-logs:
	@echo "Cleaning log files..."
	rm -rf logs
	@echo "Log files cleaned"

.PHONY: clean-venv
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_PATH)
	@echo "Virtual environment removed"

.PHONY: backup
backup:
	@echo "Creating backup of current outputs..."
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p backups/$$TIMESTAMP; \
	if [ -d "$(DATA_DIR)" ]; then cp -r $(DATA_DIR) backups/$$TIMESTAMP/; fi; \
	if [ -d "logs" ]; then cp -r logs backups/$$TIMESTAMP/; fi; \
	echo "Backup created: backups/$$TIMESTAMP"

.PHONY: restore
restore:
	@echo "Available backups:"
	@ls -la backups/ 2>/dev/null || echo "No backups found"
	@echo "To restore, run: cp -r backups/TIMESTAMP/data . && cp -r backups/TIMESTAMP/logs ."

# Individual Pipeline Stages
# ===========================

.PHONY: stage1
stage1: preprocessing
	@echo "Stage 1 (Preprocessing) completed"

.PHONY: stage2
stage2: feature-engineering
	@echo "Stage 2 (Feature Engineering) completed"

.PHONY: stage3
stage3: eda
	@echo "Stage 3 (EDA) completed"

# Data Quality Checks
# ===================

.PHONY: check-preprocessing
check-preprocessing:
	@if [ -f "$(PREPROCESSED_DATA)" ]; then \
		echo "Preprocessing validation:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('$(PREPROCESSED_DATA)'); \
print(f'  Shape: {df.shape}'); \
print(f'  Missing values: {df.isnull().sum().sum()}'); \
print(f'  Data types: {dict(df.dtypes.value_counts())}'); \
print('  Status: VALID' if df.shape[0] > 0 and df.shape[1] > 0 else '  Status: INVALID')"; \
	else \
		echo "Preprocessed data not found. Run 'make preprocessing' first."; \
	fi

.PHONY: check-feature-engineering
check-feature-engineering:
	@if [ -f "$(FEATURE_ENG_DATA)" ]; then \
		echo "Feature engineering validation:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('$(FEATURE_ENG_DATA)'); \
print(f'  Shape: {df.shape}'); \
print(f'  Features created: {df.shape[1] - 13} new features'); \
print(f'  Encoded features: {len([c for c in df.columns if any(enc in c for enc in [\"_OH_\", \"_LABEL_\", \"_TARGET_\", \"_BINARY_\", \"_HASH_\", \"_TFIDF_\"])])}'); \
print(f'  Scaled features: {len([c for c in df.columns if \"_SCALED\" in c])}'); \
print('  Status: VALID')"; \
	else \
		echo "Feature engineering data not found. Run 'make feature-engineering' first."; \
	fi

.PHONY: check-final
check-final:
	@if [ -f "$(FINAL_DATA)" ]; then \
		echo "Final dataset validation:"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "\
import pandas as pd; \
import numpy as np; \
df = pd.read_csv('$(FINAL_DATA)'); \
numeric_cols = len(df.select_dtypes(include=[np.number]).columns); \
all_numeric = numeric_cols == df.shape[1]; \
print(f'  Shape: {df.shape}'); \
print(f'  All numerical: {all_numeric}'); \
print(f'  Missing values: {df.isnull().sum().sum()}'); \
print(f'  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB'); \
print(f'  Status: {\"VALID\" if all_numeric and df.shape[0] > 0 else \"INVALID\"}')"; \
	else \
		echo "Final dataset not found. Run 'make feature-engineering' first."; \
	fi

.PHONY: check-eda
check-eda:
	@if [ -d "$(EDA_RESULTS)" ]; then \
		echo "EDA results validation:"; \
		echo "  Directories: $$(find $(EDA_RESULTS) -type d | wc -l) created"; \
		echo "  Files: $$(find $(EDA_RESULTS) -type f | wc -l) generated"; \
		echo "  Visualizations: $$(find $(EDA_RESULTS) -name '*.png' | wc -l) plots"; \
		echo "  Data files: $$(find $(EDA_RESULTS) -name '*.csv' | wc -l) CSV files"; \
		echo "  Reports: $$(find $(EDA_RESULTS) -name '*.txt' | wc -l) text reports"; \
		echo "  Status: VALID"; \
	else \
		echo "EDA results not found. Run 'make eda' first."; \
	fi

# Quick Commands
# ==============

.PHONY: quick-run
quick-run:
	@echo "Quick pipeline run (main pipeline only)..."
	$(VENV_ACTIVATE) && $(PYTHON) main.py

.PHONY: show-results
show-results:
	@echo "Pipeline Results Summary:"
	@echo "========================"
	@make --no-print-directory check-preprocessing
	@echo ""
	@make --no-print-directory check-feature-engineering
	@echo ""
	@make --no-print-directory check-final
	@echo ""
	@make --no-print-directory check-eda

# Development Utilities
# =====================

.PHONY: jupyter
jupyter:
	@echo "Starting Jupyter notebook server..."
	$(VENV_ACTIVATE) && jupyter notebook notebooks/

.PHONY: python-shell
python-shell:
	@echo "Starting Python shell with project imports..."
	$(VENV_ACTIVATE) && $(PYTHON) -c "\
import sys; \
sys.path.append('.'); \
import pandas as pd; \
import numpy as np; \
from case_src.pipeline.main_pipeline import MainPipeline; \
from case_src.pipeline.preprocessing_pipeline import PreprocessingPipeline; \
from case_src.pipeline.feature_engineering_pipeline import FeatureEngineeringPipeline; \
from case_src.pipeline.eda_pipeline import EDAPipeline; \
print('Project modules loaded. Available: MainPipeline, PreprocessingPipeline, FeatureEngineeringPipeline, EDAPipeline'); \
import code; \
code.interact(local=locals())"

.PHONY: requirements
requirements:
	@echo "Generating requirements.txt..."
	$(VENV_ACTIVATE) && pip freeze > requirements_generated.txt
	@echo "Generated requirements saved to requirements_generated.txt"

# File Dependencies
# =================

$(PREPROCESSED_DATA): $(SOURCE_DATA)
	@make --no-print-directory preprocessing

$(FEATURE_ENG_DATA): $(PREPROCESSED_DATA)
	@make --no-print-directory feature-engineering

$(FINAL_DATA): $(FEATURE_ENG_DATA)
	@echo "Final dataset is created as part of feature engineering"

$(EDA_RESULTS): $(FEATURE_ENG_DATA)
	@make --no-print-directory eda

# Error Handling
# ==============

.PHONY: debug
debug:
	@echo "Debug information:"
	@echo "=================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Virtual environment: $$(if [ -d '$(VENV_PATH)' ]; then echo 'EXISTS'; else echo 'MISSING'; fi)"
	@echo "Source data: $$(if [ -f '$(SOURCE_DATA)' ]; then echo 'EXISTS'; else echo 'MISSING'; fi)"
	@echo "Working directory: $$(pwd)"
	@echo "Available Python modules:"
	@$(VENV_ACTIVATE) && $(PYTHON) -c "import pkg_resources; [print(f'  {pkg.key}=={pkg.version}') for pkg in sorted(pkg_resources.working_set, key=lambda x: x.key)]" | head -10

.PHONY: fix-permissions
fix-permissions:
	@echo "Fixing file permissions..."
	chmod +x case_src/pipeline/*.py
	chmod -R 755 $(DATA_DIR) 2>/dev/null || true
	chmod -R 755 logs 2>/dev/null || true
	@echo "Permissions fixed"

# Documentation
# =============

.PHONY: docs
docs:
	@echo "Pipeline Documentation"
	@echo "====================="
	@echo ""
	@echo "1. PREPROCESSING PIPELINE:"
	@echo "   - Loads raw Excel data"
	@echo "   - Handles missing values with patient-specific imputation"
	@echo "   - Standardizes column names and data types"
	@echo "   - Outputs: $(PREPROCESSED_DATA)"
	@echo ""
	@echo "2. FEATURE ENGINEERING PIPELINE:"
	@echo "   - Creates advanced medical features"
	@echo "   - Applies comprehensive categorical encoding"
	@echo "   - Scales numerical features"
	@echo "   - Removes useless columns"
	@echo "   - Outputs: $(FEATURE_ENG_DATA) and $(FINAL_DATA)"
	@echo ""
	@echo "3. EDA PIPELINE:"
	@echo "   - Comprehensive exploratory data analysis"
	@echo "   - Statistical analysis and visualizations"
	@echo "   - Feature importance analysis"
	@echo "   - Outputs: $(EDA_RESULTS)/ directory"
	@echo ""
	@echo "TARGET VARIABLE: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)"
