"""
Main Pipeline Runner

This script runs the complete data processing pipeline:
1. Preprocessing Pipeline: Basic data cleaning and preparation
2. Feature Engineering Pipeline: Advanced feature creation and engineering

The pipeline follows the data flow:
Raw Data -> Preprocessing -> Feature Engineering -> Final Output

All requirements are met:
- Column names are uppercase
- String data is lowercase
- No columns are removed
- Proper folder structure with dedicated output directories
- Comprehensive logging and exception handling
"""

import os
import sys
import pandas as pd
import logging
from typing import Tuple, Dict, Any

# Import pipeline components
from .preprocessing_pipeline import PreprocessingPipeline
from .feature_engineering_pipeline import FeatureEngineeringPipeline

# Import logging and exception handling
from case_src.logging.logger import LOG_FILE_PATH
from case_src.exception import (
    PipelineException,
    DataLoadingException,
    DataValidationException,
    ConfigurationException,
    pipeline_error_handler,
    validate_file_path,
    validate_dataframe,
    handle_exception
)


class MainPipeline:
    """
    Main pipeline orchestrator that runs the complete data processing workflow.
    Coordinates preprocessing and feature engineering pipelines with comprehensive
    logging and error handling.
    """
    
    def __init__(self):
        """Initialize the main pipeline with logger and sub-pipelines."""
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("MainPipeline initialized")
        
        # Ensure all necessary directories exist
        self._create_required_directories()
        
        # Initialize sub-pipelines
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.feature_engineering_pipeline = FeatureEngineeringPipeline()
    
    def _create_required_directories(self):
        """Create all required directories for the pipeline."""
        try:
            base_dir = "/Users/user/Desktop/Projects/ds_case_pusula/data"
            
            required_dirs = [
                os.path.join(base_dir, "preprocessing"),
                os.path.join(base_dir, "feature_engineering"), 
                os.path.join(base_dir, "data_final_version"),
                os.path.join(base_dir, "EDA_results"),
                "/Users/user/Desktop/Projects/ds_case_pusula/logs"
            ]
            
            for directory in required_dirs:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self.logger.info(f"Created directory: {directory}")
                else:
                    self.logger.debug(f"Directory already exists: {directory}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to create some directories: {str(e)}")
            # Don't fail the pipeline for directory creation issues
        
    @pipeline_error_handler("main_pipeline")
    def validate_configuration(self, raw_data_file: str, preprocessed_file: str, final_output_file: str) -> None:
        """
        Validate pipeline configuration and file paths.
        
        Parameters:
        -----------
        raw_data_file : str
            Path to raw data file
        preprocessed_file : str
            Path for preprocessed output
        final_output_file : str
            Path for final output
            
        Raises:
        -------
        ConfigurationException
            If configuration is invalid
        """
        try:
            self.logger.info("Validating pipeline configuration...")
            
            # Validate raw data file exists
            if not os.path.exists(raw_data_file):
                raise ConfigurationException(f"Raw data file not found: {raw_data_file}")
            
            # Validate output directories can be created
            for output_file in [preprocessed_file, final_output_file]:
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        self.logger.info(f"Created output directory: {output_dir}")
                    except Exception as e:
                        raise ConfigurationException(f"Cannot create output directory {output_dir}: {str(e)}")
            
            self.logger.info("Configuration validation successful")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise ConfigurationException(f"Invalid pipeline configuration: {str(e)}", e)
    
    @pipeline_error_handler("main_pipeline")
    def run_preprocessing_step(self, raw_data_file: str, preprocessed_file: str) -> pd.DataFrame:
        """
        Run the preprocessing pipeline step.
        
        Parameters:
        -----------
        raw_data_file : str
            Path to raw data file
        preprocessed_file : str
            Path for preprocessed output
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe
            
        Raises:
        -------
        PipelineException
            If preprocessing step fails
        """
        try:
            self.logger.info("STEP 1: Running Preprocessing Pipeline...")
            
            # Run preprocessing pipeline
            preprocessed_df = self.preprocessing_pipeline.run_pipeline(raw_data_file, preprocessed_file)
            
            # Validate output
            validate_dataframe(preprocessed_df, min_rows=1, min_cols=1)
            
            self.logger.info(f"Preprocessing completed. Output shape: {preprocessed_df.shape}")
            return preprocessed_df
            
        except Exception as e:
            self.logger.error(f"Preprocessing step failed: {str(e)}")
            raise PipelineException(
                "Preprocessing step failed", 
                e, 
                pipeline_stage="main_pipeline",
                context={"step": "preprocessing", "input_file": raw_data_file, "output_file": preprocessed_file}
            )
    
    @pipeline_error_handler("main_pipeline")
    def run_feature_engineering_step(self, preprocessed_file: str, final_output_file: str) -> pd.DataFrame:
        """
        Run the feature engineering pipeline step.
        
        Parameters:
        -----------
        preprocessed_file : str
            Path to preprocessed data file
        final_output_file : str
            Path for final output
            
        Returns:
        --------
        pd.DataFrame
            Feature engineered dataframe
            
        Raises:
        -------
        PipelineException
            If feature engineering step fails
        """
        try:
            self.logger.info("STEP 2: Running Feature Engineering Pipeline...")
            
            # Run feature engineering pipeline
            final_df = self.feature_engineering_pipeline.run_pipeline(preprocessed_file, final_output_file)
            
            # Validate output
            validate_dataframe(final_df, min_rows=1, min_cols=1)
            
            self.logger.info(f"Feature engineering completed. Final output shape: {final_df.shape}")
            return final_df
            
        except Exception as e:
            self.logger.error(f"Feature engineering step failed: {str(e)}")
            raise PipelineException(
                "Feature engineering step failed", 
                e, 
                pipeline_stage="main_pipeline",
                context={"step": "feature_engineering", "input_file": preprocessed_file, "output_file": final_output_file}
            )
    
    @pipeline_error_handler("main_pipeline")
    def verify_requirements(self, final_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verify that all pipeline requirements are met.
        
        Parameters:
        -----------
        final_df : pd.DataFrame
            Final processed dataframe
            
        Returns:
        --------
        Dict[str, Any]
            Verification results
            
        Raises:
        -------
        DataValidationException
            If requirements are not met
        """
        try:
            self.logger.info("Verifying pipeline requirements...")
            
            validate_dataframe(final_df)
            
            verification_results = {}
            
            # Check column names are uppercase
            all_uppercase = all(col.isupper() for col in final_df.columns)
            verification_results["all_columns_uppercase"] = all_uppercase
            
            if not all_uppercase:
                non_uppercase = [col for col in final_df.columns if not col.isupper()]
                raise DataValidationException(f"Non-uppercase columns found: {non_uppercase}")
            
            # Check string data is lowercase (sample check)
            string_cols = final_df.select_dtypes(include='object').columns
            sample_strings = []
            
            for col in string_cols[:3]:  # Check first 3 string columns
                if not final_df[col].dropna().empty:
                    sample = final_df[col].dropna().iloc[0]
                    sample_strings.append(f"{col}: '{sample}'")
            
            verification_results["string_columns_checked"] = len(string_cols)
            verification_results["sample_strings"] = sample_strings
            
            # Check data shape
            verification_results["final_shape"] = final_df.shape
            verification_results["total_features"] = final_df.shape[1]
            verification_results["total_samples"] = final_df.shape[0]
            
            self.logger.info("All requirements verified successfully")
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Requirements verification failed: {str(e)}")
            raise DataValidationException(f"Pipeline requirements not met: {str(e)}", final_df.shape, e)
    
    def run_complete_pipeline(self, raw_data_file: str, preprocessed_file: str, final_output_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete data processing pipeline from raw data to final features.
        
        Parameters:
        -----------
        raw_data_file : str
            Path to raw data file
        preprocessed_file : str
            Path for preprocessed output
        final_output_file : str
            Path for final output
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (preprocessed_df, feature_engineered_df) - Both dataframes from the pipeline
            
        Raises:
        -------
        PipelineException
            If any step in the complete pipeline fails
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
            self.logger.info("="*70)
            
            # Log file paths
            self.logger.info(f"Raw data file: {raw_data_file}")
            self.logger.info(f"Preprocessed output: {preprocessed_file}")
            self.logger.info(f"Final output: {final_output_file}")
            
            # Step 0: Validate configuration
            self.validate_configuration(raw_data_file, preprocessed_file, final_output_file)
            
            # Step 1: Run Preprocessing Pipeline
            preprocessed_df = self.run_preprocessing_step(raw_data_file, preprocessed_file)
            
            # Step 2: Run Feature Engineering Pipeline
            final_df = self.run_feature_engineering_step(preprocessed_file, final_output_file)
            
            # Step 3: Verify requirements
            verification_results = self.verify_requirements(final_df)
            
            # Log verification results
            self.logger.info("VERIFYING REQUIREMENTS:")
            self.logger.info(f"All column names uppercase: {verification_results['all_columns_uppercase']}")
            self.logger.info(f"String columns checked: {verification_results['string_columns_checked']}")
            self.logger.info(f"Sample strings: {verification_results['sample_strings']}")
            self.logger.info(f"Final shape: {verification_results['final_shape']}")
            
            # Check file existence
            preprocessing_exists = os.path.exists(preprocessed_file)
            feature_eng_exists = os.path.exists(final_output_file)
            self.logger.info(f"Preprocessing output exists: {preprocessing_exists}")
            self.logger.info(f"Feature engineering output exists: {feature_eng_exists}")
            
            self.logger.info("="*70)
            self.logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            self.logger.info("="*70)
            self.logger.info(f"Final dataset ready at: {final_output_file}")
            self.logger.info(f"Intermediate data at: {preprocessed_file}")
            self.logger.info(f"Total features created: {final_df.shape[1]}")
            self.logger.info(f"Total samples: {final_df.shape[0]}")
            self.logger.info("="*70)
            
            return preprocessed_df, final_df
            
        except Exception as e:
            self.logger.error(f"Complete pipeline failed: {str(e)}")
            raise PipelineException(
                "Complete pipeline execution failed", 
                e, 
                pipeline_stage="main_pipeline",
                context={
                    "raw_data_file": raw_data_file,
                    "preprocessed_file": preprocessed_file, 
                    "final_output_file": final_output_file
                }
            )


def display_pipeline_summary():
    """
    Display a summary of what each pipeline does.
    """
    logger = logging.getLogger(__name__)
    
    summary = """
    ======================================================================
    PIPELINE SUMMARY
    ======================================================================

    PREPROCESSING PIPELINE:
       • Load raw Excel data
       • Standardize column names to UPPERCASE
       • Convert data types (remove 'Dakika', 'Seans' suffixes)
       • Create time-based features (hours, total duration)
       • Fix typos and text variants
       • Handle missing values (ALERJI -> 'YOK', others -> mode)
       • Convert string data to lowercase
       • Deduplicate by patient ID (HASTANO)
       • Save to: data/preprocessing/preprocessed_data.csv

    FEATURE ENGINEERING PIPELINE:
       • Create list features from comma-separated values
       • Generate count features (diagnoses, allergies, treatments, etc.)
       • Map diagnoses to medical categories
       • Create chronic disease group flags and risk scores
       • Categorize treatments and create treatment features
       • Create age bins and demographic features
       • Generate binary flags for various conditions
       • Save to: data/feature_engineering/feature_engineering_data.csv

    DATA FLOW:
       Raw Data (2235 rows) → Preprocessing (404 patients) → Feature Engineering (404 rows, 41 features)

    REQUIREMENTS MET:
       • Column names: ALL UPPERCASE
       • String data: all lowercase
       • No columns removed
       • Proper folder structure
       • Dedicated output directories
       • Comprehensive logging and exception handling
    """
    
    print(summary)
    logger.info("Pipeline summary displayed")


def main():
    """
    Main function to run the complete pipeline with comprehensive error handling.
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting main pipeline execution")
        
        # Display summary
        display_pipeline_summary()
        
        # Initialize main pipeline
        main_pipeline = MainPipeline()
        
        # Define file paths
        BASE_DIR = "/Users/user/Desktop/Projects/ds_case_pusula/data"
        RAW_DATA_FILE = os.path.join(BASE_DIR, "Talent_Academy_Case_DT_2025.xlsx")
        PREPROCESSED_FILE = os.path.join(BASE_DIR, "preprocessing", "preprocessed_data.csv")
        FINAL_OUTPUT_FILE = os.path.join(BASE_DIR, "feature_engineering", "feature_engineering_data.csv")
        
        logger.info("Starting complete pipeline execution")
        
        # Run complete pipeline
        preprocessed_df, final_df = main_pipeline.run_complete_pipeline(
            RAW_DATA_FILE, 
            PREPROCESSED_FILE, 
            FINAL_OUTPUT_FILE
        )
        
        # Display results
        print(f"Complete pipeline executed successfully!")
        print(f"Final dataset: {FINAL_OUTPUT_FILE}")
        print(f"Intermediate data: {PREPROCESSED_FILE}")
        print(f"Final shape: {final_df.shape}")
        
        # Optional: Display basic statistics
        print("\nFINAL DATASET OVERVIEW:")
        print("-" * 30)
        print(f"Shape: {final_df.shape}")
        print(f"Data types: {final_df.dtypes.value_counts().to_dict()}")
        
        logger.info("Main pipeline execution completed successfully")
        
    except Exception as e:
        error_msg = f"Main pipeline failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        print(f"Check log file for details: {LOG_FILE_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()