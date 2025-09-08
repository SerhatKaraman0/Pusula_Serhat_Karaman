#!/usr/bin/env python3
"""
Complete Data Science Pipeline Runner

This script orchestrates the complete data science pipeline:
1. Preprocessing Pipeline: Data cleaning and preparation
2. Feature Engineering Pipeline: Advanced feature creation
3. EDA Pipeline: Comprehensive exploratory data analysis

Usage:
    python main.py                    # Run complete pipeline
    python main.py --stage preprocessing    # Run only preprocessing
    python main.py --stage feature-engineering    # Run only feature engineering
    python main.py --stage eda              # Run only EDA
    python main.py --help                   # Show help

Requirements:
- All pipeline components must be properly installed
- Raw data file must exist at specified location
- Proper directory structure for outputs
"""

import os
import sys
import argparse
import logging
from typing import Optional, Tuple
import pandas as pd

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import pipeline components
from case_src.pipeline.main_pipeline import MainPipeline, display_pipeline_summary
from case_src.pipeline.eda_pipeline import EDAPipeline

# Import logging and exception handling
from case_src.logging.logger import LOG_FILE_PATH
from case_src.exception import (
    PipelineException,
    DataLoadingException,
    DataValidationException,
    ConfigurationException
)


class CompletePipelineRunner:
    """
    Complete pipeline runner that orchestrates all data processing stages.
    
    This class manages the execution of:
    1. Data preprocessing (cleaning, standardization)
    2. Feature engineering (advanced feature creation)
    3. Exploratory data analysis (comprehensive analysis and visualization)
    """
    
    def __init__(self):
        """Initialize the complete pipeline runner with proper configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Define standard file paths
        self.base_dir = "/Users/user/Desktop/Projects/ds_case_pusula/data"
        self.raw_data_file = os.path.join(self.base_dir, "Talent_Academy_Case_DT_2025.xlsx")
        self.preprocessed_file = os.path.join(self.base_dir, "preprocessing", "preprocessed_data.csv")
        self.feature_engineered_file = os.path.join(self.base_dir, "feature_engineering", "feature_engineering_data.csv")
        self.eda_output_dir = os.path.join(self.base_dir, "EDA_results")
        
        # Initialize pipeline components
        self.main_pipeline = None
        self.eda_pipeline = None
        
        self.logger.info("CompletePipelineRunner initialized")
    
    def validate_file_paths(self) -> None:
        """
        Validate that required files and directories exist or can be created.
        
        Raises:
        -------
        ConfigurationException
            If required files don't exist or directories can't be created
        """
        try:
            # Check if raw data file exists
            if not os.path.exists(self.raw_data_file):
                raise ConfigurationException(f"Raw data file not found: {self.raw_data_file}")
            
            # Ensure output directories exist
            os.makedirs(os.path.dirname(self.preprocessed_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.feature_engineered_file), exist_ok=True)
            os.makedirs(self.eda_output_dir, exist_ok=True)
            
            self.logger.info("File path validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"File path validation failed: {str(e)}")
            raise ConfigurationException(f"File path validation failed: {str(e)}")
    
    def run_preprocessing_stage(self) -> pd.DataFrame:
        """
        Run the preprocessing pipeline stage.
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe
            
        Raises:
        -------
        PipelineException
            If preprocessing stage fails
        """
        try:
            self.logger.info("="*60)
            self.logger.info("STAGE 1: PREPROCESSING PIPELINE")
            self.logger.info("="*60)
            
            if self.main_pipeline is None:
                self.main_pipeline = MainPipeline()
            
            # Run preprocessing only
            preprocessing_pipeline = self.main_pipeline.preprocessing_pipeline
            
            # Load raw data
            raw_df = preprocessing_pipeline.load_data(self.raw_data_file)
            
            # Run preprocessing steps
            processed_df = preprocessing_pipeline.run_preprocessing_pipeline(raw_df, self.preprocessed_file)
            
            self.logger.info(f"Preprocessing completed. Output saved to: {self.preprocessed_file}")
            self.logger.info(f"Preprocessed data shape: {processed_df.shape}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Preprocessing stage failed: {str(e)}")
            raise PipelineException("Preprocessing stage failed", e, "preprocessing")
    
    def run_feature_engineering_stage(self, preprocessed_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the feature engineering pipeline stage.
        
        Parameters:
        -----------
        preprocessed_df : pd.DataFrame, optional
            Preprocessed dataframe. If None, will load from file.
            
        Returns:
        --------
        pd.DataFrame
            Feature-engineered dataframe
            
        Raises:
        -------
        PipelineException
            If feature engineering stage fails
        """
        try:
            self.logger.info("="*60)
            self.logger.info("STAGE 2: FEATURE ENGINEERING PIPELINE")
            self.logger.info("="*60)
            
            if self.main_pipeline is None:
                self.main_pipeline = MainPipeline()
            
            # Load preprocessed data if not provided
            if preprocessed_df is None:
                if not os.path.exists(self.preprocessed_file):
                    raise DataLoadingException(self.preprocessed_file, "Preprocessed file not found. Run preprocessing first.")
                preprocessed_df = pd.read_csv(self.preprocessed_file)
                self.logger.info(f"Loaded preprocessed data from: {self.preprocessed_file}")
            
            # Run feature engineering
            feature_engineering_pipeline = self.main_pipeline.feature_engineering_pipeline
            feature_engineered_df = feature_engineering_pipeline.run_feature_engineering_pipeline(
                preprocessed_df, self.feature_engineered_file
            )
            
            self.logger.info(f"Feature engineering completed. Output saved to: {self.feature_engineered_file}")
            self.logger.info(f"Feature-engineered data shape: {feature_engineered_df.shape}")
            
            return feature_engineered_df
            
        except Exception as e:
            self.logger.error(f"Feature engineering stage failed: {str(e)}")
            raise PipelineException("Feature engineering stage failed", e, "feature_engineering")
    
    def run_eda_stage(self, feature_engineered_df: Optional[pd.DataFrame] = None) -> None:
        """
        Run the EDA pipeline stage.
        
        Parameters:
        -----------
        feature_engineered_df : pd.DataFrame, optional
            Feature-engineered dataframe. If None, will load from file.
            
        Raises:
        -------
        PipelineException
            If EDA stage fails
        """
        try:
            self.logger.info("="*60)
            self.logger.info("STAGE 3: EXPLORATORY DATA ANALYSIS PIPELINE")
            self.logger.info("="*60)
            
            if self.eda_pipeline is None:
                self.eda_pipeline = EDAPipeline()
            
            # Use provided dataframe or load from file
            if feature_engineered_df is None:
                if not os.path.exists(self.feature_engineered_file):
                    raise DataLoadingException(self.feature_engineered_file, "Feature-engineered file not found. Run feature engineering first.")
                input_file = self.feature_engineered_file
                self.logger.info(f"Will load feature-engineered data from: {input_file}")
            else:
                # Save temporary file for EDA pipeline
                temp_file = os.path.join(self.base_dir, "temp_feature_data.csv")
                feature_engineered_df.to_csv(temp_file, index=False)
                input_file = temp_file
                self.logger.info("Using provided feature-engineered dataframe")
            
            # Run complete EDA
            self.eda_pipeline.run_complete_eda(input_file, self.eda_output_dir)
            
            # Clean up temporary file if created
            temp_file = os.path.join(self.base_dir, "temp_feature_data.csv")
            if os.path.exists(temp_file) and feature_engineered_df is not None:
                os.remove(temp_file)
                self.logger.info("Cleaned up temporary file")
            
            self.logger.info(f"EDA completed. Results saved to: {self.eda_output_dir}")
            
        except Exception as e:
            self.logger.error(f"EDA stage failed: {str(e)}")
            raise PipelineException("EDA stage failed", e, "eda")
    
    def run_complete_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete pipeline: preprocessing -> feature engineering -> EDA.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (preprocessed_df, feature_engineered_df)
            
        Raises:
        -------
        PipelineException
            If any stage of the complete pipeline fails
        """
        try:
            self.logger.info("="*80)
            self.logger.info("STARTING COMPLETE DATA SCIENCE PIPELINE")
            self.logger.info("="*80)
            
            # Validate configuration
            self.validate_file_paths()
            
            # Display pipeline summary
            display_pipeline_summary()
            
            # Stage 1: Preprocessing
            preprocessed_df = self.run_preprocessing_stage()
            
            # Stage 2: Feature Engineering
            feature_engineered_df = self.run_feature_engineering_stage(preprocessed_df)
            
            # Stage 3: EDA
            self.run_eda_stage(feature_engineered_df)
            
            self.logger.info("="*80)
            self.logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            self.logger.info("="*80)
            self.logger.info(f"Raw data processed: {self.raw_data_file}")
            self.logger.info(f"Preprocessed data: {self.preprocessed_file}")
            self.logger.info(f"Feature-engineered data: {self.feature_engineered_file}")
            self.logger.info(f"EDA results: {self.eda_output_dir}")
            self.logger.info(f"Final dataset shape: {feature_engineered_df.shape}")
            self.logger.info("="*80)
            
            return preprocessed_df, feature_engineered_df
            
        except Exception as e:
            self.logger.error(f"Complete pipeline failed: {str(e)}")
            raise PipelineException("Complete pipeline failed", e, "complete_pipeline")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser for pipeline execution.
    
    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Complete Data Science Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run complete pipeline
  python main.py --stage preprocessing     # Run only preprocessing
  python main.py --stage feature-engineering  # Run only feature engineering
  python main.py --stage eda               # Run only EDA
  python main.py --verbose                 # Run with verbose logging
  python main.py --dry-run                 # Show what would be executed
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['preprocessing', 'feature-engineering', 'eda', 'complete'],
        default='complete',
        help='Pipeline stage to run (default: complete)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    return parser


def setup_logging(log_level: str = 'INFO', verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_level : str
        Logging level
    verbose : bool
        Enable verbose logging
    """
    if verbose:
        log_level = 'DEBUG'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE_PATH, mode='a')
        ]
    )


def print_pipeline_status(stage: str, dry_run: bool = False) -> None:
    """
    Print pipeline execution status.
    
    Parameters:
    -----------
    stage : str
        Pipeline stage being executed
    dry_run : bool
        Whether this is a dry run
    """
    action = "Would execute" if dry_run else "Executing"
    
    print(f"\n{action} pipeline stage: {stage}")
    print("-" * 50)
    
    stage_descriptions = {
        'preprocessing': 'Data cleaning, standardization, and basic feature creation',
        'feature-engineering': 'Advanced feature engineering and transformation',
        'eda': 'Comprehensive exploratory data analysis and visualization',
        'complete': 'Full pipeline: preprocessing → feature engineering → EDA'
    }
    
    print(f"Description: {stage_descriptions.get(stage, 'Unknown stage')}")
    
    if not dry_run:
        print(f"Log file: {LOG_FILE_PATH}")
    print()


def main():
    """
    Main function to run the complete data science pipeline.
    
    Handles command-line arguments and orchestrates pipeline execution
    with comprehensive error handling and logging.
    """
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Print execution status
        print_pipeline_status(args.stage, args.dry_run)
        
        if args.dry_run:
            print("Dry run completed. No actual processing performed.")
            return
        
        # Initialize pipeline runner
        pipeline_runner = CompletePipelineRunner()
        
        # Execute requested stage
        if args.stage == 'complete':
            logger.info("Starting complete pipeline execution")
            preprocessed_df, feature_engineered_df = pipeline_runner.run_complete_pipeline()
            
            # Print summary
            print("PIPELINE EXECUTION SUMMARY")
            print("=" * 40)
            print(f"Preprocessing completed: {preprocessed_df.shape[0]:,} records")
            print(f"Feature engineering completed: {feature_engineered_df.shape[1]:,} features")
            print(f"EDA analysis completed: {pipeline_runner.eda_output_dir}")
            print(f"Log file: {LOG_FILE_PATH}")
            
        elif args.stage == 'preprocessing':
            logger.info("Starting preprocessing stage")
            preprocessed_df = pipeline_runner.run_preprocessing_stage()
            print(f"Preprocessing completed: {preprocessed_df.shape}")
            
        elif args.stage == 'feature-engineering':
            logger.info("Starting feature engineering stage")
            feature_engineered_df = pipeline_runner.run_feature_engineering_stage()
            print(f"Feature engineering completed: {feature_engineered_df.shape}")
            
        elif args.stage == 'eda':
            logger.info("Starting EDA stage")
            pipeline_runner.run_eda_stage()
            print(f"EDA completed. Results in: {pipeline_runner.eda_output_dir}")
        
        print("\nPipeline execution completed successfully!")
        logger.info("Main function completed successfully")
        
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user")
        logger.warning("Pipeline execution interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        print(f"Check log file for details: {LOG_FILE_PATH}")
        
        if args.verbose:
            import traceback
            print("\nDetailed error information:")
            traceback.print_exc()
        
        logger.error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
