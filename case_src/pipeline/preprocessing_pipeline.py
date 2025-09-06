"""
Preprocessing Pipeline

This module handles basic data cleaning and preparation tasks including:
- Data loading and column standardization
- Data type conversions
- Text cleaning and typo fixes
- Missing value handling
- String data lowercasing
- Data deduplication and aggregation
- Basic column transformations
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any

# Import project utilities
from case_src.utils.preprocess_utils.utils import PreprocessDataFrame
from case_src.utils.analyze_utils.utils import AnalyzeDataFrame
from case_src.utils.visualize_utils.utils import VisualizeDataFrame

# Import logging and exception handling
from case_src.logging.logger import LOG_FILE_PATH
from case_src.exception import (
    PreprocessingException,
    DataLoadingException,
    DataSavingException,
    DataValidationException,
    pipeline_error_handler,
    validate_file_path,
    validate_dataframe,
    handle_exception
)


class PreprocessingPipeline:
    """
    A pipeline for preprocessing medical patient data.
    Handles basic data cleaning, type conversions, and standardization.
    """
    
    def __init__(self):
        """Initialize the preprocessing pipeline with utilities and logger."""
        self.analyze_df = AnalyzeDataFrame()
        self.visualize_df = VisualizeDataFrame()
        self.preprocess_df = PreprocessDataFrame()
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("PreprocessingPipeline initialized")
        
    @pipeline_error_handler("preprocessing")
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
            
        Raises:
        -------
        DataLoadingException
            If file loading fails
        """
        try:
            self.logger.info(f"Loading data from Excel file: {file_path}")
            
            # Validate file path
            validate_file_path(file_path, must_exist=True)
            
            # Load data
            df = pd.read_excel(file_path)
            
            # Validate loaded data
            validate_dataframe(df, min_rows=1, min_cols=1)
            
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise DataLoadingException(file_path, e)
    
    @pipeline_error_handler("preprocessing")
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all column names to uppercase.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with uppercase column names
            
        Raises:
        -------
        PreprocessingException
            If column standardization fails
        """
        try:
            self.logger.info("Standardizing column names to uppercase...")
            
            validate_dataframe(df)
            df = df.copy()
            
            original_columns = df.columns.tolist()
            df.columns = df.columns.str.upper()
            
            self.logger.info(f"Column names standardized: {len(original_columns)} columns converted to uppercase")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to standardize column names: {str(e)}")
            raise PreprocessingException("column_standardization", e, {"original_columns": len(df.columns)})
    
    @pipeline_error_handler("preprocessing")
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert UYGULAMASURESI and TEDAVISURESI columns to proper numeric types.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with converted data types
            
        Raises:
        -------
        PreprocessingException
            If data type conversion fails
        """
        try:
            self.logger.info("Converting data types...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Check required columns exist
            required_cols = ["UYGULAMASURESI", "TEDAVISURESI"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise PreprocessingException(
                    "data_type_conversion", 
                    ValueError(f"Missing required columns: {missing_cols}"),
                    {"missing_columns": missing_cols}
                )
            
            # Convert UYGULAMASURESI from "X Dakika" to integer minutes
            df["UYGULAMASURESI"] = df["UYGULAMASURESI"].str.replace("Dakika", "", regex=False)
            df["UYGULAMASURESI"] = df["UYGULAMASURESI"].str.strip()
            df["UYGULAMASURESI"] = df["UYGULAMASURESI"].astype(int)
            
            # Convert TEDAVISURESI from "X Seans" to integer sessions
            df["TEDAVISURESI"] = df["TEDAVISURESI"].str.replace("Seans", "", regex=False)
            df["TEDAVISURESI"] = df["TEDAVISURESI"].str.strip()
            df["TEDAVISURESI"] = df["TEDAVISURESI"].astype(int)
            
            self.logger.info("Data types converted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to convert data types: {str(e)}")
            raise PreprocessingException("data_type_conversion", e)
    
    @pipeline_error_handler("preprocessing")
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from duration columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional time features
            
        Raises:
        -------
        PreprocessingException
            If time feature creation fails
        """
        try:
            self.logger.info("Creating time-based features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Create hour version of application duration
            df["UYGULAMASURESI_SAAT"] = df["UYGULAMASURESI"] / 60
            
            # Rename columns for clarity
            df.rename(columns={
                "UYGULAMASURESI": "UYGULAMASURESI_DAKIKA",
                "TEDAVISURESI": "TEDAVISURESI_SEANS_SAYI"
            }, inplace=True)
            
            # Calculate total treatment time
            df["TEDAVISURESI_TOPLAM_DAKIKA"] = df["TEDAVISURESI_SEANS_SAYI"] * df["UYGULAMASURESI_DAKIKA"]
            df["TEDAVISURESI_TOPLAM_SAAT"] = df["TEDAVISURESI_SEANS_SAYI"] * df["UYGULAMASURESI_SAAT"]
            
            self.logger.info("Time-based features created successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create time features: {str(e)}")
            raise PreprocessingException("time_feature_creation", e)
    
    @pipeline_error_handler("preprocessing")
    def fix_typos_and_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix common typos and variants in text data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with fixed typos
            
        Raises:
        -------
        PreprocessingException
            If typo fixing fails
        """
        try:
            self.logger.info("Fixing typos and text variants...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Define mapping for common typos and variants
            fix_map = {
                "Hiportiroidizm": "Hipotiroidizm",
                "Hipotirodizm": "Hipotiroidizm",
                "Volteren": "Voltaren",
                "VOLTAREN": "Voltaren"
            }
            
            # Apply replacements to all object (string) columns
            string_cols = df.select_dtypes(include="object").columns
            for col in string_cols:
                for wrong, correct in fix_map.items():
                    df[col] = df[col].str.replace(wrong, correct, regex=False)
            
            self.logger.info(f"Typos fixed in {len(string_cols)} string columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fix typos: {str(e)}")
            raise PreprocessingException("typo_fixing", e)
    
    @pipeline_error_handler("preprocessing")
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using patient-specific imputation strategy.
        
        For each missing value:
        1. First try to fill using mode/mean from the same patient's other records (same HASTANO)
        2. If no other records exist for that patient, use overall column mode/mean
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with handled missing values using patient-specific imputation
            
        Raises:
        -------
        PreprocessingException
            If missing value handling fails
        """
        try:
            self.logger.info("Handling missing values with patient-specific imputation...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Count missing values before processing
            missing_before = df.isnull().sum().sum()
            
            # Special handling for ALERJI column
            if "ALERJI" in df.columns:
                df["ALERJI"] = df["ALERJI"].fillna("YOK")
            
            # Check if HASTANO column exists for patient-specific imputation
            if "HASTANO" not in df.columns:
                self.logger.warning("HASTANO column not found. Using simple mode/mean imputation.")
                return self._simple_imputation(df, missing_before)
            
            # Patient-specific imputation for each column
            columns_to_process = [col for col in df.columns if col not in ["HASTANO", "ALERJI"]]
            
            for col in columns_to_process:
                if df[col].isnull().any():
                    self.logger.info(f"Processing missing values in column: {col}")
                    df = self._patient_specific_imputation(df, col)
            
            # Count missing values after processing
            missing_after = df.isnull().sum().sum()
            
            self.logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to handle missing values: {str(e)}")
            raise PreprocessingException("missing_value_handling", e)
    
    def _patient_specific_imputation(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Perform patient-specific imputation for a single column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        column : str
            Column name to process
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with imputed values for the specified column
        """
        # Get indices of missing values
        missing_mask = df[column].isnull()
        
        if not missing_mask.any():
            return df  # No missing values
        
        # Determine if column is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        
        # Calculate global fallback value
        if is_numeric:
            global_fill_value = df[column].mean()
            self.logger.info(f"Global mean for {column}: {global_fill_value:.2f}")
        else:
            mode_values = df[column].mode()
            global_fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
            self.logger.info(f"Global mode for {column}: {global_fill_value}")
        
        # Track imputation statistics
        patient_filled = 0
        global_filled = 0
        
        # Process each missing value
        for idx in df[missing_mask].index:
            patient_id = df.loc[idx, "HASTANO"]
            
            # Get all records for this patient (excluding the current missing record)
            patient_records = df[(df["HASTANO"] == patient_id) & (df.index != idx)]
            patient_values = patient_records[column].dropna()
            
            if len(patient_values) > 0:
                # Use patient-specific value
                if is_numeric:
                    fill_value = patient_values.mean()
                else:
                    patient_mode = patient_values.mode()
                    fill_value = patient_mode.iloc[0] if not patient_mode.empty else global_fill_value
                
                df.loc[idx, column] = fill_value
                patient_filled += 1
            else:
                # Use global value
                df.loc[idx, column] = global_fill_value
                global_filled += 1
        
        self.logger.info(f"Column {column}: {patient_filled} filled with patient-specific values, "
                        f"{global_filled} filled with global values")
        
        return df
    
    def _simple_imputation(self, df: pd.DataFrame, missing_before: int) -> pd.DataFrame:
        """
        Fallback to simple mode/mean imputation when HASTANO is not available.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        missing_before : int
            Count of missing values before processing
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with simple imputation applied
        """
        # Fill numeric columns with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                self.logger.info(f"Filled {col} missing values with mean: {mean_val:.2f}")
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include="object").columns
        for col in categorical_cols:
            if col != "ALERJI" and df[col].isnull().any():  # ALERJI already handled
                mode = df[col].mode()
                if not mode.empty:
                    fill_value = mode.iloc[0]
                    df[col] = df[col].fillna(fill_value)
                    self.logger.info(f"Filled {col} missing values with mode: {fill_value}")
        
        # Count missing values after processing
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"Simple imputation completed: {missing_before} -> {missing_after}")
        
        return df
    
    @pipeline_error_handler("preprocessing")
    def lowercase_string_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all string data to lowercase.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with lowercase string data
            
        Raises:
        -------
        PreprocessingException
            If string lowercasing fails
        """
        try:
            self.logger.info("Converting string data to lowercase...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Apply lowercase to all object columns
            string_cols = df.select_dtypes(include="object").columns
            df[string_cols] = df[string_cols].apply(lambda x: x.str.lower())
            
            self.logger.info(f"String data converted to lowercase in {len(string_cols)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to convert string data to lowercase: {str(e)}")
            raise PreprocessingException("string_lowercasing", e)
    
    @pipeline_error_handler("preprocessing")
    def deduplicate_by_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate data by grouping by HASTANO and aggregating.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Deduplicated dataframe
            
        Raises:
        -------
        PreprocessingException
            If deduplication fails
        """
        try:
            self.logger.info("Deduplicating data by patient ID...")
            
            validate_dataframe(df)
            
            if "HASTANO" not in df.columns:
                raise PreprocessingException(
                    "deduplication",
                    ValueError("HASTANO column not found"),
                    {"available_columns": df.columns.tolist()}
                )
            
            df = df.copy()
            original_shape = df.shape
            
            # Separate numeric and categorical columns for aggregation
            numeric_cols = []
            categorical_cols = []
            
            for col in df.columns:
                if col == "HASTANO":  # Skip patient ID column
                    continue
                if np.issubdtype(df[col].dtype, np.number):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Create aggregation dictionary
            agg_dict = {col: "mean" for col in numeric_cols}
            agg_dict.update({col: lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan for col in categorical_cols})
            
            # Group by patient ID and aggregate
            grouped_df = df.groupby("HASTANO").agg(agg_dict).reset_index()
            
            self.logger.info(f"Data deduplicated. Shape changed from {original_shape} to {grouped_df.shape}")
            return grouped_df
            
        except Exception as e:
            self.logger.error(f"Failed to deduplicate data: {str(e)}")
            raise PreprocessingException("deduplication", e)
    
    def save_data(self, df: pd.DataFrame, output_file_path: str) -> None:
        """
        Save preprocessed data to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to save
        output_file_path : str
            Path to save the data
            
        Raises:
        -------
        DataSavingException
            If data saving fails
        """
        try:
            self.logger.info(f"Saving preprocessed data to: {output_file_path}")
            
            validate_dataframe(df)
            validate_file_path(output_file_path, must_exist=False)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Save data
            df.to_csv(output_file_path, index=False)
            
            self.logger.info(f"Data saved successfully. Shape: {df.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {output_file_path}: {str(e)}")
            raise DataSavingException(output_file_path, e)
    
    def run_pipeline(self, input_file_path: str, output_file_path: str) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        input_file_path : str
            Path to the input Excel file
        output_file_path : str
            Path to save the preprocessed data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe
            
        Raises:
        -------
        PreprocessingException
            If any step in the pipeline fails
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STARTING PREPROCESSING PIPELINE")
            self.logger.info("="*50)
            
            # Step 1: Load data
            df = self.load_data(input_file_path)
            
            # Step 2: Standardize column names
            df = self.standardize_column_names(df)
            
            # Step 3: Convert data types
            df = self.convert_data_types(df)
            
            # Step 4: Create time features
            df = self.create_time_features(df)
            
            # Step 5: Fix typos and variants
            df = self.fix_typos_and_variants(df)
            
            # Step 6: Handle missing values
            df = self.handle_missing_values(df)
            
            # Step 7: Convert string data to lowercase
            df = self.lowercase_string_data(df)
            
            # Step 8: Deduplicate by patient
            # df = self.deduplicate_by_patient(df)
            
            # Step 9: Save preprocessed data
            self.save_data(df, output_file_path)
            
            self.logger.info("="*50)
            self.logger.info("PREPROCESSING PIPELINE COMPLETED")
            self.logger.info(f"Final data shape: {df.shape}")
            self.logger.info("="*50)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise PreprocessingException("pipeline_execution", e, {"input_file": input_file_path, "output_file": output_file_path})


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        pipeline = PreprocessingPipeline()
        
        # Define file paths
        BASE_DIR = "/Users/user/Desktop/Projects/ds_case_pusula/data"
        INPUT_FILE = os.path.join(BASE_DIR, "Talent_Academy_Case_DT_2025.xlsx")
        OUTPUT_FILE = os.path.join(BASE_DIR, "preprocessing", "preprocessed_data.csv")
        
        logger.info(f"Starting preprocessing pipeline with input: {INPUT_FILE}")
        
        # Run pipeline
        df = pipeline.run_pipeline(INPUT_FILE, OUTPUT_FILE)
        
        logger.info(f"Preprocessing completed successfully!")
        logger.info(f"Preprocessed data saved to: {OUTPUT_FILE}")
        print(f"Preprocessing completed successfully!")
        print(f"Preprocessed data saved to: {OUTPUT_FILE}")
        print(f"Final data shape: {df.shape}")
        
    except Exception as e:
        error_msg = f"Preprocessing failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        raise


if __name__ == "__main__":
    main()