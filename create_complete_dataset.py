#!/usr/bin/env python3
"""
Create Complete Dataset (No Columns Dropped)

This script creates a version of the dataset where no columns are dropped,
preserving all original features and engineered features for comprehensive analysis.
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
logger = logging.getLogger(__name__)

def ensure_all_numeric_preserve_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric format without dropping any columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with mixed types
        
    Returns:
    --------
    pd.DataFrame
        Fully numeric dataframe with all columns preserved
    """
    df_numeric = df.copy()
    
    # Convert object columns to numeric using label encoding
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            logger.info(f"Converting object column to numeric: {col}")
            le = LabelEncoder()
            # Handle missing values by converting to string first
            df_numeric[col] = df_numeric[col].astype(str)
            df_numeric[col] = le.fit_transform(df_numeric[col])
        
        elif df_numeric[col].dtype == 'bool':
            logger.info(f"Converting boolean column to numeric: {col}")
            df_numeric[col] = df_numeric[col].astype(int)
    
    # Ensure all columns are numeric
    for col in df_numeric.columns:
        if not pd.api.types.is_numeric_dtype(df_numeric[col]):
            logger.warning(f"Column {col} is still not numeric, forcing conversion")
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    return df_numeric

def create_complete_dataset():
    """
    Create a complete dataset with all columns preserved and converted to numeric format.
    """
    try:
        logger.info("="*60)
        logger.info("CREATING COMPLETE DATASET (NO COLUMNS DROPPED)")
        logger.info("="*60)
        
        # Input and output paths
        input_file = "/Users/user/Desktop/Projects/ds_case_pusula/data/feature_engineering/feature_engineering_data.csv"
        output_file = "/Users/user/Desktop/Projects/ds_case_pusula/data/data_final_version/complete_dataset_no_drops.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Load the feature engineered data
        logger.info(f"Loading feature engineered data from: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Display column information
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"Data types distribution:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            logger.info(f"  {dtype}: {count} columns")
        
        # Convert all columns to numeric while preserving everything
        logger.info("Converting all columns to numeric format...")
        df_complete_numeric = ensure_all_numeric_preserve_all(df)
        
        # Verify all columns are numeric
        non_numeric_cols = []
        for col in df_complete_numeric.columns:
            if not pd.api.types.is_numeric_dtype(df_complete_numeric[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            logger.warning(f"Warning: {len(non_numeric_cols)} columns are still not numeric: {non_numeric_cols[:5]}...")
        else:
            logger.info("✓ All columns successfully converted to numeric format")
        
        # Check for missing values
        missing_values = df_complete_numeric.isnull().sum().sum()
        logger.info(f"Total missing values: {missing_values}")
        
        # Fill any remaining missing values with 0 (or median for better approach)
        if missing_values > 0:
            logger.info("Filling missing values with 0...")
            df_complete_numeric = df_complete_numeric.fillna(0)
        
        # Save the complete dataset
        logger.info(f"Saving complete dataset to: {output_file}")
        df_complete_numeric.to_csv(output_file, index=False)
        
        # Final statistics
        logger.info("="*60)
        logger.info("COMPLETE DATASET CREATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Original shape: {df.shape}")
        logger.info(f"Final shape: {df_complete_numeric.shape}")
        logger.info(f"Columns preserved: {len(df_complete_numeric.columns)} (NO DROPS)")
        logger.info(f"All columns numeric: {'Yes' if len(non_numeric_cols) == 0 else 'No'}")
        logger.info(f"Missing values: {df_complete_numeric.isnull().sum().sum()}")
        logger.info(f"Memory usage: {df_complete_numeric.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display first few column names for verification
        logger.info(f"First 10 columns: {list(df_complete_numeric.columns[:10])}")
        logger.info(f"Last 10 columns: {list(df_complete_numeric.columns[-10:])}")
        
        logger.info("✓ Complete dataset created successfully!")
        
        return df_complete_numeric
        
    except Exception as e:
        logger.error(f"Failed to create complete dataset: {str(e)}")
        raise

def compare_datasets():
    """
    Compare the original cleaned dataset with the new complete dataset.
    """
    try:
        logger.info("="*60)
        logger.info("DATASET COMPARISON")
        logger.info("="*60)
        
        # Load both datasets
        cleaned_file = "/Users/user/Desktop/Projects/ds_case_pusula/data/data_final_version/final_cleaned_data.csv"
        complete_file = "/Users/user/Desktop/Projects/ds_case_pusula/data/data_final_version/complete_dataset_no_drops.csv"
        
        df_cleaned = pd.read_csv(cleaned_file)
        df_complete = pd.read_csv(complete_file)
        
        logger.info(f"Cleaned dataset shape: {df_cleaned.shape}")
        logger.info(f"Complete dataset shape: {df_complete.shape}")
        logger.info(f"Additional columns in complete dataset: {df_complete.shape[1] - df_cleaned.shape[1]}")
        
        # Find columns that were dropped
        cleaned_cols = set(df_cleaned.columns)
        complete_cols = set(df_complete.columns)
        
        dropped_cols = complete_cols - cleaned_cols
        logger.info(f"Columns that were dropped in cleaned version: {len(dropped_cols)}")
        if dropped_cols:
            logger.info(f"Sample dropped columns: {list(dropped_cols)[:10]}")
        
        logger.info("Dataset comparison completed!")
        
    except Exception as e:
        logger.error(f"Failed to compare datasets: {str(e)}")

if __name__ == "__main__":
    # Create the complete dataset
    df_complete = create_complete_dataset()
    
    # Compare with the cleaned version
    compare_datasets()
    
    print("\n" + "="*60)
    print("COMPLETE DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"Location: /Users/user/Desktop/Projects/ds_case_pusula/data/data_final_version/complete_dataset_no_drops.csv")
    print(f"Shape: {df_complete.shape}")
    print("All original and engineered features preserved!")
    print("Ready for comprehensive analysis and Tableau visualization!")


