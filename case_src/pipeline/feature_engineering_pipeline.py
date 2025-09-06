"""
Feature Engineering Pipeline

This module handles advanced feature creation and engineering tasks including:
- List-based feature creation from comma-separated values
- Count features for various medical conditions
- Category mapping for diagnoses
- Medical condition grouping and risk scoring
- Treatment categorization
- Age binning and demographic features
- Complex aggregations and binary encodings
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder, CountEncoder

# Import project utilities
from case_src.utils.preprocess_utils.utils import PreprocessDataFrame
from case_src.utils.analyze_utils.utils import AnalyzeDataFrame
from case_src.utils.visualize_utils.utils import VisualizeDataFrame

# Import logging and exception handling
from case_src.logging.logger import LOG_FILE_PATH
from case_src.exception import (
    FeatureEngineeringException,
    DataLoadingException,
    DataSavingException,
    DataValidationException,
    pipeline_error_handler,
    validate_file_path,
    validate_dataframe,
    handle_exception
)


class FeatureEngineeringPipeline:
    """
    A pipeline for creating advanced features from preprocessed medical patient data.
    Handles feature creation, medical condition analysis, and risk scoring.
    """
    
    def __init__(self):
        """Initialize the feature engineering pipeline with utilities and logger."""
        self.analyze_df = AnalyzeDataFrame()
        self.visualize_df = VisualizeDataFrame()
        self.preprocess_df = PreprocessDataFrame()
        
        # Set target column for encoding
        self.target_column = "TEDAVISURESI_SEANS_SAYI"
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("FeatureEngineeringPipeline initialized")
        
    @pipeline_error_handler("feature_engineering")
    def load_preprocessed_data(self, file_path: str) -> pd.DataFrame:
        """
        Load preprocessed data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the preprocessed CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded preprocessed dataframe
            
        Raises:
        -------
        DataLoadingException
            If file loading fails
        """
        try:
            self.logger.info(f"Loading preprocessed data from: {file_path}")
            
            # Validate file path
            validate_file_path(file_path, must_exist=True)
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Validate loaded data
            validate_dataframe(df, min_rows=1, min_cols=1)
            
            self.logger.info(f"Preprocessed data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load preprocessed data from {file_path}: {str(e)}")
            raise DataLoadingException(file_path, e)
    
    @pipeline_error_handler("feature_engineering")
    def create_list_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create list-based features from comma-separated values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with list features
            
        Raises:
        -------
        FeatureEngineeringException
            If list feature creation fails
        """
        try:
            self.logger.info("Creating list-based features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Create TANILAR_LIST from TANILAR column
            if "TANILAR" in df.columns:
                df["TANILAR_LIST"] = df["TANILAR"].fillna("").apply(
                    lambda x: [d.strip() for d in x.split(",") if d.strip()]
                )
                self.logger.info("TANILAR_LIST feature created")
            else:
                self.logger.warning("TANILAR column not found, skipping TANILAR_LIST creation")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create list features: {str(e)}")
            raise FeatureEngineeringException("list_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def create_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create count features for various medical conditions and treatments.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with count features
            
        Raises:
        -------
        FeatureEngineeringException
            If count feature creation fails
        """
        try:
            self.logger.info("Creating count features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            features_created = []
            
            # Count of diagnoses
            if "TANILAR_LIST" in df.columns:
                df["TANI_SAYI"] = df["TANILAR_LIST"].apply(len)
                features_created.append("TANI_SAYI")
            
            # Count of allergies
            if "ALERJI" in df.columns:
                df["ALERJI_SAYI"] = df["ALERJI"].apply(
                    lambda x: 0 if x == "yok" else len([a.strip() for a in x.split(",") if a.strip()])
                )
                features_created.append("ALERJI_SAYI")
            
            # Count of application areas
            if "UYGULAMAYERLERI" in df.columns:
                df["UYGULAMAYERLERI_SAYI"] = df["UYGULAMAYERLERI"].fillna("").apply(
                    lambda x: len([item.strip() for item in x.split(",") if item.strip()])
                )
                features_created.append("UYGULAMAYERLERI_SAYI")
            
            # Count of chronic diseases
            if "KRONIKHASTALIK" in df.columns:
                df["KRONIKHASTALIK_SAYI"] = df["KRONIKHASTALIK"].fillna("").apply(
                    lambda x: len([item.strip() for item in x.split(",") if item.strip()])
                )
                features_created.append("KRONIKHASTALIK_SAYI")
            
            # Count of treatments (using + separator)
            if "TEDAVIADI" in df.columns:
                df["TEDAVIADI_SAYI"] = df["TEDAVIADI"].fillna("").apply(
                    lambda x: len([item.strip() for item in x.split("+") if item.strip()])
                )
                features_created.append("TEDAVIADI_SAYI")
            
            self.logger.info(f"Count features created: {features_created}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create count features: {str(e)}")
            raise FeatureEngineeringException("count_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def map_diagnosis_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map diagnoses to categories using external category mapping.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with diagnosis categories
            
        Raises:
        -------
        FeatureEngineeringException
            If diagnosis category mapping fails
        """
        try:
            self.logger.info("Mapping diagnosis categories...")
            
            validate_dataframe(df)
            df = df.copy()
            
            if "TANILAR_LIST" not in df.columns:
                self.logger.warning("TANILAR_LIST column not found, skipping diagnosis category mapping")
                df["TANI_CATEGORY_LIST"] = ""
                return df
            
            try:
                # Load diagnosis categories
                category_file = "/Users/user/Desktop/Projects/ds_case_pusula/tanilar_with_categories.csv"
                tani_categories = pd.read_csv(category_file)
                
                # Explode the TANILAR_LIST to create one row per diagnosis
                df_exploded = df.explode("TANILAR_LIST")
                
                # Merge with categories
                merged = df_exploded.merge(
                    tani_categories, 
                    left_on="TANILAR_LIST", 
                    right_on="Diagnosis", 
                    how="left"
                )
                
                # Group back and create category list
                df["TANI_CATEGORY_LIST"] = (
                    merged.groupby(merged.index)["Category"]
                    .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
                )
                
                self.logger.info("Diagnosis categories mapped successfully")
                
            except FileNotFoundError:
                self.logger.warning("Category mapping file not found. Creating empty category list.")
                df["TANI_CATEGORY_LIST"] = ""
            except Exception as e:
                self.logger.warning(f"Failed to load category mapping: {str(e)}. Creating empty category list.")
                df["TANI_CATEGORY_LIST"] = ""
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to map diagnosis categories: {str(e)}")
            raise FeatureEngineeringException("diagnosis_category_mapping", e)
    
    @pipeline_error_handler("feature_engineering")
    def create_chronic_disease_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to chronic diseases including groupings and risk scores.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with chronic disease features
            
        Raises:
        -------
        FeatureEngineeringException
            If chronic disease feature creation fails
        """
        try:
            self.logger.info("Creating chronic disease features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            if "KRONIKHASTALIK" not in df.columns:
                self.logger.warning("KRONIKHASTALIK column not found, skipping chronic disease features")
                return df
            
            # Define disease groups
            groups = {
                "diabetes": ["diyabet"],
                "hypertension": ["hipertansiyon"],
                "cardiac": ["kalp yetmezliği", "aritmi"],
                "respiratory": ["astım"],
                "muscular_dystrophy": ["becker musküler distrofisi", "duchenne musküler distrofisi",
                                       "limb-girdle musküler distrofi", "fascioscapulohumeral distrofi"],
                "thyroid": ["hipotiroidizm", "hipertiroidizm", "guatr"]
            }
            
            # Define risk weights for different conditions
            risk_weights = {
                "kalp yetmezliği": 2,
                "aritmi": 2,
                "duchenne musküler distrofisi": 2,
                "myastenia gravis": 2,
                "becker musküler distrofisi": 1,
                "limb-girdle musküler distrofi": 1,
                "fascioscapulohumeral distrofi": 1,
                "hipotiroidizm": 1,
                "hipertiroidizm": 1,
                "diyabet": 1,
                "hipertansiyon": 1,
                "astım": 1,
                "polimiyozit": 1
            }
            
            def process_chronic(row):
                """Process chronic disease string into list."""
                if pd.isna(row):
                    return []
                return [x.strip().lower() for x in row.split(",")]
            
            def calc_risk(lst):
                """Calculate risk score for a list of conditions."""
                return sum(risk_weights.get(x, 0) for x in lst)
            
            # Create chronic disease list
            df["CHRONIC_LIST"] = df["KRONIKHASTALIK"].apply(process_chronic)
            
            # Create multi-morbidity flag
            if "KRONIKHASTALIK_SAYI" in df.columns:
                df["MULTI_MORBIDITY_FLAG"] = (df["KRONIKHASTALIK_SAYI"] > 1).astype(int)
            
            # Create disease group flags
            for group, keywords in groups.items():
                df[f"HAS_{group.upper()}"] = df["CHRONIC_LIST"].apply(
                    lambda lst: int(any(k in lst for k in keywords))
                )
            
            # Calculate risk score
            df["RISK_SKORU"] = df["CHRONIC_LIST"].apply(calc_risk)
            
            # Clean up temporary column
            df = df.drop(columns=["CHRONIC_LIST"])
            
            features_created = ["MULTI_MORBIDITY_FLAG", "RISK_SKORU"] + [f"HAS_{group.upper()}" for group in groups.keys()]
            self.logger.info(f"Chronic disease features created: {features_created}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create chronic disease features: {str(e)}")
            raise FeatureEngineeringException("chronic_disease_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def create_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create treatment-related features and categorizations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with treatment features
            
        Raises:
        -------
        FeatureEngineeringException
            If treatment feature creation fails
        """
        try:
            self.logger.info("Creating treatment features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            if "TEDAVIADI" not in df.columns:
                self.logger.warning("TEDAVIADI column not found, skipping treatment features")
                return df
            
            # Define treatment groups
            treatment_groups = {
                "physiotherapy": ["fizik", "rehabilitasyon", "egzersiz", "trapez", "skapular", "boyun", "sırt"],
                "respiratory": ["solunum", "nefes", "oksijen", "pulmoner"],
                "electrotherapy": ["tens", "ultrason", "elektro", "stimülasyon"],
                "massage": ["masaj", "manuel terapi", "gevşetme"],
                "orthopedic": ["eklem", "protez", "ortopedik", "implant"],
                "other": []
            }
            
            def map_treatment_category(text):
                """Map treatment text to category."""
                if pd.isna(text):
                    return "unknown"
                text = text.lower()
                for cat, keywords in treatment_groups.items():
                    if any(kw in text for kw in keywords):
                        return cat
                return "other"
            
            # Create treatment category
            df["TEDAVI_KATEGORISI"] = df["TEDAVIADI"].apply(map_treatment_category)
            
            # Count unique treatments
            df["TEDAVI_SAYISI_UNIQUE"] = df["TEDAVIADI"].apply(
                lambda x: len(set([t.strip().lower() for t in x.split(",")])) if pd.notna(x) else 0
            )
            
            # Get most frequent treatment
            df["EN_SIK_TEDAVI"] = df["TEDAVIADI"].apply(
                lambda x: x.split(",")[0].strip().lower() if pd.notna(x) else np.nan
            )
            
            # Create binary flags for each treatment category
            for cat in treatment_groups.keys():
                df[f"IS_{cat.upper()}"] = (df["TEDAVI_KATEGORISI"] == cat).astype(int)
            
            features_created = ["TEDAVI_KATEGORISI", "TEDAVI_SAYISI_UNIQUE", "EN_SIK_TEDAVI"] + [f"IS_{cat.upper()}" for cat in treatment_groups.keys()]
            self.logger.info(f"Treatment features created: {features_created}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create treatment features: {str(e)}")
            raise FeatureEngineeringException("treatment_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-based features and categorizations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with age features
            
        Raises:
        -------
        FeatureEngineeringException
            If age feature creation fails
        """
        try:
            self.logger.info("Creating age features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            if "YAS" not in df.columns:
                self.logger.warning("YAS column not found, skipping age features")
                return df
            
            # Create age labels/bins
            df["YAS_LABELS"] = pd.cut(
                df["YAS"],
                bins=[0, 2, 12, 18, 30, 45, 65, 92],
                labels=[
                    "BEBEK",
                    "COCUK", 
                    "ERGEN",
                    "GENC_YETISKIN",
                    "ORTA_YAS",
                    "YASLI",
                    "ILERI_YAS"
                ]
            )
            
            self.logger.info("Age features created: YAS_LABELS")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create age features: {str(e)}")
            raise FeatureEngineeringException("age_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def process_kangrubu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process KANGRUBU (blood group) column to extract blood type and Rh factor.
        
        Creates two new features:
        - KANGRUBU_TYPE: Blood type (0, A, B, AB)
        - KANGRUBU_POSITIVE: Rh factor (1 for positive, 0 for negative)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with blood group features
            
        Raises:
        -------
        FeatureEngineeringException
            If blood group feature creation fails
        """
        try:
            self.logger.info("Processing KANGRUBU features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            if "KANGRUBU" not in df.columns:
                self.logger.warning("KANGRUBU column not found, skipping blood group features")
                return df
            
            def parse_blood_group(blood_group_str):
                """
                Parse blood group string to extract type and Rh factor.
                
                Parameters:
                -----------
                blood_group_str : str
                    Blood group string like "0 Rh+", "A Rh-", etc.
                    
                Returns:
                --------
                tuple
                    (blood_type, is_positive) where blood_type is str and is_positive is int
                """
                if pd.isna(blood_group_str) or blood_group_str == "":
                    return None, None
                
                # Convert to string and clean
                blood_str = str(blood_group_str).strip().upper()
                
                # Extract blood type (0, A, B, AB)
                blood_type = None
                if blood_str.startswith("0 "):
                    blood_type = "0"
                elif blood_str.startswith("A "):
                    blood_type = "A"
                elif blood_str.startswith("B "):
                    blood_type = "B"
                elif blood_str.startswith("AB "):
                    blood_type = "AB"
                else:
                    # Try to extract from the beginning of the string
                    if "AB" in blood_str:
                        blood_type = "AB"
                    elif "A" in blood_str:
                        blood_type = "A"
                    elif "B" in blood_str:
                        blood_type = "B"
                    elif "0" in blood_str:
                        blood_type = "0"
                
                # Extract Rh factor
                is_positive = None
                if "RH+" in blood_str or "+" in blood_str:
                    is_positive = 1
                elif "RH-" in blood_str or "-" in blood_str:
                    is_positive = 0
                
                return blood_type, is_positive
            
            # Apply parsing function
            parsed_results = df["KANGRUBU"].apply(parse_blood_group)
            
            # Extract blood type and Rh factor into separate columns
            df["KANGRUBU_TYPE"] = [result[0] for result in parsed_results]
            df["KANGRUBU_POSITIVE"] = [result[1] for result in parsed_results]
            
            # Log statistics
            type_counts = df["KANGRUBU_TYPE"].value_counts()
            positive_counts = df["KANGRUBU_POSITIVE"].value_counts()
            
            self.logger.info(f"Blood type distribution: {type_counts.to_dict()}")
            self.logger.info(f"Rh factor distribution: {positive_counts.to_dict()}")
            self.logger.info("KANGRUBU features created: KANGRUBU_TYPE, KANGRUBU_POSITIVE")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to process KANGRUBU features: {str(e)}")
            raise FeatureEngineeringException("kangrubu_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def create_repetitive_data_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create REPETITIVE_DATA feature that counts how many times each row appears in the dataset.
        
        This function identifies duplicate rows and adds a column showing the count of
        identical rows for each record in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with REPETITIVE_DATA column added
            
        Raises:
        -------
        FeatureEngineeringException
            If repetitive data feature creation fails
        """
        try:
            self.logger.info("Creating repetitive data feature...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Count duplicates for each row
            # This counts how many times each unique combination of hashable columns appears
            # We need to exclude list columns as they are unhashable
            
            # Identify hashable columns (exclude list/unhashable types)
            hashable_columns = []
            for col in df.columns:
                try:
                    # Check if column contains lists or other unhashable types
                    # Test with a few non-null values to determine if column is hashable
                    non_null_values = df[col].dropna()
                    if len(non_null_values) == 0:
                        # Column has only null values, it's hashable
                        hashable_columns.append(col)
                        continue
                    
                    # Test the first few non-null values
                    test_values = non_null_values.head(min(5, len(non_null_values)))
                    for val in test_values:
                        if isinstance(val, (list, dict, set)):
                            # Definitely unhashable
                            raise TypeError("Unhashable type detected")
                        try:
                            hash(val)
                        except TypeError:
                            # Value is unhashable
                            raise TypeError("Unhashable value detected")
                    
                    # If we get here, column seems hashable
                    hashable_columns.append(col)
                    
                except (TypeError, AttributeError):
                    # Skip unhashable columns (like lists)
                    self.logger.info(f"Skipping unhashable column '{col}' from duplicate detection")
                    continue
            
            if not hashable_columns:
                self.logger.warning("No hashable columns found for duplicate detection. Setting all REPETITIVE_DATA to 1.")
                df["REPETITIVE_DATA"] = 1
            else:
                self.logger.info(f"Using {len(hashable_columns)} hashable columns for duplicate detection: {hashable_columns}")
                
                # Use a more robust approach for groupby with potential NaN values
                try:
                    df["REPETITIVE_DATA"] = df.groupby(hashable_columns, dropna=False, observed=False).transform('size')
                except Exception as groupby_error:
                    self.logger.warning(f"Groupby failed with error: {str(groupby_error)}. Falling back to alternative method.")
                    # Alternative method: create a hash-based approach
                    df_subset = df[hashable_columns].copy()
                    # Convert to string representation for hashing
                    df_subset_str = df_subset.astype(str)
                    # Create a combined key
                    combined_key = df_subset_str.apply(lambda row: '|'.join(row.values), axis=1)
                    df["REPETITIVE_DATA"] = combined_key.groupby(combined_key).transform('size')
            
            # Log statistics about duplicates
            duplicate_stats = df["REPETITIVE_DATA"].value_counts().sort_index()
            total_unique_rows = len(df["REPETITIVE_DATA"].unique())
            total_rows = len(df)
            
            # Calculate duplicate rows using only hashable columns to avoid the same error
            if hashable_columns:
                duplicate_rows = total_rows - len(df[hashable_columns].drop_duplicates())
            else:
                duplicate_rows = 0  # No hashable columns means we can't detect duplicates
            
            self.logger.info(f"Total rows: {total_rows}")
            self.logger.info(f"Unique row patterns: {total_unique_rows}")
            self.logger.info(f"Duplicate rows: {duplicate_rows}")
            self.logger.info(f"Repetition distribution: {duplicate_stats.to_dict()}")
            
            # Log some examples of highly repetitive data
            max_repetitions = df["REPETITIVE_DATA"].max()
            if max_repetitions > 1:
                most_repetitive = df[df["REPETITIVE_DATA"] == max_repetitions].iloc[0:1]
                self.logger.info(f"Most repetitive pattern appears {max_repetitions} times")
                
                # Show which columns have the same values for the most repetitive pattern
                if len(most_repetitive) > 0:
                    sample_row = most_repetitive.iloc[0]
                    self.logger.info(f"Example of most repetitive pattern (first few columns): "
                                   f"HASTANO={sample_row.get('HASTANO', 'N/A')}, "
                                   f"YAS={sample_row.get('YAS', 'N/A')}, "
                                   f"CINSIYET={sample_row.get('CINSIYET', 'N/A')}")
            
            self.logger.info("REPETITIVE_DATA feature created successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create repetitive data feature: {str(e)}")
            raise FeatureEngineeringException("repetitive_data_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def create_hastano_count_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create HASTANO_COUNT feature that counts how many records each patient (HASTANO) has.
        
        This function counts the number of records for each unique patient ID and adds
        a column showing this count for each record.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with HASTANO_COUNT column added
            
        Raises:
        -------
        FeatureEngineeringException
            If HASTANO count feature creation fails
        """
        try:
            self.logger.info("Creating HASTANO count feature...")
            
            validate_dataframe(df)
            df = df.copy()
            
            if "HASTANO" not in df.columns:
                self.logger.warning("HASTANO column not found, skipping HASTANO count feature")
                df["HASTANO_COUNT"] = 1  # Default to 1 if no HASTANO column
                return df
            
            # Count occurrences of each HASTANO
            hastano_counts = df["HASTANO"].value_counts()
            
            # Map the counts back to the dataframe
            df["HASTANO_COUNT"] = df["HASTANO"].map(hastano_counts)
            
            # Log statistics about patient records
            unique_patients = len(hastano_counts)
            total_records = len(df)
            patients_with_multiple_records = (hastano_counts > 1).sum()
            max_records_per_patient = hastano_counts.max()
            avg_records_per_patient = hastano_counts.mean()
            
            # Distribution of record counts per patient
            count_distribution = hastano_counts.value_counts().sort_index()
            
            self.logger.info(f"Total records: {total_records}")
            self.logger.info(f"Unique patients: {unique_patients}")
            self.logger.info(f"Patients with multiple records: {patients_with_multiple_records}")
            self.logger.info(f"Max records per patient: {max_records_per_patient}")
            self.logger.info(f"Average records per patient: {avg_records_per_patient:.2f}")
            self.logger.info(f"Distribution of records per patient: {count_distribution.to_dict()}")
            
            # Log examples of patients with most records
            if max_records_per_patient > 1:
                most_frequent_patients = hastano_counts[hastano_counts == max_records_per_patient]
                self.logger.info(f"Patient(s) with most records ({max_records_per_patient}): {most_frequent_patients.index.tolist()}")
                
                # Show sample data for a patient with multiple records
                sample_patient = most_frequent_patients.index[0]
                sample_records = df[df["HASTANO"] == sample_patient][["HASTANO", "YAS", "CINSIYET"]].head(3)
                self.logger.info(f"Sample records for patient {sample_patient}:")
                for idx, row in sample_records.iterrows():
                    self.logger.info(f"  Record {idx}: Age={row.get('YAS', 'N/A')}, Gender={row.get('CINSIYET', 'N/A')}")
            
            self.logger.info("HASTANO_COUNT feature created successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create HASTANO count feature: {str(e)}")
            raise FeatureEngineeringException("hastano_count_feature_creation", e)
    
    @pipeline_error_handler("feature_engineering")
    def scale_numerical_features(self, df: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features using different scaling methods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        scaler_type : str
            Type of scaler to use ('standard', 'minmax', 'robust')
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with scaled numerical features
            
        Raises:
        -------
        FeatureEngineeringException
            If scaling fails
        """
        try:
            self.logger.info(f"Scaling numerical features using {scaler_type} scaler...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Define columns to exclude from scaling
            exclude_columns = [
                'HASTANO',  # ID column
                'TANILAR_LIST',  # List column
                'MULTI_MORBIDITY_FLAG',  # Binary flags
                'HAS_DIABETES', 'HAS_HYPERTENSION', 'HAS_CARDIAC', 'HAS_RESPIRATORY',
                'HAS_MUSCULAR_DYSTROPHY', 'HAS_THYROID',
                'IS_PHYSIOTHERAPY', 'IS_RESPIRATORY', 'IS_ELECTROTHERAPY', 
                'IS_MASSAGE', 'IS_ORTHOPEDIC', 'IS_OTHER',
                'KANGRUBU_POSITIVE',  # Binary column
                'REPETITIVE_DATA', 'HASTANO_COUNT'  # Count columns that might be useful unscaled
            ]
            
            # Get numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove excluded columns
            cols_to_scale = [col for col in numerical_cols if col not in exclude_columns]
            
            if not cols_to_scale:
                self.logger.warning("No numerical columns found for scaling")
                return df
            
            # Initialize scaler
            if scaler_type.lower() == 'standard':
                scaler = StandardScaler()
                scaler_name = "StandardScaler (mean=0, std=1)"
            elif scaler_type.lower() == 'minmax':
                scaler = MinMaxScaler()
                scaler_name = "MinMaxScaler (range=[0,1])"
            elif scaler_type.lower() == 'robust':
                scaler = RobustScaler()
                scaler_name = "RobustScaler (median=0, IQR=1)"
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            # Apply scaling
            original_stats = df[cols_to_scale].describe()
            
            # Fit and transform
            scaled_data = scaler.fit_transform(df[cols_to_scale])
            
            # Create new column names for scaled features
            scaled_columns = [f"{col}_SCALED" for col in cols_to_scale]
            
            # Add scaled columns to dataframe
            scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=df.index)
            df = pd.concat([df, scaled_df], axis=1)
            
            # Log scaling statistics
            scaled_stats = df[scaled_columns].describe()
            
            self.logger.info(f"Applied {scaler_name} to {len(cols_to_scale)} numerical columns")
            self.logger.info(f"Columns scaled: {cols_to_scale}")
            self.logger.info(f"New scaled columns created: {scaled_columns}")
            
            # Log before/after statistics for first few columns
            for i, col in enumerate(cols_to_scale[:3]):  # Show stats for first 3 columns
                orig_col = col
                scaled_col = scaled_columns[i]
                self.logger.info(f"{orig_col}: mean {original_stats.loc['mean', orig_col]:.3f} → {scaled_stats.loc['mean', scaled_col]:.3f}, "
                               f"std {original_stats.loc['std', orig_col]:.3f} → {scaled_stats.loc['std', scaled_col]:.3f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to scale numerical features: {str(e)}")
            raise FeatureEngineeringException("numerical_scaling", e)
    
    @pipeline_error_handler("feature_engineering")
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply various encoding techniques to categorical features based on their characteristics.
        
        Encoding Strategy:
        - Binary columns (2 unique values): Label Encoding
        - Low cardinality (≤10 unique): One-Hot Encoding
        - Medium cardinality (11-50 unique): Target Encoding + Binary Encoding
        - High cardinality (>50 unique): Hashing Encoding + Count Encoding
        - Text-like columns: TF-IDF Encoding (limited features)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical features
            
        Raises:
        -------
        FeatureEngineeringException
            If encoding fails
        """
        try:
            self.logger.info("Encoding categorical features...")
            
            validate_dataframe(df)
            df = df.copy()
            
            # Identify categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove columns that shouldn't be encoded
            exclude_columns = ['TANILAR_LIST']  # List columns
            categorical_cols = [col for col in categorical_cols if col not in exclude_columns]
            
            if not categorical_cols:
                self.logger.warning("No categorical columns found for encoding")
                return df
            
            # Target variable for target encoding
            target_col = self.target_column if self.target_column in df.columns else None
            
            encoding_summary = []
            
            for col in categorical_cols:
                try:
                    self.logger.info(f"Processing column: {col}")
                    
                    # Get basic statistics
                    unique_count = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    
                    # Clean the column - fill nulls with 'Unknown'
                    df[col] = df[col].fillna('Unknown')
                    
                    # Determine encoding strategy based on cardinality
                    if unique_count <= 2:
                        # Binary columns: Label Encoding
                        encoding_type = "Label Encoding"
                        le = LabelEncoder()
                        encoded_col = f"{col}_LABEL_ENCODED"
                        df[encoded_col] = le.fit_transform(df[col].astype(str))
                        
                        encoding_summary.append({
                            'Column': col,
                            'Unique_Values': unique_count,
                            'Encoding_Type': encoding_type,
                            'New_Columns': [encoded_col],
                            'New_Features_Count': 1
                        })
                        
                    elif unique_count <= 10:
                        # Low cardinality: One-Hot Encoding
                        encoding_type = "One-Hot Encoding"
                        
                        # Create dummy variables
                        dummy_df = pd.get_dummies(df[col], prefix=f"{col}_OH", dummy_na=False)
                        df = pd.concat([df, dummy_df], axis=1)
                        
                        new_cols = dummy_df.columns.tolist()
                        encoding_summary.append({
                            'Column': col,
                            'Unique_Values': unique_count,
                            'Encoding_Type': encoding_type,
                            'New_Columns': new_cols,
                            'New_Features_Count': len(new_cols)
                        })
                        
                    elif unique_count <= 50:
                        # Medium cardinality: Target Encoding + Binary Encoding
                        encoding_type = "Target + Binary Encoding"
                        new_cols = []
                        
                        # Target Encoding (if target is available)
                        if target_col and target_col in df.columns:
                            te = TargetEncoder(cols=[col])
                            target_encoded = te.fit_transform(df[col], df[target_col])
                            target_encoded_col = f"{col}_TARGET_ENCODED"
                            df[target_encoded_col] = target_encoded
                            new_cols.append(target_encoded_col)
                        
                        # Binary Encoding
                        be = BinaryEncoder(cols=[col])
                        binary_encoded = be.fit_transform(df[col])
                        
                        # Rename binary encoded columns
                        binary_cols = [f"{col}_BINARY_{i}" for i in range(len(binary_encoded.columns))]
                        binary_encoded.columns = binary_cols
                        df = pd.concat([df, binary_encoded], axis=1)
                        new_cols.extend(binary_cols)
                        
                        encoding_summary.append({
                            'Column': col,
                            'Unique_Values': unique_count,
                            'Encoding_Type': encoding_type,
                            'New_Columns': new_cols,
                            'New_Features_Count': len(new_cols)
                        })
                        
                    else:
                        # High cardinality: Hashing + Count Encoding
                        encoding_type = "Hashing + Count Encoding"
                        new_cols = []
                        
                        # Hashing Encoding (limit to 8 features to avoid explosion)
                        he = HashingEncoder(cols=[col], n_components=8)
                        hash_encoded = he.fit_transform(df[col])
                        hash_cols = [f"{col}_HASH_{i}" for i in range(len(hash_encoded.columns))]
                        hash_encoded.columns = hash_cols
                        df = pd.concat([df, hash_encoded], axis=1)
                        new_cols.extend(hash_cols)
                        
                        # Count Encoding
                        ce = CountEncoder(cols=[col])
                        count_encoded = ce.fit_transform(df[col])
                        count_col = f"{col}_COUNT_ENCODED"
                        df[count_col] = count_encoded
                        new_cols.append(count_col)
                        
                        encoding_summary.append({
                            'Column': col,
                            'Unique_Values': unique_count,
                            'Encoding_Type': encoding_type,
                            'New_Columns': new_cols,
                            'New_Features_Count': len(new_cols)
                        })
                    
                    self.logger.info(f"Applied {encoding_type} to {col} ({unique_count} unique values)")
                    
                except Exception as col_error:
                    self.logger.error(f"Failed to encode column {col}: {str(col_error)}")
                    # Continue with other columns
                    continue
            
            # Handle text-like columns with TF-IDF (for diagnosis and treatment columns)
            text_columns = ['TANILAR', 'TEDAVIADI', 'KRONIKHASTALIK']
            
            for col in text_columns:
                if col in df.columns and col in categorical_cols:
                    try:
                        self.logger.info(f"Applying TF-IDF encoding to text column: {col}")
                        
                        # Prepare text data
                        text_data = df[col].fillna('').astype(str)
                        
                        # Apply TF-IDF with limited features
                        tfidf = TfidfVectorizer(
                            max_features=10,  # Limit to top 10 features
                            stop_words=None,  # Keep all words for medical terms
                            lowercase=True,
                            ngram_range=(1, 1)  # Only unigrams
                        )
                        
                        tfidf_matrix = tfidf.fit_transform(text_data)
                        feature_names = tfidf.get_feature_names_out()
                        
                        # Create DataFrame with TF-IDF features
                        tfidf_df = pd.DataFrame(
                            tfidf_matrix.toarray(), 
                            columns=[f"{col}_TFIDF_{name}" for name in feature_names],
                            index=df.index
                        )
                        
                        df = pd.concat([df, tfidf_df], axis=1)
                        
                        encoding_summary.append({
                            'Column': col,
                            'Unique_Values': df[col].nunique(),
                            'Encoding_Type': 'TF-IDF Encoding',
                            'New_Columns': tfidf_df.columns.tolist(),
                            'New_Features_Count': len(tfidf_df.columns)
                        })
                        
                        self.logger.info(f"Applied TF-IDF encoding to {col} (created {len(tfidf_df.columns)} features)")
                        
                    except Exception as tfidf_error:
                        self.logger.error(f"Failed to apply TF-IDF to {col}: {str(tfidf_error)}")
                        continue
            
            # Log encoding summary
            total_new_features = sum([item['New_Features_Count'] for item in encoding_summary])
            self.logger.info(f"Encoding completed: {len(encoding_summary)} columns processed")
            self.logger.info(f"Total new encoded features created: {total_new_features}")
            
            # Save encoding summary
            if encoding_summary:
                encoding_df = pd.DataFrame(encoding_summary)
                
                # Log summary by encoding type
                encoding_type_summary = encoding_df.groupby('Encoding_Type').agg({
                    'Column': 'count',
                    'New_Features_Count': 'sum'
                }).rename(columns={'Column': 'Columns_Count'})
                
                self.logger.info("Encoding summary by type:")
                for enc_type, row in encoding_type_summary.iterrows():
                    self.logger.info(f"  {enc_type}: {row['Columns_Count']} columns → {row['New_Features_Count']} features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to encode categorical features: {str(e)}")
            raise FeatureEngineeringException("categorical_encoding", e)
    
    @pipeline_error_handler("feature_engineering")
    def remove_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove useless columns to create a clean, ML-ready dataset.
        
        Removes:
        - Original categorical columns that have been encoded
        - Columns with zero or very low variance
        - Single-value columns
        - List columns that can't be used in ML
        - Redundant columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe with useless columns removed
            
        Raises:
        -------
        FeatureEngineeringException
            If column removal fails
        """
        try:
            self.logger.info("Removing useless columns for ML-ready dataset...")
            
            validate_dataframe(df)
            df = df.copy()
            
            original_shape = df.shape
            columns_to_remove = []
            removal_reasons = {}
            
            # 1. Remove original categorical columns that have encoded versions
            original_categorical_cols = [
                'CINSIYET', 'KANGRUBU', 'UYRUK', 'KRONIKHASTALIK', 'BOLUM', 
                'ALERJI', 'TANILAR', 'TEDAVIADI', 'UYGULAMAYERLERI'
            ]
            
            for col in original_categorical_cols:
                if col in df.columns:
                    # Check if there are encoded versions
                    encoded_versions = [c for c in df.columns if c.startswith(f'{col}_') and 
                                      any(enc in c for enc in ['_OH_', '_LABEL_', '_TARGET_', '_BINARY_', '_HASH_', '_COUNT_', '_TFIDF_'])]
                    if encoded_versions:
                        columns_to_remove.append(col)
                        removal_reasons[col] = f"Redundant - has {len(encoded_versions)} encoded versions"
            
            # 2. Remove columns with zero variance (single values)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].var() == 0 or df[col].nunique() <= 1:
                    columns_to_remove.append(col)
                    removal_reasons[col] = f"Zero variance - {df[col].nunique()} unique value(s)"
            
            # 3. Remove columns with extremely low variance (nearly constant)
            for col in numeric_cols:
                if col not in columns_to_remove and df[col].var() < 0.001:
                    columns_to_remove.append(col)
                    removal_reasons[col] = f"Very low variance: {df[col].var():.6f}"
            
            # 4. Remove list columns that can't be used in ML
            for col in df.columns:
                if col not in columns_to_remove and df[col].dtype == 'object':
                    # Check if contains lists
                    sample_vals = df[col].dropna().head(3)
                    for val in sample_vals:
                        if isinstance(val, list) or (isinstance(val, str) and val.startswith('[')):
                            columns_to_remove.append(col)
                            removal_reasons[col] = "Contains list data - not ML compatible"
                            break
            
            # 5. Remove highly correlated features (keep one from each highly correlated pair)
            correlation_threshold = 0.95
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().abs()
                
                # Find pairs of highly correlated features
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > correlation_threshold:
                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
                
                # Remove one column from each highly correlated pair (keep the one with higher target correlation)
                if self.target_column in df.columns:
                    target_corrs = numeric_df.corrwith(df[self.target_column]).abs()
                    
                    for col1, col2, corr_val in high_corr_pairs:
                        if col1 not in columns_to_remove and col2 not in columns_to_remove:
                            # Keep the one with higher target correlation
                            if target_corrs[col1] >= target_corrs[col2]:
                                columns_to_remove.append(col2)
                                removal_reasons[col2] = f"Highly correlated with {col1} (r={corr_val:.3f})"
                            else:
                                columns_to_remove.append(col1)
                                removal_reasons[col1] = f"Highly correlated with {col2} (r={corr_val:.3f})"
            
            # 6. Remove ID columns that are not useful for ML
            id_columns = ['HASTANO']  # Patient ID is not useful for general ML models
            for col in id_columns:
                if col in df.columns and col not in columns_to_remove:
                    columns_to_remove.append(col)
                    removal_reasons[col] = "ID column - not useful for ML"
            
            # Remove duplicates from removal list
            columns_to_remove = list(set(columns_to_remove))
            
            # Actually remove the columns
            if columns_to_remove:
                df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
                
                # Log removal details
                self.logger.info(f"Removed {len(columns_to_remove)} useless columns")
                self.logger.info(f"Shape change: {original_shape} → {df_cleaned.shape}")
                
                # Log removal reasons
                removal_summary = {}
                for col in columns_to_remove:
                    reason_type = removal_reasons[col].split(' - ')[0] if ' - ' in removal_reasons[col] else removal_reasons[col].split(':')[0]
                    if reason_type not in removal_summary:
                        removal_summary[reason_type] = []
                    removal_summary[reason_type].append(col)
                
                self.logger.info("Removal summary by reason:")
                for reason, cols in removal_summary.items():
                    self.logger.info(f"  {reason}: {len(cols)} columns")
                
                # Log first few removed columns as examples
                self.logger.info(f"Examples of removed columns: {columns_to_remove[:5]}")
                
                return df_cleaned
            else:
                self.logger.info("No useless columns identified for removal")
                return df
            
        except Exception as e:
            self.logger.error(f"Failed to remove useless columns: {str(e)}")
            raise FeatureEngineeringException("useless_column_removal", e)
    
    @pipeline_error_handler("feature_engineering")
    def ensure_all_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns in the final dataset are numerical (int or float).
        
        Converts:
        - Boolean columns to int (0/1)
        - Remaining categorical columns to numerical using appropriate encoding
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with all numerical columns
            
        Raises:
        -------
        FeatureEngineeringException
            If numerical conversion fails
        """
        try:
            self.logger.info("Converting all columns to numerical format...")
            
            validate_dataframe(df)
            df = df.copy()
            
            original_shape = df.shape
            conversion_summary = []
            
            # 1. Convert boolean columns to int (0/1)
            bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
            if bool_cols:
                for col in bool_cols:
                    df[col] = df[col].astype(int)
                    conversion_summary.append({
                        'Column': col,
                        'Original_Type': 'bool',
                        'New_Type': 'int64',
                        'Conversion': 'Boolean to Integer (0/1)'
                    })
                
                self.logger.info(f"Converted {len(bool_cols)} boolean columns to integers")
            
            # 2. Convert remaining object columns
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            for col in object_cols:
                try:
                    unique_count = df[col].nunique()
                    
                    # Fill any remaining nulls
                    df[col] = df[col].fillna('Unknown')
                    
                    if unique_count <= 2:
                        # Binary categorical: Simple label encoding
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        conversion_type = "Label Encoding (Binary)"
                        new_type = "int64"
                        
                    elif unique_count <= 20:
                        # Low cardinality: Label encoding (ordinal)
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        conversion_type = "Label Encoding (Ordinal)"
                        new_type = "int64"
                        
                    else:
                        # High cardinality: Use count encoding
                        value_counts = df[col].value_counts()
                        df[col] = df[col].map(value_counts)
                        conversion_type = "Count Encoding"
                        new_type = "int64"
                    
                    conversion_summary.append({
                        'Column': col,
                        'Original_Type': 'object',
                        'New_Type': new_type,
                        'Conversion': conversion_type,
                        'Unique_Values': unique_count
                    })
                    
                    self.logger.info(f"Converted {col} using {conversion_type} ({unique_count} unique values)")
                    
                except Exception as col_error:
                    self.logger.error(f"Failed to convert column {col}: {str(col_error)}")
                    # Try simple label encoding as fallback
                    try:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        conversion_summary.append({
                            'Column': col,
                            'Original_Type': 'object',
                            'New_Type': 'int64',
                            'Conversion': 'Fallback Label Encoding'
                        })
                        self.logger.info(f"Applied fallback label encoding to {col}")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback conversion also failed for {col}: {str(fallback_error)}")
                        continue
            
            # 3. Ensure all columns are now numerical
            final_dtypes = df.dtypes
            non_numeric_remaining = []
            
            for col in df.columns:
                if final_dtypes[col] not in ['int64', 'float64', 'int32', 'float32']:
                    non_numeric_remaining.append(col)
            
            if non_numeric_remaining:
                self.logger.warning(f"Warning: {len(non_numeric_remaining)} columns still not numerical: {non_numeric_remaining}")
                # Force conversion of any remaining non-numeric columns
                for col in non_numeric_remaining:
                    try:
                        # Try to convert to numeric, coercing errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Fill any NaN with 0
                        df[col] = df[col].fillna(0)
                        self.logger.info(f"Force converted {col} to numeric")
                    except Exception as force_error:
                        self.logger.error(f"Could not force convert {col}: {str(force_error)}")
            
            # Final verification
            final_dtypes = df.dtypes
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            total_cols = len(df.columns)
            
            self.logger.info(f"Numerical conversion completed")
            self.logger.info(f"Shape: {original_shape} → {df.shape}")
            self.logger.info(f"Numerical columns: {numeric_cols}/{total_cols}")
            
            if conversion_summary:
                # Log conversion summary
                conversion_df = pd.DataFrame(conversion_summary)
                conversion_type_summary = conversion_df['Conversion'].value_counts()
                self.logger.info("Conversion summary by type:")
                for conv_type, count in conversion_type_summary.items():
                    self.logger.info(f"  {conv_type}: {count} columns")
            
            # Final data type check
            final_type_counts = df.dtypes.value_counts()
            self.logger.info("Final data types:")
            for dtype, count in final_type_counts.items():
                self.logger.info(f"  {dtype}: {count} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to ensure all columns are numerical: {str(e)}")
            raise FeatureEngineeringException("numerical_conversion", e)
    
    def save_final_data(self, df: pd.DataFrame, output_file_path: str) -> None:
        """
        Save final cleaned data to the specified directory.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Final cleaned dataframe
        output_file_path : str
            Path to save the final data
            
        Raises:
        -------
        DataSavingException
            If data saving fails
        """
        try:
            self.logger.info(f"Saving final cleaned data to: {output_file_path}")
            
            validate_dataframe(df)
            validate_file_path(output_file_path, must_exist=False)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Save data
            df.to_csv(output_file_path, index=False)
            
            self.logger.info(f"Final data saved successfully. Shape: {df.shape}")
            
            # Log final dataset statistics
            self.logger.info("Final dataset summary:")
            self.logger.info(f"  Total features: {df.shape[1]}")
            self.logger.info(f"  Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}")
            self.logger.info(f"  Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
            self.logger.info(f"  Boolean features: {len(df.select_dtypes(include=['bool']).columns)}")
            self.logger.info(f"  Missing values: {df.isnull().sum().sum()}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final data to {output_file_path}: {str(e)}")
            raise DataSavingException(output_file_path, e)
    
    def save_data(self, df: pd.DataFrame, output_file_path: str) -> None:
        """
        Save feature engineered data to CSV file.
        
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
            self.logger.info(f"Saving feature engineered data to: {output_file_path}")
            
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
        Run the complete feature engineering pipeline.
        
        Parameters:
        -----------
        input_file_path : str
            Path to the preprocessed CSV file
        output_file_path : str
            Path to save the feature engineered data
            
        Returns:
        --------
        pd.DataFrame
            Feature engineered dataframe
            
        Raises:
        -------
        FeatureEngineeringException
            If any step in the pipeline fails
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STARTING FEATURE ENGINEERING PIPELINE")
            self.logger.info("="*50)
            
            # Step 1: Load preprocessed data
            df = self.load_preprocessed_data(input_file_path)
            
            # Step 2: Create list features
            df = self.create_list_features(df)
            
            # Step 3: Create count features
            df = self.create_count_features(df)
            
            # Step 4: Map diagnosis categories
            df = self.map_diagnosis_categories(df)
            
            # Step 5: Create chronic disease features
            df = self.create_chronic_disease_features(df)
            
            # Step 6: Create treatment features
            df = self.create_treatment_features(df)
            
            # Step 7: Create age features
            df = self.create_age_features(df)
            
            # Step 8: Process KANGRUBU features
            df = self.process_kangrubu_features(df)
            
            # Step 9: Create repetitive data feature
            df = self.create_repetitive_data_feature(df)
            
            # Step 10: Create HASTANO count feature
            df = self.create_hastano_count_feature(df)
            
            # Step 11: Encode categorical features
            df = self.encode_categorical_features(df)
            
            # Step 12: Scale numerical features
            df = self.scale_numerical_features(df, scaler_type='standard')
            
            # Step 13: Save feature engineered data (with all features)
            self.save_data(df, output_file_path)
            
            # Step 14: Remove useless columns for final ML-ready dataset
            df_cleaned = self.remove_useless_columns(df)
            
            # Step 15: Ensure all columns are numerical
            df_numeric = self.ensure_all_numeric(df_cleaned)
            
            # Step 16: Save final cleaned data
            final_output_path = output_file_path.replace('feature_engineering', 'data_final_version').replace('feature_engineering_data.csv', 'final_cleaned_data.csv')
            self.save_final_data(df_numeric, final_output_path)
            
            self.logger.info("="*50)
            self.logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
            self.logger.info(f"Feature engineering data shape: {df.shape}")
            self.logger.info(f"Final cleaned data shape: {df_numeric.shape}")
            self.logger.info("="*50)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering pipeline failed: {str(e)}")
            raise FeatureEngineeringException("pipeline_execution", e, {"input_file": input_file_path, "output_file": output_file_path})


def main():
    """
    Main function to run the feature engineering pipeline.
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        pipeline = FeatureEngineeringPipeline()
        
        # Define file paths
        BASE_DIR = "/Users/user/Desktop/Projects/ds_case_pusula/data"
        INPUT_FILE = os.path.join(BASE_DIR, "preprocessing", "preprocessed_data.csv")
        OUTPUT_FILE = os.path.join(BASE_DIR, "feature_engineering", "feature_engineering_data.csv")
        
        # Ensure data_final_version directory exists
        os.makedirs(os.path.join(BASE_DIR, "data_final_version"), exist_ok=True)
        
        logger.info(f"Starting feature engineering pipeline with input: {INPUT_FILE}")
        
        # Run pipeline
        df = pipeline.run_pipeline(INPUT_FILE, OUTPUT_FILE)
        
        logger.info(f"Feature engineering completed successfully!")
        logger.info(f"Feature engineered data saved to: {OUTPUT_FILE}")
        print(f"✅ Feature engineering completed successfully!")
        print(f"📁 Feature engineered data saved to: {OUTPUT_FILE}")
        print(f"📊 Final data shape: {df.shape}")
        
    except Exception as e:
        error_msg = f"❌ Feature engineering failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        raise


if __name__ == "__main__":
    main()