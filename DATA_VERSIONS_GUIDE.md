# Data Versions Guide

This document explains the different versions of datasets available in the project and their intended use cases.

## Dataset Overview

The project contains multiple versions of the dataset at different processing stages, each optimized for specific use cases:

### 1. **Raw Dataset**
- **File:** `data/Talent_Academy_Case_DT_2025.xlsx`
- **Shape:** 2,235 rows × Original columns
- **Description:** Original, unprocessed data from the hospital system
- **Use Case:** Initial data exploration, understanding source data structure
- **Features:** Raw medical records with original formatting and data types

### 2. **Preprocessed Dataset**
- **File:** `data/preprocessing/preprocessed_data.csv`
- **Shape:** 2,235 rows × Cleaned columns
- **Description:** Cleaned and standardized data after preprocessing pipeline
- **Use Case:** Clean data analysis, basic visualizations
- **Key Transformations:**
  - Column names standardized to UPPERCASE
  - Data types converted (removed suffixes like "Dakika", "Seans")
  - Missing values handled (ALERJI → "YOK", others → patient-specific imputation)
  - Text data normalized to lowercase
  - Typos corrected

### 3. **Feature-Engineered Dataset**
- **File:** `data/feature_engineering/feature_engineering_data.csv`
- **Shape:** 2,235 rows × 258 columns
- **Description:** Complete dataset with all original features plus engineered features
- **Use Case:** Comprehensive analysis, feature exploration, model development
- **Key Features:**
  - All original columns preserved
  - Advanced engineered features (risk scores, chronic disease flags, etc.)
  - Categorical encodings (one-hot, target, binary, hash)
  - TF-IDF features for text analysis
  - Scaled versions of numerical features
  - Mixed data types (numeric, categorical, boolean)

### 4. **ML-Ready Dataset (Cleaned)**
- **File:** `data/data_final_version/final_cleaned_data.csv`
- **Shape:** 2,235 rows × 140 columns
- **Description:** Machine learning optimized dataset with reduced features
- **Use Case:** Direct machine learning model training and evaluation
- **Key Characteristics:**
  - All numeric data types
  - Reduced feature set (118 columns removed for efficiency)
  - Optimized for model performance
  - No missing values

### 5. **Complete Dataset (No Drops)** ⭐ **NEW**
- **File:** `data/data_final_version/complete_dataset_no_drops.csv`
- **Shape:** 2,235 rows × 258 columns
- **Description:** **Complete dataset with ALL features preserved and converted to numeric**
- **Use Case:** **Comprehensive analysis, Tableau dashboards, complete feature exploration**
- **Key Advantages:**
  - **All 258 features preserved** (118 more than cleaned version)
  - **Fully numeric format** (ready for any analysis tool)
  - **No information loss** from feature removal
  - **Perfect for Tableau visualization** and comprehensive analytics
  - **Includes both original and engineered features**

## Feature Categories in Complete Dataset

### Original Features (Preserved)
- **Patient Info:** HASTANO, YAS, CINSIYET, KANGRUBU, UYRUK
- **Medical Data:** KRONIKHASTALIK, BOLUM, ALERJI, TANILAR, TEDAVIADI
- **Treatment Details:** TEDAVISURESI_SAANS_SAYI, UYGULAMAYERLERI, etc.

### Engineered Features
- **Count Features:** TANI_SAYI, ALERJI_SAYI, KRONIKHASTALIK_SAYI
- **Binary Flags:** HAS_DIABETES, HAS_HYPERTENSION, HAS_CARDIAC, etc.
- **Risk Scores:** RISK_SKORU, MULTI_MORBIDITY_FLAG
- **Category Mappings:** TANI_CATEGORY_LIST, TEDAVI_KATEGORISI
- **Encoded Features:** Hash encodings, target encodings, binary encodings
- **TF-IDF Features:** Text analysis features for medical terms
- **Scaled Features:** Standardized versions of numerical variables

## Recommended Usage by Tool

### For Tableau Dashboards
**Use:** `complete_dataset_no_drops.csv`
- **Why:** All features available for comprehensive visualization
- **Benefits:** Maximum flexibility, no information loss, fully numeric

### For Machine Learning
**Use:** `final_cleaned_data.csv` (optimized) or `complete_dataset_no_drops.csv` (comprehensive)
- **Optimized:** Faster training, reduced overfitting risk
- **Comprehensive:** Full feature exploration, feature importance analysis

### For Statistical Analysis
**Use:** `complete_dataset_no_drops.csv`
- **Why:** Complete feature set for thorough statistical investigation

### For Data Exploration
**Use:** `feature_engineering_data.csv` or `complete_dataset_no_drops.csv`
- **Why:** Rich feature set with original data types or fully numeric version

## Dataset Comparison

| Dataset | Rows | Columns | Data Types | Use Case |
|---------|------|---------|------------|----------|
| Raw | 2,235 | Original | Mixed | Source exploration |
| Preprocessed | 2,235 | Cleaned | Mixed | Clean analysis |
| Feature-Engineered | 2,235 | 258 | Mixed | Feature exploration |
| ML-Ready (Cleaned) | 2,235 | 140 | Numeric | Model training |
| **Complete (No Drops)** | **2,235** | **258** | **Numeric** | **Comprehensive analysis** |

## Key Advantages of Complete Dataset

1. **No Information Loss:** All 258 features preserved
2. **Tableau Ready:** Fully numeric format compatible with all visualization tools
3. **Comprehensive Analysis:** Access to both original and engineered features
4. **Maximum Flexibility:** Suitable for any type of analysis or modeling
5. **Production Ready:** Clean, validated, and properly formatted

## File Locations

```
data/
├── Talent_Academy_Case_DT_2025.xlsx                    # Raw data
├── preprocessing/
│   └── preprocessed_data.csv                           # Preprocessed
├── feature_engineering/
│   └── feature_engineering_data.csv                    # Feature-engineered
└── data_final_version/
    ├── final_cleaned_data.csv                          # ML-ready (reduced)
    └── complete_dataset_no_drops.csv                   # Complete (all features) ⭐
```

## Recommendation

**For most analysis tasks, especially Tableau dashboards and comprehensive analytics, use the `complete_dataset_no_drops.csv` file as it provides the best balance of completeness, usability, and compatibility.**


