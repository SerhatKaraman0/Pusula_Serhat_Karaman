"""
Exploratory Data Analysis (EDA) Pipeline

This module performs comprehensive exploratory data analysis on the feature-engineered medical data.
Target variable: TEDAVISURESI_SEANS_SAYI (Treatment Duration in Sessions)

EDA Techniques Included:
- Basic statistical analysis and data profiling
- Distribution analysis and visualization
- Correlation analysis and heatmaps
- Target variable analysis and relationships
- Categorical variable analysis
- Outlier detection and analysis
- Feature importance analysis
- Missing value patterns
- Advanced visualizations and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
import os
import logging
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# Import project utilities
from case_src.utils.analyze_utils.utils import AnalyzeDataFrame
from case_src.utils.visualize_utils.utils import VisualizeDataFrame
from case_src.logging.logger import LOG_FILE_PATH
from case_src.exception import (
    DataLoadingException,
    DataValidationException,
    pipeline_error_handler,
    validate_file_path,
    validate_dataframe,
    handle_exception
)

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EDAException(Exception):
    """Custom exception for EDA pipeline errors."""
    pass


class EDAPipeline:
    """
    Comprehensive Exploratory Data Analysis Pipeline for medical treatment data.
    
    Analyzes feature-engineered data with TEDAVISURESI_SEANS_SAYI as target variable.
    Creates organized visualizations and statistical analyses.
    """
    
    def __init__(self):
        """Initialize the EDA pipeline with utilities and configuration."""
        self.analyze_df = AnalyzeDataFrame()
        self.visualize_df = VisualizeDataFrame()
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("EDAPipeline initialized")
        
        # Target variable
        self.target_column = "TEDAVISURESI_SEANS_SAYI"
        
        # Ensure base EDA directory exists
        self._ensure_base_directory()
        
        # Configure matplotlib for high-quality plots
        self._configure_matplotlib()
    
    def _ensure_base_directory(self):
        """Ensure the base EDA results directory exists."""
        try:
            base_eda_dir = "/Users/user/Desktop/Projects/ds_case_pusula/data/EDA_results"
            if not os.path.exists(base_eda_dir):
                os.makedirs(base_eda_dir, exist_ok=True)
                self.logger.info(f"Created base EDA directory: {base_eda_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to create base EDA directory: {str(e)}")
    
    def _configure_matplotlib(self):
        """Configure matplotlib settings for high-quality plots."""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
        
    def create_output_directories(self, base_path: str) -> Dict[str, str]:
        """
        Create organized directory structure for EDA results.
        
        Parameters:
        -----------
        base_path : str
            Base directory path for EDA results
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping analysis types to their directory paths
        """
        directories = {
            'basic_stats': os.path.join(base_path, '01_basic_statistics'),
            'distributions': os.path.join(base_path, '02_distributions'),
            'correlations': os.path.join(base_path, '03_correlations'),
            'target_analysis': os.path.join(base_path, '04_target_analysis'),
            'categorical_analysis': os.path.join(base_path, '05_categorical_analysis'),
            'outliers': os.path.join(base_path, '06_outlier_analysis'),
            'feature_importance': os.path.join(base_path, '07_feature_importance'),
            'advanced_viz': os.path.join(base_path, '08_advanced_visualizations'),
            'summary_reports': os.path.join(base_path, '09_summary_reports')
        }
        
        # Create directories
        for dir_name, dir_path in directories.items():
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        return directories
    
    @pipeline_error_handler("eda")
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load feature-engineered data for EDA analysis.
        
        Parameters:
        -----------
        file_path : str
            Path to the feature-engineered CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe for analysis
        """
        try:
            self.logger.info(f"Loading data from: {file_path}")
            
            validate_file_path(file_path, must_exist=True)
            df = pd.read_csv(file_path)
            validate_dataframe(df, min_rows=1, min_cols=1)
            
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Validate target column exists
            if self.target_column not in df.columns:
                raise EDAException(f"Target column '{self.target_column}' not found in data")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise DataLoadingException(file_path, e)
    
    def basic_statistical_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Perform basic statistical analysis and data profiling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save results
        """
        self.logger.info("Performing basic statistical analysis...")
        
        # Basic info
        info_dict = {
            'Dataset Shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Data Types': dict(df.dtypes.value_counts())
        }
        
        # Save basic info
        with open(os.path.join(output_dir, 'dataset_overview.txt'), 'w') as f:
            f.write("DATASET OVERVIEW\n")
            f.write("=" * 50 + "\n\n")
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")
        
        # Descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        desc_stats = df[numeric_cols].describe()
        desc_stats.to_csv(os.path.join(output_dir, 'descriptive_statistics_numeric.csv'))
        
        # Categorical column analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        cat_summary = {}
        for col in categorical_cols:
            cat_summary[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            }
        
        cat_df = pd.DataFrame(cat_summary).T
        cat_df.to_csv(os.path.join(output_dir, 'categorical_summary.csv'))
        
        # Missing values analysis
        missing_analysis = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
        }).sort_values('Missing_Count', ascending=False)
        
        missing_analysis.to_csv(os.path.join(output_dir, 'missing_values_analysis.csv'), index=False)
        
        self.logger.info("Basic statistical analysis completed")
    
    def distribution_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Analyze and visualize distributions of all variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save visualizations
        """
        self.logger.info("Performing distribution analysis...")
        
        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create distribution plots for numeric variables
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
                # Add statistics
                mean_val = df[col].mean()
                median_val = df[col].median()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
                axes[i].legend()
        
        # Remove empty subplots
        for i in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'numeric_distributions.png'))
        plt.close()
        
        # Target variable detailed distribution
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(df[self.target_column], bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {self.target_column}')
        plt.xlabel(self.target_column)
        plt.ylabel('Frequency')
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(df[self.target_column])
        plt.title(f'Box Plot of {self.target_column}')
        plt.ylabel(self.target_column)
        
        # QQ plot
        plt.subplot(2, 2, 3)
        stats.probplot(df[self.target_column], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {self.target_column}')
        
        # Value counts
        plt.subplot(2, 2, 4)
        value_counts = df[self.target_column].value_counts().sort_index()
        plt.bar(value_counts.index, value_counts.values, alpha=0.7)
        plt.title(f'Value Counts of {self.target_column}')
        plt.xlabel(self.target_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_variable_distribution.png'))
        plt.close()
        
        # Categorical distributions (top categories only)
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols[:6]:  # Limit to first 6 categorical columns
            plt.figure(figsize=(16, 8))
            
            # Get top categories (limit based on number of unique values)
            value_counts = df[col].value_counts()
            n_categories = min(12, len(value_counts))  # Max 12 categories to avoid overcrowding
            top_categories = value_counts.head(n_categories)
            
            plt.subplot(1, 2, 1)
            # Create bar plot with better formatting
            bars = plt.bar(range(len(top_categories)), top_categories.values, alpha=0.7, color='skyblue')
            plt.title(f'Top {n_categories} Categories in {col}', fontsize=14, pad=20)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Set x-axis labels with rotation and better spacing
            plt.xticks(range(len(top_categories)), 
                      [label[:20] + '...' if len(str(label)) > 20 else str(label) for label in top_categories.index], 
                      rotation=45, ha='right', fontsize=10)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.subplot(1, 2, 2)
            # Create pie chart with better label handling
            def autopct_format(pct):
                return f'{pct:.1f}%' if pct > 3 else ''  # Only show percentage if > 3%
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
            wedges, texts, autotexts = plt.pie(top_categories.values, 
                                              labels=[label[:15] + '...' if len(str(label)) > 15 else str(label) 
                                                     for label in top_categories.index],
                                              autopct=autopct_format,
                                              colors=colors,
                                              startangle=90)
            
            # Improve text formatting
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            for text in texts:
                text.set_fontsize(9)
            
            plt.title(f'Distribution of {col} (Top {n_categories})', fontsize=14, pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'categorical_distribution_{col}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Distribution analysis completed")
    
    def correlation_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Perform correlation analysis and create correlation matrices.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save correlation matrices
        """
        self.logger.info("Performing correlation analysis...")
        
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols]
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Create correlation heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Pearson correlation heatmap
        mask1 = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask1, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax1, fmt='.2f')
        ax1.set_title('Pearson Correlation Matrix')
        
        # Spearman correlation heatmap
        mask2 = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask2, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax2, fmt='.2f')
        ax2.set_title('Spearman Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrices.png'))
        plt.close()
        
        # Target variable correlations
        target_corr_pearson = pearson_corr[self.target_column].drop(self.target_column).sort_values(key=abs, ascending=False)
        target_corr_spearman = spearman_corr[self.target_column].drop(self.target_column).sort_values(key=abs, ascending=False)
        
        # Plot target correlations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top 15 Pearson correlations with target
        top_pearson = target_corr_pearson.head(15)
        top_pearson.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title(f'Top 15 Pearson Correlations with {self.target_column}')
        ax1.set_xlabel('Correlation Coefficient')
        
        # Top 15 Spearman correlations with target
        top_spearman = target_corr_spearman.head(15)
        top_spearman.plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_title(f'Top 15 Spearman Correlations with {self.target_column}')
        ax2.set_xlabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_correlations.png'))
        plt.close()
        
        # Save correlation data
        pearson_corr.to_csv(os.path.join(output_dir, 'pearson_correlation_matrix.csv'))
        spearman_corr.to_csv(os.path.join(output_dir, 'spearman_correlation_matrix.csv'))
        target_corr_pearson.to_csv(os.path.join(output_dir, 'target_pearson_correlations.csv'))
        target_corr_spearman.to_csv(os.path.join(output_dir, 'target_spearman_correlations.csv'))
        
        self.logger.info("Correlation analysis completed")
    
    def target_variable_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Comprehensive analysis of the target variable and its relationships.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save target analysis results
        """
        self.logger.info("Performing target variable analysis...")
        
        target_data = df[self.target_column]
        
        # Statistical summary
        target_stats = {
            'Count': len(target_data),
            'Mean': target_data.mean(),
            'Median': target_data.median(),
            'Mode': target_data.mode().iloc[0],
            'Standard Deviation': target_data.std(),
            'Variance': target_data.var(),
            'Skewness': target_data.skew(),
            'Kurtosis': target_data.kurtosis(),
            'Min': target_data.min(),
            'Max': target_data.max(),
            'Range': target_data.max() - target_data.min(),
            'IQR': target_data.quantile(0.75) - target_data.quantile(0.25),
            'Unique Values': target_data.nunique()
        }
        
        # Save target statistics
        with open(os.path.join(output_dir, 'target_statistics.txt'), 'w') as f:
            f.write(f"TARGET VARIABLE ANALYSIS: {self.target_column}\n")
            f.write("=" * 60 + "\n\n")
            for key, value in target_stats.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # Target vs categorical variables
        categorical_cols = ['CINSIYET', 'KANGRUBU_TYPE', 'UYRUK', 'TEDAVI_KATEGORISI', 'YAS_LABELS']
        
        for col in categorical_cols:
            if col in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Box plot
                plt.subplot(1, 2, 1)
                df.boxplot(column=self.target_column, by=col, ax=plt.gca())
                plt.title(f'{self.target_column} by {col}')
                plt.suptitle('')  # Remove default title
                
                # Bar plot of means
                plt.subplot(1, 2, 2)
                means = df.groupby(col)[self.target_column].mean().sort_values(ascending=False)
                means.plot(kind='bar')
                plt.title(f'Average {self.target_column} by {col}')
                plt.xticks(rotation=45)
                plt.ylabel(f'Average {self.target_column}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'target_vs_{col}.png'))
                plt.close()
        
        # Target vs numeric variables (scatter plots)
        numeric_cols = ['YAS', 'TANI_SAYI', 'ALERJI_SAYI', 'KRONIKHASTALIK_SAYI', 'RISK_SKORU', 'HASTANO_COUNT']
        
        n_plots = len(numeric_cols)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if col in df.columns and i < len(axes):
                axes[i].scatter(df[col], df[self.target_column], alpha=0.6)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(self.target_column)
                axes[i].set_title(f'{self.target_column} vs {col}')
                
                # Add trend line
                z = np.polyfit(df[col].dropna(), df[self.target_column][df[col].notna()], 1)
                p = np.poly1d(z)
                axes[i].plot(df[col], p(df[col]), "r--", alpha=0.8)
                
                # Add correlation
                corr, _ = pearsonr(df[col].dropna(), df[self.target_column][df[col].notna()])
                axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Remove empty subplots
        for i in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_vs_numeric_variables.png'))
        plt.close()
        
        self.logger.info("Target variable analysis completed")
    
    def categorical_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Analyze categorical variables and their relationships.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save categorical analysis results
        """
        self.logger.info("Performing categorical variable analysis...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Chi-square test results
        chi_square_results = []
        
        for col in categorical_cols:
            if col != self.target_column:  # Skip if target is categorical
                try:
                    # Create contingency table
                    contingency = pd.crosstab(df[col], pd.cut(df[self.target_column], bins=5))
                    
                    # Perform chi-square test
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    chi_square_results.append({
                        'Variable': col,
                        'Chi2_Statistic': chi2,
                        'P_Value': p_value,
                        'Degrees_of_Freedom': dof,
                        'Significant': p_value < 0.05
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Chi-square test failed for {col}: {str(e)}")
        
        # Save chi-square results
        if chi_square_results:
            chi_df = pd.DataFrame(chi_square_results)
            chi_df.to_csv(os.path.join(output_dir, 'chi_square_tests.csv'), index=False)
        
        # Categorical variable relationships
        important_cats = ['CINSIYET', 'KANGRUBU_TYPE', 'TEDAVI_KATEGORISI', 'YAS_LABELS']
        
        for i, cat1 in enumerate(important_cats):
            for cat2 in important_cats[i+1:]:
                if cat1 in df.columns and cat2 in df.columns:
                    plt.figure(figsize=(10, 6))
                    
                    # Create crosstab
                    crosstab = pd.crosstab(df[cat1], df[cat2])
                    
                    # Heatmap
                    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Cross-tabulation: {cat1} vs {cat2}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'crosstab_{cat1}_vs_{cat2}.png'))
                    plt.close()
        
        self.logger.info("Categorical analysis completed")
    
    def outlier_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Detect and analyze outliers in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save outlier analysis results
        """
        self.logger.info("Performing outlier analysis...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = []
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            
            outlier_summary.append({
                'Column': col,
                'IQR_Outliers': len(iqr_outliers),
                'IQR_Percentage': (len(iqr_outliers) / len(data)) * 100,
                'Z_Score_Outliers': len(z_outliers),
                'Z_Score_Percentage': (len(z_outliers) / len(data)) * 100,
                'Lower_Bound_IQR': lower_bound,
                'Upper_Bound_IQR': upper_bound
            })
        
        # Save outlier summary
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df.to_csv(os.path.join(output_dir, 'outlier_summary.csv'), index=False)
        
        # Create individual box plots for better visibility
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            
            # Create box plot
            box_data = df[col].dropna()
            box_plot = plt.boxplot(box_data, patch_artist=True)
            
            # Customize box plot
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][0].set_alpha(0.7)
            
            # Add statistics
            q1 = box_data.quantile(0.25)
            q3 = box_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = box_data[(box_data < lower_bound) | (box_data > upper_bound)]
            outlier_count = len(outliers)
            
            plt.title(f'Box Plot: {col}\nOutliers: {outlier_count} ({outlier_count/len(box_data)*100:.1f}%)')
            plt.ylabel(col)
            plt.xlabel('Distribution')
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
            
            # Add statistical annotations
            stats_text = f'Q1: {q1:.2f}\nMedian: {box_data.median():.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}'
            plt.text(1.1, box_data.median(), stats_text, fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'boxplot_{col}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Also create a combined overview plot
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                box_data = df[col].dropna()
                box_plot = axes[i].boxplot(box_data, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightblue')
                box_plot['boxes'][0].set_alpha(0.7)
                
                axes[i].set_title(f'{col}')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Outlier Analysis - All Variables Overview', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'outlier_boxplots_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Outlier analysis completed")
    
    def feature_importance_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Analyze feature importance using various methods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save feature importance results
        """
        self.logger.info("Performing feature importance analysis...")
        
        # Prepare data for analysis
        # Encode categorical variables
        df_encoded = df.copy()
        le = LabelEncoder()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df_encoded.columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Remove target and ID columns
        feature_cols = [col for col in df_encoded.columns if col not in [self.target_column, 'HASTANO']]
        X = df_encoded[feature_cols]
        y = df_encoded[self.target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
        
        # Plot feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Random Forest importance
        top_rf = rf_importance.head(15)
        ax1.barh(range(len(top_rf)), top_rf['Importance'])
        ax1.set_yticks(range(len(top_rf)))
        ax1.set_yticklabels(top_rf['Feature'])
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Random Forest Feature Importance (Top 15)')
        ax1.invert_yaxis()
        
        # Mutual Information
        top_mi = mi_importance.head(15)
        ax2.barh(range(len(top_mi)), top_mi['Mutual_Information'])
        ax2.set_yticks(range(len(top_mi)))
        ax2.set_yticklabels(top_mi['Feature'])
        ax2.set_xlabel('Mutual Information Score')
        ax2.set_title('Mutual Information Feature Importance (Top 15)')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # Save importance scores
        rf_importance.to_csv(os.path.join(output_dir, 'random_forest_importance.csv'), index=False)
        mi_importance.to_csv(os.path.join(output_dir, 'mutual_information_importance.csv'), index=False)
        
        self.logger.info("Feature importance analysis completed")
    
    def advanced_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Create advanced visualizations and interactive plots.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save advanced visualizations
        """
        self.logger.info("Creating advanced visualizations...")
        
        # Pairplot for key numeric variables
        key_numeric = ['YAS', 'TANI_SAYI', 'KRONIKHASTALIK_SAYI', 'RISK_SKORU', self.target_column]
        available_numeric = [col for col in key_numeric if col in df.columns]
        
        if len(available_numeric) > 1:
            plt.figure(figsize=(12, 10))
            sns.pairplot(df[available_numeric], diag_kind='hist', plot_kws={'alpha': 0.6})
            plt.savefig(os.path.join(output_dir, 'pairplot_key_variables.png'))
            plt.close()
        
        # Violin plots for target vs categorical variables
        categorical_vars = ['CINSIYET', 'KANGRUBU_TYPE', 'TEDAVI_KATEGORISI']
        
        for cat_var in categorical_vars:
            if cat_var in df.columns:
                plt.figure(figsize=(10, 6))
                sns.violinplot(data=df, x=cat_var, y=self.target_column)
                plt.title(f'Distribution of {self.target_column} by {cat_var}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'violin_plot_{cat_var}.png'))
                plt.close()
        
        # Heatmap of missing values pattern
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Values Pattern')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'))
        plt.close()
        
        self.logger.info("Advanced visualizations completed")
    
    def log_transformation_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Analyze skewed variables and apply log transformations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save log transformation analysis
        """
        self.logger.info("Performing log transformation analysis for skewed variables...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewness_results = []
        
        # Create output subdirectory
        log_transform_dir = os.path.join(output_dir, 'log_transformations')
        os.makedirs(log_transform_dir, exist_ok=True)
        
        for col in numeric_cols:
            if col != self.target_column and df[col].min() > 0:  # Only positive values for log transform
                data = df[col].dropna()
                if len(data) > 0:
                    original_skew = data.skew()
                    
                    # Apply log transformation
                    log_data = np.log1p(data)  # log1p is log(1+x), handles zeros better
                    log_skew = log_data.skew()
                    
                    skewness_results.append({
                        'Variable': col,
                        'Original_Skewness': original_skew,
                        'Log_Transformed_Skewness': log_skew,
                        'Skewness_Reduction': abs(original_skew) - abs(log_skew),
                        'Highly_Skewed': abs(original_skew) > 1,
                        'Log_Transform_Beneficial': abs(log_skew) < abs(original_skew)
                    })
                    
                    # Create visualization for highly skewed variables
                    if abs(original_skew) > 1:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        
                        # Original distribution
                        axes[0, 0].hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                        axes[0, 0].set_title(f'Original {col}\nSkewness: {original_skew:.3f}')
                        axes[0, 0].set_xlabel(col)
                        axes[0, 0].set_ylabel('Frequency')
                        
                        # Log transformed distribution
                        axes[0, 1].hist(log_data, bins=30, alpha=0.7, color='green', edgecolor='black')
                        axes[0, 1].set_title(f'Log Transformed {col}\nSkewness: {log_skew:.3f}')
                        axes[0, 1].set_xlabel(f'log({col})')
                        axes[0, 1].set_ylabel('Frequency')
                        
                        # Q-Q plots
                        stats.probplot(data, dist="norm", plot=axes[1, 0])
                        axes[1, 0].set_title(f'Q-Q Plot: Original {col}')
                        
                        stats.probplot(log_data, dist="norm", plot=axes[1, 1])
                        axes[1, 1].set_title(f'Q-Q Plot: Log Transformed {col}')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(log_transform_dir, f'log_transform_{col}.png'), 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
        
        # Save skewness analysis results
        if skewness_results:
            skewness_df = pd.DataFrame(skewness_results)
            skewness_df.to_csv(os.path.join(output_dir, 'skewness_analysis.csv'), index=False)
            
            # Create summary plot of skewness reduction
            beneficial_transforms = skewness_df[skewness_df['Log_Transform_Beneficial'] == True]
            
            if len(beneficial_transforms) > 0:
                plt.figure(figsize=(12, 8))
                
                x_pos = np.arange(len(beneficial_transforms))
                width = 0.35
                
                plt.bar(x_pos - width/2, beneficial_transforms['Original_Skewness'], 
                       width, label='Original Skewness', alpha=0.7, color='red')
                plt.bar(x_pos + width/2, beneficial_transforms['Log_Transformed_Skewness'], 
                       width, label='Log Transformed Skewness', alpha=0.7, color='green')
                
                plt.xlabel('Variables')
                plt.ylabel('Skewness')
                plt.title('Skewness Reduction Through Log Transformation')
                plt.xticks(x_pos, beneficial_transforms['Variable'], rotation=45, ha='right')
                plt.legend()
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Highly Skewed Threshold')
                plt.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'skewness_reduction_summary.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        self.logger.info("Log transformation analysis completed")
    
    def outlier_investigation(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Investigate outliers in treatment duration and other key variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save outlier investigation results
        """
        self.logger.info("Investigating outliers in treatment duration...")
        
        # Create outlier investigation subdirectory
        outlier_invest_dir = os.path.join(output_dir, 'outlier_investigation')
        os.makedirs(outlier_invest_dir, exist_ok=True)
        
        # Focus on target variable outliers
        target_data = df[self.target_column].dropna()
        q1 = target_data.quantile(0.25)
        q3 = target_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        outlier_mask = (df[self.target_column] < lower_bound) | (df[self.target_column] > upper_bound)
        outliers_df = df[outlier_mask].copy()
        
        # Analyze outlier characteristics
        if len(outliers_df) > 0:
            # Save outlier records
            outliers_df.to_csv(os.path.join(outlier_invest_dir, 'treatment_duration_outliers.csv'), index=False)
            
            # Outlier analysis by categorical variables
            categorical_vars = ['CINSIYET', 'KANGRUBU_TYPE', 'TEDAVI_KATEGORISI', 'YAS_LABELS']
            
            outlier_analysis = {}
            for var in categorical_vars:
                if var in df.columns:
                    outlier_dist = outliers_df[var].value_counts()
                    normal_dist = df[~outlier_mask][var].value_counts()
                    
                    outlier_analysis[var] = {
                        'outlier_distribution': outlier_dist.to_dict(),
                        'normal_distribution': normal_dist.to_dict()
                    }
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Outliers distribution
                    outlier_dist.plot(kind='bar', ax=ax1, color='red', alpha=0.7)
                    ax1.set_title(f'Outliers Distribution by {var}')
                    ax1.set_xlabel(var)
                    ax1.set_ylabel('Count')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Normal vs Outlier comparison
                    comparison_data = pd.DataFrame({
                        'Normal': normal_dist,
                        'Outliers': outlier_dist
                    }).fillna(0)
                    
                    comparison_data.plot(kind='bar', ax=ax2, alpha=0.7)
                    ax2.set_title(f'Normal vs Outlier Distribution by {var}')
                    ax2.set_xlabel(var)
                    ax2.set_ylabel('Count')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(outlier_invest_dir, f'outlier_analysis_{var}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Treatment duration outlier patterns
            plt.figure(figsize=(14, 8))
            
            plt.subplot(2, 2, 1)
            plt.scatter(df.index[~outlier_mask], df[self.target_column][~outlier_mask], 
                       alpha=0.6, color='blue', label='Normal', s=20)
            plt.scatter(df.index[outlier_mask], df[self.target_column][outlier_mask], 
                       alpha=0.8, color='red', label='Outliers', s=30)
            plt.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7, label='Upper Bound')
            plt.axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7, label='Lower Bound')
            plt.xlabel('Record Index')
            plt.ylabel(self.target_column)
            plt.title('Treatment Duration Outliers Over Records')
            plt.legend()
            
            # Age vs Treatment Duration
            plt.subplot(2, 2, 2)
            plt.scatter(df['YAS'][~outlier_mask], df[self.target_column][~outlier_mask], 
                       alpha=0.6, color='blue', label='Normal', s=20)
            plt.scatter(df['YAS'][outlier_mask], df[self.target_column][outlier_mask], 
                       alpha=0.8, color='red', label='Outliers', s=30)
            plt.xlabel('Age')
            plt.ylabel(self.target_column)
            plt.title('Age vs Treatment Duration (Outliers Highlighted)')
            plt.legend()
            
            # HASTANO_COUNT vs Treatment Duration
            plt.subplot(2, 2, 3)
            if 'HASTANO_COUNT' in df.columns:
                plt.scatter(df['HASTANO_COUNT'][~outlier_mask], df[self.target_column][~outlier_mask], 
                           alpha=0.6, color='blue', label='Normal', s=20)
                plt.scatter(df['HASTANO_COUNT'][outlier_mask], df[self.target_column][outlier_mask], 
                           alpha=0.8, color='red', label='Outliers', s=30)
                plt.xlabel('Patient Visit Count')
                plt.ylabel(self.target_column)
                plt.title('Patient Visits vs Treatment Duration')
                plt.legend()
            
            # Risk Score vs Treatment Duration
            plt.subplot(2, 2, 4)
            if 'RISK_SKORU' in df.columns:
                plt.scatter(df['RISK_SKORU'][~outlier_mask], df[self.target_column][~outlier_mask], 
                           alpha=0.6, color='blue', label='Normal', s=20)
                plt.scatter(df['RISK_SKORU'][outlier_mask], df[self.target_column][outlier_mask], 
                           alpha=0.8, color='red', label='Outliers', s=30)
                plt.xlabel('Risk Score')
                plt.ylabel(self.target_column)
                plt.title('Risk Score vs Treatment Duration')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(outlier_invest_dir, 'outlier_patterns.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Outlier investigation completed")
    
    def patient_pattern_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Explore patient-specific patterns using HASTANO_COUNT.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save patient pattern analysis
        """
        self.logger.info("Exploring patient-specific patterns using HASTANO_COUNT...")
        
        if 'HASTANO_COUNT' not in df.columns:
            self.logger.warning("HASTANO_COUNT column not found, skipping patient pattern analysis")
            return
        
        # Create patient patterns subdirectory
        patient_dir = os.path.join(output_dir, 'patient_patterns')
        os.makedirs(patient_dir, exist_ok=True)
        
        # Patient visit frequency analysis
        visit_freq = df['HASTANO_COUNT'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        visit_freq.plot(kind='bar', alpha=0.7, color='skyblue')
        plt.title('Distribution of Patient Visit Frequencies')
        plt.xlabel('Number of Visits per Patient')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=45)
        
        # Treatment duration vs visit frequency
        plt.subplot(2, 2, 2)
        avg_treatment_by_visits = df.groupby('HASTANO_COUNT')[self.target_column].mean()
        avg_treatment_by_visits.plot(kind='line', marker='o', alpha=0.7, color='green')
        plt.title('Average Treatment Duration by Visit Frequency')
        plt.xlabel('Number of Visits per Patient')
        plt.ylabel(f'Average {self.target_column}')
        plt.grid(True, alpha=0.3)
        
        # Box plot: Treatment duration by visit frequency groups
        plt.subplot(2, 2, 3)
        visit_groups = pd.cut(df['HASTANO_COUNT'], bins=[0, 1, 5, 10, float('inf')], 
                             labels=['Single Visit', '2-5 Visits', '6-10 Visits', '10+ Visits'])
        df_temp = df.copy()
        df_temp['Visit_Group'] = visit_groups
        
        visit_group_data = [df_temp[df_temp['Visit_Group'] == group][self.target_column].dropna() 
                           for group in ['Single Visit', '2-5 Visits', '6-10 Visits', '10+ Visits']]
        
        plt.boxplot(visit_group_data, labels=['Single Visit', '2-5 Visits', '6-10 Visits', '10+ Visits'])
        plt.title('Treatment Duration by Visit Frequency Groups')
        plt.xlabel('Visit Frequency Group')
        plt.ylabel(self.target_column)
        plt.xticks(rotation=45)
        
        # Chronic disease burden vs visit frequency
        plt.subplot(2, 2, 4)
        if 'KRONIKHASTALIK_SAYI' in df.columns:
            chronic_by_visits = df.groupby('HASTANO_COUNT')['KRONIKHASTALIK_SAYI'].mean()
            chronic_by_visits.plot(kind='line', marker='s', alpha=0.7, color='red')
            plt.title('Chronic Disease Burden by Visit Frequency')
            plt.xlabel('Number of Visits per Patient')
            plt.ylabel('Average Number of Chronic Diseases')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(patient_dir, 'patient_visit_patterns.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # High-frequency patients analysis
        high_freq_threshold = df['HASTANO_COUNT'].quantile(0.9)  # Top 10% most frequent patients
        high_freq_patients = df[df['HASTANO_COUNT'] >= high_freq_threshold]
        
        if len(high_freq_patients) > 0:
            # Characteristics of high-frequency patients
            characteristics = {}
            categorical_vars = ['CINSIYET', 'KANGRUBU_TYPE', 'TEDAVI_KATEGORISI', 'YAS_LABELS']
            
            for var in categorical_vars:
                if var in df.columns:
                    high_freq_dist = high_freq_patients[var].value_counts(normalize=True)
                    all_patients_dist = df[var].value_counts(normalize=True)
                    characteristics[var] = {
                        'high_frequency': high_freq_dist.to_dict(),
                        'all_patients': all_patients_dist.to_dict()
                    }
            
            # Save high-frequency patient analysis
            high_freq_patients.to_csv(os.path.join(patient_dir, 'high_frequency_patients.csv'), index=False)
            
            # Visualization of high-frequency patient characteristics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, var in enumerate(categorical_vars[:4]):
                if var in df.columns and i < len(axes):
                    high_freq_dist = high_freq_patients[var].value_counts()
                    all_dist = df[var].value_counts()
                    
                    # Normalize for comparison
                    high_freq_norm = high_freq_dist / high_freq_dist.sum() * 100
                    all_norm = all_dist / all_dist.sum() * 100
                    
                    comparison_df = pd.DataFrame({
                        'High Frequency Patients': high_freq_norm,
                        'All Patients': all_norm
                    }).fillna(0)
                    
                    comparison_df.plot(kind='bar', ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'{var} Distribution Comparison')
                    axes[i].set_xlabel(var)
                    axes[i].set_ylabel('Percentage')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(patient_dir, 'high_frequency_patient_characteristics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Patient pattern analysis completed")
    
    def correlation_based_feature_engineering(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Suggest feature engineering based on correlation findings.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save correlation-based feature engineering suggestions
        """
        self.logger.info("Analyzing correlations for feature engineering opportunities...")
        
        # Create correlation feature engineering subdirectory
        corr_fe_dir = os.path.join(output_dir, 'correlation_feature_engineering')
        os.makedirs(corr_fe_dir, exist_ok=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols]
        
        # Calculate correlations
        correlation_matrix = numeric_df.corr()
        target_correlations = correlation_matrix[self.target_column].drop(self.target_column).abs().sort_values(ascending=False)
        
        # Feature engineering suggestions
        suggestions = []
        
        # 1. Ratio features from highly correlated variables
        high_corr_vars = target_correlations[target_correlations > 0.3].index.tolist()
        
        for i, var1 in enumerate(high_corr_vars):
            for var2 in high_corr_vars[i+1:]:
                if var1 != var2 and df[var2].min() > 0:  # Avoid division by zero
                    ratio_name = f"{var1}_to_{var2}_ratio"
                    ratio_values = df[var1] / df[var2]
                    ratio_corr = ratio_values.corr(df[self.target_column])
                    
                    suggestions.append({
                        'Feature_Type': 'Ratio',
                        'Feature_Name': ratio_name,
                        'Variables_Used': f"{var1}, {var2}",
                        'Target_Correlation': ratio_corr,
                        'Description': f"Ratio of {var1} to {var2}",
                        'Beneficial': abs(ratio_corr) > max(target_correlations[var1], target_correlations[var2])
                    })
        
        # 2. Interaction features
        for i, var1 in enumerate(high_corr_vars[:5]):  # Limit to top 5 to avoid explosion
            for var2 in high_corr_vars[i+1:6]:
                if var1 != var2:
                    interaction_name = f"{var1}_x_{var2}_interaction"
                    interaction_values = df[var1] * df[var2]
                    interaction_corr = interaction_values.corr(df[self.target_column])
                    
                    suggestions.append({
                        'Feature_Type': 'Interaction',
                        'Feature_Name': interaction_name,
                        'Variables_Used': f"{var1}, {var2}",
                        'Target_Correlation': interaction_corr,
                        'Description': f"Product of {var1} and {var2}",
                        'Beneficial': abs(interaction_corr) > max(target_correlations[var1], target_correlations[var2])
                    })
        
        # 3. Polynomial features for highly correlated variables
        for var in high_corr_vars[:3]:  # Top 3 variables
            if df[var].min() >= 0:  # Non-negative values for square
                poly_name = f"{var}_squared"
                poly_values = df[var] ** 2
                poly_corr = poly_values.corr(df[self.target_column])
                
                suggestions.append({
                    'Feature_Type': 'Polynomial',
                    'Feature_Name': poly_name,
                    'Variables_Used': var,
                    'Target_Correlation': poly_corr,
                    'Description': f"Square of {var}",
                    'Beneficial': abs(poly_corr) > target_correlations[var]
                })
        
        # Save suggestions
        if suggestions:
            suggestions_df = pd.DataFrame(suggestions)
            suggestions_df = suggestions_df.sort_values('Target_Correlation', key=abs, ascending=False)
            suggestions_df.to_csv(os.path.join(corr_fe_dir, 'feature_engineering_suggestions.csv'), index=False)
            
            # Visualize beneficial suggestions
            beneficial_suggestions = suggestions_df[suggestions_df['Beneficial'] == True]
            
            if len(beneficial_suggestions) > 0:
                plt.figure(figsize=(12, 8))
                
                feature_types = beneficial_suggestions['Feature_Type'].value_counts()
                colors = ['skyblue', 'lightgreen', 'lightcoral']
                
                plt.subplot(2, 1, 1)
                feature_types.plot(kind='bar', alpha=0.7, color=colors[:len(feature_types)])
                plt.title('Types of Beneficial Feature Engineering Suggestions')
                plt.xlabel('Feature Type')
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                
                plt.subplot(2, 1, 2)
                top_suggestions = beneficial_suggestions.head(10)
                plt.barh(range(len(top_suggestions)), top_suggestions['Target_Correlation'].abs(), 
                        alpha=0.7, color='green')
                plt.yticks(range(len(top_suggestions)), top_suggestions['Feature_Name'])
                plt.xlabel('Absolute Correlation with Target')
                plt.title('Top 10 Feature Engineering Suggestions by Target Correlation')
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                plt.savefig(os.path.join(corr_fe_dir, 'feature_engineering_suggestions.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        self.logger.info("Correlation-based feature engineering analysis completed")
    
    def generate_summary_report(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Generate comprehensive summary report of EDA findings.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save summary report
        """
        self.logger.info("Generating summary report...")
        
        # Collect key insights
        target_stats = df[self.target_column].describe()
        
        report = f"""
EXPLORATORY DATA ANALYSIS SUMMARY REPORT
========================================

Dataset Overview:
-----------------
• Total Records: {df.shape[0]:,}
• Total Features: {df.shape[1]:,}
• Missing Values: {df.isnull().sum().sum():,}
• Duplicate Rows: {df.duplicated().sum():,}

Target Variable Analysis ({self.target_column}):
-----------------------------------------------
• Mean: {target_stats['mean']:.2f}
• Median: {target_stats['50%']:.2f}
• Standard Deviation: {target_stats['std']:.2f}
• Range: {target_stats['min']:.0f} - {target_stats['max']:.0f}
• Unique Values: {df[self.target_column].nunique()}

Data Types Distribution:
-----------------------
"""
        
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            report += f"• {dtype}: {count} columns\n"
        
        report += f"""

Key Findings:
------------
• The target variable ranges from {target_stats['min']:.0f} to {target_stats['max']:.0f} sessions
• Average treatment duration is {target_stats['mean']:.1f} sessions
• Dataset contains {df['HASTANO'].nunique()} unique patients
• Most common treatment duration: {df[self.target_column].mode().iloc[0]} sessions

Recommendations:
---------------
• Consider log transformation for skewed variables
• Investigate outliers in treatment duration
• Explore patient-specific patterns using HASTANO_COUNT
• Consider feature engineering based on correlation findings
• Address missing values using domain knowledge

Analysis Complete: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open(os.path.join(output_dir, 'EDA_Summary_Report.txt'), 'w') as f:
            f.write(report)
        
        self.logger.info("Summary report generated")
    
    def run_complete_eda(self, input_file: str, output_base_dir: str) -> None:
        """
        Run the complete EDA pipeline.
        
        Parameters:
        -----------
        input_file : str
            Path to the feature-engineered CSV file
        output_base_dir : str
            Base directory for EDA results
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STARTING COMPREHENSIVE EDA PIPELINE")
            self.logger.info("="*50)
            
            # Create output directories
            output_dirs = self.create_output_directories(output_base_dir)
            
            # Load data
            df = self.load_data(input_file)
            
            # Run all EDA analyses
            self.basic_statistical_analysis(df, output_dirs['basic_stats'])
            self.distribution_analysis(df, output_dirs['distributions'])
            self.correlation_analysis(df, output_dirs['correlations'])
            self.target_variable_analysis(df, output_dirs['target_analysis'])
            self.categorical_analysis(df, output_dirs['categorical_analysis'])
            self.outlier_analysis(df, output_dirs['outliers'])
            self.feature_importance_analysis(df, output_dirs['feature_importance'])
            self.advanced_visualizations(df, output_dirs['advanced_viz'])
            
            # Run new advanced analyses
            self.log_transformation_analysis(df, output_dirs['advanced_viz'])
            self.outlier_investigation(df, output_dirs['outliers'])
            self.patient_pattern_analysis(df, output_dirs['advanced_viz'])
            self.correlation_based_feature_engineering(df, output_dirs['feature_importance'])
            
            self.generate_summary_report(df, output_dirs['summary_reports'])
            
            self.logger.info("="*50)
            self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Results saved to: {output_base_dir}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"EDA pipeline failed: {str(e)}")
            raise EDAException(f"Complete EDA pipeline failed: {str(e)}")


def main():
    """Main function to run the EDA pipeline."""
    
    # Set up logger
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        eda_pipeline = EDAPipeline()
        
        # Define file paths
        input_file = "/Users/user/Desktop/Projects/ds_case_pusula/data/feature_engineering/feature_engineering_data.csv"
        output_dir = "/Users/user/Desktop/Projects/ds_case_pusula/data/EDA_results"
        
        logger.info(f"Starting EDA pipeline with input: {input_file}")
        
        # Run complete EDA
        eda_pipeline.run_complete_eda(input_file, output_dir)
        
        logger.info("EDA pipeline completed successfully!")
        print("EDA pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        error_msg = f"EDA pipeline failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        raise


if __name__ == "__main__":
    main()
