"""
Exception handling module for the data science pipeline.

This module provides custom exception classes and error handling utilities
for robust error management across all pipeline stages.
"""

from .exception import (
    PipelineException,
    DataLoadingException,
    DataValidationException,
    PreprocessingException,
    FeatureEngineeringException,
    DataSavingException,
    ConfigurationException,
    handle_exception,
    pipeline_error_handler,
    validate_file_path,
    validate_dataframe
)

__all__ = [
    "PipelineException",
    "DataLoadingException", 
    "DataValidationException",
    "PreprocessingException",
    "FeatureEngineeringException",
    "DataSavingException",
    "ConfigurationException",
    "handle_exception",
    "pipeline_error_handler",
    "validate_file_path",
    "validate_dataframe"
]

