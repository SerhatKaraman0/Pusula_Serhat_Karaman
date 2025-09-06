"""
Custom Exception Handler for Data Science Pipeline

This module provides custom exception classes and error handling utilities
for the data science pipeline project. It includes specific exceptions for
different pipeline stages and a comprehensive error handling system.
"""

import sys
import traceback
from typing import Optional, Any, Dict
import logging


class PipelineException(Exception):
    """
    Base exception class for all pipeline-related errors.
    
    This class provides enhanced error reporting with detailed context
    about where and why the error occurred in the pipeline.
    """
    
    def __init__(
        self, 
        error_message: str, 
        error_detail: Optional[Exception] = None,
        pipeline_stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pipeline exception.
        
        Parameters:
        -----------
        error_message : str
            Human-readable error message
        error_detail : Exception, optional
            Original exception that caused this error
        pipeline_stage : str, optional
            Stage of pipeline where error occurred (e.g., 'preprocessing', 'feature_engineering')
        context : dict, optional
            Additional context information about the error
        """
        super().__init__(error_message)
        
        self.error_message = error_message
        self.pipeline_stage = pipeline_stage or "unknown"
        self.context = context or {}
        
        # Extract detailed error information if original exception provided
        if error_detail is not None:
            self.error_detail = self._extract_error_detail(error_detail)
        else:
            self.error_detail = None
            
    def _extract_error_detail(self, error_detail: Exception) -> str:
        """
        Extract detailed error information from the original exception.
        
        Parameters:
        -----------
        error_detail : Exception
            Original exception
            
        Returns:
        --------
        str
            Formatted error detail string
        """
        exc_type, exc_obj, exc_tb = sys.exc_info()
        
        if exc_tb is not None:
            filename = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in file [{filename}] at line [{line_number}]: {str(error_detail)}"
        else:
            return str(error_detail)
    
    def __str__(self) -> str:
        """
        Return a formatted string representation of the exception.
        
        Returns:
        --------
        str
            Formatted error message with context
        """
        error_parts = [
            f"Pipeline Stage: {self.pipeline_stage}",
            f"Error: {self.error_message}"
        ]
        
        if self.error_detail:
            error_parts.append(f"Detail: {self.error_detail}")
            
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            error_parts.append(f"Context: {context_str}")
            
        return " | ".join(error_parts)


class DataLoadingException(PipelineException):
    """Exception raised when data loading fails."""
    
    def __init__(self, file_path: str, error_detail: Optional[Exception] = None):
        context = {"file_path": file_path}
        super().__init__(
            f"Failed to load data from file: {file_path}",
            error_detail=error_detail,
            pipeline_stage="data_loading",
            context=context
        )


class DataValidationException(PipelineException):
    """Exception raised when data validation fails."""
    
    def __init__(self, validation_error: str, data_shape: Optional[tuple] = None, error_detail: Optional[Exception] = None):
        context = {"data_shape": data_shape} if data_shape else {}
        super().__init__(
            f"Data validation failed: {validation_error}",
            error_detail=error_detail,
            pipeline_stage="data_validation",
            context=context
        )


class PreprocessingException(PipelineException):
    """Exception raised during preprocessing stage."""
    
    def __init__(self, preprocessing_step: str, error_detail: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Preprocessing failed at step: {preprocessing_step}",
            error_detail=error_detail,
            pipeline_stage="preprocessing",
            context=context
        )


class FeatureEngineeringException(PipelineException):
    """Exception raised during feature engineering stage."""
    
    def __init__(self, feature_step: str, error_detail: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Feature engineering failed at step: {feature_step}",
            error_detail=error_detail,
            pipeline_stage="feature_engineering",
            context=context
        )


class DataSavingException(PipelineException):
    """Exception raised when data saving fails."""
    
    def __init__(self, file_path: str, error_detail: Optional[Exception] = None):
        context = {"file_path": file_path}
        super().__init__(
            f"Failed to save data to file: {file_path}",
            error_detail=error_detail,
            pipeline_stage="data_saving",
            context=context
        )


class ConfigurationException(PipelineException):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, config_issue: str, error_detail: Optional[Exception] = None):
        super().__init__(
            f"Configuration error: {config_issue}",
            error_detail=error_detail,
            pipeline_stage="configuration",
            context={}
        )


def handle_exception(
    func_name: str,
    exception: Exception,
    logger: Optional[logging.Logger] = None,
    pipeline_stage: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> Optional[PipelineException]:
    """
    Handle exceptions in a standardized way across the pipeline.
    
    Parameters:
    -----------
    func_name : str
        Name of the function where the exception occurred
    exception : Exception
        The original exception
    logger : logging.Logger, optional
        Logger instance to log the error
    pipeline_stage : str, optional
        Stage of pipeline where error occurred
    context : dict, optional
        Additional context information
    reraise : bool, default=True
        Whether to reraise the exception after handling
        
    Returns:
    --------
    PipelineException or None
        Wrapped exception if not reraised, None otherwise
    """
    # Create appropriate exception type based on the original exception
    if isinstance(exception, FileNotFoundError):
        pipeline_exc = DataLoadingException(str(exception), exception)
    elif isinstance(exception, ValueError):
        pipeline_exc = DataValidationException(str(exception), error_detail=exception)
    elif isinstance(exception, KeyError):
        pipeline_exc = PreprocessingException(f"Missing key in {func_name}", exception, context)
    elif isinstance(exception, PipelineException):
        # Already a pipeline exception, just pass through
        pipeline_exc = exception
    else:
        # Generic pipeline exception
        pipeline_exc = PipelineException(
            f"Unexpected error in {func_name}: {str(exception)}",
            error_detail=exception,
            pipeline_stage=pipeline_stage,
            context=context
        )
    
    # Log the error if logger provided
    if logger:
        logger.error(f"Exception in {func_name}: {str(pipeline_exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Reraise or return the exception
    if reraise:
        raise pipeline_exc
    else:
        return pipeline_exc


def pipeline_error_handler(pipeline_stage: str, logger: Optional[logging.Logger] = None):
    """
    Decorator for handling pipeline errors in a standardized way.
    
    Parameters:
    -----------
    pipeline_stage : str
        Stage of pipeline (e.g., 'preprocessing', 'feature_engineering')
    logger : logging.Logger, optional
        Logger instance to use for error logging
        
    Returns:
    --------
    function
        Decorated function with error handling
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                handle_exception(
                    func_name=func.__name__,
                    exception=e,
                    logger=logger,
                    pipeline_stage=pipeline_stage,
                    context=context,
                    reraise=True
                )
        return wrapper
    return decorator


def validate_file_path(file_path: str, must_exist: bool = True) -> None:
    """
    Validate file path and raise appropriate exception if invalid.
    
    Parameters:
    -----------
    file_path : str
        Path to validate
    must_exist : bool, default=True
        Whether the file must exist
        
    Raises:
    -------
    DataLoadingException
        If file path is invalid or file doesn't exist when required
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise DataLoadingException("Invalid file path: empty or not a string")
    
    if must_exist and not os.path.exists(file_path):
        raise DataLoadingException(f"File does not exist: {file_path}")
    
    # Check if directory exists for saving
    if not must_exist:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise DataSavingException(f"Cannot create directory for file: {file_path}", e)


def validate_dataframe(df: Any, min_rows: int = 1, min_cols: int = 1, required_columns: Optional[list] = None) -> None:
    """
    Validate pandas DataFrame and raise appropriate exception if invalid.
    
    Parameters:
    -----------
    df : Any
        Object to validate as DataFrame
    min_rows : int, default=1
        Minimum number of rows required
    min_cols : int, default=1
        Minimum number of columns required
    required_columns : list, optional
        List of required column names
        
    Raises:
    -------
    DataValidationException
        If DataFrame validation fails
    """
    if not hasattr(df, 'shape'):
        raise DataValidationException("Object is not a pandas DataFrame")
    
    if df.shape[0] < min_rows:
        raise DataValidationException(f"DataFrame has insufficient rows: {df.shape[0]} < {min_rows}", df.shape)
    
    if df.shape[1] < min_cols:
        raise DataValidationException(f"DataFrame has insufficient columns: {df.shape[1]} < {min_cols}", df.shape)
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationException(f"Missing required columns: {missing_cols}", df.shape)


# Import os for file operations
import os

