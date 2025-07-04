import pandas as pd
import numpy as np

def duplicate_handling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicates in the DataFrame by removing them.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    return df

def missing_value_handling(df: pd.DataFrame) -> pd.DataFrame:
    ...

def date_time_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types of columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with converted data types.
    """
    # Example conversion
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def outlier_handling(df: pd.DataFrame) -> pd.DataFrame:
    ...

def categorical_data_standardizing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical data in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with standardized categorical data.
    """
    # Example standardization
    if 'category' in df.columns:
        df['category'] = df['category'].str.lower().str.strip()
    return df

def numerical_data_standardizing(df:pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Standardize numerical data in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of numerical columns to standardize.
        
    Returns:
        pd.DataFrame: DataFrame with standardized numerical data.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: np.sign(x) * np.log1p(abs(x)) if x > 0 else 0)
    return df