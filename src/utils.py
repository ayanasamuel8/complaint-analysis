import pandas as pd
import numpy as np
import sys
sys.path.append('../src')
def load_data(file_path: str):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        print(file_path)
        data = pd.read_csv(file_path)
        print(f"✅ Loaded {len(data)} rows from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(data: pd.DataFrame, file_path: str):
    """
    Save DataFrame to a CSV file.
    
    Args:
        data (pd.DataFrame): Data to save.
        file_path (str): Path to save the CSV file.
        
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        data.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False