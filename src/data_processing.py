import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_KEYWORDS = [
    "credit card",
    "personal loan",
    "buy now, pay later",
    "savings account",
    "money transfer"
]

# ---------- Utility Functions ----------

def duplicate_handling(df: pd.DataFrame) -> pd.DataFrame:
    initial = df.shape[0]
    df = df.drop_duplicates()
    print(f"✅ Removed {initial - df.shape[0]} duplicate rows.")
    return df

def missing_value_handling(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df[df["Consumer complaint narrative"].notna()]
    after = df.shape[0]
    print(f"✅ Dropped {before - after} rows with missing narratives.")
    return df

def date_time_conversion(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [col for col in df.columns if "date" in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def outlier_handling(df: pd.DataFrame, column: str, max_words: int = 3000) -> pd.DataFrame:
    df["NarrativeLength"] = df[column].apply(lambda x: len(x.split()))
    before = df.shape[0]
    df = df[df["NarrativeLength"] < max_words]
    print(f"✅ Removed {before - df.shape[0]} extreme outlier narratives (>{max_words} words).")
    return df.drop(columns=["NarrativeLength"])

def categorical_data_standardizing(df: pd.DataFrame) -> pd.DataFrame:
    if 'Product' in df.columns:
        df['Product'] = df['Product'].str.lower().str.strip()
    return df

def numerical_data_standardizing(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.sign(x) * np.log1p(abs(x)) if x > 0 else 0)
    return df

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def product_filter(product: str) -> bool:
    if pd.isna(product):
        return False
    product = product.lower()
    return any(keyword in product for keyword in TARGET_KEYWORDS)

# ---------- EDA & Processing Pipeline ----------

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:

    df = duplicate_handling(df)
    df = missing_value_handling(df)
    df = date_time_conversion(df)
    df = categorical_data_standardizing(df)

    print("🔍 Filtering target products...")
    df = df[df["Product"].apply(product_filter)]

    print("🧹 Cleaning narratives...")
    df["Cleaned_Narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    print("📊 Handling outliers...")
    df = outlier_handling(df, "Cleaned_Narrative", max_words=3000)

    return df

# ---------- EDA Visualization ----------

def generate_visuals(df: pd.DataFrame):
    print("\n📈 Generating EDA visuals...")
    # Product distribution
    product_counts = df["Product"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df["Product"], order=product_counts.index)
    plt.title("Complaint Count per Product")
    plt.xlabel("Count")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.show()

    # Narrative length
    df["NarrativeLength"] = df["Cleaned_Narrative"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(df["NarrativeLength"], bins=50, kde=True)
    plt.title("Distribution of Cleaned Narrative Lengths")
    plt.xlabel("Word Count")
    plt.tight_layout()
    plt.show()