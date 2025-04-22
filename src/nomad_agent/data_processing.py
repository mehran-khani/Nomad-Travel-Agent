"""Data loading and processing functions for the Nomad Travel Agent."""

import pandas as pd
from typing import List, Tuple, Optional
import requests
from . import config

def load_and_process_wikivoyage_data(url: str = config.WIKIVOYAGE_CSV_URL) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Loads, cleans, and preprocesses Wikivoyage data."""
    print(f"Loading Wikivoyage data from {url}...")
    df_raw = None
    try:
        # Handle DtypeWarning by explicitly setting dtypes
        df_raw = pd.read_csv(url, low_memory=False)
        print(f"Successfully loaded data. Shape: {df_raw.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, []

    print("Starting Data Cleaning and Preprocessing...")
    # Select and rename relevant columns
    relevant_columns = {
        'article': 'city',
        'title': 'name',
        'type': 'poi_type',
        'description': 'description',
        'address': 'address',
        'alt': 'alt'
    }

    # Check if all keys exist before selecting
    existing_cols = [col for col in relevant_columns.keys() if col in df_raw.columns]
    df_processed = df_raw[existing_cols].rename(columns=relevant_columns)

    # Fill missing values and convert to string
    text_cols_to_fill = ['name', 'poi_type', 'description', 'address', 'alt', 'city']
    for col in text_cols_to_fill:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna("").astype(str)
        else:
            df_processed[col] = ""

    # Create content for RAG
    required_for_rag = ['name', 'poi_type', 'description', 'address', 'alt']
    df_processed['content_for_rag'] = ""
    for col in required_for_rag:
        if col in df_processed.columns:
            df_processed['content_for_rag'] += f"{col.capitalize()}: " + df_processed[col] + "; "

    # Clean up the RAG content
    df_processed['content_for_rag'] = df_processed['content_for_rag'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Filter empty rows
    if 'city' in df_processed.columns:
        df_processed = df_processed[df_processed['city'] != ""]
    if 'name' in df_processed.columns:
        df_processed = df_processed[df_processed['name'] != ""]

    print(f"Data cleaned. New shape: {df_processed.shape}")

    # Extract unique cities
    unique_cities = []
    if 'city' in df_processed.columns:
        unique_cities = df_processed['city'].unique().tolist()
        print(f"Found {len(unique_cities)} unique cities.")
    else:
        print("Warning: 'city' column not found, cannot extract unique cities.")

    return df_processed, unique_cities 