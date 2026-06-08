"""CSV path constants and the load_csv() helper.

load_csv() returns (dataframe, error_message_or_None) — pure, Streamlit-free
in its happy path (uses st.warning only when a file is empty). Phase B1
removed the previous module-level errors-list mutation.
"""
import os

import pandas as pd
import streamlit as st


COMPONENTS_CSV = "components.csv"
RECIPE_CSV = "product_recipe.csv"
FIXED_CSV = "fixed_costs.csv"
WEIGHTS_CSV = "product_weights.csv"
AIR_RATES_CSV = "air_freight_rates.csv"
PALLETS_CSV = "pallets.csv"
PACKING_CSV = "product_packing.csv"
GROUPS_CSV = "product_groups.csv"

INTEREST_COST_ITEM_NAME = "Calc. Interest"


def load_csv(file_path, required_cols, numeric_cols=None, decimal_char='.', string_cols=None):
    """Load a CSV. Returns (dataframe, error_message_or_None)."""
    if not os.path.exists(file_path):
        return None, f"File missing: '{os.path.abspath(file_path)}'"
    try:
        df = pd.read_csv(file_path, decimal=decimal_char)
        missing = [c for c in required_cols if c not in df.columns]
        if missing: raise ValueError(f"'{file_path}' missing required columns: {missing}")

        if df.empty and required_cols:
            st.warning(f"Warning: File '{file_path}' loaded as empty. Check content and headers.")

        if numeric_cols:
            for col in numeric_cols:
                if col not in df.columns: continue

                is_interest_cost_row = False
                if file_path.endswith(FIXED_CSV) and col == 'MonthlyCost' and 'CostItem' in df.columns:
                     df['CostItem'] = df['CostItem'].astype(str).fillna('')
                     is_interest_cost_row = df['CostItem'].str.strip().str.lower() == INTEREST_COST_ITEM_NAME.lower()

                df[col] = pd.to_numeric(df[col], errors='coerce')

                has_nan = df[col].isnull()
                if isinstance(is_interest_cost_row, pd.Series):
                     rows_to_check_for_nan = has_nan & (~is_interest_cost_row)
                else:
                     rows_to_check_for_nan = has_nan

                if rows_to_check_for_nan.any():
                    error_indices = df.index[rows_to_check_for_nan].tolist()
                    raise ValueError(f"Column '{col}' in '{file_path}' contains non-numeric values or blanks at row indices (starting 0): {error_indices[:5]}...")

        if string_cols:
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('').str.strip()

        return df, None
    except Exception as e:
        return None, f"Error processing '{file_path}': {e}"
