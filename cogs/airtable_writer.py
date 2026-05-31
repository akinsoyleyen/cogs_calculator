"""Append the air-freight pricing matrix to an Airtable "big table".

Mirrors cogs/github_writer.py: a thin, secrets-driven module triggered by a
button. `build_ledger_rows` is a pure transform (matrix -> Airtable records)
so it is unit-testable without network or Streamlit.

Requires three secrets in .streamlit/secrets.toml:
- airtable_token    (personal access token, data.records:write on the base)
- airtable_base_id  (e.g. "appXXXXXXXXXXXXXX")
- airtable_table    (e.g. "COGS Ledger")
"""
from urllib.parse import quote

import requests
import streamlit as st

_API = "https://api.airtable.com/v0"
_BATCH = 10  # Airtable accepts at most 10 records per create request.


def build_ledger_rows(
    matrix_df,
    *,
    product,
    price_basis,
    target_profit_percent,
    raw_cost_per_kg,
    rebate_percentage,
    fixed_cost_mode,
    boxes_per_pallet,
    logged_at_iso,
    batch_id,
):
    """Turn an active pricing matrix into one Airtable record dict per destination.

    matrix_df: index = destination names, columns = pallet counts (2,4,6,10),
    values = price per box (USD). Returns a list of plain dicts whose keys match
    the "COGS Ledger" field names.
    """
    rows = []
    for dest, row in matrix_df.iterrows():
        record = {
            "Logged At": logged_at_iso,
            "Batch ID": batch_id,
            "Product": product,
            "Destination": str(dest),
            "Price basis": price_basis,
            "Target profit %": target_profit_percent,
            "Raw $/kg": raw_cost_per_kg,
            "Rebate %": rebate_percentage,
            "Fixed-cost mode": fixed_cost_mode,
            "Boxes/pallet": boxes_per_pallet,
        }
        for col in matrix_df.columns:
            record[f"{int(col)} pallets"] = round(float(row[col]), 2)
        rows.append(record)
    return rows
