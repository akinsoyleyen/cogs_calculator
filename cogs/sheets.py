"""Google Sheets matrix push for landed-cost calculator."""
from datetime import datetime
from typing import Optional

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

from cogs.calculator import compute_landed_cost

PALLET_COUNTS = (2, 4, 6, 10)
WORKSHEET_TITLE = "COGS Matrix"
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


@st.cache_resource(show_spinner=False)
def get_gspread_client() -> gspread.Client:
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=_SCOPES)
    return gspread.authorize(creds)


def build_air_matrix(
    *,
    base_inputs: dict,
    dfs: dict,
    pallet_counts: tuple[int, ...] = PALLET_COUNTS,
) -> pd.DataFrame:
    """Recompute final_cost_per_box_usd for every (destination, pallet count) pair.

    base_inputs: keys from the calculator signature, minus num_pallets, quantity_boxes,
    destination, shipment_type (we override those). It includes everything else the
    calculator needs.

    dfs: {product_weights_df, product_recipe_df, components_df, pallets_df, fixed_df,
          air_rates_df, product_packing_df}.
    """
    product_packing_df = dfs["product_packing_df"]
    air_rates_df = dfs["air_rates_df"]

    if air_rates_df is None or air_rates_df.empty:
        raise ValueError("Air rates CSV is missing or empty — cannot build matrix.")

    product_id = str(base_inputs["selected_product"])
    packing = product_packing_df[product_packing_df["ProductID"] == product_id]
    if packing.empty or not pd.notna(packing["BoxesPerPallet"].iloc[0]):
        raise ValueError(
            f"Boxes/Pallet not configured for '{product_id}' in product_packing.csv — "
            "matrix push needs an auto-calculable quantity."
        )
    boxes_per_pallet = int(packing["BoxesPerPallet"].iloc[0])
    if boxes_per_pallet <= 0:
        raise ValueError(f"Boxes/Pallet for '{product_id}' must be > 0.")

    destinations = sorted(air_rates_df["Destination"].unique().tolist())

    matrix = pd.DataFrame(
        index=destinations,
        columns=list(pallet_counts),
        dtype=float,
    )

    for dest in destinations:
        for n in pallet_counts:
            res = compute_landed_cost(
                quantity_boxes=n * boxes_per_pallet,
                num_pallets=n,
                shipment_type="Air",
                destination=dest,
                manual_logistics_cost_usd=0.0,
                product_weights_df=dfs["product_weights_df"],
                product_recipe_df=dfs["product_recipe_df"],
                components_df=dfs["components_df"],
                pallets_df=dfs["pallets_df"],
                fixed_df=dfs["fixed_df"],
                air_rates_df=air_rates_df,
                **{k: v for k, v in base_inputs.items()
                   if k not in {"shipment_type", "destination",
                                "manual_logistics_cost_usd", "num_pallets",
                                "quantity_boxes"}},
            )
            matrix.at[dest, n] = round(res.final_cost_per_box_usd, 4)

    matrix.index.name = "Destination"
    return matrix


def push_matrix(
    client: gspread.Client,
    sheet_url: str,
    *,
    product: str,
    matrix_df: pd.DataFrame,
    metadata: dict,
    worksheet_title: str = WORKSHEET_TITLE,
) -> str:
    """Append a timestamped metadata block + matrix to the sheet. Returns sheet URL."""
    sh = client.open_by_url(sheet_url)
    try:
        ws = sh.worksheet(worksheet_title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_title, rows=200, cols=12)
        ws.append_row(["Cost-per-box (USD) matrix — appended runs grow downwards."])

    rows: list[list] = []
    rows.append([])  # blank separator from previous block

    # Metadata header — six labelled rows.
    for k in ("Date", "Product", "Raw Cost / KG (USD)",
              "Rebate %", "Fixed Cost Mode", "Reporting Currency"):
        rows.append([k, metadata.get(k, "")])

    rows.append([])  # blank between metadata and matrix

    # Matrix header + body
    header = ["Destination"] + [f"{int(c)} pallets" for c in matrix_df.columns]
    rows.append(header)
    for dest, row in matrix_df.iterrows():
        rows.append([dest, *[float(v) for v in row.tolist()]])

    rows.append([])
    rows.append([])

    ws.append_rows(rows, value_input_option="USER_ENTERED")
    return _worksheet_url(sh, ws)


def _worksheet_url(sh: gspread.Spreadsheet, ws: gspread.Worksheet) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sh.id}/edit#gid={ws.id}"
