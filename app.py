# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import requests
import os
from datetime import datetime
import plotly.express as px
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import base64
import numpy as np  # Add this import at the top for np.unique
import plotly.graph_objects as go

# --- Constants ---
FALLBACK_TRY_TO_USD = 0.025  # Fallback rate if API fails
FALLBACK_USD_TO_EUR = 0.85   # Fallback USD to EUR rate if API fails
INTEREST_RATE = 0.05  # 5% interest rate
INTEREST_COST_ITEM_NAME = "Calc. Interest"  # Name for calculated interest cost

# --- API URLs ---
FRANKFURTER_API_URL = "https://api.frankfurter.app/latest?from=TRY&to=USD"
FRANKFURTER_USD_EUR_API_URL = "https://api.frankfurter.app/latest?from=USD&to=EUR"

# --- Global variables ---
usd_to_eur_rate = FALLBACK_USD_TO_EUR  # Initialize with fallback rate

# --- File Paths ---
COMPONENTS_CSV = "components.csv"
RECIPE_CSV = "product_recipe.csv"
FIXED_CSV = "fixed_costs.csv"
WEIGHTS_CSV = "product_weights.csv"
AIR_RATES_CSV = "air_freight_rates.csv"
PALLETS_CSV = "pallets.csv"
PACKING_CSV = "product_packing.csv"

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="Cost Calculator")

# --- Add Logo to Sidebar Here ---
with st.sidebar:
    try:
        # Adjust the width as needed
        st.image("assets/Logo.png", width=100)
    except FileNotFoundError:
        st.error("Logo file not found. Make sure 'your_logo.png' is in the correct path.")
    except Exception as e:
        st.error(f"An error occurred loading the logo: {e}")

st.title("Product Cost & Logistics Calculator")

# --- Exchange Rate Functions ---
@st.cache_data(ttl=3600)
def get_usd_to_eur_rate():
    try:
        response = requests.get(FRANKFURTER_USD_EUR_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data.get('rates', {}).get('EUR')
        date = data.get('date', 'N/A')
        if rate:
            st.session_state['usd_eur_rate_source'] = f"API ({date})"
            st.session_state['usd_eur_rate_date'] = date
            return float(rate)
        else:
            st.session_state['usd_eur_rate_source'] = "API Error (Rate Missing)"
            st.session_state['usd_eur_rate_date'] = None
            return None
    except Exception as e:
        st.session_state['usd_eur_rate_source'] = f"API Failed ({type(e).__name__})"
        st.session_state['usd_eur_rate_date'] = None
        return None

# --- Helper function to format costs with both USD and Euro ---
def format_cost_usd_eur(usd_amount):
    """Format a USD amount to show both USD and Euro values"""
    if usd_amount == 0:
        return "$0.00 / €0.00"
    
    # Get the current USD to EUR rate from sidebar
    global usd_to_eur_rate
    if usd_to_eur_rate is None:
        usd_to_eur_rate = FALLBACK_USD_TO_EUR
    
    eur_amount = usd_amount * usd_to_eur_rate
    return f"${usd_amount:,.2f} / €{eur_amount:,.2f}"

def format_cost_usd_only(usd_amount):
    """Format a USD amount to show only USD values"""
    if usd_amount == 0:
        return "$0.00"
    return f"${usd_amount:,.2f}"

def format_cost_eur_only(usd_amount):
    """Format a USD amount to show only EUR values"""
    if usd_amount == 0:
        return "€0.00"
    
    # Get the current USD to EUR rate from sidebar
    global usd_to_eur_rate
    if usd_to_eur_rate is None:
        usd_to_eur_rate = FALLBACK_USD_TO_EUR
    
    eur_amount = usd_amount * usd_to_eur_rate
    return f"€{eur_amount:,.2f}"

def format_cost_by_mode(usd_amount, mode):
    """Format cost based on display mode"""
    if mode == "EUR Only":
        return format_cost_eur_only(usd_amount)
    else:  # USD Only (default)
        return format_cost_usd_only(usd_amount)

def calculate_profit_margins(cost_per_box, sales_price_per_box):
    """Calculate profit margins for given cost and sales price"""
    if sales_price_per_box <= 0:
        return {"profit_per_box": 0, "profit_margin_percent": 0, "roi_percent": 0}
    
    profit_per_box = sales_price_per_box - cost_per_box
    profit_margin_percent = (profit_per_box / sales_price_per_box) * 100 if sales_price_per_box > 0 else 0
    roi_percent = (profit_per_box / cost_per_box) * 100 if cost_per_box > 0 else 0
    
    return {
        "profit_per_box": profit_per_box,
        "profit_margin_percent": profit_margin_percent,
        "roi_percent": roi_percent
    }

# --- Export Helper Functions ---
def create_csv_export(summary_data_dict, profit_data, calculation_details):
    """Create CSV export of calculation results"""
    export_data = []
    
    # Add calculation details
    export_data.append(["Calculation Details"])
    export_data.append(["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    export_data.append(["Product", calculation_details.get('product', 'N/A')])
    export_data.append(["Quantity", calculation_details.get('quantity', 'N/A')])
    export_data.append(["Shipment Type", calculation_details.get('shipment_type', 'N/A')])
    export_data.append(["Fixed Cost Mode", calculation_details.get('fixed_cost_mode', 'N/A')])
    export_data.append([])
    
    # Add cost breakdown
    export_data.append(["Cost Breakdown"])
    for key, value in summary_data_dict.items():
        if value is not None:
            export_data.append([key, value])
    export_data.append([])
    
    # Add profit analysis if available
    if profit_data:
        export_data.append(["Profit Analysis"])
        export_data.append(["Cost per Box (USD)", f"${profit_data.get('final_cost_per_box_usd', 0):,.2f}"])
        export_data.append(["Cost per Box (EUR)", f"€{profit_data.get('final_cost_per_box_usd', 0) * usd_to_eur_rate:,.2f}"])
    
    return pd.DataFrame(export_data)

def create_excel_export(summary_data_dict, profit_data, calculation_details, sensitivity_data):
    """Create Excel export with multiple sheets and formatting"""
    wb = Workbook()
    
    # Remove default sheet
    if wb.active is not None:
        wb.remove(wb.active)
    
    # Summary sheet
    ws_summary = wb.create_sheet("Cost Summary")
    ws_summary.append(["Cost Calculator - Summary Report"])
    ws_summary.append([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    ws_summary.append([])
    
    # Add calculation details
    ws_summary.append(["Calculation Details"])
    ws_summary.append(["Product", calculation_details.get('product', 'N/A')])
    ws_summary.append(["Quantity", calculation_details.get('quantity', 'N/A')])
    ws_summary.append(["Shipment Type", calculation_details.get('shipment_type', 'N/A')])
    ws_summary.append([])
    
    # Add cost breakdown
    ws_summary.append(["Cost Breakdown"])
    for key, value in summary_data_dict.items():
        if value is not None:
            ws_summary.append([key, value])
    
    # Profit Analysis sheet
    if profit_data:
        ws_profit = wb.create_sheet("Profit Analysis")
        ws_profit.append(["Profit Analysis"])
        ws_profit.append([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        ws_profit.append([])
        ws_profit.append(["Cost per Box (USD)", f"${profit_data.get('final_cost_per_box_usd', 0):,.2f}"])
        ws_profit.append(["Cost per Box (EUR)", f"€{profit_data.get('final_cost_per_box_usd', 0) * usd_to_eur_rate:,.2f}"])
        
        # Add sensitivity analysis
        if sensitivity_data:
            ws_profit.append([])
            ws_profit.append(["Sensitivity Analysis"])
            headers = list(sensitivity_data[0].keys())
            ws_profit.append(headers)
            for row in sensitivity_data:
                ws_profit.append([row[header] for header in headers])
    
    return wb

def get_download_link(data, filename, file_type):
    """Generate download link for files"""
    if file_type == "csv":
        csv = data.to_csv(index=False, header=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    elif file_type == "excel":
        buffer = io.BytesIO()
        data.save(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel</a>'
    return href

# --- Function to Fetch Exchange Rate ---
# ... (function remains the same) ...
@st.cache_data(ttl=3600)
def get_try_to_usd_rate():
    try:
        response = requests.get(FRANKFURTER_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data.get('rates', {}).get('USD')
        date = data.get('date', 'N/A')
        if rate:
            st.session_state['rate_source'] = f"API ({date})"
            st.session_state['rate_date'] = date
            return float(rate)
        else:
            st.session_state['rate_source'] = "API Error (Rate Missing)"
            st.session_state['rate_date'] = None
            return None
    except Exception as e:
        st.session_state['rate_source'] = f"API Failed ({type(e).__name__})"
        st.session_state['rate_date'] = None
        return None

# --- Initialize session state ---
# ... (initialization remains the same) ...
if 'rate_source' not in st.session_state: st.session_state['rate_source'] = "Not Fetched"
if 'rate_date' not in st.session_state: st.session_state['rate_date'] = None
# Add USD to EUR rate tracking
if 'usd_eur_rate_source' not in st.session_state: st.session_state['usd_eur_rate_source'] = "Not Fetched"
if 'usd_eur_rate_date' not in st.session_state: st.session_state['usd_eur_rate_date'] = None
# Add defaults for values needed by other pages if not present
if 'calculation_done' not in st.session_state: st.session_state['calculation_done'] = False
if 'final_cost_per_box_usd' not in st.session_state: st.session_state['final_cost_per_box_usd'] = 0.0


# --- Exchange Rate Handling (Includes Radio Button) ---
# ... (logic remains the same) ...
st.sidebar.subheader("Exchange Rate (TRY->USD)")
rate_source_choice = st.sidebar.radio("Select Rate Source:", ("Automatic (Frankfurter API)", "Manual Input"), key="rate_choice", index=0)
exchange_rate = None; manual_rate_input = None; rate_display_value = "N/A"; rate_display_source = "Not Set"
if rate_source_choice == "Automatic (Frankfurter API)":
    exchange_rate = get_try_to_usd_rate()
    if exchange_rate is None: st.sidebar.warning(f"⚠️ API Failed ({st.session_state.get('rate_source', '?')}). Enter rate manually:"); manual_rate_input = st.sidebar.number_input("Enter TRY to USD Rate (Fallback):", min_value=0.0001, value=FALLBACK_TRY_TO_USD, step=0.0001, format="%.6f", key="manual_rate_fallback"); exchange_rate = manual_rate_input; rate_display_source = "Manual Fallback"; rate_display_value = f"{exchange_rate:.6f}" if exchange_rate is not None else "Error"
    else: rate_display_source = st.session_state.get('rate_source', 'API'); rate_display_value = f"{exchange_rate:.6f}"; st.sidebar.success("Live rate fetched!")
elif rate_source_choice == "Manual Input":
    manual_rate_input = st.sidebar.number_input("Enter TRY to USD Rate Manually:", min_value=0.0001, value=FALLBACK_TRY_TO_USD, step=0.0001, format="%.6f", key="manual_rate_direct")
    exchange_rate = manual_rate_input; rate_display_source = "Manual Input"; rate_display_value = f"{exchange_rate:.6f}" if exchange_rate is not None else "Error"
if exchange_rate is None or not isinstance(exchange_rate, (int, float)) or exchange_rate <= 0: exchange_rate = FALLBACK_TRY_TO_USD; rate_display_source = "Fallback Default"; rate_display_value = f"{exchange_rate:.6f}"; st.sidebar.warning("Using default fallback rate.")
st.sidebar.metric(label=f"TRY to USD Rate Used ({rate_display_source})", value=rate_display_value)

# --- USD to EUR Exchange Rate Handling ---
st.sidebar.subheader("Exchange Rate (USD->EUR)")

usd_eur_rate_source_choice = st.sidebar.radio("Select USD->EUR Rate Source:", ("Automatic (Frankfurter API)", "Manual Input"), key="usd_eur_rate_choice", index=0)
manual_usd_eur_rate_input = None; usd_eur_rate_display_value = "N/A"; usd_eur_rate_display_source = "Not Set"

if usd_eur_rate_source_choice == "Automatic (Frankfurter API)":
    usd_to_eur_rate = get_usd_to_eur_rate()
    if usd_to_eur_rate is None: 
        st.sidebar.warning(f"⚠️ API Failed ({st.session_state.get('usd_eur_rate_source', '?')}). Enter rate manually:")
        manual_usd_eur_rate_input = st.sidebar.number_input("Enter USD to EUR Rate (Fallback):", min_value=0.0001, value=FALLBACK_USD_TO_EUR, step=0.0001, format="%.4f", key="manual_usd_eur_rate_fallback")
        usd_to_eur_rate = manual_usd_eur_rate_input
        usd_eur_rate_display_source = "Manual Fallback"
        usd_eur_rate_display_value = f"{usd_to_eur_rate:.4f}" if usd_to_eur_rate is not None else "Error"
    else: 
        usd_eur_rate_display_source = st.session_state.get('usd_eur_rate_source', 'API')
        usd_eur_rate_display_value = f"{usd_to_eur_rate:.4f}"
        st.sidebar.success("Live USD->EUR rate fetched!")
elif usd_eur_rate_source_choice == "Manual Input":
    manual_usd_eur_rate_input = st.sidebar.number_input("Enter USD to EUR Rate Manually:", min_value=0.0001, value=FALLBACK_USD_TO_EUR, step=0.0001, format="%.4f", key="manual_usd_eur_rate_direct")
    usd_to_eur_rate = manual_usd_eur_rate_input
    usd_eur_rate_display_source = "Manual Input"
    usd_eur_rate_display_value = f"{usd_to_eur_rate:.4f}" if usd_to_eur_rate is not None else "Error"

if usd_to_eur_rate is None or not isinstance(usd_to_eur_rate, (int, float)) or usd_to_eur_rate <= 0: 
    usd_to_eur_rate = FALLBACK_USD_TO_EUR
    usd_eur_rate_display_source = "Fallback Default"
    usd_eur_rate_display_value = f"{usd_to_eur_rate:.4f}"
    st.sidebar.warning("Using default USD->EUR fallback rate.")

st.sidebar.metric(label=f"USD to EUR Rate Used ({usd_eur_rate_display_source})", value=usd_eur_rate_display_value)

# --- Rate Refresh Button ---
col_refresh1, col_refresh2 = st.sidebar.columns(2)
if col_refresh1.button("🔄 Refresh TRY->USD", help="Refresh TRY to USD exchange rate"):
    st.cache_data.clear()
    st.rerun()
if col_refresh2.button("🔄 Refresh USD->EUR", help="Refresh USD to EUR exchange rate"):
    st.cache_data.clear()
    st.rerun()

# --- Currency Display Toggle ---
st.sidebar.subheader("Currency Display")
currency_display_mode = st.sidebar.selectbox(
    "Display Currency:",
    ["USD Only", "EUR Only"],
    help="Choose how to display costs in the results"
)


# --- Initialize DataFrames BEFORE loading ---
# ... (initialization remains the same) ...
components_df_try_loaded = None; product_recipe_df = None; fixed_df_usd_loaded = None
product_weights_df = None; air_rates_df = None; pallets_df = None; product_packing_df = None # Added packing_df
components_df = None; fixed_df = None; errors = []

# --- Load Data from CSV Files (with improved interest cost handling) ---
# ... (load_csv function remains largely the same, validation improved slightly) ...
def load_csv(file_path, required_cols, numeric_cols=None, decimal_char='.', string_cols=None):
    global errors
    if not os.path.exists(file_path):
        errors.append(f"File missing: '{os.path.abspath(file_path)}'")
        return None
    try:
        df = pd.read_csv(file_path, decimal=decimal_char)
        missing = [c for c in required_cols if c not in df.columns]
        if missing: raise ValueError(f"'{file_path}' missing required columns: {missing}")

        # Basic check if dataframe is empty after loading
        if df.empty and required_cols:
            st.warning(f"Warning: File '{file_path}' loaded as empty. Check content and headers.")
            # Optionally return empty dataframe with expected columns?
            # return pd.DataFrame(columns=required_cols)

        if numeric_cols:
            for col in numeric_cols:
                if col not in df.columns: continue # Skip if numeric col isn't present (might be optional)

                # Special handling for Interest Cost row in fixed_costs.csv MonthlyCost
                is_interest_cost_row = False
                if file_path.endswith(FIXED_CSV) and col == 'MonthlyCost' and 'CostItem' in df.columns:
                     # Ensure CostItem is string before comparison
                     df['CostItem'] = df['CostItem'].astype(str).fillna('')
                     is_interest_cost_row = df['CostItem'].str.strip().str.lower() == INTEREST_COST_ITEM_NAME.lower()

                # Attempt conversion to numeric, coercing errors to NaN
                original_type = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Check for NaNs *except* in the interest cost row for MonthlyCost
                has_nan = df[col].isnull()
                if isinstance(is_interest_cost_row, pd.Series): # Ensure it's a boolean Series
                     rows_to_check_for_nan = has_nan & (~is_interest_cost_row)
                else: # If not the special case, check all rows
                     rows_to_check_for_nan = has_nan

                if rows_to_check_for_nan.any():
                    # Identify specific rows with errors if possible (more helpful message)
                    error_indices = df.index[rows_to_check_for_nan].tolist()
                    raise ValueError(f"Column '{col}' in '{file_path}' contains non-numeric values or blanks at row indices (starting 0): {error_indices[:5]}...") # Show first few errors

        if string_cols:
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('').str.strip() # Convert, fill NA with empty string, strip whitespace

        return df
    except Exception as e:
        errors.append(f"Error processing '{file_path}': {e}")
        return None

try: # Load all data
    components_df_try_loaded = load_csv(COMPONENTS_CSV, ['ComponentName', 'CostPerUnit', 'WeightKG'], ['CostPerUnit', 'WeightKG'])
    product_recipe_df = load_csv(RECIPE_CSV, ['ProductID', 'ComponentName', 'QuantityPerProduct'], ['QuantityPerProduct'], string_cols=['ProductID', 'ComponentName']) # Ensure ComponentName is string
    fixed_df_usd_loaded = load_csv(FIXED_CSV, ['CostItem', 'MonthlyCost', 'Category'], ['MonthlyCost'], string_cols=['Category','CostItem'])
    product_weights_df = load_csv(WEIGHTS_CSV, ['ProductID', 'NetWeightKG'], ['NetWeightKG'], decimal_char=',', string_cols=['ProductID']) # Watch decimal char if issues arise
    air_rates_df = load_csv(AIR_RATES_CSV, ['Destination', 'MinWeightKG', 'PricePerKG_USD', 'AirwayBill_USD'], ['MinWeightKG', 'PricePerKG_USD', 'AirwayBill_USD'])
    pallets_df = load_csv(PALLETS_CSV, ['PalletType', 'CostUSD', 'WeightKG'], ['CostUSD', 'WeightKG'])
    product_packing_df = load_csv(PACKING_CSV, ['ProductID', 'BoxesPerPallet'], ['BoxesPerPallet'], string_cols=['ProductID']) # Load packing data

    # Explicitly convert BoxesPerPallet to numeric to avoid type issues
    if product_packing_df is not None:
        product_packing_df['BoxesPerPallet'] = pd.to_numeric(product_packing_df['BoxesPerPallet'], errors='coerce')

    if errors: raise ValueError("Errors occurred during file loading. See details above.")

    # Process data only if loading succeeded
    components_df = components_df_try_loaded.copy()
    components_df['CostPerUnit_USD'] = components_df['CostPerUnit'] * exchange_rate
    components_df['ComponentName'] = components_df['ComponentName'].astype(str).str.strip() # Ensure string type and stripped

    fixed_df = fixed_df_usd_loaded.copy()
    fixed_df = fixed_df.rename(columns={'MonthlyCost': 'MonthlyCost_USD'})
    fixed_df['Category'] = fixed_df['Category'].astype(str).str.strip().str.title()
    fixed_df['CostItem'] = fixed_df['CostItem'].astype(str).str.strip() # Ensure string type and stripped

    # Validate fixed cost categories
    valid_categories = ['Primary', 'Secondary']
    invalid_cats_df = fixed_df[~fixed_df['Category'].isin(valid_categories)]
    if not invalid_cats_df.empty:
        invalid_cats_list = invalid_cats_df['Category'].unique().tolist()
        errors.append(f"Invalid Category values in '{FIXED_CSV}': {invalid_cats_list}. Expected 'Primary' or 'Secondary'.")

    # Ensure ProductIDs are string type for consistent merging/lookup across all relevant DFs
    if product_weights_df is not None: product_weights_df['ProductID'] = product_weights_df['ProductID'].astype(str)
    else: errors.append(f"'{WEIGHTS_CSV}' failed to load or is empty.")
    if product_recipe_df is not None: product_recipe_df['ProductID'] = product_recipe_df['ProductID'].astype(str)
    else: errors.append(f"'{RECIPE_CSV}' failed to load or is empty.")
    if product_packing_df is not None: product_packing_df['ProductID'] = product_packing_df['ProductID'].astype(str)
    # No critical error if packing is missing, handled later

    if errors: raise ValueError("Errors occurred during data processing. See details above.")

except Exception as e:
    st.error(" / ".join(errors) if errors else f"An unexpected error occurred during data loading/processing: {e}")
    st.stop() # Stop execution if critical files/processing failed

# --- Get unique lists ---
# ... (logic remains the same) ...
try:
    product_ids = sorted(product_weights_df['ProductID'].unique().tolist()) # Products must have weight defined
    air_destinations = sorted(air_rates_df['Destination'].unique().tolist()) if air_rates_df is not None else []
    pallet_types = ["None"] + sorted(pallets_df['PalletType'].unique().tolist()) if pallets_df is not None and 'PalletType' in pallets_df else ["None"]
except Exception as e: st.error(f"Could not extract lists: {e}"); product_ids = []; air_destinations = []; pallet_types = ["None"]; st.stop()


# --- Streamlit User Interface ---
st.sidebar.header("Calculation Inputs")
# Product Selection
# ... (logic remains the same) ...
if not product_ids: st.sidebar.warning("No products available."); selected_product = None
else: selected_product = st.sidebar.selectbox("Select Product:", product_ids)

# --- Quantity / Pallet Linking ---
# ... (logic remains the same) ...
st.sidebar.subheader("Quantity & Pallets")
auto_calc_boxes = st.sidebar.checkbox("Calculate Box Quantity from Pallets?", value=False)

boxes_per_pallet = 0; calculated_boxes = 0; can_auto_calc = False

if auto_calc_boxes and selected_product and product_packing_df is not None:
    packing_info = product_packing_df[product_packing_df['ProductID'] == str(selected_product)]
    if not packing_info.empty:
        boxes_per_pallet = packing_info['BoxesPerPallet'].iloc[0]
        if pd.notna(boxes_per_pallet) and boxes_per_pallet > 0: can_auto_calc = True
        else: st.sidebar.warning(f"Boxes/Pallet is zero or invalid for {selected_product}.")
    else: st.sidebar.warning(f"Packing info (Boxes/Pallet) not found for {selected_product} in '{PACKING_CSV}'."); auto_calc_boxes = False

num_pallets = st.sidebar.number_input("Number of Pallets:", min_value=0, value=1, step=1)

if auto_calc_boxes and can_auto_calc:
    calculated_boxes = num_pallets * boxes_per_pallet
    quantity_input = st.sidebar.number_input( "Quantity (Boxes/Units):", min_value=1, value=calculated_boxes, step=1, disabled=True, help=f"Auto: {num_pallets} pallets * {boxes_per_pallet} boxes/pallet")
    st.sidebar.caption(f"Using {boxes_per_pallet} Boxes/Pallet for {selected_product}")
else:
     quantity_input = st.sidebar.number_input("Quantity (Boxes/Units):", min_value=1, value=100, step=1, disabled=False, help="Enter manually, or check box above to calculate from pallets.")

# Pallet Type Selection
# ... (logic remains the same) ...
if not pallet_types: st.sidebar.warning("No pallet types loaded."); selected_pallet_type = "None"
else: none_index = pallet_types.index("None") if "None" in pallet_types else 0; selected_pallet_type = st.sidebar.selectbox("Select Pallet Type:", pallet_types, index=none_index)

# Raw Product Cost & Other Costs
# ... (logic remains the same) ...
st.sidebar.subheader("Costs") # Group remaining costs
raw_cost_per_kg_try = st.sidebar.number_input("Raw Product Cost per KG (TRY):", min_value=0.0, value=50.0, step=0.1, format="%.2f")

# --- Show Fixed Cost Totals in Radio Options ---
if fixed_df is not None:
    total_primary = fixed_df[fixed_df['Category'] == 'Primary']['MonthlyCost_USD'].sum()
    total_all = fixed_df[fixed_df['Category'].isin(['Primary', 'Secondary'])]['MonthlyCost_USD'].sum()
else:
    total_primary = 0.0
    total_all = 0.0
fixed_cost_options = [
    f"All Costs (Primary+Secondary) (${'{:,.2f}'.format(total_all)})",
    f"Primary Costs Only (${'{:,.2f}'.format(total_primary)})",
    "Add 10% of total value"
]
fixed_cost_selection = st.sidebar.radio("Include Fixed Costs:", fixed_cost_options, index=0)

# Map selection to logic
if "Primary Costs Only" in fixed_cost_selection:
    fixed_categories_to_include = ['Primary']
    fixed_cost_label_suffix = "(Primary Only)"
    fixed_cost_mode = "standard"
elif "10% of total value" in fixed_cost_selection:
    fixed_categories_to_include = []  # Not used in this mode
    fixed_cost_label_suffix = "(10% of Total Value)"
    fixed_cost_mode = "percent"
else:
    fixed_categories_to_include = ['Primary', 'Secondary']
    fixed_cost_label_suffix = "(All)"
    fixed_cost_mode = "standard"

unexpected_cost_try = st.sidebar.number_input("Unexpected Costs (Total TRY for Batch):", min_value=0.0, value=0.0, step=10.0, format="%.2f")
include_variable_costs = st.sidebar.checkbox("Include Variable Component Costs in COGS?", value=True)


# Logistics Inputs
# ... (logic remains the same) ...
st.sidebar.subheader("Logistics")
shipment_types = ["Select...", "Air", "Container", "Truck"]; selected_shipment_type = st.sidebar.selectbox("Select Shipment Type:", shipment_types)
selected_destination = None
if selected_shipment_type == "Air":
    if not air_destinations: st.sidebar.warning("No air destinations loaded.")
    else: selected_destination = st.sidebar.selectbox("Select Destination (Air):", ["Select..."] + air_destinations) # Add select prompt

manual_logistics_cost_usd = 0.0
if selected_shipment_type in ["Container", "Truck"]:
    # Determine currency for display based on mode
    logistics_currency = "EUR" if currency_display_mode == "EUR Only" else "USD"
    manual_logistics_cost_input = st.sidebar.number_input(
        f"Enter Fixed {selected_shipment_type} Price ({logistics_currency}):",
        min_value=0.0,
        value=4000.0,
        step=50.0,
        format="%.2f"
    )
    # Convert EUR to USD if needed
    if currency_display_mode == "EUR Only":
        manual_logistics_cost_usd = manual_logistics_cost_input / usd_to_eur_rate
    else:
        manual_logistics_cost_usd = manual_logistics_cost_input

# *** NEW: Rebate Input ***
st.sidebar.subheader("Sales Adjustments")
rebate_rate_input = st.sidebar.number_input(
    "Retailer Rebate/Fee (%):",
    min_value=0.0,
    value=0.0,        # Default to 0%
    step=0.1,
    format="%.2f",    # Allow one decimal place for percentage
    help="Enter a percentage (e.g., 5 for 5%) to be added to the Total Delivered Cost."
)


# --- Calculation Logic ---
# ... (check remains the same) ...
calculation_ready = (
    selected_product is not None and
    selected_pallet_type is not None and
    fixed_cost_selection is not None and
    selected_shipment_type != "Select..." and
    (selected_shipment_type != "Air" or (selected_destination is not None and selected_destination != "Select...")) and
    quantity_input >= 1
)

if calculation_ready and st.sidebar.button("Calculate Costs"):

    # --- Initialize variables ---
    # ... (Initialize ALL calculation variables used below to 0.0 or empty dataframes/strings) ...
    total_raw_cost_usd=0.0; raw_cost_per_box_usd=0.0; total_variable_comp_cost_usd=0.0; total_per_unit_variable_comp_cost_usd=0.0; total_pallet_cost_usd=0.0; pallet_cost_per_box_usd=0.0; total_variable_costs_incl_pallets_usd=0.0; variable_costs_incl_pallets_per_box_usd=0.0; total_allocated_fixed_cost_usd=0.0; fixed_cost_per_unit_usd=0.0; total_cogs_usd=0.0; cogs_per_box_usd=0.0; cogs_per_kg_usd=0.0; total_logistics_cost_usd=0.0; logistics_per_box_usd=0.0; logistics_per_kg_gross_usd=0.0; total_unexpected_cost_usd=0.0; unexpected_cost_per_box_usd=0.0; total_delivered_cost_usd=0.0; delivered_cost_per_box_usd=0.0; delivered_cost_per_kg_net_usd=0.0; total_packaging_weight_kg=0.0; calculated_gross_weight_kg_per_box=0.0; final_shipping_gross_weight_kg=0.0; total_pallet_weight_kg=0.0; total_net_weight_kg=0.0; freight_or_fixed_logistics_cost=0.0; fixed_logistics_price=0.0; awb_cost=0.0; logistics_rate_per_kg=0.0; interest_cost_usd=0.0;
    # *** NEW: Rebate variables ***
    rebate_percentage = 0.0; rebate_amount_usd = 0.0; final_total_cost_usd = 0.0; final_cost_per_box_usd = 0.0

    product_variable_costs_detailed = pd.DataFrame() # Ensure it's defined
    calc_errors = [] # List to hold calculation errors

    try: # Main calculation try block
        selected_product_str = str(selected_product)

        # 1. Get Pallet Specs & Cost/Weight
        cost_per_pallet_usd = 0.0; weight_per_pallet_kg = 0.0
        if selected_pallet_type != "None" and num_pallets > 0 and pallets_df is not None:
            pallet_spec = pallets_df[pallets_df['PalletType'] == selected_pallet_type]
            if not pallet_spec.empty:
                cost_per_pallet_usd = pallet_spec['CostUSD'].iloc[0]
                weight_per_pallet_kg = pallet_spec['WeightKG'].iloc[0]
            else: st.warning(f"Specs missing for pallet '{selected_pallet_type}'. Cost/Weight assumed 0.")
        total_pallet_cost_usd = num_pallets * cost_per_pallet_usd
        total_pallet_weight_kg = num_pallets * weight_per_pallet_kg
        pallet_cost_per_box_usd = total_pallet_cost_usd / quantity_input if quantity_input > 0 else 0

        # 2. Get Net Weight
        product_weight_info = product_weights_df[product_weights_df['ProductID'] == selected_product_str]
        if product_weight_info.empty: raise ValueError(f"Net Weight missing for '{selected_product_str}' in '{WEIGHTS_CSV}'.")
        net_weight_kg = product_weight_info['NetWeightKG'].iloc[0]
        if not net_weight_kg > 0: raise ValueError(f"Net Weight for '{selected_product_str}' must be > 0.")

        # 3. Raw Product Cost
        raw_cost_per_kg_usd = raw_cost_per_kg_try * exchange_rate
        raw_cost_per_box_usd = raw_cost_per_kg_usd * net_weight_kg
        total_raw_cost_usd = raw_cost_per_box_usd * quantity_input

        # 4. Variable Component Costs & Packaging Weight
        total_packaging_weight_kg = 0.0 # Initialize packaging weight

        # Always calculate packaging weight from recipe for logistics, regardless of cost inclusion
        product_recipe = product_recipe_df[product_recipe_df['ProductID'] == selected_product_str]
        if not product_recipe.empty:
            recipe_components = product_recipe['ComponentName'].unique()
            available_components = components_df['ComponentName'].unique()
            missing_in_components_df = [comp for comp in recipe_components if comp not in available_components]
            if missing_in_components_df:
                 raise ValueError(f"Component(s) in recipe for '{selected_product_str}' not found in '{COMPONENTS_CSV}': {missing_in_components_df}")

            product_variable_costs_detailed = pd.merge(
                product_recipe, components_df[['ComponentName', 'CostPerUnit_USD', 'WeightKG']],
                on='ComponentName', how='left'
            )
            if product_variable_costs_detailed.isnull().values.any():
                missing = pd.Series(product_variable_costs_detailed[product_variable_costs_detailed.isnull().any(axis=1)]['ComponentName']).unique().tolist()
                raise ValueError(f"Details missing after merge for component(s): {missing}. Check '{COMPONENTS_CSV}'.")

            product_variable_costs_detailed['QuantityPerProduct'] = pd.to_numeric(product_variable_costs_detailed['QuantityPerProduct'], errors='coerce')
            if product_variable_costs_detailed['QuantityPerProduct'].isnull().any():
                raise ValueError("Non-numeric 'QuantityPerProduct' found in recipe.")

            # Calculate Weight first
            product_variable_costs_detailed['LineItemWeightKG'] = product_variable_costs_detailed['WeightKG'] * product_variable_costs_detailed['QuantityPerProduct']
            total_packaging_weight_kg = product_variable_costs_detailed['LineItemWeightKG'].sum()

            # Calculate Cost only if included
            if include_variable_costs:
                product_variable_costs_detailed['LineItemCost_USD'] = product_variable_costs_detailed['CostPerUnit_USD'] * product_variable_costs_detailed['QuantityPerProduct']
                total_per_unit_variable_comp_cost_usd = product_variable_costs_detailed['LineItemCost_USD'].sum()
                total_variable_comp_cost_usd = total_per_unit_variable_comp_cost_usd * quantity_input
            else:
                # Ensure costs are zero if not included
                product_variable_costs_detailed['LineItemCost_USD'] = 0.0
                total_per_unit_variable_comp_cost_usd = 0.0
                total_variable_comp_cost_usd = 0.0
        else:
            # No recipe found, variable costs and packaging weight are zero
            total_per_unit_variable_comp_cost_usd = 0.0
            total_variable_comp_cost_usd = 0.0
            total_packaging_weight_kg = 0.0
            st.warning(f"No recipe found for ProductID '{selected_product_str}' in '{RECIPE_CSV}'. Variable costs & packaging weight assumed 0.")

        # Total Variable (Components + Pallets)
        total_variable_costs_incl_pallets_usd = total_variable_comp_cost_usd + total_pallet_cost_usd
        variable_costs_incl_pallets_per_box_usd = total_variable_costs_incl_pallets_usd / quantity_input if quantity_input > 0 else 0

        # --- Intermediate Weights ---
        calculated_gross_weight_kg_per_box = net_weight_kg + total_packaging_weight_kg
        total_batch_gross_weight_boxes_only = calculated_gross_weight_kg_per_box * quantity_input
        final_shipping_gross_weight_kg = total_batch_gross_weight_boxes_only + total_pallet_weight_kg

        # --- 6. Logistics Costs (Freight/Fixed ONLY) ---
        # ... (logic remains the same) ...
        freight_or_fixed_logistics_cost = 0.0; logistics_rate_per_kg = 0.0; awb_cost = 0.0; fixed_logistics_price = 0.0; logistics_cost_source = "N/A"
        if selected_shipment_type == "Air":
            if air_rates_df is not None and not air_rates_df.empty:
                 applicable_rates = air_rates_df[ (air_rates_df['Destination'] == selected_destination) & (air_rates_df['MinWeightKG'] <= final_shipping_gross_weight_kg) ].sort_values('MinWeightKG', ascending=False)
                 if not applicable_rates.empty:
                     rate_row = applicable_rates.iloc[0]; logistics_rate_per_kg = rate_row['PricePerKG_USD']; awb_cost = rate_row['AirwayBill_USD']
                     freight_or_fixed_logistics_cost = (final_shipping_gross_weight_kg * logistics_rate_per_kg) + awb_cost; logistics_cost_source = f"Air: {rate_row['MinWeightKG']}+ KG Tier"
                 else: st.warning(f"No AIR rate for {selected_destination} at {final_shipping_gross_weight_kg:.2f} KG."); logistics_cost_source = "Air: No Rate Found"; freight_or_fixed_logistics_cost = 0.0
            else: st.warning(f"Air rates file '{AIR_RATES_CSV}' missing or empty."); logistics_cost_source = "Air: Rates Missing"; freight_or_fixed_logistics_cost = 0.0
        elif selected_shipment_type in ["Container", "Truck"]:
             freight_or_fixed_logistics_cost = manual_logistics_cost_usd; fixed_logistics_price = manual_logistics_cost_usd; logistics_cost_source = f"{selected_shipment_type}: Manual Input"
        total_logistics_cost_usd = freight_or_fixed_logistics_cost
        logistics_per_box_usd = total_logistics_cost_usd / quantity_input if quantity_input > 0 else 0
        logistics_per_kg_gross_usd = total_logistics_cost_usd / final_shipping_gross_weight_kg if final_shipping_gross_weight_kg > 0 else 0

        # --- 7. Unexpected Cost ---
        # ... (logic remains the same) ...
        total_unexpected_cost_usd = unexpected_cost_try * exchange_rate
        unexpected_cost_per_box_usd = total_unexpected_cost_usd / quantity_input if quantity_input > 0 else 0

        # --- 8. Fixed Costs (10% Option Calculated Here, after Logistics) ---
        interest_base_cost = 0.0
        if fixed_cost_mode == "percent":
            total_value_for_percent = total_raw_cost_usd + total_variable_costs_incl_pallets_usd + total_logistics_cost_usd + total_unexpected_cost_usd
            fixed_cost_10_percent = total_value_for_percent * 0.10
            total_allocated_fixed_cost_usd = fixed_cost_10_percent  # Only the 10% value, do not add interest here
        else:
            # Remove debug output for production
            standard_fixed_df = fixed_df[ (fixed_df['CostItem'].str.strip().str.lower() != INTEREST_COST_ITEM_NAME.lower()) & (fixed_df['Category'].isin(fixed_categories_to_include)) ]
            total_standard_fixed_cost_usd = standard_fixed_df['MonthlyCost_USD'].sum()
            total_allocated_fixed_cost_usd = total_standard_fixed_cost_usd

        # Interest should account for the entire amount we pre-finance (raw, variable, fixed, logistics, unexpected)
        interest_base_cost = (
            total_raw_cost_usd
            + total_variable_costs_incl_pallets_usd
            + total_allocated_fixed_cost_usd
            + total_logistics_cost_usd
            + total_unexpected_cost_usd
        )
        interest_cost_usd = interest_base_cost * INTEREST_RATE
        fixed_cost_per_unit_usd = total_allocated_fixed_cost_usd / quantity_input if quantity_input > 0 else 0
        total_cogs_usd = total_raw_cost_usd + total_variable_costs_incl_pallets_usd + total_allocated_fixed_cost_usd + interest_cost_usd
        cogs_per_box_usd = total_cogs_usd / quantity_input if quantity_input > 0 else 0; total_net_weight_kg = net_weight_kg * quantity_input; cogs_per_kg_usd = total_cogs_usd / total_net_weight_kg if total_net_weight_kg > 0 else 0

        # --- 11. Total Delivered Cost (Before Rebate) ---
        # ... (logic remains the same) ...
        total_delivered_cost_usd = total_cogs_usd + total_logistics_cost_usd + total_unexpected_cost_usd
        delivered_cost_per_box_usd = total_delivered_cost_usd / quantity_input if quantity_input > 0 else 0
        delivered_cost_per_kg_net_usd = total_delivered_cost_usd / total_net_weight_kg if total_net_weight_kg > 0 else 0

        # *** NEW: 12. RETAILER REBATE/FEE CALCULATION ***
        rebate_percentage = rebate_rate_input # Get percentage from sidebar input
        # Calculate the rebate amount based on the total delivered cost
        rebate_amount_usd = total_delivered_cost_usd * (rebate_percentage / 100.0)
        # Calculate the final total cost including the rebate
        final_total_cost_usd = total_delivered_cost_usd + rebate_amount_usd
        # Calculate the final cost per box including the rebate
        final_cost_per_box_usd = final_total_cost_usd / quantity_input if quantity_input > 0 else 0

        # *** NEW: 13. PROFIT MARGIN CALCULATIONS ***
        # Store profit analysis data
        st.session_state['profit_analysis'] = {
            'final_cost_per_box_usd': final_cost_per_box_usd,
            'cogs_per_box_usd': cogs_per_box_usd,
            'delivered_cost_per_box_usd': delivered_cost_per_box_usd
        }

        # --- STORE RESULTS IN SESSION STATE ---
        # Store values needed potentially by other pages (like Batch Sales, Profit Calc)
        st.session_state['last_calc_product'] = selected_product_str
        st.session_state['last_calc_quantity'] = quantity_input
        st.session_state['cogs_per_box_usd'] = cogs_per_box_usd # COGS per box
        st.session_state['unexpected_cost_per_box_usd'] = unexpected_cost_per_box_usd
        st.session_state['calculated_gross_weight_kg_per_box'] = calculated_gross_weight_kg_per_box
        st.session_state['air_rates_df'] = air_rates_df # Pass air rates table if needed elsewhere
        st.session_state['selected_shipment_type'] = selected_shipment_type
        st.session_state['delivered_cost_per_box_usd'] = delivered_cost_per_box_usd # Cost BEFORE rebate
        # *** NEW: Store final cost per box including rebate ***
        st.session_state['final_cost_per_box_usd'] = final_cost_per_box_usd # Cost AFTER rebate
        st.session_state['last_calc_rebate_rate'] = rebate_percentage # Store rebate info too
        st.session_state['last_calc_rebate_amount'] = rebate_amount_usd

        # Update summary data dictionary to include rebate and final total
        summary_data = {
            "1. Raw Product": format_cost_by_mode(total_raw_cost_usd, currency_display_mode),
            "2. Variable Costs (incl. Pallets)": format_cost_by_mode(total_variable_costs_incl_pallets_usd, currency_display_mode),
            f"3. Fixed Costs {fixed_cost_label_suffix}": (
                format_cost_by_mode(fixed_cost_10_percent, currency_display_mode)
                if fixed_cost_mode == "percent"
                else format_cost_by_mode(total_allocated_fixed_cost_usd, currency_display_mode)
            ),
            f"   (Calc. Interest @ 5.0%)": format_cost_by_mode(interest_cost_usd, currency_display_mode),
            "   Subtotal COGS": format_cost_by_mode(total_cogs_usd, currency_display_mode),
            f"4. {selected_shipment_type} Freight/Fee": format_cost_by_mode(freight_or_fixed_logistics_cost, currency_display_mode),
            "   Subtotal Logistics": format_cost_by_mode(total_logistics_cost_usd, currency_display_mode),
            "5. Unexpected Costs": format_cost_by_mode(total_unexpected_cost_usd, currency_display_mode),
            "      Delivered Cost (Before Rebate)": format_cost_by_mode(total_delivered_cost_usd, currency_display_mode),
            f"6. Rebate/Fee ({rebate_percentage:.1f}%)": format_cost_by_mode(rebate_amount_usd, currency_display_mode),
            "Grand Total Cost (incl Rebate)": format_cost_by_mode(final_total_cost_usd, currency_display_mode)
        }
        # Remove None values from summary (like zero interest) for cleaner display
        st.session_state['summary_data'] = {k: v for k, v in summary_data.items() if v is not None}

        st.session_state['calculation_done'] = True
        st.session_state['last_calc_product'] = selected_product
        st.session_state['last_calc_quantity'] = quantity_input
        st.session_state['cogs_per_box_usd'] = cogs_per_box_usd
        st.session_state['unexpected_cost_per_box_usd'] = unexpected_cost_per_box_usd
        st.session_state['calculated_gross_weight_kg_per_box'] = calculated_gross_weight_kg_per_box
        st.session_state['air_rates_df'] = air_rates_df
        st.session_state['selected_shipment_type'] = selected_shipment_type
        st.session_state['final_cost_per_box_usd'] = final_cost_per_box_usd
        st.session_state['selected_pallet_type'] = selected_pallet_type
        st.session_state['boxes_per_pallet'] = boxes_per_pallet if boxes_per_pallet > 0 else 1  # Always set, fallback to 1 if not valid
        st.session_state['weight_per_pallet_kg'] = weight_per_pallet_kg
        st.session_state['num_pallets'] = num_pallets
        # OVERWRITE THE CLEANED DICTIONARY WITH THE ORIGINAL ONE
        st.session_state['summary_data'] = summary_data

    # --- Exception Handling for Main Calculation ---
    except ValueError as ve: calc_errors.append(f"Data Input Error: {ve}")
    except KeyError as ke: calc_errors.append(f"Data Lookup Error: Missing key {ke}. Check file headers/content.")
    except ZeroDivisionError: calc_errors.append("Calculation Error: Division by zero (check Quantity).")
    except AssertionError as ae: calc_errors.append(f"Data Validation Failed: {ae}")
    except Exception as e_main_calc: calc_errors.append(f"Unexpected calculation error: {e_main_calc}")

    # --- Display Results OR Errors ---
    if calc_errors:
        st.error("Calculation failed. Please resolve errors:")
        for err in calc_errors: st.error(f"- {err}")
        st.session_state['calculation_done'] = False # Ensure flag is false on error
    elif st.session_state.get('calculation_done', False): # Proceed only if calculation was successful
        st.header("Calculation Results (in USD)")

        # Update top display line to include rebate % if applicable
        display_destination = f" | Dest: `{selected_destination}`" if selected_shipment_type == "Air" else ""
        display_pallet = f" | Pallet: `{selected_pallet_type}` (x{num_pallets})" if selected_pallet_type not in [None, "None"] else ""
        display_rebate = f" | Rebate: `{rebate_percentage:.1f}%`" if rebate_percentage > 0 else "" # Add rebate display
        st.write(
            f"**Product:** `{selected_product_str}` | **Qty:** `{quantity_input}` | **Ship Via:** `{selected_shipment_type}`"
            f"{display_destination}{display_pallet}{display_rebate} | **Fixed Costs:** `{fixed_cost_selection}`" # Added rebate here
        )

        colw1, colw2, colw3 = st.columns(3)
        colw1.metric("Net Wt/Box", f"{net_weight_kg:.3f} KG")
        colw2.metric("Gross Wt/Box", f"{calculated_gross_weight_kg_per_box:.3f} KG")
        colw3.metric("Final Shipping Wt (Batch)", f"{final_shipping_gross_weight_kg:.3f} KG", f"{num_pallets} Pallets")

        st.markdown("---")
        # Renamed Delivered Cost tab
        tab_cogs, tab_logistics, tab_total, tab_summary, tab_profit = st.tabs([
            "💰 COGS",
            f"🚚 Logistics ({selected_shipment_type})",
            "📦 Delivered Cost (+Rebate)", # Renamed
            "📊 Batch Summary Detail",
            "📈 Profit Analysis"
        ])

        # --- COGS Tab ---
        # ... (remains the same) ...
        with tab_cogs:
            st.subheader(f"Cost of Goods Sold {fixed_cost_label_suffix}")
            col1a, col1b = st.columns(2); col1a.metric("Total COGS", format_cost_by_mode(total_cogs_usd, currency_display_mode)); col1b.metric("COGS / Box", format_cost_by_mode(cogs_per_box_usd, currency_display_mode)); col1b.metric("COGS / KG (Net)", format_cost_by_mode(cogs_per_kg_usd, currency_display_mode))
            st.subheader("COGS Components (Total)"); col2a, col2b, col2c = st.columns(3); col2a.metric("Raw Product", format_cost_by_mode(total_raw_cost_usd, currency_display_mode)); col2b.metric("Variable (incl. Pallets)", format_cost_by_mode(total_variable_costs_incl_pallets_usd, currency_display_mode), help=f"Only included if checkbox is checked. Pallet cost always included."); col2c.metric(f"Fixed Costs {fixed_cost_label_suffix}", format_cost_by_mode(total_allocated_fixed_cost_usd, currency_display_mode), help=f"Includes Calc. Interest: {format_cost_by_mode(interest_cost_usd, currency_display_mode)}" if interest_cost_usd > 0 else None)
            st.subheader("COGS Breakdown (Per Box)"); st.write(f"- Raw Product: {format_cost_by_mode(raw_cost_per_box_usd, currency_display_mode)}"); st.write(f"- Variable (incl. Pallets): {format_cost_by_mode(variable_costs_incl_pallets_per_box_usd, currency_display_mode)}"); st.caption(f"  (Components: {format_cost_by_mode(total_per_unit_variable_comp_cost_usd, currency_display_mode)} + Pallets: {format_cost_by_mode(pallet_cost_per_box_usd, currency_display_mode)})"); st.write(f"- Fixed Costs {fixed_cost_label_suffix}: {format_cost_by_mode(fixed_cost_per_unit_usd, currency_display_mode)}"); st.write(f"**= Total COGS Per Box:** {format_cost_by_mode(cogs_per_box_usd, currency_display_mode)}")

        # --- Logistics Tab ---
        # ... (remains the same) ...
        with tab_logistics:
            st.subheader(f"{selected_shipment_type} Logistics Cost ({logistics_cost_source})"); col3a, col3b = st.columns(2); col3a.metric("Total Logistics Cost", format_cost_by_mode(total_logistics_cost_usd, currency_display_mode)); col3b.metric("Logistics / Box", format_cost_by_mode(logistics_per_box_usd, currency_display_mode));
            if final_shipping_gross_weight_kg > 0: col3b.metric("Logistics / KG (Gross Ship Wt)", format_cost_by_mode(logistics_per_kg_gross_usd, currency_display_mode))
            st.subheader("Logistics Components (Total)");
            if selected_shipment_type == "Air": col4a, col4b = st.columns(2); freight_cost = final_shipping_gross_weight_kg * logistics_rate_per_kg; col4a.metric("Freight Cost", format_cost_by_mode(freight_cost, currency_display_mode), f"{final_shipping_gross_weight_kg:.2f}kg @ {format_cost_by_mode(logistics_rate_per_kg, currency_display_mode)}/kg"); col4b.metric("Airway Bill Cost", format_cost_by_mode(awb_cost, currency_display_mode))
            elif selected_shipment_type in ["Container", "Truck"]:
                # For Container/Truck, show the price in the selected currency
                if currency_display_mode == "EUR Only":
                    fixed_price_display = manual_logistics_cost_input
                else:
                    fixed_price_display = fixed_logistics_price
                st.metric(f"{selected_shipment_type} Fixed Price", format_cost_by_mode(fixed_price_display, currency_display_mode))
            else: st.write("N/A")

        # --- Total Delivered Cost Tab (Modified for Rebate) ---
        with tab_total:
            st.subheader("Total Delivered Cost (COGS + Logistics + Unexpected + Rebate)") # Updated title
            # Show costs *before* rebate first
            st.metric("Total Delivered Cost (Before Rebate)", format_cost_by_mode(total_delivered_cost_usd, currency_display_mode))
            st.write(f"*Calculation: {format_cost_by_mode(total_cogs_usd, currency_display_mode)} (COGS) + {format_cost_by_mode(total_logistics_cost_usd, currency_display_mode)} (Logistics) + {format_cost_by_mode(total_unexpected_cost_usd, currency_display_mode)} (Unexpected)*")
            st.markdown("---")
            # Show rebate calculation
            st.subheader(f"Retailer Rebate/Fee Adjustment ({rebate_percentage:.1f}%)")
            col_rebate1, col_rebate2 = st.columns(2)
            col_rebate1.metric("Rebate/Fee Amount (Total Batch)", format_cost_by_mode(rebate_amount_usd, currency_display_mode))
            # Show final costs *after* rebate
            col_rebate2.metric("Final Total Cost (Incl. Rebate)", format_cost_by_mode(final_total_cost_usd, currency_display_mode))
            col_rebate2.metric("Final Cost / Box (Incl. Rebate)", format_cost_by_mode(final_cost_per_box_usd, currency_display_mode))
            if total_net_weight_kg > 0:
                 final_cost_per_kg_net_usd = final_total_cost_usd / total_net_weight_kg
                 col_rebate2.metric("Final Cost / KG Net (Incl. Rebate)", format_cost_by_mode(final_cost_per_kg_net_usd, currency_display_mode))
            st.write(f"*Final Calculation: {format_cost_by_mode(total_delivered_cost_usd, currency_display_mode)} (Delivered Cost) + {format_cost_by_mode(rebate_amount_usd, currency_display_mode)} (Rebate)*")


        # --- Batch Summary Detail Tab (Modified for Rebate) ---
        with tab_summary:
            st.subheader("Total Batch Cost Summary")
            summary_data_dict = st.session_state.get('summary_data', {})
            if summary_data_dict:
                # Pie Chart: Shows breakdown BEFORE rebate for clarity on operational costs
                plot_data = {
                    'Raw Product': total_raw_cost_usd,
                    'Variable (incl. Pallets)': total_variable_costs_incl_pallets_usd,
                    'Fixed Costs': total_allocated_fixed_cost_usd,
                    'Freight/Fee': freight_or_fixed_logistics_cost,
                    'Unexpected': total_unexpected_cost_usd
                }
                plot_data_filtered = {k: v for k, v in plot_data.items() if v > 0}
                if plot_data_filtered:
                    plot_df = pd.DataFrame(list(plot_data_filtered.items()), columns=pd.Index(["Cost Category", "Total Cost (USD)"]))
                    fig = px.pie(plot_df, names='Cost Category', values='Total Cost (USD)',
                                 title='Cost Breakdown by Component (Before Rebate)', # Updated title
                                 hole=.3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                else: st.write("No primary cost components > 0 for chart.")

                # Summary Table: Shows the FULL breakdown including rebate and final total
                st.subheader("Full Cost Breakdown Table")
                summary_df = pd.DataFrame(list(summary_data_dict.items()), columns=pd.Index(["Cost Category", "Total Cost (USD/EUR)"]))
                st.dataframe(summary_df, use_container_width=True)
            else: st.write("Summary data not available (Run calculation first).")


        # --- Profit Analysis Tab ---
        with tab_profit:
            st.subheader("Profit Margin Analysis")
            
            # Get stored profit analysis data
            profit_data = st.session_state.get('profit_analysis', {})
            if profit_data:
                final_cost_per_box = profit_data.get('final_cost_per_box_usd', 0)
                cogs_per_box = profit_data.get('cogs_per_box_usd', 0)
                delivered_cost_per_box = profit_data.get('delivered_cost_per_box_usd', 0)
                
                # Display current costs
                col_cost1, col_cost2, col_cost3 = st.columns(3)
                col_cost1.metric("COGS per Box", format_cost_by_mode(cogs_per_box, currency_display_mode))
                col_cost2.metric("Delivered Cost per Box", format_cost_by_mode(delivered_cost_per_box, currency_display_mode))
                col_cost3.metric("Final Cost per Box (incl. Rebate)", format_cost_by_mode(final_cost_per_box, currency_display_mode))
                
                st.markdown("---")
                
                # Interactive Profit Calculator
                st.subheader("Profit Calculator")
                sales_price_input = st.number_input(
                    "Enter Sales Price per Box (USD):",
                    min_value=0.0,
                    value=final_cost_per_box * 1.2,  # Default 20% markup
                    step=0.01,
                    format="%.2f"
                )
                
                if sales_price_input > 0:
                    # Calculate profit margins
                    profit_analysis = calculate_profit_margins(final_cost_per_box, sales_price_input)
                    
                    col_profit1, col_profit2, col_profit3 = st.columns(3)
                    col_profit1.metric("Profit per Box", format_cost_by_mode(profit_analysis['profit_per_box'], currency_display_mode))
                    col_profit2.metric("Profit Margin", f"{profit_analysis['profit_margin_percent']:.1f}%")
                    col_profit3.metric("ROI", f"{profit_analysis['roi_percent']:.1f}%")
                    
                    # Total profit for batch
                    total_profit = profit_analysis['profit_per_box'] * quantity_input
                    st.metric("Total Profit for Batch", format_cost_by_mode(total_profit, currency_display_mode))
                    
                    # Cost Sensitivity Analysis
                    st.markdown("---")
                    st.subheader("Cost Sensitivity Analysis")
                    
                    # Create sensitivity table
                    markup_percentages = [5, 10, 15, 20, 25, 30, 40, 50]
                    sensitivity_data = []
                    
                    for markup in markup_percentages:
                        sales_price = final_cost_per_box * (1 + markup / 100)
                        analysis = calculate_profit_margins(final_cost_per_box, sales_price)
                        sensitivity_data.append({
                            "Markup %": markup,
                            "Sales Price": format_cost_by_mode(sales_price, currency_display_mode),
                            "Profit per Box": format_cost_by_mode(analysis['profit_per_box'], currency_display_mode),
                            "Profit Margin %": f"{analysis['profit_margin_percent']:.1f}%",
                            "ROI %": f"{analysis['roi_percent']:.1f}%"
                        })
                    
                    sensitivity_df = pd.DataFrame(sensitivity_data)
                    st.dataframe(sensitivity_df, use_container_width=True)

                    # Add a graph for Cost Sensitivity Analysis
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sensitivity_df['Markup %'], y=sensitivity_df['Profit per Box'].apply(lambda x: float(x.replace('$','').replace(',','').replace('€',''))), mode='lines+markers', name='Profit per Box'))
                    fig.add_trace(go.Scatter(x=sensitivity_df['Markup %'], y=sensitivity_df['Profit Margin %'].apply(lambda x: float(x.replace('%',''))), mode='lines+markers', name='Profit Margin %'))
                    fig.add_trace(go.Scatter(x=sensitivity_df['Markup %'], y=sensitivity_df['ROI %'].apply(lambda x: float(x.replace('%',''))), mode='lines+markers', name='ROI %'))
                    fig.update_layout(title='Cost Sensitivity Analysis', xaxis_title='Markup %', yaxis_title='Value', legend_title='Metric')
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.info("Run a calculation first to see profit analysis.")

        # --- Optional Variable Breakdown ---
        with st.expander("Show Detailed Variable Cost & Weight Breakdown (Per Box) - Components Only"):
             # Ensure product_variable_costs_detailed is available and has the cost column if needed
            if not product_variable_costs_detailed.empty and 'LineItemCost_USD' in product_variable_costs_detailed.columns:
                 # Select columns carefully in case Cost was excluded
                 cols_to_show = ['ComponentName', 'QuantityPerProduct', 'WeightKG', 'LineItemWeightKG']
                 if include_variable_costs:
                     cols_to_show.extend(['CostPerUnit_USD', 'LineItemCost_USD'])

                 breakdown_display = product_variable_costs_detailed[cols_to_show]
                 # Rename columns for display
                 rename_map = {
                     'CostPerUnit_USD': 'Cost/Unit ($)', 'LineItemCost_USD': 'TotalCost ($)',
                     'WeightKG': 'Wt/Unit (KG)', 'LineItemWeightKG': 'TotalWt (KG)'
                 }
                 breakdown_display = breakdown_display.rename(columns=rename_map)

                 # Define formatters, applying only if column exists
                 formatters = {
                    'QuantityPerProduct': '{:,.2f}',
                    'Wt/Unit (KG)': '{:,.4f}', 'TotalWt (KG)': '{:,.4f}'
                 }
                 if include_variable_costs:
                     formatters['Cost/Unit ($)'] = '${:,.6f}'
                     formatters['TotalCost ($)'] = '${:,.6f}'
                 st.dataframe(breakdown_display.set_index('ComponentName').style.format(formatters))
                 st.write(f"**Total Component Packaging Weight (per Box):** {total_packaging_weight_kg:.4f} KG")
                 if not include_variable_costs:
                     st.caption("*Variable Component Costs were excluded from COGS calculation based on checkbox selection.*")
            else: st.write("(No variable components found in recipe or breakdown unavailable)")

# --- Export Section ---
st.markdown("---")
st.subheader("📊 Export Results")

if st.session_state.get('calculation_done', False):
    # Get data for export
    summary_data_dict = st.session_state.get('summary_data', {})
    profit_data = st.session_state.get('profit_analysis', {})
    calculation_details = {
        'product': st.session_state.get('last_calc_product', 'N/A'),
        'quantity': st.session_state.get('last_calc_quantity', 'N/A'),
        'shipment_type': st.session_state.get('selected_shipment_type', 'N/A'),
        'fixed_cost_mode': fixed_cost_mode if 'fixed_cost_mode' in locals() else 'N/A'
    }
    
    # Create sensitivity data for export
    sensitivity_data = []
    if profit_data and 'final_cost_per_box_usd' in profit_data:
        final_cost_per_box = profit_data['final_cost_per_box_usd']
        markup_percentages = [5, 10, 15, 20, 25, 30, 40, 50]
        for markup in markup_percentages:
            sales_price = final_cost_per_box * (1 + markup / 100)
            analysis = calculate_profit_margins(final_cost_per_box, sales_price)
            sensitivity_data.append({
                "Markup %": markup,
                "Sales Price (USD)": f"${sales_price:,.2f}",
                "Profit per Box (USD)": f"${analysis['profit_per_box']:,.2f}",
                "Profit Margin %": f"{analysis['profit_margin_percent']:.1f}%",
                "ROI %": f"{analysis['roi_percent']:.1f}%"
            })
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        st.write("**Export as CSV:**")
        csv_data = create_csv_export(summary_data_dict, profit_data, calculation_details)
        csv_filename = f"cost_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_link = get_download_link(csv_data, csv_filename, "csv")
        st.markdown(csv_link, unsafe_allow_html=True)
    
    with col_export2:
        st.write("**Export as Excel:**")
        excel_data = create_excel_export(summary_data_dict, profit_data, calculation_details, sensitivity_data)
        excel_filename = f"cost_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        excel_link = get_download_link(excel_data, excel_filename, "excel")
        st.markdown(excel_link, unsafe_allow_html=True)
    
    st.caption("💡 Export includes cost breakdown, profit analysis, and sensitivity analysis (Excel only)")
else:
    st.info("Run a calculation first to enable export options.")

# --- UI Improvements and Advanced Features ---
st.markdown("---")
st.subheader("🎨 Advanced Features")

# Cost History Tracking
col_theme2 = st.columns(1)[0]
with col_theme2:
    st.write("**Cost History:**")
    if st.button("💾 Save Current Calculation", help="Save this calculation to history"):
        if st.session_state.get('calculation_done', False):
            # Save calculation to session state history
            if 'cost_history' not in st.session_state:
                st.session_state['cost_history'] = []
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'product': st.session_state.get('last_calc_product', 'N/A'),
                'quantity': st.session_state.get('last_calc_quantity', 'N/A'),
                'final_cost_per_box': st.session_state.get('final_cost_per_box_usd', 0),
                'shipment_type': st.session_state.get('selected_shipment_type', 'N/A'),
                'fixed_cost_mode': fixed_cost_mode if 'fixed_cost_mode' in locals() else 'N/A'
            }
            st.session_state['cost_history'].append(history_entry)
            st.success("Calculation saved to history!")

# Display Cost History
if 'cost_history' in st.session_state and st.session_state['cost_history']:
    st.subheader("📈 Cost History")
    history_df = pd.DataFrame(st.session_state['cost_history'])
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    history_df['final_cost_per_box'] = history_df['final_cost_per_box'].apply(lambda x: format_cost_by_mode(x, currency_display_mode))
    
    # Rename columns for display
    history_df = history_df.rename(columns={
        'timestamp': 'Date/Time',
        'product': 'Product',
        'quantity': 'Quantity',
        'final_cost_per_box': 'Cost per Box',
        'shipment_type': 'Shipment',
        'fixed_cost_mode': 'Fixed Cost Mode'
    })
    
    st.dataframe(history_df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state['cost_history'] = []
        st.rerun()

# --- Input prompts if calculation isn't ready ---
else:
    # Check specific conditions to give more targeted advice
    if not selected_product: st.warning("Select a product to begin.")
    elif quantity_input < 1: st.warning("Enter a valid Quantity (>= 1).")
    elif selected_shipment_type == "Select...": st.warning("Select a shipment type.")
    elif selected_shipment_type == "Air" and (selected_destination is None or selected_destination == "Select..."): st.warning("Select a destination for Air shipment.")
    elif selected_pallet_type is None: st.warning("Select a pallet type (use 'None' if no pallets).")
    else: st.info("Adjust inputs in the sidebar and click 'Calculate Costs'.") # Default message


# --- Display raw data ---
# ... (remains the same) ...
st.markdown("---")
with st.expander("Show Raw Data Loaded from Files"):
    # Display DataFrames safely checking if they exist
    if pallets_df is not None: st.write(f"**Pallet Specifications (`{PALLETS_CSV}`):**"); st.dataframe(pallets_df.style.format({'CostUSD': '${:,.2f}', 'WeightKG': '{:,.2f} KG'})); st.write("---")
    if air_rates_df is not None: st.write(f"**Air Freight Rates (`{AIR_RATES_CSV}` - USD):**"); st.dataframe(air_rates_df.style.format({'MinWeightKG': '{:,.1f} KG','PricePerKG_USD': '${:,.3f}','AirwayBill_USD': '${:,.2f}'})); st.write("---")
    if product_weights_df is not None: st.write(f"**Product Net Weights (`{WEIGHTS_CSV}`):**"); st.dataframe(product_weights_df.style.format({'NetWeightKG': '{:.3f} KG'})); st.write("---")
    if product_packing_df is not None: st.write(f"**Product Packing (`{PACKING_CSV}`):**"); st.dataframe(product_packing_df.style.format({'BoxesPerPallet': '{:,.0f}'})); st.write("---") # Added Packing DF
    if components_df is not None: st.write(f"**Processed Components (`{COMPONENTS_CSV}` - USD Cost, KG Weight):**"); st.dataframe(components_df[['ComponentName', 'CostPerUnit_USD', 'WeightKG']].style.format({'CostPerUnit_USD': '${:,.6f}', 'WeightKG': '{:,.4f} KG'})); st.write("---")
    if product_recipe_df is not None: st.write(f"**Product Recipes (`{RECIPE_CSV}`):**"); st.dataframe(product_recipe_df); st.write("---")
    if fixed_df is not None: st.write(f"**Processed Fixed Costs (`{FIXED_CSV}` - USD Monthly):**"); st.dataframe(fixed_df[['CostItem', 'MonthlyCost_USD', 'Category']].style.format({'MonthlyCost_USD': '${:,.2f}'}))
