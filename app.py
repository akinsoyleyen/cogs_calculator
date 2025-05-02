# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import requests
import os
from datetime import datetime
import plotly.express as px

# --- Define File Paths & Constants---
# ... (paths remain the same) ...
COMPONENTS_CSV = "components.csv"; RECIPE_CSV = "product_recipe.csv"; FIXED_CSV = "fixed_costs.csv"
WEIGHTS_CSV = "product_weights.csv"; AIR_RATES_CSV = "air_freight_rates.csv"; PALLETS_CSV = "pallets.csv"
PACKING_CSV = "product_packing.csv" # New packing file
FRANKFURTER_API_URL = "https://api.frankfurter.app/latest?from=TRY&to=USD"; FALLBACK_TRY_TO_USD = 0.0262
INTEREST_RATE = 0.02; INTEREST_COST_ITEM_NAME = "Interest Cost" # Make sure this matches CSV exactly (case-insensitive check used later)

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

# st.info(f"""
# **Data Files:** `{COMPONENTS_CSV}`(TRY), `{RECIPE_CSV}`, `{FIXED_CSV}`(USD), `{WEIGHTS_CSV}`, `{AIR_RATES_CSV}`(USD), `{PALLETS_CSV}`(USD), `{PACKING_CSV}`.
# """)

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
fixed_cost_options = ["All Costs (Primary + Secondary)", "Primary Costs Only"]; fixed_cost_selection = st.sidebar.radio("Include Fixed Costs:", fixed_cost_options, index=0)
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
if selected_shipment_type in ["Container", "Truck"]: manual_logistics_cost_usd = st.sidebar.number_input(f"Enter Fixed {selected_shipment_type} Price (USD):", min_value=0.0, value=4000.0, step=50.0, format="%.2f")

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
    total_raw_cost_usd=0.0; raw_cost_per_box_usd=0.0; total_variable_comp_cost_usd=0.0; total_per_unit_variable_comp_cost_usd=0.0; total_pallet_cost_usd=0.0; pallet_cost_per_box_usd=0.0; total_variable_costs_incl_pallets_usd=0.0; variable_costs_incl_pallets_per_box_usd=0.0; total_allocated_fixed_cost_usd=0.0; fixed_cost_per_unit_usd=0.0; total_cogs_usd=0.0; cogs_per_box_usd=0.0; cogs_per_kg_usd=0.0; total_logistics_cost_usd=0.0; logistics_per_box_usd=0.0; logistics_per_kg_gross_usd=0.0; total_unexpected_cost_usd=0.0; unexpected_cost_per_box_usd=0.0; total_delivered_cost_usd=0.0; delivered_cost_per_box_usd=0.0; delivered_cost_per_kg_net_usd=0.0; total_packaging_weight_kg=0.0; calculated_gross_weight_kg_per_box=0.0; final_shipping_gross_weight_kg=0.0; total_pallet_weight_kg=0.0; total_net_weight_kg=0.0; freight_or_fixed_logistics_cost=0.0; fixed_logistics_price=0.0; awb_cost=0.0; logistics_rate_per_kg=0.0; fixed_cost_label_suffix="(All)"; interest_cost_usd=0.0;
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
                missing = product_variable_costs_detailed[product_variable_costs_detailed.isnull().any(axis=1)]['ComponentName'].unique().tolist()
                raise ValueError(f"Details missing after merge for component(s): {missing}. Check '{COMPONENTS_CSV}'.")

            product_variable_costs_detailed['QuantityPerProduct'] = pd.to_numeric(product_variable_costs_detailed['QuantityPerProduct'], errors='coerce')
            if product_variable_costs_detailed['QuantityPerProduct'].isnull().any(): raise ValueError("Non-numeric 'QuantityPerProduct' found in recipe.")

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

        # --- 5. Standard Fixed Costs (Filtered, Excl. Interest) ---
        # ... (logic remains the same) ...
        if fixed_cost_selection == "Primary Costs Only": fixed_categories_to_include = ['Primary']; fixed_cost_label_suffix = "(Primary Only)"
        else: fixed_categories_to_include = ['Primary', 'Secondary']; fixed_cost_label_suffix = "(All)"
        standard_fixed_df = fixed_df[ (fixed_df['CostItem'].str.strip().str.lower() != INTEREST_COST_ITEM_NAME.lower()) & (fixed_df['Category'].isin(fixed_categories_to_include)) ]
        total_standard_fixed_cost_usd = standard_fixed_df['MonthlyCost_USD'].sum()

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

        # --- 8. Interest Cost (Based on sum of others) ---
        # ... (logic remains the same) ...
        interest_cost_usd = 0.0
        interest_base_cost = total_raw_cost_usd + total_variable_costs_incl_pallets_usd + total_standard_fixed_cost_usd + total_logistics_cost_usd + total_unexpected_cost_usd
        interest_item_row = fixed_df[fixed_df['CostItem'].str.strip().str.lower() == INTEREST_COST_ITEM_NAME.lower()]
        if not interest_item_row.empty:
             if interest_item_row['Category'].iloc[0] in fixed_categories_to_include: interest_cost_usd = interest_base_cost * INTEREST_RATE

        # --- 9. FINAL Fixed Costs & COGS ---
        # ... (logic remains the same) ...
        total_allocated_fixed_cost_usd = total_standard_fixed_cost_usd + interest_cost_usd
        fixed_cost_per_unit_usd = total_allocated_fixed_cost_usd / quantity_input if quantity_input > 0 else 0
        total_cogs_usd = total_raw_cost_usd + total_variable_costs_incl_pallets_usd + total_allocated_fixed_cost_usd
        cogs_per_box_usd = total_cogs_usd / quantity_input if quantity_input > 0 else 0; total_net_weight_kg = net_weight_kg * quantity_input; cogs_per_kg_usd = total_cogs_usd / total_net_weight_kg if total_net_weight_kg > 0 else 0

        # --- 10. Total Delivered Cost (Before Rebate) ---
        # ... (logic remains the same) ...
        total_delivered_cost_usd = total_cogs_usd + total_logistics_cost_usd + total_unexpected_cost_usd
        delivered_cost_per_box_usd = total_delivered_cost_usd / quantity_input if quantity_input > 0 else 0
        delivered_cost_per_kg_net_usd = total_delivered_cost_usd / total_net_weight_kg if total_net_weight_kg > 0 else 0

        # *** NEW: 11. RETAILER REBATE/FEE CALCULATION ***
        rebate_percentage = rebate_rate_input # Get percentage from sidebar input
        # Calculate the rebate amount based on the total delivered cost
        rebate_amount_usd = total_delivered_cost_usd * (rebate_percentage / 100.0)
        # Calculate the final total cost including the rebate
        final_total_cost_usd = total_delivered_cost_usd + rebate_amount_usd
        # Calculate the final cost per box including the rebate
        final_cost_per_box_usd = final_total_cost_usd / quantity_input if quantity_input > 0 else 0

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
            "1. Raw Product": total_raw_cost_usd,
            "2. Variable Costs (incl. Pallets)": total_variable_costs_incl_pallets_usd,
            f"3. Fixed Costs {fixed_cost_label_suffix}": total_allocated_fixed_cost_usd,
            f"   (Calc. Interest @ {INTEREST_RATE:.1%})": interest_cost_usd if interest_cost_usd > 0 else None, # Show only if non-zero
            "   Subtotal COGS": total_cogs_usd,
            f"4. {selected_shipment_type} Freight/Fee": freight_or_fixed_logistics_cost,
            "   Subtotal Logistics": total_logistics_cost_usd,
            "5. Unexpected Costs": total_unexpected_cost_usd,
            "      Delivered Cost (Before Rebate)": total_delivered_cost_usd, # Show pre-rebate subtotal
            f"6. Rebate/Fee ({rebate_percentage:.1f}%)": rebate_amount_usd,   # Show rebate amount
            "Grand Total Cost (incl Rebate)": final_total_cost_usd        # Show final total with rebate
        }
        # Remove None values from summary (like zero interest) for cleaner display
        st.session_state['summary_data'] = {k: v for k, v in summary_data.items() if v is not None}

        st.session_state['calculation_done'] = True

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
        tab_cogs, tab_logistics, tab_total, tab_summary = st.tabs([
            "💰 COGS",
            f"🚚 Logistics ({selected_shipment_type})",
            "📦 Delivered Cost (+Rebate)", # Renamed
            "📊 Batch Summary Detail"
        ])

        # --- COGS Tab ---
        # ... (remains the same) ...
        with tab_cogs:
            st.subheader(f"Cost of Goods Sold {fixed_cost_label_suffix}")
            col1a, col1b = st.columns(2); col1a.metric("Total COGS", f"${total_cogs_usd:,.2f}"); col1b.metric("COGS / Box", f"${cogs_per_box_usd:,.2f}"); col1b.metric("COGS / KG (Net)", f"${cogs_per_kg_usd:,.2f}")
            st.subheader("COGS Components (Total)"); col2a, col2b, col2c = st.columns(3); col2a.metric("Raw Product", f"${total_raw_cost_usd:,.2f}"); col2b.metric("Variable (incl. Pallets)", f"${total_variable_costs_incl_pallets_usd:,.2f}", help=f"Only included if checkbox is checked. Pallet cost always included."); col2c.metric(f"Fixed Costs {fixed_cost_label_suffix}", f"${total_allocated_fixed_cost_usd:,.2f}", help=f"Includes Calc. Interest: ${interest_cost_usd:,.2f}" if interest_cost_usd > 0 else None)
            st.subheader("COGS Breakdown (Per Box)"); st.write(f"- Raw Product: ${raw_cost_per_box_usd:,.3f}"); st.write(f"- Variable (incl. Pallets): ${variable_costs_incl_pallets_per_box_usd:,.3f}"); st.caption(f"  (Components: ${total_per_unit_variable_comp_cost_usd:,.3f} + Pallets: ${pallet_cost_per_box_usd:,.3f})"); st.write(f"- Fixed Costs {fixed_cost_label_suffix}: ${fixed_cost_per_unit_usd:,.3f}"); st.write(f"**= Total COGS Per Box:** ${cogs_per_box_usd:,.3f}")

        # --- Logistics Tab ---
        # ... (remains the same) ...
        with tab_logistics:
            st.subheader(f"{selected_shipment_type} Logistics Cost ({logistics_cost_source})"); col3a, col3b = st.columns(2); col3a.metric("Total Logistics Cost", f"${total_logistics_cost_usd:,.2f}"); col3b.metric("Logistics / Box", f"${logistics_per_box_usd:,.2f}");
            if final_shipping_gross_weight_kg > 0: col3b.metric("Logistics / KG (Gross Ship Wt)", f"${logistics_per_kg_gross_usd:,.3f}")
            st.subheader("Logistics Components (Total)");
            if selected_shipment_type == "Air": col4a, col4b = st.columns(2); freight_cost = final_shipping_gross_weight_kg * logistics_rate_per_kg; col4a.metric("Freight Cost", f"${freight_cost:,.2f}", f"{final_shipping_gross_weight_kg:.2f}kg @ ${logistics_rate_per_kg:.2f}/kg"); col4b.metric("Airway Bill Cost", f"${awb_cost:,.2f}")
            elif selected_shipment_type in ["Container", "Truck"]: st.metric(f"{selected_shipment_type} Fixed Price", f"${fixed_logistics_price:,.2f}")
            else: st.write("N/A")

        # --- Total Delivered Cost Tab (Modified for Rebate) ---
        with tab_total:
            st.subheader("Total Delivered Cost (COGS + Logistics + Unexpected + Rebate)") # Updated title
            # Show costs *before* rebate first
            st.metric("Total Delivered Cost (Before Rebate)", f"${total_delivered_cost_usd:,.2f}")
            st.write(f"*Calculation: ${total_cogs_usd:,.2f} (COGS) + ${total_logistics_cost_usd:,.2f} (Logistics) + ${total_unexpected_cost_usd:,.2f} (Unexpected)*")
            st.markdown("---")
            # Show rebate calculation
            st.subheader(f"Retailer Rebate/Fee Adjustment ({rebate_percentage:.1f}%)")
            col_rebate1, col_rebate2 = st.columns(2)
            col_rebate1.metric("Rebate/Fee Amount (Total Batch)", f"${rebate_amount_usd:,.2f}")
            # Show final costs *after* rebate
            col_rebate2.metric("Final Total Cost (Incl. Rebate)", f"${final_total_cost_usd:,.2f}")
            col_rebate2.metric("Final Cost / Box (Incl. Rebate)", f"${final_cost_per_box_usd:,.2f}")
            if total_net_weight_kg > 0:
                 final_cost_per_kg_net_usd = final_total_cost_usd / total_net_weight_kg
                 col_rebate2.metric("Final Cost / KG Net (Incl. Rebate)", f"${final_cost_per_kg_net_usd:,.2f}")
            st.write(f"*Final Calculation: ${total_delivered_cost_usd:,.2f} (Delivered Cost) + ${rebate_amount_usd:,.2f} (Rebate)*")


        # --- Batch Summary Detail Tab (Modified for Rebate) ---
        with tab_summary:
            st.subheader("Total Batch Cost Summary")
            summary_data_dict = st.session_state.get('summary_data', {})
            if summary_data_dict:
                # Pie Chart: Shows breakdown BEFORE rebate for clarity on operational costs
                plot_data = {
                    'Raw Product': summary_data_dict.get("1. Raw Product", 0),
                    'Variable (incl. Pallets)': summary_data_dict.get("2. Variable Costs (incl. Pallets)", 0),
                    'Fixed Costs': summary_data_dict.get(f"3. Fixed Costs {fixed_cost_label_suffix}", 0),
                    'Freight/Fee': summary_data_dict.get(f"4. {selected_shipment_type} Freight/Fee", 0),
                    'Unexpected': summary_data_dict.get("5. Unexpected Costs", 0)
                }
                plot_data_filtered = {k: v for k, v in plot_data.items() if v > 0}
                if plot_data_filtered:
                    plot_df = pd.DataFrame(list(plot_data_filtered.items()), columns=['Cost Category', 'Total Cost (USD)'])
                    fig = px.pie(plot_df, names='Cost Category', values='Total Cost (USD)',
                                 title='Cost Breakdown by Component (Before Rebate)', # Updated title
                                 hole=.3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                else: st.write("No primary cost components > 0 for chart.")

                # Summary Table: Shows the FULL breakdown including rebate and final total
                st.subheader("Full Cost Breakdown Table")
                summary_df = pd.DataFrame(list(summary_data_dict.items()), columns=['Cost Category', 'Total Cost (USD)'])
                st.dataframe(summary_df.style.format({'Total Cost (USD)': '${:,.2f}'}), use_container_width=True)
            else: st.write("Summary data not available (Run calculation first).")


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