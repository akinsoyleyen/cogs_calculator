import pandas as pd
import streamlit as st
import os
from datetime import datetime
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from theme import apply_theme, plot_style
from cogs.exchange import (
    DISPLAY_CURRENCIES,
    CURRENCY_SYMBOLS,
    FALLBACK_USD_RATES,
    get_usd_to_target_rate,
)
from cogs.formatters import (
    format_cost,
    format_cost_by_mode,
    format_cost_usd_only,
    format_cost_eur_only,
)
from cogs.exporters import (
    calculate_profit_margins,
    create_csv_export,
    create_excel_export,
    get_download_link,
)
from cogs.data_loader import (
    COMPONENTS_CSV,
    RECIPE_CSV,
    FIXED_CSV,
    WEIGHTS_CSV,
    AIR_RATES_CSV,
    PALLETS_CSV,
    PACKING_CSV,
    INTEREST_COST_ITEM_NAME,
    load_csv,
)
from cogs.calculator import compute_landed_cost

exchange_rate = 1.0  # TRY rate no longer used; retained as a no-op multiplier

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="Ledger — Cost & Logistics", page_icon="◐")

apply_theme("main")

# --- Sidebar masthead ---
with st.sidebar:
    try:
        st.image("assets/Logo.png", width=96)
    except FileNotFoundError:
        st.error("Logo file not found. Make sure 'Logo.png' is in assets/.")
    except Exception as e:
        st.error(f"An error occurred loading the logo: {e}")
    st.markdown(
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-weight:600;"
        "font-size:1.05rem;color:var(--ink);letter-spacing:-0.02em;margin-top:8px;'>"
        "Ledger<span style='color:var(--pink)'>.</span></div>"
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.62rem;color:var(--ink-muted);"
        "letter-spacing:0.2em;text-transform:uppercase;margin-bottom:18px;'>Cost · Logistics · Margin</div>",
        unsafe_allow_html=True,
    )

# --- Masthead ---
st.markdown(
    "<h1 style='margin:6px 0 6px'>Cost &amp; <em>Logistics</em> Calculator</h1>"
    "<p style='font-size:0.92rem;color:var(--ink-muted);max-width:62ch;"
    "margin:0 0 24px;line-height:1.5;'>"
    "Everything in USD. One FX conversion at the end.</p>",
    unsafe_allow_html=True,
)


def _sidebar_section(label: str) -> None:
    """Mono-uppercase mini-label with a soft divider above. Replaces st.subheader in the sidebar."""
    st.sidebar.markdown(
        "<div style='border-top:1px solid var(--rule);margin:14px 0 10px;'></div>"
        f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        "font-size:0.68rem;letter-spacing:0.16em;text-transform:uppercase;"
        f"color:var(--ink-muted);margin-bottom:6px;'>{label}</div>",
        unsafe_allow_html=True,
    )

# --- Initialize session state ---
if 'calculation_done' not in st.session_state: st.session_state['calculation_done'] = False
if 'final_cost_per_box_usd' not in st.session_state: st.session_state['final_cost_per_box_usd'] = 0.0


# --- Display Currency (single FX at the end) ---
_sidebar_section("Reporting currency")
display_currency = st.sidebar.selectbox(
    "Display results in:",
    DISPLAY_CURRENCIES,
    index=0,
    help="All inputs are in USD. Pick any currency here to convert the final output — one rate, applied at the end."
)
display_symbol = CURRENCY_SYMBOLS.get(display_currency, display_currency + " ")

fx_mode = st.sidebar.radio(
    "FX rate source",
    ("Live (Frankfurter API)", "Manual"),
    key="fx_mode",
    index=0,
    disabled=(display_currency == "USD")
)

if display_currency == "USD":
    display_fx_rate = 1.0
    fx_source = "native"
    fx_date = None
elif fx_mode == "Live (Frankfurter API)":
    live_rate, fx_date = get_usd_to_target_rate(display_currency)
    if live_rate is not None:
        display_fx_rate = live_rate
        fx_source = f"API ({fx_date})"
    else:
        st.sidebar.warning(f"Live rate unavailable for USD→{display_currency}. Enter manually:")
        display_fx_rate = st.sidebar.number_input(
            f"USD→{display_currency} rate",
            min_value=0.000001,
            value=float(FALLBACK_USD_RATES.get(display_currency, 1.0)),
            step=0.0001,
            format="%.6f",
            key=f"manual_fx_fallback_{display_currency}",
        )
        fx_source = "Manual fallback"
else:
    display_fx_rate = st.sidebar.number_input(
        f"USD→{display_currency} rate",
        min_value=0.000001,
        value=float(FALLBACK_USD_RATES.get(display_currency, 1.0)),
        step=0.0001,
        format="%.6f",
        key=f"manual_fx_{display_currency}",
    )
    fx_source = "Manual"

st.sidebar.metric(
    label=f"USD → {display_currency}  ({fx_source})",
    value=f"{display_fx_rate:.4f}" if display_currency != "USD" else "1.0000",
)

if st.sidebar.button("Refresh live FX", help="Clear cache and re-fetch FX rates", disabled=(display_currency == "USD")):
    st.cache_data.clear()
    st.rerun()

# Push display state into session_state so cogs.formatters can read it.
st.session_state["display_currency"] = display_currency
st.session_state["display_fx_rate"] = display_fx_rate
st.session_state["display_symbol"] = display_symbol

# Kept as aliases so legacy code paths still work
currency_display_mode = f"{display_currency} Only"
usd_to_eur_rate = display_fx_rate if display_currency == "EUR" else FALLBACK_USD_RATES["EUR"]


# --- Initialize DataFrames BEFORE loading ---
components_df_try_loaded = None; product_recipe_df = None; fixed_df_usd_loaded = None
product_weights_df = None; air_rates_df = None; pallets_df = None; product_packing_df = None
components_df = None; fixed_df = None

errors = []
try:
    components_df_try_loaded, err = load_csv(COMPONENTS_CSV, ['ComponentName', 'CostPerUnit', 'WeightKG'], ['CostPerUnit', 'WeightKG'])
    if err: errors.append(err)
    product_recipe_df, err = load_csv(RECIPE_CSV, ['ProductID', 'ComponentName', 'QuantityPerProduct'], ['QuantityPerProduct'], string_cols=['ProductID', 'ComponentName'])
    if err: errors.append(err)
    fixed_df_usd_loaded, err = load_csv(FIXED_CSV, ['CostItem', 'MonthlyCost', 'Category'], ['MonthlyCost'], string_cols=['Category','CostItem'])
    if err: errors.append(err)
    product_weights_df, err = load_csv(WEIGHTS_CSV, ['ProductID', 'NetWeightKG'], ['NetWeightKG'], decimal_char=',', string_cols=['ProductID'])
    if err: errors.append(err)
    air_rates_df, err = load_csv(AIR_RATES_CSV, ['Destination', 'MinWeightKG', 'PricePerKG_USD', 'AirwayBill_USD'], ['MinWeightKG', 'PricePerKG_USD', 'AirwayBill_USD'])
    if err: errors.append(err)
    pallets_df, err = load_csv(PALLETS_CSV, ['PalletType', 'CostUSD', 'WeightKG'], ['CostUSD', 'WeightKG'])
    if err: errors.append(err)
    product_packing_df, err = load_csv(PACKING_CSV, ['ProductID', 'BoxesPerPallet'], ['BoxesPerPallet'], string_cols=['ProductID'])
    if err: errors.append(err)

    if product_packing_df is not None:
        product_packing_df['BoxesPerPallet'] = pd.to_numeric(product_packing_df['BoxesPerPallet'], errors='coerce')

    if errors: raise ValueError("Errors occurred during file loading. See details above.")

    components_df = components_df_try_loaded.copy()
    components_df['CostPerUnit_USD'] = components_df['CostPerUnit'] * exchange_rate
    components_df['ComponentName'] = components_df['ComponentName'].astype(str).str.strip()

    fixed_df = fixed_df_usd_loaded.copy()
    fixed_df = fixed_df.rename(columns={'MonthlyCost': 'MonthlyCost_USD'})
    fixed_df['Category'] = fixed_df['Category'].astype(str).str.strip().str.title()
    fixed_df['CostItem'] = fixed_df['CostItem'].astype(str).str.strip()

    valid_categories = ['Primary', 'Secondary']
    invalid_cats_df = fixed_df[~fixed_df['Category'].isin(valid_categories)]
    if not invalid_cats_df.empty:
        invalid_cats_list = invalid_cats_df['Category'].unique().tolist()
        errors.append(f"Invalid Category values in '{FIXED_CSV}': {invalid_cats_list}. Expected 'Primary' or 'Secondary'.")

    if product_weights_df is not None: product_weights_df['ProductID'] = product_weights_df['ProductID'].astype(str)
    else: errors.append(f"'{WEIGHTS_CSV}' failed to load or is empty.")
    if product_recipe_df is not None: product_recipe_df['ProductID'] = product_recipe_df['ProductID'].astype(str)
    else: errors.append(f"'{RECIPE_CSV}' failed to load or is empty.")
    if product_packing_df is not None: product_packing_df['ProductID'] = product_packing_df['ProductID'].astype(str)

    if errors: raise ValueError("Errors occurred during data processing. See details above.")

except Exception as e:
    st.error(" / ".join(errors) if errors else f"An unexpected error occurred during data loading/processing: {e}")
    st.stop()

# --- Get unique lists ---
try:
    product_ids = sorted(product_weights_df['ProductID'].unique().tolist()) # Products must have weight defined
    air_destinations = sorted(air_rates_df['Destination'].unique().tolist()) if air_rates_df is not None else []
    pallet_types = ["None"] + sorted(pallets_df['PalletType'].unique().tolist()) if pallets_df is not None and 'PalletType' in pallets_df else ["None"]
except Exception as e: st.error(f"Could not extract lists: {e}"); product_ids = []; air_destinations = []; pallet_types = ["None"]; st.stop()


# --- Streamlit User Interface ---
_sidebar_section("Product")
if not product_ids: st.sidebar.warning("No products available."); selected_product = None
else: selected_product = st.sidebar.selectbox("Select Product:", product_ids)

# --- Quantity / Pallet Linking ---
_sidebar_section("Quantity & pallets")
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
    calculated_boxes = int(max(1, num_pallets * boxes_per_pallet))
    quantity_input = st.sidebar.number_input( "Quantity (Boxes/Units):", min_value=1, value=calculated_boxes, step=1, disabled=True, help=f"Auto: {num_pallets} pallets * {boxes_per_pallet} boxes/pallet")
    st.sidebar.caption(f"Using {int(boxes_per_pallet)} Boxes/Pallet for {selected_product}")
else:
     quantity_input = st.sidebar.number_input("Quantity (Boxes/Units):", min_value=1, value=100, step=1, disabled=False, help="Enter manually, or check box above to calculate from pallets.")

# Pallet Type Selection
if not pallet_types: st.sidebar.warning("No pallet types loaded."); selected_pallet_type = "None"
else: none_index = pallet_types.index("None") if "None" in pallet_types else 0; selected_pallet_type = st.sidebar.selectbox("Select Pallet Type:", pallet_types, index=none_index)

# Raw Product Cost & Other Costs
_sidebar_section("Costs")

# Raw fruit is purchased in TRY → live-convert to USD for the calc.
_usd_try_rate, _usd_try_date = get_usd_to_target_rate("TRY")
if _usd_try_rate is None:
    _usd_try_rate = float(FALLBACK_USD_RATES.get("TRY", 38.0))
    _usd_try_source = "fallback"
else:
    _usd_try_source = f"live · {_usd_try_date}"

_col_try, _col_usd = st.sidebar.columns(2)
raw_cost_per_kg_try_input = _col_try.number_input(
    "Raw cost (TRY/kg):",
    min_value=0.0,
    value=float(_usd_try_rate),  # ~1 USD worth by default
    step=1.0,
    format="%.2f",
    help="Enter fruit cost in Turkish Lira. Converted to USD live for the calculation.",
)
raw_cost_per_kg_try = raw_cost_per_kg_try_input / _usd_try_rate if _usd_try_rate > 0 else 0.0
_col_usd.number_input(
    "= USD/kg:",
    value=float(raw_cost_per_kg_try),
    disabled=True,
    format="%.4f",
    help="Auto-converted from TRY using the live Frankfurter rate.",
)
st.sidebar.caption(f"USD → TRY: {_usd_try_rate:.4f} ({_usd_try_source})")

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

# Logistics Inputs
_sidebar_section("Logistics")
shipment_types = ["Select...", "Air", "Container", "Truck"]; selected_shipment_type = st.sidebar.selectbox("Select Shipment Type:", shipment_types)
selected_destination = None
if selected_shipment_type == "Air":
    if not air_destinations: st.sidebar.warning("No air destinations loaded.")
    else: selected_destination = st.sidebar.selectbox("Select Destination (Air):", ["Select..."] + air_destinations) # Add select prompt

manual_logistics_cost_usd = 0.0
manual_logistics_cost_input = 0.0  # referenced in logistics-tab display block below
if selected_shipment_type in ["Container", "Truck"]:
    manual_logistics_cost_input = st.sidebar.number_input(
        f"Fixed {selected_shipment_type} Price (USD):",
        min_value=0.0,
        value=4000.0,
        step=50.0,
        format="%.2f",
        help="Enter the flat freight price in USD. The final output is converted to your reporting currency at the end.",
    )
    manual_logistics_cost_usd = manual_logistics_cost_input

_sidebar_section("Sales adjustments")
rebate_rate_input = st.sidebar.number_input(
    "Retailer Rebate/Fee (%):",
    min_value=0.0,
    value=0.0,
    step=0.1,
    format="%.2f",
    help="Enter a percentage (e.g., 5 for 5%) to be added to the Total Delivered Cost.",
)
target_profit_percent = st.sidebar.number_input(
    "Target Profit (%):",
    min_value=0.0,
    value=0.0,
    step=0.5,
    format="%.1f",
    help="Markup over cost shown in the matrix (and email). Sell price = cost × (1 + profit/100). 0 leaves the matrix as raw cost.",
)

with st.sidebar.expander("Advanced", expanded=False):
    interest_rate_percent = st.number_input(
        "Interest Rate (%):",
        min_value=0.0,
        value=5.0,
        step=0.1,
        format="%.2f",
        help="Applied on top of intermediate cost. Persists across pages via session state.",
    )
    unexpected_cost_try = st.number_input(
        "Unexpected Costs (USD for batch):",
        min_value=0.0, value=0.0, step=10.0, format="%.2f",
    )
    include_variable_costs = st.checkbox(
        "Include Variable Component Costs in COGS?", value=True,
    )

interest_rate = interest_rate_percent / 100.0
st.session_state["interest_rate_percent"] = interest_rate_percent


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

    selected_product_str = str(selected_product)
    calc_errors = []
    result = None

    try:
        result = compute_landed_cost(
            selected_product=selected_product_str,
            quantity_boxes=quantity_input,
            selected_pallet_type=selected_pallet_type,
            num_pallets=num_pallets,
            raw_cost_per_kg_usd=raw_cost_per_kg_try,
            include_variable_costs=include_variable_costs,
            fixed_cost_mode=fixed_cost_mode,
            fixed_categories=fixed_categories_to_include,
            interest_rate=interest_rate,
            unexpected_cost_usd=unexpected_cost_try,
            rebate_percentage=rebate_rate_input,
            shipment_type=selected_shipment_type,
            destination=selected_destination,
            manual_logistics_cost_usd=manual_logistics_cost_usd,
            product_weights_df=product_weights_df,
            product_recipe_df=product_recipe_df,
            components_df=components_df,
            pallets_df=pallets_df,
            fixed_df=fixed_df,
            air_rates_df=air_rates_df,
        )
        for w in result.warnings:
            st.warning(w)
    except ValueError as ve: calc_errors.append(f"Data Input Error: {ve}")
    except KeyError as ke: calc_errors.append(f"Data Lookup Error: Missing key {ke}. Check file headers/content.")
    except ZeroDivisionError: calc_errors.append("Calculation Error: Division by zero (check Quantity).")
    except AssertionError as ae: calc_errors.append(f"Data Validation Failed: {ae}")
    except Exception as e_main_calc: calc_errors.append(f"Unexpected calculation error: {e_main_calc}")

    if result is not None and not calc_errors:
        # Unpack result into the local names the display blocks below already use.
        net_weight_kg = result.net_weight_kg_per_box
        calculated_gross_weight_kg_per_box = result.gross_weight_kg_per_box
        final_shipping_gross_weight_kg = result.final_shipping_gross_weight_kg
        total_net_weight_kg = result.total_net_weight_kg
        total_packaging_weight_kg = result.total_packaging_weight_kg

        total_raw_cost_usd = result.total_raw_cost_usd
        raw_cost_per_box_usd = result.raw_cost_per_box_usd
        total_variable_comp_cost_usd = result.total_variable_comp_cost_usd
        total_per_unit_variable_comp_cost_usd = result.total_per_unit_variable_comp_cost_usd
        total_pallet_cost_usd = result.total_pallet_cost_usd
        pallet_cost_per_box_usd = result.pallet_cost_per_box_usd
        total_variable_costs_incl_pallets_usd = result.total_variable_costs_incl_pallets_usd
        variable_costs_incl_pallets_per_box_usd = result.variable_costs_incl_pallets_per_box_usd
        total_allocated_fixed_cost_usd = result.total_allocated_fixed_cost_usd
        fixed_cost_10_percent = result.fixed_cost_10_percent
        fixed_cost_per_unit_usd = result.fixed_cost_per_unit_usd
        interest_cost_usd = result.interest_cost_usd
        total_cogs_usd = result.total_cogs_usd
        cogs_per_box_usd = result.cogs_per_box_usd
        cogs_per_kg_usd = result.cogs_per_kg_usd

        total_logistics_cost_usd = result.total_logistics_cost_usd
        freight_or_fixed_logistics_cost = result.freight_or_fixed_logistics_cost
        logistics_rate_per_kg = result.logistics_rate_per_kg
        awb_cost = result.awb_cost
        fixed_logistics_price = result.fixed_logistics_price
        logistics_per_box_usd = result.logistics_per_box_usd
        logistics_per_kg_gross_usd = result.logistics_per_kg_gross_usd
        logistics_cost_source = result.logistics_cost_source

        total_unexpected_cost_usd = result.total_unexpected_cost_usd
        unexpected_cost_per_box_usd = result.unexpected_cost_per_box_usd

        total_delivered_cost_usd = result.total_delivered_cost_usd
        delivered_cost_per_box_usd = result.delivered_cost_per_box_usd
        delivered_cost_per_kg_net_usd = result.delivered_cost_per_kg_net_usd

        rebate_percentage = result.rebate_percentage
        rebate_amount_usd = result.rebate_amount_usd
        final_total_cost_usd = result.final_total_cost_usd
        final_cost_per_box_usd = result.final_cost_per_box_usd

        weight_per_pallet_kg = result.weight_per_pallet_kg
        product_variable_costs_detailed = result.product_variable_costs_detailed

        st.session_state['profit_analysis'] = {
            'final_cost_per_box_usd': final_cost_per_box_usd,
            'cogs_per_box_usd': cogs_per_box_usd,
            'delivered_cost_per_box_usd': delivered_cost_per_box_usd,
        }

        summary_data = {
            "1. Raw Product": format_cost_by_mode(total_raw_cost_usd, currency_display_mode),
            "2. Variable Costs (incl. Pallets)": format_cost_by_mode(total_variable_costs_incl_pallets_usd, currency_display_mode),
            f"3. Fixed Costs {fixed_cost_label_suffix}": (
                format_cost_by_mode(fixed_cost_10_percent, currency_display_mode)
                if fixed_cost_mode == "percent"
                else format_cost_by_mode(total_allocated_fixed_cost_usd, currency_display_mode)
            ),
            f"   (Calc. Interest @ {interest_rate_percent:.1f}%)": format_cost_by_mode(interest_cost_usd, currency_display_mode),
            "   Subtotal COGS": format_cost_by_mode(total_cogs_usd, currency_display_mode),
            f"4. {selected_shipment_type} Freight/Fee": format_cost_by_mode(freight_or_fixed_logistics_cost, currency_display_mode),
            "   Subtotal Logistics": format_cost_by_mode(total_logistics_cost_usd, currency_display_mode),
            "5. Unexpected Costs": format_cost_by_mode(total_unexpected_cost_usd, currency_display_mode),
            "      Delivered Cost (Before Rebate)": format_cost_by_mode(total_delivered_cost_usd, currency_display_mode),
            f"6. Rebate/Fee ({rebate_percentage:.1f}%)": format_cost_by_mode(rebate_amount_usd, currency_display_mode),
            "Grand Total Cost (incl Rebate)": format_cost_by_mode(final_total_cost_usd, currency_display_mode),
        }

        st.session_state['calculation_done'] = True
        st.session_state['summary_data'] = summary_data
        st.session_state['last_calc_product'] = selected_product_str
        st.session_state['last_calc_quantity'] = quantity_input
        st.session_state['cogs_per_box_usd'] = cogs_per_box_usd
        st.session_state['unexpected_cost_per_box_usd'] = unexpected_cost_per_box_usd
        st.session_state['calculated_gross_weight_kg_per_box'] = calculated_gross_weight_kg_per_box
        st.session_state['air_rates_df'] = air_rates_df
        st.session_state['selected_shipment_type'] = selected_shipment_type
        st.session_state['delivered_cost_per_box_usd'] = delivered_cost_per_box_usd
        st.session_state['final_cost_per_box_usd'] = final_cost_per_box_usd
        st.session_state['last_calc_rebate_rate'] = rebate_percentage
        st.session_state['last_calc_rebate_amount'] = rebate_amount_usd
        st.session_state['selected_pallet_type'] = selected_pallet_type
        st.session_state['boxes_per_pallet'] = boxes_per_pallet if boxes_per_pallet > 0 else 1
        st.session_state['weight_per_pallet_kg'] = weight_per_pallet_kg
        st.session_state['num_pallets'] = num_pallets

    # --- Display Results OR Errors ---
    if calc_errors:
        st.error("Calculation failed. Please resolve errors:")
        for err in calc_errors: st.error(f"- {err}")
        st.session_state['calculation_done'] = False # Ensure flag is false on error
    elif st.session_state.get('calculation_done', False): # Proceed only if calculation was successful
        st.header(f"Calculation Results (in {display_currency})")

        # Update top display line to include rebate % if applicable
        display_destination = f" | Dest: `{selected_destination}`" if selected_shipment_type == "Air" else ""
        display_pallet = f" | Pallet: `{selected_pallet_type}` (x{num_pallets})" if selected_pallet_type not in [None, "None"] else ""
        display_rebate = f" | Rebate: `{rebate_percentage:.1f}%`" if rebate_percentage > 0 else "" # Add rebate display
        st.write(
            f"**Product:** `{selected_product_str}` | **Qty:** `{quantity_input}` | **Ship Via:** `{selected_shipment_type}`"
            f"{display_destination}{display_pallet}{display_rebate} | **Fixed Costs:** `{fixed_cost_selection}`" # Added rebate here
        )

        st.markdown(
            "<dl style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));"
            "gap:18px;margin:8px 0 18px;padding:0;'>"
            "<div><dt style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            "font-size:0.65rem;letter-spacing:0.16em;text-transform:uppercase;"
            "color:var(--ink-muted);margin:0 0 4px;'>Net wt / box</dt>"
            f"<dd style='margin:0;font-size:1.1rem;font-weight:600;color:var(--ink);'>{net_weight_kg:.3f} kg</dd></div>"
            "<div><dt style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            "font-size:0.65rem;letter-spacing:0.16em;text-transform:uppercase;"
            "color:var(--ink-muted);margin:0 0 4px;'>Gross wt / box</dt>"
            f"<dd style='margin:0;font-size:1.1rem;font-weight:600;color:var(--ink);'>{calculated_gross_weight_kg_per_box:.3f} kg</dd></div>"
            "<div><dt style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            "font-size:0.65rem;letter-spacing:0.16em;text-transform:uppercase;"
            "color:var(--ink-muted);margin:0 0 4px;'>Shipping wt (batch)</dt>"
            f"<dd style='margin:0;font-size:1.1rem;font-weight:600;color:var(--ink);'>{final_shipping_gross_weight_kg:.2f} kg "
            f"<span style='font-size:0.78rem;font-weight:500;color:var(--ink-muted);'>· {num_pallets} pallets</span></dd></div>"
            "</dl>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        tab_cost, tab_detail, tab_profit = st.tabs([
            "Cost",
            "Detail",
            "Profit",
        ])

        with tab_cost:
            # Top-line: three stat metrics — COGS / Logistics / Delivered+Rebate
            top_a, top_b, top_c = st.columns(3)
            top_a.metric("Total COGS", format_cost_by_mode(total_cogs_usd, currency_display_mode), help=f"Per box: {format_cost_by_mode(cogs_per_box_usd, currency_display_mode)} · per kg net: {format_cost_by_mode(cogs_per_kg_usd, currency_display_mode)}")
            top_b.metric("Logistics", format_cost_by_mode(total_logistics_cost_usd, currency_display_mode), help=f"{logistics_cost_source} · per box: {format_cost_by_mode(logistics_per_box_usd, currency_display_mode)}")
            top_c.metric("Delivered (incl. rebate)", format_cost_by_mode(final_total_cost_usd, currency_display_mode), help=f"Per box: {format_cost_by_mode(final_cost_per_box_usd, currency_display_mode)}")

            st.markdown("---")

            # COGS subsection
            st.subheader(f"COGS {fixed_cost_label_suffix}")
            col2a, col2b, col2c = st.columns(3)
            col2a.metric("Raw Product", format_cost_by_mode(total_raw_cost_usd, currency_display_mode))
            col2b.metric("Variable (incl. Pallets)", format_cost_by_mode(total_variable_costs_incl_pallets_usd, currency_display_mode), help="Only included if checkbox is checked. Pallet cost always included.")
            col2c.metric(f"Fixed Costs {fixed_cost_label_suffix}", format_cost_by_mode(total_allocated_fixed_cost_usd, currency_display_mode), help=f"Includes Calc. Interest: {format_cost_by_mode(interest_cost_usd, currency_display_mode)}" if interest_cost_usd > 0 else None)
            st.caption(
                f"Per box — Raw: {format_cost_by_mode(raw_cost_per_box_usd, currency_display_mode)}"
                f" · Variable: {format_cost_by_mode(variable_costs_incl_pallets_per_box_usd, currency_display_mode)}"
                f" (Components {format_cost_by_mode(total_per_unit_variable_comp_cost_usd, currency_display_mode)}"
                f" + Pallets {format_cost_by_mode(pallet_cost_per_box_usd, currency_display_mode)})"
                f" · Fixed: {format_cost_by_mode(fixed_cost_per_unit_usd, currency_display_mode)}"
                f" → COGS/box {format_cost_by_mode(cogs_per_box_usd, currency_display_mode)}"
            )

            st.markdown("---")

            # Logistics subsection
            st.subheader(f"{selected_shipment_type} logistics — {logistics_cost_source}")
            if selected_shipment_type == "Air":
                col4a, col4b = st.columns(2)
                freight_cost = final_shipping_gross_weight_kg * logistics_rate_per_kg
                col4a.metric("Freight", format_cost_by_mode(freight_cost, currency_display_mode), f"{final_shipping_gross_weight_kg:.2f} kg @ {format_cost_by_mode(logistics_rate_per_kg, currency_display_mode)}/kg")
                col4b.metric("Airway Bill", format_cost_by_mode(awb_cost, currency_display_mode))
            elif selected_shipment_type in ("Container", "Truck"):
                st.metric(f"{selected_shipment_type} Fixed Price", format_cost(fixed_logistics_price))
            else:
                st.write("N/A")
            if final_shipping_gross_weight_kg > 0:
                st.caption(f"Per box: {format_cost_by_mode(logistics_per_box_usd, currency_display_mode)} · Per kg gross: {format_cost_by_mode(logistics_per_kg_gross_usd, currency_display_mode)}")

            st.markdown("---")

            # Delivered + Rebate subsection
            st.subheader(f"Delivered cost · rebate {rebate_percentage:.1f}%")
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Before rebate", format_cost_by_mode(total_delivered_cost_usd, currency_display_mode))
            col_d2.metric("Rebate amount", format_cost_by_mode(rebate_amount_usd, currency_display_mode))
            col_d3.metric("After rebate", format_cost_by_mode(final_total_cost_usd, currency_display_mode))
            per_box_kg_caption = f"Per box: {format_cost_by_mode(final_cost_per_box_usd, currency_display_mode)}"
            if total_net_weight_kg > 0:
                final_cost_per_kg_net_usd = final_total_cost_usd / total_net_weight_kg
                per_box_kg_caption += f" · Per kg net: {format_cost_by_mode(final_cost_per_kg_net_usd, currency_display_mode)}"
            st.caption(per_box_kg_caption)
            st.caption(
                f"= {format_cost_by_mode(total_cogs_usd, currency_display_mode)} (COGS) "
                f"+ {format_cost_by_mode(total_logistics_cost_usd, currency_display_mode)} (Logistics) "
                f"+ {format_cost_by_mode(total_unexpected_cost_usd, currency_display_mode)} (Unexpected) "
                f"+ {format_cost_by_mode(rebate_amount_usd, currency_display_mode)} (Rebate)"
            )

        # --- Detail tab (former Batch Summary Detail) ---
        with tab_detail:
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
                    _ps = plot_style()
                    fig = px.pie(plot_df, names='Cost Category', values='Total Cost (USD)',
                                 title='Cost breakdown by component (before rebate)',
                                 hole=.58,
                                 color_discrete_sequence=_ps["palette"])
                    fig.update_traces(textposition='outside', textinfo='percent+label',
                                      marker=dict(line=dict(color=_ps["marker_edge"], width=2)))
                    fig.update_layout(
                        paper_bgcolor=_ps["paper_bg"], plot_bgcolor=_ps["plot_bg"],
                        font=dict(family="Space Grotesk, sans-serif", color=_ps["font_color"], size=13),
                        title=dict(font=dict(family="Space Grotesk, sans-serif", size=16, color=_ps["title_color"]), x=0, xanchor="left"),
                        legend=dict(font=dict(family="JetBrains Mono, monospace", size=11, color=_ps["font_color"])),
                        margin=dict(l=10, r=10, t=60, b=10)
                    )
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
                    _ps = plot_style()
                    _line_palette = _ps["line_palette"]
                    def _coerce_num(s):
                        return float(str(s).replace('$','').replace(',','').replace('€','').replace('£','').replace('₺','').replace('%','').replace('C$','').replace('A$','').replace('S$','').replace('CHF ','').replace('¥','').replace('د.إ ','').strip())
                    fig.add_trace(go.Scatter(x=sensitivity_df['Markup %'], y=sensitivity_df['Profit per Box'].apply(_coerce_num), mode='lines+markers', name='Profit per Box', line=dict(color=_line_palette[0], width=2.2), marker=dict(size=7)))
                    fig.add_trace(go.Scatter(x=sensitivity_df['Markup %'], y=sensitivity_df['Profit Margin %'].apply(_coerce_num), mode='lines+markers', name='Profit Margin %', line=dict(color=_line_palette[1], width=2.2), marker=dict(size=7)))
                    fig.add_trace(go.Scatter(x=sensitivity_df['Markup %'], y=sensitivity_df['ROI %'].apply(_coerce_num), mode='lines+markers', name='ROI %', line=dict(color=_line_palette[2], width=2.2), marker=dict(size=7)))
                    fig.update_layout(
                        title=dict(text='Cost sensitivity', font=dict(family="Space Grotesk, sans-serif", size=16, color=_ps["title_color"]), x=0, xanchor="left"),
                        xaxis=dict(title='Markup %', gridcolor=_ps["grid_color"], zerolinecolor=_ps["grid_color"], linecolor=_ps["axis_line"]),
                        yaxis=dict(title='Value', gridcolor=_ps["grid_color"], zerolinecolor=_ps["grid_color"], linecolor=_ps["axis_line"]),
                        paper_bgcolor=_ps["paper_bg"], plot_bgcolor=_ps["plot_bg"],
                        font=dict(family="Space Grotesk, sans-serif", color=_ps["font_color"], size=13),
                        legend=dict(font=dict(size=11, color=_ps["font_color"])), margin=dict(l=20, r=20, t=60, b=40)
                    )
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

# --- Pricing matrix (for pasting into emails) ---
st.markdown("---")
_matrix_is_sell = target_profit_percent > 0
_matrix_title_word = "Sell-price" if _matrix_is_sell else "Cost"
st.subheader(f"{'💰' if _matrix_is_sell else '📋'} {_matrix_title_word} matrix — air destinations × pallet count")

matrix_product_id = str(
    st.session_state.get('last_calc_product') or selected_product or ""
).strip()

if st.session_state.get('calculation_done', False) and selected_shipment_type == "Air" and matrix_product_id:
    if _matrix_is_sell:
        st.caption(
            f"Sell price per box (USD) = cost × (1 + {target_profit_percent:.1f}%). "
            "Other inputs (raw cost, rebate, fixed-cost mode) follow the sidebar."
        )
    else:
        st.caption(
            "Cost per box (USD), recomputed for every air destination at 2 / 4 / 6 / 10 pallets. "
            "Set **Target Profit** in the sidebar to switch this to sell prices."
        )
    try:
        from cogs.matrix import PALLET_COUNTS, build_air_matrix

        base_inputs = {
            "selected_product": matrix_product_id,
            "selected_pallet_type": selected_pallet_type,
            "raw_cost_per_kg_usd": raw_cost_per_kg_try,
            "include_variable_costs": include_variable_costs,
            "fixed_cost_mode": fixed_cost_mode,
            "fixed_categories": fixed_categories_to_include,
            "interest_rate": interest_rate,
            "unexpected_cost_usd": unexpected_cost_try,
            "rebate_percentage": rebate_rate_input,
        }
        dfs = {
            "product_weights_df": product_weights_df,
            "product_recipe_df": product_recipe_df,
            "components_df": components_df,
            "pallets_df": pallets_df,
            "fixed_df": fixed_df,
            "air_rates_df": air_rates_df,
            "product_packing_df": product_packing_df,
        }
        cost_matrix_df = build_air_matrix(base_inputs=base_inputs, dfs=dfs)
        profit_multiplier = 1.0 + (target_profit_percent / 100.0)
        sell_matrix_df = (cost_matrix_df * profit_multiplier).round(2)
        active_matrix_df = sell_matrix_df if _matrix_is_sell else cost_matrix_df

        # Look up boxes_per_pallet for the carton sub-header
        _packing_lookup = product_packing_df[product_packing_df["ProductID"] == matrix_product_id]
        _bpp = int(_packing_lookup["BoxesPerPallet"].iloc[0]) if not _packing_lookup.empty else 0

        # Two-row column header: top "N pallets", bottom "X cartons"
        display_matrix = active_matrix_df.copy()
        if _bpp > 0:
            display_matrix.columns = pd.MultiIndex.from_tuples(
                [(f"{int(c)} pallets", f"{int(c) * _bpp} cartons") for c in display_matrix.columns]
            )
        else:
            display_matrix.columns = [f"{int(c)} pallets" for c in display_matrix.columns]

        st.dataframe(
            display_matrix.style.format("${:,.2f}"),
            use_container_width=True,
        )

        with st.expander("Copy for email (tab-separated)", expanded=False):
            st.caption(
                f"Product: **{matrix_product_id}** · rebate **{rebate_rate_input:.1f}%**"
                f"{f' · target profit **{target_profit_percent:.1f}%**' if _matrix_is_sell else ''}"
                f" · raw **${raw_cost_per_kg_try:.3f}/kg** · "
                f"{datetime.now().strftime('%Y-%m-%d')}"
            )
            # Flatten the multi-index for clean TSV (single header row keeps Gmail paste tidy).
            tsv_matrix = active_matrix_df.copy()
            tsv_matrix.columns = (
                [f"{int(c)} pallets / {int(c) * _bpp} cartons" if _bpp > 0
                 else f"{int(c)} pallets"
                 for c in tsv_matrix.columns]
            )
            tsv = tsv_matrix.reset_index().to_csv(sep="\t", index=False, float_format="%.2f")
            st.code(tsv, language="text")
            st.caption(
                "Click the copy icon at the top-right of the block above. "
                "Pasting into Gmail / Apple Mail / Outlook compose turns it into a real table."
            )

        # --- Send via Make.com webhook ---
        st.markdown("---")
        st.subheader("✉️ Send pricing email (via Make.com)")
        import requests
        from cogs.matrix import render_matrix_html

        webhook_url = st.secrets.get("make_webhook_url", "") if hasattr(st, "secrets") else ""
        if not webhook_url:
            st.info(
                "Add `make_webhook_url = \"https://hook.eu2.make.com/...\"` to "
                "`.streamlit/secrets.toml` to enable. See README for the Make scenario setup."
            )
        else:
            col_to, col_subj = st.columns([1, 1])
            recipient_email = col_to.text_input(
                "Recipient email",
                value=st.session_state.get("last_recipient_email", ""),
                placeholder="buyer@example.com",
            )
            default_subject = f"{matrix_product_id} — pricing matrix ({datetime.now():%Y-%m-%d})"
            subject = col_subj.text_input("Subject", value=default_subject)
            note = st.text_area(
                "Note (prepended to email body)",
                value="",
                placeholder="Optional. e.g. 'As discussed, here is the latest pricing.'",
                height=80,
            )
            if st.button("Send to Make"):
                if not recipient_email or "@" not in recipient_email:
                    st.error("Enter a valid recipient email.")
                else:
                    def _rows(m):
                        return [
                            {
                                "destination": dest,
                                **{
                                    f"p{int(p)}": round(float(m.at[dest, p]), 2)
                                    for p in m.columns
                                },
                                **(
                                    {
                                        f"cartons_p{int(p)}": int(p) * _bpp
                                        for p in m.columns
                                    }
                                    if _bpp > 0
                                    else {}
                                ),
                            }
                            for dest in m.index
                        ]

                    # Flatten the multi-index header for the email HTML (single-row label).
                    email_matrix = active_matrix_df.copy()
                    email_matrix.columns = (
                        [f"{int(c)} pallets<br><span style=\"color:#6c6478;font-weight:400;\">"
                         f"{int(c) * _bpp} cartons</span>"
                         if _bpp > 0 else f"{int(c)} pallets"
                         for c in email_matrix.columns]
                    )

                    payload = {
                        "from_app": "cogs_calculator",
                        "date_iso": datetime.now().isoformat(timespec="seconds"),
                        "product": matrix_product_id,
                        "raw_cost_per_kg_usd": round(raw_cost_per_kg_try, 4),
                        "rebate_percentage": round(rebate_rate_input, 2),
                        "target_profit_percent": round(target_profit_percent, 2),
                        "matrix_kind": "sell_price" if _matrix_is_sell else "cost",
                        "boxes_per_pallet": _bpp,
                        "fixed_cost_mode": fixed_cost_selection,
                        "reporting_currency": display_currency,
                        "to_email": recipient_email,
                        "subject": subject,
                        "note": note,
                        "matrix_html": render_matrix_html(email_matrix),
                        "matrix_rows": _rows(active_matrix_df),
                        "cost_matrix_rows": _rows(cost_matrix_df),
                        "sell_matrix_rows": _rows(sell_matrix_df),
                    }
                    try:
                        with st.spinner("Sending to Make…"):
                            resp = requests.post(webhook_url, json=payload, timeout=15)
                        if resp.ok:
                            st.success(f"Sent. Make returned {resp.status_code}.")
                            st.session_state["last_recipient_email"] = recipient_email
                        else:
                            st.error(
                                f"Make returned HTTP {resp.status_code}: "
                                f"{resp.text[:200]}"
                            )
                    except requests.RequestException as e:
                        st.error(f"Request failed: {e}")
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Could not build matrix: {e}")
elif st.session_state.get('calculation_done', False):
    st.info("Matrix is for Air shipments only. Switch shipment type to Air and recalculate.")
else:
    st.info("Run a calculation first to see the matrix.")

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
if not st.session_state.get('calculation_done', False):
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
