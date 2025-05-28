import streamlit as st
import pandas as pd
import requests
from datetime import date
import math  # Add if not present

st.set_page_config(layout="wide", page_title="Batch Sales Calculator")
st.title("📊 Batch Sales & Profit Projection")


# Check if calculation results exist in session state
required_keys = [
    'calculation_done',           # Add this key
    'last_calc_product',
    'last_calc_quantity',
    'cogs_per_box_usd',
    'unexpected_cost_per_box_usd',
    'calculated_gross_weight_kg_per_box',
    'air_rates_df',
    'selected_shipment_type',
    'final_cost_per_box_usd',    # This is the correct key name
    'selected_pallet_type',      # Add this key
    'boxes_per_pallet',          # Add this key
    'weight_per_pallet_kg',      # Add this key
    'num_pallets'                # Add this key
]

# Check if calculation was done and all required keys exist
if not st.session_state.get('calculation_done', False) or not all(key in st.session_state for key in required_keys):
    st.warning("⬅️ Please run a cost calculation on the main 'Cost Calculator' page first.")
    st.stop()

# Retrieve data from session state
product = st.session_state['last_calc_product']
original_quantity = st.session_state['last_calc_quantity']
cogs_per_box_usd = st.session_state['cogs_per_box_usd']
unexpected_cost_per_box_usd = st.session_state['unexpected_cost_per_box_usd']
gross_weight_kg_per_box = st.session_state['calculated_gross_weight_kg_per_box']
air_rates_df = st.session_state['air_rates_df']
original_shipment_type = st.session_state['selected_shipment_type']
original_delivered_cost_per_box = st.session_state['final_cost_per_box_usd'] # Cost for original C/T run including rebate
summary_data = st.session_state.get('summary_data', {}) # Get summary data too

selected_pallet_type = st.session_state['selected_pallet_type']
boxes_per_pallet = st.session_state['boxes_per_pallet']
weight_per_pallet_kg = st.session_state['weight_per_pallet_kg']
num_pallets = st.session_state['num_pallets']

st.header(f"Projections for: `{product}`")
st.write(f"Using base COGS/Unexpected Costs calculated from a run with batch size: `{original_quantity}`.")
st.metric("Base COGS per Box (incl. Pallets, Interest)", f"${cogs_per_box_usd:,.3f}")
st.metric("Original Delivered Cost per Box", f"${original_delivered_cost_per_box:,.3f}")
st.metric("Base Unexpected Cost per Box", f"${unexpected_cost_per_box_usd:,.3f}")
st.metric("Pallet Configuration", 
    f"{selected_pallet_type}: {boxes_per_pallet} boxes/pallet, {weight_per_pallet_kg:.1f}kg/pallet")
st.caption("Note: Total costs & profits below vary by destination for Air. Container/Truck uses the fixed cost from the main page run. Approximations apply.")
st.markdown("---")

# --- Inputs ---
profit_markup_percentage = st.number_input(
    "Desired Profit Markup (%) on Delivered Cost",
    min_value=0.0, max_value=1000.0, value=10.0, step=1.0, format="%.1f"
)

batch_quantities = [150, 300, 450, 600, 1000]

# --- Calculations ---
if profit_markup_percentage is not None:
    all_results = []

    # --- Air Freight Calculations (per destination) ---
    st.subheader("Projections for Air Freight (by Destination)")
    if air_rates_df is not None and not air_rates_df.empty:
        air_destinations = sorted(air_rates_df['Destination'].unique().tolist())

        for destination in air_destinations:
            dest_results = {
                "Destination": destination,
                "Product": product,
                "Pallet Type": selected_pallet_type,
                "Boxes per Pallet": boxes_per_pallet
            }

            for qty in batch_quantities:
                # Calculate number of pallets needed for this quantity
                num_pallets_needed = -(-qty // boxes_per_pallet)  # ceiling division
                
                # Calculate total gross weight including pallet weight
                total_pallet_weight = num_pallets_needed * weight_per_pallet_kg
                current_total_gross_weight = (gross_weight_kg_per_box * qty) + total_pallet_weight
                
                # Rest of the air freight calculations...
                logistics_rate_per_kg = 0.0
                awb_cost = 0.0
                current_logistics_cost = 0.0
                rate_found = False

                applicable_rates = air_rates_df[
                    (air_rates_df['Destination'] == destination) &
                    (air_rates_df['MinWeightKG'] <= current_total_gross_weight)
                ].sort_values('MinWeightKG', ascending=False)

                if not applicable_rates.empty:
                    rate_row = applicable_rates.iloc[0]
                    logistics_rate_per_kg = rate_row['PricePerKG_USD']
                    awb_cost = rate_row['AirwayBill_USD']
                    current_logistics_cost = (current_total_gross_weight * logistics_rate_per_kg) + awb_cost
                    rate_found = True

                # Calculate costs for this qty/destination
                current_cogs_total = cogs_per_box_usd * qty
                current_unexpected_total = unexpected_cost_per_box_usd * qty
                current_delivered_cost_total = current_cogs_total + current_logistics_cost + current_unexpected_total
                current_delivered_cost_per_box = current_delivered_cost_total / qty if qty > 0 else 0

                # Calculate Sales and Profit based on THIS specific delivered cost
                current_sales_price_per_box = current_delivered_cost_per_box * (1 + profit_markup_percentage / 100.0)
                current_total_invoice = current_sales_price_per_box * qty
                current_total_profit = current_total_invoice - current_delivered_cost_total

                # Store results for this quantity
                dest_results[f"{qty} Boxes - Sales Price/Box"] = current_sales_price_per_box

            all_results.append(dest_results)

            #Date time for new Column
            dest_results["CalculationDate"] = date.today().isoformat()

        if all_results:
            results_df_air = pd.DataFrame(all_results).set_index("Product")
            
            # Apply formatting only to numeric columns
            numeric_columns = results_df_air.select_dtypes(include=["number"]).columns
            styled_df = results_df_air.style.format(
                {col: "${:,.2f}" for col in numeric_columns}, na_rep="-"
            )
            
            # Display the styled DataFrame
            st.dataframe(styled_df, use_container_width=True)

            # 1) Render the button and capture its click state
            push = st.button("Push to Airtable")

            # 2) Only run this block if the user actually clicked
            if push:
                # Prepare payload
                records = results_df_air.reset_index().to_dict(orient="records")
                payload = {"records": records}

                # Your Make.com webhook URL
                webhook_url = "https://hook.eu2.make.com/b1rgrihpz73sujq8qsn4gthgr9blr0or"

                resp = requests.post(webhook_url, json=payload)
                if resp.ok:
                    st.success("✅ Projections sent to Airtable!")
                else:
                    st.error(f"⚠️ Failed ({resp.status_code}): {resp.text}")
        else:
            st.write("No destinations found in Air Rates file.")
    else:
        st.warning("Air rates data was not loaded successfully. Cannot generate Air projections.")

    # --- Container / Truck Calculation (Fixed cost per box from main run) ---
    st.subheader(f"Projections for Container / Truck")
    if original_shipment_type in ["Container", "Truck"]:
        st.write(f"Using the fixed Delivered Cost per Box calculated on the main page for '{original_shipment_type}': ${original_delivered_cost_per_box:,.3f}")

        ct_sales_price = original_delivered_cost_per_box * (1 + profit_markup_percentage / 100.0)
        ct_profit_per_box = ct_sales_price - original_delivered_cost_per_box
        ct_results = []
        for qty in batch_quantities:
            ct_results.append({
                "Batch Quantity": qty,
                "Sales Price / Box (USD)": ct_sales_price,
                "Total Invoice Value (USD)": ct_sales_price * qty,
                "Approx. Total Cost (USD)": original_delivered_cost_per_box * qty,
                "Approx. Total Profit (USD)": ct_profit_per_box * qty
            })
        results_df_ct = pd.DataFrame(ct_results)
        st.dataframe(results_df_ct.style.format(subset=["Sales Price / Box (USD)", "Total Invoice Value (USD)", "Approx. Total Cost (USD)", "Approx. Total Profit (USD)"], formatter='${:,.2f}'), use_container_width=True)
        st.metric("Calculated Sales Price per Box (C/T)", f"${ct_sales_price:,.2f}", f"{profit_markup_percentage:.1f}% Markup")
    else:
        st.info("Container/Truck projections shown when Container or Truck is selected and calculated on the main page.")

else:
    st.info("Enter a profit markup percentage.")

# Display the cost summary from the main calculation for reference
st.markdown("---")
st.subheader("Cost Summary Used (Total Batch Costs from Main Calculator run)")
if summary_data:
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['Cost Category', 'Total Cost (USD)'])
    st.dataframe(summary_df.style.format({'Total Cost (USD)': '${:,.2f}'}), use_container_width=True)
else:
    st.write("Cost summary data not available from previous calculation.")