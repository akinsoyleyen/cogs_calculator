import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("💰 Profit Calculator")

# Check if calculation results exist in session state
if 'delivered_cost_per_box_usd' not in st.session_state:
    st.warning("Please run a cost calculation on the main 'Cost Calculator' page first.")
    st.stop()

# Retrieve data from session state
product = st.session_state.get('last_calc_product', 'N/A')
quantity = st.session_state.get('last_calc_quantity', 0)
cost_per_box = st.session_state.get('delivered_cost_per_box_usd', 0.0)
summary_data = st.session_state.get('summary_data', {})

st.header(f"Profit Calculation for: `{product}`")
st.write(f"Based on a batch quantity of `{quantity}` boxes/units.")
st.metric("Calculated Delivered Cost per Box", f"${cost_per_box:,.3f}")
st.markdown("---")

# Input for Sales Price
sales_price_per_box = st.number_input(
    "Enter Sales Price per Box (USD):",
    min_value=0.0,
    value=cost_per_box * 1.1, # Default to 10% markup
    step=0.01,
    format="%.2f"
)

# Calculate Profit
if sales_price_per_box > 0 and quantity > 0:
    profit_per_box = sales_price_per_box - cost_per_box
    total_profit = profit_per_box * quantity
    profit_margin = (profit_per_box / sales_price_per_box) * 100 if sales_price_per_box > 0 else 0

    st.subheader("Profit Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Profit per Box (USD)", f"${profit_per_box:,.2f}")
    col2.metric("Total Profit for Batch (USD)", f"${total_profit:,.2f}")
    col3.metric("Profit Margin (%)", f"{profit_margin:,.1f}%")

    st.markdown("---")
    st.subheader("Cost Summary Used (Total Batch Costs)")
    if summary_data:
        # Convert summary dictionary to DataFrame for display
        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Cost Category', 'Total Cost (USD)'])
        st.dataframe(summary_df.style.format({'Total Cost (USD)': '${:,.2f}'}), use_container_width=True)
    else:
        st.write("Cost summary data not available.")

elif quantity == 0:
     st.warning("Quantity from calculation was zero.")
else:
     st.info("Enter a Sales Price greater than $0.00 to calculate profit.")