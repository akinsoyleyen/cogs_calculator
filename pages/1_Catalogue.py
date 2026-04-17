# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st

COMPONENTS_CSV = "components.csv"
RECIPE_CSV = "product_recipe.csv"
WEIGHTS_CSV = "product_weights.csv"
PACKING_CSV = "product_packing.csv"
PALLETS_CSV = "pallets.csv"
FIXED_CSV = "fixed_costs.csv"
AIR_RATES_CSV = "air_freight_rates.csv"

st.set_page_config(page_title="Catalogue — Ledger", layout="wide", page_icon="◐")

# --- Shared styling (matches Main.py) ---
_SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..700&family=Public+Sans:ital,wght@0,300..700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --ink: oklch(0.22 0.012 80);
  --ink-soft: oklch(0.38 0.012 80);
  --ink-muted: oklch(0.52 0.012 80);
  --paper: oklch(0.975 0.008 85);
  --paper-2: oklch(0.955 0.010 85);
  --rule: oklch(0.88 0.012 85);
  --olive: oklch(0.42 0.08 140);
  --olive-ink: oklch(0.30 0.06 140);
  --olive-wash: oklch(0.93 0.035 140);
  --claret: oklch(0.48 0.14 25);
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper) !important;
  color: var(--ink);
  font-family: "Public Sans", ui-sans-serif, system-ui, sans-serif;
}
#MainMenu, footer { visibility: hidden; }
h1 { font-family: "Bricolage Grotesque", sans-serif; font-variation-settings: "wght" 500, "wdth" 85, "opsz" 64; font-size: clamp(2rem, 3.2vw, 2.8rem); letter-spacing: -0.025em; line-height: 1.02; color: var(--ink); margin: 12px 0 4px; }
h2 { font-family: "Bricolage Grotesque", sans-serif; font-variation-settings: "wght" 520, "wdth" 92; font-size: 1.35rem; letter-spacing: -0.015em; color: var(--ink); margin-top: 32px; }
h3 { font-family: "Bricolage Grotesque", sans-serif; font-variation-settings: "wght" 550, "wdth" 100; font-size: 1.05rem; color: var(--ink); }
label, [data-testid="stWidgetLabel"] p { font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-soft) !important; }
[data-testid="stSidebar"] { background: var(--paper-2); border-right: 1px solid var(--rule); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 32px 0; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  font-family: "Public Sans", sans-serif; font-weight: 500; font-size: 0.88rem; letter-spacing: 0.02em;
  border-radius: 2px; border: 1px solid var(--ink); background: var(--ink); color: var(--paper);
  padding: 10px 18px; transition: background 160ms ease, transform 160ms ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background: var(--olive); border-color: var(--olive); transform: translateY(-1px);
}
input, textarea, [data-baseweb="input"] input { font-family: "Public Sans", sans-serif !important; background: var(--paper) !important; border-radius: 2px !important; color: var(--ink) !important; }
[data-testid="stNumberInput"] input { font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum"; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] {
  font-family: "Bricolage Grotesque", sans-serif; font-variation-settings: "wght" 520, "wdth" 92;
  font-size: 0.92rem; color: var(--ink-muted); background: transparent; padding: 12px 20px;
  border-radius: 0; border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom: 2px solid var(--olive) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--rule); border-radius: 2px; }
[data-testid="stAlert"] { border-radius: 2px; border: 1px solid var(--rule); background: var(--paper); }
[data-testid="stAlert"][kind="success"] { background: var(--olive-wash); }
[data-testid="stMetric"] {
  background: var(--paper); border: 1px solid var(--rule); border-radius: 2px;
  padding: 16px 24px;
}
[data-testid="stMetricLabel"] p { font-size: 0.7rem !important; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-muted) !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum"; font-weight: 500 !important; font-size: 1.45rem !important; color: var(--ink) !important; }
</style>
"""
if hasattr(st, "html"):
    st.html(_SHARED_CSS)
else:
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)


# --- Helpers ---
def load_csv(path, required_cols, decimal_char="."):
    if not os.path.exists(path):
        st.error(f"Missing required file: `{path}`")
        st.stop()
    df = pd.read_csv(path, decimal=decimal_char)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"`{path}` is missing required columns: {missing}")
        st.stop()
    return df


def save_csv(df, path, decimal_char="."):
    df.to_csv(path, index=False, decimal=decimal_char)


def clean_str_col(series):
    return series.astype(str).fillna("").str.strip()


def dedupe_blank_rows(df, key_col):
    df = df.copy()
    df[key_col] = clean_str_col(df[key_col])
    df = df[df[key_col] != ""]
    return df


# --- Masthead ---
st.markdown(
    "<div style='font-family:\"Public Sans\",sans-serif;font-size:0.72rem;color:var(--ink-muted);"
    "letter-spacing:0.18em;text-transform:uppercase;margin-bottom:-4px;'>"
    "Catalogue &middot; Costs &middot; Logistics</div>",
    unsafe_allow_html=True,
)
st.title("Catalogue")
st.markdown(
    "<p style='font-family:\"Public Sans\",sans-serif;font-size:1rem;color:var(--ink-soft);"
    "max-width:72ch;margin-top:-2px;margin-bottom:20px;line-height:1.55;'>"
    "Every cost, count and coefficient the calculator reads — in one place. "
    "Edit inline, add new rows, delete old ones, then save.</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Products", "Components", "Pallets", "Fixed Costs", "Air Freight"])


# =========================================================================
# PRODUCTS — merged view of weights + packing, with per-product recipe editor
# =========================================================================
with tabs[0]:
    weights_df = load_csv(WEIGHTS_CSV, ["ProductID", "NetWeightKG"], decimal_char=",")
    packing_df = load_csv(PACKING_CSV, ["ProductID", "BoxesPerPallet"])
    recipe_df = load_csv(RECIPE_CSV, ["ProductID", "ComponentName", "QuantityPerProduct"])
    components_df = load_csv(COMPONENTS_CSV, ["ComponentName", "CostPerUnit", "WeightKG"])

    component_options = sorted(clean_str_col(components_df["ComponentName"]).unique().tolist())

    weights_df["ProductID"] = clean_str_col(weights_df["ProductID"])
    packing_df["ProductID"] = clean_str_col(packing_df["ProductID"])
    recipe_df["ProductID"] = clean_str_col(recipe_df["ProductID"])
    recipe_df["ComponentName"] = clean_str_col(recipe_df["ComponentName"])

    merged = weights_df.merge(packing_df, on="ProductID", how="outer")
    merged = merged[["ProductID", "NetWeightKG", "BoxesPerPallet"]]

    col_top_a, col_top_b, col_top_c = st.columns([1, 1, 1])
    col_top_a.metric("Products", f"{merged['ProductID'].nunique()}")
    col_top_b.metric("Components in use", f"{recipe_df['ComponentName'].nunique()}")
    col_top_c.metric("Recipe lines", f"{len(recipe_df)}")

    st.subheader("Products master list")
    st.caption("Edit cells directly. Use the + row at the bottom to add, or tick the trash icon to remove.")
    edited_products = st.data_editor(
        merged,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ProductID": st.column_config.TextColumn("Product ID", required=True, width="large"),
            "NetWeightKG": st.column_config.NumberColumn(
                "Net weight (kg)", min_value=0.0, step=0.1, format="%.3f", required=True
            ),
            "BoxesPerPallet": st.column_config.NumberColumn(
                "Boxes per pallet", min_value=1, step=1, format="%d", required=True
            ),
        },
        key="products_editor",
    )

    if st.button("Save products", key="save_products"):
        cleaned = dedupe_blank_rows(edited_products, "ProductID")
        dupes = cleaned["ProductID"][cleaned["ProductID"].duplicated()].unique().tolist()
        if dupes:
            st.error(f"Duplicate Product IDs: {', '.join(dupes)}")
        elif cleaned["NetWeightKG"].isnull().any() or (cleaned["NetWeightKG"] <= 0).any():
            st.error("Every product needs a Net weight greater than 0.")
        elif cleaned["BoxesPerPallet"].isnull().any() or (cleaned["BoxesPerPallet"] < 1).any():
            st.error("Every product needs Boxes per pallet ≥ 1.")
        else:
            try:
                weights_out = cleaned[["ProductID", "NetWeightKG"]].copy()
                packing_out = cleaned[["ProductID", "BoxesPerPallet"]].copy()
                packing_out["BoxesPerPallet"] = packing_out["BoxesPerPallet"].astype(int)
                save_csv(weights_out, WEIGHTS_CSV, decimal_char=",")
                save_csv(packing_out, PACKING_CSV)

                kept_ids = set(cleaned["ProductID"].tolist())
                recipe_cleaned = recipe_df[recipe_df["ProductID"].isin(kept_ids)].copy()
                save_csv(recipe_cleaned, RECIPE_CSV)

                st.success(
                    f"Saved {len(cleaned)} products. "
                    f"Recipes for removed products were dropped."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write CSVs: {e}")

    st.markdown("---")
    st.subheader("Recipe editor")
    product_ids = sorted(
        [p for p in clean_str_col(edited_products["ProductID"]).unique().tolist() if p]
    )
    if not product_ids:
        st.info("Add a product above first.")
    elif not component_options:
        st.warning("Add components in the Components tab before assigning recipes.")
    else:
        sel = st.selectbox("Select product", product_ids, key="recipe_product")
        product_recipe = recipe_df[recipe_df["ProductID"] == sel][
            ["ComponentName", "QuantityPerProduct"]
        ].reset_index(drop=True)

        if product_recipe.empty:
            product_recipe = pd.DataFrame(
                {"ComponentName": [component_options[0]], "QuantityPerProduct": [1.0]}
            )

        edited_recipe = st.data_editor(
            product_recipe,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "ComponentName": st.column_config.SelectboxColumn(
                    "Component", options=component_options, required=True
                ),
                "QuantityPerProduct": st.column_config.NumberColumn(
                    "Qty per product", min_value=0.0, step=0.1, format="%.3f", required=True
                ),
            },
            key=f"recipe_editor_{sel}",
        )

        if st.button(f"Save recipe for '{sel}'", key=f"save_recipe_{sel}"):
            clean_recipe = edited_recipe.dropna(how="all").copy()
            clean_recipe["ComponentName"] = clean_str_col(clean_recipe["ComponentName"])
            clean_recipe = clean_recipe[clean_recipe["ComponentName"] != ""]

            if clean_recipe.empty:
                st.error("Recipe must contain at least one component.")
            elif (clean_recipe["QuantityPerProduct"].fillna(0) <= 0).any():
                st.error("All recipe quantities must be greater than 0.")
            else:
                try:
                    others = recipe_df[recipe_df["ProductID"] != sel].copy()
                    clean_recipe["ProductID"] = sel
                    clean_recipe = clean_recipe[
                        ["ProductID", "ComponentName", "QuantityPerProduct"]
                    ]
                    out = pd.concat([others, clean_recipe], ignore_index=True)
                    save_csv(out, RECIPE_CSV)
                    st.success(f"Recipe for '{sel}' saved ({len(clean_recipe)} line(s)).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to write `{RECIPE_CSV}`: {e}")


# =========================================================================
# COMPONENTS
# =========================================================================
with tabs[1]:
    components_df = load_csv(COMPONENTS_CSV, ["ComponentName", "CostPerUnit", "WeightKG"])
    if "ComponentType" not in components_df.columns:
        components_df["ComponentType"] = ""
    components_df = components_df[["ComponentName", "ComponentType", "CostPerUnit", "WeightKG"]]

    st.subheader("Components")
    st.caption("Unit cost is in Turkish Lira — the calculator converts to USD at run time.")
    edited_components = st.data_editor(
        components_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ComponentName": st.column_config.TextColumn("Component name", required=True, width="large"),
            "ComponentType": st.column_config.TextColumn("Type", width="small"),
            "CostPerUnit": st.column_config.NumberColumn(
                "Cost / unit (TRY)", min_value=0.0, step=0.01, format="%.4f", required=True
            ),
            "WeightKG": st.column_config.NumberColumn(
                "Weight (kg)", min_value=0.0, step=0.01, format="%.4f", required=True
            ),
        },
        key="components_editor",
    )

    if st.button("Save components", key="save_components"):
        cleaned = dedupe_blank_rows(edited_components, "ComponentName")
        dupes = cleaned["ComponentName"][cleaned["ComponentName"].duplicated()].unique().tolist()
        if dupes:
            st.error(f"Duplicate component names: {', '.join(dupes)}")
        elif cleaned["CostPerUnit"].isnull().any() or cleaned["WeightKG"].isnull().any():
            st.error("Cost and weight are required on every row.")
        else:
            try:
                save_csv(cleaned, COMPONENTS_CSV)
                st.success(f"Saved {len(cleaned)} components.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write `{COMPONENTS_CSV}`: {e}")


# =========================================================================
# PALLETS
# =========================================================================
with tabs[2]:
    pallets_df = load_csv(PALLETS_CSV, ["PalletType", "CostUSD", "WeightKG"])

    st.subheader("Pallets")
    st.caption("Unit cost and tare weight per pallet type.")
    edited_pallets = st.data_editor(
        pallets_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "PalletType": st.column_config.TextColumn("Pallet type", required=True, width="large"),
            "CostUSD": st.column_config.NumberColumn(
                "Cost (USD)", min_value=0.0, step=0.5, format="%.2f", required=True
            ),
            "WeightKG": st.column_config.NumberColumn(
                "Weight (kg)", min_value=0.0, step=0.1, format="%.2f", required=True
            ),
        },
        key="pallets_editor",
    )

    if st.button("Save pallets", key="save_pallets"):
        cleaned = dedupe_blank_rows(edited_pallets, "PalletType")
        dupes = cleaned["PalletType"][cleaned["PalletType"].duplicated()].unique().tolist()
        if dupes:
            st.error(f"Duplicate pallet types: {', '.join(dupes)}")
        else:
            try:
                save_csv(cleaned, PALLETS_CSV)
                st.success(f"Saved {len(cleaned)} pallet types.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write `{PALLETS_CSV}`: {e}")


# =========================================================================
# FIXED COSTS
# =========================================================================
with tabs[3]:
    fixed_df = load_csv(FIXED_CSV, ["CostItem", "MonthlyCost", "Category"])
    fixed_df = fixed_df[["CostItem", "Category", "MonthlyCost"]]

    st.subheader("Fixed costs")
    st.caption("Monthly overhead. Category determines whether the item is counted in Primary or Secondary totals.")
    edited_fixed = st.data_editor(
        fixed_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "CostItem": st.column_config.TextColumn("Cost item", required=True, width="large"),
            "Category": st.column_config.SelectboxColumn(
                "Category", options=["Primary", "Secondary"], required=True
            ),
            "MonthlyCost": st.column_config.NumberColumn(
                "Monthly (USD)", min_value=0.0, step=10.0, format="%.2f"
            ),
        },
        key="fixed_editor",
    )

    if st.button("Save fixed costs", key="save_fixed"):
        cleaned = dedupe_blank_rows(edited_fixed, "CostItem")
        dupes = cleaned["CostItem"][cleaned["CostItem"].duplicated()].unique().tolist()
        if dupes:
            st.error(f"Duplicate cost items: {', '.join(dupes)}")
        elif cleaned["Category"].isnull().any():
            st.error("Every row needs a Category.")
        else:
            out = cleaned[["CostItem", "MonthlyCost", "Category"]]
            try:
                save_csv(out, FIXED_CSV)
                st.success(f"Saved {len(out)} fixed-cost items.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write `{FIXED_CSV}`: {e}")


# =========================================================================
# AIR FREIGHT RATES
# =========================================================================
with tabs[4]:
    air_df = load_csv(
        AIR_RATES_CSV,
        ["Destination", "MinWeightKG", "PricePerKG_USD", "AirwayBill_USD"],
    )

    st.subheader("Air freight rates")
    st.caption("Tiered rates per destination. The calculator picks the tier matching the shipment's gross weight.")
    edited_air = st.data_editor(
        air_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Destination": st.column_config.TextColumn("Destination", required=True),
            "MinWeightKG": st.column_config.NumberColumn(
                "Min weight (kg)", min_value=0.0, step=10.0, format="%.1f", required=True
            ),
            "PricePerKG_USD": st.column_config.NumberColumn(
                "Price / kg (USD)", min_value=0.0, step=0.01, format="%.3f", required=True
            ),
            "AirwayBill_USD": st.column_config.NumberColumn(
                "Airway bill (USD)", min_value=0.0, step=1.0, format="%.2f", required=True
            ),
        },
        key="air_editor",
    )

    if st.button("Save air rates", key="save_air"):
        cleaned = edited_air.copy()
        cleaned["Destination"] = clean_str_col(cleaned["Destination"])
        cleaned = cleaned[cleaned["Destination"] != ""]
        dupes = cleaned[cleaned.duplicated(subset=["Destination", "MinWeightKG"], keep=False)]
        if not dupes.empty:
            st.error("Duplicate (Destination, Min weight) pairs — each tier must be unique.")
        elif cleaned[["MinWeightKG", "PricePerKG_USD", "AirwayBill_USD"]].isnull().any().any():
            st.error("All numeric fields are required.")
        else:
            try:
                save_csv(cleaned, AIR_RATES_CSV)
                st.success(f"Saved {len(cleaned)} air-rate tiers.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write `{AIR_RATES_CSV}`: {e}")
