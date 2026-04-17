# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st

COMPONENTS_CSV = "components.csv"
RECIPE_CSV = "product_recipe.csv"
WEIGHTS_CSV = "product_weights.csv"
PACKING_CSV = "product_packing.csv"

st.set_page_config(page_title="Add Product — Ledger", layout="wide", page_icon="◐")

# Reuse the same editorial styling as the main app
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
[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 2px; font-family: "JetBrains Mono", monospace; }
[data-testid="stAlert"] { border-radius: 2px; border: 1px solid var(--rule); background: var(--paper); }
[data-testid="stAlert"][kind="success"] { background: var(--olive-wash); }
</style>
"""
if hasattr(st, "html"):
    st.html(_SHARED_CSS)
else:
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)

st.markdown(
    "<div style='font-family:\"Public Sans\",sans-serif;font-size:0.72rem;color:var(--ink-muted);"
    "letter-spacing:0.18em;text-transform:uppercase;margin-bottom:-4px;'>"
    "Catalogue &middot; Components &middot; Recipes</div>",
    unsafe_allow_html=True,
)
st.title("Add a Product")
st.markdown(
    "<p style='font-family:\"Public Sans\",sans-serif;font-size:1rem;color:var(--ink-soft);"
    "max-width:64ch;margin-top:-2px;margin-bottom:24px;line-height:1.55;'>"
    "Register a new SKU with its recipe, weight and packing so the calculator can price it.</p>",
    unsafe_allow_html=True,
)


def load_csv_or_stop(path, required_cols, decimal_char="."):
    if not os.path.exists(path):
        st.error(f"Missing required file: `{path}`")
        st.stop()
    df = pd.read_csv(path, decimal=decimal_char)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"`{path}` is missing required columns: {missing}")
        st.stop()
    return df


components_df = load_csv_or_stop(COMPONENTS_CSV, ["ComponentName", "CostPerUnit", "WeightKG"])
recipe_df = load_csv_or_stop(RECIPE_CSV, ["ProductID", "ComponentName", "QuantityPerProduct"])
weights_df = load_csv_or_stop(WEIGHTS_CSV, ["ProductID", "NetWeightKG"], decimal_char=",")
packing_df = load_csv_or_stop(PACKING_CSV, ["ProductID", "BoxesPerPallet"])

component_options = sorted(components_df["ComponentName"].astype(str).str.strip().unique().tolist())
existing_products = set(weights_df["ProductID"].astype(str).str.strip().tolist())
existing_components = set(components_df["ComponentName"].astype(str).str.strip().tolist())

tabs = st.tabs(["Add Product", "Add Component"])

with tabs[0]:
    show_product_form = True
    if not component_options:
        show_product_form = False
        st.warning(
            "No components found in `components.csv`. Add components in the next tab to enable product recipes."
        )

    if show_product_form:
        with st.form("add_product_form"):
            product_id = st.text_input("Product ID", placeholder="e.g., 15kg Telescopic Lemon")
            net_weight = st.number_input("Net weight (kg)", min_value=0.0, step=0.1, format="%.3f")
            boxes_per_pallet = st.number_input("Boxes per pallet", min_value=1, step=1)

            st.markdown("Recipe components")
            starter = pd.DataFrame(
                {"ComponentName": [component_options[0]], "QuantityPerProduct": [1.0]}
            )
            recipe_table = st.data_editor(
                starter,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "ComponentName": st.column_config.SelectboxColumn(
                        "Component",
                        options=component_options,
                        required=True,
                    ),
                    "QuantityPerProduct": st.column_config.NumberColumn(
                        "Qty per product",
                        min_value=0.0,
                        step=0.1,
                        format="%.3f",
                        required=True,
                    ),
                },
            )

            submitted = st.form_submit_button("Add product")

        if submitted:
            product_id_clean = product_id.strip()
            errors = []

            if not product_id_clean:
                errors.append("Product ID is required.")
            if product_id_clean in existing_products:
                errors.append(f"Product '{product_id_clean}' already exists in `{WEIGHTS_CSV}`.")

            recipe_table = recipe_table.dropna(how="all")
            if recipe_table.empty:
                errors.append("At least one recipe component is required.")
            else:
                invalid_components = recipe_table[
                    ~recipe_table["ComponentName"].astype(str).isin(component_options)
                ]
                if not invalid_components.empty:
                    errors.append("Recipe contains invalid components not in `components.csv`.")
                if (recipe_table["QuantityPerProduct"].astype(float) <= 0).any():
                    errors.append("All recipe quantities must be greater than 0.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                # Append to weights CSV
                new_weights_row = pd.DataFrame(
                    [{"ProductID": product_id_clean, "NetWeightKG": float(net_weight)}]
                )
                weights_out = pd.concat([weights_df, new_weights_row], ignore_index=True)

                # Append to packing CSV
                new_packing_row = pd.DataFrame(
                    [{"ProductID": product_id_clean, "BoxesPerPallet": int(boxes_per_pallet)}]
                )
                packing_out = pd.concat([packing_df, new_packing_row], ignore_index=True)

                # Append to recipe CSV
                recipe_rows = recipe_table.copy()
                recipe_rows["ProductID"] = product_id_clean
                recipe_out = pd.concat([recipe_df, recipe_rows], ignore_index=True)
                recipe_out = recipe_out[["ProductID", "ComponentName", "QuantityPerProduct"]]

                try:
                    weights_out.to_csv(WEIGHTS_CSV, index=False, decimal=",")
                    packing_out.to_csv(PACKING_CSV, index=False)
                    recipe_out.to_csv(RECIPE_CSV, index=False)
                    st.success(f"Added '{product_id_clean}' to product files.")
                    st.write("New product details:")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "ProductID": [product_id_clean],
                                "NetWeightKG": [net_weight],
                                "BoxesPerPallet": [boxes_per_pallet],
                            }
                        ),
                        use_container_width=True,
                    )
                    st.dataframe(recipe_rows.reset_index(drop=True), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to write CSVs: {e}")

with tabs[1]:
    st.subheader("Add Component")

    with st.form("add_component_form"):
        component_name = st.text_input("Component name", placeholder="e.g., 40x60x9")
        component_type = st.text_input("Component type", placeholder="e.g., Box")
        cost_per_unit = st.number_input(
            "Cost per unit", min_value=0.0, step=0.01, format="%.4f"
        )
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, step=0.01, format="%.4f")
        component_submitted = st.form_submit_button("Add component")

    if component_submitted:
        name_clean = component_name.strip()
        type_clean = component_type.strip()
        errors = []

        if not name_clean:
            errors.append("Component name is required.")
        if not type_clean:
            errors.append("Component type is required.")
        if name_clean in existing_components:
            errors.append(f"Component '{name_clean}' already exists in `{COMPONENTS_CSV}`.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            new_component_row = pd.DataFrame(
                [
                    {
                        "ComponentName": name_clean,
                        "ComponentType": type_clean,
                        "CostPerUnit": float(cost_per_unit),
                        "WeightKG": float(weight_kg),
                    }
                ]
            )
            components_out = pd.concat([components_df, new_component_row], ignore_index=True)

            try:
                components_out.to_csv(COMPONENTS_CSV, index=False)
                st.success(f"Added '{name_clean}' to `{COMPONENTS_CSV}`.")
                st.dataframe(new_component_row, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to write `{COMPONENTS_CSV}`: {e}")
