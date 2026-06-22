# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
import pandas as pd
import streamlit as st

from theme import apply_theme
from cogs.github_writer import github_is_configured, push_paths
from cogs.produce import reconcile_groups


def _persist_to_github(paths: list[str], label: str) -> None:
    """Try to commit the saved CSV(s) to GitHub. Silently no-op if not configured."""
    if not github_is_configured():
        st.info(
            "Edits saved locally only — Streamlit Cloud wipes them on reload. "
            "Add `github_token` and `github_repo` to `.streamlit/secrets.toml` "
            "to auto-commit. See README."
        )
        return
    msg = f"Catalogue: update {label} ({datetime.now():%Y-%m-%d %H:%M})"
    try:
        with st.spinner(f"Pushing {label} to GitHub…"):
            result = push_paths(paths, msg)
        shas = [s for s in result.values() if s != "unchanged"]
        if shas:
            st.toast(f"Pushed to GitHub ({shas[0][:7]}). Cloud redeploy in ~30-60 s.")
        else:
            st.toast("GitHub already matched local — nothing to push.")
    except Exception as e:
        st.error(
            f"Saved locally but GitHub push failed: {e}. "
            "Edits will revert on next Cloud reload."
        )

COMPONENTS_CSV = "components.csv"
RECIPE_CSV = "product_recipe.csv"
WEIGHTS_CSV = "product_weights.csv"
PACKING_CSV = "product_packing.csv"
PALLETS_CSV = "pallets.csv"
FIXED_CSV = "fixed_costs.csv"
AIR_RATES_CSV = "air_freight_rates.csv"
GROUPS_CSV = "product_groups.csv"

st.set_page_config(page_title="Catalogue — Ledger", layout="wide", page_icon="◐")

apply_theme("catalogue")


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


def parse_pep_list(text, carrier_suffix, airway_bill):
    """Turn a pasted PEP price list into tiered air-rate rows.

    A PEP list is one destination per line: an IATA code followed by a
    small-weight rate (Q100) and a large-weight rate (Q1000), e.g.::

        SIN 2,4 2,2

    Each line becomes two tiers — MinWeightKG 0 (Q100) and 999 (Q1000) —
    with the carrier suffix appended (``SIN`` → ``SIN-TK``) and a flat
    airway bill. Decimals may be written with a comma or a dot. Header,
    date and any other non-IATA lines are ignored; lines that look like a
    destination but carry no rate are returned as ``skipped``.
    """
    suffix = (carrier_suffix or "").strip().upper()
    tail = f"-{suffix}" if suffix else ""
    rows, skipped = [], []
    for raw in text.splitlines():
        tokens = raw.strip().split()
        if not tokens:
            continue
        code = tokens[0].upper()
        if not re.fullmatch(r"[A-Z]{2,4}", code):
            continue  # header row, date range, or stray text
        nums = []
        for tok in tokens[1:]:
            try:
                nums.append(float(tok.replace(",", ".")))
            except ValueError:
                pass
        if not nums:
            skipped.append(code)
            continue
        q100 = nums[0]
        q1000 = nums[1] if len(nums) > 1 else nums[0]
        dest = f"{code}{tail}"
        rows.append({"Destination": dest, "MinWeightKG": 0.0,
                     "PricePerKG_USD": q100, "AirwayBill_USD": airway_bill})
        rows.append({"Destination": dest, "MinWeightKG": 999.0,
                     "PricePerKG_USD": q1000, "AirwayBill_USD": airway_bill})
    parsed = pd.DataFrame(
        rows, columns=["Destination", "MinWeightKG", "PricePerKG_USD", "AirwayBill_USD"]
    )
    return parsed, skipped


# --- Masthead ---
st.markdown(
    "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.66rem;color:var(--ink-muted);"
    "letter-spacing:0.24em;text-transform:uppercase;margin-bottom:-4px;'>"
    "Catalogue · Costs · Logistics</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='margin-top:4px'>Edit <em>everything</em>.</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-family:\"Space Grotesk\",sans-serif;font-size:1rem;color:var(--ink-soft);"
    "max-width:72ch;margin-top:-2px;margin-bottom:20px;line-height:1.55;'>"
    "Every cost, count and coefficient the calculator reads — in one place, all in USD. "
    "Edit inline, add or remove rows, then save.</p>",
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
    groups_df = (
        load_csv(GROUPS_CSV, ["ProductID", "Produce"])
        if os.path.exists(GROUPS_CSV)
        else pd.DataFrame(columns=["ProductID", "Produce", "AirEligible"])
    )

    component_options = sorted(clean_str_col(components_df["ComponentName"]).unique().tolist())

    weights_df["ProductID"] = clean_str_col(weights_df["ProductID"])
    packing_df["ProductID"] = clean_str_col(packing_df["ProductID"])
    recipe_df["ProductID"] = clean_str_col(recipe_df["ProductID"])
    recipe_df["ComponentName"] = clean_str_col(recipe_df["ComponentName"])

    merged = weights_df.merge(packing_df, on="ProductID", how="outer")
    merged = merged[["ProductID", "NetWeightKG", "BoxesPerPallet"]]
    # Bring in the produce family + air-eligibility (seeding any missing/new IDs).
    groups_aligned = reconcile_groups(groups_df, merged["ProductID"].tolist())
    merged = merged.merge(groups_aligned, on="ProductID", how="left")
    merged = merged[["ProductID", "Produce", "NetWeightKG", "BoxesPerPallet", "AirEligible"]]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Products", f"{merged['ProductID'].nunique()}")
    col_b.metric("Produce families", f"{merged['Produce'].nunique()}")
    col_c.metric("Components in use", f"{recipe_df['ComponentName'].nunique()}")
    col_d.metric("Recipe lines", f"{len(recipe_df)}")

    st.subheader("Products master list")
    st.caption(
        "Edit cells directly. **Produce** is the fruit each pack type is priced under on "
        "the Season Pricing page — leave it blank to auto-guess from the name on save. "
        "Use the + row at the bottom to add, or tick the trash icon to remove."
    )
    edited_products = st.data_editor(
        merged,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ProductID": st.column_config.TextColumn("Product ID", required=True, width="large"),
            "Produce": st.column_config.TextColumn(
                "Produce (fruit)", width="medium",
                help="Fruit family for the Season Pricing page (e.g. 'Cherry 16 x 350g' → 'Cherry'). "
                     "Blank = auto-guessed from the name on save.",
            ),
            "NetWeightKG": st.column_config.NumberColumn(
                "Net weight (kg)", min_value=0.0, step=0.1, format="%.3f", required=True
            ),
            "BoxesPerPallet": st.column_config.NumberColumn(
                "Boxes per pallet", min_value=1, step=1, format="%d", required=True
            ),
            "AirEligible": st.column_config.CheckboxColumn(
                "Air-eligible", default=True,
                help="Off for Container/Truck-only variants so they're left out of the Season "
                     "air matrix by default.",
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

                # Produce grouping: reconcile fills blank Produce (auto-guess) and
                # normalises Air-eligible, so the editor is the full source of truth.
                groups_out = reconcile_groups(
                    cleaned[["ProductID", "Produce", "AirEligible"]],
                    cleaned["ProductID"].tolist(),
                )
                save_csv(groups_out, GROUPS_CSV)

                kept_ids = set(cleaned["ProductID"].tolist())
                recipe_cleaned = recipe_df[recipe_df["ProductID"].isin(kept_ids)].copy()
                save_csv(recipe_cleaned, RECIPE_CSV)

                st.success(
                    f"Saved {len(cleaned)} products. "
                    f"Recipes for removed products were dropped."
                )
                _persist_to_github([WEIGHTS_CSV, PACKING_CSV, RECIPE_CSV, GROUPS_CSV], "products")
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
                    _persist_to_github([RECIPE_CSV], f"recipe for {sel}")
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
    st.caption("Unit cost is in USD — one currency across the app, converted to your reporting currency only at the end.")
    edited_components = st.data_editor(
        components_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ComponentName": st.column_config.TextColumn("Component name", required=True, width="large"),
            "ComponentType": st.column_config.TextColumn("Type", width="small"),
            "CostPerUnit": st.column_config.NumberColumn(
                "Cost / unit (USD)", min_value=0.0, step=0.01, format="%.4f", required=True
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
                _persist_to_github([COMPONENTS_CSV], "components")
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
                _persist_to_github([PALLETS_CSV], "pallets")
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
                _persist_to_github([FIXED_CSV], "fixed costs")
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

    # ---- Bulk import from a PEP price list -----------------------------------
    with st.expander("⬆ Bulk import from a PEP price list", expanded=False):
        st.caption(
            "Paste the airline's PEP list — one destination per line, e.g. "
            "`SIN 2,4 2,2` (IATA · Q100 · Q1000). Each line becomes two tiers "
            "(0 kg = Q100, 999 kg = Q1000). Comma or dot decimals both work; "
            "header and date lines are ignored."
        )
        pep_text = st.text_area(
            "PEP list", height=160, key="pep_text",
            placeholder="ALA 2,2 1,9\nAMM 2 1,8\nAMS 1,9 1,7\n…",
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            pep_suffix = st.text_input("Carrier suffix", value="TK", key="pep_suffix")
        with c2:
            pep_awb = st.number_input(
                "Airway bill (USD)", min_value=0.0, value=115.5, step=1.0,
                format="%.2f", key="pep_awb",
            )
        with c3:
            pep_mode = st.radio(
                "On import",
                ["Replace all rates", "Merge / update destinations"],
                key="pep_mode",
                help="Replace wipes the table. Merge updates the tiers for "
                     "destinations in the paste and keeps every other row.",
            )

        if st.button("Preview import", key="pep_preview_btn"):
            parsed, skipped = parse_pep_list(pep_text, pep_suffix, pep_awb)
            st.session_state["pep_parsed"] = parsed
            st.session_state["pep_skipped"] = skipped

        parsed = st.session_state.get("pep_parsed")
        if parsed is not None:
            if parsed.empty:
                st.warning("Nothing parsed — check the pasted text.")
            else:
                dests = parsed["Destination"].nunique()
                st.success(f"Parsed {dests} destinations → {len(parsed)} tier rows.")
                skipped = st.session_state.get("pep_skipped") or []
                if skipped:
                    st.warning("No rate found for: " + ", ".join(skipped))
                st.dataframe(parsed, use_container_width=True, hide_index=True)
                bc1, bc2 = st.columns([1, 1])
                if bc1.button("Confirm & save", key="pep_confirm"):
                    # PEP lists only carry the 0 kg (Q100) and 999 kg (Q1000)
                    # tiers; any other tier (e.g. a negotiated 4999 kg rate) is
                    # preserved so it isn't silently wiped on import.
                    pep_tiers = {0.0, 999.0}
                    incoming = set(parsed["Destination"])
                    extras = air_df[~air_df["MinWeightKG"].isin(pep_tiers)]
                    if pep_mode.startswith("Replace"):
                        extras = extras[extras["Destination"].isin(incoming)]
                        final = pd.concat([parsed, extras], ignore_index=True)
                    else:
                        drop = air_df["Destination"].isin(incoming) & air_df["MinWeightKG"].isin(pep_tiers)
                        final = pd.concat([air_df[~drop], parsed], ignore_index=True)
                    final = final.sort_values(
                        ["Destination", "MinWeightKG"]
                    ).reset_index(drop=True)
                    try:
                        save_csv(final, AIR_RATES_CSV)
                        st.success(f"Imported — {len(final)} air-rate tiers saved.")
                        _persist_to_github([AIR_RATES_CSV], "air freight rates (PEP import)")
                        st.session_state.pop("pep_parsed", None)
                        st.session_state.pop("pep_skipped", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to write `{AIR_RATES_CSV}`: {e}")
                if bc2.button("Cancel", key="pep_cancel"):
                    st.session_state.pop("pep_parsed", None)
                    st.session_state.pop("pep_skipped", None)
                    st.rerun()

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
                _persist_to_github([AIR_RATES_CSV], "air freight rates")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write `{AIR_RATES_CSV}`: {e}")
