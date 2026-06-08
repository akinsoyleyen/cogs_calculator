# -*- coding: utf-8 -*-
"""Season Pricing — price one raw price per fruit across all its pack types.

You type a single raw price per produce family (e.g. "Cherry"); the app fans it
out across every pack type of that fruit and every air destination, builds the
same destination × {2,4,6,10}-pallet matrix the main calculator uses, stacks the
lot into one table, and (optionally) logs every row to the Airtable COGS Ledger.
Prices are entered in TRY/kg and live-converted to USD, exactly like app.py.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from theme import apply_theme
from cogs.exchange import FALLBACK_USD_RATES, get_usd_to_target_rate
from cogs.data_loader import (
    AIR_RATES_CSV,
    COMPONENTS_CSV,
    FIXED_CSV,
    GROUPS_CSV,
    PACKING_CSV,
    PALLETS_CSV,
    RECIPE_CSV,
    WEIGHTS_CSV,
    load_csv,
)
from cogs.produce import reconcile_groups
from cogs.matrix import build_air_matrices, matrices_to_long
from cogs.airtable_writer import airtable_is_configured, build_ledger_rows, log_matrix
from cogs.github_writer import github_is_configured, push_paths

st.set_page_config(page_title="Season Pricing — Ledger", layout="wide", page_icon="◐")
apply_theme("season")


def _persist_to_github(paths: list[str], label: str) -> None:
    """Commit saved CSV(s) to GitHub. Silently no-ops if not configured.
    Mirrors the helper in pages/1_Catalogue.py (duplicated by design)."""
    if not github_is_configured():
        st.info(
            "Edits saved locally only — Streamlit Cloud wipes them on reload. "
            "Add `github_token` and `github_repo` to `.streamlit/secrets.toml` "
            "to auto-commit. See README."
        )
        return
    msg = f"Season Pricing: update {label} ({datetime.now():%Y-%m-%d %H:%M})"
    try:
        with st.spinner(f"Pushing {label} to GitHub…"):
            result = push_paths(paths, msg)
        shas = [s for s in result.values() if s != "unchanged"]
        if shas:
            st.toast(f"Pushed to GitHub ({shas[0][:7]}). Cloud redeploy in ~30-60 s.")
        else:
            st.toast("GitHub already matched local — nothing to push.")
    except Exception as e:
        st.error(f"Saved locally but GitHub push failed: {e}.")


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


# --- Masthead ---
st.markdown(
    "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.66rem;color:var(--ink-muted);"
    "letter-spacing:0.24em;text-transform:uppercase;margin-bottom:-4px;'>"
    "Season · Air · Matrix</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='margin-top:4px'>Price the whole <em>season</em> at once.</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-family:\"Space Grotesk\",sans-serif;font-size:1rem;color:var(--ink-soft);"
    "max-width:74ch;margin-top:-2px;margin-bottom:18px;line-height:1.55;'>"
    "One raw price per fruit, fanned out across every pack type and every air "
    "destination. Fill in the produce in season, build the matrix, push it to the "
    "ledger.</p>",
    unsafe_allow_html=True,
)


# --- Load data (mirrors app.py's column handling) ---
errors: list[str] = []
weights_df, e = load_csv(WEIGHTS_CSV, ["ProductID", "NetWeightKG"], ["NetWeightKG"], decimal_char=",", string_cols=["ProductID"])
errors += [e] if e else []
packing_df, e = load_csv(PACKING_CSV, ["ProductID", "BoxesPerPallet"], ["BoxesPerPallet"], string_cols=["ProductID"])
errors += [e] if e else []
recipe_df, e = load_csv(RECIPE_CSV, ["ProductID", "ComponentName", "QuantityPerProduct"], ["QuantityPerProduct"], string_cols=["ProductID", "ComponentName"])
errors += [e] if e else []
components_raw, e = load_csv(COMPONENTS_CSV, ["ComponentName", "CostPerUnit", "WeightKG"], ["CostPerUnit", "WeightKG"])
errors += [e] if e else []
pallets_df, e = load_csv(PALLETS_CSV, ["PalletType", "CostUSD", "WeightKG"], ["CostUSD", "WeightKG"])
errors += [e] if e else []
fixed_raw, e = load_csv(FIXED_CSV, ["CostItem", "MonthlyCost", "Category"], ["MonthlyCost"], string_cols=["Category", "CostItem"])
errors += [e] if e else []
air_rates_df, e = load_csv(AIR_RATES_CSV, ["Destination", "MinWeightKG", "PricePerKG_USD", "AirwayBill_USD"], ["MinWeightKG", "PricePerKG_USD", "AirwayBill_USD"])
errors += [e] if e else []
# Grouping is non-fatal: if missing/partial, reconcile_groups seeds it from the products.
groups_raw, _ = load_csv(GROUPS_CSV, ["ProductID", "Produce"], string_cols=["ProductID", "Produce"])

if errors:
    st.error(" / ".join(errors))
    st.stop()

components_df = components_raw.copy()
components_df["CostPerUnit_USD"] = components_df["CostPerUnit"]  # USD-native pipeline
components_df["ComponentName"] = components_df["ComponentName"].astype(str).str.strip()

fixed_df = fixed_raw.rename(columns={"MonthlyCost": "MonthlyCost_USD"}).copy()
fixed_df["Category"] = fixed_df["Category"].astype(str).str.strip().str.title()

packing_df["BoxesPerPallet"] = pd.to_numeric(packing_df["BoxesPerPallet"], errors="coerce")
for _df in (weights_df, packing_df, recipe_df):
    _df["ProductID"] = _df["ProductID"].astype(str)

air_destinations = sorted(air_rates_df["Destination"].unique().tolist())

# Products that can actually be priced need both a weight and a positive boxes/pallet.
packable = packing_df[packing_df["BoxesPerPallet"].notna() & (packing_df["BoxesPerPallet"] > 0)]
buildable_ids = sorted(set(weights_df["ProductID"]) & set(packable["ProductID"]))

groups = reconcile_groups(groups_raw if groups_raw is not None else pd.DataFrame(), buildable_ids)
produce_of = dict(zip(groups["ProductID"], groups["Produce"]))
eligible_of = dict(zip(groups["ProductID"], groups["AirEligible"]))
bpp_of = {pid: int(b) for pid, b in zip(packing_df["ProductID"], packing_df["BoxesPerPallet"]) if pd.notna(b)}
produce_families = sorted(groups["Produce"].unique().tolist())
packtype_count = groups.groupby("Produce")["ProductID"].size().to_dict()


# --- FX (TRY → USD), reusing the cached Frankfurter fetch ---
fx_rate, fx_date = get_usd_to_target_rate("TRY")
fx_live = fx_rate is not None
if not fx_live:
    fx_rate = float(FALLBACK_USD_RATES.get("TRY", 38.0))


# --- Shipment assumptions (apply to every fruit in the batch) ---
st.markdown("#### Assumptions")
total_primary = fixed_df[fixed_df["Category"] == "Primary"]["MonthlyCost_USD"].sum()
total_all = fixed_df[fixed_df["Category"].isin(["Primary", "Secondary"])]["MonthlyCost_USD"].sum()
fixed_choice = st.radio(
    "Fixed costs",
    [
        f"All (Primary+Secondary) (${total_all:,.2f})",
        f"Primary only (${total_primary:,.2f})",
        "10% of total value",
    ],
    index=0,
    horizontal=True,
)
if "Primary only" in fixed_choice:
    fixed_categories, fixed_cost_mode = ["Primary"], "standard"
elif "10% of total value" in fixed_choice:
    fixed_categories, fixed_cost_mode = [], "percent"
else:
    fixed_categories, fixed_cost_mode = ["Primary", "Secondary"], "standard"

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.3, 1.2])
target_profit = c1.number_input("Target profit (%)", min_value=0.0, value=0.0, step=0.5, format="%.1f",
                                 help="Sell price = cost × (1 + profit/100). 0 leaves the table as cost.")
rebate_pct = c2.number_input("Rebate/fee (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
interest_pct = c3.number_input("Interest (%)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
pallet_types = ["None"] + sorted(pallets_df["PalletType"].unique().tolist())
_pallet_idx = pallet_types.index("Air Pallet") if "Air Pallet" in pallet_types else 0
selected_pallet_type = c4.selectbox("Pallet type", pallet_types, index=_pallet_idx)
include_variable_costs = c5.checkbox("Variable costs", value=True, help="Include packaging component costs in COGS.")

interest_rate = interest_pct / 100.0
multiplier = 1.0 + target_profit / 100.0
price_basis = "Sell price" if target_profit > 0 else "Cost"

if fx_live:
    st.success(f"USD → TRY = {fx_rate:.4f}  ·  live · {fx_date}")
else:
    st.caption(f"USD → TRY = {fx_rate:.4f}  ·  fallback")


# --- Produce price table (one row per fruit) ---
st.markdown("#### Raw prices")
st.caption("Type the lira price you received for each fruit in season. Leave the rest at 0.")
base_prices = pd.DataFrame(
    {
        "Produce": produce_families,
        "Raw price (TRY/kg)": 0.0,
        "Pack types": [int(packtype_count.get(p, 0)) for p in produce_families],
    }
)
edited_prices = st.data_editor(
    base_prices,
    hide_index=True,
    use_container_width=True,
    key="season_price_editor",
    column_config={
        "Produce": st.column_config.TextColumn("Produce (fruit)", disabled=True, width="large"),
        "Raw price (TRY/kg)": st.column_config.NumberColumn("Raw price (TRY/kg)", min_value=0.0, step=1.0, format="%.2f"),
        "Pack types": st.column_config.NumberColumn("Pack types", disabled=True, format="%d", width="small"),
    },
)

edited_prices = edited_prices.copy()
edited_prices["Raw price (TRY/kg)"] = pd.to_numeric(edited_prices["Raw price (TRY/kg)"], errors="coerce").fillna(0.0)
priced = edited_prices[edited_prices["Raw price (TRY/kg)"] > 0]
try_by_produce = dict(zip(priced["Produce"], priced["Raw price (TRY/kg)"]))
usd_by_produce = {p: (v / fx_rate if fx_rate > 0 else 0.0) for p, v in try_by_produce.items()}

if try_by_produce:
    preview = pd.DataFrame(
        {
            "Produce": list(try_by_produce),
            "TRY/kg": list(try_by_produce.values()),
            "USD/kg": [round(usd_by_produce[p], 4) for p in try_by_produce],
        }
    )
    st.dataframe(
        preview.style.format({"TRY/kg": "{:,.2f}", "USD/kg": "${:,.4f}"}),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("Enter a price for at least one fruit to build the matrix.")


# --- Pack-type & destination scope ---
priced_set = set(try_by_produce)
candidate_pids = [pid for pid in buildable_ids if produce_of.get(pid) in priced_set]
n_eligible = sum(1 for pid in candidate_pids if eligible_of.get(pid, True))

sc1, sc2 = st.columns([1, 1])
with sc1:
    include_non_air = st.checkbox(
        f"Include non-air-eligible pack types ({len(candidate_pids) - n_eligible} hidden)",
        value=False,
        help="Off by default so Container/Truck-only variants don't get bogus air prices. "
        "Manage this per pack type under 'Edit produce grouping' below.",
        disabled=not candidate_pids,
    )
with sc2:
    selected_dests = st.multiselect(
        "Destinations", options=air_destinations, default=air_destinations, key="season_dests",
    )

build_pids = [pid for pid in candidate_pids if include_non_air or eligible_of.get(pid, True)]

st.caption(
    f"Will build **{len(build_pids)}** pack type(s) × **{len(selected_dests)}** destination(s)."
    if build_pids and selected_dests
    else "Nothing to build yet."
)

build_clicked = st.button(
    "Build season matrix",
    type="primary",
    disabled=not (build_pids and selected_dests),
)

if build_clicked:
    air_for_build = air_rates_df[air_rates_df["Destination"].isin(selected_dests)]
    dfs = {
        "product_weights_df": weights_df,
        "product_recipe_df": recipe_df,
        "components_df": components_df,
        "pallets_df": pallets_df,
        "fixed_df": fixed_df,
        "air_rates_df": air_for_build,
        "product_packing_df": packing_df,
    }
    base_common = {
        "selected_pallet_type": selected_pallet_type,
        "include_variable_costs": include_variable_costs,
        "fixed_cost_mode": fixed_cost_mode,
        "fixed_categories": fixed_categories,
        "interest_rate": interest_rate,
        "unexpected_cost_usd": 0.0,
        "rebate_percentage": rebate_pct,
    }
    raw_usd_by_pid = {pid: usd_by_produce[produce_of[pid]] for pid in build_pids}
    with st.spinner(f"Pricing {len(build_pids)} pack types across {len(selected_dests)} destinations…"):
        matrices, build_errors = build_air_matrices(
            product_ids=build_pids,
            raw_cost_per_kg_usd_by_product=raw_usd_by_pid,
            base_common=base_common,
            dfs=dfs,
        )
        long_df = matrices_to_long(
            matrices, produce_of=produce_of, boxes_per_pallet_of=bpp_of, multiplier=multiplier,
        )
    st.session_state["season_result"] = {
        "long": long_df,
        "cost_matrices": matrices,
        "errors": build_errors,
        "multiplier": multiplier,
        "price_basis": price_basis,
        "target_profit": target_profit,
        "rebate": rebate_pct,
        "fixed_cost_mode": fixed_cost_mode,
        "produce_of": produce_of,
        "bpp_of": bpp_of,
        "usd_by_produce": usd_by_produce,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# --- Results ---
res = st.session_state.get("season_result")
if res and not res["long"].empty:
    st.markdown("---")
    st.markdown("#### Matrix")
    st.caption(
        f"Built {res['built_at']} · **{res['price_basis']}** per box (USD) · "
        f"{len(res['cost_matrices'])} pack types · {len(res['long'])} rows."
    )
    long_df = res["long"]
    pallet_cols = [c for c in long_df.columns if c.endswith("pallets")]
    st.dataframe(
        long_df.style.format({c: "${:,.2f}" for c in pallet_cols}),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Copy for spreadsheet (tab-separated)", expanded=False):
        st.code(long_df.to_csv(sep="\t", index=False, float_format="%.2f"), language="text")
        st.caption("Click the copy icon, then paste into Excel / Google Sheets / an email.")

    if res["errors"]:
        with st.expander(f"⚠️ {len(res['errors'])} pack type(s) skipped", expanded=False):
            for pid, msg in res["errors"].items():
                st.write(f"**{pid}** — {msg}")

    # --- Log all to ledger ---
    st.markdown("---")
    st.subheader("📌 Log all to ledger (Airtable)")
    if not airtable_is_configured():
        st.info(
            "Add `airtable_token`, `airtable_base_id`, and `airtable_table` to "
            "`.streamlit/secrets.toml` to enable. See README."
        )
    else:
        st.caption(
            f"Appends {len(res['long'])} rows — one per (pack type × destination) — to the "
            "COGS Ledger, reusing the single-product schema (`Product` = pack type)."
        )
        if st.button("Log all to ledger"):
            try:
                now = datetime.now()
                all_rows: list[dict] = []
                for pid, cost_matrix in res["cost_matrices"].items():
                    active = (cost_matrix * res["multiplier"]).round(2)
                    produce = res["produce_of"].get(pid, "")
                    all_rows.extend(
                        build_ledger_rows(
                            active,
                            product=pid,
                            price_basis=res["price_basis"],
                            target_profit_percent=round(float(res["target_profit"]), 2),
                            raw_cost_per_kg=round(float(res["usd_by_produce"].get(produce, 0.0)), 4),
                            rebate_percentage=round(float(res["rebate"]), 2),
                            fixed_cost_mode=res["fixed_cost_mode"],
                            boxes_per_pallet=int(res["bpp_of"].get(pid, 0) or 0),
                            logged_at_iso=now.astimezone().isoformat(timespec="seconds"),
                            batch_id=f"{now:%Y-%m-%d %H:%M} · season lot",
                        )
                    )
                with st.spinner("Logging to Airtable…"):
                    n = log_matrix(all_rows)
                st.success(f"Logged {n} rows across {len(res['cost_matrices'])} pack types.")
            except Exception as e:
                st.error(f"Could not log to ledger: {e}")


# --- Produce grouping editor ---
st.markdown("---")
with st.expander("⚙️ Edit produce grouping (which pack types belong to which fruit)", expanded=False):
    st.caption(
        "Set the fruit each pack type is priced under, and whether it ships by air. "
        "Turn **Air-eligible** off for Container/Truck-only variants so they stay out of "
        "the air sweep."
    )
    edited_groups = st.data_editor(
        groups,
        hide_index=True,
        use_container_width=True,
        key="season_groups_editor",
        column_config={
            "ProductID": st.column_config.TextColumn("Pack type (Product ID)", disabled=True, width="large"),
            "Produce": st.column_config.TextColumn("Produce (fruit)", required=True),
            "AirEligible": st.column_config.CheckboxColumn("Air-eligible", default=True),
        },
    )
    if st.button("Save grouping"):
        out = edited_groups.copy()
        out["ProductID"] = out["ProductID"].astype(str).str.strip()
        out["Produce"] = out["Produce"].astype(str).str.strip()
        if (out["Produce"] == "").any():
            st.error("Every pack type needs a Produce name.")
        else:
            try:
                _save_csv(out[["ProductID", "Produce", "AirEligible"]], GROUPS_CSV)
                st.success("Saved produce grouping.")
                _persist_to_github([GROUPS_CSV], "produce grouping")
                st.rerun()
            except Exception as ex:
                st.error(f"Failed to write `{GROUPS_CSV}`: {ex}")
