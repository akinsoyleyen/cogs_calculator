"""Pure landed-cost calculation. UI-free so it can be called in loops (matrix push)."""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from cogs.data_loader import (
    AIR_RATES_CSV,
    COMPONENTS_CSV,
    INTEREST_COST_ITEM_NAME,
    RECIPE_CSV,
    WEIGHTS_CSV,
)


@dataclass
class CalcResult:
    # weights
    net_weight_kg_per_box: float = 0.0
    gross_weight_kg_per_box: float = 0.0
    final_shipping_gross_weight_kg: float = 0.0
    total_pallet_weight_kg: float = 0.0
    total_net_weight_kg: float = 0.0
    total_packaging_weight_kg: float = 0.0

    # per-batch USD
    total_raw_cost_usd: float = 0.0
    total_variable_comp_cost_usd: float = 0.0
    total_pallet_cost_usd: float = 0.0
    total_variable_costs_incl_pallets_usd: float = 0.0
    total_allocated_fixed_cost_usd: float = 0.0
    fixed_cost_10_percent: float = 0.0
    interest_cost_usd: float = 0.0
    total_cogs_usd: float = 0.0
    total_logistics_cost_usd: float = 0.0
    freight_or_fixed_logistics_cost: float = 0.0
    awb_cost: float = 0.0
    logistics_rate_per_kg: float = 0.0
    fixed_logistics_price: float = 0.0
    total_unexpected_cost_usd: float = 0.0
    total_delivered_cost_usd: float = 0.0
    rebate_amount_usd: float = 0.0
    final_total_cost_usd: float = 0.0

    # per-box USD
    raw_cost_per_box_usd: float = 0.0
    variable_costs_incl_pallets_per_box_usd: float = 0.0
    total_per_unit_variable_comp_cost_usd: float = 0.0
    pallet_cost_per_box_usd: float = 0.0
    fixed_cost_per_unit_usd: float = 0.0
    cogs_per_box_usd: float = 0.0
    logistics_per_box_usd: float = 0.0
    unexpected_cost_per_box_usd: float = 0.0
    delivered_cost_per_box_usd: float = 0.0
    final_cost_per_box_usd: float = 0.0

    # per-kg USD
    cogs_per_kg_usd: float = 0.0
    logistics_per_kg_gross_usd: float = 0.0
    delivered_cost_per_kg_net_usd: float = 0.0

    # meta
    logistics_cost_source: str = "N/A"
    rebate_percentage: float = 0.0
    cost_per_pallet_usd: float = 0.0
    weight_per_pallet_kg: float = 0.0

    # breakdown
    product_variable_costs_detailed: pd.DataFrame = field(default_factory=pd.DataFrame)

    # collected non-fatal messages — caller displays via st.warning
    warnings: list[str] = field(default_factory=list)


def compute_landed_cost(
    *,
    selected_product: str,
    quantity_boxes: int,
    selected_pallet_type: str,
    num_pallets: int,
    raw_cost_per_kg_usd: float,
    include_variable_costs: bool,
    fixed_cost_mode: str,                 # "standard" | "percent"
    fixed_categories: list[str],          # ['Primary'] or ['Primary','Secondary']
    interest_rate: float,                 # decimal (e.g. 0.05)
    unexpected_cost_usd: float,
    rebate_percentage: float,
    shipment_type: str,                   # "Air" | "Container" | "Truck"
    destination: Optional[str],           # required if Air
    manual_logistics_cost_usd: float,     # used if Container/Truck
    product_weights_df: pd.DataFrame,
    product_recipe_df: pd.DataFrame,
    components_df: pd.DataFrame,
    pallets_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    air_rates_df: Optional[pd.DataFrame],
) -> CalcResult:
    r = CalcResult()
    selected_product_str = str(selected_product)

    # 1. Pallet specs & cost
    if selected_pallet_type != "None" and num_pallets > 0 and pallets_df is not None:
        pallet_spec = pallets_df[pallets_df['PalletType'] == selected_pallet_type]
        if not pallet_spec.empty:
            r.cost_per_pallet_usd = float(pallet_spec['CostUSD'].iloc[0])
            r.weight_per_pallet_kg = float(pallet_spec['WeightKG'].iloc[0])
        else:
            r.warnings.append(f"Specs missing for pallet '{selected_pallet_type}'. Cost/Weight assumed 0.")
    r.total_pallet_cost_usd = num_pallets * r.cost_per_pallet_usd
    r.total_pallet_weight_kg = num_pallets * r.weight_per_pallet_kg
    r.pallet_cost_per_box_usd = r.total_pallet_cost_usd / quantity_boxes if quantity_boxes > 0 else 0.0

    # 2. Net weight
    product_weight_info = product_weights_df[product_weights_df['ProductID'] == selected_product_str]
    if product_weight_info.empty:
        raise ValueError(f"Net Weight missing for '{selected_product_str}' in '{WEIGHTS_CSV}'.")
    r.net_weight_kg_per_box = float(product_weight_info['NetWeightKG'].iloc[0])
    if not r.net_weight_kg_per_box > 0:
        raise ValueError(f"Net Weight for '{selected_product_str}' must be > 0.")

    # 3. Raw product cost
    r.raw_cost_per_box_usd = raw_cost_per_kg_usd * r.net_weight_kg_per_box
    r.total_raw_cost_usd = r.raw_cost_per_box_usd * quantity_boxes

    # 4. Variable components + packaging weight (always compute weight for logistics)
    product_recipe = product_recipe_df[product_recipe_df['ProductID'] == selected_product_str]
    if not product_recipe.empty:
        recipe_components = product_recipe['ComponentName'].unique()
        available_components = components_df['ComponentName'].unique()
        missing = [c for c in recipe_components if c not in available_components]
        if missing:
            raise ValueError(
                f"Component(s) in recipe for '{selected_product_str}' not found in "
                f"'{COMPONENTS_CSV}': {missing}"
            )

        detailed = pd.merge(
            product_recipe,
            components_df[['ComponentName', 'CostPerUnit_USD', 'WeightKG']],
            on='ComponentName', how='left',
        )
        if detailed.isnull().values.any():
            still_missing = pd.Series(
                detailed[detailed.isnull().any(axis=1)]['ComponentName']
            ).unique().tolist()
            raise ValueError(
                f"Details missing after merge for component(s): {still_missing}. "
                f"Check '{COMPONENTS_CSV}'."
            )

        detailed['QuantityPerProduct'] = pd.to_numeric(detailed['QuantityPerProduct'], errors='coerce')
        if detailed['QuantityPerProduct'].isnull().any():
            raise ValueError("Non-numeric 'QuantityPerProduct' found in recipe.")

        detailed['LineItemWeightKG'] = detailed['WeightKG'] * detailed['QuantityPerProduct']
        r.total_packaging_weight_kg = float(detailed['LineItemWeightKG'].sum())

        if include_variable_costs:
            detailed['LineItemCost_USD'] = detailed['CostPerUnit_USD'] * detailed['QuantityPerProduct']
            r.total_per_unit_variable_comp_cost_usd = float(detailed['LineItemCost_USD'].sum())
            r.total_variable_comp_cost_usd = r.total_per_unit_variable_comp_cost_usd * quantity_boxes
        else:
            detailed['LineItemCost_USD'] = 0.0
            r.total_per_unit_variable_comp_cost_usd = 0.0
            r.total_variable_comp_cost_usd = 0.0

        r.product_variable_costs_detailed = detailed
    else:
        r.warnings.append(
            f"No recipe found for ProductID '{selected_product_str}' in '{RECIPE_CSV}'. "
            "Variable costs & packaging weight assumed 0."
        )

    # 5. Totals: variable (incl. pallets), weights
    r.total_variable_costs_incl_pallets_usd = r.total_variable_comp_cost_usd + r.total_pallet_cost_usd
    r.variable_costs_incl_pallets_per_box_usd = (
        r.total_variable_costs_incl_pallets_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    )

    r.gross_weight_kg_per_box = r.net_weight_kg_per_box + r.total_packaging_weight_kg
    total_batch_gross_weight_boxes_only = r.gross_weight_kg_per_box * quantity_boxes
    r.final_shipping_gross_weight_kg = total_batch_gross_weight_boxes_only + r.total_pallet_weight_kg

    # 6. Logistics (Air uses tier table; Container/Truck use manual)
    if shipment_type == "Air":
        if air_rates_df is not None and not air_rates_df.empty and destination:
            applicable = air_rates_df[
                (air_rates_df['Destination'] == destination)
                & (air_rates_df['MinWeightKG'] <= r.final_shipping_gross_weight_kg)
            ].sort_values('MinWeightKG', ascending=False)
            if not applicable.empty:
                rate_row = applicable.iloc[0]
                r.logistics_rate_per_kg = float(rate_row['PricePerKG_USD'])
                r.awb_cost = float(rate_row['AirwayBill_USD'])
                r.freight_or_fixed_logistics_cost = (
                    r.final_shipping_gross_weight_kg * r.logistics_rate_per_kg + r.awb_cost
                )
                r.logistics_cost_source = f"Air: {rate_row['MinWeightKG']}+ KG Tier"
            else:
                r.warnings.append(
                    f"No AIR rate for {destination} at {r.final_shipping_gross_weight_kg:.2f} KG."
                )
                r.logistics_cost_source = "Air: No Rate Found"
        else:
            r.warnings.append(f"Air rates file '{AIR_RATES_CSV}' missing or empty.")
            r.logistics_cost_source = "Air: Rates Missing"
    elif shipment_type in ("Container", "Truck"):
        r.freight_or_fixed_logistics_cost = manual_logistics_cost_usd
        r.fixed_logistics_price = manual_logistics_cost_usd
        r.logistics_cost_source = f"{shipment_type}: Manual Input"

    r.total_logistics_cost_usd = r.freight_or_fixed_logistics_cost
    r.logistics_per_box_usd = r.total_logistics_cost_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    r.logistics_per_kg_gross_usd = (
        r.total_logistics_cost_usd / r.final_shipping_gross_weight_kg
        if r.final_shipping_gross_weight_kg > 0 else 0.0
    )

    # 7. Unexpected
    r.total_unexpected_cost_usd = unexpected_cost_usd
    r.unexpected_cost_per_box_usd = (
        r.total_unexpected_cost_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    )

    # 8. Fixed costs — 10% mode depends on totals incl. logistics
    if fixed_cost_mode == "percent":
        total_value_for_percent = (
            r.total_raw_cost_usd
            + r.total_variable_costs_incl_pallets_usd
            + r.total_logistics_cost_usd
            + r.total_unexpected_cost_usd
        )
        r.fixed_cost_10_percent = total_value_for_percent * 0.10
        r.total_allocated_fixed_cost_usd = r.fixed_cost_10_percent
    else:
        standard_fixed = fixed_df[
            (fixed_df['CostItem'].str.strip().str.lower() != INTEREST_COST_ITEM_NAME.lower())
            & (fixed_df['Category'].isin(fixed_categories))
        ]
        r.total_allocated_fixed_cost_usd = float(standard_fixed['MonthlyCost_USD'].sum())

    # 9. Interest (on everything we pre-finance)
    interest_base = (
        r.total_raw_cost_usd
        + r.total_variable_costs_incl_pallets_usd
        + r.total_allocated_fixed_cost_usd
        + r.total_logistics_cost_usd
        + r.total_unexpected_cost_usd
    )
    r.interest_cost_usd = interest_base * interest_rate
    r.fixed_cost_per_unit_usd = (
        r.total_allocated_fixed_cost_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    )

    # 10. COGS roll-up
    r.total_cogs_usd = (
        r.total_raw_cost_usd
        + r.total_variable_costs_incl_pallets_usd
        + r.total_allocated_fixed_cost_usd
        + r.interest_cost_usd
    )
    r.cogs_per_box_usd = r.total_cogs_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    r.total_net_weight_kg = r.net_weight_kg_per_box * quantity_boxes
    r.cogs_per_kg_usd = (
        r.total_cogs_usd / r.total_net_weight_kg if r.total_net_weight_kg > 0 else 0.0
    )

    # 11. Delivered cost (before rebate)
    r.total_delivered_cost_usd = (
        r.total_cogs_usd + r.total_logistics_cost_usd + r.total_unexpected_cost_usd
    )
    r.delivered_cost_per_box_usd = (
        r.total_delivered_cost_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    )
    r.delivered_cost_per_kg_net_usd = (
        r.total_delivered_cost_usd / r.total_net_weight_kg if r.total_net_weight_kg > 0 else 0.0
    )

    # 12. Rebate / fee on top
    r.rebate_percentage = rebate_percentage
    r.rebate_amount_usd = r.total_delivered_cost_usd * (rebate_percentage / 100.0)
    r.final_total_cost_usd = r.total_delivered_cost_usd + r.rebate_amount_usd
    r.final_cost_per_box_usd = (
        r.final_total_cost_usd / quantity_boxes if quantity_boxes > 0 else 0.0
    )

    return r
