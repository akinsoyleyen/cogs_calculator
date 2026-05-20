"""Build the air-freight cost matrix: 11 destinations × {2,4,6,10} pallets."""
import pandas as pd

from cogs.calculator import compute_landed_cost

PALLET_COUNTS = (2, 4, 6, 10)


def render_matrix_html(matrix_df: pd.DataFrame) -> str:
    """Inline-styled HTML table — safe for Gmail/Outlook/Apple Mail bodies."""
    style_th = (
        "text-align:left;padding:8px 12px;"
        "border-bottom:2px solid #15121f;font-weight:600;"
        "font-family:Arial,sans-serif;font-size:13px;color:#15121f;"
    )
    style_th_num = style_th.replace("text-align:left", "text-align:right")
    style_td = (
        "padding:6px 12px;border-bottom:1px solid #e5e2d9;"
        "font-family:Arial,sans-serif;font-size:13px;color:#15121f;"
    )
    style_td_num = (
        "text-align:right;padding:6px 12px;border-bottom:1px solid #e5e2d9;"
        "font-family:Menlo,Consolas,monospace;font-size:13px;color:#15121f;"
    )

    head = "".join(
        [f'<th style="{style_th}">Destination</th>']
        + [f'<th style="{style_th_num}">{c}</th>' for c in matrix_df.columns]
    )
    body = []
    for dest, row in matrix_df.iterrows():
        cells = [f'<td style="{style_td}">{dest}</td>'] + [
            f'<td style="{style_td_num}">${float(v):,.2f}</td>' for v in row
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<table style="border-collapse:collapse;background:#fafaf7;">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def build_air_matrix(
    *,
    base_inputs: dict,
    dfs: dict,
    pallet_counts: tuple[int, ...] = PALLET_COUNTS,
) -> pd.DataFrame:
    """Recompute final_cost_per_box_usd for every (destination, pallet count) pair.

    base_inputs: keys from the calculator signature minus num_pallets, quantity_boxes,
    destination, shipment_type, manual_logistics_cost_usd (those are overridden here).

    dfs: {product_weights_df, product_recipe_df, components_df, pallets_df, fixed_df,
          air_rates_df, product_packing_df}.
    """
    product_packing_df = dfs["product_packing_df"]
    air_rates_df = dfs["air_rates_df"]

    if air_rates_df is None or air_rates_df.empty:
        raise ValueError("Air rates CSV is missing or empty — cannot build matrix.")

    product_id = str(base_inputs["selected_product"])
    packing = product_packing_df[product_packing_df["ProductID"] == product_id]
    if packing.empty or not pd.notna(packing["BoxesPerPallet"].iloc[0]):
        raise ValueError(
            f"Boxes/Pallet not configured for '{product_id}' in product_packing.csv — "
            "matrix needs an auto-calculable quantity."
        )
    boxes_per_pallet = int(packing["BoxesPerPallet"].iloc[0])
    if boxes_per_pallet <= 0:
        raise ValueError(f"Boxes/Pallet for '{product_id}' must be > 0.")

    destinations = sorted(air_rates_df["Destination"].unique().tolist())
    matrix = pd.DataFrame(
        index=destinations,
        columns=list(pallet_counts),
        dtype=float,
    )

    for dest in destinations:
        for n in pallet_counts:
            res = compute_landed_cost(
                quantity_boxes=n * boxes_per_pallet,
                num_pallets=n,
                shipment_type="Air",
                destination=dest,
                manual_logistics_cost_usd=0.0,
                product_weights_df=dfs["product_weights_df"],
                product_recipe_df=dfs["product_recipe_df"],
                components_df=dfs["components_df"],
                pallets_df=dfs["pallets_df"],
                fixed_df=dfs["fixed_df"],
                air_rates_df=air_rates_df,
                **{k: v for k, v in base_inputs.items()
                   if k not in {"shipment_type", "destination",
                                "manual_logistics_cost_usd", "num_pallets",
                                "quantity_boxes"}},
            )
            matrix.at[dest, n] = round(res.final_cost_per_box_usd, 4)

    matrix.index.name = "Destination"
    return matrix
