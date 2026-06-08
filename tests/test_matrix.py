import pandas as pd
import pytest

from cogs.matrix import build_air_matrices, matrices_to_long


def _dfs():
    """Minimal in-memory frames mirroring the post-load shapes app.py builds
    (note components use CostPerUnit_USD, fixed uses MonthlyCost_USD)."""
    return {
        "product_weights_df": pd.DataFrame(
            {"ProductID": ["A 5kg", "B 5kg"], "NetWeightKG": [5.0, 4.0]}
        ),
        # Empty recipe → variable cost / packaging weight 0; components unused.
        "product_recipe_df": pd.DataFrame(
            columns=["ProductID", "ComponentName", "QuantityPerProduct"]
        ),
        "components_df": pd.DataFrame(
            columns=["ComponentName", "CostPerUnit_USD", "WeightKG"]
        ),
        "pallets_df": pd.DataFrame(
            {"PalletType": ["Air Pallet"], "CostUSD": [10.0], "WeightKG": [5.5]}
        ),
        "fixed_df": pd.DataFrame(
            {"CostItem": ["Customs"], "MonthlyCost_USD": [200.0], "Category": ["Primary"]}
        ),
        "air_rates_df": pd.DataFrame(
            {
                "Destination": ["BAH-TK", "BAH-TK", "DXB-TK", "DXB-TK"],
                "MinWeightKG": [0, 999, 0, 999],
                "PricePerKG_USD": [2.3, 2.1, 2.5, 2.4],
                "AirwayBill_USD": [115.5, 115.5, 115.5, 115.5],
            }
        ),
        "product_packing_df": pd.DataFrame(
            {"ProductID": ["A 5kg", "B 5kg"], "BoxesPerPallet": [100, 75]}
        ),
    }


def _base_common():
    return {
        "selected_pallet_type": "Air Pallet",
        "include_variable_costs": True,
        "fixed_cost_mode": "standard",
        "fixed_categories": ["Primary"],
        "interest_rate": 0.05,
        "unexpected_cost_usd": 0.0,
        "rebate_percentage": 0.0,
    }


def test_build_air_matrices_one_frame_per_product():
    matrices, errors = build_air_matrices(
        product_ids=["A 5kg", "B 5kg"],
        raw_cost_per_kg_usd_by_product={"A 5kg": 3.0, "B 5kg": 2.0},
        base_common=_base_common(),
        dfs=_dfs(),
    )
    assert set(matrices) == {"A 5kg", "B 5kg"}
    assert errors == {}
    assert list(matrices["A 5kg"].columns) == [2, 4, 6, 10]
    assert sorted(matrices["A 5kg"].index.tolist()) == ["BAH-TK", "DXB-TK"]


def test_build_air_matrices_isolates_bad_product():
    matrices, errors = build_air_matrices(
        product_ids=["A 5kg", "Ghost 9kg"],  # Ghost has no packing/weight
        raw_cost_per_kg_usd_by_product={"A 5kg": 3.0, "Ghost 9kg": 1.0},
        base_common=_base_common(),
        dfs=_dfs(),
    )
    assert "A 5kg" in matrices       # one bad product does not abort the batch
    assert "Ghost 9kg" not in matrices
    assert "Ghost 9kg" in errors


def test_build_air_matrices_price_is_per_product():
    dfs = _dfs()
    cheap, _ = build_air_matrices(
        product_ids=["A 5kg"],
        raw_cost_per_kg_usd_by_product={"A 5kg": 1.0},
        base_common=_base_common(),
        dfs=dfs,
    )
    pricey, _ = build_air_matrices(
        product_ids=["A 5kg"],
        raw_cost_per_kg_usd_by_product={"A 5kg": 9.0},
        base_common=_base_common(),
        dfs=dfs,
    )
    # Higher raw price → higher cost per box at the same destination/pallet count.
    assert pricey["A 5kg"].at["BAH-TK", 2] > cheap["A 5kg"].at["BAH-TK", 2]


def test_matrices_to_long_shape_and_columns():
    matrices, _ = build_air_matrices(
        product_ids=["A 5kg", "B 5kg"],
        raw_cost_per_kg_usd_by_product={"A 5kg": 3.0, "B 5kg": 2.0},
        base_common=_base_common(),
        dfs=_dfs(),
    )
    long = matrices_to_long(
        matrices,
        produce_of={"A 5kg": "Apple", "B 5kg": "Banana"},
        boxes_per_pallet_of={"A 5kg": 100, "B 5kg": 75},
    )
    assert len(long) == 4  # 2 products × 2 destinations
    assert list(long.columns) == [
        "Produce", "Pack type", "Boxes/pallet", "Destination",
        "2 pallets", "4 pallets", "6 pallets", "10 pallets",
    ]
    assert set(long["Produce"]) == {"Apple", "Banana"}
    assert set(long["Pack type"]) == {"A 5kg", "B 5kg"}


def test_matrices_to_long_multiplier_scales_values():
    matrices, _ = build_air_matrices(
        product_ids=["A 5kg"],
        raw_cost_per_kg_usd_by_product={"A 5kg": 3.0},
        base_common=_base_common(),
        dfs=_dfs(),
    )
    cost = matrices_to_long(matrices, produce_of={"A 5kg": "Apple"})
    sell = matrices_to_long(matrices, produce_of={"A 5kg": "Apple"}, multiplier=2.0)
    assert "Boxes/pallet" not in cost.columns  # omitted when not supplied
    c = float(cost["2 pallets"].iloc[0])
    s = float(sell["2 pallets"].iloc[0])
    assert s == pytest.approx(c * 2.0, abs=0.02)


def test_matrices_to_long_empty_is_safe():
    long = matrices_to_long({}, produce_of={})
    assert list(long.columns) == ["Produce", "Pack type", "Destination"]
    assert long.empty
