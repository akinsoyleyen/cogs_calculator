import pandas as pd

from cogs.airtable_writer import build_ledger_rows


def _matrix():
    return pd.DataFrame(
        {2: [10.0, 20.0], 4: [9.5, 19.0], 6: [9.0, 18.0], 10: [8.5, 17.0]},
        index=pd.Index(["BAH-TK", "DXB-EK"], name="Destination"),
    )


def test_build_ledger_rows_one_record_per_destination():
    rows = build_ledger_rows(
        _matrix(),
        product="Cherries",
        price_basis="Sell price",
        target_profit_percent=15.0,
        raw_cost_per_kg=3.5,
        rebate_percentage=5.0,
        fixed_cost_mode="standard",
        boxes_per_pallet=120,
        logged_at_iso="2026-05-31T14:30:00",
        batch_id="2026-05-31 14:30 · Cherries",
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["Destination"] == "BAH-TK"
    assert first["Product"] == "Cherries"
    assert first["2 pallets"] == 10.0
    assert first["4 pallets"] == 9.5
    assert first["6 pallets"] == 9.0
    assert first["10 pallets"] == 8.5
    assert first["Price basis"] == "Sell price"
    assert first["Target profit %"] == 15.0
    assert first["Raw $/kg"] == 3.5
    assert first["Rebate %"] == 5.0
    assert first["Fixed-cost mode"] == "standard"
    assert first["Boxes/pallet"] == 120
    assert first["Batch ID"] == "2026-05-31 14:30 · Cherries"
    assert first["Logged At"] == "2026-05-31T14:30:00"


def test_build_ledger_rows_rounds_to_cents():
    m = pd.DataFrame({2: [10.005], 4: [9.001], 6: [9.0], 10: [8.5]},
                     index=pd.Index(["BAH-TK"], name="Destination"))
    rows = build_ledger_rows(
        m, product="P", price_basis="Cost", target_profit_percent=0.0,
        raw_cost_per_kg=1.0, rebate_percentage=0.0, fixed_cost_mode="standard",
        boxes_per_pallet=100, logged_at_iso="2026-05-31T00:00:00", batch_id="b",
    )
    assert rows[0]["2 pallets"] == 10.0  # 10.005 is ~10.00499 in IEEE-754, rounds to 10.00
    assert rows[0]["4 pallets"] == 9.0
