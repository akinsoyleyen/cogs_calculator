import pandas as pd
import pytest

import cogs.airtable_writer as aw
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
    # Values chosen away from any .xx5 midpoint so the result is platform-independent.
    m = pd.DataFrame({2: [10.347], 4: [9.002], 6: [9.0], 10: [8.5]},
                     index=pd.Index(["BAH-TK"], name="Destination"))
    rows = build_ledger_rows(
        m, product="P", price_basis="Cost", target_profit_percent=0.0,
        raw_cost_per_kg=1.0, rebate_percentage=0.0, fixed_cost_mode="standard",
        boxes_per_pallet=100, logged_at_iso="2026-05-31T00:00:00", batch_id="b",
    )
    assert rows[0]["2 pallets"] == 10.35  # rounds up
    assert rows[0]["4 pallets"] == 9.0    # rounds down


def test_build_ledger_rows_concatenated_across_products():
    # The Season Pricing page logs many pack types in one batch by concatenating
    # build_ledger_rows per product. Product + raw cost must travel per row.
    rows = []
    for product, raw in (("Cherry 10 x 500g", 3.5), ("Apricot 16 x 350g", 2.8)):
        rows.extend(
            build_ledger_rows(
                _matrix(),
                product=product,
                price_basis="Cost",
                target_profit_percent=0.0,
                raw_cost_per_kg=raw,
                rebate_percentage=0.0,
                fixed_cost_mode="standard",
                boxes_per_pallet=100,
                logged_at_iso="2026-06-08T12:00:00",
                batch_id="2026-06-08 12:00 · season lot",
            )
        )

    assert len(rows) == 4  # 2 products × 2 destinations
    assert {r["Product"] for r in rows} == {"Cherry 10 x 500g", "Apricot 16 x 350g"}
    cherry = [r for r in rows if r["Product"] == "Cherry 10 x 500g"]
    assert all(r["Raw $/kg"] == 3.5 for r in cherry)


def test_airtable_is_configured_true(monkeypatch):
    monkeypatch.setattr(aw.st, "secrets", {
        "airtable_token": "patX",
        "airtable_base_id": "appX",
        "airtable_table": "COGS Ledger",
    })
    assert aw.airtable_is_configured() is True


def test_airtable_is_configured_false_when_missing(monkeypatch):
    monkeypatch.setattr(aw.st, "secrets", {"airtable_token": "patX"})
    assert aw.airtable_is_configured() is False


class _FakeResp:
    def __init__(self, ok=True, status_code=200, text="", payload=None):
        self.ok = ok
        self.status_code = status_code
        self.text = text
        self._payload = payload or {"records": []}

    def json(self):
        return self._payload


def test_log_matrix_batches_and_posts(monkeypatch):
    monkeypatch.setattr(aw.st, "secrets", {
        "airtable_token": "patX",
        "airtable_base_id": "appBASE",
        "airtable_table": "COGS Ledger",
    })
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResp(ok=True, payload={"records": json["records"]})

    monkeypatch.setattr(aw.requests, "post", fake_post)

    rows = [{"Destination": f"D{i}"} for i in range(11)]
    n = aw.log_matrix(rows)

    assert n == 11
    assert len(calls) == 2  # 10 + 1
    assert calls[0]["url"].endswith("/appBASE/COGS%20Ledger")
    assert calls[0]["headers"]["Authorization"] == "Bearer patX"
    assert calls[0]["json"]["typecast"] is True
    assert len(calls[0]["json"]["records"]) == 10
    assert calls[0]["json"]["records"][0] == {"fields": {"Destination": "D0"}}
    assert len(calls[1]["json"]["records"]) == 1


def test_log_matrix_empty_is_noop(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not POST for empty rows")
    monkeypatch.setattr(aw.requests, "post", boom)
    assert aw.log_matrix([]) == 0


def test_log_matrix_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(aw.st, "secrets", {
        "airtable_token": "patX",
        "airtable_base_id": "appBASE",
        "airtable_table": "T",
    })

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(ok=False, status_code=422, text="Unknown field name")

    monkeypatch.setattr(aw.requests, "post", fake_post)

    with pytest.raises(RuntimeError) as exc:
        aw.log_matrix([{"Destination": "D0"}])
    assert "422" in str(exc.value)


def test_log_matrix_reports_partial_count_when_later_batch_fails(monkeypatch):
    # The normal ~11-destination load is two batches (10 + 1). If the first
    # batch lands and the second fails, the error must say how many rows wrote
    # so the user knows a retry would duplicate them.
    monkeypatch.setattr(aw.st, "secrets", {
        "airtable_token": "patX",
        "airtable_base_id": "appBASE",
        "airtable_table": "T",
    })
    seen = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["n"] += 1
        if seen["n"] == 1:
            return _FakeResp(ok=True, payload={"records": json["records"]})
        return _FakeResp(ok=False, status_code=429, text="Rate limited")

    monkeypatch.setattr(aw.requests, "post", fake_post)

    rows = [{"Destination": f"D{i}"} for i in range(11)]
    with pytest.raises(RuntimeError) as exc:
        aw.log_matrix(rows)
    msg = str(exc.value)
    assert "after 10" in msg  # first batch of 10 landed before batch 2 failed
    assert "429" in msg
