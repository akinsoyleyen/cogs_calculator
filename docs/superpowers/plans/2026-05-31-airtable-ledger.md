# Airtable Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click "📌 Log to ledger" button that pushes the full air-freight pricing matrix (every destination × 2/4/6/10 pallets, at the target-profit sell price) into an Airtable "COGS Ledger" table.

**Architecture:** A new thin module `cogs/airtable_writer.py` (mirroring `cogs/github_writer.py`) provides a pure record-builder (`build_ledger_rows`) and an HTTP appender (`log_matrix`). The Streamlit app's matrix section gets a button that builds records from the already-computed `active_matrix_df` and POSTs them. Wide format: one Airtable record per destination, pallet tiers as columns.

**Tech Stack:** Python 3.12, Streamlit, pandas, `requests` (all already in the project), Airtable REST API. Tests via pytest (already installed) with `monkeypatch` (no new test libraries).

**Spec:** `docs/superpowers/specs/2026-05-31-airtable-ledger-design.md`

---

## File Structure

- **Create** `cogs/airtable_writer.py` — Airtable integration. Three public functions:
  - `airtable_is_configured() -> bool`
  - `build_ledger_rows(matrix_df, *, ...) -> list[dict]` (pure, the testable core)
  - `log_matrix(rows) -> int` (batched POST)
- **Create** `tests/test_airtable_writer.py` — unit tests for all three functions.
- **Create** `requirements-dev.txt` — records the pytest dev dependency.
- **Modify** `app.py` — add the "Log to ledger" button in the matrix section.
- **Modify** `README.md` — add an "Airtable ledger" setup section + secrets keys.
- **Provision** (runtime action, not a file) — create the Airtable base + "COGS Ledger" table.

---

## Task 1: Pure record-builder `build_ledger_rows`

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/test_airtable_writer.py`
- Create: `cogs/airtable_writer.py`

- [ ] **Step 1: Create the dev-dependency file**

Create `requirements-dev.txt`:

```
pytest>=7.4
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_airtable_writer.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_airtable_writer.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_ledger_rows'` (module does not exist yet).

- [ ] **Step 4: Write the minimal implementation**

Create `cogs/airtable_writer.py`:

```python
"""Append the air-freight pricing matrix to an Airtable "big table".

Mirrors cogs/github_writer.py: a thin, secrets-driven module triggered by a
button. `build_ledger_rows` is a pure transform (matrix -> Airtable records)
so it is unit-testable without network or Streamlit.

Requires three secrets in .streamlit/secrets.toml:
- airtable_token    (personal access token, data.records:write on the base)
- airtable_base_id  (e.g. "appXXXXXXXXXXXXXX")
- airtable_table    (e.g. "COGS Ledger")
"""
from urllib.parse import quote

import requests
import streamlit as st

_API = "https://api.airtable.com/v0"
_BATCH = 10  # Airtable accepts at most 10 records per create request.


def build_ledger_rows(
    matrix_df,
    *,
    product,
    price_basis,
    target_profit_percent,
    raw_cost_per_kg,
    rebate_percentage,
    fixed_cost_mode,
    boxes_per_pallet,
    logged_at_iso,
    batch_id,
):
    """Turn an active pricing matrix into one Airtable record dict per destination.

    matrix_df: index = destination names, columns = pallet counts (2,4,6,10),
    values = price per box (USD). Returns a list of plain dicts whose keys match
    the "COGS Ledger" field names.
    """
    rows = []
    for dest, row in matrix_df.iterrows():
        record = {
            "Logged At": logged_at_iso,
            "Batch ID": batch_id,
            "Product": product,
            "Destination": str(dest),
            "Price basis": price_basis,
            "Target profit %": target_profit_percent,
            "Raw $/kg": raw_cost_per_kg,
            "Rebate %": rebate_percentage,
            "Fixed-cost mode": fixed_cost_mode,
            "Boxes/pallet": boxes_per_pallet,
        }
        for col in matrix_df.columns:
            record[f"{int(col)} pallets"] = round(float(row[col]), 2)
        rows.append(record)
    return rows
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_airtable_writer.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt tests/test_airtable_writer.py cogs/airtable_writer.py
git commit -m "Add build_ledger_rows: matrix -> Airtable records"
```

---

## Task 2: `airtable_is_configured`

**Files:**
- Modify: `tests/test_airtable_writer.py`
- Modify: `cogs/airtable_writer.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_airtable_writer.py`:

```python
import cogs.airtable_writer as aw


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_airtable_writer.py -k configured -v`
Expected: FAIL with `AttributeError: module 'cogs.airtable_writer' has no attribute 'airtable_is_configured'`.

- [ ] **Step 3: Implement**

Add to `cogs/airtable_writer.py` (below the imports, above `build_ledger_rows`):

```python
def airtable_is_configured() -> bool:
    if not hasattr(st, "secrets"):
        return False
    try:
        return (
            bool(st.secrets.get("airtable_token"))
            and bool(st.secrets.get("airtable_base_id"))
            and bool(st.secrets.get("airtable_table"))
        )
    except Exception:
        return False
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_airtable_writer.py -k configured -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_airtable_writer.py cogs/airtable_writer.py
git commit -m "Add airtable_is_configured secrets check"
```

---

## Task 3: `log_matrix` — batched POST to Airtable

**Files:**
- Modify: `tests/test_airtable_writer.py`
- Modify: `cogs/airtable_writer.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_airtable_writer.py`:

```python
import pytest


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_airtable_writer.py -k log_matrix -v`
Expected: FAIL with `AttributeError: module 'cogs.airtable_writer' has no attribute 'log_matrix'`.

- [ ] **Step 3: Implement**

Add to the end of `cogs/airtable_writer.py`:

```python
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {st.secrets['airtable_token']}",
        "Content-Type": "application/json",
    }


def _url() -> str:
    base_id = st.secrets["airtable_base_id"]
    table = st.secrets["airtable_table"]
    return f"{_API}/{base_id}/{quote(table)}"


def log_matrix(rows: list[dict]) -> int:
    """Append `rows` (one dict per destination) to Airtable.

    Returns the number of records written. Raises RuntimeError on API failure.
    """
    if not rows:
        return 0
    url = _url()
    headers = _headers()
    written = 0
    for i in range(0, len(rows), _BATCH):
        batch = rows[i : i + _BATCH]
        payload = {"records": [{"fields": r} for r in batch], "typecast": True}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if not resp.ok:
            raise RuntimeError(
                f"Airtable write failed ({resp.status_code}): {resp.text[:300]}"
            )
        written += len(resp.json().get("records", batch))
    return written
```

- [ ] **Step 4: Run the whole test file to verify all pass**

Run: `python -m pytest tests/test_airtable_writer.py -v`
Expected: PASS (all tests, 7 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_airtable_writer.py cogs/airtable_writer.py
git commit -m "Add log_matrix: batched Airtable record append"
```

---

## Task 4: Wire the "Log to ledger" button into `app.py`

**Files:**
- Modify: `app.py` (matrix section — insert immediately before the line `        # --- Send via Make.com webhook ---`, which is inside the matrix `try:` block where `active_matrix_df`, `_matrix_is_sell`, `matrix_product_id`, `target_profit_percent`, `rebate_rate_input`, `raw_cost_per_kg_try`, `fixed_cost_mode`, and `_bpp` are all in scope)

- [ ] **Step 1: Insert the button block**

In `app.py`, find this line (inside the `try:` that builds the matrix):

```python
        # --- Send via Make.com webhook ---
```

Insert the following **immediately above** that line (same 8-space indentation):

```python
        # --- Log to ledger (Airtable) ---
        st.markdown("---")
        st.subheader("📌 Log to ledger (Airtable)")
        from cogs.airtable_writer import (
            airtable_is_configured,
            build_ledger_rows,
            log_matrix,
        )

        if not airtable_is_configured():
            st.info(
                "Add `airtable_token`, `airtable_base_id`, and `airtable_table` to "
                "`.streamlit/secrets.toml` to enable one-click logging. See README."
            )
        else:
            st.caption(
                "Logs one row per destination (with 2/4/6/10-pallet "
                f"{'sell prices' if _matrix_is_sell else 'costs'}) to the COGS Ledger."
            )
            if st.button("Log matrix to ledger"):
                try:
                    _now = datetime.now()
                    _rows_out = build_ledger_rows(
                        active_matrix_df,
                        product=matrix_product_id,
                        price_basis="Sell price" if _matrix_is_sell else "Cost",
                        target_profit_percent=round(float(target_profit_percent), 2),
                        raw_cost_per_kg=round(float(raw_cost_per_kg_try), 4),
                        rebate_percentage=round(float(rebate_rate_input), 2),
                        fixed_cost_mode=fixed_cost_mode,
                        boxes_per_pallet=_bpp,
                        logged_at_iso=_now.isoformat(timespec="seconds"),
                        batch_id=f"{_now:%Y-%m-%d %H:%M} · {matrix_product_id}",
                    )
                    with st.spinner("Logging to Airtable…"):
                        _n = log_matrix(_rows_out)
                    st.success(f"Logged {_n} rows to the ledger.")
                except Exception as e:
                    st.error(f"Could not log to ledger: {e}")

```

- [ ] **Step 2: Verify the file still compiles**

Run: `python -m py_compile app.py`
Expected: no output, exit code 0 (syntax is valid). `app.py` runs Streamlit at import time, so `py_compile` — not import — is the right syntax check.

- [ ] **Step 3: Run the full test suite (nothing should have broken)**

Run: `python -m pytest -v`
Expected: PASS (7 passed).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add 'Log to ledger' button to the pricing matrix"
```

---

## Task 5: Provision the Airtable base + "COGS Ledger" table

This is a runtime action performed via the connected Airtable tools (not a code change). It also requires one user action (generating a token).

- [ ] **Step 1: Pick / create the workspace and base**

Use the Airtable tools: list workspaces, then create a base named **"COGS Calculator"** (or reuse an existing one the user names). Record the returned **base id** (`appXXXXXXXXXXXXXX`).

- [ ] **Step 2: Create the "COGS Ledger" table with these fields**

| Field | Type |
|---|---|
| Logged At | Date (include time) |
| Batch ID | Single line text |
| Product | Single line text |
| Destination | Single line text |
| 2 pallets | Currency (USD, precision 2) |
| 4 pallets | Currency (USD, precision 2) |
| 6 pallets | Currency (USD, precision 2) |
| 10 pallets | Currency (USD, precision 2) |
| Price basis | Single select — options: `Sell price`, `Cost` |
| Target profit % | Number (precision 1) |
| Raw $/kg | Number (precision 3) |
| Rebate % | Number (precision 1) |
| Fixed-cost mode | Single line text |
| Boxes/pallet | Number (precision 0) |

Field names must match `build_ledger_rows` output **exactly** (the POST uses `typecast=true`, which coerces values but does **not** create missing fields).

- [ ] **Step 3: User generates a token (manual)**

Ask the user to create an Airtable **personal access token** with scope `data.records:write` (and `schema.bases:read` is harmless) limited to the new base, at https://airtable.com/create/tokens.

- [ ] **Step 4: Add secrets**

Add to `.streamlit/secrets.toml` locally (gitignored) and to the Streamlit Cloud Secrets panel:

```toml
airtable_token   = "patXXXXXXXXXXXXXX"
airtable_base_id = "appXXXXXXXXXXXXXX"   # from Step 1
airtable_table   = "COGS Ledger"
```

- [ ] **Step 5: Smoke-test the write path against the real base**

With secrets in place, run a one-off from the repo root:

```bash
python -c "import streamlit as st; from cogs import airtable_writer as aw; print(aw.airtable_is_configured())"
```

Expected: prints `True`. (A full end-to-end check happens in Task 6.)

---

## Task 6: Documentation + end-to-end verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a README section**

Add this section to `README.md` after the "Make.com email scenario" section:

```markdown
## Airtable ledger

The **📌 Log to ledger** button under the Air pricing matrix appends one row per
destination (2/4/6/10-pallet prices, plus product, target profit, rebate, and
raw cost) to an Airtable "COGS Ledger" table — your accumulating big table of
every product you have costed.

Setup (one-time, ~3 min):

1. Create an Airtable base with a table named `COGS Ledger` and the fields listed
   in `docs/superpowers/specs/2026-05-31-airtable-ledger-design.md`.
2. Generate a personal access token (`data.records:write` on that base) at
   https://airtable.com/create/tokens.
3. Add to `.streamlit/secrets.toml` (and the Streamlit Cloud Secrets panel):
   ```toml
   airtable_token   = "patXXXXXXXXXXXXXX"
   airtable_base_id = "appXXXXXXXXXXXXXX"
   airtable_table   = "COGS Ledger"
   ```

If the secrets are absent, the button shows a hint and does nothing — the rest of
the app is unaffected.
```

- [ ] **Step 2: Commit the docs**

```bash
git add README.md
git commit -m "Document the Airtable ledger feature"
```

- [ ] **Step 3: End-to-end manual verification**

1. Run: `streamlit run app.py`
2. Pick a product, set **Shipment type = Air**, enter a raw cost, set a **Target Profit %** (e.g. 15), and run the calculation.
3. Scroll to the pricing matrix → **📌 Log to ledger** → click **Log matrix to ledger**.
4. Expected: a green "Logged N rows to the ledger." toast (N = number of destinations).
5. Open the Airtable base and confirm: one row per destination, `Price basis = Sell price`, the 2/4/6/10-pallet columns populated, and the context columns (product, target profit, rebate, raw $/kg) filled.
6. Set **Target Profit % = 0**, recalculate, log again, and confirm the new rows have `Price basis = Cost`.

- [ ] **Step 4: Mark the plan complete**

All tasks done. The branch is ready for the finishing-a-development-branch skill (merge / PR / cleanup).

---

## Notes for the implementer

- **No new runtime dependencies.** `requests` and `pandas` are already in `requirements.txt`; pytest is dev-only (`requirements-dev.txt`).
- **Why `build_ledger_rows` is separate from the button:** Streamlit UI code is awkward to unit-test, so all logic that *can* be pure (matrix → records) lives in a pure function that is fully tested. The button is a thin shell.
- **`raw_cost_per_kg_try` naming:** despite the `_try` suffix this variable holds the USD raw cost per kg (legacy name — see `app.py` `base_inputs["raw_cost_per_kg_usd"] = raw_cost_per_kg_try`). Store it as-is.
- **Duplicates:** clicking twice logs twice by design (each click is a snapshot). No upsert in v1.
