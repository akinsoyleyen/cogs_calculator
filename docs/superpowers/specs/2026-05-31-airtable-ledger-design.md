# Airtable Ledger — Design

**Date:** 2026-05-31
**Status:** Approved direction; pending spec review
**Feature:** One-click logging of the air-freight pricing matrix into an Airtable "big table"

## Goal

Add a one-click **"Log to ledger"** action to the COGS calculator that pushes the full
air-freight pricing matrix — every destination × every pallet tier, at the target-profit
sell price — into an Airtable table that accumulates across products and calculations.

## Background / current state

- The Streamlit app computes a per-calculation cost breakdown.
- In **Air** mode it renders a matrix (`cogs/matrix.py::build_air_matrix`):
  destinations × `{2, 4, 6, 10}` pallets, valued at `final_cost_per_box_usd`.
- When **Target Profit %** > 0, the app derives a sell-price matrix:
  `sell = cost × (1 + target_profit / 100)` (see `app.py` ~line 945).
- Output today is ephemeral: a CSV/Excel download, or an email via the Make.com webhook.
  There is **no persistent ledger** that accumulates results across calculations.
- Pattern to mirror: `cogs/github_writer.py` — a thin module that pushes data out using a
  token from `st.secrets`, triggered by a button. The new writer follows the same shape.

## Decisions

- **Store:** Airtable — already connected, clean REST API, the user's prior tool that
  "worked very well."
- **Row shape:** **Wide** (Option B) — one record per destination, pallet tiers as columns.
  Mirrors the email/matrix layout the user already works with.
- **Values in the pallet columns:** the **active matrix** — sell price when Target Profit > 0,
  otherwise raw cost. A `Price basis` field records which, so the numbers are never ambiguous.
- **One click** logs every destination of the current matrix (~11 rows) as one batch.
- **Canonical currency:** USD (the matrix is computed in USD; reporting-currency FX is applied
  only for on-screen display).

## Airtable schema — table **"COGS Ledger"**

| Field | Type | Notes |
|---|---|---|
| Logged At | Date (with time) | App-sent ISO timestamp |
| Batch ID | Single line text | Groups one click, e.g. `2026-05-31 14:30 · Cherries` |
| Product | Single line text | |
| Destination | Single line text | |
| 2 pallets | Currency (USD) | Sell price/box (or cost if no target profit) |
| 4 pallets | Currency (USD) | |
| 6 pallets | Currency (USD) | |
| 10 pallets | Currency (USD) | |
| Price basis | Single select | `Sell price` or `Cost` |
| Target profit % | Number (1 dp) | |
| Raw $/kg | Number (3 dp) | Context; with Target profit % lets cost be recovered |
| Rebate % | Number (1 dp) | |
| Fixed-cost mode | Single line text | |
| Boxes/pallet | Number (integer) | Packing density at log time |

Pallet columns are static (`2/4/6/10`) to match `cogs/matrix.py::PALLET_COUNTS`. If the tiers
ever change, the columns and this spec change with them — the accepted trade-off of the wide
shape.

## Data flow

1. User runs an **Air** calculation; the matrix renders (existing behavior).
2. User clicks **📌 Log to ledger** (new button in the matrix section).
3. The app builds one record per destination from the already-in-memory `active_matrix_df`,
   attaching the context fields + Batch ID + timestamp.
4. `cogs/airtable_writer.py::log_matrix(rows)` POSTs them to Airtable
   (batched ≤ 10 records/request, `typecast=true` so text values are accepted as-is).
5. On success: a toast "Logged N rows to ledger" + a link to the Airtable view.
   On failure: `st.error` with the Airtable status + message (no silent failure), matching
   `github_writer`'s behavior.

## Module API — `cogs/airtable_writer.py`

- `airtable_is_configured() -> bool` — true if `airtable_token`, `airtable_base_id`, and
  `airtable_table` secrets are present.
- `log_matrix(rows: list[dict]) -> int` — append records, return the count written.
  Raises `RuntimeError` with the API detail on failure.
- Internal helpers (`_headers`, `_url`) mirror `github_writer.py`.

## Config / secrets (`.streamlit/secrets.toml` + Streamlit Cloud Secrets)

```toml
airtable_token   = "patXXXXXXXXXXXXXX"   # personal access token, data.records:write on the base
airtable_base_id = "appXXXXXXXXXXXXXX"
airtable_table   = "COGS Ledger"
```

If absent, the button shows the same "add these secrets to enable" info banner used by the
Make.com and GitHub features. Local-only development without secrets degrades gracefully.

## Provisioning

Claude creates the base + "COGS Ledger" table with the schema above over the existing Airtable
connection. The user generates a PAT (records write on the base) and pastes the three secrets.
Base creation requires choosing a workspace — confirm at build time.

## Error handling

- No token → info banner; the button is a no-op (consistent with existing patterns).
- API error → surfaced via `st.error` with the Airtable status + message; a partial-batch
  failure reports how many rows made it.
- The button only appears when the matrix is available: Air mode + calculation done + valid
  product.

## Out of scope (YAGNI for v1)

- Reading the ledger back into the app (just link out to Airtable).
- De-duplication / upsert — each click is a fresh snapshot; duplicates are accepted.
- Logging non-Air (Container/Truck) calculations.
- Multi-currency storage — USD only; reporting-currency conversion can come later.
- Long-format (one row per destination × pallet) — explicitly chose wide.

## Open questions

- Which Airtable workspace to create the base in (confirm at build time).
- Whether to also store cost columns alongside sell price — default **no**; `Price basis`,
  `Raw $/kg`, and `Target profit %` make cost recoverable.
