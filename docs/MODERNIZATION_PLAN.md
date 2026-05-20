# COGS Calculator — Modernization Plan (revised against V2.33)

## Context

This is a revision of an earlier plan written when the codebase was at V2.17. Since then the remote `origin/main` has advanced 18 commits to **V2.33** without any of the original plan's hygiene items being executed. Several plan phases are now obsolete because of unrelated refactors that landed in V2.18→V2.33:

- **V2.24** removed Docker.
- **V2.28** introduced `pages/1_Catalogue.py` (Catalogue cockpit, 330 lines).
- **V2.30** moved everything to a single-currency USD model with one display-FX at the end. This **eliminates the `global usd_to_eur_rate` problem** (Phase B1 of the old plan).
- **V2.30/2.29** also **deleted** `pages/2_Batch_Sales_Calculator.py` and `pages/3_Batch_Sales_Calculator_New.py`. The Make.com webhook URLs lived in those files, so **Phase C of the old plan is now N/A** — there are no webhooks left to move into `st.secrets`.
- **V2.31** added `pages/2_FX_Calculator.py` (147 lines, new manual FX page, no webhooks).
- **V2.32 / V2.33** added `theme.py` (402 lines) and a Dark Neon theme.

Net result: the structural refactor (Phase D) is more justified than before because total app code grew (`app.py` 990 + `theme.py` 402 + two pages = ~1,870 lines), but most of the *hygiene* items the original plan started with were never landed and one of them (committed `__pycache__/app.cpython-312.pyc`) actually got worse. This revised plan keeps what's still needed, drops what's obsolete, and re-orders so the highest-value fix lands first.

---

## Phase A — Setup hygiene (no behavior change)

Land first, in this order. Each is independently revertible.

### A1. Add `.gitignore` and untrack already-committed build artifacts (HIGHER PRIORITY than original plan)

File (new): `.gitignore`

Contents:

```
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.env
.streamlit/secrets.toml
.DS_Store
*.egg-info/
.vscode/
```

Then untrack files that were committed in error:

- `__pycache__/app.cpython-312.pyc` — present on origin/main, was not on V2.17. `git rm --cached -r __pycache__/`.
- `.DS_Store` — committed on origin (8196 bytes). `git rm --cached .DS_Store`.

The original plan was conservative ("add to ignore but do not delete"). Since the `.pyc` file is regenerated on every run and shouldn't ever have been committed, **untrack it** in the same commit as adding the ignore. `.DS_Store` similarly.

### A2. Pin `requirements.txt`

File: `requirements.txt`

Currently still 6 unpinned names: `streamlit`, `pandas`, `requests`, `plotly`, `fpdf2`, `openpyxl`.

Replace with exact `==` pins matching what Streamlit Cloud is currently resolving. To get the right versions:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip freeze > requirements.lock
```

Copy the relevant pins from `requirements.lock` into `requirements.txt`. Use exact pins (not `~=`) — Streamlit's minor releases have removed APIs and a compatible-release pin would still let those land.

### A3. Modernize `.devcontainer/devcontainer.json`

File: `.devcontainer/devcontainer.json`

- `mcr.microsoft.com/devcontainers/python:1-3.11-bullseye` → `mcr.microsoft.com/devcontainers/python:1-3.12-bookworm`. Bullseye is past regular EOL.
- Drop `--server.enableCORS false --server.enableXsrfProtection false` from `postAttachCommand`. Modern Streamlit auto-detects the proxy; the flags weaken default security. The command becomes simply `streamlit run app.py`.

### A4. Add `README.md`

File (new): `README.md`

Cover:
- One-paragraph description (Streamlit-based COGS + Logistics calculator, single-currency USD with display FX at the end).
- Local run: `pip install -r requirements.txt && streamlit run app.py`.
- CSV files it reads: `components.csv`, `product_recipe.csv`, `fixed_costs.csv`, `product_weights.csv`, `air_freight_rates.csv`, `pallets.csv`, `product_packing.csv`.
- Pages: main calculator, Catalogue cockpit (`pages/1_Catalogue.py`).
- Theme system: `theme.py` provides `apply_theme(page)` with Light + Dark Neon options.
- Cost model section: fixed-cost selector represents overhead allocated to *this shipment*, not the whole month.

The devcontainer.json `openFiles` array references `README.md` — this fixes a long-standing dangling reference.

### A5. Drop the UTF-8 magic comment

File: `app.py:1`

Remove `# -*- coding: utf-8 -*-`. Python 3 sources are UTF-8 by default since PEP 3120 (2008).

---

## Phase B — State cleanup (small refactor, behavior-preserving)

Only one item from the original Phase B survives. The old B1 (`global usd_to_eur_rate`) is **moot** — the V2.30 single-currency refactor removed that variable.

### B1 (revised). Convert `errors` list to a return value

File: `app.py` — locate the remaining `global errors` declaration (still present in V2.33; exact line numbers shifted by V2.30).

`errors = []` is module-level and mutated via `global errors` from inside `load_csv()`. Streamlit reruns the script top-to-bottom, so this resets every rerun — fine in practice but fragile and confusing.

Change: `load_csv()` returns `(dataframe, error_or_none)`. The CSV-loading block collects errors locally and renders them via `st.error()` if any. No `global` needed.

This is a prerequisite for Phase D4 (`cogs/data_loader.py`).

---

## ~~Phase C — Webhook secrets~~ (REMOVED — N/A on V2.33)

The Make.com webhook URLs lived in `pages/2_Batch_Sales_Calculator.py` and `pages/3_Batch_Sales_Calculator_New.py`. Both files were deleted before V2.30. New pages have no webhooks (verified by grep on `origin/main`).

No action needed. Future webhook integrations should pull URLs from `st.secrets` from day one.

---

## Phase D — Structure split (extract pure helpers)

Re-scope against V2.33's `app.py` (990 lines). The package layout below uses `theme.py` as precedent: a sibling top-level module, not a deep package.

New package: `cogs/` (sibling to `app.py` and `theme.py`).

### D1. `cogs/exchange.py`

Move from app.py: `FALLBACK_USD_RATES`, `FRANKFURTER_URL_TEMPLATE`, `get_usd_to_target_rate()` (around line 90 in V2.33 app.py — verify before editing). The legacy `exchange_rate = 1.0` no-op alias and `usd_to_eur_rate` derivation can stay in `app.py` as a backward-compat shim if anything still reads them, or be removed if grep shows zero callers.

The `@st.cache_data(ttl=3600)` decorator works fine in an imported module.

### D2. `cogs/formatters.py`

Move currency formatting helpers (locate via grep for `format_cost`, `display_symbol` consumers). After V2.30 these are simpler than in V2.17 because there's only one display currency at a time.

### D3. `cogs/exporters.py`

Move `create_csv_export()`, `create_excel_export()`, `get_download_link()`, `calculate_profit_margins()` (locate by name in V2.33 app.py).

### D4. `cogs/data_loader.py`

Move `load_csv()` and the CSV path constants `COMPONENTS_CSV` … `PACKING_CSV`. Depends on Phase B1 (returns errors instead of mutating global).

### D5. `app.py` becomes the orchestrator

Imports the four `cogs/` modules, runs sidebar setup, calls into helpers, builds tabs and exports. The COGS calculation block and the tab rendering **stay in `app.py`** — they're tightly coupled to many `st.session_state` keys and sidebar widgets, and splitting them is high-risk for low gain.

Target: `app.py` drops from 990 → ~600-700 lines.

### Out of scope (still deferred)

- Hardening broad `except Exception:` clauses in `load_csv()` and elsewhere — behavior-changing.
- Splitting the COGS calc block into a pure `compute_cogs()` function (the original plan's E4) — keep deferred until A–D land and a regression test exists.

---

## Phase E — Calculation logic review

Carry forward from the original plan. Re-verify line numbers against V2.33 before editing — they will have shifted.

### E0. Make interest rate a sidebar input

File: `app.py` — `INTEREST_RATE = 0.05` is still on line 18 of V2.33 app.py.

- Delete the constant.
- In the sidebar, alongside the existing FX section, add `interest_rate_percent = st.sidebar.number_input("Interest Rate (%):", min_value=0.0, value=5.0, step=0.1, format="%.2f")` and convert: `interest_rate = interest_rate_percent / 100.0`.
- Replace both call sites of `INTEREST_RATE` (grep to find current line numbers).
- Update the summary label from `"(Calc. Interest @ 5.0%)"` to use the chosen rate.
- Persist via `st.session_state["interest_rate_percent"]` so the Catalogue and FX pages can read it consistently.

### E1, E2, E3. Dead-code cleanup

Original plan items still apply but **line numbers must be re-verified** against V2.33 `app.py`:

- E1: dead dict filter on `summary_data` immediately overwritten — grep for `summary_data` writes.
- E2: duplicated `st.session_state['…'] = …` block — grep for `last_calc_product`.
- E3: stale `# ... (logic remains the same) ...` placeholders — grep for that exact string.

All three are no-behavior-change deletions.

### E-D1, E-D2 (documentation only)

- Add a "Cost model" section to README (per A4) explaining fixed costs are intentionally per-batch.
- Add a one-line comment at the rebate-on-top-of-delivered site stating the convention.

---

## Critical files

- `requirements.txt` — A2
- `.devcontainer/devcontainer.json` — A3
- `app.py` — A5, B1, D1-D4 (extractions), E0, E1-E3 (cleanup)
- New: `.gitignore`, `README.md`, `cogs/__init__.py`, `cogs/exchange.py`, `cogs/formatters.py`, `cogs/exporters.py`, `cogs/data_loader.py`
- Untracked: `__pycache__/`, `.DS_Store` (via `git rm --cached`)

---

## Verification

After each phase, run locally and walk the golden path before pushing.

1. **Install fresh:** clean venv → `pip install -r requirements.txt` resolves with the pins.
2. **Boot:** `streamlit run app.py` opens at `http://localhost:8501`. Logo loads, no console errors.
3. **FX rates:** sidebar shows `API (YYYY-MM-DD)` for USD→target. Change display currency and confirm rates re-fetch.
4. **COGS calc:** select a product, set quantity, choose pallet/box mode, confirm Total Cost / Box and the 5 tabs (COGS, Logistics, Total, Summary pie, Profit Analysis) render with non-zero values.
5. **Exports:** CSV and Excel download links produce files that open with expected columns.
6. **Catalogue + FX pages:** navigate via sidebar. Catalogue cockpit lets you edit products/components/logistics inline. FX calculator computes manual conversions.
7. **Theme switch:** toggle Dark Neon — colors and fonts change everywhere.
8. **Phase E sanity:** new Interest Rate sidebar input — change 5% → 7% and confirm COGS Subtotal moves up by ≈2% × interest_base. Summary label shows "(Calc. Interest @ 7.0%)".
9. **Numeric regression:** before refactor, run one canonical input (e.g. Cherry 10×500g, qty 750, 10 pallets, Air to DXB, default rates) and screenshot. After each phase, repeat — numbers must match exactly.
10. **Untracked artifacts:** confirm `git status` after a fresh `streamlit run app.py` shows no new `__pycache__/*.pyc` candidates for staging.

If any phase breaks the golden path, revert just that phase — they're designed to be independent commits.
