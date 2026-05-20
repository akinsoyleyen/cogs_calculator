# COGS Calculator

A Streamlit-based **Cost of Goods Sold + Logistics calculator** for fresh-produce shipments. Everything is computed in USD; a single display-FX is applied at the end so results can be reported in any of 10 currencies (USD, EUR, GBP, TRY, CAD, CHF, JPY, AED, SGD, AUD).

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

A devcontainer is provided (`.devcontainer/devcontainer.json`) — open the repo in GitHub Codespaces or VS Code's Dev Containers extension and the environment is set up automatically.

## Pages

- **Main calculator** (`app.py`) — pick a product, set quantity / pallets, choose fixed-cost mode, get a per-box breakdown across COGS, Logistics, Total, Summary pie, and Profit Analysis tabs.
- **Catalogue** (`pages/1_Catalogue.py`) — inline editor for the seven CSV data files. Edit products, components, weights, packing, recipes, fixed costs, pallets, and air-freight rates without leaving the browser.
- **FX Calculator** (`pages/2_FX_Calculator.py`) — manual currency conversion using live rates from Frankfurter.

## Theme

`theme.py` exposes `apply_theme(page_name)` and `plot_style()`. Two themes ship: **Light** (default, SwingScope-inspired) and **Dark Neon**. Theme is selected via the sidebar on each page and persists across pages via `st.session_state`.

## Data files

All CSVs live at the repo root. Edit them through the Catalogue page or directly with a text editor.

| File | Purpose |
|---|---|
| `components.csv` | Per-component USD cost and weight (boxes, clamshells, labels, viyols, bags). |
| `product_recipe.csv` | Bill-of-materials: which components go into which product, in what quantity. |
| `product_weights.csv` | Product weight per unit (kg). Comma decimal separator. |
| `product_packing.csv` | Boxes per pallet for each product. |
| `fixed_costs.csv` | Per-batch overhead options (Primary, Secondary, etc.). See "Cost model" below. |
| `pallets.csv` | Pallet weights and per-pallet cost. |
| `air_freight_rates.csv` | Tiered air-freight pricing per destination, by minimum chargeable weight. |

## Cost model

The fixed-cost selector represents overhead allocated to **this shipment**, not the whole month. Options like "All ($1,400)", "Primary only ($200)", or "10% of total value" are interpreted per-batch — every app run is one batch.

The interest-rate input (default 5%) is added on top of intermediate cost. Rebates (when configured per-pallet) are added to the delivered cost rather than subtracted from revenue — i.e. a $20 delivered cost with a 10% rebate becomes $22 final cost.

## Streamlit Cloud secrets

FX rates use the public Frankfurter API and need no secrets.

The **Push matrix to Google Sheets** feature does — it writes a cost-per-box matrix (11 destinations × {2, 4, 6, 10} pallets) to a sheet you own. Setup:

1. In Google Cloud Console, create or pick a project and enable the **Sheets API** and **Drive API**.
2. Create a service account; download its JSON key.
3. Create the destination Google Sheet (or reuse one). Share it with the service-account email (`...@...iam.gserviceaccount.com`) with **Editor** permission.
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:
   - `sheet_url` — the full URL of the sheet
   - `[gcp_service_account]` — paste in the service-account JSON, key-for-key
5. Locally: `secrets.toml` lives under `.streamlit/` and is gitignored. On Streamlit Cloud, paste the same keys under Project Settings → Secrets.

Each push appends a timestamped block (metadata header + the matrix) to a worksheet called `COGS Matrix`. Old blocks are preserved; runs grow downwards.

## Modernization plan

A phased modernization plan lives at [docs/MODERNIZATION_PLAN.md](docs/MODERNIZATION_PLAN.md).
