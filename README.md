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

## Cost matrix for emails

After running a calculation in **Air** mode, scroll to **📋 Cost matrix — air destinations × pallet count**. The app recomputes cost-per-box for every destination at 2 / 4 / 6 / 10 pallets, holding raw cost, rebate, and fixed-cost mode constant. Two ways to use it:

- **Manual paste** — expand *Copy for email (tab-separated)* and use the copy icon. Pasting into Gmail / Apple Mail / Outlook turns the TSV into a real HTML table.
- **Auto-send via Make.com** — fill in the recipient email under *Send pricing email* and click **Send to Make**. Requires the webhook setup below.

## Make.com email scenario

The **Send to Make** button POSTs a JSON payload to a Make.com custom webhook. Make does the actual email send, so credentials live with Make (your existing Gmail/Outlook connection) rather than in this repo.

Setup (one-time, ~5 min):

1. In Make, create a new scenario. First module: **Webhooks → Custom webhook → Add**. Copy the webhook URL.
2. Second module: **Gmail → Send an Email** (or your provider — Outlook, SMTP, etc.). Map:
   - *To* → `{{1.to_email}}`
   - *Subject* → `{{1.subject}}`
   - *Content type* → **HTML**
   - *Content* → `<p>{{1.note}}</p>{{1.matrix_html}}` (or any layout you like; `matrix_html` is the pre-rendered table)
3. Save & turn the scenario on. Test by clicking "Run once" in Make, then sending from the app — Make should pick up the request and trigger an email.
4. Add the webhook URL to `.streamlit/secrets.toml`:
   ```toml
   make_webhook_url = "https://hook.eu2.make.com/abcdef1234567890"
   ```
   Locally `.streamlit/secrets.toml` is gitignored. On Streamlit Cloud, paste the same key under Project Settings → Secrets.

Payload reference (what the webhook receives):

```json
{
  "from_app": "cogs_calculator",
  "date_iso": "2026-05-20T14:30:00",
  "product": "Cherries",
  "raw_cost_per_kg_usd": 3.5,
  "rebate_percentage": 5.0,
  "fixed_cost_mode": "All Costs (Primary+Secondary) ($X,XXX.XX)",
  "reporting_currency": "USD",
  "to_email": "buyer@example.com",
  "subject": "Cherries — pricing matrix (2026-05-20)",
  "note": "As discussed, here is the latest pricing.",
  "matrix_html": "<table>...</table>",
  "matrix_rows": [
    {"destination": "BAH-TK", "p2": 12.34, "p4": 11.22, "p6": 10.80, "p10": 10.50},
    ...
  ]
}
```

## Modernization plan

A phased modernization plan lives at [docs/MODERNIZATION_PLAN.md](docs/MODERNIZATION_PLAN.md).
