# -*- coding: utf-8 -*-
import requests
import streamlit as st

DISPLAY_CURRENCIES = ["USD", "EUR", "GBP", "TRY", "CAD", "CHF", "JPY", "AED", "SGD", "AUD"]
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "TRY": "₺", "CAD": "C$",
    "CHF": "CHF ", "JPY": "¥", "AED": "د.إ ", "SGD": "S$", "AUD": "A$",
}
FALLBACK_CROSS_RATES_USD = {
    "USD": 1.0, "EUR": 0.90, "GBP": 0.78, "TRY": 38.00, "CAD": 1.38,
    "CHF": 0.90, "JPY": 148.0, "AED": 3.67, "SGD": 1.34, "AUD": 1.52,
}
FRANKFURTER_URL = "https://api.frankfurter.app/latest?from={base}&to={target}"

st.set_page_config(page_title="FX Calculator — Ledger", layout="wide", page_icon="◐")

_SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --paper:   #fafaf7;
  --paper-2: #f4f3ee;
  --card:    #ffffff;
  --rule:      rgba(20, 16, 30, 0.08);
  --rule-soft: rgba(20, 16, 30, 0.14);
  --ink:       #15121f;
  --ink-soft:  #3a3545;
  --ink-muted: #6c6478;
  --pink:   #e11d74;
  --up:     #0b8f7e;
  --down:   #c23b3b;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper) !important;
  color: var(--ink);
  font-family: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;
  letter-spacing: -0.005em;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
h1 { font-family: "Space Grotesk", sans-serif; font-weight: 600; font-size: clamp(2rem, 3.2vw, 2.6rem); letter-spacing: -0.03em; line-height: 1.05; color: var(--ink); margin: 12px 0 4px; }
h1 em { font-style: normal; color: var(--pink); }
h2 { font-family: "Space Grotesk", sans-serif; font-weight: 600; font-size: 1.3rem; letter-spacing: -0.015em; color: var(--ink); margin-top: 32px; }
h3 { font-family: "JetBrains Mono", monospace; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-muted); margin-top: 24px; margin-bottom: 8px; }
label, [data-testid="stWidgetLabel"] p {
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.66rem !important; font-weight: 500 !important;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-muted) !important;
}
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--rule); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 32px 0; }
.stButton > button, [data-testid="stFormSubmitButton"] button {
  font-family: "JetBrains Mono", monospace; font-weight: 600; font-size: 0.76rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  border-radius: 2px; border: 1px solid var(--ink); background: var(--ink); color: var(--paper);
  padding: 11px 18px; transition: background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background: var(--pink); border-color: var(--pink); color: #fff;
  transform: translateY(-1px); box-shadow: 0 2px 10px rgba(225,29,116,0.22);
}
input, textarea, [data-baseweb="input"] input { font-family: "Space Grotesk", sans-serif !important; background: var(--card) !important; border-radius: 2px !important; color: var(--ink) !important; }
[data-testid="stNumberInput"] input { font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum"; }
[data-testid="stAlert"] { border-radius: 2px; border: 1px solid var(--rule); background: var(--card); }
[data-testid="stAlert"][kind="info"]    { background: var(--paper-2); border-left: 2px solid var(--pink); }
[data-testid="stAlert"][kind="success"] { background: color-mix(in oklab, var(--up) 6%, var(--card)); border-left: 2px solid var(--up); }
[data-testid="stAlert"][kind="error"]   { background: color-mix(in oklab, var(--down) 6%, var(--card)); border-left: 2px solid var(--down); }
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--rule); border-left: 2px solid var(--pink);
  border-radius: 2px; padding: 16px 24px;
}
[data-testid="stMetricLabel"] p {
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.62rem !important; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--ink-muted) !important; font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
  font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum";
  font-weight: 500 !important; font-size: 1.8rem !important; color: var(--ink) !important;
  letter-spacing: -0.02em;
}
.stCaption, [data-testid="stCaptionContainer"] {
  font-family: "Space Grotesk", sans-serif; font-size: 0.8rem; color: var(--ink-muted);
}
.fx-result-card {
  background: var(--card); border: 1px solid var(--rule); border-left: 2px solid var(--pink);
  border-radius: 2px; padding: 28px 32px; margin-top: 8px;
}
.fx-result-label {
  font-family: "JetBrains Mono", monospace; font-size: 0.62rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 10px;
}
.fx-result-value {
  font-family: "JetBrains Mono", monospace; font-feature-settings: "tnum";
  font-size: 2.6rem; color: var(--ink); letter-spacing: -0.02em; line-height: 1;
}
.fx-result-sub {
  font-family: "JetBrains Mono", monospace; font-size: 0.72rem;
  color: var(--ink-muted); margin-top: 14px; letter-spacing: 0.04em;
}
</style>
"""
if hasattr(st, "html"):
    st.html(_SHARED_CSS)
else:
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_rate(base: str, target: str):
    if base == target:
        return 1.0, "native"
    try:
        r = requests.get(FRANKFURTER_URL.format(base=base, target=target), timeout=10)
        r.raise_for_status()
        data = r.json()
        rate = data.get("rates", {}).get(target)
        date = data.get("date")
        if rate:
            return float(rate), date
    except Exception:
        pass
    usd_to_base = FALLBACK_CROSS_RATES_USD.get(base)
    usd_to_target = FALLBACK_CROSS_RATES_USD.get(target)
    if usd_to_base and usd_to_target:
        return usd_to_target / usd_to_base, None
    return None, None


def fmt(amount: float, currency: str) -> str:
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{sym}{amount:,.2f}"


# --- Masthead ---
st.markdown(
    "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.66rem;color:var(--ink-muted);"
    "letter-spacing:0.24em;text-transform:uppercase;margin-bottom:-4px;'>"
    "FX · Quick Conversion</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='margin-top:4px'>Convert <em>anything</em>.</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-family:\"Space Grotesk\",sans-serif;font-size:1rem;color:var(--ink-soft);"
    "max-width:64ch;margin-top:-2px;margin-bottom:24px;line-height:1.55;'>"
    "Enter an amount in one currency and see it in another. Rates are live from Frankfurter; "
    "override manually if you need to price against a specific rate."
    "</p>",
    unsafe_allow_html=True,
)

# --- Session defaults ---
if "fx_from" not in st.session_state:
    st.session_state.fx_from = "TRY"
if "fx_to" not in st.session_state:
    st.session_state.fx_to = "USD"

# --- Controls ---
col_amt, col_from, col_swap, col_to = st.columns([2, 1.3, 0.5, 1.3])

with col_amt:
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=1000.0,
        step=10.0,
        format="%.2f",
    )

with col_from:
    from_currency = st.selectbox("From", DISPLAY_CURRENCIES, key="fx_from")

with col_swap:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("⇄", help="Swap currencies", use_container_width=True):
        st.session_state.fx_from, st.session_state.fx_to = (
            st.session_state.fx_to,
            st.session_state.fx_from,
        )
        st.rerun()

with col_to:
    to_currency = st.selectbox("To", DISPLAY_CURRENCIES, key="fx_to")

# --- Rate source ---
live_rate, rate_date = fetch_rate(from_currency, to_currency)

col_src, col_refresh = st.columns([3, 1])
with col_src:
    source = st.radio(
        "Rate source",
        ["Live (Frankfurter)", "Manual override"],
        horizontal=True,
        label_visibility="visible",
    )
with col_refresh:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("Refresh live FX", use_container_width=True):
        fetch_rate.clear()
        st.rerun()

if source == "Manual override":
    default_rate = live_rate if live_rate else 1.0
    rate = st.number_input(
        f"Manual rate — 1 {from_currency} = ? {to_currency}",
        min_value=0.0,
        value=float(default_rate),
        step=0.0001,
        format="%.6f",
    )
    rate_note = "manual override"
else:
    if live_rate is None:
        st.error("Could not fetch a live rate and no fallback available for this pair.")
        st.stop()
    rate = live_rate
    rate_note = f"live · {rate_date}" if rate_date and rate_date != "native" else rate_date or "live"

# --- Result ---
converted = amount * rate
inverse = (1.0 / rate) if rate else 0.0

st.markdown(
    f"""
    <div class='fx-result-card'>
      <div class='fx-result-label'>{fmt(amount, from_currency)} equals</div>
      <div class='fx-result-value'>{fmt(converted, to_currency)}</div>
      <div class='fx-result-sub'>
        1 {from_currency} = {rate:,.6f} {to_currency}
        &nbsp;·&nbsp; 1 {to_currency} = {inverse:,.6f} {from_currency}
        &nbsp;·&nbsp; {rate_note}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# --- Quick reference table: 1 unit of From → all other currencies at live rates ---
st.markdown("<h3>Quick reference — 1 " + from_currency + " in other currencies</h3>", unsafe_allow_html=True)
ref_cols = st.columns(5)
for idx, cur in enumerate([c for c in DISPLAY_CURRENCIES if c != from_currency]):
    r, _ = fetch_rate(from_currency, cur)
    with ref_cols[idx % 5]:
        if r is None:
            st.metric(cur, "—")
        else:
            st.metric(cur, fmt(r, cur))

st.caption(
    "Frankfurter is a free ECB-sourced rate API. Rates refresh at most once an hour; "
    "hit Refresh live FX to clear the cache."
)
