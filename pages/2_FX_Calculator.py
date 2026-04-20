# -*- coding: utf-8 -*-
import requests
import streamlit as st

from theme import apply_theme

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

apply_theme("fx")


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
