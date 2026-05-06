"""Exchange-rate constants and live-fetch helper.

Single source of truth for the 10 display currencies, fallback rates,
the Frankfurter API endpoint, and the cached USD->target rate fetch.
"""
import requests
import streamlit as st


DISPLAY_CURRENCIES = ["USD", "EUR", "GBP", "TRY", "CAD", "CHF", "JPY", "AED", "SGD", "AUD"]

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "TRY": "₺", "CAD": "C$",
    "CHF": "CHF ", "JPY": "¥", "AED": "د.إ ", "SGD": "S$", "AUD": "A$"
}

FALLBACK_USD_RATES = {  # USD -> target, used when Frankfurter fails
    "USD": 1.0, "EUR": 0.90, "GBP": 0.78, "TRY": 38.00, "CAD": 1.38,
    "CHF": 0.90, "JPY": 148.0, "AED": 3.67, "SGD": 1.34, "AUD": 1.52
}

FRANKFURTER_URL_TEMPLATE = "https://api.frankfurter.app/latest?from=USD&to={target}"


@st.cache_data(ttl=3600)
def get_usd_to_target_rate(target_currency: str):
    """Fetch live USD -> target rate from Frankfurter. Returns (rate, date|None)."""
    if target_currency == "USD":
        return 1.0, "native"
    try:
        response = requests.get(FRANKFURTER_URL_TEMPLATE.format(target=target_currency), timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data.get('rates', {}).get(target_currency)
        date = data.get('date', None)
        if rate:
            return float(rate), date
    except Exception:
        pass
    return None, None
