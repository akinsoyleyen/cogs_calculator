"""Currency display formatters.

Reads the active display currency / FX rate / symbol from st.session_state.
app.py populates those keys after the sidebar selectbox renders.
"""
import streamlit as st


def _display_state():
    return (
        st.session_state.get("display_symbol", "$"),
        st.session_state.get("display_fx_rate", 1.0),
        st.session_state.get("display_currency", "USD"),
    )


def format_cost(usd_amount):
    """Convert a USD amount to the chosen display currency and format it."""
    symbol, fx_rate, _ = _display_state()
    if usd_amount is None:
        return f"{symbol}0.00"
    amt = float(usd_amount) * fx_rate
    return f"{symbol}{amt:,.2f}"


def format_cost_by_mode(usd_amount, _mode_unused=None):
    return format_cost(usd_amount)


def format_cost_usd_only(usd_amount):
    return f"${0.0 if usd_amount is None else float(usd_amount):,.2f}"


def format_cost_eur_only(usd_amount):
    return format_cost(usd_amount)
