# -*- coding: utf-8 -*-
"""Login gate shared by every page.

Usage (right after st.set_page_config):

    from auth import require_login
    require_login()

Credentials are read from st.secrets:

    [auth]
    username = "admin"
    password_hash = "pbkdf2_sha256$600000$<salt_hex>$<hash_hex>"

Generate a hash with:  python auth.py hash
If the [auth] section is missing the app refuses to load (fails closed).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
import time

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 600_000
_SESSION_KEY = "auth_user"


def hash_password(password: str, iterations: int = _ITERATIONS, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations)
    return f"{_ALGO}${iterations}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, expected = stored.split("$")
        if algo != _ALGO:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), expected)
    except Exception:
        return False


def _auth_config():
    import streamlit as st
    try:
        cfg = st.secrets.get("auth")
    except Exception:
        cfg = None
    if not cfg or not cfg.get("username") or not cfg.get("password_hash"):
        return None
    return {"username": str(cfg["username"]), "password_hash": str(cfg["password_hash"])}


def _logout() -> None:
    import streamlit as st
    st.session_state.pop(_SESSION_KEY, None)


def require_login() -> str:
    """Block the page until the user is logged in. Returns the username."""
    import streamlit as st
    from theme import inject_theme_css

    cfg = _auth_config()
    if cfg is None:
        inject_theme_css()
        st.error(
            "Login is not configured, so the app is locked.\n\n"
            "Add this to `.streamlit/secrets.toml` on the server (see "
            "`.streamlit/secrets.toml.example` and the README), then restart:\n\n"
            "```toml\n[auth]\nusername = \"admin\"\n"
            "password_hash = \"pbkdf2_sha256$600000$<salt>$<hash>\"\n```\n\n"
            "Create a hash with `python auth.py hash`."
        )
        st.stop()

    user = st.session_state.get(_SESSION_KEY)
    if user == cfg["username"]:
        with st.sidebar:
            st.button(f"Log out · {user}", key="logout_btn", on_click=_logout, use_container_width=True)
        return user

    # ---- Login screen ----
    inject_theme_css()
    st.html("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
            "[data-testid='collapsedControl']{display:none!important;}</style>")
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown(
            '<div class="lx-hero" style="grid-template-columns:1fr;margin-top:6vh">'
            '<div><div class="lx-masthead" style="margin:0 0 6px;border:0;padding:0">'
            '<div class="lx-kicker">Aksun · Cost · Logistics · Margin</div>'
            '<div class="lx-kicker">Secure access</div></div>'
            '<div class="lx-big">Ledger<span class="lx-em">.</span></div>'
            '<p>Sign in to open the cost calculator, catalogue and season pricing.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in  →", type="primary", use_container_width=True)
        if submitted:
            ok_user = hmac.compare_digest(username.strip(), cfg["username"])
            ok_pass = verify_password(password, cfg["password_hash"])
            if ok_user and ok_pass:
                st.session_state[_SESSION_KEY] = cfg["username"]
                st.rerun()
            else:
                time.sleep(1.0)  # slow down brute force a little
                st.error("Wrong username or password.")
    st.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hash":
        import getpass
        pw = getpass.getpass("New password: ")
        if pw != getpass.getpass("Repeat: "):
            sys.exit("Passwords do not match.")
        print(hash_password(pw))
    else:
        print("Usage: python auth.py hash")
