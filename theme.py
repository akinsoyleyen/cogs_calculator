# -*- coding: utf-8 -*-
"""Shared theming for the cogs_calculator Streamlit app.

Two themes are available:
- "light": SwingScope Light (warm paper, deep ink, pink accent) — default
- "dark":  Dark Neon (near-black, neon pink + cyan, subtle glows)

Call apply_theme() at the top of each page. It injects the CSS,
renders a sidebar selector, and returns the active theme key.
plot_style() returns a dict of plotly colors for the active theme.
"""
import streamlit as st


_LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --paper:   #fafaf7;
  --paper-2: #f4f3ee;
  --paper-3: #ebe9e1;
  --card:    #ffffff;
  --rule:      rgba(20, 16, 30, 0.08);
  --rule-soft: rgba(20, 16, 30, 0.14);
  --ink:       #15121f;
  --ink-soft:  #3a3545;
  --ink-muted: #6c6478;
  --ink-faint: #a8a0b0;
  --pink:   #e11d74;
  --amber:  #b87d00;
  --violet: #6b3fd4;
  --cyan:   #0b8f7e;
  --up:     #0b8f7e;
  --down:   #c23b3b;
  --glow:   0 2px 10px rgba(225, 29, 116, 0.22);
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
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  font-family: "JetBrains Mono", monospace; font-weight: 600; font-size: 0.76rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  border-radius: 2px; border: 1px solid var(--ink); background: var(--ink); color: var(--paper);
  padding: 11px 18px; transition: background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background: var(--pink); border-color: var(--pink); color: #fff;
  transform: translateY(-1px); box-shadow: var(--glow);
}
input, textarea, [data-baseweb="input"] input { font-family: "Space Grotesk", sans-serif !important; background: var(--card) !important; border-radius: 2px !important; color: var(--ink) !important; }
[data-testid="stNumberInput"] input { font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum"; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] {
  font-family: "JetBrains Mono", monospace; font-weight: 500; font-size: 0.72rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-muted); background: transparent; padding: 12px 16px;
  border-radius: 0; border-bottom: 1px solid transparent; margin-bottom: -1px;
}
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom: 1px solid var(--ink) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--rule); border-radius: 2px; background: var(--card); }
[data-testid="stDataFrame"] [role="columnheader"], [data-testid="stDataEditor"] [role="columnheader"] {
  font-family: "JetBrains Mono", monospace !important; font-size: 0.62rem !important; font-weight: 600 !important;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-muted) !important; background: var(--paper-2) !important;
}
[data-testid="stAlert"] { border-radius: 2px; border: 1px solid var(--rule); background: var(--card); }
[data-testid="stAlert"][kind="success"] { background: color-mix(in oklab, var(--up) 6%, var(--card)); border-left: 2px solid var(--up); }
[data-testid="stAlert"][kind="error"]   { background: color-mix(in oklab, var(--down) 6%, var(--card)); border-left: 2px solid var(--down); }
[data-testid="stAlert"][kind="info"]    { background: var(--paper-2); border-left: 2px solid var(--pink); }
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
  font-weight: 500 !important; font-size: 1.5rem !important; color: var(--ink) !important;
  letter-spacing: -0.02em;
}
.stCaption, [data-testid="stCaptionContainer"] {
  font-family: "Space Grotesk", sans-serif; font-size: 0.8rem; color: var(--ink-muted);
}
.fx-result-card {
  background: var(--card); border: 1px solid var(--rule); border-left: 2px solid var(--pink);
  border-radius: 2px; padding: 28px 32px; margin-top: 8px;
}
.fx-result-label { font-family: "JetBrains Mono", monospace; font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 10px; }
.fx-result-value { font-family: "JetBrains Mono", monospace; font-feature-settings: "tnum"; font-size: 2.6rem; color: var(--ink); letter-spacing: -0.02em; line-height: 1; }
.fx-result-sub   { font-family: "JetBrains Mono", monospace; font-size: 0.72rem; color: var(--ink-muted); margin-top: 14px; letter-spacing: 0.04em; }
</style>
"""

_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --paper:   #07060d;
  --paper-2: #0f0d1a;
  --paper-3: #17152a;
  --card:    #121022;
  --rule:      rgba(255, 255, 255, 0.08);
  --rule-soft: rgba(255, 255, 255, 0.16);
  --ink:       #ece8f5;
  --ink-soft:  #bab3cc;
  --ink-muted: #7a7294;
  --ink-faint: #5a536e;
  --pink:   #ff2d8a;
  --cyan:   #00f0ff;
  --violet: #a06bff;
  --amber:  #ffb347;
  --up:     #00ff9d;
  --down:   #ff4569;
  --glow:   0 0 18px rgba(255, 45, 138, 0.55), 0 0 2px rgba(255, 45, 138, 0.8);
  --glow-cyan: 0 0 14px rgba(0, 240, 255, 0.45);
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at top left, #120f22 0%, var(--paper) 55%) !important;
  color: var(--ink);
  font-family: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;
  letter-spacing: -0.005em;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
h1 {
  font-family: "Space Grotesk", sans-serif; font-weight: 600;
  font-size: clamp(2rem, 3.2vw, 2.6rem); letter-spacing: -0.03em; line-height: 1.05;
  color: var(--ink); margin: 12px 0 4px;
}
h1 em {
  font-style: normal; color: var(--pink);
  text-shadow: 0 0 18px rgba(255, 45, 138, 0.5), 0 0 2px rgba(255, 45, 138, 0.9);
}
h2 { font-family: "Space Grotesk", sans-serif; font-weight: 600; font-size: 1.3rem; letter-spacing: -0.015em; color: var(--ink); margin-top: 32px; }
h3 { font-family: "JetBrains Mono", monospace; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: var(--cyan); margin-top: 24px; margin-bottom: 8px; text-shadow: 0 0 8px rgba(0, 240, 255, 0.25); }
label, [data-testid="stWidgetLabel"] p {
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.66rem !important; font-weight: 500 !important;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-muted) !important;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d0b1a 0%, #07060d 100%);
  border-right: 1px solid var(--rule);
}
hr { border: 0; border-top: 1px solid var(--rule); margin: 32px 0; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  font-family: "JetBrains Mono", monospace; font-weight: 600; font-size: 0.76rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  border-radius: 2px; border: 1px solid var(--pink); background: transparent; color: var(--pink);
  padding: 11px 18px; transition: background 160ms ease, transform 160ms ease, box-shadow 160ms ease, color 160ms ease;
  text-shadow: 0 0 6px rgba(255, 45, 138, 0.5);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background: var(--pink); color: #0b0918;
  transform: translateY(-1px); box-shadow: var(--glow);
  text-shadow: none;
}
input, textarea, [data-baseweb="input"] input {
  font-family: "Space Grotesk", sans-serif !important;
  background: var(--card) !important; border-radius: 2px !important;
  color: var(--ink) !important;
}
[data-baseweb="input"], [data-baseweb="select"] > div {
  background: var(--card) !important; border-color: var(--rule-soft) !important;
}
[data-testid="stNumberInput"] input { font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum"; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] {
  font-family: "JetBrains Mono", monospace; font-weight: 500; font-size: 0.72rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-muted); background: transparent; padding: 12px 16px;
  border-radius: 0; border-bottom: 1px solid transparent; margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 1px solid var(--cyan) !important;
  background: transparent !important;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border: 1px solid var(--rule); border-radius: 2px; background: var(--card);
}
[data-testid="stDataFrame"] [role="columnheader"], [data-testid="stDataEditor"] [role="columnheader"] {
  font-family: "JetBrains Mono", monospace !important; font-size: 0.62rem !important; font-weight: 600 !important;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--cyan) !important;
  background: var(--paper-2) !important;
}
[data-testid="stAlert"] { border-radius: 2px; border: 1px solid var(--rule); background: var(--card); }
[data-testid="stAlert"][kind="success"] { background: rgba(0, 255, 157, 0.06); border-left: 2px solid var(--up); }
[data-testid="stAlert"][kind="error"]   { background: rgba(255, 69, 105, 0.06); border-left: 2px solid var(--down); }
[data-testid="stAlert"][kind="info"]    { background: rgba(0, 240, 255, 0.05); border-left: 2px solid var(--cyan); }
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 2px solid var(--pink);
  border-radius: 2px; padding: 16px 24px;
  box-shadow: 0 0 0 1px rgba(255, 45, 138, 0.05), 0 0 20px rgba(255, 45, 138, 0.05);
}
[data-testid="stMetricLabel"] p {
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.62rem !important; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--ink-muted) !important; font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
  font-family: "JetBrains Mono", monospace !important; font-feature-settings: "tnum";
  font-weight: 500 !important; font-size: 1.5rem !important; color: var(--ink) !important;
  letter-spacing: -0.02em;
  text-shadow: 0 0 10px rgba(236, 232, 245, 0.12);
}
.stCaption, [data-testid="stCaptionContainer"] {
  font-family: "Space Grotesk", sans-serif; font-size: 0.8rem; color: var(--ink-muted);
}
.fx-result-card {
  background: linear-gradient(135deg, rgba(255,45,138,0.06) 0%, var(--card) 60%);
  border: 1px solid var(--rule);
  border-left: 2px solid var(--pink);
  border-radius: 2px; padding: 28px 32px; margin-top: 8px;
  box-shadow: 0 0 24px rgba(255, 45, 138, 0.10);
}
.fx-result-label { font-family: "JetBrains Mono", monospace; font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--cyan); margin-bottom: 10px; text-shadow: 0 0 6px rgba(0, 240, 255, 0.3); }
.fx-result-value { font-family: "JetBrains Mono", monospace; font-feature-settings: "tnum"; font-size: 2.6rem; color: var(--ink); letter-spacing: -0.02em; line-height: 1; text-shadow: 0 0 16px rgba(255, 45, 138, 0.35); }
.fx-result-sub   { font-family: "JetBrains Mono", monospace; font-size: 0.72rem; color: var(--ink-muted); margin-top: 14px; letter-spacing: 0.04em; }
</style>
"""


THEMES = {
    "light": {
        "label": "Light · SwingScope",
        "css": _LIGHT_CSS,
        "plot": {
            "paper_bg": "#ffffff",
            "plot_bg":  "#ffffff",
            "font_color": "#15121f",
            "title_color": "#15121f",
            "grid_color": "rgba(20,16,30,0.08)",
            "axis_line": "#d9d5e0",
            "palette":      ["#e11d74", "#6b3fd4", "#b87d00", "#0b8f7e", "#15121f", "#a8a0b0"],
            "line_palette": ["#e11d74", "#6b3fd4", "#0b8f7e"],
            "marker_edge": "#ffffff",
        },
    },
    "dark": {
        "label": "Dark · Neon",
        "css": _DARK_CSS,
        "plot": {
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg":  "rgba(0,0,0,0)",
            "font_color": "#ece8f5",
            "title_color": "#ff2d8a",
            "grid_color": "rgba(255,255,255,0.06)",
            "axis_line": "rgba(255,255,255,0.18)",
            "palette":      ["#ff2d8a", "#00f0ff", "#a06bff", "#00ff9d", "#ffb347", "#ece8f5"],
            "line_palette": ["#ff2d8a", "#00f0ff", "#00ff9d"],
            "marker_edge": "#121022",
        },
    },
}


def apply_theme(key_suffix: str = "global") -> str:
    """Render the theme selector in the sidebar and inject the active CSS.

    Returns the active theme key ("light" or "dark").
    """
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    with st.sidebar:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.62rem;"
            "letter-spacing:0.2em;text-transform:uppercase;color:#8a8498;margin-bottom:6px;'>"
            "Appearance</div>",
            unsafe_allow_html=True,
        )
        labels = [THEMES[k]["label"] for k in ("light", "dark")]
        keys = ["light", "dark"]
        current_idx = keys.index(st.session_state.theme)
        chosen_label = st.radio(
            "Theme",
            labels,
            index=current_idx,
            key=f"theme_radio_{key_suffix}",
            label_visibility="collapsed",
        )
        chosen_key = keys[labels.index(chosen_label)]
        if chosen_key != st.session_state.theme:
            st.session_state.theme = chosen_key
            st.rerun()

    css = THEMES[st.session_state.theme]["css"]
    if hasattr(st, "html"):
        st.html(css)
    else:
        st.markdown(css, unsafe_allow_html=True)
    return st.session_state.theme


def plot_style() -> dict:
    """Return plotly colour tokens for the active theme."""
    return THEMES[st.session_state.get("theme", "light")]["plot"]
