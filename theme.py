# -*- coding: utf-8 -*-
"""Shared theming for the cogs_calculator Streamlit app.

Two themes are available:
- "light": Editorial Light — warm paper, deep ink, pink accent (default)
- "dark":  Dark Neon — near-black, neon pink + cyan accents

Single typeface (Inter) throughout for consistency; tabular numerals
on metric values so figures stay aligned. Call apply_theme() at the
top of each page. plot_style() returns plotly colors for the active theme.
"""
import streamlit as st


# Shared font stack and element reset used by both themes.
# Every readable surface is covered so that text in dropdowns, tables,
# alerts, expanders, code blocks and the sidebar nav never disappears.
_BASE_CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  {tokens}
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {{
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: var(--paper) !important;
  color: var(--ink) !important;
  letter-spacing: -0.005em;
}}
[data-testid="stAppViewContainer"] .main .block-container {{
  color: var(--ink);
}}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* Headings */
h1, h2, h3, h4, h5, h6 {{ color: var(--ink) !important; font-weight: 600; letter-spacing: -0.02em; }}
h1 {{ font-size: clamp(1.9rem, 3vw, 2.4rem); line-height: 1.1; margin: 12px 0 6px; }}
h1 em {{ font-style: normal; color: var(--pink); }}
h2 {{ font-size: 1.35rem; margin-top: 28px; }}
h3 {{ font-size: 1rem; color: var(--ink-soft) !important; margin-top: 20px; margin-bottom: 8px; font-weight: 600; letter-spacing: -0.01em; text-transform: none; }}
h4 {{ font-size: 0.95rem; font-weight: 600; color: var(--ink) !important; }}

/* Body text */
p, li, span, div {{ color: var(--ink); }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stText"] {{
  color: var(--ink) !important;
  font-size: 0.95rem;
  line-height: 1.55;
}}
a {{ color: var(--pink); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Widget labels — sentence case, gentle weight, readable size */
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
  font-family: "Inter", sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  color: var(--ink-soft) !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background: var(--sidebar-bg); border-right: 1px solid var(--rule); }}
[data-testid="stSidebar"] * {{ color: var(--ink) !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  color: var(--ink-soft) !important;
}}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span {{ color: var(--ink) !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
  color: var(--pink) !important; font-weight: 600;
}}

/* Horizontal rule */
hr {{ border: 0; border-top: 1px solid var(--rule); margin: 28px 0; }}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {{
  font-family: "Inter", sans-serif;
  font-weight: 600;
  font-size: 0.9rem;
  letter-spacing: 0;
  text-transform: none;
  border-radius: 4px;
  border: 1px solid var(--btn-border);
  background: var(--btn-bg);
  color: var(--btn-fg);
  padding: 9px 18px;
  transition: background 160ms ease, color 160ms ease, box-shadow 160ms ease, transform 120ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
  background: var(--pink);
  border-color: var(--pink);
  color: #ffffff;
  transform: translateY(-1px);
  box-shadow: var(--glow);
}}

/* Inputs */
input, textarea {{
  font-family: "Inter", sans-serif !important;
  background: var(--card) !important;
  color: var(--ink) !important;
  border-radius: 4px !important;
}}
[data-baseweb="input"], [data-baseweb="input"] input,
[data-baseweb="textarea"], [data-baseweb="textarea"] textarea {{
  background: var(--card) !important;
  color: var(--ink) !important;
  border-color: var(--rule-soft) !important;
}}
[data-testid="stNumberInput"] input {{
  font-feature-settings: "tnum";
}}
[data-testid="stNumberInput"] button {{
  background: var(--card) !important; color: var(--ink) !important;
  border-color: var(--rule-soft) !important;
}}

/* Selectbox + dropdown popover */
[data-baseweb="select"] > div {{ background: var(--card) !important; color: var(--ink) !important; border-color: var(--rule-soft) !important; }}
[data-baseweb="select"] * {{ color: var(--ink) !important; }}
[role="listbox"], [data-baseweb="popover"], [data-baseweb="menu"] {{
  background: var(--card) !important; color: var(--ink) !important;
  border: 1px solid var(--rule-soft) !important;
}}
[role="option"] {{ color: var(--ink) !important; }}
[role="option"]:hover {{ background: var(--paper-2) !important; }}
[role="option"][aria-selected="true"] {{ background: var(--paper-2) !important; color: var(--pink) !important; }}

/* Radio + checkbox */
[data-baseweb="radio"] *, [data-baseweb="checkbox"] * {{ color: var(--ink) !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{
  font-family: "Inter", sans-serif;
  font-weight: 500;
  font-size: 0.92rem;
  letter-spacing: 0;
  text-transform: none;
  color: var(--ink-muted);
  background: transparent;
  padding: 10px 16px;
  border-radius: 0;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--ink); }}
.stTabs [aria-selected="true"] {{
  color: var(--tab-active-fg) !important;
  border-bottom: 2px solid var(--tab-active-fg) !important;
  background: transparent !important;
  font-weight: 600;
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* DataFrames + DataEditor */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
  border: 1px solid var(--rule); border-radius: 4px; background: var(--card);
}}
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] * {{
  color: var(--ink) !important;
}}
[data-testid="stDataFrame"] [role="columnheader"], [data-testid="stDataEditor"] [role="columnheader"] {{
  font-family: "Inter", sans-serif !important;
  font-size: 0.8rem !important; font-weight: 600 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  color: var(--ink-soft) !important;
  background: var(--paper-2) !important;
}}
[data-testid="stDataFrame"] [role="gridcell"], [data-testid="stDataEditor"] [role="gridcell"] {{
  background: var(--card) !important;
  color: var(--ink) !important;
  border-color: var(--rule) !important;
}}

/* Alerts (info / success / error / warning) */
[data-testid="stAlert"] {{ border-radius: 4px; border: 1px solid var(--rule); background: var(--card); }}
[data-testid="stAlert"] * {{ color: var(--ink) !important; }}
[data-testid="stAlert"][kind="success"], [data-testid="stAlert"][data-baseweb*="success"] {{
  background: var(--alert-success-bg); border-left: 3px solid var(--up);
}}
[data-testid="stAlert"][kind="error"], [data-testid="stAlert"][data-baseweb*="error"] {{
  background: var(--alert-error-bg); border-left: 3px solid var(--down);
}}
[data-testid="stAlert"][kind="info"], [data-testid="stAlert"][data-baseweb*="info"] {{
  background: var(--alert-info-bg); border-left: 3px solid var(--pink);
}}
[data-testid="stAlert"][kind="warning"], [data-testid="stAlert"][data-baseweb*="warning"] {{
  background: var(--alert-warning-bg); border-left: 3px solid var(--amber);
}}

/* Metrics */
[data-testid="stMetric"] {{
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--pink);
  border-radius: 4px; padding: 14px 20px;
  box-shadow: var(--metric-shadow);
}}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {{
  font-family: "Inter", sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  color: var(--ink-soft) !important;
}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
  font-family: "Inter", sans-serif !important;
  font-feature-settings: "tnum";
  font-weight: 600 !important;
  font-size: 1.6rem !important;
  color: var(--ink) !important;
  letter-spacing: -0.02em;
}}
[data-testid="stMetricDelta"] * {{ color: var(--ink-soft) !important; }}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"], small {{
  font-family: "Inter", sans-serif;
  font-size: 0.82rem;
  color: var(--ink-muted) !important;
}}

/* Expanders */
[data-testid="stExpander"] {{ border: 1px solid var(--rule); border-radius: 4px; background: var(--card); }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--ink) !important; font-weight: 500; }}
[data-testid="stExpander"] details[open] > summary {{ border-bottom: 1px solid var(--rule); }}

/* Code */
code, pre, kbd {{
  background: var(--paper-2) !important;
  color: var(--ink) !important;
  border-radius: 3px;
  padding: 1px 6px;
  font-family: "Inter", "SF Mono", ui-monospace, monospace;
  font-feature-settings: "tnum";
}}
pre {{ padding: 12px 14px; }}

/* Tooltip */
[role="tooltip"] {{ background: var(--card) !important; color: var(--ink) !important; border: 1px solid var(--rule-soft) !important; }}

/* Divider text on sliders */
[data-testid="stSlider"] * {{ color: var(--ink) !important; }}

/* FX result card (used on the FX Calculator page) */
.fx-result-card {{
  background: var(--fx-card-bg);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--pink);
  border-radius: 4px;
  padding: 28px 32px;
  margin-top: 8px;
  box-shadow: var(--fx-card-shadow);
}}
.fx-result-label {{
  font-family: "Inter", sans-serif;
  font-size: 0.85rem;
  font-weight: 500;
  letter-spacing: 0;
  text-transform: none;
  color: var(--ink-soft);
  margin-bottom: 10px;
}}
.fx-result-value {{
  font-family: "Inter", sans-serif;
  font-feature-settings: "tnum";
  font-weight: 700;
  font-size: 2.6rem;
  color: var(--ink);
  letter-spacing: -0.02em;
  line-height: 1;
}}
.fx-result-sub {{
  font-family: "Inter", sans-serif;
  font-feature-settings: "tnum";
  font-size: 0.85rem;
  color: var(--ink-muted);
  margin-top: 14px;
  letter-spacing: 0;
}}
</style>
"""


_LIGHT_TOKENS = """
  --paper:   #fafaf7;
  --paper-2: #f1efe9;
  --paper-3: #e5e2d9;
  --card:    #ffffff;
  --sidebar-bg: #ffffff;
  --rule:      rgba(20, 16, 30, 0.10);
  --rule-soft: rgba(20, 16, 30, 0.18);
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
  --btn-bg: #15121f;
  --btn-fg: #ffffff;
  --btn-border: #15121f;
  --tab-active-fg: #15121f;
  --glow: 0 2px 10px rgba(225,29,116,0.22);
  --metric-shadow: none;
  --alert-success-bg: #ecf7f4;
  --alert-error-bg:   #fbeeee;
  --alert-info-bg:    #f5eef3;
  --alert-warning-bg: #faf1e2;
  --fx-card-bg: #ffffff;
  --fx-card-shadow: none;
"""


# Dark Neon: near-black paper, light ink everywhere, pink + cyan accents.
# Contrast is kept high; no large text depends on a glow to be readable.
_DARK_TOKENS = """
  --paper:   #0b0a12;
  --paper-2: #17162a;
  --paper-3: #22213a;
  --card:    #16152a;
  --sidebar-bg: #0f0e1c;
  --rule:      rgba(255, 255, 255, 0.10);
  --rule-soft: rgba(255, 255, 255, 0.22);
  --ink:       #f4f2fb;
  --ink-soft:  #c9c4dc;
  --ink-muted: #9a94b0;
  --ink-faint: #6e6885;
  --pink:   #ff3d8f;
  --amber:  #ffb347;
  --violet: #b487ff;
  --cyan:   #5cf0ff;
  --up:     #5cf0a8;
  --down:   #ff6b7f;
  --btn-bg: transparent;
  --btn-fg: #ff3d8f;
  --btn-border: #ff3d8f;
  --tab-active-fg: #5cf0ff;
  --glow: 0 0 14px rgba(255, 61, 143, 0.45);
  --metric-shadow: 0 0 0 1px rgba(255, 61, 143, 0.06);
  --alert-success-bg: rgba(92, 240, 168, 0.07);
  --alert-error-bg:   rgba(255, 107, 127, 0.08);
  --alert-info-bg:    rgba(92, 240, 255, 0.06);
  --alert-warning-bg: rgba(255, 179, 71, 0.07);
  --fx-card-bg: linear-gradient(135deg, rgba(255,61,143,0.08) 0%, #16152a 55%);
  --fx-card-shadow: 0 0 24px rgba(255, 61, 143, 0.10);
"""


THEMES = {
    "light": {
        "label": "Light · Editorial",
        "css": _BASE_CSS_TEMPLATE.format(tokens=_LIGHT_TOKENS),
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
        "css": _BASE_CSS_TEMPLATE.format(tokens=_DARK_TOKENS),
        "plot": {
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg":  "rgba(0,0,0,0)",
            "font_color": "#f4f2fb",
            "title_color": "#ff3d8f",
            "grid_color": "rgba(255,255,255,0.10)",
            "axis_line": "rgba(255,255,255,0.24)",
            "palette":      ["#ff3d8f", "#5cf0ff", "#b487ff", "#5cf0a8", "#ffb347", "#f4f2fb"],
            "line_palette": ["#ff3d8f", "#5cf0ff", "#5cf0a8"],
            "marker_edge": "#16152a",
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
            "<div style='font-size:0.78rem;font-weight:500;color:#6c6478;"
            "margin-bottom:6px;'>Appearance</div>",
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
