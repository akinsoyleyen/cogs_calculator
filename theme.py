# -*- coding: utf-8 -*-
"""Shared theming for the cogs_calculator Streamlit app.

Two futuristic themes:
- "light": Gallery — warm paper with dot grain, black ink, 1px bento grid
  lines, wide grotesk display type (Syne / Space Grotesk), round buttons.
- "dark":  Orange Tech — near-black, #FF6A1A accent with glow, wide tech
  display type (Orbitron), thin-bordered cards, angled buttons.

Call apply_theme() at the top of each page (after require_login()).
plot_style() returns plotly colours for the active theme. stat_cards(),
hero() and section_label() are small HTML helpers shared by the pages.
"""
import html as _html

import streamlit as st

DEFAULT_THEME = "dark"

_FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@600;700;800"
    "&family=Orbitron:wght@500;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');"
)

# NOTE: token names (--pink, --ink, --rule, ...) are kept from the previous
# theme because app.py / pages use them in inline styles. --pink is now
# simply "the accent colour" of the active theme.
_BASE_CSS_TEMPLATE = """
<style>
{fonts}
:root {{
  {tokens}
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"] * {{
  font-family: var(--font-body);
}}
html, body, [data-testid="stAppViewContainer"] {{
  background-color: var(--paper) !important;
  background-image: var(--bg-texture) !important;
  background-size: var(--bg-texture-size);
  color: var(--ink) !important;
}}
.material-symbols-rounded, [data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"], [class*="material-symbols"] {{
  font-family: "Material Symbols Rounded" !important;
}}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; max-width: 1500px; }}

/* Headings */
h1, h2, h3, h4, h5, h6 {{
  color: var(--ink) !important; font-family: var(--font-display) !important;
  letter-spacing: var(--display-tracking); text-transform: var(--display-case);
}}
h1 {{ font-size: clamp(2rem, 4vw, 3.4rem) !important; font-weight: 800 !important; line-height: 1.02 !important; margin: 8px 0 10px; }}
.lx-em {{ font-style: normal; color: var(--pink) !important; text-shadow: var(--text-glow); font-size: inherit !important; font-family: inherit !important; }}
h1 em, h1 em *, .lx-big em {{ font-size: inherit !important; line-height: inherit !important; font-family: inherit !important; }}
h1 em {{ font-style: normal; color: var(--pink); text-shadow: var(--text-glow); }}
h2 {{ font-size: 1.45rem !important; font-weight: 700 !important; margin-top: 30px; }}
h3 {{ font-size: 1.05rem !important; font-weight: 700 !important; margin-top: 22px; margin-bottom: 8px; }}
h4 {{ font-size: 0.95rem !important; font-weight: 700 !important; }}
h1 a, h2 a, h3 a {{ display: none !important; }}

/* Body text */
p, li, span, div {{ color: var(--ink); }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em,
[data-testid="stText"] {{ color: var(--ink) !important; font-size: 0.95rem; line-height: 1.55; }}
a {{ color: var(--pink); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Widget labels: small uppercase nav-style */
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
  font-family: var(--font-body) !important; font-size: 0.72rem !important;
  font-weight: 600 !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important; color: var(--ink-soft) !important;
}}
[data-baseweb="radio"] label, [data-baseweb="radio"] p,
[data-baseweb="checkbox"] label, [data-baseweb="checkbox"] p,
[data-testid="stCheckbox"] p, [data-testid="stRadio"] [role="radiogroup"] p {{
  text-transform: none !important; letter-spacing: 0 !important; font-size: 0.88rem !important; font-weight: 500 !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background: var(--sidebar-bg) !important; border-right: 1px solid var(--rule-strong); }}
[data-testid="stSidebar"] * {{ color: var(--ink) !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: var(--ink-soft) !important; }}
[data-testid="stSidebarNav"] {{ border-bottom: 1px solid var(--rule-strong); padding-bottom: 8px; }}
[data-testid="stSidebarNav"] a span {{
  font-family: var(--font-body) !important; text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.74rem; font-weight: 600;
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--paper-2) !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: var(--pink) !important; }}

hr {{ border: 0; border-top: 1px solid var(--rule-strong); margin: 30px 0; }}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {{
  font-family: var(--font-body) !important; font-weight: 700; font-size: 0.78rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  border-radius: var(--btn-radius); clip-path: var(--btn-clip);
  border: 1px solid var(--btn-border); background: var(--btn-bg); color: var(--btn-fg);
  padding: 10px 22px; min-height: 42px;
  transition: background 160ms ease, color 160ms ease, box-shadow 160ms ease, transform 120ms ease;
}}
.stButton > button p, .stDownloadButton > button p, [data-testid="stFormSubmitButton"] button p {{ color: inherit !important; font-size: inherit; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
  background: var(--btn-hover-bg); border-color: var(--btn-hover-bg);
  color: var(--btn-hover-fg) !important; transform: translateY(-1px); box-shadow: var(--glow);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: var(--pink); border-color: var(--pink); color: var(--on-accent) !important; box-shadow: var(--glow);
}}

button[kind="primary"] *, [data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] *,
[data-testid="stSidebar"] button[kind="primary"] *, .stButton > button:hover *, .stDownloadButton > button:hover * {{
  color: var(--on-accent) !important;
}}
.stButton > button[kind="secondary"]:not(:hover) *, .stDownloadButton > button:not(:hover) * {{ color: var(--btn-fg) !important; }}

/* Inputs */
input, textarea {{ font-family: var(--font-body) !important; background: var(--field) !important; color: var(--ink) !important; }}
[data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="input"] input,
[data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="textarea"] textarea {{
  background: var(--field) !important; color: var(--ink) !important;
  border-color: var(--rule-soft) !important; border-radius: var(--field-radius) !important;
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
  border-color: var(--pink) !important; box-shadow: var(--focus-ring) !important;
}}
[data-testid="stNumberInput"] input {{ font-feature-settings: "tnum"; }}
[data-testid="stNumberInput"] button {{ background: var(--field) !important; color: var(--ink) !important; border-color: var(--rule-soft) !important; }}
[data-testid="stNumberInput"] button:hover {{ background: var(--pink) !important; color: var(--on-accent) !important; }}

/* Selectbox + dropdown popover */
[data-baseweb="select"] > div {{ background: var(--field) !important; color: var(--ink) !important; border-color: var(--rule-soft) !important; border-radius: var(--field-radius) !important; }}
[data-baseweb="select"] * {{ color: var(--ink) !important; }}
[role="listbox"], [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="popover"] ul {{
  background: var(--card) !important; color: var(--ink) !important; border: 1px solid var(--rule-soft) !important;
}}
[data-baseweb="popover"] * {{ font-family: var(--font-body); }}
[role="option"] {{ color: var(--ink) !important; }}
[role="option"]:hover {{ background: var(--paper-2) !important; }}
[role="option"][aria-selected="true"] {{ background: var(--paper-2) !important; color: var(--pink) !important; }}

[data-baseweb="tag"] {{ background: var(--pink) !important; border-radius: var(--chip-radius) !important; }}
[data-baseweb="tag"] * {{ color: var(--on-accent) !important; }}
[data-baseweb="radio"] *, [data-baseweb="checkbox"] * {{ color: var(--ink) !important; }}
[data-testid="stSlider"] * {{ color: var(--ink) !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--rule-strong); }}
.stTabs [data-baseweb="tab"] {{
  font-family: var(--font-display) !important; font-weight: 700; font-size: 0.8rem;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-muted);
  background: transparent; padding: 14px 26px; border-radius: 0;
  border-right: 1px solid var(--rule); border-bottom: 2px solid transparent; margin-bottom: -1px;
}}
.stTabs [data-baseweb="tab"] p {{ font-family: inherit !important; font-size: inherit !important; color: inherit !important; }}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--ink); }}
.stTabs [aria-selected="true"] {{
  color: var(--tab-active-fg) !important; border-bottom: 2px solid var(--pink) !important;
  background: var(--paper-2) !important; text-shadow: var(--text-glow);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* DataFrames + DataEditor (glide grid draws on canvas; see --grid-filter) */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
  border: 1px solid var(--rule-strong); border-radius: var(--card-radius); background: var(--card); padding: 2px;
}}
[data-testid="stDataFrame"] > div, [data-testid="stDataEditor"] > div {{ filter: var(--grid-filter); }}
[data-testid="stTable"] table {{ border-collapse: collapse; width: 100%; }}
[data-testid="stTable"] th {{ background: var(--paper-2); color: var(--ink-soft) !important; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.1em; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ border: 1px solid var(--rule) !important; padding: 6px 10px; }}

/* Alerts */
[data-testid="stAlert"] {{ border-radius: var(--card-radius); border: 1px solid var(--rule-soft); background: var(--card); }}
[data-testid="stAlert"] > div {{ background: transparent !important; }}
[data-testid="stAlert"] * {{ color: var(--ink) !important; }}
[data-testid="stAlertContentSuccess"] {{ border-left: 3px solid var(--up); padding-left: 10px; }}
[data-testid="stAlertContentError"] {{ border-left: 3px solid var(--down); padding-left: 10px; }}
[data-testid="stAlertContentInfo"] {{ border-left: 3px solid var(--pink); padding-left: 10px; }}
[data-testid="stAlertContentWarning"] {{ border-left: 3px solid var(--amber); padding-left: 10px; }}

/* Metrics → big stat cards */
[data-testid="stMetric"] {{
  background: var(--card); border: 1px solid var(--rule-strong); border-radius: var(--card-radius);
  padding: 18px 22px; box-shadow: var(--metric-shadow); position: relative;
}}
[data-testid="stMetric"]::before {{ content: ""; position: absolute; left: 0; top: 0; width: 28px; height: 2px; background: var(--pink); box-shadow: var(--glow); }}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {{
  font-family: var(--font-body) !important; font-size: 0.7rem !important; font-weight: 600 !important;
  letter-spacing: 0.14em !important; text-transform: uppercase !important; color: var(--ink-muted) !important;
}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
  font-family: var(--font-num) !important; font-feature-settings: "tnum"; font-weight: 800 !important;
  font-size: 2rem !important; color: var(--ink) !important; letter-spacing: var(--num-tracking);
}}
[data-testid="stMetricDelta"] * {{ color: var(--ink-soft) !important; }}

.stCaption, [data-testid="stCaptionContainer"], small {{ font-size: 0.8rem; color: var(--ink-muted) !important; }}
[data-testid="stCaptionContainer"] * {{ color: var(--ink-muted) !important; }}

/* Expanders */
[data-testid="stExpander"] details {{ border: 1px solid var(--rule-strong) !important; border-radius: var(--card-radius) !important; background: var(--card); }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--ink) !important; font-weight: 600; }}
[data-testid="stExpander"] summary p {{ text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.78rem !important; }}
[data-testid="stExpander"] details[open] > summary {{ border-bottom: 1px solid var(--rule); }}
[data-testid="stExpander"] summary:hover * {{ color: var(--pink) !important; }}

code, pre, kbd {{
  background: var(--paper-2) !important; color: var(--ink) !important; border-radius: 3px; padding: 1px 6px;
  font-family: "JetBrains Mono", ui-monospace, monospace !important; font-feature-settings: "tnum";
}}
pre {{ padding: 12px 14px; }}
[role="tooltip"], [data-baseweb="tooltip"] * {{ background: var(--card) !important; color: var(--ink) !important; }}

/* ---------- Ledger UI components (stat cards / hero / sections) ---------- */
.lx-grid {{
  display: grid; gap: 0; margin: 10px 0 18px;
  border: 1px solid var(--rule-strong); border-radius: var(--card-radius);
  background: var(--card); overflow: hidden;
}}
.lx-cell {{ padding: 20px 22px 18px; border-right: 1px solid var(--rule-strong); position: relative; min-width: 0; }}
.lx-cell:last-child {{ border-right: 0; }}
.lx-cell .lx-label {{
  font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-muted);
  font-weight: 600; display: flex; justify-content: space-between; align-items: center;
}}
.lx-cell .lx-label .lx-glyph {{ font-size: 1rem; color: var(--ink); font-weight: 300; }}
.lx-cell .lx-value {{
  font-family: var(--font-num); font-feature-settings: "tnum"; font-weight: 800;
  font-size: var(--stat-size); line-height: 1.02; letter-spacing: var(--num-tracking);
  color: var(--ink); margin: 12px 0 6px; overflow-wrap: anywhere;
}}
.lx-cell.lx-accent .lx-value {{ color: var(--pink); text-shadow: var(--text-glow); }}
.lx-cell .lx-hint {{ font-size: 0.78rem; color: var(--ink-muted); line-height: 1.35; }}
.lx-cell.lx-small .lx-value {{ font-size: 1.25rem; margin: 8px 0 4px; }}

.lx-section {{
  display: flex; align-items: center; gap: 10px; margin: 18px 0 8px;
  font-size: 0.66rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink-muted); font-weight: 700;
}}
.lx-section .lx-num {{
  display: inline-flex; align-items: center; justify-content: center; min-width: 24px; height: 24px; padding: 0 6px;
  border: 1px solid var(--rule-soft); border-radius: var(--chip-radius); clip-path: var(--chip-clip);
  font-family: var(--font-num); font-size: 0.68rem; color: var(--pink); letter-spacing: 0.05em;
}}
.lx-section .lx-line {{ flex: 1; height: 1px; background: var(--rule-strong); }}

.lx-hero {{
  display: grid; grid-template-columns: 1.3fr 1fr; border: 1px solid var(--rule-strong);
  border-radius: var(--card-radius); background: var(--card); overflow: hidden; margin: 8px 0 18px;
}}
.lx-hero > div {{ padding: 30px 32px; }}
.lx-hero > div + div {{ border-left: 1px solid var(--rule-strong); }}
.lx-hero .lx-big {{
  font-family: var(--font-display); font-weight: 800; font-size: clamp(2.2rem, 5vw, 4.4rem);
  line-height: 0.95; text-transform: var(--display-case); letter-spacing: var(--display-tracking); color: var(--ink);
}}
.lx-hero .lx-big em {{ font-style: normal; color: var(--pink); text-shadow: var(--text-glow); }}
.lx-hero p {{ color: var(--ink-soft) !important; max-width: 46ch; margin-top: 14px; }}
.lx-steps {{ list-style: none; padding: 0; margin: 0; }}
.lx-steps li {{ display: flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--rule); color: var(--ink-soft); font-size: 0.9rem; }}
.lx-steps li:last-child {{ border-bottom: 0; }}
.lx-steps b {{ font-family: var(--font-num); color: var(--pink); min-width: 28px; }}

.lx-masthead {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 20px;
  border-bottom: 1px solid var(--rule-strong); padding-bottom: 14px; margin-bottom: 18px; }}
.lx-masthead .lx-kicker {{ font-size: 0.66rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-muted); font-weight: 600; }}
.lx-brand {{ font-family: var(--font-display); font-weight: 800; font-size: 1.25rem; letter-spacing: var(--display-tracking);
  text-transform: var(--display-case); color: var(--ink); }}
.lx-brand span {{ color: var(--pink); text-shadow: var(--text-glow); }}
.lx-formula {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.8rem; color: var(--ink-muted);
  border: 1px dashed var(--rule-soft); border-radius: var(--card-radius); padding: 8px 12px; margin: 4px 0 12px; }}
.lx-formula b {{ color: var(--ink); font-weight: 600; }}

/* FX result card */
.fx-result-card {{ background: var(--card); border: 1px solid var(--rule-strong); border-radius: var(--card-radius);
  padding: 28px 32px; margin-top: 8px; box-shadow: var(--metric-shadow); }}
.fx-result-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 10px; }}
.fx-result-value {{ font-family: var(--font-num); font-feature-settings: "tnum"; font-weight: 800; font-size: 2.6rem;
  color: var(--ink); letter-spacing: var(--num-tracking); line-height: 1; }}
.fx-result-sub {{ font-size: 0.85rem; color: var(--ink-muted); margin-top: 14px; }}

@media (max-width: 900px) {{
  .lx-grid {{ grid-template-columns: 1fr !important; }}
  .lx-cell {{ border-right: 0; border-bottom: 1px solid var(--rule-strong); }}
  .lx-hero {{ grid-template-columns: 1fr; }}
  .lx-hero > div + div {{ border-left: 0; border-top: 1px solid var(--rule-strong); }}
}}
</style>
"""


# Light · Gallery — warm paper, dot grain, black ink, round buttons.
_LIGHT_TOKENS = """
  --font-body: "Space Grotesk", -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-display: "Syne", "Space Grotesk", sans-serif;
  --font-num: "Space Grotesk", sans-serif;
  --display-case: none;
  --display-tracking: -0.03em;
  --num-tracking: -0.04em;
  --stat-size: clamp(1.8rem, 3.2vw, 3rem);
  --paper:   #ebe9e3;
  --paper-2: #e1ded6;
  --paper-3: #d4d1c8;
  --card:    #efede8;
  --field:   #f6f5f1;
  --sidebar-bg: #e4e2db;
  --bg-texture: radial-gradient(rgba(20,20,20,0.07) 1px, transparent 1px);
  --bg-texture-size: 6px 6px;
  --rule:        rgba(17,17,17,0.10);
  --rule-soft:   rgba(17,17,17,0.22);
  --rule-strong: rgba(17,17,17,0.18);
  --ink:       #111111;
  --ink-soft:  #333230;
  --ink-muted: #6b6963;
  --ink-faint: #a5a29a;
  --pink:   #111111;
  --on-accent: #efede8;
  --amber:  #a8740a;
  --violet: #4a4a4a;
  --cyan:   #2f6f62;
  --up:     #2f6f62;
  --down:   #b33a2e;
  --btn-bg: #cfccc4;
  --btn-fg: #111111;
  --btn-border: transparent;
  --btn-hover-bg: #111111;
  --btn-hover-fg: #efede8;
  --btn-radius: 999px;
  --btn-clip: none;
  --chip-radius: 999px;
  --chip-clip: none;
  --card-radius: 0px;
  --field-radius: 0px;
  --tab-active-fg: #111111;
  --glow: none;
  --text-glow: none;
  --focus-ring: 0 0 0 1px #111111;
  --metric-shadow: none;
  --grid-filter: none;
"""

# Dark · Orange Tech — near black, orange glow, angled buttons.
_DARK_TOKENS = """
  --font-body: "Space Grotesk", -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-display: "Orbitron", "Space Grotesk", sans-serif;
  --font-num: "Orbitron", "Space Grotesk", sans-serif;
  --display-case: uppercase;
  --display-tracking: 0.04em;
  --num-tracking: 0.01em;
  --stat-size: clamp(1.5rem, 2.6vw, 2.5rem);
  --paper:   #0a0a0b;
  --paper-2: #17130f;
  --paper-3: #221a13;
  --card:    #121214;
  --field:   #19191c;
  --sidebar-bg: #0d0d0f;
  --bg-texture: radial-gradient(1200px 500px at 85% -10%, rgba(255,106,26,0.10), transparent 60%),
                linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  --bg-texture-size: 100% 100%, 48px 48px, 48px 48px;
  --rule:        rgba(255,255,255,0.08);
  --rule-soft:   rgba(255,255,255,0.20);
  --rule-strong: rgba(255,106,26,0.28);
  --ink:       #f3efe9;
  --ink-soft:  #c9c2b8;
  --ink-muted: #8f877c;
  --ink-faint: #5d574f;
  --pink:   #ff6a1a;
  --on-accent: #0a0a0b;
  --amber:  #ffb347;
  --violet: #ff9a5c;
  --cyan:   #ffd2a8;
  --up:     #55e0a0;
  --down:   #ff5a5a;
  --btn-bg: transparent;
  --btn-fg: #ff6a1a;
  --btn-border: #ff6a1a;
  --btn-hover-bg: #ff6a1a;
  --btn-hover-fg: #0a0a0b;
  --btn-radius: 0px;
  --btn-clip: polygon(12px 0, 100% 0, calc(100% - 12px) 100%, 0 100%);
  --chip-radius: 0px;
  --chip-clip: polygon(5px 0, 100% 0, calc(100% - 5px) 100%, 0 100%);
  --card-radius: 2px;
  --field-radius: 2px;
  --tab-active-fg: #ff6a1a;
  --glow: 0 0 18px rgba(255,106,26,0.45);
  --text-glow: 0 0 18px rgba(255,106,26,0.45);
  --focus-ring: 0 0 0 1px #ff6a1a, 0 0 12px rgba(255,106,26,0.35);
  --metric-shadow: 0 0 0 1px rgba(255,106,26,0.04), 0 10px 30px rgba(0,0,0,0.35);
  --grid-filter: invert(0.93) hue-rotate(180deg) saturate(1.4);
"""


THEMES = {
    "light": {
        "label": "Light",
        "css": _BASE_CSS_TEMPLATE.format(fonts=_FONTS, tokens=_LIGHT_TOKENS),
        "plot": {
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg":  "rgba(0,0,0,0)",
            "font_color": "#111111",
            "title_color": "#111111",
            "grid_color": "rgba(17,17,17,0.10)",
            "axis_line": "rgba(17,17,17,0.35)",
            "palette":      ["#111111", "#6b6963", "#a8740a", "#2f6f62", "#b33a2e", "#a5a29a"],
            "line_palette": ["#111111", "#a8740a", "#2f6f62"],
            "marker_edge": "#efede8",
        },
    },
    "dark": {
        "label": "Dark",
        "css": _BASE_CSS_TEMPLATE.format(fonts=_FONTS, tokens=_DARK_TOKENS),
        "plot": {
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg":  "rgba(0,0,0,0)",
            "font_color": "#f3efe9",
            "title_color": "#ff6a1a",
            "grid_color": "rgba(255,255,255,0.08)",
            "axis_line": "rgba(255,106,26,0.35)",
            "palette":      ["#ff6a1a", "#ffb347", "#f3efe9", "#8f877c", "#ff9a5c", "#55e0a0"],
            "line_palette": ["#ff6a1a", "#ffb347", "#f3efe9"],
            "marker_edge": "#121214",
        },
    },
}


def _inject(css: str) -> None:
    if hasattr(st, "html"):
        st.html(css)
    else:
        st.markdown(css, unsafe_allow_html=True)


def current_theme() -> str:
    if st.session_state.get("theme") not in THEMES:
        st.session_state.theme = DEFAULT_THEME
    return st.session_state.theme


def inject_theme_css() -> str:
    """Inject the active theme's CSS without rendering any widget."""
    key = current_theme()
    _inject(THEMES[key]["css"])
    return key


def apply_theme(key_suffix: str = "global") -> str:
    """Render the Light/Dark toggle at the top of the sidebar and inject CSS.

    Returns the active theme key ("light" or "dark").
    """
    current_theme()
    with st.sidebar:
        dark_on = st.toggle(
            "Dark mode",
            value=(st.session_state.theme == "dark"),
            key=f"theme_toggle_{key_suffix}",
            help="Switch between the Light (gallery) and Dark (orange tech) look.",
        )
        chosen_key = "dark" if dark_on else "light"
        # No st.rerun(): the CSS below is injected in this same run, and an
        # early rerun would discard the widget values rendered after it.
        st.session_state.theme = chosen_key
    return inject_theme_css()


def plot_style() -> dict:
    """Return plotly colour tokens for the active theme."""
    return THEMES[st.session_state.get("theme", DEFAULT_THEME)]["plot"]


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def stat_cards(items, small: bool = False) -> None:
    """Render a bento row of big-number stat cards.

    items: iterable of (label, value, hint) or (label, value, hint, accent).
    Values are inserted as HTML-escaped text.
    """
    items = list(items)
    n = len(items) or 1
    cells = []
    for it in items:
        label, value, hint = it[0], it[1], it[2]
        accent = len(it) > 3 and it[3]
        cls = "lx-cell" + (" lx-accent" if accent else "") + (" lx-small" if small else "")
        hint_html = f'<div class="lx-hint">{_html.escape(str(hint))}</div>' if hint else ""
        cells.append(
            f'<div class="{cls}"><div class="lx-label">{_html.escape(str(label))}'
            f'<span class="lx-glyph">+</span></div>'
            f'<div class="lx-value">{_html.escape(str(value))}</div>{hint_html}</div>'
        )
    st.markdown(
        f'<div class="lx-grid" style="grid-template-columns:repeat({n},minmax(0,1fr));">'
        f'{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def section_label(label: str, num: str | None = None, container=None) -> None:
    """Numbered uppercase section divider (works in sidebar via container)."""
    target = container if container is not None else st
    num_html = f'<span class="lx-num">{_html.escape(num)}</span>' if num else ""
    target.markdown(
        f'<div class="lx-section">{num_html}<span>{_html.escape(label)}</span>'
        f'<span class="lx-line"></span></div>',
        unsafe_allow_html=True,
    )


def masthead(kicker: str, title_html: str, right_html: str = "") -> None:
    """Page masthead. title_html is trusted HTML (may contain <em>)."""
    st.markdown(
        f'<div class="lx-masthead"><div><div class="lx-kicker">{_html.escape(kicker)}</div>'
        f'<h1 style="margin:6px 0 0">{title_html}</h1></div>'
        f'<div class="lx-kicker" style="text-align:right">{right_html}</div></div>',
        unsafe_allow_html=True,
    )
