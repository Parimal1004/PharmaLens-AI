"""
styles.py
----------
Visual design layer for PharmaLens AI (UI only - no application logic).

Light "clinical blue" theme:
  * Soft blue-tinted page background with a faint cyan / violet glow
  * White cards with thin borders and light shadows
  * Cyan accent for actions, violet used only in the brand gradient
  * Inter for text, JetBrains Mono for SMILES and numbers
  * Semantic red / orange / amber / green kept for risk levels

Public helpers keep the old names so app.py keeps working:
global_css, header_bar, risk_badge, score_pill, mono, shap_bar, footer_bar, COLORS
New helpers: section_header, card_header.
"""

import html as _html

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    # text / headings
    "primary": "#0F2A4A",
    "primary_hover": "#0B1F38",
    "on_surface": "#0F172A",
    "on_surface_variant": "#475569",
    "secondary": "#64748B",
    # accents
    "accent": "#0891B2",
    "accent_hover": "#0E7490",
    "accent_soft": "rgba(8,145,178,0.10)",
    "accent_border": "rgba(8,145,178,0.35)",
    "violet": "#7C3AED",
    "ice_blue": "#E0F2FE",
    "sky": "#0891B2",
    # backgrounds
    "bg": "#EEF4FB",
    "sidebar_bg": "#F7FAFE",
    "surface_1": "#FFFFFF",   # cards
    "surface": "#F5F8FC",     # nested tiles / wells
    "elevated": "#FFFFFF",    # buttons, inputs
    "border": "#DCE5F0",
    "border_strong": "#C3D0E0",
    # risk levels
    "high_solid": "#B91C1C", "high_bg": "#FEF2F2", "high_dot": "#EF4444", "high_border": "#FECACA",
    "medium_solid": "#C2410C", "medium_bg": "#FFF7ED", "medium_dot": "#F97316", "medium_border": "#FED7AA",
    "low_solid": "#A16207", "low_bg": "#FEFCE8", "low_dot": "#EAB308", "low_border": "#FEF08A",
    "safe_solid": "#15803D", "safe_bg": "#F0FDF4", "safe_dot": "#22C55E", "safe_border": "#BBF7D0",
}

RISK_STYLE = {
    "high": ("high_solid", "high_bg", "high_dot", "high_border", "High Risk"),
    "medium": ("medium_solid", "medium_bg", "medium_dot", "medium_border", "Medium Risk"),
    "low": ("low_solid", "low_bg", "low_dot", "low_border", "Low Risk"),
    "none": ("safe_solid", "safe_bg", "safe_dot", "safe_border", "Safe / No Flag"),
    "Low": ("safe_solid", "safe_bg", "safe_dot", "safe_border", "Low"),
    "Low-Moderate": ("low_solid", "low_bg", "low_dot", "low_border", "Low-Moderate"),
    "Moderate": ("medium_solid", "medium_bg", "medium_dot", "medium_border", "Moderate"),
    "High": ("high_solid", "high_bg", "high_dot", "high_border", "High"),
}


def _c(name):
    return COLORS[name]


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
def global_css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(900px 420px at 8% -8%, rgba(8,145,178,0.13), transparent 60%),
        radial-gradient(800px 420px at 100% 0%, rgba(124,58,237,0.10), transparent 60%),
        {_c('bg')} !important;
    background-attachment: fixed !important;
    color: {_c('on_surface')};
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}

[data-testid="stMainBlockContainer"], .block-container {{
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---- Typography ---- */
h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    color: {_c('primary')} !important;
    letter-spacing: -0.015em;
    font-weight: 700 !important;
}}
p, li, label, [data-testid="stMarkdownContainer"] {{
    font-family: 'Inter', sans-serif;
    color: {_c('on_surface')};
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: {_c('secondary')};
    font-size: 0.86rem;
}}
hr {{ border-color: {_c('border')} !important; margin: 1.25rem 0 !important; }}
code {{ color: {_c('accent_hover')}; background: {_c('surface')}; }}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background-color: {_c('sidebar_bg')};
    border-right: 1px solid {_c('border')};
}}
[data-testid="stSidebar"] > div {{ padding-top: 1.25rem; }}

/* ---- Cards ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {_c('surface_1')};
    border: 1px solid {_c('border')};
    border-radius: 1.1rem;
    box-shadow: 0 10px 30px rgba(15,42,74,0.08);
    padding: 0.9rem 1rem;
}}
/* nested tiles (candidates / alternatives) */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {_c('surface')};
    box-shadow: none;
    border-radius: 0.9rem;
    padding: 0.7rem 0.75rem;
}}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {{
    background-color: {_c('elevated')};
    color: {_c('on_surface')};
    border: 1px solid {_c('border_strong')};
    border-radius: 0.65rem;
    min-height: 2.5rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    box-shadow: none;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {_c('accent')};
    color: {_c('accent_hover')};
    background-color: {_c('elevated')};
}}
.stButton > button:focus-visible {{
    outline: 3px solid {_c('accent_soft')};
    outline-offset: 1px;
}}
/* primary action (Generate) */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, #0891B2 0%, #6366F1 100%);
    color: #FFFFFF;
    border: none;
    font-weight: 700;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(135deg, #0E7490 0%, #4F46E5 100%);
    color: #FFFFFF;
}}

/* ---- Sliders ---- */
[data-testid="stSlider"] [role="slider"] {{
    background-color: {_c('accent')} !important;
    box-shadow: 0 0 0 4px {_c('accent_soft')};
}}
[data-testid="stSliderThumbValue"], [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {{
    color: {_c('accent')} !important;
}}
[data-testid="stSlider"] label p {{ font-weight: 600; color: {_c('on_surface_variant')}; }}

/* ---- Expanders ---- */
[data-testid="stExpander"] {{
    border: 1px solid {_c('border')};
    border-radius: 0.85rem;
    background-color: {_c('surface')};
}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{
    font-weight: 600;
    color: {_c('primary')};
}}

/* ---- Alerts ---- */
[data-testid="stAlert"] {{ border-radius: 0.8rem; border: 1px solid {_c('border')}; }}

/* ---- Progress bar ---- */
[data-testid="stProgress"] > div > div {{ background-color: {_c('border')} !important; }}
[data-testid="stProgress"] > div > div > div > div {{
    background: linear-gradient(90deg, #0891B2, #7C3AED) !important;
}}

/* ---- Text input / SMILES entry ---- */
[data-testid="stTextInput"] input {{
    font-family: 'JetBrains Mono', monospace;
    min-height: 3.1rem;
    padding: 0.7rem 1rem;
    border-radius: 0.75rem;
    border: 1.5px solid {_c('border_strong')};
    background: {_c('surface')};
    color: {_c('on_surface')};
    font-size: 0.98rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
[data-testid="stTextInput"] input::placeholder {{ color: {_c('secondary')}; opacity: 0.8; }}
[data-testid="stTextInput"] input:hover {{ border-color: {_c('accent')}; }}
[data-testid="stTextInput"] input:focus {{
    border-color: {_c('accent')} !important;
    box-shadow: 0 0 0 4px {_c('accent_soft')} !important;
}}
[data-testid="stTextInput"] label p {{ color: {_c('on_surface_variant')} !important; font-weight: 600; }}
.pl-smiles-label {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.6rem 0 0.1rem;
    color: {_c('accent')};
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
.pl-smiles-label::before {{
    content: "";
    width: 0.5rem; height: 0.5rem;
    border-radius: 50%;
    background: {_c('accent')};
    box-shadow: 0 0 10px {_c('accent')};
}}
.pl-smiles-help {{ margin: 0 0 0.4rem; color: {_c('secondary')}; font-size: 0.86rem; }}

/* ---- Checkbox ---- */
[data-testid="stCheckbox"] label p {{ font-size: 0.86rem; color: {_c('on_surface_variant')}; }}

/* ---- Metrics ---- */
[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; color: {_c('primary')}; }}

/* ---- SMILES chip ---- */
.pl-mono {{
    font-family: 'JetBrains Mono', monospace;
    background-color: #FFFFFF;
    border: 1px solid {_c('border')};
    border-radius: 0.5rem;
    padding: 0.3rem 0.6rem;
    font-size: 0.78rem;
    color: {_c('on_surface_variant')};
    display: inline-block;
    word-break: break-all;
}}
.pl-mono.trunc {{
    display: block;
    max-width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    word-break: normal;
    margin: 0.35rem 0 0.5rem;
}}

/* ---- Section header ---- */
.pl-section {{
    display: flex; align-items: center; gap: 0.75rem;
    margin: 0.1rem 0 0.35rem;
}}
.pl-section .num {{
    width: 2rem; height: 2rem; border-radius: 0.6rem;
    display: grid; place-items: center;
    background: linear-gradient(135deg, #0891B2, #7C3AED);
    color: #FFFFFF; font-weight: 800; font-size: 0.95rem;
}}
.pl-section .title {{
    font-size: 1.25rem; font-weight: 700; color: {_c('primary')}; letter-spacing: -0.015em;
}}

/* ---- Card header row ---- */
.pl-cardhead {{
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
}}
.pl-cardhead .t {{ font-weight: 700; color: {_c('primary')}; font-size: 0.98rem; }}

/* ---- Footer ---- */
.pl-footer {{
    text-align: center;
    color: {_c('secondary')};
    font-size: 0.8rem;
    padding: 1rem 0.25rem 0.25rem;
    border-top: 1px solid {_c('border')};
    margin-top: 1.5rem;
}}
</style>
"""


# ---------------------------------------------------------------------------
# HTML snippet helpers
# ---------------------------------------------------------------------------
def header_bar(logo_svg_base64: str) -> str:
    return f"""
<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.75rem;
            padding:0.9rem 1.25rem; margin-bottom:1.5rem; background:{_c('surface_1')};
            border:1px solid {_c('border')}; border-radius:1.1rem; box-shadow:0 10px 30px rgba(15,42,74,0.08);">
  <div style="display:flex; align-items:center; gap:0.9rem;">
    <img src="data:image/svg+xml;base64,{logo_svg_base64}" style="height:46px; width:46px;" alt="PharmaLens AI logo" />
    <div style="line-height:1.15;">
      <div style="font-family:'Inter',sans-serif; font-size:1.45rem; font-weight:800; letter-spacing:-0.02em; color:{_c('primary')};">
        PharmaLens<span style="color:{_c('accent')};"> AI</span>
      </div>
      <div style="font-family:'Inter',sans-serif; font-size:0.7rem; font-weight:600; letter-spacing:0.14em;
                  text-transform:uppercase; color:{_c('secondary')}; margin-top:0.2rem;">
        Transparent Drug Discovery
      </div>
    </div>
  </div>
  <div style="display:inline-flex; align-items:center; gap:0.45rem; padding:0.35rem 0.85rem;
              border-radius:9999px; background:{_c('accent_soft')}; border:1px solid {_c('accent_border')};">
    <span style="width:8px;height:8px;border-radius:50%;background:{_c('accent')}; box-shadow:0 0 8px {_c('accent')};"></span>
    <span style="font-family:'Inter',sans-serif; font-size:0.78rem; font-weight:600; color:{_c('accent_hover')};">
      Explainable AI &middot; No Black Box
    </span>
  </div>
</div>
"""


def section_header(number: int, title: str) -> str:
    return f'<div class="pl-section"><div class="num">{number}</div><div class="title">{_html.escape(title)}</div></div>'


def card_header(title: str, right_html: str = "") -> str:
    return f'<div class="pl-cardhead"><span class="t">{_html.escape(title)}</span>{right_html.strip()}</div>'


def risk_badge(risk_level: str, label_override: str | None = None) -> str:
    solid, bg, dot, border, default_label = RISK_STYLE.get(risk_level, RISK_STYLE["none"])
    label = label_override or default_label
    return f"""
<span style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.22rem 0.7rem;
             border-radius:9999px; background:{_c(bg)}; border:1px solid {_c(border)};
             font-family:'Inter',sans-serif; font-size:0.78rem; font-weight:600; color:{_c(solid)};">
  <span style="width:8px;height:8px;border-radius:50%;background:{_c(dot)};"></span>{label}
</span>
"""


def score_pill(score: float, max_score: float = 10) -> str:
    if score == 0:
        level = "none"
    elif score <= max_score * 0.3:
        level = "low"
    elif score <= max_score * 0.6:
        level = "medium"
    else:
        level = "high"
    solid, bg, dot, border, _ = RISK_STYLE[level]
    return f"""
<span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.2rem 0.65rem;
             border-radius:9999px; background:{_c(bg)}; border:1px solid {_c(border)};
             font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700; color:{_c(solid)};">
  {score:g}/{max_score:g}
</span>
"""


def mono(text: str, truncate: bool = False) -> str:
    safe = _html.escape(str(text))
    if truncate:
        return f'<span class="pl-mono trunc" title="{safe}">{safe}</span>'
    return f'<span class="pl-mono">{safe}</span>'


def shap_bar(feature_label: str, value: float, contribution: float, max_abs: float) -> str:
    """One diverging SHAP bar row."""
    pct = 0 if max_abs == 0 else min(abs(contribution) / max_abs, 1.0) * 50  # half-width max
    is_positive = contribution >= 0
    color = _c("high_dot") if is_positive else _c("safe_dot")
    side_style = "left:50%; border-radius:0 4px 4px 0;" if is_positive else "right:50%; border-radius:4px 0 0 4px;"
    sign = "+" if is_positive else "-"
    return f"""
<div style="display:grid; grid-template-columns: 1fr 2fr; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
  <div style="font-size:0.82rem; color:{_c('on_surface')}; font-weight:500;">{feature_label} <span style="color:{_c('secondary')}; font-size:0.75rem;">({value:.2g})</span></div>
  <div style="position:relative; height:1.4rem; background:{_c('bg')}; border:1px solid {_c('border')}; border-radius:0.35rem;">
    <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:{_c('border_strong')};"></div>
    <div style="position:absolute; {side_style} top:0.15rem; bottom:0.15rem; width:{pct}%; background:{color};
                display:flex; align-items:center; justify-content:{'flex-end' if is_positive else 'flex-start'};
                padding:0 0.3rem; font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700; color:white;">
      {sign}{abs(contribution)*100:.1f}%
    </div>
  </div>
</div>
"""


def footer_bar() -> str:
    return f"""
<div class="pl-footer">
  PharmaLens AI &middot; Educational tool &mdash; not a substitute for laboratory validation.
</div>
"""