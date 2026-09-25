"""
The playground's visual system: one palette, two typefaces, a few components.

Streamlit draws the widgets; this module sets the type, the spacing and the
handful of custom blocks (step headers, the verdict card, model badges) so
that every page reads the same way. Colours come from Streamlit's own theme
variables where they exist, so the viewer's light or dark setting is
respected, and from the tokens below elsewhere.
"""

import streamlit as st

LIBRARY = "#2F7D5B"   # moss: what this library implements
BASELINE = "#4A6FA5"  # slate: what it is measured against
WARN = "#C58A1A"
BAD = "#B4433A"

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root { --nt-lib: %(lib)s; --nt-base: %(base)s; --nt-warn: %(warn)s; --nt-bad: %(bad)s; }
html, body, .stApp, [class*="st-"], .stMarkdown, .stCaption, p, li, label { font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; }
h1, h2, h3, .nt-display { font-family: "Bricolage Grotesque", "IBM Plex Sans", Georgia, serif !important; letter-spacing: -0.01em; text-wrap: balance; }
h1 { font-size: 2.6rem !important; font-weight: 700 !important; line-height: 1.05 !important; margin-bottom: 0.2em !important; }
h2 { font-size: 1.7rem !important; font-weight: 700 !important; margin-top: 1.6em !important; }
h3 { font-size: 1.2rem !important; font-weight: 600 !important; }
code, pre, .stCode, [data-testid="stDataFrame"] { font-family: "IBM Plex Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }
.stMarkdown p { max-width: 68ch; line-height: 1.55; }
.stCaption, .stCaption p { max-width: 68ch; }
[data-testid="stMetricValue"] { font-family: "IBM Plex Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }
/* lead paragraph under a title */
.nt-lead { font-size: 1.15rem; line-height: 1.5; max-width: 62ch; color: var(--text-color); opacity: 0.9; margin: 0 0 1.2rem 0; }
/* eyebrow label */
.nt-eyebrow { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600; color: var(--nt-lib); margin-bottom: 0.2rem; }
/* numbered step: the number is large because the order is real */
.nt-step { display: flex; gap: 1rem; align-items: baseline; margin: 2.2rem 0 0.6rem 0; }
.nt-step .n { font-family: "Bricolage Grotesque", serif; font-size: 2.4rem; font-weight: 700; line-height: 1; color: var(--nt-lib); min-width: 2.2rem; }
.nt-step h2 { margin: 0 !important; }
.nt-step p { margin: 0.15rem 0 0 0; opacity: 0.8; max-width: 62ch; }
/* verdict card */
.nt-verdict { border-left: 6px solid var(--nt-lib); background: var(--secondary-background-color); padding: 1rem 1.2rem; border-radius: 6px; margin: 0.6rem 0 1rem 0; }
.nt-verdict .k { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600; opacity: 0.7; }
.nt-verdict .v { font-family: "Bricolage Grotesque", serif; font-size: 1.7rem; font-weight: 700; line-height: 1.15; margin: 0.1rem 0 0.3rem 0; }
.nt-verdict p { margin: 0; max-width: 70ch; }
/* badges */
.nt-badge { display: inline-block; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 999px; color: white; }
.nt-badge.lib { background: var(--nt-lib); }
.nt-badge.base { background: var(--nt-base); }
/* quieter Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }
div[data-testid="stPlotlyChart"] { margin-bottom: 0.2rem; }
</style>
""" % dict(lib=LIBRARY, base=BASELINE, warn=WARN, bad=BAD)


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


def title(text, lead=None, eyebrow=None):
    if eyebrow:
        st.markdown(f'<div class="nt-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(text)
    if lead:
        st.markdown(f'<p class="nt-lead">{lead}</p>', unsafe_allow_html=True)


def step(n, heading, text=None):
    st.markdown(
        f'<div class="nt-step"><div class="n">{n}</div><div><h2>{heading}</h2>'
        + (f"<p>{text}</p>" if text else "") + "</div></div>",
        unsafe_allow_html=True,
    )


def verdict(label, value, text):
    st.markdown(
        f'<div class="nt-verdict"><div class="k">{label}</div><div class="v">{value}</div><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def badge(group):
    cls = "lib" if group == "neural-trees" else "base"
    label = "neural-trees" if group == "neural-trees" else "baseline"
    return f'<span class="nt-badge {cls}">{label}</span>'
