"""Visual layer: brand tokens, CSS, and small HTML components for Streamlit.

Kuku theme: pure black, white type, a chunky soft serif for headlines like the
Kuku wordmark, and the coral-red dot as the only accent colour.
"""
from __future__ import annotations

import html

import streamlit as st

# Design tokens - every colour in the app comes from here.
T = {
    "bg": "#000000",          # Kuku black
    "surface": "#0e0e0e",     # cards
    "surface2": "#161616",    # inputs, inner panels
    "surface3": "#1f1f1f",    # pressed, selected
    "line": "rgba(255,255,255,0.09)",
    "line2": "rgba(255,255,255,0.16)",
    "ink": "#ffffff",         # Kuku wordmark white
    "ink2": "#d4d4d4",
    "muted": "#9a9a9a",
    "red": "#F24A4A",         # Kuku dot - the only accent
    "brand": "#ffffff",       # charts: primary series
    "bar": "#3a3a3a",         # charts: secondary bars
    "fit": "#3ddc84",
    "tight": "#f5b942",
    "overflow": "#ff6b6b",
    "neutral": "#6e6e6e",
}
STATUS = {
    "fit": ("●", "Fits", T["fit"]),
    "tight": ("◐", "Tight", T["tight"]),
    "overflow": ("▲", "Overflow", T["overflow"]),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,600..900,0..100&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;600&display=swap');
.kserif, .hero h1, .sec .h, .tile .v, .mast .word, .empty b, .step .n {{ font-family:'Fraunces', Georgia, serif !important; font-variation-settings:'SOFT' 100, 'WONK' 0; }}

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, label {{
  font-family: 'Inter', 'Noto Sans Devanagari', ui-sans-serif, system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}}
.stApp {{ background: {T['bg']}; color: {T['ink']}; }}
header[data-testid="stHeader"] {{ background: rgba(0,0,0,.78); backdrop-filter: saturate(160%) blur(18px);
  border-bottom: 1px solid {T['line']}; }}
.block-container {{ padding-top: 4.4rem; max-width: 1160px; }}
.mono {{ font-family: 'JetBrains Mono', monospace !important; }}
.dot {{ color: {T['red']}; }}
::selection {{ background: rgba(242,74,74,.25); }}

/* top nav */
[data-testid="stTopNavLinkContainer"] a span, [data-testid="stTopNavLinkContainer"] a p {{ font-weight:500; color:{T['ink2']}; }}

/* masthead */
.mast {{ display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
  padding: 2px 0 18px; border-bottom: 1px solid {T['line']}; margin-bottom: 40px; }}
.mast .word {{ font-weight:900; font-size:clamp(48px, 7vw, 76px); letter-spacing:-.03em; color:{T['ink']}; line-height:.95; }}
.mast .word i {{ display:inline-block; width:.27em; height:.27em; border-radius:50%; background:{T['red']}; margin-left:.05em; }}
.mast .meta {{ display:inline-flex; align-items:center; gap:10px; height:34px; padding:0 14px; border-radius:999px;
  border:1px solid {T['line']}; font-size:13px; color:{T['ink2']}; }}
.mast .meta b {{ color:{T['ink']}; font-weight:600; }}

/* hero */
.hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:32px; flex-wrap:wrap; margin: 0 0 44px; }}
.hero > div {{ max-width: 680px; }}
.hero .eyebrow {{ font-size:12px; font-weight:600; color:{T['muted']}; letter-spacing:.14em; text-transform:uppercase; }}
.hero h1 {{ font-size: clamp(36px, 5vw, 60px); font-weight:800; letter-spacing:-.025em; line-height:1.02;
  margin:14px 0 0; color:{T['ink']}; }}
.hero p {{ font-size:17px; color:{T['ink2']}; line-height:1.6; max-width:460px; margin:0; }}

/* metric cards */
.tiles {{ display:flex; flex-wrap:wrap; gap:20px; margin: 4px 0 28px; }}
.tile {{ flex:1 1 220px; min-width:0; background:{T['surface']}; border:1px solid {T['line']}; border-radius:20px; padding:22px 24px; }}
.tile .k {{ color:{T['ink2']}; font-size:13.5px; font-weight:500; }}
.tile .v {{ font-size:clamp(28px, 2.9vw, 40px); white-space:nowrap; font-weight:800; margin-top:14px; color:{T['ink']}; letter-spacing:-.035em; line-height:1.05;
  font-variant-numeric: tabular-nums; }}
.tile .s {{ color:{T['ink2']}; font-size:13.5px; margin-top:8px; }}
.tile.hot .v:after {{ content:""; display:inline-block; width:10px; height:10px; border-radius:50%; background:{T['red']}; margin-left:6px; }}

/* section heading */
.sec {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap; margin: 44px 0 18px; }}
.sec .h {{ font-size: clamp(24px, 2.6vw, 30px); font-weight:800; color:{T['ink']}; letter-spacing:-.025em; line-height:1.15; }}
.sec span {{ color:{T['muted']}; font-size:13px; }}

/* cards */
.card {{ background:{T['surface']}; border:1px solid {T['line']}; border-radius:20px; padding:20px 22px; }}
.card p {{ color:{T['ink2']}; margin: 6px 0 0; font-size:14.5px; line-height:1.6; }}
.card h4, .card .h4 {{ margin:0; font-size:16px; font-weight:600; color:{T['ink']}; letter-spacing:-.01em; }}
.step {{ display:flex; gap:16px; align-items:flex-start; }}
.step .n {{ flex:0 0 32px; height:32px; border-radius:10px; display:grid; place-items:center; font-size:16px; font-weight:800;
  background:{T['surface3']}; color:{T['ink2']}; border:1px solid {T['line']}; }}
.tag {{ display:inline-flex; align-items:center; height:22px; font-size:12px; font-weight:500; padding:0 9px;
  border-radius:999px; margin-left:10px; vertical-align:middle; }}
.tag.ai {{ color:{T['red']}; background:rgba(242,74,74,.13); }}
.tag.code {{ color:{T['ink2']}; background:{T['surface3']}; border:1px solid {T['line']}; }}
.tag.human {{ color:{T['tight']}; background:rgba(245,185,66,.12); }}

/* line rows (studio) */
.lrow {{ background:{T['surface']}; border:1px solid {T['line']}; border-radius:20px; padding:18px 20px; margin-bottom:12px; }}
.lrow.review {{ border-color: rgba(255,107,107,.35); }}
.lhead {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; color:{T['muted']}; font-size:13px; margin-bottom:12px; }}
.lhead .who {{ color:{T['ink']}; font-weight:600; font-size:14px; }}
.lhead .tc {{ font: 500 12.5px 'JetBrains Mono', monospace; color:{T['ink']}; background:{T['surface3']};
  border:1px solid {T['line']}; padding:3px 8px; border-radius:7px; }}
.lgrid {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; }}
@media (max-width: 900px) {{ .lgrid {{ grid-template-columns: 1fr; }} }}
.lcol {{ background:{T['surface2']}; border:1px solid {T['line']}; border-radius:14px; padding:12px 14px; }}
.lcol .lab {{ font-size:12.5px; font-weight:500; color:{T['muted']}; display:flex; justify-content:space-between; align-items:center; gap:8px; }}
.lcol .txt {{ color:{T['ink']}; font-size:15px; margin-top:8px; line-height:1.55; }}
.lcol .meta {{ color:{T['muted']}; font-size:12.5px; margin-top:8px; font-family:'JetBrains Mono', monospace; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; height:24px; font-size:12px; font-weight:500; padding:0 10px; border-radius:999px; }}
.bar {{ height:6px; border-radius:999px; background:rgba(255,255,255,.08); margin-top:10px; position:relative; }}
.bar i {{ position:absolute; left:0; top:0; bottom:0; border-radius:inherit; }}
.bar b {{ position:absolute; top:-4px; bottom:-4px; width:2px; border-radius:2px; background:{T['ink']}; }}

/* streamlit widgets */
.stButton>button, .stDownloadButton>button {{ border-radius:999px; font-weight:500; height:42px;
  background:{T['surface2']}; border:1px solid {T['line2']}; color:{T['ink']}; transition: all .15s; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:{T['surface3']}; border-color:rgba(255,255,255,.24); color:{T['ink']}; }}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {{ background:{T['ink']}; color:#000; border:1px solid {T['ink']}; }}
.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover {{ background:{T['red']}; border-color:{T['red']}; color:#fff; }}
div[data-baseweb="select"]>div, .stTextInput input, .stTextArea textarea {{ border-radius:12px !important;
  background:{T['surface2']} !important; border-color:{T['line2']} !important; }}
[data-testid="stExpander"] {{ border-radius:14px; border:1px solid {T['line']}; background:{T['surface']}; }}
[data-testid="stFileUploaderDropzone"] {{ background:{T['surface']}; border:1px dashed {T['line2']}; border-radius:14px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:20px; }}
.empty {{ display:flex; flex-direction:column; align-items:center; gap:8px; text-align:center; padding: 36px 24px;
  border:1px dashed {T['line2']}; border-radius:20px; color:{T['ink2']}; font-size:14.5px; }}
.empty b {{ color:{T['ink']}; font-size:20px; font-weight:800; }}
.footer {{ color:{T['ink2']}; font-size:13px; text-align:center; margin: 56px 0 20px; padding-top:28px; border-top:1px solid {T['line']}; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def masthead():
    st.markdown(
        '<div class="mast"><div class="word">dubfit<i></i></div>'
        '<div class="meta"><b>Timing-aware dubbing</b> · built for the Kuku assignment</div></div>',
        unsafe_allow_html=True)


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def hero(eyebrow: str, title_html: str, sub: str):
    st.markdown(f'<div class="hero"><div><div class="eyebrow">{e(eyebrow)}</div><h1>{title_html}</h1></div>'
                f'<p>{e(sub)}</p></div>', unsafe_allow_html=True)


def tiles(items: list[tuple]):
    """items: (label, value, sub, hot?)"""
    cells = "".join(
        f'<div class="tile{" hot" if (len(it) > 3 and it[3]) else ""}"><div class="k">{e(it[0])}</div>'
        f'<div class="v">{e(it[1])}</div><div class="s">{e(it[2])}</div></div>' for it in items)
    st.markdown(f'<div class="tiles">{cells}</div>', unsafe_allow_html=True)


def section(title: str, note: str = ""):
    st.markdown(f'<div class="sec"><div class="h">{e(title)}</div><span>{e(note)}</span></div>', unsafe_allow_html=True)


def chip(status: str, ratio: float | None = None) -> str:
    icon, label, color = STATUS.get(status, ("○", status or "—", T["neutral"]))
    r = f" · {ratio:.0%}" if ratio is not None else ""
    return f'<span class="chip" style="color:{color};background:{color}1f">{icon} {label}{r}</span>'


def fitbar(ratio: float, status: str) -> str:
    color = STATUS.get(status, ("", "", T["neutral"]))[2]
    w = min(ratio, 1.5) / 1.5 * 100
    mark = 1 / 1.5 * 100
    return f'<div class="bar"><i style="width:{w:.1f}%;background:{color}"></i><b style="left:{mark:.1f}%"></b></div>'


def tc(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def empty(title: str, body: str):
    st.markdown(f'<div class="empty"><b>{e(title)}</b>{e(body)}</div>', unsafe_allow_html=True)


def plotly_layout(fig, height=320, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Noto Sans Devanagari, sans-serif", color=T["ink2"], size=12),
        hoverlabel=dict(bgcolor="#161616", bordercolor="rgba(255,255,255,0.16)", font=dict(color=T["ink"], size=13)),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.08)")
    return fig