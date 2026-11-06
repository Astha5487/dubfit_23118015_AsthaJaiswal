"""DubFit - dubbed dialogue that fits the frame.  Run:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dubfit import ui
from dubfit.analysis import aggregate, load_results, source_pressure
from dubfit.checks import judge_agreement, load_ruler
from dubfit.budget import LANGUAGES
from dubfit.llm import LLM, PRESETS
from dubfit.parser import load_show
from dubfit.pipeline import run_episode, status_for, summarize
from dubfit.tts import edge_available, espeak_available, measure
from dubfit.ui import STATUS, T, e

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
TARGETS = {k: v for k, v in LANGUAGES.items() if k != "en"}

st.set_page_config(page_title="dubfit · timing-aware dubbing", page_icon="🔴", layout="wide", initial_sidebar_state="collapsed")
ui.inject()
ui.masthead()


# ---------------------------------------------------------------- state & data
@st.cache_resource(show_spinner=False)
def _parse(key: str, payload):
    return load_show(payload)


@st.cache_data(show_spinner=False)
def _pressure(key: str, engine: str) -> dict:
    return source_pressure(st.session_state.show, engine=engine)


def init_state():
    ss = st.session_state
    ss.setdefault("show", None)
    ss.setdefault("show_key", None)
    ss.setdefault("show_name", "")
    ss.setdefault("results_dir", OUT)
    ss.setdefault("results", load_results(OUT))
    if ss.show is None:
        local = sorted(DATA.glob("*.zip")) + ([DATA] if DATA.exists() and any(DATA.rglob("*.txt")) else [])
        if local:
            src = local[0]
            ss.show = _parse(f"local:{src}", str(src))
            ss.show_key = hashlib.sha1(str(src).encode()).hexdigest()[:10]
            ss.show_name = _zip_title(src.read_bytes(), src.stem) if src.is_file() else "Local scripts"


def _zip_title(data: bytes, fallback: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            dirs = [n.split("/")[0] for n in z.namelist() if "/" in n and not n.startswith("__MACOSX")]
            return max(set(dirs), key=dirs.count) if dirs else fallback
    except zipfile.BadZipFile:
        return fallback


def set_show_from_upload(file):
    data = file.getvalue()
    key = hashlib.sha1(data).hexdigest()[:10]
    eps = _parse(key, data)
    if not eps:
        st.error("No timed dialogue found in this zip. DubFit expects Kuku-format .txt screenplays with "
                 "'**SPEAKER (Emotion):**' blocks followed by 'Start: (mm:ss.s) | End: (mm:ss.s)'.")
        return
    title = _zip_title(data, Path(file.name).stem)
    if title != st.session_state.get("show_name"):
        # A different show: keep its results in their own folder so episode numbers never collide.
        slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")[:60] or key
        st.session_state.results_dir = OUT / "shows" / slug
        st.session_state.results = load_results(st.session_state.results_dir)
    st.session_state.show = eps
    st.session_state.show_key = key
    st.session_state.show_name = _zip_title(data, Path(file.name).stem)


def save_result(res: dict):
    st.session_state.results.setdefault(res["lang"], {})[res["episode"]] = res
    try:
        folder = Path(st.session_state.get("results_dir", OUT)) / res["lang"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"episode-{res['episode']:03d}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    except OSError:
        pass


def outputs_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for lang, eps in st.session_state.results.items():
            for n, r in sorted(eps.items()):
                slim = {**r, "lines": [{k: v for k, v in ln.items() if not k.endswith("audio")} for ln in r["lines"]]}
                for ln in slim["lines"]:
                    for kk in ("baseline", "dubfit"):
                        if isinstance(ln.get(kk), dict):
                            ln[kk] = {k: v for k, v in ln[kk].items() if k != "audio"}
                z.writestr(f"outputs/{lang}/episode-{n:03d}.json", json.dumps(slim, indent=2, ensure_ascii=False))
        if st.session_state.show:
            sp = _pressure(st.session_state.show_key, "espeak")
            z.writestr("outputs/source_pressure.json", json.dumps(sp, indent=2, ensure_ascii=False))
        rc = load_ruler(st.session_state.get("results_dir", OUT))
        if rc:
            z.writestr("outputs/ruler_check.json", json.dumps(rc, indent=2, ensure_ascii=False))
        for lang, eps in st.session_state.results.items():
            z.writestr(f"outputs/judge_check_{lang}.json", json.dumps(judge_agreement(eps), indent=2, ensure_ascii=False))
    return buf.getvalue()


def need_show() -> bool:
    if st.session_state.show:
        return True
    ui.empty("No show loaded yet", "Upload a zip of episode screenplays on the Run page to begin.")
    return False


init_state()
default_engine = "espeak" if espeak_available() else "edge"


def engine_for(lang: str) -> str:
    """Use the neural voice for any language where the ruler check found eSpeak unreliable."""
    rc = load_ruler(OUT)
    if rc and edge_available():
        for group, s in rc["summary"].items():
            if group.startswith(LANGUAGES[lang]["name"]) and (s.get("status_agreement") or 1) < 0.7:
                return "edge"
    return default_engine


# ---------------------------------------------------------------- charts
def chart_growth(sp: dict):
    xs = list(sp["break_at_growth"].keys())
    ys = [v * 100 for v in sp["break_at_growth"].values()]
    fig = go.Figure(go.Bar(
        x=[f"+{x}" if x != "0%" else "Same length" for x in xs], y=ys,
        marker_color=[T["red"] if x == "30%" else T["bar"] for x in xs],
        marker_line_width=0, text=[f"{y:.0f}%" for y in ys], textposition="outside",
        textfont=dict(color=T["ink"], size=13), hovertemplate="%{x}: %{y:.1f}% of lines overflow<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.update_yaxes(title="Lines that no longer fit (%)", range=[0, max(ys) * 1.25 + 1], ticksuffix="%")
    fig.update_xaxes(title="Dub spoken longer than the English by…", type="category")
    return ui.plotly_layout(fig, 300, legend=False)


def chart_episodes(sp: dict):
    pe = sp["per_episode"]
    fig = go.Figure(go.Bar(
        x=[p["episode"] for p in pe], y=[p["no_room_share"] * 100 for p in pe],
        marker_color=[T["ink"] if p["no_room_share"] >= 0.5 else T["bar"] for p in pe],
        marker_line_width=0, customdata=[[p["lines"], p["median_wps"]] for p in pe],
        hovertemplate="Episode %{x}<br>%{y:.0f}% of lines have <20% headroom<br>%{customdata[0]} lines · "
                      "%{customdata[1]} words/sec<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_yaxes(title="Lines with < 20% headroom", ticksuffix="%")
    fig.update_xaxes(title="Episode", dtick=5)
    return ui.plotly_layout(fig, 260, legend=False)


def chart_timeline(rows: list[dict], has_dub: bool):
    tracks = ["English source"] + (["Plain translation", "DubFit"] if has_dub else [])
    fig = go.Figure()
    # window rails
    fig.add_trace(go.Bar(
        y=[t for t in tracks for _ in rows], x=[r["limit"] for _ in tracks for r in rows],
        base=[r["start"] for _ in tracks for r in rows], orientation="h", width=0.62,
        marker=dict(color="rgba(255,255,255,0.03)", line=dict(color="rgba(255,255,255,0.24)", width=1)),
        hoverinfo="skip", name="Time window", showlegend=True))
    spoken = {"English source": None, "Plain translation": "baseline", "DubFit": "dubfit"}
    for status in ("fit", "tight", "overflow"):
        ys, xs, bs, cd = [], [], [], []
        for t in tracks:
            for r in rows:
                if spoken[t] is None:
                    sec, txt = r["en_seconds"], r["en"]
                    ratio, stt = status_for(sec, r["limit"])
                else:
                    d = r[spoken[t]]
                    sec, txt, ratio, stt = d["seconds"], d["text"], d["ratio"], d["status"]
                if stt != status:
                    continue
                ys.append(t); xs.append(sec); bs.append(r["start"])
                cd.append([r["speaker"].title(), ui.tc(r["start"]), f"{r['limit']:.2f}", f"{sec:.2f}", f"{ratio:.0%}", txt])
        if not ys:
            continue
        icon, label, color = STATUS[status]
        fig.add_trace(go.Bar(
            y=ys, x=xs, base=bs, orientation="h", width=0.36, name=label,
            marker=dict(color=color, line=dict(width=0), cornerradius=3), customdata=cd,
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Window %{customdata[2]}s · spoken "
                          "%{customdata[3]}s (%{customdata[4]})<br>%{customdata[5]}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(tracks)), gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Episode time (seconds)", rangeslider=dict(visible=True, thickness=0.06, bgcolor=T["surface"]))
    return ui.plotly_layout(fig, 170 + 70 * len(tracks))


def chart_before_after(agg: dict):
    fig = go.Figure()
    for status in ("fit", "tight", "overflow"):
        icon, label, color = STATUS[status]
        vals = [agg["baseline"][status] * 100, agg["dubfit"][status] * 100]
        fig.add_trace(go.Bar(
            y=["Plain translation", "DubFit"], x=vals, orientation="h", name=label,
            marker=dict(color=color, line=dict(color=T["bg"], width=2)),
            text=[f"{v:.0f}%" if v >= 6 else "" for v in vals], textposition="inside",
            textfont=dict(color="#000000", size=13),
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>"))
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=["DubFit", "Plain translation"])
    return ui.plotly_layout(fig, 210)


def chart_overflow_by_episode(results: dict):
    eps = sorted(results)
    fig = go.Figure()
    for key, name, color, dash in (("baseline", "Plain translation", T["neutral"], "dot"), ("dubfit", "DubFit", T["ink"], "solid")):
        ys = [results[n]["summary"][f"{key}_overflow"] * 100 for n in eps]
        fig.add_trace(go.Scatter(x=eps, y=ys, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
                                 hovertemplate=f"Episode %{{x}} · {name}: %{{y:.0f}}% overflow<extra></extra>"))
    fig.update_yaxes(title="Lines overflowing", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(title="Episode")
    return ui.plotly_layout(fig, 280)


def chart_ruler(rc: dict):
    fig = go.Figure()
    colors = [T["ink"], T["red"], T["neutral"]]
    mx = 0.0
    for i, grp in enumerate(rc["summary"]):
        g = [r for r in rc["lines"] if r["group"] == grp]
        mx = max([mx] + [r["espeak"] for r in g] + [r["neural"] for r in g])
        fig.add_trace(go.Scatter(
            x=[r["espeak"] for r in g], y=[r["neural"] for r in g], mode="markers", name=grp,
            marker=dict(size=8, color=colors[i % 3], opacity=.75, line=dict(color=T["bg"], width=1)),
            customdata=[[r["text"]] for r in g],
            hovertemplate="eSpeak %{x:.2f}s · neural %{y:.2f}s<br>%{customdata[0]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Same length",
                             line=dict(color=T["muted"], width=1, dash="dot"), hoverinfo="skip"))
    fig.update_xaxes(title="eSpeak (seconds)", rangemode="tozero")
    fig.update_yaxes(title="Neural voice (seconds)", rangemode="tozero")
    return ui.plotly_layout(fig, 340)


def chart_meaning(agg: dict):
    ks = [1, 2, 3, 4, 5]
    vs = [agg["meaning_dist"].get(k, 0) for k in ks]
    fig = go.Figure(go.Bar(x=[str(k) for k in ks], y=vs, marker_color=T["brand"], text=vs, textposition="outside",
                           textfont=dict(color=T["ink"]), hovertemplate="Score %{x}: %{y} lines<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.update_xaxes(title="Meaning preserved (1 wrong → 5 same)")
    fig.update_yaxes(title="Lines", range=[0, max(vs + [1]) * 1.2])
    return ui.plotly_layout(fig, 260, legend=False)


# ---------------------------------------------------------------- pages
def page_overview():
    ui.hero("Problem 3 · Adapting microdramas", 'Dubs that fit the frame<span class="dot">.</span>',
            "Every line in a Kuku screenplay is locked to a time window. Translations run longer, so AI dubbing "
            "rushes, clips or drifts off the lips. DubFit translates each line to fit its window and proves it "
            "with real speech timing before anything reaches the voice model.")
    if not need_show():
        return
    sp = _pressure(st.session_state.show_key, default_engine)
    ui.tiles([
        ("Episodes", f"{sp['episodes']}", (st.session_state.show_name or "Loaded")[:60]),
        ("Dialogue lines", f"{sp['lines']:,}", f"median window {sp['median_window']}s"),
        ("Already fast", f"{sp['share_wps_3plus']:.0%}", "of English lines run ≥ 3 words/sec"),
        ("Break at +30% length", f"{sp['break_at_growth']['30%']:.0%}", "of lines no longer fit", True),
    ])
    c1, c2 = st.columns([1.05, 1])
    with c1:
        ui.section("How fast the show breaks", "measured with TTS on every English line")
        st.plotly_chart(chart_growth(sp), width="stretch", config={"displayModeBar": False})
    with c2:
        ui.section("Where the pressure is", "episodes with the least room")
        st.plotly_chart(chart_episodes(sp), width="stretch", config={"displayModeBar": False})

    res = st.session_state.results
    if res:
        ui.section("Results so far", "real runs, from the Quality page")
        cols = st.columns(len(res))
        for col, (lang, eps) in zip(cols, res.items()):
            agg = aggregate(eps)
            with col:
                ui.tiles([(f"{LANGUAGES[lang]['name']}", f"{agg['baseline']['overflow']:.0%} → {agg['dubfit']['overflow']:.0%}",
                           f"overflowing lines · {agg['episodes']} episodes · {agg['lines']} lines", True)])
    else:
        st.info("No translation runs yet. Open **Run** to dub episodes with any LLM, then come back here.", icon="🎬")
    with st.expander("How the ruler is calibrated"):
        st.markdown(
            f"- Each line is spoken with **{sp['engine']}** TTS at a fixed rate, and silence is trimmed.\n"
            f"- At that rate **{sp['en_fit_share']:.0%}** of the original English lines fit their own windows, so the ruler "
            "accepts the source script. The leftover {:.0%} are lines that were already tight in English.\n"
            "- **Headroom** is the share of a window a line leaves unused. A line with under 20% headroom breaks as soon as a "
            "translation is 20% longer.\n- Voice-over and off-screen lines (V.O./O.S.) may run into the silence after them, "
            "because nobody's lips have to match.".format(1 - sp["en_fit_share"]))


def page_studio():
    if not need_show():
        return
    eps = {e.number: e for e in st.session_state.show}
    res = st.session_state.results
    langs = list(TARGETS)
    c1, c2, c3 = st.columns([1, 0.9, 1.9])
    first = next((i for i, l in enumerate(langs) if res.get(l)), 0)
    lang = c1.selectbox("Language", langs, index=first,
                        format_func=lambda k: LANGUAGES[k]["name"] + (f"  ✓ {len(res[k])} dubbed" if res.get(k) else ""))
    have = set(res.get(lang, {}))
    ep_n = c2.selectbox("Episode", sorted(eps), format_func=lambda n: f"Episode {n}" + ("  ✓ dubbed" if n in have else ""))
    view = c3.segmented_control("Show", ["All lines", "Needs review", "Fixed by DubFit", "To label"], default="All lines")
    ep = eps[ep_n]
    result = res.get(lang, {}).get(ep_n)

    st.markdown(f'<div class="card" style="margin:6px 0 4px"><div class="h4">Episode {ep_n}</div><p>{e(ep.logline)}</p></div>',
                unsafe_allow_html=True)

    if result:
        rows = result["lines"]
        s = summarize(rows)
        ui.tiles([
            ("Plain translation", f"{s['baseline_fit'] + s['baseline_tight']:.0%}", f"fit or tight · {s['baseline_overflow']:.0%} overflow"),
            ("DubFit", f"{s['dubfit_fit'] + s['dubfit_tight']:.0%}", f"fit or tight · {s['dubfit_overflow']:.0%} overflow", True),
            ("Lines DubFit changed", s.get("changed", s["rewritten"]), "only lines that overflowed are touched"),
            ("For a human", s["needs_review"], "still long or meaning drift"),
        ])
    else:
        rows = []
        for ln in ep.lines:
            g = "F" if any(k in ln.speaker.upper() for k in ("LADY", "JULIANA", "CATHERINE", "CLARISSA", "MOTHER")) else "M"
            m = measure(ln.text, "en", default_engine, g)
            rows.append({"id": ln.id, "speaker": ln.speaker, "emotion": ln.emotion, "start": ln.start, "end": ln.end,
                         "limit": ln.window if ln.on_screen else ln.soft_window, "on_screen": ln.on_screen,
                         "en": ln.text, "en_seconds": m["seconds"], "en_audio": m["audio"], "gender": g})
        dub_now_panel(ep, lang)

    ui.section("Timeline", "each bar is one spoken line inside its window · drag the slider to zoom")
    st.plotly_chart(chart_timeline(rows, bool(result)), width="stretch", config={"displayModeBar": False})

    if result:
        b1, b2, b3 = st.columns(3)
        df = pd.DataFrame([{"id": r["id"], "start": r["start"], "end": r["end"], "speaker": r["speaker"],
                            "emotion": r["emotion"], "english": r["en"], "dub": r["dubfit"]["text"],
                            "window_s": r["limit"], "spoken_s": r["dubfit"]["seconds"], "status": r["dubfit"]["status"],
                            "approved": r.get("approved", False), "needs_review": r.get("needs_review", False)} for r in rows])
        b1.download_button("Download dub script (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           f"dubfit_{lang}_ep{ep_n:03d}.csv", "text/csv", type="primary", width="stretch")
        srt = "\n".join(f"{i + 1}\n{_srt(r['start'])} --> {_srt(r['end'])}\n{r['dubfit']['text']}\n" for i, r in enumerate(rows))
        b2.download_button("Download subtitles (SRT)", srt.encode("utf-8"), f"dubfit_{lang}_ep{ep_n:03d}.srt",
                           width="stretch")
        b3.download_button("Download raw result (JSON)", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                           f"dubfit_{lang}_ep{ep_n:03d}.json", width="stretch")

    ui.section("Lines", "open a line to listen, edit and re-measure")
    shown = rows
    if result and view == "Needs review":
        shown = [r for r in rows if r.get("needs_review")]
    elif result and view == "Fixed by DubFit":
        shown = [r for r in rows if r["baseline"]["status"] == "overflow" and r["dubfit"]["status"] != "overflow"]
    elif result and view == "To label":
        changed = [r for r in rows if r["dubfit"].get("changed")]
        shown = [r for r in changed if not r.get("human_label")]
        st.caption(f"Labelled {len(changed) - len(shown)} of {len(changed)} lines DubFit changed in this episode. "
                   "Your labels are used to check the AI judge on the Quality page.")
    if not shown:
        ui.empty("Nothing here", "No lines match this filter.")
    for r in shown:
        line_card(r, lang, result)


KEY_ENV = {"groq/": "GROQ_API_KEY", "gemini/": "GEMINI_API_KEY", "openai/": "OPENAI_API_KEY", "anthropic/": "ANTHROPIC_API_KEY"}


def _model_and_key() -> tuple[str, str, str]:
    ss = st.session_state
    model = ss.get("run_model") or os.environ.get("DUBFIT_MODEL") or PRESETS[list(PRESETS)[0]][0]
    env = next((v for k, v in KEY_ENV.items() if model.startswith(k)), "")
    key = ss.get("run_key") or ""
    if not key and env:
        try:
            key = st.secrets.get(env, "") if _has_secrets() else ""
        except Exception:
            key = ""
        key = key or os.environ.get(env, "")
    return model, env, key


def dub_now_panel(ep, lang: str):
    model, env, key = _model_and_key()
    name = LANGUAGES[lang]["name"]
    with st.container(border=True):
        st.markdown(f"**Episode {ep.number} isn't dubbed in {name} yet.** The English track is shown below. "
                    f"Dub it now with `{model}`. It takes about 1–2 minutes on a free tier, and the same pipeline "
                    "and checks are used as on the Run page.")
        c1, c2 = st.columns([2, 1])
        if env and not key:
            key = c1.text_input(f"{env} (change the model on the Run page)", type="password", key=f"key-{lang}-{ep.number}")
        else:
            c1.caption(f"Model and key come from the Run page{' or your environment' if key else ''}.")
        if c2.button(f"Dub Episode {ep.number} in {name}", type="primary", width="stretch",
                      disabled=bool(env) and not key, key=f"dub-{lang}-{ep.number}"):
            st.session_state.run_model, st.session_state.run_key = model, key
            llm = LLM(model, api_key=key or None, min_interval=3.0, verbose=False)
            with st.status(f"Dubbing Episode {ep.number} in {name}…", expanded=True) as box:
                try:
                    res = run_episode(ep, lang, llm, engine=engine_for(lang), back_check=True,
                                      progress=lambda msg, frac: box.write(msg))
                    save_result(res)
                    box.update(label="Done", state="complete")
                except Exception as ex:
                    box.update(label="Failed", state="error")
                    st.error(f"{type(ex).__name__}: {str(ex)[:400]}")
                    return
            st.rerun()


def _srt(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def line_card(r: dict, lang: str, result: dict | None):
    en_ratio, en_status = status_for(r["en_seconds"], r["limit"])
    head = (f'<div class="lhead"><span class="tc">{ui.tc(r["start"])} → {ui.tc(r["start"] + r["limit"])}</span>'
            f'<span class="who">{e(r["speaker"].title())}</span><span>{e(r["emotion"])}</span>'
            f'<span>{r["limit"]:.2f}s window{" · voice-over" if not r["on_screen"] else ""}</span>'
            + (f'<span>budget {r["budget"]} syllables</span>' if "budget" in r else "")
            + ("".join(f'<span class="chip" style="color:{T["overflow"]};background:{T["overflow"]}1f;border:1px solid '
                       f'{T["overflow"]}55">⚑ {e(x)}</span>' for x in r.get("review_reasons", [])))
            + ('<span class="chip" style="color:#3ddc84;background:#3ddc841f">✓ approved</span>'
               if r.get("approved") else "") + "</div>")
    cols = [_col("English", r["en"], r["en_seconds"], en_ratio, en_status)]
    if result:
        b, d = r["baseline"], r["dubfit"]
        cols.append(_col("Plain translation", b["text"], b["seconds"], b["ratio"], b["status"], b.get("syllables")))
        extra = ""
        if r.get("back_translation"):
            extra = f'<div class="meta">↩ {e(r["back_translation"])}' + (
                f' · meaning {r["meaning_score"]}/5' if r.get("meaning_score") else "") + "</div>"
        cols.append(_col("DubFit" + (f" · try {d.get('attempts', 1)}" if d.get("attempts", 1) > 1 else ""), d["text"],
                         d["seconds"], d["ratio"], d["status"], d.get("syllables"), extra))
    klass = "lrow review" if r.get("needs_review") and not r.get("approved") else "lrow"
    grid_style = "" if result else ' style="grid-template-columns:1fr"'
    st.markdown(f'<div class="{klass}">{head}<div class="lgrid"{grid_style}>{"".join(cols)}</div></div>', unsafe_allow_html=True)

    if result and r["dubfit"].get("changed"):
        lab = r.get("human_label")
        g1, g2, g3 = st.columns([1, 1, 4])
        if g1.button("✓ Good", key=f"hg-{r['id']}", type="primary" if lab == "good" else "secondary", width="stretch"):
            r["human_label"] = None if lab == "good" else "good"
            save_result(result); st.rerun()
        if g2.button("✗ Bad", key=f"hb-{r['id']}", type="primary" if lab == "bad" else "secondary", width="stretch"):
            r["human_label"] = None if lab == "bad" else "bad"
            save_result(result); st.rerun()
        g3.caption(f"Your verdict: **{lab}**" if lab else
                   "Is the DubFit line correct and natural? Your verdict checks the AI judge.")

    with st.expander(f"Listen & edit · {r['id']}", expanded=False):
        a1, a2, a3 = st.columns(3)
        if r.get("en_audio") and Path(r["en_audio"]).exists():
            a1.caption("English"); a1.audio(r["en_audio"])
        if result:
            for col, key, lab in ((a2, "baseline", "Plain translation"), (a3, "dubfit", "DubFit")):
                p = r[key].get("audio")
                if p and Path(p).exists():
                    col.caption(lab); col.audio(p)
                elif r[key]["text"]:
                    if col.button(f"Generate {lab} audio", key=f"gen-{key}-{r['id']}"):
                        m = measure(r[key]["text"], lang, default_engine, r.get("gender", "F"))
                        r[key]["audio"] = m["audio"]; save_result(result); st.rerun()
            new = st.text_area("Edit the dub line", r["dubfit"]["text"], key=f"edit-{r['id']}", height=70)
            c1, c2, c3 = st.columns([1, 1, 2])
            if c1.button("Re-measure", key=f"rm-{r['id']}", type="primary"):
                m = measure(new, lang, result.get("engine", default_engine), r.get("gender", "F"))
                ratio, stt = status_for(m["seconds"], r["limit"])
                hist = r["dubfit"].get("history", []) + [{k: r["dubfit"][k] for k in ("text", "seconds", "ratio", "status")}]
                r["dubfit"] = {**r["dubfit"], "text": new, "seconds": m["seconds"], "ratio": ratio, "status": stt,
                               "audio": m["audio"], "history": hist, "edited_by_human": True}
                r["review_reasons"] = ["still too long"] if stt == "overflow" else []
                r["needs_review"] = bool(r["review_reasons"])
                save_result(result); st.rerun()
            if c2.button("Approve" if not r.get("approved") else "Unapprove", key=f"ap-{r['id']}"):
                r["approved"] = not r.get("approved", False)
                save_result(result); st.rerun()
            if r["dubfit"].get("history"):
                c3.caption("Earlier versions: " + " → ".join(f"{h['ratio']:.0%}" for h in r["dubfit"]["history"]))


def _col(label, text, seconds, ratio, status, syl=None, extra=""):
    meta = f"{seconds:.2f}s spoken" + (f" · {syl} syllables" if syl is not None else "")
    return (f'<div class="lcol"><div class="lab"><span>{e(label)}</span>{ui.chip(status, ratio)}</div>'
            f'<div class="txt">{e(text) or "<i>—</i>"}</div>{ui.fitbar(ratio, status)}<div class="meta">{meta}</div>{extra}</div>')


def page_run():
    ui.section("1 · Load a show", "a zip of episode .txt screenplays, Kuku format")
    up = st.file_uploader("Screenplay zip", type=["zip"], label_visibility="collapsed")
    if up is not None and st.session_state.get("_uploaded") != up.file_id:
        set_show_from_upload(up)
        st.session_state._uploaded = up.file_id
    if st.session_state.show:
        n_lines = sum(len(x.lines) for x in st.session_state.show)
        st.success(f"**{st.session_state.show_name}** · {len(st.session_state.show)} episodes · {n_lines:,} dialogue lines", icon="📚")
    with st.expander("Or load results from an earlier run (.zip of outputs/)"):
        rz = st.file_uploader("Results zip", type=["zip"], key="rz", label_visibility="collapsed")
        if rz is not None and st.session_state.get("_rz") != rz.file_id:
            with zipfile.ZipFile(rz) as z:
                k = 0
                for nm in z.namelist():
                    if nm.endswith(".json") and "/episode-" in nm:
                        r = json.loads(z.read(nm))
                        st.session_state.results.setdefault(r["lang"], {})[r["episode"]] = r
                        k += 1
            st.session_state._rz = rz.file_id
            st.success(f"Loaded {k} episode results.")
    if not st.session_state.show:
        return

    ui.section("2 · Choose the model", "any LiteLLM model works · swap it in one field")
    c1, c2 = st.columns([1, 1])
    preset = c1.selectbox("Model", list(PRESETS))
    model_default, env_name = PRESETS[preset]
    model = c2.text_input("LiteLLM model string", model_default, placeholder="provider/model-name")
    server_key = (st.secrets.get(env_name, "") if env_name and _has_secrets() else "") or os.environ.get(env_name or "", "")
    typed = st.text_input(f"API key{f' ({env_name})' if env_name else ''}", "", type="password",
                          placeholder="Using the app's built-in key" if server_key else "Paste a key (free at console.groq.com)",
                          help="Optional when the app has a built-in key. A key you paste is used only for this session and never saved.")
    api_key = typed or server_key  # the built-in key stays on the server and is never sent to the browser
    st.session_state.run_model, st.session_state.run_key = model, api_key

    ui.section("3 · What to dub", "")
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    langs = c1.multiselect("Languages", list(TARGETS), default=["hi"], format_func=lambda k: TARGETS[k]["name"])
    nums = [x.number for x in st.session_state.show]
    rng = c2.select_slider("Episodes", options=nums, value=(nums[0], nums[min(2, len(nums) - 1)]))
    engines = ["auto"] + [x for x, ok in (("espeak", espeak_available()), ("edge", edge_available())) if ok]
    engine = c3.selectbox("Timing voice", engines, format_func=lambda x: {"auto": "Auto (neural where eSpeak failed the check)",
                                                                          "espeak": "eSpeak (offline ruler)",
                                                                          "edge": "Edge neural (online)"}[x])
    o1, o2 = st.columns(2)
    back = o1.toggle("Back-translation meaning check", value=True)
    interval = o2.slider("Pause between LLM calls (s)", 0.0, 10.0, 4.0, 0.5, help="Keeps free tiers under their rate limits.")
    chosen = [x for x in st.session_state.show if rng[0] <= x.number <= rng[1]]
    st.caption(f"{len(chosen)} episodes × {len(langs)} languages ≈ up to {len(chosen) * len(langs) * (5 if back else 3)} LLM calls "
               "(fewer when lines already fit; cached calls are free).")

    if st.button("Run DubFit", type="primary", disabled=not (langs and model)):
        bar = st.progress(0.0)
        log = st.empty()
        total = len(chosen) * len(langs)
        done = 0
        for lang in langs:
            llm = LLM(model, api_key=api_key or None, min_interval=interval)
            for ep in chosen:
                def prog(msg, frac, _ep=ep, _lang=lang, _done=done):
                    bar.progress(min(1.0, (_done + frac) / total))
                    log.markdown(f"**{TARGETS[_lang]['name']} · Episode {_ep.number}** · {msg}")
                try:
                    res = run_episode(ep, lang, llm, engine=engine_for(lang) if engine == "auto" else engine,
                                      back_check=back, progress=prog)
                    save_result(res)
                except Exception as ex:
                    st.error(f"Episode {ep.number} ({lang}) failed: {type(ex).__name__}: {str(ex)[:300]}")
                    if "auth" in type(ex).__name__.lower():
                        break
                done += 1
                bar.progress(done / total)
            st.caption(f"{TARGETS[lang]['name']}: {llm.usage.calls} calls · {llm.usage.cached} cached · "
                       f"{llm.usage.prompt_tokens + llm.usage.completion_tokens:,} tokens")
        log.markdown("**Done.** Open **Studio** or **Quality**.")

    if st.session_state.results:
        ui.section("4 · Raw outputs", "every run, unedited, as JSON")
        st.download_button("Download outputs.zip", outputs_zip(), "dubfit_outputs.zip", "application/zip", type="primary")


def _has_secrets() -> bool:
    try:
        return bool(st.secrets._secrets is not None or len(st.secrets))
    except Exception:
        return False


def page_quality():
    res = st.session_state.results
    if not res:
        ui.empty("No runs yet", "Dub some episodes on the Run page, or load a results zip there.")
        return
    lang = st.segmented_control("Language", list(res), default=list(res)[0], format_func=lambda k: LANGUAGES[k]["name"]) or list(res)[0]
    eps = res[lang]
    agg = aggregate(eps)
    low = sum(v for k, v in agg["meaning_dist"].items() if k <= 3)
    ui.tiles([
        ("Overflowing lines", f"{agg['baseline']['overflow']:.0%} → {agg['dubfit']['overflow']:.0%}", "plain translation → DubFit", True),
        ("Fit without stretching", f"{agg['baseline']['fit']:.0%} → {agg['dubfit']['fit']:.0%}", "spoken inside the window"),
        ("Meaning kept", f"{agg['meaning_avg'] or '—'}/5", f"{low} changed lines scored ≤ 3"),
        ("Sent to a human", agg["needs_review"], f"of {agg['lines']} lines"),
    ])
    per_ep = agg["wall_seconds"] / max(1, agg["episodes"])
    st.caption(f"{agg['episodes']} episodes · {agg['lines']} lines · {agg['calls']} LLM calls · "
               f"{agg['tokens']:,} tokens · about {per_ep:.0f}s per episode · free tier")

    ui.section("Before and after", "share of all lines, measured with TTS")
    st.plotly_chart(chart_before_after(agg), width="stretch", config={"displayModeBar": False})

    ui.section("Checks on the checkers", "are the judge and the timing voice trustworthy?")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**AI judge vs a person**")
        ja = judge_agreement(eps)
        if ja["compared"]:
            st.markdown(
                f"You labelled **{ja['labelled']}** of {ja['changed_lines']} changed lines and rated "
                f"**{ja['human_good_share']:.0%}** acceptable. The judge caught **{ja['both_bad']} of "
                f"{ja['both_bad'] + ja['judge_missed']}** lines you marked bad, with **{ja['false_alarm']}** false alarms "
                f"({ja['agreement']:.0%} agreement).")
            if ja["disagreements"]:
                with st.expander(f"{len(ja['disagreements'])} disagreements"):
                    st.dataframe(pd.DataFrame(ja["disagreements"])[["line", "english", "dub", "judge_score", "human", "kind"]],
                                 width="stretch", hide_index=True)
        else:
            st.caption("Not labelled yet. In Studio, use the “To label” filter and mark 30–40 changed lines Good or Bad.")
    with c2:
        st.markdown("**Timing voice (eSpeak) vs a neural voice**")
        rc = load_ruler(st.session_state.get("results_dir", OUT))
        if rc:
            st.dataframe(pd.DataFrame([{
                "Lines": g.replace(" (DubFit)", ""), "n": s["lines"], "Correlation": s["pearson_r"],
                "Same verdict": f"{s['status_agreement']:.0%}", "Neural pace": f"{s['speed_factor']:.2f}×",
            } for g, s in rc["summary"].items()]), width="stretch", hide_index=True)
            weak = [(g.replace(" (DubFit)", ""), s) for g, s in rc["summary"].items() if (s["status_agreement"] or 0) < 0.7]
            for g, s in weak:
                st.caption(f"⚠ {g}: the neural voice is {s['speed_factor'] - 1:.0%} slower than eSpeak, so eSpeak is too "
                           "lenient here. Measure this language with the neural voice (--engine edge).")
            if not weak:
                st.caption("eSpeak gives the same verdict as the neural voice for every language checked.")
        else:
            st.caption("Not checked yet. Run  python validate_ruler.py --show data/scripts.zip  then refresh.")

    ui.section("Review queue", "where DubFit still needs a person · worst first")
    rows = [ln for r in eps.values() for ln in r["lines"] if ln.get("needs_review")]
    rows.sort(key=lambda x: (-(x["dubfit"]["ratio"]), x.get("meaning_score") or 5))
    if rows:
        st.dataframe(pd.DataFrame([{
            "line": x["id"], "English": x["en"], "DubFit": x["dubfit"]["text"],
            "spoken / window": x["dubfit"]["ratio"], "meaning": x.get("meaning_score"),
            "why": ", ".join(x.get("review_reasons", []))} for x in rows]),
            width="stretch", hide_index=True,
            column_config={"spoken / window": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=2)})
    else:
        st.success("Nothing left for review in this language.")

    ui.section("Where it fails", "")
    st.markdown(
        "- **Tiny windows.** One-word lines (\"Why?\", \"No...\") under about 0.8s have almost no room.\n"
        "- **Formal titles and idioms** (\"Your Grace\", \"my lord\") have no short equivalent.\n"
        "- **The timing voice is a stand-in** for Kuku's production voice. Hindi needed the neural voice to be accurate.\n"
        "- **The judge is an LLM.** It can miss lost sarcasm, so low scores mean \"look\", not \"wrong\".")

def page_how():
    ui.hero("Method", 'One person, one step, one checkable output<span class="dot">.</span>',
            "The person is the localization editor. The step is turning an English screenplay into a dub-ready script. "
            "The output is a translated script where every line carries a measured fit score.")
    steps = [
        ("Parse", "code", "Read every dialogue line with its start, end, speaker, emotion and on/off-screen tag. Every timestamp comes from the script."),
        ("Translate naturally", "ai", "The LLM translates the whole episode the way a subtitle translator would, with the show logline as context."),
        ("Measure", "code", "Speak every line with TTS, trim the silence and compare it to the window: fits, tight (up to 15% over, fixable by a light time-stretch) or overflow."),
        ("Keep what fits", "code", "Lines that fit or are tight stay exactly as translated. An earlier version re-translated every line under a syllable limit; the model over-compressed lines that already fit and damaged meaning, so DubFit now touches only lines that overflow."),
        ("Budget", "code", "For each overflowing line, set a syllable limit from its window, its measured length and its emotion (whispers are slower, shouts faster), using the timing voice's calibrated syllables per second."),
        ("Shorten with context", "ai", "The LLM writes 2 shorter options per overflowing line, seeing the English, the previous and next lines, the speaker and the emotion. Meaning comes first, then natural grammar, then length."),
        ("Pick by measurement", "code", "Both options are spoken and measured. DubFit keeps the one that fits while keeping the most content, for up to 2 rounds."),
        ("Back-check meaning", "ai", "Every changed line gets a blind back-translation, which an LLM judge compares with the English (1–5), checking for reversed meaning, changed subject, lost negation and lost sarcasm."),
        ("Human review", "human", "Lines still too long or scoring ≤ 3 go to a queue, where the editor listens, edits, re-measures and approves."),
    ]
    for i, (t, kind, body) in enumerate(steps, 1):
        tag = {"ai": '<span class="tag ai">AI</span>', "code": '<span class="tag code">Code</span>',
               "human": '<span class="tag human">Human</span>'}[kind]
        st.markdown(f'<div class="card step" style="margin-bottom:8px"><div class="n">{i}</div><div><div class="h4">{e(t)}{tag}</div>'
                    f'<p>{e(body)}</p></div></div>', unsafe_allow_html=True)
    ui.section("Assumptions", "where the brief was silent")
    st.markdown(
        "- The English timings fit the original performance, so they are the target windows.\n"
        "- On-screen lines must end inside their window (lip-sync). V.O./O.S. lines may run into the silence before the next line.\n"
        "- Up to 15% over is acceptable after a light time-stretch. More than that sounds rushed.\n"
        "- Speaker gender is guessed from the name and only used to pick the timing voice.\n"
        "- Kuku's real dubbing voice is unknown, so the timing voice is a stand-in. Swap in the production TTS for exact numbers.")
    ui.section("Scale", "")
    st.markdown(
        "- **2–5 LLM calls per episode per language**, batched per episode and cached on disk. Lines that already fit cost nothing extra. TTS timing with eSpeak is free and runs locally.\n"
        "- Every new show reuses the same pipeline unchanged. Only the screenplay changes.\n"
        "- Swapping the model is one string (Gemini, Groq, OpenAI, Claude, or a local Ollama model).")


pages = [
    st.Page(page_overview, title="Overview", icon=":material/insights:", default=True),
    st.Page(page_studio, title="Studio", url_path="studio", icon=":material/graphic_eq:"),
    st.Page(page_run, title="Run", url_path="run", icon=":material/play_circle:"),
    st.Page(page_quality, title="Quality", url_path="quality", icon=":material/verified:"),
    st.Page(page_how, title="How it works", url_path="how", icon=":material/account_tree:"),
]
nav = st.navigation(pages, position="top")
nav.run()
st.markdown('<div class="footer">dubfit · timing measured, not guessed</div>',
            unsafe_allow_html=True)