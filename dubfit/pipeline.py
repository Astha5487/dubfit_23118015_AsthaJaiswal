"""DubFit pipeline for one episode and one target language.

  1. translate  - natural LLM translation of the whole episode               [AI]
  2. measure    - speak every line with TTS, compare it to its window        [code: the ground truth]
  3. keep       - lines that fit (or are within 15%) stay exactly as written [code]
  4. fix        - only overflowing lines: AI writes 2 shorter options with
                  full context; code measures both and keeps the one that
                  fits while keeping the most content. Up to 2 rounds.       [AI + code]
  5. back-check - blind back-translation + meaning judge, changed lines only [AI]
  6. flag       - still too long, or meaning drift -> human review queue     [code]

Design note: an earlier version re-translated *every* line under a syllable
budget. The model treated the budget as a target and compressed lines that
already fit, which damaged meaning. Touching only the lines that overflow fixed it.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from .budget import LANGUAGES, budget, guess_gender, syllables
from .llm import LLM
from .parser import Episode, Line
from .tts import calibrate, measure

TIGHT = 1.15          # up to 15% over can be fixed with a light time-stretch
MAX_ROUNDS = 2

SYSTEM_BASELINE = """You translate English screenplay dialogue for a vertical microdrama into {lang}.
Translate each line naturally and faithfully, as a professional subtitle translator would.
Keep character names unchanged. {script_rule}
Return only JSON: {{"lines": [{{"id": "...", "text": "..."}}]}} with one entry per input id."""

SYSTEM_FIX = """You are an expert dubbing adapter for vertical microdramas, writing {lang} dialogue that an AI voice
will speak over existing video. Each line below was measured and runs too long for its video window.

For EACH line write 2 shorter options in {lang}:
- option 1: a light trim, close to the current translation, about max_syllables or a little over
- option 2: a bolder rewrite that clearly fits within max_syllables

Rules, in priority order:
1. Keep the meaning of the English line: who does what to whom, negation, questions, and any sarcasm or irony.
   Never reverse or change the meaning. Never change who the subject or object is.
2. Write correct, natural, spoken {lang} with correct grammar and gender agreement, the way people talk in a TV drama.
   No broken fragments.
3. Shorten by removing filler, repetition and words the scene already makes obvious (use the previous and next
   lines for context), or by using a shorter natural phrasing. Keep names unless dropping one is the only way to fit.
{script_rule}
Return only JSON: {{"lines": [{{"id": "...", "options": ["option 1", "option 2"]}}]}}"""

SYSTEM_BACK = """Translate each {lang} line into plain English, literally and faithfully.
Do not improve or guess what it 'should' say. Return only JSON: {{"lines": [{{"id": "...", "text": "..."}}]}}"""

SYSTEM_JUDGE = """You check dubbing quality. For each line compare the ORIGINAL English with a BACK-TRANSLATION of the dub.
Score meaning preservation 1-5: 5 same meaning and tone, 4 minor nuance lost, 3 noticeable meaning or tone change,
2 important meaning lost, 1 wrong or reversed. Shortening and paraphrase are fine if the intent survives.
Check carefully: reversed meaning, changed subject/object, lost negation, lost sarcasm or irony.
Give a note under 12 words only when score <= 3.
Return only JSON: {"lines": [{"id": "...", "score": 5, "note": ""}]}"""

SCRIPT_RULES = {
    "hi": "Write Hindi in Devanagari script, in everyday conversational Hindi (Hindustani), not formal Sanskritized Hindi.",
    "es": "Use neutral Latin American Spanish.",
    "pt": "Use Brazilian Portuguese.",
}


def status_for(seconds: float, limit: float) -> tuple[float, str]:
    ratio = seconds / limit if limit > 0 else 9.9
    if ratio <= 1.0:
        return round(ratio, 3), "fit"
    if ratio <= TIGHT:
        return round(ratio, 3), "tight"
    return round(ratio, 3), "overflow"


def _limit(line: Line) -> float:
    # Lip-synced lines must end with the mouth; voice-over may run into the silence after it.
    return line.window if line.on_screen else line.soft_window


def _map(resp: dict) -> dict:
    return {str(x.get("id")): (x.get("text") or "").strip() for x in resp.get("lines", []) if x.get("id")}


def _options(resp: dict) -> dict:
    out = {}
    for x in resp.get("lines", []):
        if not x.get("id"):
            continue
        opts = x.get("options") or ([x["text"]] if x.get("text") else [])
        out[str(x["id"])] = [str(o).strip() for o in opts if str(o).strip()]
    return out


def run_episode(ep: Episode, lang: str, llm: LLM, engine: str = "espeak", back_check: bool = True,
                progress: Callable[[str, float], None] | None = None) -> dict:
    t0 = time.time()
    name = LANGUAGES[lang]["name"]
    rule = SCRIPT_RULES.get(lang, "")
    say = progress or (lambda msg, frac: None)
    rates = {g: calibrate(lang, engine, g) for g in ("F", "M")}
    lines = ep.lines
    by_line = {ln.id: ln for ln in lines}

    rows = []
    for ln in lines:
        g = guess_gender(ln.speaker)
        en = measure(ln.text, "en", engine, g)
        rows.append({
            "id": ln.id, "episode": ep.number, "scene": ln.scene, "speaker": ln.speaker, "gender": g,
            "emotion": ln.emotion, "on_screen": ln.on_screen, "start": ln.start, "end": ln.end,
            "window": ln.window, "soft_window": ln.soft_window, "limit": round(_limit(ln), 3),
            "en": ln.text, "en_seconds": en["seconds"], "en_audio": en["audio"],
            "budget": budget(_limit(ln), ln.emotion, rates[g]),
        })

    def score(text: str, r: dict) -> dict:
        m = measure(text, lang, engine, r["gender"])
        ratio, st = status_for(m["seconds"], r["limit"])
        return {"text": text, "syllables": syllables(text, lang), "seconds": m["seconds"],
                "ratio": ratio, "status": st, "audio": m["audio"]}

    # 1. natural translation of the whole episode
    say("Translating the episode", 0.1)
    base = _map(llm.json(SYSTEM_BASELINE.format(lang=name, script_rule=rule), json.dumps({
        "task": "baseline", "show_context": ep.logline,
        "lines": [{"id": r["id"], "speaker": r["speaker"], "emotion": r["emotion"], "en": r["en"]} for r in rows],
    }, ensure_ascii=False)))

    # 2-3. measure; lines that fit are kept untouched
    say("Measuring with TTS", 0.3)
    for r in rows:
        r["baseline"] = score(base.get(r["id"], ""), r)
        r["dubfit"] = {**r["baseline"], "attempts": 1, "history": [], "changed": False}

    # 4. fix only the overflowing lines, measured candidates
    for rnd in range(MAX_ROUNDS):
        bad = [r for r in rows if r["dubfit"]["status"] == "overflow" and r["dubfit"]["text"]]
        if not bad:
            break
        say(f"Shortening {len(bad)} overflowing lines (round {rnd + 1})", 0.45 + 0.15 * rnd)
        payload = []
        for r in bad:
            ln = by_line[r["id"]]
            d = r["dubfit"]
            target = max(1, min(r["budget"], int(d["syllables"] / d["ratio"]))) if d["ratio"] > 0 else r["budget"]
            payload.append({
                "id": r["id"], "speaker": r["speaker"], "emotion": r["emotion"],
                "previous_line": ln.prev_text, "en": r["en"], "next_line": ln.next_text,
                "current_translation": d["text"], "current_syllables": d["syllables"],
                "spoken_seconds": d["seconds"], "window_seconds": r["limit"], "max_syllables": target,
            })
        opts = _options(llm.json(SYSTEM_FIX.format(lang=name, script_rule=rule), json.dumps({
            "task": "fix", "show_context": ep.logline, "lines": payload}, ensure_ascii=False)))
        for r in bad:
            cands = [score(t, r) for t in opts.get(r["id"], [])]
            if not cands:
                continue
            fitting = [c for c in cands if c["status"] != "overflow"]
            # Prefer a candidate that fits while keeping the most words; otherwise the shortest one.
            best = max(fitting, key=lambda c: c["syllables"]) if fitting else min(cands, key=lambda c: c["ratio"])
            if best["ratio"] < r["dubfit"]["ratio"]:
                hist = r["dubfit"]["history"] + [{k: r["dubfit"][k] for k in ("text", "seconds", "ratio", "status")}]
                r["dubfit"] = {**best, "attempts": r["dubfit"]["attempts"] + 1, "history": hist, "changed": True,
                               "candidates": [{k: c[k] for k in ("text", "seconds", "ratio", "status")} for c in cands]}

    # 5. blind back-translation + judge, only for lines DubFit changed
    changed = [r for r in rows if r["dubfit"].get("changed")]
    if back_check and changed:
        say(f"Meaning check on {len(changed)} changed lines", 0.85)
        back = _map(llm.json(SYSTEM_BACK.format(lang=name), json.dumps({
            "task": "back", "lines": [{"id": r["id"], "text": r["dubfit"]["text"]} for r in changed],
        }, ensure_ascii=False)))
        judge = llm.json(SYSTEM_JUDGE, json.dumps({
            "task": "judge",
            "lines": [{"id": r["id"], "original": r["en"], "back_translation": back.get(r["id"], "")} for r in changed],
        }, ensure_ascii=False))
        jmap = {str(x.get("id")): x for x in judge.get("lines", [])}
        for r in changed:
            j = jmap.get(r["id"], {})
            r["back_translation"] = back.get(r["id"], "")
            try:
                r["meaning_score"] = int(j.get("score")) if j.get("score") is not None else None
            except (TypeError, ValueError):
                r["meaning_score"] = None
            r["meaning_note"] = j.get("note", "")

    # 6. flag for a human
    for r in rows:
        reasons = []
        if r["dubfit"]["status"] == "overflow":
            reasons.append("still too long")
        if not r["dubfit"]["text"]:
            reasons.append("missing translation")
        if r.get("meaning_score") is not None and r["meaning_score"] <= 3:
            reasons.append("meaning drift")
        r["review_reasons"] = reasons
        r["needs_review"] = bool(reasons)

    say("Done", 1.0)
    return {
        "episode": ep.number, "lang": lang, "language": name, "model": llm.model, "engine": engine,
        "pipeline": "v2-fix-only-overflow",
        "logline": ep.logline, "duration": ep.duration, "syl_per_sec": rates,
        "summary": summarize(rows), "usage": llm.usage.to_dict(),
        "wall_seconds": round(time.time() - t0, 1), "lines": rows,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1

    def rate(key, st):
        return round(sum(r[key]["status"] == st for r in rows) / n, 4)

    scores = [r["meaning_score"] for r in rows if r.get("meaning_score") is not None]
    return {
        "lines": len(rows),
        "baseline_fit": rate("baseline", "fit"), "baseline_tight": rate("baseline", "tight"),
        "baseline_overflow": rate("baseline", "overflow"),
        "dubfit_fit": rate("dubfit", "fit"), "dubfit_tight": rate("dubfit", "tight"),
        "dubfit_overflow": rate("dubfit", "overflow"),
        "changed": sum(bool(r["dubfit"].get("changed")) for r in rows),
        "needs_review": sum(r.get("needs_review", False) for r in rows),
        "meaning_avg": round(sum(scores) / len(scores), 2) if scores else None,
        "meaning_low": sum(s <= 3 for s in scores),
        "rewritten": sum(r["dubfit"].get("attempts", 1) > 1 for r in rows),
    }
