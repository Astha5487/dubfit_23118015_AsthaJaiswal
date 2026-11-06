"""Show-level analysis: how much room each line has before a dub stops fitting.

Runs on the English source only (no LLM), so it works on any new show instantly.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from .budget import guess_gender, syllables
from .parser import Episode
from .tts import measure


def source_pressure(episodes: list[Episode], engine: str = "espeak") -> dict:
    lines = []
    for ep in episodes:
        for ln in ep.lines:
            words = len(ln.text.split())
            sec = measure(ln.text, "en", engine, guess_gender(ln.speaker))["seconds"]
            limit = ln.window if ln.on_screen else ln.soft_window
            ratio = sec / limit if limit else 9.9
            lines.append({
                "id": ln.id, "episode": ep.number, "speaker": ln.speaker, "emotion": ln.emotion,
                "text": ln.text, "window": ln.window, "limit": round(limit, 3), "on_screen": ln.on_screen,
                "words": words, "wps": round(words / ln.window, 3) if ln.window else 0,
                "syllables": syllables(ln.text, "en"), "en_seconds": sec,
                "ratio": round(ratio, 3), "headroom": round(1 - ratio, 3),
            })
    n = len(lines) or 1
    per_ep = []
    for ep in episodes:
        el = [x for x in lines if x["episode"] == ep.number]
        if not el:
            continue
        per_ep.append({
            "episode": ep.number, "lines": len(el), "duration": ep.duration,
            "median_wps": round(st.median(x["wps"] for x in el), 2),
            # A line with < 20% headroom breaks if translation grows it by 20%+.
            "no_room_share": round(sum(x["headroom"] < 0.2 for x in el) / len(el), 3),
            "en_overflow_share": round(sum(x["ratio"] > 1 for x in el) / len(el), 3),
        })
    return {
        "engine": engine,
        "episodes": len(per_ep), "lines": len(lines),
        "on_screen_lines": sum(x["on_screen"] for x in lines),
        "median_window": round(st.median(x["window"] for x in lines), 2) if lines else 0,
        "median_wps": round(st.median(x["wps"] for x in lines), 2) if lines else 0,
        "share_wps_3plus": round(sum(x["wps"] >= 3 for x in lines) / n, 3),
        "en_fit_share": round(sum(x["ratio"] <= 1 for x in lines) / n, 3),
        # How many lines break if a translation is X% longer when spoken
        "break_at_growth": {f"{g}%": round(sum(x["ratio"] * (1 + g / 100) > 1 for x in lines) / n, 3)
                            for g in (0, 10, 20, 30, 40)},
        "per_episode": per_ep,
        "lines_detail": lines,
    }


def load_results(folder: str | Path) -> dict[str, dict[int, dict]]:
    """outputs/<lang>/episode-XXX.json -> {lang: {episode: result}}"""
    out: dict[str, dict[int, dict]] = {}
    root = Path(folder)
    if not root.exists():
        return out
    for f in sorted(root.glob("*/episode-*.json")):
        try:
            r = json.loads(f.read_text())
            out.setdefault(r["lang"], {})[r["episode"]] = r
        except Exception:
            continue
    return out


def aggregate(results: dict[int, dict]) -> dict:
    rows = [ln for r in results.values() for ln in r["lines"]]
    n = len(rows) or 1

    def share(key, stt):
        return sum(x[key]["status"] == stt for x in rows) / n

    scores = [x["meaning_score"] for x in rows if x.get("meaning_score") is not None]
    usage = [r.get("usage", {}) for r in results.values()]
    return {
        "episodes": len(results), "lines": len(rows),
        "baseline": {s: share("baseline", s) for s in ("fit", "tight", "overflow")},
        "dubfit": {s: share("dubfit", s) for s in ("fit", "tight", "overflow")},
        "needs_review": sum(x.get("needs_review", False) for x in rows),
        "meaning_avg": round(sum(scores) / len(scores), 2) if scores else None,
        "meaning_dist": {k: scores.count(k) for k in range(1, 6)},
        "rewritten": sum(x["dubfit"].get("attempts", 1) > 1 for x in rows),
        "calls": sum(u.get("calls", 0) for u in usage),
        "tokens": sum(u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) for u in usage),
        "cost_usd": round(sum(u.get("cost_usd", 0) for u in usage), 4),
        "wall_seconds": round(sum(r.get("wall_seconds", 0) for r in results.values()), 1),
    }
