"""Checks on DubFit's own measuring tools.

1. Ruler check   - does the eSpeak timing voice agree with a real neural voice?
2. Judge check   - does the LLM meaning judge agree with a human reviewer?
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3)


def summarize_ruler(rows: list[dict]) -> dict:
    """rows: {group, espeak, neural, limit, espeak_status, neural_status, neural_scaled_status}"""
    out = {}
    for group in sorted({r["group"] for r in rows}):
        g = [r for r in rows if r["group"] == group]
        xs, ys = [r["espeak"] for r in g], [r["neural"] for r in g]
        n = len(g)
        out[group] = {
            "lines": n,
            "pearson_r": pearson(xs, ys),
            "speed_factor": round(st.median(y / x for x, y in zip(xs, ys) if x > 0), 3) if n else None,
            "mean_abs_diff_s": round(st.mean(abs(x - y) for x, y in zip(xs, ys)), 3) if n else None,
            "status_agreement": round(sum(r["espeak_status"] == r["neural_status"] for r in g) / n, 3) if n else None,
            "status_agreement_scaled": round(sum(r["espeak_status"] == r["neural_scaled_status"] for r in g) / n, 3) if n else None,
            "overflow_agreement": round(sum((r["espeak_status"] == "overflow") == (r["neural_status"] == "overflow")
                                            for r in g) / n, 3) if n else None,
        }
    return out


def load_ruler(out_dir: str | Path) -> dict | None:
    p = Path(out_dir) / "ruler_check.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def judge_agreement(results: dict[int, dict]) -> dict:
    """Compare the LLM judge (score <= 3 = bad) with human Good/Bad labels on changed lines."""
    labelled = [ln for r in results.values() for ln in r["lines"]
                if ln.get("human_label") in ("good", "bad")]
    changed = [ln for r in results.values() for ln in r["lines"] if ln.get("dubfit", {}).get("changed")]
    scored = [ln for ln in labelled if ln.get("meaning_score") is not None]
    tp = sum(ln["human_label"] == "bad" and ln["meaning_score"] <= 3 for ln in scored)
    fn = sum(ln["human_label"] == "bad" and ln["meaning_score"] > 3 for ln in scored)
    fp = sum(ln["human_label"] == "good" and ln["meaning_score"] <= 3 for ln in scored)
    tn = sum(ln["human_label"] == "good" and ln["meaning_score"] > 3 for ln in scored)
    n = len(scored)
    disagreements = [{
        "line": ln["id"], "english": ln["en"], "dub": ln["dubfit"]["text"],
        "back_translation": ln.get("back_translation", ""), "judge_score": ln["meaning_score"],
        "human": ln["human_label"], "kind": "judge missed it" if ln["human_label"] == "bad" else "false alarm",
    } for ln in scored if (ln["human_label"] == "bad") != (ln["meaning_score"] <= 3)]
    return {
        "changed_lines": len(changed),
        "labelled": len(labelled),
        "human_good_share": round(sum(ln["human_label"] == "good" for ln in labelled) / len(labelled), 3) if labelled else None,
        "compared": n,
        "both_bad": tp, "judge_missed": fn, "false_alarm": fp, "both_good": tn,
        "agreement": round((tp + tn) / n, 3) if n else None,
        "judge_recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        "judge_precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "disagreements": disagreements,
    }