"""Batch runner: produce the raw outputs folder for the whole input pack.

  python run_batch.py --show path/to/scripts.zip --langs hi es --model gemini/gemini-3.8-flash
  python run_batch.py --show scripts/ --langs hi --episodes 1-10 --engine edge

Also writes outputs/source_pressure.json (English-only analysis, no LLM needed):
  python run_batch.py --show scripts.zip --source-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dubfit.analysis import source_pressure
from dubfit.llm import LLM
from dubfit.parser import load_show
from dubfit.pipeline import run_episode


def parse_range(spec: str | None, available: list[int]) -> list[int]:
    if not spec:
        return available
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return [e for e in available if e in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", required=True, help="zip file or folder of episode .txt files")
    ap.add_argument("--langs", nargs="+", default=["hi", "es"])
    ap.add_argument("--model", default=os.environ.get("DUBFIT_MODEL", "gemini/gemini-3.8-flash"))
    ap.add_argument("--engine", default="espeak", choices=["espeak", "edge"])
    ap.add_argument("--episodes", default=None, help="e.g. 1-10,15")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--min-interval", type=float, default=4.0, help="seconds between LLM calls (free-tier rate limits)")
    ap.add_argument("--no-back-check", action="store_true")
    ap.add_argument("--source-only", action="store_true")
    a = ap.parse_args()

    episodes = load_show(a.show)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(episodes)} episodes, {sum(len(e.lines) for e in episodes)} dialogue lines")

    sp = source_pressure(episodes, engine="espeak")
    (out / "source_pressure.json").write_text(json.dumps(sp, indent=2, ensure_ascii=False))
    print("English source analysis ->", out / "source_pressure.json")
    if a.source_only:
        return

    chosen = parse_range(a.episodes, [e.number for e in episodes])
    for lang in a.langs:
        llm = LLM(a.model, min_interval=a.min_interval)
        (out / lang).mkdir(exist_ok=True)
        for ep in episodes:
            if ep.number not in chosen:
                continue
            dest = out / lang / f"episode-{ep.number:03d}.json"
            try:
                res = run_episode(ep, lang, llm, engine=a.engine, back_check=not a.no_back_check)
            except Exception as e:
                print(f"  [{lang}] ep {ep.number}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
                continue
            dest.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            s = res["summary"]
            print(f"  [{lang}] ep {ep.number:>3}: baseline fit {s['baseline_fit']:.0%} -> DubFit fit "
                  f"{s['dubfit_fit']:.0%} | overflow {s['baseline_overflow']:.0%} -> {s['dubfit_overflow']:.0%}"
                  f" | review {s['needs_review']}")
        print(f"[{lang}] usage: {llm.usage.to_dict()}")


if __name__ == "__main__":
    main()
