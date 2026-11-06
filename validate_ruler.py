"""Check the timing ruler: does eSpeak agree with a real neural voice?

Speaks a sample of lines with BOTH eSpeak (the offline ruler DubFit uses) and a
Microsoft neural voice (edge-tts, free, needs internet), then compares:
  - how well durations correlate (Pearson r)
  - the speed difference between the voices (median neural / eSpeak)
  - how often both voices give the same verdict: fits / tight / overflow

  python validate_ruler.py --show data/scripts.zip                # English + every dubbed language in outputs/
  python validate_ruler.py --show data/scripts.zip --n 100        # smaller sample

Writes outputs/ruler_check.json (shown on the Quality page).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dubfit.analysis import load_results
from dubfit.budget import guess_gender
from dubfit.checks import summarize_ruler
from dubfit.parser import load_show
from dubfit.pipeline import status_for
from dubfit.tts import edge_available, espeak_available, measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", required=True)
    ap.add_argument("--n", type=int, default=150, help="lines to sample per group (English, each dubbed language)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not (espeak_available() and edge_available()):
        raise SystemExit("Needs both espeak-ng and edge-tts installed (pip install edge-tts).")

    rnd = random.Random(a.seed)
    samples = []  # (group, lang, text, gender, limit, id)
    eps = load_show(a.show)
    en_lines = [ln for ep in eps for ln in ep.lines]
    for ln in rnd.sample(en_lines, min(a.n, len(en_lines))):
        samples.append(("English source", "en", ln.text, guess_gender(ln.speaker),
                        ln.window if ln.on_screen else ln.soft_window, ln.id))
    for lang, results in load_results(a.out).items():
        lines = [ln for r in results.values() for ln in r["lines"] if ln["dubfit"]["text"]]
        for ln in rnd.sample(lines, min(a.n, len(lines))):
            samples.append((f"{results[next(iter(results))]['language']} (DubFit)", lang, ln["dubfit"]["text"],
                            ln.get("gender", "F"), ln["limit"], ln["id"]))

    print(f"Measuring {len(samples)} lines with eSpeak and a neural voice ...")
    rows = []
    for i, (group, lang, text, g, limit, lid) in enumerate(samples, 1):
        try:
            es = measure(text, lang, "espeak", g)["seconds"]
            nv = measure(text, lang, "edge", g)["seconds"]
        except Exception as ex:
            print(f"  skip {lid}: {type(ex).__name__}: {str(ex)[:100]}")
            continue
        if es <= 0 or nv <= 0:
            continue
        rows.append({"group": group, "lang": lang, "id": lid, "text": text, "limit": limit,
                     "espeak": es, "neural": nv})
        if i % 25 == 0:
            print(f"  {i}/{len(samples)}")

    summary_raw = summarize_ruler([{**r, "espeak_status": "", "neural_status": "", "neural_scaled_status": ""} for r in rows])
    for r in rows:
        factor = summary_raw[r["group"]]["speed_factor"] or 1.0
        r["espeak_status"] = status_for(r["espeak"], r["limit"])[1]
        r["neural_status"] = status_for(r["neural"], r["limit"])[1]
        # Same verdict once the neural voice is slowed/sped to eSpeak's overall pace:
        r["neural_scaled_status"] = status_for(r["neural"] / factor, r["limit"])[1]
    summary = summarize_ruler(rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ruler_check.json").write_text(json.dumps({"summary": summary, "lines": rows}, indent=2, ensure_ascii=False))
    for group, s in summary.items():
        print(f"{group:28s} n={s['lines']:>4}  r={s['pearson_r']}  neural/eSpeak speed={s['speed_factor']}  "
              f"same verdict={s['status_agreement']:.0%}  (pace-matched {s['status_agreement_scaled']:.0%})  "
              f"overflow yes/no agree={s['overflow_agreement']:.0%}")
    print("Saved", out / "ruler_check.json")


if __name__ == "__main__":
    main()