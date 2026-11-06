"""Text-to-speech timing: the ground truth for 'does this line fit?'.

Two engines:
  * espeak  - offline, free, deterministic. Robotic voice, but a stable ruler.
  * edge    - Microsoft neural voices via edge-tts (free, needs internet).
Durations are measured on the *speech* only (leading/trailing silence trimmed),
then cached on disk so re-runs cost nothing.
"""
from __future__ import annotations

import array
import re
import time
import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from .budget import LANGUAGES

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "tts"
CACHE.mkdir(parents=True, exist_ok=True)

# espeak speaking rate (words/min). Held fixed across languages so the ruler is the same.
ESPEAK_RATE = 175


def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def _key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _trimmed_wav_seconds(path: Path, threshold: int = 500) -> float:
    with wave.open(str(path), "rb") as w:
        rate, width, n = w.getframerate(), w.getsampwidth(), w.getnframes()
        frames = w.readframes(n)
    if width != 2 or not frames:
        return n / float(rate or 1)
    samples = array.array("h", frames)
    first = next((i for i, s in enumerate(samples) if abs(s) > threshold), 0)
    last = next((i for i in range(len(samples) - 1, -1, -1) if abs(samples[i]) > threshold), len(samples) - 1)
    return max(0.0, (last - first) / float(rate))


def _espeak(text: str, lang: str, out: Path) -> float:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    voice = LANGUAGES[lang]["espeak"]
    subprocess.run([exe, "-v", voice, "-s", str(ESPEAK_RATE), "-w", str(out), text],
                   check=True, capture_output=True, timeout=30)
    return _trimmed_wav_seconds(out)


async def _edge_async(text: str, voice: str, out: Path) -> float:
    import edge_tts
    try:
        comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    except TypeError:  # older edge-tts
        comm = edge_tts.Communicate(text, voice)
    first, last = None, 0.0
    with open(out, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                start = chunk["offset"] / 1e7
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                first = start if first is None else min(first, start)
                last = max(last, end)
    return max(0.0, last - (first or 0.0))


def _edge(text: str, lang: str, gender: str, out: Path) -> float:
    voice = LANGUAGES[lang]["edge"][gender]
    try:
        return asyncio.run(_edge_async(text, voice, out))
    except RuntimeError:  # already inside an event loop (e.g. some notebook hosts)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_edge_async(text, voice, out))
        finally:
            loop.close()


def measure(text: str, lang: str, engine: str = "espeak", gender: str = "F") -> dict:
    """Speak `text` and return {'seconds': float, 'audio': path, 'engine': str}."""
    text = (text or "").strip()
    if not text:
        return {"seconds": 0.0, "audio": "", "engine": engine}
    ext = "wav" if engine == "espeak" else "mp3"
    k = _key(engine, lang, gender, ESPEAK_RATE if engine == "espeak" else "", text)
    meta, audio = CACHE / f"{k}.json", CACHE / f"{k}.{ext}"
    if meta.exists() and audio.exists():
        return json.loads(meta.read_text())
    if engine == "espeak":
        secs = _espeak(text, lang, audio)
    else:
        if not re.search(r"\w", text):  # punctuation only, e.g. "..." - nothing to speak
            return {"seconds": 0.0, "audio": "", "engine": engine}
        last = None
        for attempt in range(3):  # the free neural voice sometimes drops a request
            try:
                secs = _edge(text, lang, gender, audio)
                break
            except Exception as ex:
                last = ex
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Neural voice failed 3 times on: {text[:80]!r} ({type(last).__name__})")
    res = {"seconds": round(secs, 3), "audio": str(audio), "engine": engine}
    meta.write_text(json.dumps(res))
    return res


# Fixed calibration sentences per language: used to learn how many syllables per
# second the chosen engine actually speaks, so budgets match the ruler.
CALIBRATION = {
    "en": ["I never thought you would come back for me.", "Gerald is waiting for me at the altar.",
           "You knew this before I walked through those doors.", "Mother spent years writing this book."],
    "hi": ["मैंने कभी नहीं सोचा था कि तुम मेरे लिए वापस आओगे।", "जेराल्ड वेदी पर मेरा इंतज़ार कर रहा है।",
           "तुम यह पहले से जानते थे।", "माँ ने यह किताब लिखने में सालों लगाए।"],
    "es": ["Nunca pensé que volverías por mí.", "Gerald me está esperando en el altar.",
           "Lo sabías antes de que yo cruzara esas puertas.", "Mamá pasó años escribiendo este libro."],
    "pt": ["Eu nunca pensei que você voltaria por mim.", "Gerald está me esperando no altar.",
           "Você sabia disso antes de eu passar por aquelas portas.", "Mamãe passou anos escrevendo este livro."],
}


def calibrate(lang: str, engine: str = "espeak", gender: str = "F") -> float:
    """Measured syllables/second of this engine+language (cached)."""
    from .budget import syllables
    k = CACHE / f"calib-{engine}-{lang}-{gender}-{ESPEAK_RATE}.json"
    if k.exists():
        return json.loads(k.read_text())["syl_per_sec"]
    syl = secs = 0.0
    for s in CALIBRATION[lang]:
        syl += syllables(s, lang)
        secs += measure(s, lang, engine, gender)["seconds"]
    rate = round(syl / secs, 3) if secs else 5.5
    k.write_text(json.dumps({"syl_per_sec": rate}))
    return rate
