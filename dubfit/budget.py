"""Syllable counting and per-line timing budgets. Pure code, no AI."""
from __future__ import annotations

import re

# Languages DubFit supports. 'tts' voices are picked per speaker gender.
LANGUAGES = {
    "hi": {"name": "Hindi", "espeak": "hi", "edge": {"F": "hi-IN-SwaraNeural", "M": "hi-IN-MadhurNeural"}},
    "es": {"name": "Spanish (Latin America)", "espeak": "es-419", "edge": {"F": "es-MX-DaliaNeural", "M": "es-MX-JorgeNeural"}},
    "pt": {"name": "Portuguese (Brazil)", "espeak": "pt-br", "edge": {"F": "pt-BR-FranciscaNeural", "M": "pt-BR-AntonioNeural"}},
    "en": {"name": "English (source)", "espeak": "en-us", "edge": {"F": "en-US-JennyNeural", "M": "en-US-GuyNeural"}},
}

# Emotion tags change how fast a line can naturally be delivered.
SLOW = re.compile(r"whisper|somber|sorrow|gentle|tender|reflect|haunt|grave|weak|faint|trembl|choking|broken|reveren|soft|quiet|hesitant|loving|vulnerable", re.I)
FAST = re.compile(r"outrag|panic|frantic|furious|shout|scream|barked|urgent|desperate|snap|rapid|enraged|yell|hysteric|sharp|angry", re.I)


def emotion_factor(emotion: str) -> float:
    if FAST.search(emotion or ""):
        return 1.10
    if SLOW.search(emotion or ""):
        return 0.88
    return 1.0


_EN_VOWELS = re.compile(r"[aeiouy]+")
_ROMANCE_VOWELS = "aeiouáéíóúâêôãõàüï"
_STRONG = set("aeoáéóâêôãõà")
DEV_INDEP_VOWEL = re.compile(r"[ऄ-औॠॡ]")
DEV_CONSONANT = re.compile(r"[क-हक़-य़]")
DEV_VIRAMA = "्"
DEV_MATRA = re.compile(r"[ा-ौॢॣ]")


def _en_word(w: str) -> int:
    w = w.lower()
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(_EN_VOWELS.findall(w)))


def _romance_word(w: str) -> int:
    w = w.lower()
    count, prev_vowel, prev_strong = 0, False, False
    for ch in w:
        if ch in _ROMANCE_VOWELS:
            strong = ch in _STRONG
            if not prev_vowel or (strong and prev_strong):  # hiatus: two strong vowels split
                count += 1
            prev_vowel, prev_strong = True, strong
        else:
            prev_vowel, prev_strong = False, False
    return max(1, count)


def _hi_word(w: str) -> int:
    count = len(DEV_INDEP_VOWEL.findall(w))
    chars = list(w)
    for i, ch in enumerate(chars):
        if DEV_CONSONANT.match(ch):
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt == "़":  # nukta
                nxt = chars[i + 2] if i + 2 < len(chars) else ""
            if nxt != DEV_VIRAMA:
                count += 1
    # Schwa deletion: a word ending in a bare consonant loses its final vowel ("राम" = ram).
    stripped = re.sub(r"[ऀ-ः़]+$", "", w)
    if count > 1 and stripped and DEV_CONSONANT.match(stripped[-1]):
        count -= 1
    if count == 0 and re.search(r"[A-Za-z]", w):  # English words inside Hindi lines
        return _en_word(w)
    return max(1, count) if re.search(r"\w", w) else 0


def syllables(text: str, lang: str) -> int:
    words = re.findall(r"[\wऀ-ॿ']+", text or "")
    if lang == "hi":
        return sum(_hi_word(w) for w in words)
    if lang in ("es", "pt"):
        return sum(_romance_word(w) for w in words)
    return sum(_en_word(w) for w in words)


def budget(window: float, emotion: str, syl_per_sec: float) -> int:
    """Max syllables a target-language line can carry inside its window."""
    return max(1, int(window * syl_per_sec * emotion_factor(emotion)))


def guess_gender(speaker: str) -> str:
    s = speaker.upper()
    female = ("LADY", "MOTHER", "DUCHESS", "MAID", "WOMAN", "QUEEN", "SISTER", "GIRL", "MATRIARCH",
              "HOUSEKEEPER", "JULIANA", "CATHERINE", "CLARISSA", "PRUDENCE", "NOBLEWOMAN", "PRINCESS",
              "WIFE", "WIDOW", "AUNT", "NURSE", "GOVERNESS", "MISS", "MRS", "BRIDE", "DAUGHTER")
    return "F" if any(k in s for k in female) else "M"
