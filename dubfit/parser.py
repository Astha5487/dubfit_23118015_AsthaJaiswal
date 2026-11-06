"""Screenplay parser: turns Kuku-format episode scripts into timed dialogue lines.

Pure code, no AI. Every timestamp in the output comes straight from the script,
so nothing downstream can invent a timing.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "TRANSITION"}

BLOCK = re.compile(
    r"^\*\*(?P<head>[^*\n]+?):\*\*[ \t]*\n"
    r"Start:\s*\((?P<s>\d+:\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((?P<e>\d+:\d+(?:\.\d+)?)\)[ \t]*\n"
    r"(?P<body>.*?)(?=\n[ \t]*\n|\n\*\*|\Z)",
    re.S | re.M,
)
SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.M | re.I)
SCENE_HEADING = re.compile(r"^Scene heading:\s*(.+)$", re.M | re.I)
SCENE_END = re.compile(r"^Scene end:\s*(\d+:\d+(?:\.\d+)?)", re.M | re.I)
EP_NUM = re.compile(r"^EPISODE NUMBER:\s*Episode\s*(\d+)", re.M | re.I)
LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.+)$", re.M | re.I)
VOICE_TAGS = re.compile(r"(?<![A-Z])(V\.O\.?|O\.S\.?|O\.C\.?|OFF-SCREEN|VOICE CONTINUING)(?![A-Z])", re.I)


def ts(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + float(seconds)


@dataclass
class Line:
    id: str
    episode: int
    index: int
    scene: int
    scene_heading: str
    speaker: str
    emotion: str
    text: str
    start: float
    end: float
    on_screen: bool                # False for V.O. / O.S. -> no lip-sync needed
    soft_end: float = 0.0          # when the next line starts (or scene ends)
    prev_text: str = ""
    next_text: str = ""

    @property
    def window(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def soft_window(self) -> float:
        return round(max(self.soft_end, self.end) - self.start, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window"] = self.window
        d["soft_window"] = self.soft_window
        return d


@dataclass
class Episode:
    number: int
    name: str
    logline: str
    lines: list[Line] = field(default_factory=list)
    duration: float = 0.0


def _split_head(head: str) -> tuple[str, str, bool]:
    """'LADY JULIANA (V.O.) (Cold, Resolute)' -> ('LADY JULIANA', 'Cold, Resolute', False)."""
    parens = re.findall(r"\(([^)]*)\)", head)
    speaker = re.sub(r"\s*\(.*", "", head).strip()
    on_screen = not any(VOICE_TAGS.search(p) for p in parens) and not VOICE_TAGS.search(speaker)
    emotions = []
    for p in parens:
        cleaned = VOICE_TAGS.sub("", p).replace("/", ",")
        cleaned = ", ".join(x.strip() for x in cleaned.split(",") if x.strip() and x.strip() not in {"Continues"})
        if cleaned:
            emotions.append(cleaned)
    return speaker, ", ".join(emotions), on_screen


def parse_episode(text: str, name: str = "") -> Episode:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")  # Windows/Word line endings
    m = EP_NUM.search(text)
    number = int(m.group(1)) if m else int((re.findall(r"\d+", name) or [0])[0])
    lm = LOGLINE.search(text)
    ep = Episode(number=number, name=name, logline=lm.group(1).strip() if lm else "")

    scenes = [(sm.start(), int(sm.group(1))) for sm in SCENE.finditer(text)]
    headings = [(hm.start(), hm.group(1).strip()) for hm in SCENE_HEADING.finditer(text)]
    scene_ends = {}
    for (pos, num) in scenes:
        nxt = [p for p, _ in scenes if p > pos]
        chunk = text[pos: nxt[0] if nxt else len(text)]
        se = SCENE_END.search(chunk)
        if se:
            scene_ends[num] = ts(se.group(1))

    def locate(pos: int) -> tuple[int, str]:
        num = max([n for p, n in scenes if p <= pos], default=1)
        head = max([(p, h) for p, h in headings if p <= pos], default=(0, ""))[1]
        return num, head

    max_end = 0.0
    for b in BLOCK.finditer(text):
        start, end = ts(b.group("s")), ts(b.group("e"))
        max_end = max(max_end, end)
        speaker, emotion, on_screen = _split_head(b.group("head").strip())
        body = " ".join(b.group("body").split())
        if speaker.upper() in NON_DIALOGUE or not body.startswith(("\"", "“")):
            continue
        scene, heading = locate(b.start())
        clean = body.strip().strip("\"“”").strip()
        idx = len(ep.lines)
        ep.lines.append(Line(
            id=f"E{number:03d}-L{idx + 1:03d}", episode=number, index=idx, scene=scene,
            scene_heading=heading, speaker=speaker, emotion=emotion, text=clean,
            start=start, end=end, on_screen=on_screen,
        ))
    ep.duration = max(max_end, max(scene_ends.values(), default=0.0))

    for i, ln in enumerate(ep.lines):
        nxt = ep.lines[i + 1] if i + 1 < len(ep.lines) else None
        if nxt and nxt.scene == ln.scene:
            ln.soft_end = max(ln.end, nxt.start)
        else:
            ln.soft_end = max(ln.end, scene_ends.get(ln.scene, ln.end))
        ln.prev_text = ep.lines[i - 1].text if i else ""
        ln.next_text = nxt.text if nxt else ""
    return ep


def load_show(source) -> list[Episode]:
    """Load episodes from a folder path, a .zip path, or uploaded zip bytes."""
    files: list[tuple[str, str]] = []
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        for p in sorted(Path(source).rglob("*.txt")):
            files.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    else:
        with zipfile.ZipFile(source) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                    files.append((Path(n).name, z.read(n).decode("utf-8", errors="replace")))
    eps = [parse_episode(t, n) for n, t in files]
    eps = [e for e in eps if e.lines]
    return sorted(eps, key=lambda e: e.number)
