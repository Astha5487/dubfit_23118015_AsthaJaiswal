"""Run: python -m pytest -q   (uses a synthetic snippet, not Kuku content)"""
from dubfit.budget import syllables, budget
from dubfit.parser import parse_episode
from dubfit.pipeline import status_for

SAMPLE = '''EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test.
--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOM - DAY
Scene start: 00:00.00
Scene end: 00:10.00

**CAMERA:**
Start: (00:00.00) | End: (00:02.00)
Wide shot.

**ANNA (Whispering):**
Start: (00:01.00) | End: (00:02.50)
"Are you there?"

**BEN (O.S.) (Sharp):**
Start: (00:04.00) | End: (00:05.00)
"Go home."
'''


def test_parser():
    ep = parse_episode(SAMPLE, "x.txt")
    assert ep.number == 7 and len(ep.lines) == 2
    a, b = ep.lines
    assert (a.speaker, a.emotion, a.on_screen, a.window) == ("ANNA", "Whispering", True, 1.5)
    assert a.soft_end == 4.0 and b.on_screen is False and b.soft_end == 10.0


def test_syllables():
    assert syllables("नमस्ते, आप कैसे हैं? राम घर गया।", "hi") == 11
    assert syllables("¿Por qué estás sosteniendo su mano?", "es") == 11
    assert budget(2.0, "Whispering", 6.0) < budget(2.0, "Shouting", 6.0)


def test_status():
    assert status_for(1.0, 1.0)[1] == "fit"
    assert status_for(1.1, 1.0)[1] == "tight"
    assert status_for(1.3, 1.0)[1] == "overflow"
