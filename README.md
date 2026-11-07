# dubfit. Dubbed dialogue that fits the frame

**Kuku AI Generalist Assignment · Problem 3 (Find your own problem) · Adapting AI microdramas**

Every line in a Kuku screenplay is locked to a time window, for example `"Not anymore."` from 00:18.0 to 00:19.4.
Translations usually come out longer than the English, so the dubbed line overflows its window. The AI voice then
rushes, gets cut off or falls out of lip-sync, and an editor has to fix lines by hand, for every show in every language.

**DubFit** turns an English screenplay into a dub-ready script in Hindi, Spanish or Portuguese where every line is
**measured** against its time window, and only lines that don't fit are changed.

> **One person · one step · one checkable output**
> The localization editor · English script → dub-ready script · a translated script with a measured fit for every line

🔗 **Live app:** [[your Streamlit link]](https://dubfit-23118015-asthajaiswal.streamlit.app/studio)  ·  🎥 **Walkthrough:** [your video link]

---

## The problem is real (measured on the input pack)

All **1,487 dialogue lines** in the 50 episodes, spoken with TTS and compared with their windows:

| Dub spoken longer than the English by | Lines that no longer fit |
|---|---|
| same length | 11% |
| +10% | 19% |
| +20% | 27% |
| **+30%** | **35%** |
| +40% | 42% |

28% of the English lines already run at 3 or more words per second, so there's little room to grow.
With a real Hindi neural voice, **plain translations overflowed 23–79% of lines per episode**.

## How it works

| # | Step | Done by |
|---|---|---|
| 1 | Parse every dialogue line with its start/end time, speaker, emotion and on-screen vs voice-over (V.O./O.S.) | Code |
| 2 | Translate the whole episode naturally | AI |
| 3 | Speak every translated line with TTS, trim silence, compare to its window: **fits / tight (≤15% over) / overflow** | Code |
| 4 | Keep every line that fits, untouched | Code |
| 5 | Overflowing lines only: write 2 shorter options with context (previous/next line, speaker, emotion) | AI |
| 6 | Speak both options and keep the one that fits while keeping the most content (up to 2 rounds) | Code |
| 7 | Blind back-translation of changed lines, then an LLM judge scores meaning 1–5 | AI |
| 8 | Lines still too long or scoring ≤ 3 go to a review queue: listen, edit, re-measure, approve | Human |

**The AI never decides timing. Code measures everything.** Voice-over lines may run into the silence after them,
because no lips need to match.

## Results (real runs)

**Hindi, 10 episodes, measured with a Microsoft neural voice**

| Episode | Overflow: plain → DubFit | Fit: plain → DubFit | Sent to review |
|---|---|---|---|
| 1 | 38% → 0% | 43% → 62% | 2 |
| 2 | 65% → 6% | 23% → 39% | 6 |
| 3 | 33% → 0% | 54% → 70% | 7 |
| 4 | 45% → 0% | 42% → 70% | 7 |
| 5 | 45% → 0% | 45% → 83% | 5 |
| 6 | 36% → 0% | 57% → 71% | 3 |
| 7 | 50% → 0% | 40% → 75% | 3 |
| 8 | [ ] | [ ] | [ ] |
| 9 | 79% → 12% | 15% → 58% | 9 |
| 10 | 23% → 0% | 59% → 77% | 2 |
| **All** | **[x]% → [y]%** | **[ ]% → [ ]%** | **[ ]** |

**Spanish, 3 episodes:** overflow 12% → 0%, fit 79% → 87%, average meaning 4.07/5.

## How I checked the checkers

**1. Is the timing voice trustworthy?** I compared eSpeak (the offline ruler) with a Microsoft neural voice on
414 random lines.

| Lines | Correlation | Same verdict | Neural pace |
|---|---|---|---|
| English | 0.979 | 87% | 1.11× |
| Spanish | 0.974 | 91% | 1.00× |
| **Hindi** | 0.968 | **42%** | **1.27×** |

eSpeak was fine for English and Spanish but **27% too fast for Hindi**, so Hindi overflow was under-reported.
Hindi is now measured with the neural voice, and the app picks the neural voice automatically for any language that fails this check.

**2. Does the AI judge agree with a person?** I labelled [N] changed Hindi lines myself. I rated [ ]% acceptable,
and the judge caught [x of y] lines I marked bad, with [ ] false alarms. Example of a miss: *"You have shown them exactly what your
word is worth"* became *"how valuable your promise is"*. The sarcasm was lost, yet the judge scored it 5/5.

**3. v1 → v2, a fix driven by real output.** v1 re-translated *every* line under a syllable limit. The model
over-compressed lines that already fit: *"You gave me your word before both our families"* became *"you broke our
promises"*, the opposite meaning. v2 touches only overflowing lines.

## Where it fails
- **Tiny windows.** One-word lines ("Why?", "No...") under about 0.8s have almost no room.
- **Formal titles and idioms** ("Your Grace", "my lord") have no short equivalent.
- **More cuts mean more meaning risk.** Real timing pressure sends 2–9 lines per episode to review.
- **The timing voice is a stand-in** for Kuku's production voice. Swap in the real TTS for exact numbers.
- **The judge is an LLM** and can miss lost sarcasm, which is why a person stays in the loop.

## Scale and cost
- **3–5 LLM calls per episode per language**, batched and cached. About **[ ]k tokens per episode** on Groq's free tier.
- Lines that already fit cost nothing extra. eSpeak timing is free and local; the neural voice is free and online.
- **Any new show runs unchanged:** upload a Kuku-format zip. Each show's results are kept separately.
- **Swapping the model is one setting** (Groq, Gemini, OpenAI, Claude or local Ollama, through LiteLLM). During this
  project Gemini returned overload errors, and I switched to Groq with no code changes.

---

## Run it

### Use the hosted app
1. Open the live app and go to **Run**. Upload the input pack zip (the screenplays aren't in this repo, for confidentiality).
2. **Studio:** pick a language and an episode, then press **Dub** or browse the results. Listen, edit, re-measure and approve lines.
3. **Quality:** see the before/after results, the checks and the review queue.

### Run locally
```bash
pip install -r requirements.txt
brew install espeak-ng            # Linux: sudo apt-get install espeak-ng
export GROQ_API_KEY="gsk_..."     # free at console.groq.com
export DUBFIT_MODEL="groq/openai/gpt-oss-120b"
mkdir -p data && cp /path/to/scripts.zip data/
streamlit run app.py
```

### Produce raw outputs (batch)
```bash
# English timing analysis only (no LLM)
python run_batch.py --show data/scripts.zip --source-only

# Dub episodes 1–10 into Hindi, measured with the neural voice
python run_batch.py --show data/scripts.zip --langs hi --episodes 1-10 --engine edge --min-interval 20

# A new show: keep its outputs separate
python run_batch.py --show newshow.zip --langs hi es --episodes 1-3 --engine edge --out outputs_newshow

# Check the timing voice against a neural voice
python validate_ruler.py --show data/scripts.zip
```
Results go to `outputs/<lang>/episode-XXX.json`, `outputs/source_pressure.json` and `outputs/ruler_check.json`.

### Deploy (Streamlit Community Cloud)
Private repo, main file `app.py`, Python 3.11. Add these secrets:
```toml
GROQ_API_KEY = "gsk_..."
DUBFIT_MODEL = "groq/openai/gpt-oss-120b"
```
`packages.txt` installs eSpeak. The API key stays on the server and is never sent to the browser.

## Project structure
```
app.py              Streamlit app: Overview · Studio · Run · Quality · How it works
run_batch.py        batch runner for raw outputs
validate_ruler.py   eSpeak vs neural voice check
dubfit/parser.py    screenplay → timed lines (code only)
dubfit/budget.py    syllable counting (hi/es/pt/en) + per-line budgets
dubfit/tts.py       eSpeak / neural TTS timing, calibration, cache
dubfit/llm.py       one swappable LLM client (LiteLLM), retries, cache
dubfit/pipeline.py  translate → measure → fix overflow only → back-check → flag
dubfit/checks.py    ruler check + judge-vs-human agreement
dubfit/analysis.py  show-level timing pressure + aggregates
dubfit/ui.py        theme and components
outputs/            raw outputs from real runs
tests/              python -m pytest -q
```

## Assumptions
- The English timestamps are the target windows, because they fit the original performance.
- On-screen lines must end inside their window. V.O./O.S. lines may run into the silence that follows.
- Up to 15% over is fixable with a light time-stretch; more sounds rushed.
- Speaker gender is guessed from the name and only picks the timing voice.

## Tools used
Python, Streamlit, LiteLLM, Groq (gpt-oss-120b), eSpeak NG, edge-tts, Plotly. AI assistants helped with planning,
code and debugging. [Add one place where an AI tool got it wrong, e.g. the first parser regex missed 62 of 72
voice-over lines, which I caught by checking against a plain text search.]

---
*The input pack is Kuku's confidential content and is not included in this repository.*
