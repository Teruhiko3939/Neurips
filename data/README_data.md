# data/

Scripts for building training and evaluation records from meeting datasets (AMI and ICSI).

---

## Directory Structure

```
data/
├── add_theme.py              # Utility for generating discussion themes via LLM
├── prepare_icsi.py           # Build script for the ICSI dataset
├── prepare_helper_icsi.py    # Parse/transform helper functions for ICSI
├── prepare_ami.py            # Build script for the AMI dataset
├── prepare_helper_ami.py     # Parse/transform helper functions for AMI
├── dataset/                  # Output directory (JSON / PKL)
├── datasets--ami/            # AMI corpus (amicorpus/manual_ann/)
└── ICSI_plus_NXT/            # ICSI_plus_NXT corpus (ICSIplus/)
```

---

## File Descriptions

### `add_theme.py`

A shared utility that queries an LLM to generate "discussion themes" from meeting transcripts and topic quotations.

- `add_theme(prompt: str) -> str`  
  Calls the LLM and returns the response as a string. Used by `prepare_icsi.py` and `prepare_ami.py`.
- When run standalone, sends a simple connectivity-check prompt.

---

### `prepare_icsi.py`

Builds meeting records from the ICSI_plus_NXT corpus.

#### Usage

```bash
# All meetings → dataset/icsi.pkl
python data/prepare_icsi.py

# Single meeting → dataset/Bmr003.json
python data/prepare_icsi.py Bmr003

# Multiple meetings → dataset/Bmr003.json, dataset/Bro003.json
python data/prepare_icsi.py Bmr003 Bro003
```

#### Overview

1. Retrieves transcript, topics, and summaries via the parse functions in `prepare_helper_icsi.py`.
2. Generates themes with `add_theme` and structures them via `json.loads` (retries once on failure).
3. In batch mode, only records that have LLM-generated themes and no duplicate themes are saved to `icsi.pkl`.

#### Output Format (per record)

```json
{
  "theme": {"theme": [{"name": "...", "description": "...", "conditions": "..."}]},
  "participants": [],
  "discussion": [{"id": "...", "topic_type": "", "description": "...", "quote": "..."}],
  "argument_units": {"discussions": [], "argument_units": []},
  "transcript": [{"id": "...", "speaker": "...", "dialogue_act": "", "text": "...", "turn": 1, "argument_relations": []}],
  "summary": {"abstractive": {"overall": "", "actions": "", "decisions": "", "problems": ""}, "decisions": [], "participant_summaries": {}},
  "missing_files": []
}
```

---

### `prepare_helper_icsi.py`

ICSI-specific helpers called by `prepare_icsi.py`.

| Function | Description |
|---|---|
| `list_meetings_with_summary()` | Returns a list of meeting IDs for which summary files exist |
| `load_words(meeting_id)` | Builds a `nite:id → text` map from Words XML |
| `load_segment_map(meeting_id, word_map)` | Builds a segment map from Segments XML |
| `get_transcript(seg_map)` | Converts the segment map into a chronologically ordered transcript |
| `get_topics(meeting_id, seg_map)` | Retrieves a topic list (with quotations) from TopicSegmentation XML |
| `get_summary(meeting_id)` | Returns the abstractive summary XML in AMI-compatible format |
| `process_meeting(meeting_id)` | Entry point: builds and returns a complete record for one meeting |

Data sources:
- `ICSI_plus_NXT/ICSIplus/Words/` — Word XML
- `ICSI_plus_NXT/ICSIplus/Segments/` — Segment XML
- `ICSI_plus_NXT/ICSIplus/Contributions/TopicSegmentation/` — Topic XML
- `ICSI_plus_NXT/ICSIplus/Contributions/Summarization/abstractive/` — Summary XML

---

### `prepare_ami.py`

Builds meeting records from the AMI corpus.

#### Usage

```bash
# Single meeting → dataset/ES2002a.json
python data/prepare_ami.py ES2002a

# All meetings → dataset/extracted_ami.pkl
python data/prepare_ami.py
```

#### Overview

1. Retrieves transcript, participants, topics, argumentation structure, and summaries via the parse functions in `prepare_helper_ami.py`.
2. Theme generation uses a theme-reuse strategy:
   - **ES / IS / TS series** (when all four phases a/b/c/d are available)  
     Reuses an existing theme from another project in the same series (e.g., ES2002 → ES2003).
   - **When reuse is not possible**, generates a theme via `add_theme`.
3. In batch mode, only records that have LLM-generated themes and no duplicate themes are saved to `extracted_ami.pkl`.

#### Output Format (per record)

Same as `prepare_icsi.py`. Includes the additional AMI-specific argumentation fields (`argument_units.discussions` / `argument_relations`).

---

### `prepare_helper_ami.py`

AMI-specific helpers called by `prepare_ami.py`.

| Function | Description |
|---|---|
| `load_ontology_map(xml_path)` | Converts ontology XML into an `id → {name, gloss}` map |
| `load_words_for_meeting(...)` | Reads Words XML for all speakers |
| `build_speaker_indices(speaker_words)` | Builds a word-ID → position-index map |
| `load_disfluency_word_ids(...)` | Returns the set of word IDs for disfluencies to be removed |
| `build_segment_text_map(...)` | Builds a segment-text map from Segments XML |
| `parse_participants(...)` | Retrieves participant information |
| `parse_topics(...)` | Retrieves a topic list (with quotations) |
| `parse_argumentation(...)` | Retrieves argumentation structure (AE / AR / discussions) |
| `integrate_argument_relations_into_transcript(...)` | Integrates argument relations into the transcript |
| `parse_transcript(...)` | Generates a chronologically ordered transcript |
| `parse_summary(...)` | Retrieves abstractive summaries, decisions, and per-participant summaries |
| `build_meeting_list(...)` | Builds the list of meeting IDs to process |

Data sources (under `datasets--ami/amicorpus/manual_ann/`):
- `words/` — Word XML
- `segments/` — Segment XML
- `topics/` — Topic XML
- `dialogueActs/` — Dialogue act XML
- `argumentation/ae/`, `ar/`, `dis/` — Argumentation structure XML
- `abstractive/` — Summary XML
- `decision/manual/` — Decision XML
- `participantSummaries/` — Per-participant summary XML
- `corpusResources/meetings.xml` — Participant metadata

---

## Dataset Licenses

### AMI Meeting Corpus

| Item | Details |
|---|---|
| Full name | AMI Meeting Corpus |
| Provider | The AMI Project Consortium (University of Edinburgh, et al.) |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Reference | https://groups.inf.ed.ac.uk/ami/corpus/ |

- Secondary use (including academic and commercial) is permitted, but **copyright notice and attribution to the original source** are required.
- Text and annotation data included in this repository are subject to the CC BY 4.0 terms.

> Carletta, J. et al. (2005). The AMI Meeting Corpus: A Pre-announcement. *MLMI 2005*, LNCS 3869.

---

### ICSI Meeting Corpus / ICSI-plus NXT

| Item | Details |
|---|---|
| Full name | ICSI Meeting Corpus |
| Provider | International Computer Science Institute (ICSI), Berkeley |
| LDC Catalog | [LDC2004S02](https://catalog.ldc.upenn.edu/LDC2004S02) |
| ICSI-plus NXT | Extended version with additional annotations by Morgan, N. et al. |
| License | LDC User Agreement (non-commercial research use). Audio data requires an LDC member license. Text and annotation parts are available for research use. |
| Reference | https://groups.inf.ed.ac.uk/ami/icsi/ |

- **Audio files** (`.sph`, etc.) are subject to the LDC license and are not included in this repository.
- Only the text and annotation XML under `ICSI_plus_NXT/ICSIplus/` is used in this repository.

> Janin, A. et al. (2003). The ICSI Meeting Corpus. *ICASSP 2003*.

---

## Dependencies

```
add_theme.py          ← models/llm.py, const/consts.py, utils/run_helper.py
prepare_icsi.py       ← prepare_helper_icsi.py, add_theme.py
prepare_ami.py        ← prepare_helper_ami.py, add_theme.py
prepare_helper_icsi.py ← standard library only
prepare_helper_ami.py  ← standard library only
```
