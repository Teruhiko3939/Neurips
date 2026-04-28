#!/usr/bin/env python3
"""Build AMI meeting records from manual_ann.

Usage:
  python prepare_ami.py ES2002a   # JSON output to dataset/<id>.json
  python prepare_ami.py           # PKL output to dataset/ami.pkl
"""

import json
import os
import pickle
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepare_helper_ami import (
    build_meeting_list,
    build_segment_text_map,
    build_speaker_indices,
    integrate_argument_relations_into_transcript,
    load_disfluency_word_ids,
    load_ontology_map,
    load_words_for_meeting,
    parse_argumentation,
    parse_participants,
    parse_summary,
    parse_topics,
    parse_transcript,
)

from add_theme import add_theme

ANN_ROOT = os.path.join(SCRIPT_DIR, "ICSI_plus_NXT", "ICSIplus")
ICSI_ROOT = os.path.join(SCRIPT_DIR, "datasets--ami", "amicorpus", "manual_ann")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset")

PROMPT_TEMPLATE = """\
Design a plausible discussion theme and meeting conditions that could produce the following meeting.
Output must be **JSON only** exactly as in the example below, with no additional text.

===========(Meeting)===========
{quotes}
===========(Output Example)===========
{{"theme": [{{"name": "New Product Go-to-Market Strategy", "description": "This meeting discusses the go-to-market strategy for a new product.", "conditions": "One participant each from marketing, sales, and development. Meeting duration: 1 hour."}}]}}
"""

ESITS_PREFIX = re.compile(r'^(ES|IS|TS)')
BASE_SUFFIX  = re.compile(r'^(.+)([abcd])$')
LLM_used = 0

def _series_prefix(mid: str) -> str | None:
    m = ESITS_PREFIX.match(mid)
    return m.group(1) if m else None


def _base_id(mid: str) -> str:
    m = BASE_SUFFIX.match(mid)
    return m.group(1) if m else mid


def _find_reuse_theme(meeting_id: str, dataset: dict, complete_bases: set[str]) -> dict | None:
    """Return an existing theme from the same ES/IS/TS series (different project only), or None."""
    prefix = _series_prefix(meeting_id)
    if not prefix:  # EN/IB/IN → no reuse
        return None
    base = _base_id(meeting_id)
    if base not in complete_bases:  # incomplete series (not all a/b/c/d) → LLM generate
        return None
    # Same series prefix, different project only (ES2002 → ES2003, not ES2002a → ES2002b)
    for mid, rec in dataset.items():
        if _series_prefix(mid) == prefix and _base_id(mid) != base and rec.get("theme") is not None:
            return rec["theme"]
    return None


def process_meeting(ann_root: str, meeting_id: str, maps: tuple, reuse_theme: dict | None = None) -> dict:
    topic_map, da_map, ae_map, ar_map = maps

    optional_checks = [
        ("topics",            os.path.join(ann_root, "topics", f"{meeting_id}.topic.xml")),
        ("abstractive",       os.path.join(ann_root, "abstractive", f"{meeting_id}.abssumm.xml")),
        ("decision",          os.path.join(ann_root, "decision", "manual", f"{meeting_id}.decision.xml")),
        ("argumentation_ar",  os.path.join(ann_root, "argumentation", "ar", f"{meeting_id}.argumentationrels.xml")),
        ("argumentation_dis", os.path.join(ann_root, "argumentation", "dis", f"{meeting_id}.discussions.xml")),
    ]
    missing_files = [label for label, path in optional_checks if not os.path.exists(path)]

    if "topics" in missing_files:
        print(f"  skip {meeting_id}: no topics file")
        return {
            "theme": None,
            "participants": [],
            "discussion": [],
            "argument_units": {},
            "transcript": [],
            "summary": {},
            "missing_files": missing_files,
        }

    speaker_words, _, _ = load_words_for_meeting(ann_root, meeting_id)
    speaker_id_to_pos = build_speaker_indices(speaker_words)
    disfluency_word_ids = load_disfluency_word_ids(ann_root, meeting_id, speaker_words, speaker_id_to_pos)
    seg_map, seg_ids, seg_pos, seg_bounds = build_segment_text_map(
        ann_root, meeting_id, speaker_words, speaker_id_to_pos, disfluency_word_ids
    )

    participants = parse_participants(ann_root, meeting_id)
    topics = parse_topics(ann_root, meeting_id, topic_map, speaker_words, speaker_id_to_pos, disfluency_word_ids)
    arg = parse_argumentation(
        ann_root, meeting_id, ae_map, ar_map,
        speaker_words, speaker_id_to_pos, disfluency_word_ids,
        seg_map, seg_ids, seg_pos, seg_bounds,
    )
    transcript = parse_transcript(ann_root, meeting_id, da_map, speaker_words, speaker_id_to_pos, disfluency_word_ids)
    integrate_argument_relations_into_transcript(transcript, arg)
    summary = parse_summary(ann_root, meeting_id, speaker_words, speaker_id_to_pos, disfluency_word_ids)

    topic_quotes = [str(t.get("quote", "")).strip() for t in topics]
    theme = None
    if reuse_theme is not None and transcript:
        theme = reuse_theme
        print(f"  ↩ theme reused")
    elif add_theme and topics and all(topic_quotes) and transcript:
        global LLM_used
        raw = add_theme(PROMPT_TEMPLATE.format(quotes="\n".join(topic_quotes)))
        LLM_used += 1

        try:
            theme = json.loads(raw)
        except json.JSONDecodeError:
            fix_prompt = f"The following output is invalid JSON. Please output valid JSON only.\n{raw}"
            raw2 = add_theme(fix_prompt)
            LLM_used += 1

            try:
                theme = json.loads(raw2)
            except json.JSONDecodeError as e:
                err_path = os.path.join(OUTPUT_DIR, "json_errors.txt")
                with open(err_path, "a", encoding="utf-8") as ef:
                    ef.write(f"{meeting_id}: {e}\n{raw2}\n\n")
                print(f"  ⚠ JSON parse failed for {meeting_id}, logged to {err_path}")

    print(
        f"✓ {meeting_id}: participants={len(participants)}, topics={len(topics)}, "
        f"arg_units={len(arg['argument_units'])}, discussions={len(arg['discussions'])}, "
        f"transcript_turns={len(transcript)}"
    )
    if missing_files:
        print(f"  missing_optional_files={len(missing_files)} ({', '.join(missing_files)})")

    return {
        "theme": theme,
        "participants": participants,
        "discussion": topics,
        "argument_units": arg,
        "transcript": transcript,
        "summary": summary,
        "missing_files": missing_files,
    }


def load_maps(ann_root: str) -> tuple:
    ont = os.path.join(ann_root, "ontologies")
    return (
        load_ontology_map(os.path.join(ont, "default-topics.xml")),
        load_ontology_map(os.path.join(ont, "da-types.xml")),
        load_ontology_map(os.path.join(ont, "ae-types.xml")),
        load_ontology_map(os.path.join(ont, "ar-types.xml")),
    )


def main() -> None:
    # extract_frag = True  # True: use partial utterances for theme generation / False: use full utterances

    ann_root = os.path.abspath(ANN_ROOT)
    if not os.path.isdir(ann_root):
        print(f"Error: ann-root not found: {ann_root}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    maps = load_maps(ann_root)
    meeting_ids = sys.argv[1:] or build_meeting_list(ann_root, None, "*")
    if not meeting_ids:
        print("Warning: no meeting ids found", file=sys.stderr)
        sys.exit(0)

    ok, ng = 0, 0
    dataset: dict[str, dict] = {}

    # Bases with all 4 phases (a, b, c, d) present in the meeting list
    phase_count: dict[str, set[str]] = {}
    for mid in meeting_ids:
        m = BASE_SUFFIX.match(mid)
        if m:
            phase_count.setdefault(m.group(1), set()).add(m.group(2))
    complete_bases = {base for base, phases in phase_count.items() if phases >= {"a", "b", "c", "d"}}

    for mid in meeting_ids:
        try:
            reuse = _find_reuse_theme(mid, dataset, complete_bases)
            dataset[mid] = process_meeting(ann_root, mid, maps, reuse_theme=reuse)
            ok += 1
        except Exception as e:
            print(f"✗ {mid}: {e}", file=sys.stderr)
            ng += 1

    if sys.argv[1:]:
        for mid, record in dataset.items():
            out_path = os.path.join(OUTPUT_DIR, f"{mid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"  → {out_path}")
    else:
        pkl_path = os.path.join(OUTPUT_DIR, "extracted_ami.pkl")
        # Keep only newly generated LLM themes and remove duplicate theme content.
        seen_themes: set[str] = set()
        pkl_data = {}
        for mid, rec in dataset.items():
            if "topics" in rec.get("missing_files", []):
                continue
            if not rec.get("llm_generated", False):
                continue
            theme_key = json.dumps(rec["theme"], ensure_ascii=False, sort_keys=True)
            if theme_key in seen_themes:
                continue
            seen_themes.add(theme_key)
            pkl_data[mid] = rec
        # else:
        #     pkl_path = os.path.join(OUTPUT_DIR, "ami.pkl")
        #     pkl_data = {mid: rec for mid, rec in dataset.items() if "topics" not in rec.get("missing_files", [])}
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)
        skipped = len(dataset) - len(pkl_data)
        print(f"Dataset pickle saved: {pkl_path} ({len(pkl_data)} meetings, {skipped} skipped)")

    print(f"LLM used {LLM_used} times")
    print(f"Done: {ok} success, {ng} failed")
    if ng:
        sys.exit(1)

if __name__ == "__main__":
    main()
