#!/usr/bin/env python3
"""Build ICSI meeting records from ICSI_plus_NXT.

Usage:
  python data/prepare_icsi.py              # 全ミーティング → dataset/icsi.pkl
  python data/prepare_icsi.py Bmr003       # 単一ミーティング → dataset/Bmr003.json
  python data/prepare_icsi.py Bmr003 Bro003  # 複数指定
"""

import json
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepare_helper_icsi import list_meetings_with_summary, process_meeting
from add_theme import add_theme

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "icsi.pkl")

PROMPT_TEMPLATE = """\
Design a plausible discussion theme and meeting conditions that could produce the following meeting.
Output must be **JSON only** exactly as in the example below, with no additional text.

===========(Meeting)===========
{quotes}
===========(Output Example)===========
{{"theme": [{{"name": "New Product Go-to-Market Strategy", "description": "This meeting discusses the go-to-market strategy for a new product.", "conditions": "One participant each from marketing, sales, and development. Meeting duration: 1 hour."}}]}}
"""

LLM_used = 0


def _generate_theme(meeting_id: str, record: dict) -> None:
    """record を in-place で更新し、theme を LLM で生成する。"""
    global LLM_used

    topics = record.get("discussion", [])
    transcript = record.get("transcript", [])
    topic_quotes = [str(t.get("quote", "")).strip() for t in topics]

    if not (topics and any(topic_quotes) and transcript):
        return

    raw = add_theme(PROMPT_TEMPLATE.format(quotes="\n".join(topic_quotes)))
    LLM_used += 1

    try:
        record["theme"] = json.loads(raw)
        record["llm_generated"] = True
        print(f"  ✦ theme generated")
    except json.JSONDecodeError:
        fix_prompt = f"The following output is invalid JSON. Please output valid JSON only.\n{raw}"
        raw2 = add_theme(fix_prompt)
        LLM_used += 1
        try:
            record["theme"] = json.loads(raw2)
            record["llm_generated"] = True
            print(f"  ✦ theme generated (retry)")
        except json.JSONDecodeError as e:
            err_path = os.path.join(OUTPUT_DIR, "json_errors.txt")
            with open(err_path, "a", encoding="utf-8") as ef:
                ef.write(f"{meeting_id}: {e}\n{raw2}\n\n")
            print(f"  ⚠ JSON parse failed for {meeting_id}, logged to {err_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    meeting_ids = sys.argv[1:] or list_meetings_with_summary()
    if not meeting_ids:
        print("Warning: no meeting ids found", file=sys.stderr)
        sys.exit(0)

    single_mode = len(sys.argv) > 1

    dataset: dict[str, dict] = {}
    ok, ng = 0, 0

    for meeting_id in meeting_ids:
        try:
            record = process_meeting(meeting_id)
            _generate_theme(meeting_id, record)
            dataset[meeting_id] = record
            ok += 1
        except Exception as e:
            print(f"✗ {meeting_id}: {e}", file=sys.stderr)
            ng += 1

    if single_mode:
        # 単一 or 複数指定 → JSON ファイル保存
        for mid, record in dataset.items():
            out_path = os.path.join(OUTPUT_DIR, f"{mid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"  → {out_path}")
    else:
        # 全件 → pkl 保存 (AMI と同様に LLM 生成済み・重複排除のみ)
        seen_themes: set[str] = set()
        pkl_data = {}
        for mid, rec in dataset.items():
            if not rec.get("llm_generated", False):
                continue
            if rec.get("theme") is None:
                continue
            theme_key = json.dumps(rec["theme"], ensure_ascii=False, sort_keys=True)
            if theme_key in seen_themes:
                continue
            seen_themes.add(theme_key)
            pkl_data[mid] = rec
        with open(OUTPUT_PKL, "wb") as f:
            pickle.dump(pkl_data, f)
        skipped = len(dataset) - len(pkl_data)
        print(f"\n完了: {ok} 件成功, {ng} 件失敗")
        print(f"保存先: {OUTPUT_PKL} ({len(pkl_data)} ミーティング保存, {skipped} スキップ)")

    print(f"LLM used {LLM_used} times")
    if ng:
        sys.exit(1)


if __name__ == "__main__":
    main()

