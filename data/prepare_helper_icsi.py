"""
ICSI_plus_NXT データセット ヘルパー関数

prepare_icsi.py から呼び出される。
"""

import glob
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).parent / "ICSI_plus_NXT" / "ICSIplus"
WORDS_DIR = BASE_DIR / "Words"
SEGS_DIR = BASE_DIR / "Segments"
TOPICS_DIR = BASE_DIR / "Contributions" / "TopicSegmentation"
SUMM_DIR = BASE_DIR / "Contributions" / "Summarization" / "abstractive"

NITE_NS = "http://nite.sourceforge.net/"
_NITE_ID = f"{{{NITE_NS}}}id"

MAX_TOPIC_QUOTE_SEGS = 6  # トピックごとに引用するセグメント数の上限


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def _tag(elem: ET.Element) -> str:
    t = elem.tag
    return t.split("}", 1)[1] if "}" in t else t


def _nite_id(elem: ET.Element) -> str:
    return elem.get(_NITE_ID, "") or elem.get("nite:id", "")


def _safe_float(s: str | None, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def list_meetings_with_summary() -> list[str]:
    """abstractive 要約が存在するミーティング ID の一覧を返す。"""
    return sorted(p.stem.split(".")[0] for p in SUMM_DIR.glob("*.abssumm.xml"))


# ──────────────────────────────────────────────
# Words ロード
# ──────────────────────────────────────────────

def load_words(meeting_id: str) -> dict[str, str]:
    """
    Words/{meeting}.*.words.xml を全スピーカー分読み込み、
    nite:id → テキスト のマップを返す。
    """
    word_map: dict[str, str] = {}
    for path in WORDS_DIR.glob(f"{meeting_id}.*.words.xml"):
        try:
            root = ET.parse(str(path)).getroot()
        except ET.ParseError as e:
            print(f"Warning: failed to parse {path}: {e}", file=sys.stderr)
            continue
        for elem in root.iter():
            if _tag(elem) != "w":
                continue
            wid = _nite_id(elem)
            if wid and elem.text and elem.text.strip():
                word_map[wid] = elem.text.strip()
    return word_map


# ──────────────────────────────────────────────
# Segments ロード & セグメントマップ構築
# ──────────────────────────────────────────────

def _resolve_word_range(href: str, word_map: dict[str, str]) -> list[str]:
    """
    nite:child href を解決して単語テキストのリストを返す。
    href 例: "Bmr003.A.words.xml#id(Bmr003.w.35)..id(Bmr003.w.36)"
    """
    raw_ids = re.findall(r"id\(([^)]+)\)", href)
    if not raw_ids:
        return []

    def _norm(raw: str) -> str:
        """NXT compound id: 'Bmr003.w.1,35' → 'Bmr003.w.35'"""
        if "," in raw:
            prefix, num = raw.rsplit(",", 1)
            base = re.sub(r"\.\d+$", "", prefix)
            return f"{base}.{num}"
        return raw

    if len(raw_ids) == 1:
        wid = _norm(raw_ids[0])
        return [word_map[wid]] if wid in word_map else []

    start_norm = _norm(raw_ids[0])
    end_norm = _norm(raw_ids[1])
    pm = re.match(r"^(.*\.)(\d+)$", start_norm)
    em = re.match(r"^(.*\.)(\d+)$", end_norm)
    if not pm or not em or pm.group(1) != em.group(1):
        result = []
        for wid in [start_norm, end_norm]:
            if wid in word_map:
                result.append(word_map[wid])
        return result

    prefix = pm.group(1)
    words = []
    for num in range(int(pm.group(2)), int(em.group(2)) + 1):
        wid = f"{prefix}{num}"
        if wid in word_map:
            words.append(word_map[wid])
    return words


def load_segment_map(
    meeting_id: str, word_map: dict[str, str]
) -> dict[str, dict]:
    """
    Segments/{meeting}.*.segs.xml を全スピーカー分読み込み、
    segment nite:id → {"id", "speaker", "start", "end", "text"} のマップを返す。
    """
    seg_map: dict[str, dict] = {}
    for seg_path in SEGS_DIR.glob(f"{meeting_id}.*.segs.xml"):
        try:
            root = ET.parse(str(seg_path)).getroot()
        except ET.ParseError as e:
            print(f"Warning: failed to parse {seg_path}: {e}", file=sys.stderr)
            continue
        for elem in root.iter():
            if _tag(elem) != "segment":
                continue
            seg_id = _nite_id(elem)
            if not seg_id:
                continue
            participant = elem.get("participant", "?")
            start = _safe_float(elem.get("starttime"))
            end = _safe_float(elem.get("endtime"))

            words: list[str] = []
            for child in elem:
                if _tag(child) == "child":
                    href = child.get("href", "")
                    words.extend(_resolve_word_range(href, word_map))

            text = " ".join(w for w in words if w)
            if text:
                seg_map[seg_id] = {
                    "id": seg_id,
                    "speaker": participant,
                    "start": start,
                    "end": end,
                    "text": text,
                }
    return seg_map


# ──────────────────────────────────────────────
# トランスクリプト
# ──────────────────────────────────────────────

def get_transcript(seg_map: dict[str, dict]) -> list[dict]:
    """
    セグメントマップから時刻順のトランスクリプトを生成する。
    Returns: [{"id", "speaker", "dialogue_act", "text", "turn", "argument_relations"}, ...]
    """
    from collections import defaultdict
    utterances = list(seg_map.values())
    utterances.sort(key=lambda u: (u["start"], u["end"]))

    turn_count: dict[str, int] = defaultdict(int)
    result = []
    for s in utterances:
        turn_count[s["speaker"]] += 1
        result.append({
            "id": s["id"],
            "speaker": s["speaker"],
            "dialogue_act": "",
            "text": s["text"],
            "turn": turn_count[s["speaker"]],
            "argument_relations": [],
        })
    return result


# ──────────────────────────────────────────────
# トピック抽出（引用付き）
# ──────────────────────────────────────────────

def _resolve_seg_ids_from_href(href: str, meeting_id: str) -> list[str]:
    """
    topic.xml の nite:child href からセグメント ID リストを返す。
    href 例:
      "Bmr003.C.segs.xml#id(Bmr003.segment.1,382)"
      "Bmr003.C.segs.xml#id(Bmr003.segment.1,382)..id(Bmr003.segment.1,394)"
    NXT compound id: 'Prefix.1,N' → '{meeting_id}.segment.N'
    """
    raw_ids = re.findall(r"id\(([^)]+)\)", href)
    if not raw_ids:
        return []

    def _to_seg_id(raw: str) -> str | None:
        if "," in raw:
            _, num = raw.rsplit(",", 1)
            return f"{meeting_id}.segment.{num}"
        # 通常の nite:id をそのまま使う
        return raw if raw.startswith(meeting_id) else None

    if len(raw_ids) == 1:
        sid = _to_seg_id(raw_ids[0])
        return [sid] if sid else []

    start = _to_seg_id(raw_ids[0])
    end = _to_seg_id(raw_ids[1])
    if not start or not end:
        return [s for s in [start, end] if s]

    sm = re.match(r"^(.*\.)(\d+)$", start)
    em = re.match(r"^(.*\.)(\d+)$", end)
    if not sm or not em or sm.group(1) != em.group(1):
        return [start, end]

    prefix = sm.group(1)
    return [f"{prefix}{n}" for n in range(int(sm.group(2)), int(em.group(2)) + 1)]


def get_topics(meeting_id: str, seg_map: dict[str, dict]) -> list[dict]:
    """
    TopicSegmentation/{meeting}.topic.xml からトピックを抽出する。
    各トピックに最大 MAX_TOPIC_QUOTE_SEGS 件の引用テキストを付与する。
    Returns: [{"id", "topic_type", "description", "quote"}, ...]
    """
    path = TOPICS_DIR / f"{meeting_id}.topic.xml"
    if not path.exists():
        return []

    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as e:
        print(f"Warning: failed to parse {path}: {e}", file=sys.stderr)
        return []

    topics: list[dict] = []

    def _walk(node: ET.Element) -> None:
        if _tag(node) == "topic":
            nite_id = _nite_id(node)
            desc = node.get("description", "")
            quote_parts: list[str] = []
            for child in node:
                if _tag(child) != "child":
                    continue
                if len(quote_parts) >= MAX_TOPIC_QUOTE_SEGS:
                    break
                href = child.get("href", "")
                for seg_id in _resolve_seg_ids_from_href(href, meeting_id):
                    seg = seg_map.get(seg_id)
                    if seg and len(quote_parts) < MAX_TOPIC_QUOTE_SEGS:
                        quote_parts.append(f"[{seg['speaker']}] {seg['text']}")
            topics.append({
                "id": nite_id,
                "topic_type": "",
                "description": desc,
                "quote": " ".join(quote_parts),
            })
            for child in node:
                _walk(child)
        else:
            for child in node:
                _walk(child)

    _walk(root)
    return topics


# ──────────────────────────────────────────────
# 要約抽出
# ──────────────────────────────────────────────

def get_summary(meeting_id: str) -> dict:
    """
    Summarization/abstractive/{meeting}.abssumm.xml から要約を取得する。
    Returns: AMI 互換形式
      {
        "abstractive": {"overall", "actions", "decisions", "problems"},
        "decisions": [],
        "participant_summaries": {},
      }
    """
    path = SUMM_DIR / f"{meeting_id}.abssumm.xml"
    abstractive = {"overall": "", "actions": "", "decisions": "", "problems": ""}

    if path.exists():
        with open(str(path), "rb") as f:
            content = f.read()
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            print(f"Warning: failed to parse {path}: {e}", file=sys.stderr)
            root = None

        if root is not None:
            # ICSI section → AMI key mapping
            section_map = {
                "abstract": "overall",
                "progress": "actions",
                "decisions": "decisions",
                "problems": "problems",
            }
            for section, ami_key in section_map.items():
                sentences: list[str] = []
                for elem in root.iter():
                    if _tag(elem) == section:
                        for sent in elem.iter():
                            if _tag(sent) == "sentence" and sent.text:
                                t = sent.text.strip()
                                if t:
                                    sentences.append(t)
                if sentences:
                    abstractive[ami_key] = " ".join(sentences)

    return {
        "abstractive": abstractive,
        "decisions": [],
        "participant_summaries": {},
    }


# ──────────────────────────────────────────────
# ミーティング処理エントリポイント
# ──────────────────────────────────────────────

def process_meeting(meeting_id: str) -> dict:
    """
    1 ミーティング分のレコードを構築して返す。
    Returns:
        {
            "meeting_id": str,
            "discussion": [{"id", "description", "depth", "quote"}, ...],
            "transcript": [{"speaker", "start", "end", "text"}, ...],
            "summary": {"abstract"?: str, "decisions"?: str,
                        "problems"?: str, "progress"?: str},
        }
    """
    word_map = load_words(meeting_id)
    seg_map = load_segment_map(meeting_id, word_map)
    transcript = get_transcript(seg_map)
    topics = get_topics(meeting_id, seg_map)
    summary = get_summary(meeting_id)

    print(
        f"✓ {meeting_id}: topics={len(topics)}, "
        f"transcript_turns={len(transcript)}"
    )

    return {
        "theme": None,
        "participants": [],
        "discussion": topics,
        "argument_units": {
            "discussions": [],
            "argument_units": [],
        },
        "transcript": transcript,
        "summary": summary,
        "missing_files": [],
    }
