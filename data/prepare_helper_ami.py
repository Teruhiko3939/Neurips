import fnmatch
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def strip_ns(tag: str) -> str:
	return tag.split("}", 1)[-1] if "}" in tag else tag


def get_attr_any_ns(elem: ET.Element, local_name: str) -> str | None:
	if local_name in elem.attrib:
		return elem.attrib[local_name]
	for k, v in elem.attrib.items():
		if k.endswith("}" + local_name):
			return v
	return None


def safe_parse_xml(path: str, silent_missing: bool = False) -> ET.Element | None:
	try:
		return ET.parse(path).getroot()
	except FileNotFoundError:
		if not silent_missing:
			print(f"Warning: Failed to parse {path}: file not found", file=sys.stderr)
		return None
	except Exception as e:
		print(f"Warning: Failed to parse {path}: {e}", file=sys.stderr)
		return None


def extract_id_range_from_href(href: str) -> tuple[str | None, str | None]:
	ids = re.findall(r"id\(([^)]+)\)", href)
	if not ids:
		return None, None
	if len(ids) == 1:
		return ids[0], ids[0]
	return ids[0], ids[1]


def extract_target_id_from_pointer(href: str) -> str | None:
	m = re.search(r"#id\(([^)]+)\)", href)
	return m.group(1) if m else None


def word_index(word_id: str) -> int:
	m = re.search(r"(\d+)$", word_id)
	return int(m.group(1)) if m else 10**12


def infer_speaker_from_filename(path: str) -> str:
	base = os.path.basename(path)
	parts = base.split(".")
	if len(parts) >= 3:
		return parts[1]
	return "unknown"


def load_ontology_map(xml_path: str) -> dict[str, dict[str, str]]:
	root = safe_parse_xml(xml_path)
	if root is None:
		return {}

	out: dict[str, dict[str, str]] = {}
	for elem in root.iter():
		nid = get_attr_any_ns(elem, "id")
		if nid:
			out[nid] = {
				"name": elem.attrib.get("name", ""),
				"gloss": elem.attrib.get("gloss", ""),
			}
	return out


def load_words_for_meeting(manual_ann_root: str, meeting_id: str):
	words_files = sorted(glob.glob(os.path.join(manual_ann_root, "words", f"{meeting_id}.*.words.xml")))

	speaker_words: dict[str, list[dict]] = defaultdict(list)
	id_to_word: dict[str, dict] = {}
	speaker_file_to_speaker: dict[str, str] = {}

	for wf in words_files:
		speaker = infer_speaker_from_filename(wf)
		speaker_file_to_speaker[os.path.basename(wf)] = speaker
		root = safe_parse_xml(wf)
		if root is None:
			continue

		for elem in root.iter():
			tag = strip_ns(elem.tag).lower()
			if tag != "w":
				continue

			wid = get_attr_any_ns(elem, "id")
			if not wid:
				continue

			token = (elem.text or "").strip()
			if not token:
				continue

			punc = elem.attrib.get("punc", "") == "true"
			st = elem.attrib.get("starttime", "")
			et = elem.attrib.get("endtime", "")
			try:
				stf = float(st)
			except Exception:
				stf = 1e18
			try:
				etf = float(et)
			except Exception:
				etf = stf

			w = {
				"id": wid,
				"speaker": speaker,
				"text": token,
				"punc": punc,
				"start": stf,
				"end": etf,
				"idx": word_index(wid),
			}
			speaker_words[speaker].append(w)
			id_to_word[wid] = w

	for spk in speaker_words:
		speaker_words[spk].sort(key=lambda x: (x["start"], x["idx"]))

	return speaker_words, id_to_word, speaker_file_to_speaker


def build_speaker_indices(speaker_words: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
	out = {}
	for spk, arr in speaker_words.items():
		out[spk] = {w["id"]: i for i, w in enumerate(arr)}
	return out


def render_tokens(tokens: list[dict], remove_word_ids: set[str] | None = None) -> str:
	parts: list[str] = []
	remove_word_ids = remove_word_ids or set()
	for t in tokens:
		if t["id"] in remove_word_ids:
			continue
		if t["punc"] and parts:
			parts[-1] = parts[-1] + t["text"]
		else:
			parts.append(t["text"])
	return " ".join(parts).strip()


def tokens_from_word_range(
	speaker: str,
	start_id: str,
	end_id: str,
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
) -> list[dict]:
	arr = speaker_words.get(speaker, [])
	idx_map = speaker_id_to_pos.get(speaker, {})
	if start_id not in idx_map or end_id not in idx_map:
		return []
	a = idx_map[start_id]
	b = idx_map[end_id]
	if a > b:
		a, b = b, a
	return arr[a : b + 1]


def load_disfluency_word_ids(
	manual_ann_root: str,
	meeting_id: str,
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
) -> dict[str, set[str]]:
	removable_dsfl_types = {
		"ami_dsfl_3",
		"ami_dsfl_4",
		"ami_dsfl_5",
		"ami_dsfl_12",
		"ami_dsfl_14",
		"ami_dsfl_16",
		"ami_dsfl_17",
		"ami_dsfl_19",
	}

	disfluency_files = sorted(glob.glob(os.path.join(manual_ann_root, "disfluency", f"{meeting_id}.*.disfluency.xml")))
	out: dict[str, set[str]] = defaultdict(set)

	for path in disfluency_files:
		root = safe_parse_xml(path)
		if root is None:
			continue

		for dsfl in root.iter():
			if strip_ns(dsfl.tag) != "dsfl":
				continue

			dsfl_type_id = ""
			child_hrefs: list[str] = []
			for ch in dsfl:
				tag = strip_ns(ch.tag)
				if tag == "pointer" and ch.attrib.get("role") == "dsfl-type":
					dsfl_type_id = extract_target_id_from_pointer(ch.attrib.get("href", "")) or ""
				elif tag == "child":
					child_hrefs.append(ch.attrib.get("href", ""))

			if dsfl_type_id not in removable_dsfl_types:
				continue

			for href in child_hrefs:
				file_part = href.split("#", 1)[0]
				speaker = infer_speaker_from_filename(file_part)
				sid, eid = extract_id_range_from_href(href)
				if not sid or not eid:
					continue
				toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
				for t in toks:
					out[speaker].add(t["id"])

	return out


def build_segment_text_map(
	manual_ann_root: str,
	meeting_id: str,
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
	disfluency_word_ids: dict[str, set[str]],
) -> tuple[
	dict[str, str],
	dict[str, list[str]],
	dict[str, dict[str, int]],
	dict[str, list[dict]],
]:
	segment_files = sorted(glob.glob(os.path.join(manual_ann_root, "segments", f"{meeting_id}.*.segments.xml")))
	segment_text_map: dict[str, str] = {}
	speaker_segment_ids: dict[str, list[str]] = defaultdict(list)
	speaker_segment_bounds: dict[str, list[dict]] = defaultdict(list)

	for path in segment_files:
		speaker = infer_speaker_from_filename(path)
		root = safe_parse_xml(path)
		if root is None:
			continue

		rows: list[tuple[float, float, str, str, int, int]] = []
		for seg in root.iter():
			if strip_ns(seg.tag) != "segment":
				continue

			seg_id = get_attr_any_ns(seg, "id") or ""
			start = seg.attrib.get("transcriber_start", "")
			end = seg.attrib.get("transcriber_end", "")
			try:
				stf = float(start)
			except Exception:
				stf = 1e18
			try:
				etf = float(end)
			except Exception:
				etf = stf

			sid = None
			eid = None
			for ch in seg:
				if strip_ns(ch.tag) == "child":
					sid, eid = extract_id_range_from_href(ch.attrib.get("href", ""))
					break

			if not sid or not eid:
				continue

			pos_map = speaker_id_to_pos.get(speaker, {})
			if sid not in pos_map or eid not in pos_map:
				continue

			a = pos_map[sid]
			b = pos_map[eid]
			if a > b:
				a, b = b, a

			toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
			text = render_tokens(toks, disfluency_word_ids.get(speaker, set()))
			if not text:
				continue

			rows.append((stf, etf, seg_id, text, a, b))

		rows.sort(key=lambda x: (x[0], x[1], x[2]))
		for _start, _end, seg_id, text, start_pos, end_pos in rows:
			segment_text_map[seg_id] = text
			speaker_segment_ids[speaker].append(seg_id)
			speaker_segment_bounds[speaker].append(
				{
					"segment_id": seg_id,
					"start_pos": start_pos,
					"end_pos": end_pos,
				}
			)

	speaker_segment_pos = {
		speaker: {seg_id: index for index, seg_id in enumerate(seg_ids)}
		for speaker, seg_ids in speaker_segment_ids.items()
	}
	return segment_text_map, speaker_segment_ids, speaker_segment_pos, speaker_segment_bounds


def segment_ids_from_word_range(
	speaker: str,
	start_id: str,
	end_id: str,
	speaker_id_to_pos: dict[str, dict[str, int]],
	speaker_segment_bounds: dict[str, list[dict]],
) -> list[str]:
	pos_map = speaker_id_to_pos.get(speaker, {})
	if start_id not in pos_map or end_id not in pos_map:
		return []

	a = pos_map[start_id]
	b = pos_map[end_id]
	if a > b:
		a, b = b, a

	out: list[str] = []
	for row in speaker_segment_bounds.get(speaker, []):
		if row["end_pos"] < a or row["start_pos"] > b:
			continue
		out.append(row["segment_id"])
	return out


def segment_texts_from_href(
	href: str,
	segment_text_map: dict[str, str],
	speaker_segment_ids: dict[str, list[str]],
	speaker_segment_pos: dict[str, dict[str, int]],
) -> list[str]:
	file_part = href.split("#", 1)[0]
	speaker = infer_speaker_from_filename(file_part)
	start_id, end_id = extract_id_range_from_href(href)
	if not start_id or not end_id:
		return []

	pos_map = speaker_segment_pos.get(speaker, {})
	if start_id not in pos_map or end_id not in pos_map:
		if start_id in segment_text_map:
			return [segment_text_map[start_id]]
		return []

	a = pos_map[start_id]
	b = pos_map[end_id]
	if a > b:
		a, b = b, a

	out: list[str] = []
	for seg_id in speaker_segment_ids.get(speaker, [])[a : b + 1]:
		text = segment_text_map.get(seg_id, "")
		if text:
			out.append(f"[{speaker}] {text}")
	return out


def parse_participants(manual_ann_root: str, meeting_id: str) -> list[dict]:
	path = os.path.join(manual_ann_root, "corpusResources", "meetings.xml")
	root = safe_parse_xml(path)
	if root is None:
		return []

	participants: list[dict] = []
	for meeting in root.iter():
		if strip_ns(meeting.tag) != "meeting":
			continue
		if meeting.attrib.get("observation", "") != meeting_id:
			continue

		for sp in meeting:
			if strip_ns(sp.tag) != "speaker":
				continue
			participants.append(
				{
					"speaker": sp.attrib.get("nxt_agent", ""),
					"name": sp.attrib.get("global_name", ""),
					"role": sp.attrib.get("role", ""),
				}
			)
		break

	participants.sort(key=lambda x: x["speaker"])
	return participants


def parse_topics(
	manual_ann_root: str,
	meeting_id: str,
	topic_map: dict[str, dict[str, str]],
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
	disfluency_word_ids: dict[str, set[str]],
):
	path = os.path.join(manual_ann_root, "topics", f"{meeting_id}.topic.xml")
	root = safe_parse_xml(path, silent_missing=True)
	if root is None:
		return []

	results = []
	for topic in root.iter():
		if strip_ns(topic.tag) != "topic":
			continue

		topic_id = get_attr_any_ns(topic, "id") or ""
		description = topic.attrib.get("other_description", "")
		topic_label = ""

		for ch in topic:
			if strip_ns(ch.tag) == "pointer" and ch.attrib.get("role") == "scenario_topic_type":
				tid = extract_target_id_from_pointer(ch.attrib.get("href", ""))
				if tid and tid in topic_map:
					topic_label = topic_map[tid].get("name", "") or topic_map[tid].get("gloss", "")

		quote_chunks: list[str] = []
		for ch in topic:
			if strip_ns(ch.tag) != "child":
				continue
			href = ch.attrib.get("href", "")
			file_part = href.split("#", 1)[0]
			speaker = infer_speaker_from_filename(file_part)
			sid, eid = extract_id_range_from_href(href)
			if not sid or not eid:
				continue
			toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
			txt = render_tokens(toks, disfluency_word_ids.get(speaker, set()))
			if txt:
				quote_chunks.append(f"[{speaker}] {txt}")
			if len(quote_chunks) >= 6:
				break

		results.append(
			{
				"id": topic_id,
				"topic_type": topic_label,
				"description": description,
				"quote": " ".join(quote_chunks),
			}
		)

	return results


def parse_argumentation(
	manual_ann_root: str,
	meeting_id: str,
	ae_map: dict[str, dict[str, str]],
	ar_map: dict[str, dict[str, str]],
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
	disfluency_word_ids: dict[str, set[str]],
	segment_text_map: dict[str, str],
	speaker_segment_ids: dict[str, list[str]],
	speaker_segment_pos: dict[str, dict[str, int]],
	speaker_segment_bounds: dict[str, list[dict]],
):
	base = os.path.join(manual_ann_root, "argumentation")
	dis_path = os.path.join(base, "dis", f"{meeting_id}.discussions.xml")
	ar_path = os.path.join(base, "ar", f"{meeting_id}.argumentationrels.xml")
	ae_paths = sorted(glob.glob(os.path.join(base, "ae", f"{meeting_id}.*.argumentstructs.xml")))

	arg_units: dict[str, dict] = {}
	for ap in ae_paths:
		speaker = infer_speaker_from_filename(ap)
		root = safe_parse_xml(ap)
		if root is None:
			continue

		for ae in root.iter():
			if strip_ns(ae.tag) != "ae":
				continue
			ae_id = get_attr_any_ns(ae, "id") or ""
			ae_type_id = ""
			ae_type = ""
			quote = ""
			segment_ids: list[str] = []

			for ch in ae:
				tag = strip_ns(ch.tag)
				if tag == "pointer" and ch.attrib.get("role") == "type":
					tid = extract_target_id_from_pointer(ch.attrib.get("href", "")) or ""
					ae_type_id = tid
					if tid in ae_map:
						ae_type = ae_map[tid].get("gloss", "") or ae_map[tid].get("name", "")
				elif tag == "child":
					sid, eid = extract_id_range_from_href(ch.attrib.get("href", ""))
					if sid and eid:
						toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
						quote = render_tokens(toks, disfluency_word_ids.get(speaker, set()))
						segment_ids = segment_ids_from_word_range(
							speaker,
							sid,
							eid,
							speaker_id_to_pos,
							speaker_segment_bounds,
						)

			arg_units[ae_id] = {
				"id": ae_id,
				"speaker": speaker,
				"type": ae_type,
				"type_id": ae_type_id,
				"quote": quote,
				"segment_ids": segment_ids,
			}

			if ae_type_id == "ae_6":
				arg_units.pop(ae_id, None)

	relations = []
	ar_root = safe_parse_xml(ar_path, silent_missing=True)
	if ar_root is not None:
		for ar in ar_root.iter():
			if strip_ns(ar.tag) != "ar":
				continue
			rid = get_attr_any_ns(ar, "id") or ""
			src = ""
			tgt = ""
			rel_type = ""

			for ch in ar:
				if strip_ns(ch.tag) != "pointer":
					continue
				role = ch.attrib.get("role", "")
				href = ch.attrib.get("href", "")
				tid = extract_target_id_from_pointer(href) or ""
				if role == "source":
					src = tid
				elif role == "target":
					tgt = tid
				elif role == "type" and tid in ar_map:
					rel_type = ar_map[tid].get("gloss", "") or ar_map[tid].get("name", "")

			relations.append(
				{
					"id": rid,
					"relation_type": rel_type,
					"source_id": src,
					"target_id": tgt,
				}
			)

	discussions = []
	dis_root = safe_parse_xml(dis_path, silent_missing=True)
	if dis_root is not None:
		for frag in dis_root.iter():
			if strip_ns(frag.tag) != "discussion-fragment":
				continue
			fid = get_attr_any_ns(frag, "id") or ""
			name = frag.attrib.get("name", "")
			text_chunks: list[str] = []
			for ch in frag:
				if strip_ns(ch.tag) == "child":
					text_chunks.extend(
						segment_texts_from_href(
							ch.attrib.get("href", ""),
							segment_text_map,
							speaker_segment_ids,
							speaker_segment_pos,
						)
					)
			discussions.append({"id": fid, "name": name, "text": " ".join(text_chunks)})

	return {
		"discussions": discussions,
		"argument_units": list(arg_units.values()),
		"argument_relations": relations,
	}


def integrate_argument_relations_into_transcript(transcript: list[dict], arg: dict) -> None:
	transcript_by_segment_id = {row["id"]: row for row in transcript}
	units_by_id = {row["id"]: row for row in arg.get("argument_units", [])}

	for row in transcript:
		row["argument_relations"] = []

	for rel in arg.get("argument_relations", []):
		source_unit = units_by_id.get(rel.get("source_id", ""))
		target_unit = units_by_id.get(rel.get("target_id", ""))
		if not source_unit or not target_unit:
			continue

		source_segment_ids = source_unit.get("segment_ids", [])
		target_segment_ids = target_unit.get("segment_ids", [])
		if not source_segment_ids or not target_segment_ids:
			continue

		for source_segment_id in source_segment_ids:
			source_row = transcript_by_segment_id.get(source_segment_id)
			if source_row is None:
				continue
			for target_segment_id in target_segment_ids:
				target_row = transcript_by_segment_id.get(target_segment_id)
				if target_row is None:
					continue
				source_row["argument_relations"].append(
					{
						"role": "source",
						"relation_type": rel.get("relation_type", ""),
						"related_speaker": target_row.get("speaker", ""),
						"related_turn": target_row.get("turn"),
					}
				)
				target_row["argument_relations"].append(
					{
						"role": "target",
						"relation_type": rel.get("relation_type", ""),
						"related_speaker": source_row.get("speaker", ""),
						"related_turn": source_row.get("turn"),
					}
				)

	for unit in arg.get("argument_units", []):
		unit.pop("segment_ids", None)

	arg.pop("argument_relations", None)


def parse_dialogue_acts(
	manual_ann_root: str,
	meeting_id: str,
	da_map: dict[str, dict[str, str]],
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
	disfluency_word_ids: dict[str, set[str]],
):
	da_files = sorted(glob.glob(os.path.join(manual_ann_root, "dialogueActs", f"{meeting_id}.*.dialog-act.xml")))
	by_speaker: dict[str, list[dict]] = defaultdict(list)

	for dp in da_files:
		speaker = infer_speaker_from_filename(dp)
		root = safe_parse_xml(dp)
		if root is None:
			continue

		for dact in root.iter():
			if strip_ns(dact.tag) != "dact":
				continue

			dact_id = get_attr_any_ns(dact, "id") or ""
			da_type_id = ""
			da_label = ""
			sid = None
			eid = None

			for ch in dact:
				tag = strip_ns(ch.tag)
				if tag == "pointer" and ch.attrib.get("role") == "da-aspect":
					tid = extract_target_id_from_pointer(ch.attrib.get("href", "")) or ""
					da_type_id = tid
					if tid in da_map:
						da_label = da_map[tid].get("gloss", "") or da_map[tid].get("name", "")
				elif tag == "child":
					sid, eid = extract_id_range_from_href(ch.attrib.get("href", ""))

			if not sid or not eid:
				continue

			toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
			if not toks:
				continue

			by_speaker[speaker].append(
				{
					"dact_id": dact_id,
					"da_type": da_label,
					"da_type_id": da_type_id,
					"start": toks[0]["start"],
					"end": toks[-1]["end"],
					"text": render_tokens(toks, disfluency_word_ids.get(speaker, set())),
				}
			)

	for spk in by_speaker:
		by_speaker[spk].sort(key=lambda x: (x["start"], x["end"]))

	return by_speaker


def parse_transcript(
	manual_ann_root: str,
	meeting_id: str,
	da_map: dict[str, dict[str, str]],
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
	disfluency_word_ids: dict[str, set[str]],
):
	da_by_speaker = parse_dialogue_acts(
		manual_ann_root,
		meeting_id,
		da_map,
		speaker_words,
		speaker_id_to_pos,
		disfluency_word_ids,
	)

	seg_files = sorted(glob.glob(os.path.join(manual_ann_root, "segments", f"{meeting_id}.*.segments.xml")))

	transcript: list[dict] = []
	turn_count: dict[str, int] = defaultdict(int)

	for sp in seg_files:
		speaker = infer_speaker_from_filename(sp)
		root = safe_parse_xml(sp)
		if root is None:
			continue

		seg_rows = []
		for seg in root.iter():
			if strip_ns(seg.tag) != "segment":
				continue

			seg_id = get_attr_any_ns(seg, "id") or ""
			start = seg.attrib.get("transcriber_start", "")
			end = seg.attrib.get("transcriber_end", "")
			try:
				stf = float(start)
			except Exception:
				stf = 1e18
			try:
				etf = float(end)
			except Exception:
				etf = stf

			sid = None
			eid = None
			for ch in seg:
				if strip_ns(ch.tag) == "child":
					sid, eid = extract_id_range_from_href(ch.attrib.get("href", ""))
					break

			if not sid or not eid:
				continue

			toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
			text = render_tokens(toks, disfluency_word_ids.get(speaker, set()))
			if not text:
				continue

			da_label = ""
			best_overlap = -1.0
			for da in da_by_speaker.get(speaker, []):
				ov = min(etf, da["end"]) - max(stf, da["start"])
				if ov > best_overlap:
					best_overlap = ov
					da_label = da["da_type"]

			seg_rows.append(
				{
					"id": seg_id,
					"speaker": speaker,
					"start": stf,
					"end": etf,
					"dialogue_act": da_label,
					"text": text,
				}
			)

		seg_rows.sort(key=lambda x: (x["start"], x["end"]))
		for row in seg_rows:
			turn_count[speaker] += 1
			row["turn"] = turn_count[speaker]
			out_row = {
				"id": row["id"],
				"speaker": row["speaker"],
				"dialogue_act": row["dialogue_act"],
				"text": row["text"],
				"turn": row["turn"],
				"_start": row["start"],
				"_end": row["end"],
			}
			transcript.append(out_row)

	transcript.sort(key=lambda x: (x["_start"], x["_end"], x["speaker"], x["turn"]))
	for row in transcript:
		row.pop("_start", None)
		row.pop("_end", None)
	return transcript


def collect_sentences_under(node: ET.Element, sent_tags: tuple[str, ...]) -> list[str]:
	out = []
	for e in node.iter():
		if strip_ns(e.tag) in sent_tags:
			txt = " ".join((e.text or "").split())
			if txt:
				out.append(txt)
	return out


def parse_summary(
	manual_ann_root: str,
	meeting_id: str,
	speaker_words: dict[str, list[dict]],
	speaker_id_to_pos: dict[str, dict[str, int]],
	disfluency_word_ids: dict[str, set[str]],
):
	abstractive_path = os.path.join(manual_ann_root, "abstractive", f"{meeting_id}.abssumm.xml")
	abstractive = {
		"overall": "",
		"actions": "",
		"decisions": "",
		"problems": "",
	}
	abs_root = safe_parse_xml(abstractive_path, silent_missing=True)
	if abs_root is not None:
		for child in abs_root:
			tag = strip_ns(child.tag)
			sents = collect_sentences_under(child, ("sentence", "sent"))
			if not sents:
				continue
			para = " ".join(sents)
			if tag == "abstract":
				abstractive["overall"] = para
			elif tag == "actions":
				abstractive["actions"] = para
			elif tag == "decisions":
				abstractive["decisions"] = para
			elif tag == "problems":
				abstractive["problems"] = para

	decision_path = os.path.join(manual_ann_root, "decision", "manual", f"{meeting_id}.decision.xml")
	decisions = []
	dec_root = safe_parse_xml(decision_path, silent_missing=True)
	if dec_root is not None:
		for d in dec_root.iter():
			if strip_ns(d.tag) != "decision":
				continue
			did = get_attr_any_ns(d, "id") or ""
			text_chunks: list[str] = []
			for ch in d:
				if strip_ns(ch.tag) != "child":
					continue
				href = ch.attrib.get("href", "")
				file_part = href.split("#", 1)[0]
				speaker = infer_speaker_from_filename(file_part)
				sid, eid = extract_id_range_from_href(href)
				if not sid or not eid:
					continue
				toks = tokens_from_word_range(speaker, sid, eid, speaker_words, speaker_id_to_pos)
				text = render_tokens(toks, disfluency_word_ids.get(speaker, set()))
				if text:
					text_chunks.append(f"[{speaker}] {text}")

			decisions.append(
				{
					"id": did,
					"text": " ".join(text_chunks),
				}
			)

	part_paths = sorted(glob.glob(os.path.join(manual_ann_root, "participantSummaries", f"{meeting_id}.*.summ.xml")))
	participant_summaries = {}
	for pp in part_paths:
		speaker = infer_speaker_from_filename(pp)
		root = safe_parse_xml(pp)
		if root is None:
			continue
		sections = {}
		for sec in root:
			sec_tag = strip_ns(sec.tag)
			sents = collect_sentences_under(sec, ("sentence", "sent"))
			if sents:
				sections[sec_tag] = " ".join(sents)
		participant_summaries[speaker] = sections

	return {
		"abstractive": abstractive,
		"decisions": decisions,
		"participant_summaries": participant_summaries,
	}


def build_meeting_list(manual_ann_root: str, meeting_ids: list[str] | None, pattern: str) -> list[str]:
	if meeting_ids:
		return meeting_ids

	words_files = glob.glob(os.path.join(manual_ann_root, "words", "*.words.xml"))
	mids = set()
	for wf in words_files:
		base = os.path.basename(wf)
		m = re.match(r"^([^.]+)\.[^.]+\.words\.xml$", base)
		if m:
			mids.add(m.group(1))

	return sorted(mid for mid in mids if fnmatch.fnmatch(mid, pattern))
