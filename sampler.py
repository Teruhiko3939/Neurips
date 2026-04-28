from datetime import datetime
import os
import pickle
import json
import re
import sys
import traceback
from pathlib import Path
from time import perf_counter

import openai

from agent.flow import FlowRunner
from const.consts import AGENT_LLM_MODEL
from evaluation.evaluation import Sampler
from evaluation.evaluation_RAGAS import Sampler_RAGAS
from evaluation.evaluation_RAGAS import Sampler_RAGAS
from models.llm import LLM
from run_flow import phase_discussion, phase_prompt, phase_qa
from run_single import run_once
from utils.app_write_message import set_streamlit_mode
from utils.run_helper import append_jsonl, run_with_retry, runtime_summary_record

PKL_PATH = Path("data/dataset/icsi.pkl")
SINGLE_SUBDIRS = ["1", "2", "3"]
EVAL_SOURCE_DIRS = ("single_4", "single_5", "multi_af", "multi_noaf")


def _result_subdir_from_flags(multi_flag: bool, af_using_flag: bool) -> str:
	if not multi_flag:
		return "single_5" if af_using_flag else "single_4"
	return "multi_af" if af_using_flag else "multi_noaf"


def _load_meeting(meeting_id: str) -> dict:
	"""Load and return the record for the specified meeting from PKL once."""
	data = pickle.loads(PKL_PATH.read_bytes())
	if meeting_id not in data:
		print(f"meeting not found: {meeting_id}")
		print(f"available meetings (head): {list(data.keys())[:10]}")
		sys.exit(1)
	return data[meeting_id]

def _parse_jsonl(filepath: str) -> list[dict]:
	"""Parse all records from JSONL, including multi-line JSON entries."""
	text = Path(filepath).read_text(encoding="utf-8")
	records = []
	for part in re.split(r'\n(?=\{)', text.strip()):
		part = part.strip()
		if part:
			try:
				records.append(json.loads(part, strict=False))
			except json.JSONDecodeError:
				pass
	return records


def read_jsonl_by_index(filepath: str, index: int) -> dict | None:
	return next((r for r in _parse_jsonl(filepath) if r.get("index") == index), None)


def read_jsonl_last_record(filepath: str) -> dict | None:
	"""Return the record with the maximum index field."""
	records = [r for r in _parse_jsonl(filepath) if "index" in r]
	return max(records, key=lambda r: r["index"]) if records else None


def read_jsonl_first_content(filepath: str) -> str | None:
	return next((r["content"] for r in _parse_jsonl(filepath) if "content" in r), None)


def _meeting_id_from_filename(filename: str) -> str:
	match = re.match(r"^(.*?)_\d{8}_\d{6}\.jsonl$", filename)
	return match.group(1) if match else Path(filename).stem


def _load_content_from_result_file(result_subdir: str, filepath: str) -> str | None:
	if result_subdir in ("single_4", "single_5"):
		return read_jsonl_first_content(filepath)
	record = read_jsonl_last_record(filepath)
	return record.get("content") if record else None


def _evaluate_and_save(
	meeting_id: str,
	content: str,
	sampler: Sampler,
	results_dir: str,
	result_subdir: str,
	source_file: str | None = None,
	sub_subdir: str | None = None,
) -> None:
	meeting = _load_meeting(meeting_id)
	input_query = json.dumps(meeting["theme"], ensure_ascii=False, indent=2)
	reference = json.dumps(meeting["summary"]["abstractive"], ensure_ascii=False, indent=2)

	input_query = sampler.clean_query(input_query)
	content = sampler.clean_answer(content)
	reference = sampler.clean_query(reference)

	eval_started = perf_counter()
	sampler_label = "RAGAS" if isinstance(sampler, Sampler_RAGAS) else "Eval"
	print(f"  [{sampler_label}] {meeting_id}" + (f"/{sub_subdir}" if sub_subdir else ""))
	if isinstance(sampler, Sampler_RAGAS):
		score_in_out  = sampler.evaluate_in_out(input_query, content, reference)
		# score_ref_out = sampler.evaluate_ref_out(input_query, content, reference)
		eval_elapsed = perf_counter() - eval_started
		result_record = {
			"meeting_id": meeting_id,
			"input_query": input_query,
			"content": content,
			"reference": reference,
			"score_ragas_in_out": score_in_out,
			# "score_ragas_ref_out": score_ref_out,
			"eval_elapsed_seconds": eval_elapsed,
		}
		file_prefix = "eval_RAGAS"
	else:
		score_in_out  = sampler.evaluate_in_out(input_query, content, reference)
		score_ref_out = sampler.evaluate_ref_out(input_query, content, reference)
		score_div   = sampler.evaluate_arg_div(input_query, content)
		score_opp   = sampler.evaluate_opp(input_query, content)
		# score_faith = sampler.evaluate_faith(input_query, content)
		eval_elapsed = perf_counter() - eval_started
		result_record = {
			"meeting_id": meeting_id,
			"input_query": input_query,
			"content": content,
			"reference": reference,
			"score_in_out": score_in_out,
			"score_ref_out": score_ref_out,
			"score_arg_div": score_div,
			"score_opp": score_opp,
			# "score_faith": score_faith,
			"eval_elapsed_seconds": eval_elapsed,
		}
		file_prefix = "eval"

	result_path = os.path.join(
		results_dir,
		result_subdir,
		*([sub_subdir] if sub_subdir else []),
		f"{file_prefix}_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
	)

	os.makedirs(os.path.dirname(result_path), exist_ok=True)
	if source_file is not None:
		result_record["source_file"] = source_file
	append_jsonl(result_path, result_record)


def reevaluate_existing_outputs(sampler: Sampler, results_dir: str, result_subdir: str) -> None:
	if result_subdir not in EVAL_SOURCE_DIRS:
		raise ValueError(f"unknown result_subdir: {result_subdir}")

	dir_path = os.path.join(results_dir, result_subdir)
	if not os.path.isdir(dir_path):
		print(f"[WARN] Directory does not exist: {dir_path}")
		return

	file_entries = []
	# direct files under result_subdir/
	for name in sorted(os.listdir(dir_path)):
		if name.endswith(".jsonl") and not name.startswith("eval_"):
			file_entries.append((os.path.join(dir_path, name), name, None))
	# files under numbered subdirs
	for subdir in SINGLE_SUBDIRS:
		subdir_path = os.path.join(dir_path, subdir)
		if not os.path.isdir(subdir_path):
			os.makedirs(subdir_path, exist_ok=True)
		for name in sorted(os.listdir(subdir_path)):
			if name.endswith(".jsonl") and not name.startswith("eval_"):
				file_entries.append((os.path.join(subdir_path, name), name, subdir))

	print(f"\n[{result_subdir}] {len(file_entries)} files")

	for i, (file_path, file_name, subdir) in enumerate(file_entries, 1):
		meeting_id = _meeting_id_from_filename(file_name)
		label = f"{result_subdir}/{subdir}" if subdir else result_subdir
		print(f"[{label} {i}/{len(file_entries)}] {file_name}")
		try:
			content = _load_content_from_result_file(result_subdir, file_path)
			if not content:
				raise ValueError(f"content not found in {file_name}")
			_evaluate_and_save(meeting_id, content, sampler, results_dir, result_subdir, source_file=file_name, sub_subdir=subdir)
		except Exception as e:
			print(f"[ERROR] Exception while re-evaluating source_file={file_name}: {e}")
			_record_meeting_error(results_dir, meeting_id)

def _record_meeting_error(results_dir: str, meeting_id: str) -> None:
	"""Append failed meeting_id and exception details to results."""
	error_path = os.path.join(results_dir, "error_meetings.jsonl")
	append_jsonl(error_path, {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"meeting_id": meeting_id,
		"traceback": traceback.format_exc(),
	})
	print(f"[ERROR] Recorded meeting_id={meeting_id}: {error_path}")

def run_meeting(meeting_id: str, samplers: list[Sampler], results_dir: str, multi_flag: bool, af_using_flag: bool, file_name: str | None, skip_eval: bool = False, sub_subdir: str | None = None) -> None:
	meeting     = _load_meeting(meeting_id)
	input_query = json.dumps(meeting["theme"], ensure_ascii=False, indent=2)
	reference   = json.dumps(meeting["summary"]["abstractive"], ensure_ascii=False, indent=2)

	af_tag = "af" if af_using_flag else "noaf"

	if multi_flag:
		if file_name is None:
			runner = FlowRunner()
			runner._checkpoint_file = os.path.join(
				results_dir, f"multi_{af_tag}",
				*([sub_subdir] if sub_subdir else []),
				f"{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
			)
			os.makedirs(os.path.dirname(runner._checkpoint_file), exist_ok=True)

			agenda, af = phase_prompt(runner, agenda_text=input_query, confirm_text="yes", use_af=af_using_flag)
			phase_discussion(runner, agenda, af, proceed="yes")
			phase_qa(runner, question_text=None, proceed="yes")

			if not skip_eval:
				report = runner.get_runtime_report()
				append_jsonl(runner.checkpoint_file, runtime_summary_record(
					total_elapsed_seconds=report["total_elapsed_seconds"],
					total_log_seconds=report["total_log_seconds"],
					total_net_seconds=report["total_net_seconds"],
					extra={"steps": report["steps"]},
				))
				if report["steps"]:
					for name, stats in report["steps"].items():
						print(f"{name}: count={stats['count']}, net={stats['net_seconds']:.3f} sec, log={stats['log_seconds']:.3f} sec")
					append_jsonl(runner.checkpoint_file, {"type": "step_details", "steps": report["steps"]})

			record = read_jsonl_last_record(runner.checkpoint_file)
		else:
			record = read_jsonl_last_record(os.path.join(results_dir, f"multi_{af_tag}", file_name))

		content = record.get("content")

	else:
		if file_name is None:
			model = LLM(vendor_name=AGENT_LLM_MODEL, temperature=0.8).getModel()
			_result_subdir = _result_subdir_from_flags(multi_flag, af_using_flag)
			checkpoint_file = os.path.join(
				results_dir, _result_subdir,
				*([sub_subdir] if sub_subdir else []),
				f"{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
			)
			os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

			query = f'''Based on the following "meeting information", conduct a discussion from the perspectives of multiple participants and create meeting minutes.
            \n\n
			## Meeting Information\n
			{input_query}
            \n\n
			## Instructions\n
			- Let each participant speak from their position described in "conditions" and develop the discussion.
			- Keep the discussion aligned with the purpose in "description" and derive a conclusion.
			- Output the minutes in English using the structure below.
            \n\n
			## Minutes Output Format\n
			- **Meeting Name**: (from `name`)
			- **Attendees**: (extracted from `conditions`)
			- **Agenda**: (from `description`)
			- **Main Statements (A)**: bullet points of each participant's claims
			- **Adopted Statements (A*)**: final agreed/adopted statements
			- **Conclusion (C)**: conclusion reached in the meeting
			- **Summary (S)**: short summary of the whole meeting
			- **Rationale for Conclusion**: concise reasons supporting the conclusion
			'''

			started = perf_counter()
			response = run_with_retry("Single-agent response generation", lambda: run_once(model, query))
			elapsed = perf_counter() - started

			append_jsonl(checkpoint_file, {
				"timestamp": datetime.now().isoformat(timespec="seconds"),
				"input": query,
				"content": response,
				"elapsed_seconds": elapsed,
			})
		else:
			response = read_jsonl_first_content(os.path.join(results_dir, file_name))

		content = response

	if not skip_eval:
		result_subdir = _result_subdir_from_flags(multi_flag, af_using_flag)
		for sampler in samplers:
			_evaluate_and_save(meeting_id, content, sampler, results_dir, result_subdir, source_file=file_name, sub_subdir=sub_subdir)
	else:
		print(f"[INFO] Skipped evaluation for {meeting_id}")

def main(
    multi_agent: bool = True,
    use_af: bool = True,
    reevaluate_only: bool = False,
    skip_eval: bool = False,
	RAGAS_eval: bool = False,
    source_file: str | None = None,
    meeting_id: str | None = None,
) -> None:
	"""Run meeting processing or re-evaluation.

	Args:
		multi_agent: Use multi-agent mode (vs single-agent).
		use_af: Generate and use argumentation framework in phase 1.
		reevaluate_only: Re-evaluate existing JSONL files without running new meetings.
		skip_eval: Skip evaluation and save only output (no metrics).
		source_file: JSONL file name to load results from (None = execute new run).
		meeting_id: Specific meeting ID to process (None = all meetings).
	"""
	set_streamlit_mode(False)
	results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
	os.makedirs(results_dir, exist_ok=True)

	if reevaluate_only and RAGAS_eval:
		samplers = [Sampler(), Sampler_RAGAS()]
	elif RAGAS_eval:
		samplers = [Sampler_RAGAS()]
	else:
		samplers = [Sampler()]

	if reevaluate_only:
		result_subdir = _result_subdir_from_flags(multi_agent, use_af)
		for s in samplers:
			reevaluate_existing_outputs(s, results_dir, result_subdir)
		return

	if meeting_id:
		meeting_ids = [meeting_id]
	else:
		meeting_ids = list(pickle.loads(PKL_PATH.read_bytes()).keys())

	for i, mid in enumerate(meeting_ids, 1):
		print(f"\n{'='*50}")
		print(f"[{i}/{len(meeting_ids)}] {mid}")
		for sub in SINGLE_SUBDIRS:
			result_subdir = _result_subdir_from_flags(multi_agent, use_af)
			if meeting_id is None:
				check_dirs = [
					os.path.join(results_dir, result_subdir, sub),
					os.path.join(results_dir, result_subdir + "_old", sub),
				]
				if any(
					os.path.isdir(d) and any(
						f.startswith(mid + "_") and f.endswith(".jsonl") and not f.startswith("eval_")
						for f in os.listdir(d)
					)
					for d in check_dirs
				):
					print(f"[SKIP] Already exists: {mid} in {result_subdir}[_old]/{sub}/")
					continue
			try:
				run_meeting(mid, samplers, results_dir, multi_agent, use_af, source_file, skip_eval=skip_eval, sub_subdir=sub)
			except openai.BadRequestError as e:
				print(f"[SKIP] Content filter triggered for meeting_id={mid} sub={sub}: {e}")
				_record_meeting_error(results_dir, mid)
			except Exception as e:
				print(f"[ERROR] Exception while processing meeting_id={mid} sub={sub}: {e}")
				_record_meeting_error(results_dir, mid)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run MAGI meeting processing and evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # All meetings, multi-agent with AF
  %(prog)s ES2002a            # Specific meeting
  %(prog)s -single            # All meetings, single-agent
  %(prog)s -no_af             # All meetings, multi-agent without AF
  %(prog)s -skip_E            # Output only, no evaluation
  %(prog)s ES2002a -single -no_af  # Specific meeting, single-agent, no AF
  %(prog)s -reE               # Re-evaluate existing results only
  %(prog)s -RAGAS               # Use RAGAS evaluation
  %(prog)s --source-file f.jsonl  # Load from existing results
        """
    )
    parser.add_argument("meeting_id", nargs="?", default=None, 
                        help="Specific meeting ID to process (default: all meetings)")
    parser.add_argument("-single", "--single-agent", action="store_true", 
                        help="Use single-agent mode (instead of multi-agent)")
    parser.add_argument("-no_af", "--no-af", action="store_true", 
                        help="Disable argumentation framework")
    parser.add_argument("-skip_E", "--skip-eval", action="store_true", 
                        help="Output only, skip evaluation (no metrics)")
    parser.add_argument("-reE", "--reevaluate", action="store_true", 
                        help="Re-evaluate existing results only")
    parser.add_argument("-RAGAS", "--ragas", action="store_true", 
                        help="Use RAGAS evaluation (-reE -RAGAS to run both)")
    parser.add_argument("--source-file", type=str, default=None, 
                        help="Load results from this JSONL file")
    
    args = parser.parse_args()
    
    main(
        multi_agent=not args.single_agent,
        use_af=not args.no_af,
        skip_eval=args.skip_eval,
        reevaluate_only=args.reevaluate,
        RAGAS_eval=args.ragas,
        source_file=args.source_file,
        meeting_id=args.meeting_id,
    )
