import os
import sys
from datetime import datetime
from time import perf_counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from const.consts import AGENT_LLM_MODEL
from models.llm import LLM
from utils.run_helper import append_jsonl, run_with_retry, runtime_summary_record


def _extract_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
        return "".join(texts)

    return ""


def run_once(model: Any, prompt: str) -> str:
    chunks = model.stream(prompt)
    full_response = ""
    for chunk in chunks:
        text = _extract_chunk_text(chunk)
        if text:
            full_response += text
    return full_response


def main() -> None:
    # Explicitly disable Streamlit mode for CLI execution.
    llm = LLM(vendor_name=AGENT_LLM_MODEL, temperature=0.8)
    model = llm.getModel()

    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoints_dir, f"single_{session_id}.jsonl")

    total_elapsed_seconds = 0.0
    total_log_seconds = 0.0
    turn_count = 0

    print(f"Checkpoint output: {checkpoint_file}")
    print("\n===== Single-Agent Run =====")

    while True:
        user_input = input("\nEnter the agenda (press Enter on empty input to finish): ").strip()
        if not user_input:
            break

        turn_count += 1

        input_query = f'''Based on the following "meeting information", conduct a discussion from the perspectives of multiple participants and create meeting minutes.

    ## Meeting Information
{user_input}

    ## Instructions
    - Let each participant speak from their position described in "conditions" and develop the discussion.
    - Keep the discussion aligned with the purpose in "description" and derive a conclusion.
    - Output the minutes in English using the structure below.

    ## Minutes Output Format
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
        response = run_with_retry(
            "Single-agent response generation",
            lambda: run_once(model, input_query),
        )
        elapsed = perf_counter() - started
        total_elapsed_seconds += elapsed

        print("\n--- Input ---")
        print(user_input)
        print("\n--- Output ---")
        print(response)

        log_started = perf_counter()
        append_jsonl(
            checkpoint_file,
            {
                "index": turn_count,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input": user_input,
                "output": response,
                "elapsed_seconds": elapsed,
            },
        )
        total_log_seconds += perf_counter() - log_started

    print("\n===== Runtime Report =====")
    print(f"Total elapsed time: {total_elapsed_seconds:.3f} sec")
    print(f"Logging time:       {total_log_seconds:.3f} sec")
    print(f"Net processing time (excluding logging): {max(0.0, total_elapsed_seconds - total_log_seconds):.3f} sec")

    append_jsonl(
        checkpoint_file,
        runtime_summary_record(
            total_elapsed_seconds=total_elapsed_seconds,
            total_log_seconds=total_log_seconds,
            total_net_seconds=max(0.0, total_elapsed_seconds - total_log_seconds),
            extra={"turn_count": turn_count},
        ),
    )

    print(f"Executed turns: {turn_count}")

    print("\nFinished.")


if __name__ == "__main__":
    main()
