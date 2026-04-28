import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from const.consts import AGENT_LLM_MODEL
from models.llm import LLM
from utils.run_helper import run_with_retry

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

def _message_to_text(model: Any, prompt: str) -> str:
    return "".join(
        text
        for text in (_extract_chunk_text(chunk) for chunk in model.stream(prompt))
        if text
    )


def add_theme(input_data: str) -> str:
    model = LLM(vendor_name=AGENT_LLM_MODEL, temperature=0.8).getModel()
    return run_with_retry(
        "Response generation",
        lambda: _message_to_text(model, input_data),
    )

if __name__ == "__main__":
    prompt = "Hello, Who are you?"
    result = add_theme(prompt)
    print(result)