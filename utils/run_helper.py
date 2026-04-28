import json
import time
from datetime import datetime
from typing import Any, Callable, TypeVar

import httpx
import openai


T = TypeVar("T")


NETWORK_RETRY_ERRORS = (
    openai.APIConnectionError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


def run_with_retry(action_name: str, fn: Callable[[], T], retries: int = 2) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except NETWORK_RETRY_ERRORS as e:
            if attempt >= retries:
                raise
            wait_sec = 20 * attempt
            print(
                f"\n[WARN] Connection error during {action_name}: {e}. "
                f"Retrying in {wait_sec} seconds ({attempt + 1}/{retries})"
            )
            time.sleep(wait_sec)

def append_jsonl(file_path: str, record: dict[str, Any]) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def runtime_summary_record(
    total_elapsed_seconds: float,
    total_log_seconds: float,
    total_net_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "runtime_summary",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_seconds": total_elapsed_seconds,
        "total_log_seconds": total_log_seconds,
        "total_net_seconds": total_net_seconds,
    }
    if extra:
        record.update(extra)
    return record
