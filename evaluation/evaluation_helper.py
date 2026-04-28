
import json
import logging
import os
import re
import torch
from functools import lru_cache

# ── Suppress verbose library output ─────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
for _logger_name in ("transformers", "sentence_transformers", "bertopic", "torch"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def _configured_gpu_ids() -> list[int]:
    """Return GPU IDs from MAGI_GPU_IDS (comma-separated), or an empty list if unset."""
    raw = os.getenv("MAGI_GPU_IDS", "").strip()
    if not raw:
        return []

    ids: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError:
            logging.warning("Ignoring invalid MAGI_GPU_IDS token: %s", token)
            continue
        if gpu_id < 0:
            logging.warning("Ignoring negative GPU ID in MAGI_GPU_IDS: %s", token)
            continue
        ids.append(gpu_id)
    return ids


def _n_gpu() -> int:
    """Return the number of visible GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _available_gpu_indices() -> list[int]:
    """Return usable GPU indices, honoring MAGI_GPU_IDS when provided."""
    n = _n_gpu()
    if n <= 0:
        return []

    configured = _configured_gpu_ids()
    if not configured:
        return list(range(n))

    valid = [idx for idx in configured if 0 <= idx < n]
    if not valid:
        logging.warning(
            "MAGI_GPU_IDS=%s does not match visible GPU range [0, %d]; using all visible GPUs.",
            configured,
            n - 1,
        )
        return list(range(n))
    return valid


def _gpu_for_slot(slot: int) -> int:
    """Pick a GPU index for a model slot with round-robin assignment."""
    gpus = _available_gpu_indices()
    if not gpus:
        return -1
    return gpus[slot % len(gpus)]


def _pipeline_device(slot: int) -> int:
    """Return device index for transformers pipeline (-1 means CPU)."""
    return _gpu_for_slot(slot)


def _torch_device(slot: int) -> str | None:
    """Return torch device string for libraries expecting 'cuda:x' or None for CPU."""
    gpu = _gpu_for_slot(slot)
    return f"cuda:{gpu}" if gpu >= 0 else None

def _clean_answer(text: str) -> str:
    """Remove markdown noise symbols (horizontal rules, # headers, ** bold markers)
    and merge speaker label lines (lines ending with a colon) with the following utterance line."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Completely remove horizontal rules (--- / === etc.)
        if re.fullmatch(r'[-=*_]{2,}', s):
            continue
        # Convert markdown header symbols (# ...) to [H{n}] prefix to preserve hierarchy
        m = re.match(r'^(#{1,6})\s+(.*)', s)
        if m:
            level = len(m.group(1))
            s = f'[H{level}] {m.group(2).strip()}'
            if not s:
                continue
        # Convert numbered lists to [N] form and bullets to ・ (preserves structure in one line)
        s = re.sub(r'^(\d+)\.\s+', lambda m: f'[{m.group(1)}] ', s)
        s = re.sub(r'^[-\*•]\s+', '・', s).strip()
        if not s:
            continue
        # Strip ** ** / * * markers, keep only the text
        s = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', s).strip()
        if not s:
            continue
        lines.append(s)

    # Merge speaker label lines (ending with colon) with the following line
    merged: list[str] = []
    i = 0
    while i < len(lines):
        if re.search(r'[：:]\s*$', lines[i]) and i + 1 < len(lines):
            merged.append(lines[i] + lines[i + 1])
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return '\n'.join(merged)


def _clean_query(text: str) -> str:
    """Extract only column names and string values from a JSON-format query. Returns original text on parse failure."""
    try:
        data = json.loads(text)
        values: list[str] = []

        def _collect(obj: object) -> None:
            if isinstance(obj, str):
                values.append(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        values.append(f"{k}: {v}")
                    else:
                        values.append(k)
                        _collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    _collect(item)

        _collect(data)
        return '\n'.join(values)
    except Exception:
        return text


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'[。.!?！？\n]+', text) if s.strip()]


def _pairwise_indices(n: int):
    return ((i, j) for i in range(n) for j in range(i + 1, n))


# ── Model singletons ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _cloud_embedder():
    """Azure OpenAI Embeddings singleton (cloud embedding model)."""
    from models.llm import Embedding
    from const.consts import LLM_VENDOR
    return Embedding(LLM_VENDOR.AZURE).getEmbedding()


@lru_cache(maxsize=1)
def _zero_shot():
    from transformers import pipeline
    device = _pipeline_device(slot=0)
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)


@lru_cache(maxsize=1)
def _nli():
    from transformers import pipeline
    device = _pipeline_device(slot=1)
    return pipeline("text-classification", model="roberta-large-mnli", top_k=None, device=device)

@lru_cache(maxsize=1)
def _nli_large():
    from transformers import pipeline
    device = _pipeline_device(slot=2)
    return pipeline(
        "text-classification", model="cross-encoder/nli-deberta-v3-large",  # Supports up to 1024 tokens
        device=device,
    )

@lru_cache(maxsize=1)
def _bert_scorer():
    """BERTScorer singleton (cached to avoid reloading the model on every call)."""
    from bert_score import BERTScorer
    device = _torch_device(slot=0)
    return BERTScorer(lang="ja", device=device)


# ── BERTopic topic distribution cache (shared by N_view / H_view) ───────────────

@lru_cache(maxsize=32)
def _encode(answer: str):
    """Cache Azure OpenAI embeddings (shared by Semantic_Diversity and Coherence)."""
    import numpy as np
    sentences = _split_sentences(answer)
    if not sentences:
        return np.empty((0, 1), dtype="float32")
    vecs = np.array(_cloud_embedder().embed_documents(sentences), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


@lru_cache(maxsize=32)
def _topic_distribution(answer: str) -> tuple[int, tuple[int, ...]]:
    from bertopic import BERTopic

    sentences = _split_sentences(answer)
    if len(sentences) < 2:
        return 1, (0,)

    # Pass pre-computed cloud embeddings directly to BERTopic
    embeddings = _encode(answer)
    topics, _ = BERTopic(
        language="multilingual",
        min_topic_size=2,
        verbose=False,
    ).fit_transform(sentences, embeddings=embeddings)
    valid = tuple(t for t in topics if t != -1)
    return (len(set(valid)) if valid else 1), valid

def _flatten_results(results):
    """Normalize pipeline results to always return a list of dicts."""
    if not results:
        return []
    # List of lists -> flatten
    if isinstance(results[0], list):
        return [item for sublist in results for item in sublist]
    return results