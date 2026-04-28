
"""
evaluation.py  —  Evaluation Metrics Library
=====================================================

References for each metric
--------------------------
N_view
    手法   : Azure OpenAI Embeddings + BERTopic によるセンテンスクラスタリング → クラスタ数
    文献   : Grootendorst, M. (2022). BERTopic: Neural topic modeling with a
             class-based TF-IDF procedure. arXiv:2203.05794.
             Neelakantan, A., et al. (2022). Text and Code Embeddings by
             Contrastive Pre-Training. arXiv:2201.10005.

H_view
    手法   : Azure OpenAI Embeddings + BERTopic トピック分布の Shannon Entropy
    文献   : Steenbergen, M. R., Bächtiger, A., Spörndli, M., & Steiner, J. (2003).
             Measuring Political Deliberation: A Discourse Quality Index.
             Comparative European Politics, 1(1), 21–48.
             Neelakantan, A., et al. (2022). Text and Code Embeddings by
             Contrastive Pre-Training. arXiv:2201.10005.

Distinct_2
    手法   : ユニーク bigram / 全 bigram
    文献   : Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016).
             A Diversity-Promoting Objective Function for Neural Conversation Models.
             NAACL-HLT 2016, 110–119.

Semantic_Diversity
    手法   : Azure OpenAI Embeddings による全ペア平均コサイン距離
    文献   : Neelakantan, A., Xu, T., Puri, R., Radford, A., Han, J. M., Tworek, J.,
             Yuan, Q., Tezak, N., Kim, J. W., Hallacy, C., et al. (2022).
             Text and Code Embeddings by Contrastive Pre-Training.
             arXiv:2201.10005.

Lexical_Diversity
    手法   : MTLD (Measure of Textual Lexical Diversity)
    文献   : McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation
             study of sophisticated approaches to lexical diversity assessment.
             Behavior Research Methods, 42(2), 381–392.

Argumentative_Diversity
    手法   : zero-shot NLI による Toulmin 5 スロット分類 → 検出種別数 / 5
    文献   : Chernodub, A., Oliynyk, O., Heidenreich, P., Bondarenko, A., Hagen, M.,
             Stein, B., & Bondarenko, A. (2019). TARGER: Neural Argument Mining at
             Your Fingertips. ACL 2019 demo, 195–200.
             Toulmin, S. E. (1958). The Uses of Argument. Cambridge University Press.

Oppositionality
    手法   : RoBERTa-large-MNLI による全ペア contradiction 確率の平均
    文献   : Williams, A., Nangia, N., & Bowman, S. R. (2018). A Broad-Coverage
             Challenge Corpus for Sentence Understanding through Inference.
             NAACL-HLT 2018, 1112–1122.

Novelty
    手法   : 1 - BERTScore recall(answer, query)
    文献   : Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020).
             BERTScore: Evaluating Text Generation with BERT. ICLR 2020.

Coherence
    手法   : Azure OpenAI Embeddings による連続文ペアのコサイン類似度平均 (Entity Grid の近似)
    文献   : Barzilay, R., & Lapata, M. (2008). Modeling Local Coherence: An
             Entity-Based Approach. Computational Linguistics, 34(1), 1–34.
             Neelakantan, A., et al. (2022). Text and Code Embeddings by
             Contrastive Pre-Training. arXiv:2201.10005.

Relevance
    手法   : Azure OpenAI Embeddings による query-answer コサイン類似度
    文献   : Neelakantan, A., et al. (2022). Text and Code Embeddings by
             Contrastive Pre-Training. arXiv:2201.10005.

ROUGE-1/2/L
    手法   : n-gram / LCS 一致率 F1 (参照テキストとの表層比較)
    文献   : Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of
             Summaries. ACL Workshop on Text Summarization Branches Out, 74–81.

Redundancy
    手法   : Azure OpenAI Embeddings による高類似文ペア割合 (類似度 ≥ 0.85)
    文献   : Wang, T., Wan, X., & Wang, C. (2020). DESCRIBE: A Direct Evaluation
             of Steps in Commonsense Reasoning Benchmarks. EMNLP 2020.
             Neelakantan, A., et al. (2022). Text and Code Embeddings by
             Contrastive Pre-Training. arXiv:2201.10005.

Faithfulness
    手法   : cross-encoder/nli-deberta-v3-large による query→各文 含意スコアの平均
    文献   : He, P., Gao, J., & Chen, W. (2021).
             DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
             ICLR 2021.
             Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020).
             On Faithfulness and Factuality in Abstractive Summarization.
             ACL 2020, 1906–1919.
"""

import math
from collections import Counter
from evaluation.evaluation_helper import (
    _bert_scorer, _clean_answer, _clean_query, _cloud_embedder, _encode,
    _nli, _nli_large, _pairwise_indices, _split_sentences,
    _topic_distribution, _zero_shot, _flatten_results
)

_TOULMIN_LABELS = ["claim", "data", "warrant", "backing", "rebuttal"]
_NLI_CONTRADICTION = "CONTRADICTION"
_NLI_ENTAILMENT    = "ENTAILMENT"
_REDUNDANCY_THRESH = 0.92          # Sentence pairs with similarity >= this threshold are considered "redundant"
_BATCH_SIZE = 32  # Adjust based on GPU memory (use 8 if out of memory)

def calc_N_view_score(query: str, answer: str) -> float:
    """Number of viewpoint clusters (Grootendorst 2022)."""
    return float(_topic_distribution(answer)[0])

def calc_H_view_score(query: str, answer: str) -> float:
    """Shannon Entropy [bits] of topic distribution (Steenbergen et al. 2003)."""
    topics = _topic_distribution(answer)[1]
    if not topics:
        return 0.0
    total = len(topics)
    return -sum((c / total) * math.log2(c / total) for c in Counter(topics).values())


def calc_distinct2_score(query: str, answer: str) -> float:
    """Unique bigrams / total bigrams (Li et al. 2016)."""
    tokens = answer.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return len(set(bigrams)) / len(bigrams)


def calc_semantic_diversity_score(query: str, answer: str) -> float:
    """Mean pairwise cosine distance via Azure OpenAI Embeddings (Neelakantan et al. 2022). Range [0, 1]."""
    import numpy as np

    embs = _encode(answer)  # Reuse cached embeddings
    if len(embs) < 2:
        return 0.0
    # Compute all pairwise similarities at once using numpy matrix multiplication (no Python loop needed)
    n = len(embs)
    i, j = np.triu_indices(n, k=1)
    mean_sim = float((embs @ embs.T)[i, j].mean())
    return 1.0 - mean_sim


def calc_lexical_diversity_score(query: str, answer: str) -> float:
    """MTLD (McCarthy & Jarvis 2010). A text-length-robust lexical diversity metric."""
    tokens = answer.split()
    if len(tokens) < 2:
        return 0.0

    threshold = 0.72

    def _pass(toks: list[str]) -> float:
        factors, seg_count, unique = 0.0, 0, set()
        for tok in toks:
            unique.add(tok.lower())
            seg_count += 1
            if len(unique) / seg_count <= threshold:
                factors += 1
                seg_count, unique = 0, set()
        if seg_count:
            factors += (1 - len(unique) / seg_count) / (1 - threshold)
        return len(toks) / factors if factors else float(len(toks))

    return (_pass(tokens) + _pass(list(reversed(tokens)))) / 2


def calc_argumentative_diversity_score(query: str, answer: str) -> float:
    """Proportion of detected Toulmin 5 slots (Chernodub et al. 2019; Toulmin 1958). Range [0, 1]."""
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    # Batch inference over all sentences at once (eliminates per-call loop)
    results = _zero_shot()(sentences, _TOULMIN_LABELS, multi_label=False, batch_size=_BATCH_SIZE)
    if isinstance(results, dict):  # A single sentence returns a dict instead of a list
        results = [results]
    found = {r["labels"][0] for r in results if r["scores"][0] > 0.4}
    return len(found) / len(_TOULMIN_LABELS)


def calc_oppositionality_score(query: str, answer: str) -> float:
    """Mean pairwise contradiction probability via RoBERTa-large-MNLI (Williams et al. 2018). Range [0, 1]."""
    sentences = _split_sentences(answer)
    if len(sentences) < 2:
        return 0.0
    pairs = list(_pairwise_indices(len(sentences)))
    # Batch inference over all pairs at once (aggregates O(n²) loop calls into a single batched call)
    pair_texts = [f"{sentences[i]} </s> {sentences[j]}" for i, j in pairs]
    results = _nli()(pair_texts, batch_size=_BATCH_SIZE)
    results = _flatten_results(results)
    total = sum(
        result.get("score", 0.0) if result.get("label") == _NLI_CONTRADICTION else 0.0
        for result in results
    )
    return total / len(pairs)


def calc_novelty_score(query: str, answer: str) -> float:
    """1 - BERTScore recall(answer, query) (Zhang et al. 2020). Range [0, 1]."""
    if not query.strip() or not answer.strip():
        return 0.0
    _, R, _ = _bert_scorer().score([answer], [query])
    return float(1.0 - R[0].item())


def calc_coherence_score(query: str, answer: str) -> float:
    """Mean cosine similarity of consecutive sentence pairs via Azure OpenAI Embeddings (Barzilay & Lapata 2008). Range [0, 1]."""
    embs = _encode(answer)  # Reuse cached embeddings
    if len(embs) < 2:
        return 1.0
    # Compute dot products for consecutive pairs using vectorized operations
    return float((embs[:-1] * embs[1:]).sum(axis=1).mean())


def calc_relevance_score(query: str, answer: str) -> float:
    """Query-answer cosine similarity via Azure OpenAI Embeddings (Neelakantan et al. 2022). Range [0, 1]."""
    import numpy as np
    if not query.strip() or not answer.strip():
        return 0.0
    embedder = _cloud_embedder()
    q_emb = np.array(embedder.embed_query(query), dtype="float32")
    a_emb = np.array(embedder.embed_query(answer), dtype="float32")
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
    a_emb = a_emb / (np.linalg.norm(a_emb) or 1.0)
    return float(np.dot(q_emb, a_emb))

def calc_rouge_scores(reference: str, answer: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 (Lin 2004). n-gram overlap rate against reference text. Range [0, 1]."""
    from rouge_score import rouge_scorer as _rs
    if not reference.strip() or not answer.strip():
        return {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    s = scorer.score(reference, answer)
    return {
        "ROUGE-1": s["rouge1"].fmeasure,
        "ROUGE-2": s["rouge2"].fmeasure,
        "ROUGE-L": s["rougeL"].fmeasure,
    }


def calc_redundancy_score(query: str, answer: str) -> float:
    """Proportion of highly similar sentence pairs via Azure OpenAI Embeddings (Semantic Redundancy). Range [0, 1]. Higher = more redundant."""
    import numpy as np
    embs = _encode(answer)
    if len(embs) < 2:
        return 0.0
    n = len(embs)
    i, j = np.triu_indices(n, k=1)
    sims = (embs @ embs.T)[i, j]
    return float((sims >= _REDUNDANCY_THRESH).mean())


def calc_faithfulness_score(query: str, answer: str) -> float:
    """Mean entailment score of each answer sentence conditioned on query (NLI-based). Range [0, 1]."""
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    pair_texts = [f"{query} </s> {s}" for s in sentences]

    results = _nli_large()(pair_texts, batch_size=_BATCH_SIZE, truncation=True, max_length=1024)
    results = _flatten_results(results)
    total = sum(
        result.get("score", 0.0) if result.get("label") == _NLI_ENTAILMENT else 0.0
        for result in results
    )
    return total / len(sentences)


# ── Aggregate evaluation ────────────────────────────────────────────────

class Sampler:
    """Evaluation metric sampler.
    Pre-loads all models in __init__ and computes each score in evaluate.
    """

    def __init__(self) -> None:
        """Pre-load all models into cache."""
        _cloud_embedder()
        _zero_shot()
        _nli()
        _nli_large()
        _bert_scorer()

    def clean_query(self, text: str) -> str:
        """Remove noise from a JSON-format query and return the cleaned text."""
        return _clean_query(text)

    def clean_answer(self, text: str) -> str:
        """Remove noise lines from a markdown-format answer and return the cleaned text."""
        return _clean_answer(text)

    def evaluate_in_out(self, query: str, answer: str, reference: str = "") -> dict[str, float]:
        """Compute all evaluation metrics for a query-answer pair."""
        return {
            "N_view":                  calc_N_view_score(query, answer),
            "H_view":                  calc_H_view_score(query, answer),
            # "Distinct_2":              calc_distinct2_score(query, answer),
            "Semantic_Diversity":      calc_semantic_diversity_score(query, answer),
            "Lexical_Diversity":       calc_lexical_diversity_score(query, answer),
            "Redundancy":              calc_redundancy_score(query, answer),
            "Novelty (input-output)":  calc_novelty_score(query, answer),
            # "Relevance":               calc_relevance_score(query, answer),
            # "Coherence":               calc_coherence_score(query, answer),
        }
    
    def evaluate_ref_out(self, query: str, answer: str, reference: str = "") -> dict[str, float]:
        """Compute Novelty and ROUGE for a reference-answer pair."""
        return {
            "Novelty (reference-output)": calc_novelty_score(reference, answer),
            **calc_rouge_scores(reference, answer),
        }
    
    def evaluate_arg_div(self, query: str, answer: str) -> dict[str, float]:
        """Compute Argumentative_Diversity for a query-answer pair."""
        return {
            "Argumentative_Diversity": calc_argumentative_diversity_score(query, answer),
        }

    def evaluate_faith(self, query: str, answer: str) -> dict[str, float]:
        """Compute Faithfulness for a query-answer pair."""
        return {
            "Faithfulness": calc_faithfulness_score(query, answer),
        }
    
    def evaluate_opp(self, query: str, answer: str) -> dict[str, float]:
        """Compute Oppositionality for a query-answer pair."""
        return {
            "Oppositionality":         calc_oppositionality_score(query, answer),
        }

