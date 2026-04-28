"""
evaluation_RAGAS_helper.py — Helper functions and RAGASevaluator for RAGAS-based evaluation

Provides:
- RAGASevaluator class (merged from evaluation_tools.py)
- Loading eval JSONL files from results directories
- Building temporary reference files for evaluators
- Saving RAGAS evaluation results
- Aggregating and displaying results
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness as FaithfulnessMetric,
    AnswerRelevancy as AnswerRelevancyMetric,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AspectCritic,
)
from ragas.metrics.collections import (
    SummaryScore,
    AnswerRelevancy,
    Faithfulness,
    AnswerAccuracy,
    SemanticSimilarity,
    FactualCorrectness,
)
from ragas.metrics import (
    TopicAdherenceScore,
    AgentGoalAccuracyWithReference,
    AgentGoalAccuracyWithoutReference,
)
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI, AsyncAzureOpenAI

# RAGAS公式HP内Metrics情報ソース：https://docs.ragas.io/en/latest/concepts/metrics/overview/

# =============================================================================
# データクラス
# =============================================================================

@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""

    scores: dict[str, float]
    raw_result: Any

    def __str__(self) -> str:
        score_lines = [f"{metric}: {score:.4f}" for metric, score in self.scores.items()]
        return "\n".join(score_lines)


# =============================================================================
# 環境変数チェック
# =============================================================================

def check_environment() -> bool:
    """環境変数の確認"""
    required_vars = [
        "RAGAS_CHAT_AZURE_OPENAI_API_KEY",
        "RAGAS_CHAT_AZURE_OPENAI_ENDPOINT",
        "RAGAS_OPENAI_API_VERSION",
        "RAGAS_AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"環境変数が設定されていません(CHAT): {missing}")
        return False

    required_vars = [
        "EMBEDDINGS_AZURE_OPENAI_API_KEY",
        "EMBEDDINGS_AZURE_OPENAI_ENDPOINT",
        "EMBEDDINGS_AZURE_OPENAI_API_VERSION",
        "EMBEDDINGS_AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"環境変数が設定されていません(EMBEDDINGS): {missing}")
        return False

    return True


# =============================================================================
# RAGASevaluator
# =============================================================================

class RAGASevaluator:
    """RAGAS評価クラス"""

    def __init__(self):
        if not check_environment():
            raise RuntimeError("環境変数が設定されていません")

        self.model = AzureChatOpenAI(
            azure_endpoint=os.environ["RAGAS_CHAT_AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["RAGAS_OPENAI_API_VERSION"],
            api_key=os.environ["RAGAS_CHAT_AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["RAGAS_AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["EMBEDDINGS_AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["EMBEDDINGS_AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["EMBEDDINGS_AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["EMBEDDINGS_AZURE_OPENAI_DEPLOYMENT_NAME"],
        )

        self.async_azure_client = AsyncAzureOpenAI(
            api_key=os.environ["RAGAS_CHAT_AZURE_OPENAI_API_KEY"],
            api_version=os.environ["RAGAS_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["RAGAS_CHAT_AZURE_OPENAI_ENDPOINT"],
            max_retries=3,
        )

        self.async_instructor_llm = llm_factory(
            os.environ["RAGAS_AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            client=self.async_azure_client,
        )
        self.async_instructor_llm.model_args = {'max_completion_tokens': 16000}

        self.async_modern_embeddings = OpenAIEmbeddings(
            model=os.environ["EMBEDDINGS_AZURE_OPENAI_DEPLOYMENT_NAME"],
            client=self.async_azure_client,
        )

        self.legacy_metrics = {
            'faithfulness': FaithfulnessMetric(llm=self.model),
            'answer_relevancy': AnswerRelevancyMetric(llm=self.model, embeddings=self.embeddings),
            'context_recall': ContextRecall(llm=self.model),
            'answer_correctness': AnswerCorrectness(llm=self.model, embeddings=self.embeddings),
            'context_precision': ContextPrecision(llm=self.model),
        }

        self.input_context_output_metrics = {
            'answer_relevancy_v2': AnswerRelevancy(llm=self.async_instructor_llm, embeddings=self.async_modern_embeddings),
            'faithfulness_v2': Faithfulness(llm=self.async_instructor_llm),
        }

        self.output_reference_metrics = {
            'answer_accuracy': AnswerAccuracy(llm=self.async_instructor_llm),
            'semantic_similarity': SemanticSimilarity(embeddings=self.async_modern_embeddings),
            'factual_correctness': FactualCorrectness(llm=self.async_instructor_llm),
        }

        # self.summary_metrics = {
        #     'summary_score': SummaryScore(llm=self.async_instructor_llm),
        # }

        langchain_llm = LangchainLLMWrapper(self.model)
        self.agent_metrics = {
            'topic_adherence': TopicAdherenceScore(llm=self.async_instructor_llm, mode=["precision"]),
            'agent_goal_accuracy_with_reference': AgentGoalAccuracyWithReference(llm=self.async_instructor_llm),
            'agent_goal_accuracy_without_reference': AgentGoalAccuracyWithoutReference(llm=self.async_instructor_llm),
        }

        self.base_agent_metrics = {}
        self.langchain_llm = langchain_llm

        self.collections_metrics = {
            **self.input_context_output_metrics,
            **self.output_reference_metrics,
            # **self.summary_metrics,
            **self.agent_metrics,
        }

    def _extract_score(self, evaluation_result: Any, element: str) -> float:
        """評価結果からスコアを抽出"""
        if hasattr(evaluation_result, 'to_pandas'):
            df = evaluation_result.to_pandas()
            if element in df.columns:
                return df[element].iloc[0]
        if element in evaluation_result:
            return evaluation_result[element]
        return 0.0

    def evaluate_ragas(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """従来メトリクスで評価 (faithfulness, context_precision, context_recall, answer_correctness 等)"""
        metrics = [self.legacy_metrics[elem] for elem in elements if elem in self.legacy_metrics]
        if not metrics:
            raise ValueError(f"指定されたメトリクス {elements} は利用できません。利用可能なメトリクス: {list(self.legacy_metrics.keys())}")

        with open(correct_info_path, 'r', encoding='utf-8') as f:
            correct_info = f.read()

        data = {
            "question": [question],
            "answer": [result.strip()],
            "contexts": [[correct_info]],
            "ground_truth": [correct_info],
        }
        dataset = Dataset.from_dict(data)
        raw_result = evaluate(dataset, metrics=metrics)

        scores = {}
        for element in elements:
            if element in self.legacy_metrics:
                scores[element] = self._extract_score(raw_result, element)
        return EvaluationResult(scores=scores, raw_result=raw_result)

    async def _evaluate_metric_async(self, metric, element: str, question: str, result: str, correct_info: str):
        """個別メトリクスの評価を実行（ヘルパーメソッド）"""
        if element == 'answer_accuracy':
            return await metric.ascore(user_input=question, response=result.strip(), reference=correct_info)
        elif element == 'faithfulness_v2':
            return await metric.ascore(user_input=question, response=result.strip(), retrieved_contexts=[correct_info])
        # elif element == 'summary_score':
        #     return await metric.ascore(response=result.strip(), reference_contexts=correct_info)
        elif element in ('blue_score', 'rouge_score', 'rouge_1_score', 'rouge_L_score',
                         'semantic_similarity', 'factual_correctness'):
            return await metric.ascore(response=result.strip(), reference=correct_info)
        elif element == 'answer_relevancy_v2':
            return await metric.ascore(user_input=question, response=result.strip())
        else:
            return await metric.ascore(user_input=question, response=result.strip())

    async def _evaluate_agent_metric_async(self, metric, element: str, question: str, result: str, reference_data):
        """Agent関連メトリクスの評価を実行（ヘルパーメソッド）"""
        if element == 'topic_adherence':
            sample = MultiTurnSample(
                user_input=[HumanMessage(content=question), AIMessage(content=result.strip())],
                reference_topics=reference_data,
            )
            return await metric.multi_turn_ascore(sample)
        elif element == 'agent_goal_accuracy_with_reference':
            sample = MultiTurnSample(
                user_input=[HumanMessage(content=question), AIMessage(content=result.strip())],
                reference=reference_data,
            )
            return await metric.multi_turn_ascore(sample)
        elif element == 'agent_goal_accuracy_without_reference':
            sample = MultiTurnSample(
                user_input=[HumanMessage(content=question), AIMessage(content=result.strip())],
            )
            return await metric.multi_turn_ascore(sample)
        else:
            return await metric.ascore(user_input=question, response=result.strip())

    async def _evaluate_base_agent_metric_async(self, metric, element: str, question: str, result: str):
        """Agent-baseメトリクスの評価を実行（ヘルパーメソッド）"""
        sample = SingleTurnSample(user_input=question, response=result.strip())
        return await metric.single_turn_ascore(sample)

    async def evaluate_input_context_output_async(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """入力-コンテキスト-出力メトリクスで評価（非同期）: answer_relevancy_v2, faithfulness_v2"""
        with open(correct_info_path, 'r', encoding='utf-8') as f:
            correct_info = f.read()
        scores = {}
        for element in elements:
            if element not in self.input_context_output_metrics:
                print(f"警告: '{element}' は利用できません。利用可能: {list(self.input_context_output_metrics.keys())}")
                continue
            try:
                score_result = await self._evaluate_metric_async(
                    self.input_context_output_metrics[element], element, question, result, correct_info)
                scores[element] = score_result.value
            except Exception as e:
                print(f"エラー: '{element}' の評価中にエラー: {e}")
                scores[element] = 0.0
        return EvaluationResult(scores=scores, raw_result=None)

    async def evaluate_output_reference_async(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """出力-正解メトリクスで評価（非同期）: rouge_*, semantic_similarity, answer_accuracy, factual_correctness 等"""
        with open(correct_info_path, 'r', encoding='utf-8') as f:
            correct_info = f.read()
        scores = {}
        for element in elements:
            if element not in self.output_reference_metrics:
                print(f"警告: '{element}' は利用できません。利用可能: {list(self.output_reference_metrics.keys())}")
                continue
            try:
                score_result = await self._evaluate_metric_async(
                    self.output_reference_metrics[element], element, question, result, correct_info)
                scores[element] = score_result.value if hasattr(score_result, 'value') else score_result
            except Exception as e:
                print(f"エラー: '{element}' の評価中にエラー: {e}")
                scores[element] = 0.0
        return EvaluationResult(scores=scores, raw_result=None)

    async def evaluate_summary_async(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """全体出力-正解メトリクスで評価（非同期）: summary_score"""
        with open(correct_info_path, 'r', encoding='utf-8') as f:
            correct_info = f.read()
        scores = {}
        for element in elements:
            if element not in self.summary_metrics:
                print(f"警告: '{element}' は利用できません。利用可能: {list(self.summary_metrics.keys())}")
                continue
            try:
                score_result = await self._evaluate_metric_async(
                    self.summary_metrics[element], element, question, result, correct_info)
                scores[element] = score_result.value
            except Exception as e:
                print(f"エラー: '{element}' の評価中にエラー: {e}")
                scores[element] = 0.0
        return EvaluationResult(scores=scores, raw_result=None)

    async def evaluate_agent_async(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """Agentメトリクスで評価（非同期）: topic_adherence, agent_goal_accuracy_*"""
        scores = {}
        for element in elements:
            if element not in self.agent_metrics:
                print(f"警告: '{element}' は利用できません。利用可能: {list(self.agent_metrics.keys())}")
                continue
            try:
                if element == 'topic_adherence':
                    with open(correct_info_path, 'r', encoding='utf-8') as f:
                        reference_data = [line.strip() for line in f if line.strip()]
                elif element == 'agent_goal_accuracy_with_reference':
                    with open(correct_info_path, 'r', encoding='utf-8') as f:
                        reference_data = f.read().strip()
                else:
                    reference_data = None
                score_result = await self._evaluate_agent_metric_async(
                    self.agent_metrics[element], element, question, result, reference_data)
                scores[element] = score_result if isinstance(score_result, float) else score_result.value
            except Exception as e:
                print(f"エラー: '{element}' の評価中にエラー: {e}")
                import traceback
                traceback.print_exc()
                scores[element] = 0.0
        return EvaluationResult(scores=scores, raw_result=None)

    async def evaluate_base_agent_async(self, question: str, result: str, correct_info_path: str,
                                        elements: list, agent_name: str, agent_definition: str) -> EvaluationResult:
        """Agent-baseメトリクスで評価（非同期）: aspect_critic"""
        scores = {}
        for element in elements:
            try:
                if element == 'aspect_critic':
                    metric = AspectCritic(name=agent_name, definition=agent_definition, llm=self.langchain_llm)
                elif element in self.base_agent_metrics:
                    metric = self.base_agent_metrics[element]
                else:
                    print(f"警告: '{element}' は利用できません。利用可能: aspect_critic")
                    continue
                score_result = await self._evaluate_base_agent_metric_async(metric, element, question, result)
                scores[element] = float(score_result) if isinstance(score_result, (int, float)) else score_result.value
            except Exception as e:
                print(f"エラー: '{element}' の評価中にエラー: {e}")
                import traceback
                traceback.print_exc()
                scores[element] = 0.0
        return EvaluationResult(scores=scores, raw_result=None)

    # 同期ラッパーメソッド
    def evaluate_input_context_output(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """入力-コンテキスト-出力メトリクス評価（同期）"""
        return asyncio.run(self.evaluate_input_context_output_async(question, result, correct_info_path, elements))

    def evaluate_output_reference(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """出力-正解メトリクス評価（同期）"""
        return asyncio.run(self.evaluate_output_reference_async(question, result, correct_info_path, elements))

    def evaluate_summary(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """全体出力-正解メトリクス評価（同期）"""
        return asyncio.run(self.evaluate_summary_async(question, result, correct_info_path, elements))

    def evaluate_agent(self, question: str, result: str, correct_info_path: str, elements: list) -> EvaluationResult:
        """Agentメトリクス評価（同期）"""
        return asyncio.run(self.evaluate_agent_async(question, result, correct_info_path, elements))

    def evaluate_base_agent(self, question: str, result: str, correct_info_path: str,
                            elements: list, agent_name: str, agent_definition: str) -> EvaluationResult:
        """Agent-baseメトリクス評価（同期）"""
        return asyncio.run(self.evaluate_base_agent_async(question, result, correct_info_path, elements, agent_name, agent_definition))


# =============================================================================
# ディレクトリ・ファイル取得関数
# =============================================================================

def get_results_base_dir() -> str:
    """プロジェクトルートの results/ ディレクトリの絶対パスを返す"""
    # このファイルの場所: evaluation/
    # プロジェクトルート: ../
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    results_dir = os.path.join(project_root, "results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results ディレクトリが見つかりません: {results_dir}")
    return results_dir


def get_result_directories(results_base_dir: str = None) -> Dict[str, str]:
    """利用可能な結果ディレクトリの辞書を返す（ディレクトリ名 → 絶対パス）

    Returns:
        dict: {"multi_af": "/path/to/results/multi_af", ...}
    """
    if results_base_dir is None:
        results_base_dir = get_results_base_dir()
    dirs = {}
    for entry in sorted(os.listdir(results_base_dir)):
        full_path = os.path.join(results_base_dir, entry)
        if os.path.isdir(full_path):
            dirs[entry] = full_path
    return dirs


def get_eval_jsonl_files(result_dir: str, exclude_ragas: bool = True) -> List[str]:
    """指定ディレクトリから eval_*.jsonl ファイルの一覧を返す

    Args:
        result_dir: 結果ディレクトリの絶対パス
        exclude_ragas: True の場合、ragas_eval_* ファイルを除外する

    Returns:
        list: ファイルパスのリスト（ソート済み）
    """
    files = []
    for name in sorted(os.listdir(result_dir)):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("eval_"):
            continue
        if exclude_ragas and name.startswith("eval_ragas_"):
            continue
        files.append(os.path.join(result_dir, name))
    return files


# =============================================================================
# データ読み込み関数
# =============================================================================

def load_eval_record(file_path: str) -> Dict:
    """eval JSONL ファイルの最初の行（レコード）を読み込む

    eval_*.jsonl は 1 行に JSON が格納された形式を想定。

    Args:
        file_path: eval JSONL ファイルのパス

    Returns:
        dict: {
            'meeting_id', 'input_query', 'content', 'reference',
            'source_file', 'score_in_out', 'score_ref_out'
        }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"ファイルが空です: {file_path}")
    data = json.loads(line)
    return {
        "meeting_id":   data.get("meeting_id", ""),
        "input_query":  data.get("input_query", ""),
        "content":      data.get("content", ""),
        "reference":    data.get("reference", ""),
        "source_file":  data.get("source_file", ""),
        "score_in_out": data.get("score_in_out", {}),
        "score_ref_out": data.get("score_ref_out", {}),
    }


# =============================================================================
# 一時ファイル管理
# =============================================================================

def write_temp_reference(reference_text: str) -> str:
    """参照テキストを一時ファイルに書き込み、そのパスを返す

    Args:
        reference_text: 参照テキスト（AMI データセットの正解アノテーション等）

    Returns:
        str: 作成した一時ファイルのパス（使用後は呼び出し元が削除すること）
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(reference_text)
    tmp.close()
    return tmp.name


def remove_temp_file(file_path: str) -> None:
    """一時ファイルを削除する（エラーは警告のみ）"""
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"警告: 一時ファイル削除に失敗しました ({file_path}): {e}")


# =============================================================================
# 結果保存・集計関数
# =============================================================================

def build_output_filename(original_filename: str) -> str:
    """eval_*.jsonl から ragas_eval_*.jsonl へのファイル名変換

    Examples:
        "eval_ES2002a_20260417_124930.jsonl"
        → "ragas_eval_ES2002a_20260417_124930.jsonl"
    """
    if original_filename.startswith("eval_"):
        return "ragas_" + original_filename
    return "ragas_eval_" + original_filename


def save_ragas_result(
    original_file_path: str,
    record: Dict,
    ragas_scores: Dict,
    output_dir: Optional[str] = None,
) -> str:
    """1 件の RAGAS 評価結果を JSONL ファイルに保存する

    Args:
        original_file_path: 元の eval JSONL ファイルのパス
        record: load_eval_record() の戻り値
        ragas_scores: {"score_ragas_in_out": {...}, "score_ragas_ref_out": {...}}
        output_dir: 保存先ディレクトリ（None の場合は元ファイルと同じディレクトリ）

    Returns:
        str: 保存したファイルのパス
    """
    original_dir  = os.path.dirname(original_file_path)
    original_name = os.path.basename(original_file_path)

    if output_dir is None:
        output_dir = original_dir

    output_name = build_output_filename(original_name)
    output_path = os.path.join(output_dir, output_name)

    result = {
        "meeting_id":        record["meeting_id"],
        "input_query":       record["input_query"],
        "content":           record["content"],
        "reference":         record["reference"],
        "source_eval_file":  original_name,
        **ragas_scores,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return output_path


def aggregate_results(all_results: List[Dict]) -> Dict:
    """複数会議の RAGAS スコアを集計し、メトリクスごとの平均を返す

    Args:
        all_results: [{"meeting_id": ..., "score_ragas_in_out": {...}, "score_ragas_ref_out": {...}}, ...]

    Returns:
        dict: {
            "score_ragas_in_out":  {"metric_name": {"mean": float, "scores": [float, ...]}, ...},
            "score_ragas_ref_out": {...}
        }
    """
    in_out_scores: Dict[str, List[float]]  = {}
    ref_out_scores: Dict[str, List[float]] = {}

    for r in all_results:
        for metric, val in r.get("score_ragas_in_out", {}).items():
            in_out_scores.setdefault(metric, []).append(val)
        for metric, val in r.get("score_ragas_ref_out", {}).items():
            ref_out_scores.setdefault(metric, []).append(val)

    def _summarize(scores_dict: Dict[str, List[float]]) -> Dict:
        return {
            metric: {
                "mean":   sum(vals) / len(vals) if vals else 0.0,
                "scores": vals,
            }
            for metric, vals in scores_dict.items()
        }

    return {
        "score_ragas_in_out":  _summarize(in_out_scores),
        "score_ragas_ref_out": _summarize(ref_out_scores),
    }


def print_summary(condition_name: str, aggregated: Dict, meeting_ids: List[str]) -> None:
    """集計結果をコンソールに表示する

    Args:
        condition_name: 結果ディレクトリ名（例: "multi_af"）
        aggregated: aggregate_results() の戻り値
        meeting_ids: 評価対象の meeting_id リスト
    """
    print(f"\n{'=' * 60}")
    print(f"[{condition_name}] RAGAS 評価サマリー  (n={len(meeting_ids)} 会議)")
    print(f"  会議: {', '.join(meeting_ids)}")

    for score_key in ("score_ragas_in_out", "score_ragas_ref_out"):
        section = aggregated.get(score_key, {})
        if not section:
            continue
        label = "Input-Output" if "in_out" in score_key else "Output-Reference"
        print(f"\n  [{label}]")
        for metric, data in section.items():
            scores_str = ", ".join(f"{s:.4f}" for s in data["scores"])
            print(f"    {metric:30s}: mean={data['mean']:.4f}  [{scores_str}]")

    print("=" * 60)


def save_aggregated_results(
    condition_name: str,
    aggregated: Dict,
    meeting_ids: List[str],
    result_dir: str,
    timestamp: str,
) -> str:
    """条件ごとの集計結果を JSONL ファイルに保存する

    Args:
        condition_name: ディレクトリ名（例: "multi_af"）
        aggregated: aggregate_results() の戻り値
        meeting_ids: 対象 meeting_id リスト
        result_dir: 保存先ディレクトリ
        timestamp: タイムスタンプ文字列（例: "20260417_130000"）

    Returns:
        str: 保存ファイルのパス
    """
    output_name = f"ragas_summary_{condition_name}_{timestamp}.jsonl"
    output_path = os.path.join(result_dir, output_name)

    record = {
        "condition":    condition_name,
        "meeting_ids":  meeting_ids,
        "n_meetings":   len(meeting_ids),
        "timestamp":    timestamp,
        **aggregated,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path
