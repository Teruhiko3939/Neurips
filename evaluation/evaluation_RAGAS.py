"""
evaluation_RAGAS.py — RAGAS-based evaluation for Input / Output / Reference

処理の流れ:
  1. results/ 以下の各条件ディレクトリ (multi_af, multi_noaf, single_4, single_5 等) を指定→該当ディレクトリをそれぞれ走査
  2. 各条件の eval_*.jsonl ファイルを読み込む
     - input_query  : 入力 (議題テキスト)
     - content      : 出力 (生成された議事録)
     - reference    : 参照 (AMI データセットの正解アノテーション)
  3. RAGAS メトリクスで評価
     - Input-Output 評価  : answer_relevancy_v2, faithfulness_v2, summary_score, topic_adherence
       → 生成された議事録が議題にどれだけ適切か
     - Output-Reference 評価 : rouge_1_score, rouge_L_score, semantic_similarity, context_precision, context_recall, answer_correctness, answer_accuracy, factual_correctness
       → 生成された議事録が正解アノテーションとどれだけ一致するか
  4. 結果を ragas_eval_*.jsonl として同ディレクトリに保存し、
     条件ごとの集計を ragas_summary_*.jsonl に保存する

    """

import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

from evaluation.evaluation_helper import _clean_query, _clean_answer

# evaluation_RAGAS_helper から RAGASevaluator と補助関数をインポート
from evaluation.evaluation_RAGAS_helper import (
    RAGASevaluator,
    get_results_base_dir,
    get_result_directories,
    get_eval_jsonl_files,
    load_eval_record,
    write_temp_reference,
    remove_temp_file,
    save_ragas_result,
    aggregate_results,
    print_summary,
    save_aggregated_results,
)

# =============================================================================
# デフォルト設定
# =============================================================================

# Input-Output 評価に使用するメトリクス
# 対応するメソッド:
#   evaluate_input_context_output : answer_relevancy_v2, faithfulness_v2
#   evaluate_summary              : summary_score
#   evaluate_agent                : topic_adherence
DEFAULT_IN_OUT_METRICS = [
    # "answer_relevancy_v2",  # 一時的に無効化
    # "faithfulness_v2",      # 一時的に無効化
    # "summary_score",
    # "topic_adherence",      # 一時的に無効化
]

# Output-Reference 評価に使用するメトリクス
# 対応するメソッド:
#   evaluate_output_reference : rouge_1_score, rouge_L_score, semantic_similarity,
#                               answer_accuracy, factual_correctness
#   evaluate_ragas (legacy)   : context_precision, context_recall, answer_correctness
DEFAULT_REF_OUT_METRICS = [
    # "semantic_similarity",
    # "context_precision",
    # "context_recall",
    # "answer_correctness",
    "answer_accuracy",
    "factual_correctness",
]

# メトリクス → 使用する evaluator メソッドのマッピング
# _IN_OUT_METRIC_GROUPS = {
#     "input_context_output": {"answer_relevancy_v2", "faithfulness_v2"},
#     "summary":              {"summary_score"},
#     "agent":                {"topic_adherence"},
# }
_IN_OUT_METRIC_GROUPS = {
    "input_context_output": {"answer_relevancy_v2", "faithfulness_v2"},
    # "summary":              {"summary_score"},
    "agent":                {"topic_adherence"},
}

_REF_OUT_METRIC_GROUPS = {
    # "output_reference": {"semantic_similarity", "answer_accuracy", "factual_correctness"},
    "output_reference": {"answer_accuracy", "factual_correctness"},
    # "legacy":           {"context_precision", "context_recall",
    #                      "answer_correctness", "faithfulness", "answer_relevancy"},
}


# =============================================================================
# 1 件の評価レコードを RAGAS で評価する関数
# =============================================================================

def evaluate_record(
    evaluator: RAGASevaluator,
    record: dict,
    in_out_metrics: list,
    ref_out_metrics: list,
) -> dict:
    """1 件の eval レコードを RAGAS で評価し、スコア辞書を返す

    Args:
        evaluator: RAGASevaluator インスタンス
        record: load_eval_record() の戻り値
        in_out_metrics: Input-Output 評価に使用するメトリクスのリスト
        ref_out_metrics: Output-Reference 評価に使用するメトリクスのリスト

    Returns:
        dict: {
            "score_ragas_in_out":  {metric: score, ...},
            "score_ragas_ref_out": {metric: score, ...},
        }
    """
    input_query = record["input_query"]
    content     = record["content"]
    reference   = record["reference"]
    meeting_id  = record["meeting_id"]

    ragas_scores: dict = {}

    # ------------------------------------------------------------------
    # Input-Output 評価
    # ------------------------------------------------------------------
    if in_out_metrics:
        print(f"  [Input-Output] {meeting_id}: {in_out_metrics}")
        combined_in_out: dict = {}

        # --- evaluate_input_context_output (answer_relevancy_v2, faithfulness_v2) ---
        ict_metrics = [m for m in in_out_metrics if m in _IN_OUT_METRIC_GROUPS.get("input_context_output", set())]
        if ict_metrics:
            tmp_ref = write_temp_reference(reference)
            try:
                r = evaluator.evaluate_input_context_output(
                    question=input_query,
                    result=content,
                    correct_info_path=tmp_ref,
                    elements=ict_metrics,
                )
                combined_in_out.update(r.scores)
            except Exception as e:
                print(f"  警告: input_context_output 評価エラー ({meeting_id}): {e}")
                combined_in_out.update({m: None for m in ict_metrics})
            finally:
                remove_temp_file(tmp_ref)

        # --- evaluate_summary (summary_score) ---
        sum_metrics = [m for m in in_out_metrics if m in _IN_OUT_METRIC_GROUPS.get("summary", set())]
        if sum_metrics:
            tmp_ref = write_temp_reference(reference)
            try:
                r = evaluator.evaluate_summary(
                    question=input_query,
                    result=content,
                    correct_info_path=tmp_ref,
                    elements=sum_metrics,
                )
                combined_in_out.update(r.scores)
            except Exception as e:
                print(f"  警告: summary 評価エラー ({meeting_id}): {e}")
                combined_in_out.update({m: None for m in sum_metrics})
            finally:
                remove_temp_file(tmp_ref)

        # --- evaluate_agent (topic_adherence) ---
        # reference を 1 行 1 トピックとして解釈する
        agent_metrics = [m for m in in_out_metrics if m in _IN_OUT_METRIC_GROUPS.get("agent", set())]
        if agent_metrics:
            tmp_ref = write_temp_reference(reference)
            try:
                r = evaluator.evaluate_agent(
                    question=input_query,
                    result=content,
                    correct_info_path=tmp_ref,
                    elements=agent_metrics,
                )
                combined_in_out.update(r.scores)
            except Exception as e:
                print(f"  警告: agent 評価エラー ({meeting_id}): {e}")
                combined_in_out.update({m: None for m in agent_metrics})
            finally:
                remove_temp_file(tmp_ref)

        ragas_scores["score_ragas_in_out"] = combined_in_out

    # ------------------------------------------------------------------
    # Output-Reference 評価
    # ------------------------------------------------------------------
    if ref_out_metrics:
        print(f"  [Output-Reference] {meeting_id}: {ref_out_metrics}")
        combined_ref_out: dict = {}

        # --- evaluate_output_reference (rouge, semantic_similarity, etc.) ---
        or_metrics = [m for m in ref_out_metrics if m in _REF_OUT_METRIC_GROUPS.get("output_reference", set())]
        if or_metrics:
            tmp_ref = write_temp_reference(reference)
            try:
                r = evaluator.evaluate_output_reference(
                    question=input_query,
                    result=content,
                    correct_info_path=tmp_ref,
                    elements=or_metrics,
                )
                combined_ref_out.update(r.scores)
            except Exception as e:
                print(f"  警告: output_reference 評価エラー ({meeting_id}): {e}")
                combined_ref_out.update({m: None for m in or_metrics})
            finally:
                remove_temp_file(tmp_ref)

        # --- evaluate_ragas / legacy (context_precision, context_recall, answer_correctness) ---
        legacy_metrics = [m for m in ref_out_metrics if m in _REF_OUT_METRIC_GROUPS.get("legacy", set())]
        if legacy_metrics:
            tmp_ref = write_temp_reference(reference)
            try:
                r = evaluator.evaluate_ragas(
                    question=input_query,
                    result=content,
                    correct_info_path=tmp_ref,
                    elements=legacy_metrics,
                )
                combined_ref_out.update(r.scores)
            except Exception as e:
                print(f"  警告: legacy 評価エラー ({meeting_id}): {e}")
                combined_ref_out.update({m: None for m in legacy_metrics})
            finally:
                remove_temp_file(tmp_ref)

        ragas_scores["score_ragas_ref_out"] = combined_ref_out

    return ragas_scores


# =============================================================================
# 条件ディレクトリ単位の評価関数
# =============================================================================

def evaluate_condition(
    evaluator: RAGASevaluator,
    condition_name: str,
    condition_dir: str,
    in_out_metrics: list,
    ref_out_metrics: list,
    timestamp: str,
) -> None:
    """1 つの結果ディレクトリ（条件）に対して RAGAS 評価を実行する

    Args:
        evaluator: RAGASevaluator インスタンス
        condition_name: 条件名（ディレクトリ名、例: "multi_af"）
        condition_dir: 条件ディレクトリの絶対パス
        in_out_metrics: Input-Output 評価メトリクス
        ref_out_metrics: Output-Reference 評価メトリクス
        timestamp: タイムスタンプ文字列
    """
    eval_files = get_eval_jsonl_files(condition_dir, exclude_ragas=True)

    if not eval_files:
        print(f"\n[{condition_name}] eval ファイルが見つかりません。スキップします。")
        return

    print(f"\n{'=' * 60}")
    print(f"[{condition_name}] 評価開始  ({len(eval_files)} 件)")

    all_results = []
    meeting_ids = []

    for eval_file in eval_files:
        filename = os.path.basename(eval_file)
        print(f"\n  処理中: {filename}")

        try:
            record = load_eval_record(eval_file)
        except Exception as e:
            print(f"  エラー: ファイル読み込み失敗 ({filename}): {e}")
            continue

        meeting_id = record["meeting_id"]
        meeting_ids.append(meeting_id)

        # 空データのチェック
        if not record["input_query"] or not record["content"]:
            print(f"  警告: input_query または content が空です。スキップします。")
            continue

        ragas_scores = evaluate_record(
            evaluator=evaluator,
            record=record,
            in_out_metrics=in_out_metrics,
            ref_out_metrics=ref_out_metrics,
        )

        # 結果を ragas_eval_*.jsonl に保存
        output_path = save_ragas_result(
            original_file_path=eval_file,
            record=record,
            ragas_scores=ragas_scores,
        )
        print(f"  保存完了: {os.path.basename(output_path)}")

        # 集計用にスコアを保持
        all_results.append({
            "meeting_id":          meeting_id,
            "score_ragas_in_out":  ragas_scores.get("score_ragas_in_out", {}),
            "score_ragas_ref_out": ragas_scores.get("score_ragas_ref_out", {}),
        })

    if not all_results:
        print(f"[{condition_name}] 評価結果がありません。")
        return

    # 集計・表示・保存
    aggregated = aggregate_results(all_results)
    print_summary(condition_name, aggregated, meeting_ids)

    summary_path = save_aggregated_results(
        condition_name=condition_name,
        aggregated=aggregated,
        meeting_ids=meeting_ids,
        result_dir=condition_dir,
        timestamp=timestamp,
    )
    print(f"\n[{condition_name}] 集計結果を保存しました: {os.path.basename(summary_path)}")


# =============================================================================
# Sampler_RAGAS — Sampler クラスと同様のインターフェースを提供する集約クラス
# =============================================================================

class Sampler_RAGAS:
    """RAGAS 評価メトリクスサンプラー。

    evaluation.py の Sampler クラスと同様のインターフェースで RAGAS メトリクスを提供する。
    RAGASevaluator を内包し、一時ファイルの管理を内部で完結させる。

    Usage:
        sampler = Sampler_RAGAS()
        scores = sampler.evaluate_in_out(query, answer, reference)
        scores = sampler.evaluate_ref_out(query, answer, reference)
    """

    def __init__(
        self,
        in_out_metrics: list = None,
        ref_out_metrics: list = None,
    ) -> None:
        """RAGASevaluator を初期化する。

        Args:
            in_out_metrics: Input-Output 評価に使用するメトリクスのリスト
                            (None の場合は DEFAULT_IN_OUT_METRICS を使用)
            ref_out_metrics: Output-Reference 評価に使用するメトリクスのリスト
                             (None の場合は DEFAULT_REF_OUT_METRICS を使用)
        """
        # sampler.py など外部から呼ばれる場合に備え、env.txt を自動ロードする
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _env_path = os.path.join(os.path.dirname(_current_dir), "env.txt")
        load_dotenv(_env_path)

        self.evaluator = RAGASevaluator()
        self.in_out_metrics  = in_out_metrics  if in_out_metrics  is not None else list(DEFAULT_IN_OUT_METRICS)
        self.ref_out_metrics = ref_out_metrics if ref_out_metrics is not None else list(DEFAULT_REF_OUT_METRICS)

    def clean_query(self, text: str) -> str:
        return _clean_query(text)

    def clean_answer(self, text: str) -> str:
        return _clean_answer(text)

    def evaluate_in_out(self, query: str, answer: str, reference: str = "") -> dict[str, float]:
        """Input-Output メトリクスを評価する。

        Args:
            query:     入力（議題テキスト）
            answer:    出力（生成された議事録）
            reference: 参照テキスト（topic_adherence で1行1トピックとして使用）

        Returns:
            dict: {metric_name: score, ...}
        """
        record = {"meeting_id": "", "input_query": query, "content": answer, "reference": reference}
        scores = evaluate_record(
            evaluator=self.evaluator,
            record=record,
            in_out_metrics=self.in_out_metrics,
            ref_out_metrics=[],
        )
        return scores.get("score_ragas_in_out", {})

    def evaluate_ref_out(self, query: str, answer: str, reference: str) -> dict[str, float]:
        """Output-Reference メトリクスを評価する。

        Args:
            query:     入力（議題テキスト）
            answer:    出力（生成された議事録）
            reference: 参照テキスト（正解アノテーション）

        Returns:
            dict: {metric_name: score, ...}
        """
        record = {"meeting_id": "", "input_query": query, "content": answer, "reference": reference}
        scores = evaluate_record(
            evaluator=self.evaluator,
            record=record,
            in_out_metrics=[],
            ref_out_metrics=self.ref_out_metrics,
        )
        return scores.get("score_ragas_ref_out", {})


# =============================================================================
# コマンドライン引数のパース
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGAS-based evaluation for Input / Output / Reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        metavar="DIR",
        help="評価対象の結果ディレクトリ名（未指定の場合は全ディレクトリ）。"
             "例: --dirs multi_af single_4",
    )
    parser.add_argument(
        "--in-out-metrics",
        nargs="*",
        default=DEFAULT_IN_OUT_METRICS,
        metavar="METRIC",
        help=f"Input-Output 評価メトリクス (デフォルト: {DEFAULT_IN_OUT_METRICS})",
    )
    parser.add_argument(
        "--ref-out-metrics",
        nargs="*",
        default=DEFAULT_REF_OUT_METRICS,
        metavar="METRIC",
        help=f"Output-Reference 評価メトリクス (デフォルト: {DEFAULT_REF_OUT_METRICS})",
    )
    return parser.parse_args()


# =============================================================================
# メイン関数
# =============================================================================

def main() -> None:
    # 環境変数の読み込み（プロジェクトルートの env.txt を参照）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    env_path = os.path.join(project_root, "env.txt")
    load_dotenv(env_path)

    args = parse_args()

    # 利用可能な結果ディレクトリを取得
    try:
        all_dirs = get_result_directories()
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    if not all_dirs:
        print("エラー: results/ 以下に結果ディレクトリが見つかりません。")
        sys.exit(1)

    # 対象ディレクトリを絞り込む
    if args.dirs:
        target_dirs = {
            name: path
            for name, path in all_dirs.items()
            if name in args.dirs
        }
        missing = set(args.dirs) - set(target_dirs.keys())
        if missing:
            print(f"警告: 以下のディレクトリが見つかりません: {missing}")
    else:
        target_dirs = all_dirs

    if not target_dirs:
        print("エラー: 評価対象のディレクトリが 1 件もありません。")
        sys.exit(1)

    print(f"評価対象ディレクトリ: {list(target_dirs.keys())}")
    print(f"Input-Output メトリクス:  {args.in_out_metrics}")
    print(f"Output-Reference メトリクス: {args.ref_out_metrics}")

    # RAGASevaluator を初期化（1 回だけ）
    try:
        evaluator = RAGASevaluator()
    except RuntimeError as e:
        print(f"エラー: 評価器の初期化に失敗しました: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 各条件ディレクトリを評価
    for condition_name, condition_dir in target_dirs.items():
        evaluate_condition(
            evaluator=evaluator,
            condition_name=condition_name,
            condition_dir=condition_dir,
            in_out_metrics=args.in_out_metrics or [],
            ref_out_metrics=args.ref_out_metrics or [],
            timestamp=timestamp,
        )

    print("\n全評価完了。")


if __name__ == "__main__":
    main()
