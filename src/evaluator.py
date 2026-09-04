import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from .metrics import (
    grounding_score,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_scores,
    semantic_similarity,
    token_f1,
)
from .prompt_builder import build_prompt


class Evaluator:
    """Run the complete evaluation and save reports and plots."""

    def __init__(self, retriever, gemma, answer_embedding_model, output_dir, plot_dir):
        self.retriever = retriever
        self.gemma = gemma
        self.answer_embedding_model = answer_embedding_model
        self.output_dir = Path(output_dir)
        self.plot_dir = Path(plot_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, questions, top_k):
        rows = []

        for number, question_data in enumerate(questions, start=1):
            print(f"Evaluating question {number}/{len(questions)}...")
            row = self._evaluate_one(question_data, top_k)
            rows.append(row)

        results = pd.DataFrame(rows)
        results.to_csv(
            self.output_dir / "evaluation_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary = self._create_summary(results)
        self._save_summary(summary)
        self._create_plots(results, summary)

        return results, summary

    def _evaluate_one(self, question_data, top_k):
        question = question_data["question"]
        reference = question_data["answer"]
        relevant_docs = question_data.get("source_docs", [])

        retrieved, retrieval_ms = self.retriever.search(question, top_k)
        retrieved_docs = [item["doc_id"] for item in retrieved]

        prompt = build_prompt(question, retrieved)
        answer, generation_ms = self.gemma.generate(prompt)

        rouge = rouge_scores(reference, answer)
        semantic = semantic_similarity(
            reference,
            answer,
            self.answer_embedding_model,
        )

        retrieved_texts = [item["text"] for item in retrieved]

        return {
            "question": question,
            "category": question_data.get("category", "unknown"),
            "retrieved_docs": ", ".join(retrieved_docs),
            "recall@1": recall_at_k(retrieved_docs, relevant_docs, 1),
            "recall@3": recall_at_k(retrieved_docs, relevant_docs, 3),
            "recall@5": recall_at_k(retrieved_docs, relevant_docs, 5),
            "precision@5": precision_at_k(retrieved_docs, relevant_docs, 5),
            "mrr": reciprocal_rank(retrieved_docs, relevant_docs),
            "ndcg@5": ndcg_at_k(retrieved_docs, relevant_docs, 5),
            "token_f1": token_f1(reference, answer),
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "semantic_similarity": semantic,
            "grounding": grounding_score(answer, retrieved_texts),
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
            "reference_answer": reference,
            "generated_answer": answer,
        }

    def _create_summary(self, results):
        retrieval = {
            "recall@1": float(results["recall@1"].mean()),
            "recall@3": float(results["recall@3"].mean()),
            "recall@5": float(results["recall@5"].mean()),
            "precision@5": float(results["precision@5"].mean()),
            "mrr": float(results["mrr"].mean()),
            "ndcg@5": float(results["ndcg@5"].mean()),
        }

        answer = {
            "token_f1": float(results["token_f1"].mean()),
            "rouge1": float(results["rouge1"].mean()),
            "rouge2": float(results["rouge2"].mean()),
            "rougeL": float(results["rougeL"].mean()),
            "semantic_similarity": float(results["semantic_similarity"].mean()),
            "grounding": float(results["grounding"].mean()),
        }

        performance = {
            "average_retrieval_ms": float(results["retrieval_ms"].mean()),
            "average_generation_ms": float(results["generation_ms"].mean()),
            "average_total_ms": float(results["total_ms"].mean()),
        }

        return {
            "number_of_questions": len(results),
            "retrieval": retrieval,
            "answer_quality": answer,
            "performance": performance,
        }

    def _save_summary(self, summary):
        file_path = self.output_dir / "evaluation_summary.json"

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    def _create_plots(self, results, summary):
        self._plot_retrieval(summary["retrieval"])
        self._plot_answer_quality(summary["answer_quality"])
        self._plot_latency(results)
        self._plot_category_quality(results)

    def _plot_retrieval(self, values):
        self._create_bar_plot(
            list(values.keys()),
            list(values.values()),
            "Retrieval metrics",
            "retrieval_metrics.png",
        )

    def _plot_answer_quality(self, values):
        self._create_bar_plot(
            list(values.keys()),
            list(values.values()),
            "Answer quality metrics",
            "answer_quality.png",
        )

    def _plot_latency(self, results):
        plt.figure(figsize=(9, 5))
        plt.plot(results.index + 1, results["retrieval_ms"], marker="o", label="Retrieval")
        plt.plot(results.index + 1, results["generation_ms"], marker="o", label="Gemma")
        plt.xlabel("Question number")
        plt.ylabel("Milliseconds")
        plt.title("RAG latency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_dir / "latency.png", dpi=160)
        plt.close()

    def _plot_category_quality(self, results):
        grouped = results.groupby("category")["semantic_similarity"].mean()

        self._create_bar_plot(
            list(grouped.index),
            list(grouped.values),
            "Semantic similarity by category",
            "quality_by_category.png",
            rotate_labels=True,
        )

    def _create_bar_plot(self, labels, values, title, filename, rotate_labels=False):
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(title)

        if rotate_labels:
            plt.xticks(rotation=25, ha="right")

        plt.tight_layout()
        plt.savefig(self.plot_dir / filename, dpi=160)
        plt.close()
