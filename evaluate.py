from config import (
    EMBEDDING_MODEL,
    GEMMA_MODEL,
    INDEX_FILE,
    MAX_EVALUATION_QUESTIONS,
    MAX_NEW_TOKENS,
    QUESTIONS_FILE,
    STORE_FILE,
    TEMPERATURE,
    TOP_K,
)
from src.data_loader import load_questions
from src.embedding import EmbeddingModel
from src.evaluator import Evaluator
from src.gemma_model import GemmaModel
from src.retriever import Retriever
from src.vector_database import VectorDatabase


def load_retriever():
    """Load the saved FAISS database and its embedding model."""
    vector_database = VectorDatabase()
    index = vector_database.load_index(INDEX_FILE)
    store = vector_database.load_store(STORE_FILE)
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)

    return Retriever(index, store["chunks"], embedding_model)


def load_evaluation_questions():
    """Load questions and optionally limit their number."""
    questions = load_questions(QUESTIONS_FILE)

    if MAX_EVALUATION_QUESTIONS is not None:
        questions = questions[:MAX_EVALUATION_QUESTIONS]

    return questions


def print_summary(summary):
    """Print the final evaluation numbers."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print("\nRetrieval")
    for name, value in summary["retrieval"].items():
        print(f"{name:15}: {value:.4f}")

    print("\nAnswer quality")
    for name, value in summary["answer_quality"].items():
        print(f"{name:22}: {value:.4f}")

    print("\nPerformance")
    for name, value in summary["performance"].items():
        print(f"{name:28}: {value:.2f} ms")


def main():
    questions = load_evaluation_questions()
    retriever = load_retriever()

    gemma = GemmaModel(
        GEMMA_MODEL,
        MAX_NEW_TOKENS,
        TEMPERATURE,
    )

    answer_embedding_model = EmbeddingModel(EMBEDDING_MODEL)

    evaluator = Evaluator(
        retriever,
        gemma,
        answer_embedding_model,
        "outputs/reports",
        "outputs/plots",
    )

    _, summary = evaluator.evaluate(questions, TOP_K)
    print_summary(summary)


if __name__ == "__main__":
    main()
