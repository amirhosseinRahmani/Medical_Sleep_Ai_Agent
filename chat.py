from config import EMBEDDING_MODEL, GEMMA_MODEL, INDEX_FILE, MAX_NEW_TOKENS, STORE_FILE, TEMPERATURE, TOP_K
from src.embedding import EmbeddingModel
from src.gemma_model import GemmaModel
from src.prompt_builder import build_prompt
from src.retriever import Retriever
from src.vector_database import VectorDatabase


def load_retriever():
    """Load FAISS, chunks and the embedding model."""
    vector_database = VectorDatabase()
    index = vector_database.load_index(INDEX_FILE)
    store = vector_database.load_store(STORE_FILE)
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)

    return Retriever(index, store["chunks"], embedding_model)


def print_sources(results):
    """Print the chunks used by the answer."""
    print("\nSources used:")

    for number, result in enumerate(results, start=1):
        print(
            f"[{number}] {result['title']} | "
            f"chunk={result['chunk_id']} | "
            f"score={result['score']:.4f}"
        )


def ask_question(question, retriever, gemma):
    """Run one complete RAG question."""
    results, retrieval_ms = retriever.search(question, TOP_K)
    prompt = build_prompt(question, results)
    answer, generation_ms = gemma.generate(prompt)

    print("\nAnswer:")
    print(answer)

    print_sources(results)
    print(f"\nRetrieval: {retrieval_ms:.1f} ms")
    print(f"Gemma generation: {generation_ms:.1f} ms")


def main():
    print("Loading RAG...")
    retriever = load_retriever()

    print("Loading Gemma...")
    gemma = GemmaModel(
        GEMMA_MODEL,
        MAX_NEW_TOKENS,
        TEMPERATURE,
    )

    print("\nRAG is ready.")
    print("Type 'exit' to close the program.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        ask_question(question, retriever, gemma)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
