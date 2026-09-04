import json
import time

from config import DATA_FILE, EMBEDDING_MODEL, INDEX_FILE, STORE_FILE, VECTOR_DIR
from src.data_loader import get_chunks, get_chunk_texts, load_rag_data
from src.embedding import EmbeddingModel
from src.vector_database import VectorDatabase


def create_embeddings(model, texts):
    """Create one embedding vector for every chunk."""
    start_time = time.perf_counter()
    embeddings = model.create_embeddings(texts)
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Embedding time: {elapsed_seconds:.2f} seconds")
    return embeddings, elapsed_seconds


def create_store(data, chunks, model_name, dimension, elapsed_seconds):
    """Create the JSON information that belongs to the FAISS index."""
    return {
        "embedding_model": model_name,
        "dimension": int(dimension),
        "embedding_seconds": elapsed_seconds,
        "documents": data["documents"],
        "chunks": chunks,
    }


def save_store(store, vector_database):
    """Save the JSON store."""
    vector_database.save_store(store, STORE_FILE)


def main():
    print("Loading RAG data...")
    data = load_rag_data(DATA_FILE)

    chunks = get_chunks(data)
    texts = get_chunk_texts(chunks)

    print(f"Documents: {len(data['documents'])}")
    print(f"Chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)

    embeddings, elapsed_seconds = create_embeddings(
        embedding_model,
        texts,
    )

    vector_database = VectorDatabase()
    index = vector_database.create_index(embeddings)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vector_database.save_index(index, INDEX_FILE)

    store = create_store(
        data,
        chunks,
        EMBEDDING_MODEL,
        embeddings.shape[1],
        elapsed_seconds,
    )

    save_store(store, vector_database)

    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Vectors in FAISS: {index.ntotal}")
    print(f"FAISS file: {INDEX_FILE}")
    print(f"Store file: {STORE_FILE}")
    print("Build completed successfully.")


if __name__ == "__main__":
    main()
