import json
import faiss


class VectorDatabase:
    """Create, save and load the FAISS vector database."""

    def create_index(self, embeddings):
        """Create a FAISS cosine-similarity index."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def save_index(self, index, file_path):
        """Save the FAISS index to disk."""
        faiss.write_index(index, str(file_path))

    def load_index(self, file_path):
        """Load a FAISS index from disk."""
        return faiss.read_index(str(file_path))

    def save_store(self, store, file_path):
        """Save chunks and document information next to the FAISS index."""
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(store, file, ensure_ascii=False, indent=2)

    def load_store(self, file_path):
        """Load chunks and document information."""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
