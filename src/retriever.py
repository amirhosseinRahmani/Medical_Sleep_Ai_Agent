import time


class Retriever:
    """Find the most similar chunks for a question."""

    def __init__(self, index, chunks, embedding_model):
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model

    def search(self, question, top_k):
        """Return the top-k chunks and their similarity scores."""
        start_time = time.perf_counter()

        question_vector = self.embedding_model.create_query_embedding(question)
        scores, indexes = self.index.search(question_vector, top_k)

        results = self._build_results(scores[0], indexes[0])
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return results, elapsed_ms

    def _build_results(self, scores, indexes):
        """Turn FAISS indexes into readable chunk dictionaries."""
        results = []

        for score, index in zip(scores, indexes):
            if index < 0:
                continue

            chunk = dict(self.chunks[index])
            chunk["score"] = float(score)
            results.append(chunk)

        return results
