import math
import re

from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity


def recall_at_k(retrieved_docs, relevant_docs, k):
    relevant = set(relevant_docs)
    retrieved = set(retrieved_docs[:k])

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(retrieved_docs, relevant_docs, k):
    relevant = set(relevant_docs)
    retrieved = retrieved_docs[:k]

    if not retrieved:
        return 0.0

    return sum(doc in relevant for doc in retrieved) / len(retrieved)


def reciprocal_rank(retrieved_docs, relevant_docs):
    relevant = set(relevant_docs)

    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_docs, relevant_docs, k):
    relevant = set(relevant_docs)
    gains = []

    for doc in retrieved_docs[:k]:
        gains.append(1 if doc in relevant else 0)

    dcg = 0.0

    for rank, gain in enumerate(gains, start=1):
        dcg += gain / math.log2(rank + 1)

    ideal_count = min(len(relevant), k)
    idcg = 0.0

    for rank in range(1, ideal_count + 1):
        idcg += 1 / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def token_f1(reference, answer):
    reference_tokens = _tokenize(reference)
    answer_tokens = _tokenize(answer)

    if not reference_tokens or not answer_tokens:
        return 0.0

    reference_counts = _count_tokens(reference_tokens)
    answer_counts = _count_tokens(answer_tokens)

    common = 0

    for token in reference_counts:
        common += min(reference_counts[token], answer_counts.get(token, 0))

    precision = common / len(answer_tokens)
    recall = common / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def rouge_scores(reference, answer):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(reference, answer)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def semantic_similarity(reference, answer, embedding_model):
    vectors = embedding_model.create_embeddings([reference, answer])
    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return float(score)


def grounding_score(answer, retrieved_texts):
    """Diagnostic score showing how much answer vocabulary appears in context."""
    answer_words = set(_tokenize(answer))
    context_words = set(_tokenize(" ".join(retrieved_texts)))

    if not answer_words:
        return 0.0

    return len(answer_words & context_words) / len(answer_words)


def _tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+(?:[-_/\.][a-zA-Z0-9]+)*\b", text.lower())


def _count_tokens(tokens):
    counts = {}

    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    return counts
