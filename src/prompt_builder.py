SYSTEM_MESSAGE = """
You are a sleep-medicine research assistant.
Answer the user's question using the provided scientific context.
Do not invent information that is not supported by the context.
If the context is not enough, say that clearly.
This system is for research and education, not a substitute for a clinician.
""".strip()


def build_prompt(question, results):
    """Build the exact text that will be sent to Gemma."""
    context = build_context(results)

    prompt = f"""{SYSTEM_MESSAGE}

SCIENTIFIC CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return prompt


def build_context(results):
    """Format retrieved chunks and give each one a source number."""
    parts = []

    for number, result in enumerate(results, start=1):
        title = result.get("title", "Unknown source")
        text = result["text"]

        part = f"[{number}] {title}\n{text}"
        parts.append(part)

    return "\n\n".join(parts)
