import json


def load_rag_data(file_path):
    """Read the complete rag_ready.json file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_chunks(data):
    """Return the list of chunks from rag_ready data."""
    return data["chunks"]


def get_documents(data):
    """Return the list of source documents."""
    return data["documents"]


def get_chunk_texts(chunks):
    """Return only the text of every chunk."""
    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])

    return texts


def load_questions(file_path):
    """Read the evaluation questions."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
