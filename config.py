from pathlib import Path

# The main folder of the project.
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Data files
# -----------------------------
DATA_FILE = BASE_DIR / "rag_ready.json"
QUESTIONS_FILE = BASE_DIR / "questions.json"

# -----------------------------
# Vector database
# -----------------------------
VECTOR_DIR = BASE_DIR / "vector_db"
INDEX_FILE = VECTOR_DIR / "faiss.index"
STORE_FILE = VECTOR_DIR / "store.json"

# -----------------------------
# Models
# -----------------------------
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Change this to your downloaded Gemma folder.
# Example:
# GEMMA_MODEL = r"D:\Models\gemma-3-4b-it"
GEMMA_MODEL = r"C:\Users\senator\PycharmProjects\Artificial--Intelligence\Text-AI\models\gemma"

# -----------------------------
# RAG settings
# -----------------------------
TOP_K = 5
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1

# Number of questions used by evaluation.
# Set to None to use all questions.
MAX_EVALUATION_QUESTIONS = None
