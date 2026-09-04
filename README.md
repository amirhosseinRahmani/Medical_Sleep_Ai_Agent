# SleepMedicalRAG - Clean Version

This is the first complete RAG implementation for the sleep-medicine data you prepared.

The code is intentionally organized so that each function and class has one clear job.
LangChain is NOT used here. The purpose is to understand every RAG step first.

## Your data

The project contains the real files prepared for this project:

- `rag_ready.json` - 5 documents and 173 chunks
- `chunks.json` - 173 chunks
- `questions.json` - 12 evaluation questions
- `metadata.json`
- `entities.json`
- `keywords.json`
- cleaned source text files
- `summary.md`

The RAG pipeline therefore starts from your actual prepared data rather than sample data.

## Architecture

```text
rag_ready.json
      |
      v
 data_loader
      |
      v
  EmbeddingModel
      |
      v
    FAISS
      |
      v
   Retriever
      |
      +------ question
      |
      v
 Prompt Builder
      |
      v
    Gemma
      |
      v
   Answer
```

## 1. Install packages

Create a virtual environment and run:

```bash
pip install -r requirements.txt
```

The NumPy/FAISS versions are pinned because FAISS 1.8.0 can have compatibility problems with NumPy 2.x on Windows.

## 2. Set your Gemma path

Open `config.py` and change:

```python
GEMMA_MODEL = r"C:\Models\gemma"
```

to the folder containing your downloaded Gemma model.

## 3. Build embeddings and FAISS

Run:

```bash
python build_embeddings.py
```

This does only the following:

1. Read `rag_ready.json`
2. Get the 173 chunks
3. Create one BGE embedding for every chunk
4. Create a FAISS index
5. Save the index
6. Save chunk metadata

The output is:

```text
vector_db/faiss.index
vector_db/store.json
```

## 4. Test retrieval + Gemma

After the FAISS step succeeds:

```bash
python chat.py
```

Then ask something such as:

```text
What is AHI?
```

The program shows the answer and the retrieved chunks with similarity scores.

## 5. Run evaluation

Run:

```bash
python evaluate.py
```

The evaluation uses the 12 real questions in `questions.json`.

### Retrieval metrics

- Recall@1
- Recall@3
- Recall@5
- Precision@5
- MRR
- nDCG@5

These are document-level metrics because the supplied questions identify `source_docs`, not exact gold chunk IDs.

### Answer metrics

- Token F1
- ROUGE-1
- ROUGE-2
- ROUGE-L
- Semantic similarity
- Grounding diagnostic

### Performance

- Retrieval time
- Gemma generation time
- Total response time

## 6. Generated reports

```text
outputs/
├── reports/
│   ├── evaluation_results.csv
│   └── evaluation_summary.json
│
└── plots/
    ├── retrieval_metrics.png
    ├── answer_quality.png
    ├── latency.png
    └── quality_by_category.png
```

## Important evaluation note

No automatic metric can prove that a medical answer is clinically correct.
ROUGE, token F1 and semantic similarity compare generated text with a reference answer.
The grounding score is only a diagnostic signal showing vocabulary overlap with retrieved context.

For a later research-grade evaluation, add expert-reviewed questions and exact relevant chunk IDs.
Then we can evaluate retrieval at chunk level and add a separate faithfulness/clinical-correctness protocol.

## Why there is no LangChain

LangChain can connect many of these components, but we deliberately implement the core pipeline ourselves first:

```text
Embedding -> FAISS -> Retrieval -> Prompt -> Gemma
```

Once this version works and its metrics are known, LangChain can be introduced as a convenience layer without hiding the underlying process.
