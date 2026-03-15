# Retrieval-Augmented QA on NCKU Wikipedia

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** question-answering system built on an NCKU Wikipedia knowledge source.

The project evaluates answer quality **with and without RAG** using a self-constructed QA dataset, and compares performance with **BLEU**, **ROUGE-L**, and **BERTScore**. It also explores the effect of using different language models and different RAG pipeline designs.

## Highlights

- Built a domain-specific QA system on top of NCKU Wikipedia
- Constructed a custom evaluation dataset with 8 question-answer pairs
- Compared answer quality **with and without RAG**
- Evaluated results using **BLEU**, **ROUGE-L**, and **BERTScore**
- Compared different language models
- Compared different RAG pipeline variants such as `top_k`, embedding model choice, and reranking

## Project Overview

Large language models can generate fluent answers, but they may hallucinate or produce unsupported facts when responding to domain-specific questions. To address this issue, this project uses a Retrieval-Augmented Generation pipeline:

1. split the source document into chunks
2. create dense embeddings for the chunks
3. retrieve the most relevant passages for each query
4. provide retrieved context to the language model
5. generate grounded answers based on the retrieved evidence

The goal of this project is to test whether retrieval can improve factual QA performance on a focused knowledge source.

## Knowledge Source

The knowledge base used in this project is based on **NCKU Wikipedia** content.

The document contains information such as:
- university background
- international collaborations
- sustainability achievements
- global rankings
- notable alumni
- academic programs and language of instruction

## Evaluation Dataset

A custom evaluation dataset was created for this project.

### Dataset format
The dataset is stored in `my_dataset.json` and contains **8 question-answer pairs**. Each pair is grounded in the NCKU Wikipedia source document.

### Example questions
- Which international universities have signed dual degree or collaboration agreements with NCKU?
- What special research program does NCKU participate in that is related to Academia Sinica?
- How is NCKU ranked globally according to the QS World University Rankings 2024?
- Which notable architect associated with NCKU designed Taipei 101?

## Method

### 1. Baseline: Without RAG
In the non-RAG setting, the language model answers each question directly without access to retrieved context.

This setup is used as a baseline to observe:
- hallucination behavior
- factual accuracy without external grounding
- the effect of model prior knowledge

### 2. RAG Pipeline
In the RAG setting, the system retrieves relevant passages from the NCKU Wikipedia document before generating an answer.

The core pipeline includes:
- document chunking
- dense retrieval with SentenceTransformer embeddings
- top-k passage selection
- answer generation conditioned on retrieved context

### 3. Additional Comparisons
To further analyze system behavior, this project also compares:
- different language models
- different embedding models
- different retrieval strategies
- reranking-based retrieval refinement

### Evaluation Metrics

The following metrics are used to evaluate generated answers against ground-truth references:

- **BLEU**
- **ROUGE-L**
- **BERTScore**

These metrics are used to compare:
- without RAG vs. with RAG
- different model choices
- different RAG pipeline configurations

## Results

### 1. Without RAG vs. With RAG
The project shows that RAG can significantly improve answer quality by grounding generation in retrieved evidence.

In particular, the report shows a clear improvement for Gemma under the RAG setting:
- BLEU: `0.0179 -> 0.1005`
- ROUGE-L: `0.1393 -> 0.4409`
- BERTScore F1: `0.8343 -> 0.8996`

This suggests that retrieval helps reduce hallucination and improves factual alignment.

### 2. Different Model Comparison
This project also compares different language models under the same task setting.

The report discusses:
- **Gemma-2B-it**
- **Qwen-0.5B-Instruct**

The comparison highlights that models with stronger factual prior knowledge may gain less from retrieval, while models with weaker grounding may benefit more from RAG.

### 3. Different RAG Pipeline Comparison
Several RAG variants were tested beyond the baseline pipeline:

- baseline RAG
- `top_k = 5`
- `all-mpnet-base-v2` as embedding model
- two-stage retrieval with reranking

This analysis shows that RAG quality is sensitive to pipeline design, and that simply retrieving more passages does not always improve final QA quality.

## Repository Structure

```text
.
├── README.md
├── LLM_RAGdemo.ipynb
├── report.pdf
├── my_dataset.json
├── NCKU_wiki.pdf            
├── requirements.txt
└── .gitignore
````

## Files

### `LLM_RAGdemo.ipynb`

Main notebook containing:

* model loading
* embedding model setup
* document preprocessing
* chunking
* retrieval
* prompt construction
* answer generation
* evaluation

### `report.pdf`

Final report describing:

* dataset design
* no-RAG results
* RAG results
* model comparison
* pipeline comparison
* observations and conclusions

### `my_dataset.json`

Custom QA benchmark used for evaluation.

### `NCKU_wiki.pdf`

Source knowledge document used for retrieval.
If redistribution is restricted, this file can be omitted from the public repository and described in the README instead.

## How to Run

## 1. Install dependencies

```bash
pip install transformers sentence-transformers torch numpy pandas scikit-learn nltk rouge-score bert-score jupyter
```

## 2. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
LLM_RAGdemo.ipynb
```

## 3. Prepare required files

Make sure the following files are available in the working directory:

* `my_dataset.json`
* knowledge source file such as `NCKU_wiki.pdf` or the original source document
* notebook dependencies

## 4. Run the pipeline

Run the notebook cells in order to:

* load the source document
* split it into chunks
* create embeddings
* retrieve relevant passages
* generate answers
* evaluate outputs

## Key Takeaways

* RAG improves factual QA performance on a focused knowledge source
* retrieval quality and grounding fidelity matter more than simply increasing model size
* pipeline design choices such as chunking, embedding model selection, and reranking can strongly affect final performance
* evaluating with both qualitative observations and automatic metrics gives a clearer view of system behavior
