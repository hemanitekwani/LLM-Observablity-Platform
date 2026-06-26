<div align="center">

# 🚀 RAG Observability & Evaluation Platform

### framework for evaluating, benchmarking, and monitoring Retrieval-Augmented Generation (RAG) systems.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-success)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

# 📌 Overview

Large Language Models are only as reliable as their retrieval pipeline. Traditional RAG systems usually stop after generating an answer. This project focuses on **evaluating and observing** RAG systems.

This platform answers critical engineering questions:
* **How faithful are the generated responses?**
* **Which configuration performs best?**
* **Which prompts hallucinate?**
* **Which retrieval strategy is better?**
* **How does CRAG compare against vanilla RAG?**

The platform automatically evaluates every experiment, stores metrics in MLflow, and visualizes performance through an interactive Streamlit dashboard.

---

# 🧪 Custom Metric: Adjusted Faithfulness

Standard evaluation metrics frequently penalize language models for "safe refusals"—scenarios where the model correctly reports that information is missing from the provided context. To resolve this, the Adjusted Faithfulness metric was developed to distinguish between valid safety constraints and factual hallucinations.

### The Formula
Let $F_{raw}$ be the raw faithfulness score from RAGAS, and $R$ be the set of "Refusal" responses (e.g., *"I cannot determine the answer..."*).

$$ \text{Faithfulness}_{adjusted} = 
\begin{cases} 
1.0, & \text{if Answer} \in R \\ 
F_{raw}, & \text{otherwise} 
\end{cases} $$

By normalizing safe refusals to a score of **1.0**, prevented the evaluation from penalizing intelligent models that correctly identify missing context.

---

# ✨ Features

## ✅ Multiple Retrieval Configurations
Supports experimentation with different retrieval strategies such as:
- **Vanilla RAG**
- **CRAG (Corrective RAG)**
- **Hyperparameter Tuning:** Different Chunk Sizes, Top-K, and Reranker settings.

## ✅ Automated RAG Evaluation
Every run is automatically evaluated using **RAGAS** metrics. Metrics include:
- Faithfulness & Adjusted Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Latency

## ✅ MLflow Experiment Tracking
Every experiment stores:
- Hyperparameters
- Evaluation Metrics
- Generated Answers & Artifacts
- Evaluation CSVs
*Allowing for complete experiment reproducibility.*

## ✅ Interactive Observability Dashboard
Built using **Streamlit**, providing:
- Experiment Leaderboard & Hyperparameter Inspection
- Hallucination Audit Logs
- Cross-run comparison charts
- Domain-specific performance analysis

## ✅ Hallucination Detection
The platform distinguishes between:
1. **Correct refusal:** "I cannot determine the answer from the provided context."
2. **Actual hallucinations:** Unsupported facts generated despite insufficient evidence.

---

# 🏗 System Architecture

```mermaid
flowchart LR
    A[Documents/PDFs] --> B[Chunking]
    B --> C[Vector Database]
    Q[User Question] --> D[Retriever]
    D --> E[CRAG Verification]
    E --> F[Prompt Builder]
    F --> G[LLM]
    G --> H[Generated Answer]
    H --> I[RAGAS Evaluation]
    I --> J[MLflow]
    J --> K[Streamlit Dashboard]


# 🏗 Project Structure

Enterprise-RAG-Observability/
├── rag/              # Core pipeline (retriever, generator)
├── evaluation/       # RAGAS evaluators
├── mlruns/           # MLflow experiment logs
├── app.py            # Streamlit Dashboard
└── requirements.txt

📸 Dashboard Previews








