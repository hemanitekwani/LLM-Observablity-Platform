""" 1. Retrieval Metrics - These metrics measure the effectiveness of the embedding model \
and vector database in finding the right information.
Context Precision: Assesses whether the relevant documents are ranked higher in the retrieved context. A higher score means less 
noise.
Context Recall: Measures if the retrieval system fetched all the necessary information 
required to address the user's query.
"""

"""2. Generation Metrics - These metrics evaluate the quality of the LLM-generated response based on the retrieved context.
      Faithfulness: Measures the factual consistency of the generated answer against the retrieved context. It checks for hallucinations 
      by ensuring that all claims made in the answer can be directly inferred from the context.Answer Relevancy: Evaluates how well the generated 
      text directly addresses the user's question, penalizing redundant or irrelevant information"""


import os

import pandas as pd
from datasets import Dataset
from ragas import evaluate

from ragas.metrics import faithfulness , answer_relevancy , context_precision,context_recall
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_path = os.path.join(root_dir, '.env')

load_dotenv(dotenv_path=env_path)

groq_api_key = os.getenv("GROQ_KEY")

if not groq_api_key:
    raise ValueError("GROQ_KEY missing")

groq_llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    max_retries=5
)

ragas_llm = LangchainLLMWrapper(groq_llm)

hf_embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

ragas_embedding = LangchainEmbeddingsWrapper(
    hf_embedding
)
rate_limit_config = RunConfig(timeout=60, max_retries=10, max_wait=60, max_workers=1)


def evaluate_single(question , answer , contexts , ground_truth):

    print(f"\n Answer:")

    print(f"{answer}")

    data = {
        "question":[question],
        "answer":[answer],
        "contexts":[contexts],
        "ground_truth":[ground_truth]
    }

    dataset = Dataset.from_dict(data)


    try:
        result = evaluate(
        dataset=dataset,
        metrics = [faithfulness , answer_relevancy , context_precision , context_recall],
        llm=ragas_llm,
        embeddings=ragas_embedding,
        run_config=rate_limit_config
        )

    except Exception as e:
        print(f"Ragas Error: {e}")
        return None
    

    return result


