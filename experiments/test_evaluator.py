import pandas as pd
import os
import sys
import time
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from rag.pipeline import Pipeline
from evaluation.evaluator import evaluate_single


def run_test():
    print("Loading RAG Pipeline")

    pipeline = Pipeline()

    print("Loading Evaluation Dataset...")

    question_df = pd.read_csv("data/evaluation/questions.csv")
    
    single_hop_mix = question_df[question_df["question_type"] == "single_hop"].head(7)
    multi_hop_mix = question_df[question_df["question_type"] == "multi_hop"].head(7)
    adversial_mix = question_df[question_df["question_type"] == "adversarial"].head(7)

    eval_df = pd.concat([single_hop_mix , multi_hop_mix, adversial_mix]).reset_index(drop=True)


    results = []

    eval_questions = len(eval_df)


    for idx , row in eval_df.iterrows():

        question = row["question"]
        ground_truth = row["ground_truth"]
        question_type = row["question_type"]

        print(f"[{idx + 1}/{eval_questions}]"
            f"{question_type.upper()}")
        
        print(f"Q: {question}")

        try:
            start_time = time.time()

            rag_result = pipeline.ask(question)

            latency = time.time() - start_time

            generated_answer = rag_result.get("answer",rag_result.get("result" , ""))

            retrieved_contexts = [doc.page_content for doc in rag_result.get("contexts" ,[])]
            
            
            scores = evaluate_single(question=question , answer=generated_answer,contexts=retrieved_contexts,ground_truth=ground_truth)
            
            time.sleep(2)
            
            print("\nAnswer:")

            print(generated_answer)

            print("\nScores")
            print(scores)

            results.append({
                "question_type":question_type,
                "question":question,

                "final_query_used":rag_result.get("final_query_used", question),
                "retries": rag_result.get("retries" , 0),

                "answer":generated_answer,
                "faithfulness":float(scores["faithfulness"][0]),

                "answer_relevancy":float(scores["answer_relevancy"][0]),

                "context_precision":float(scores["context_precision"][0]),

                "context_recall":float(scores["context_recall"][0]),

                "latency":latency
            })

            time.sleep(15)

        except Exception as e:
            print(f"Error {e}")

            results.append({
                "question_type":question_type,
                "question":question,

                "final_query_used":rag_result.get("final_query", question),
                "retries": rag_result.get("retries" , 0),


                "faithfulness":np.nan,
                "answer_relevancy": np.nan,
                "context_precision":np.nan,
                "context_recall":np.nan,
                "latency":np.nan
            })

    return pd.DataFrame(results)




if __name__ == "__main__":
    result_df = run_test()

    os.makedirs("results" , exist_ok=True)

    csv_string = result_df.to_csv(index=False)

    # result_df.to_csv("results/evaluation_results_v2.csv",index=False)
    


    import mlflow


    mlflow.set_experiment("RAG EVALUATION")

    with mlflow.start_run(run_name="baseline_v5_rerank_top_3_overlap_150"):

       
        mlflow.log_param("vector_fetch_k", 15)      
        mlflow.log_param("reranker_top_k", 3)        
        mlflow.log_param("reranker_model", "BAAI/bge-reranker-base")
        mlflow.log_param("chunk_size", 500)

        mlflow.log_param("architecture", "Self-Correcting CRAG")
        mlflow.log_param("max_retries", 2)

        mlflow.log_param("system_prompt", """
        You are a strict data extraction audit system.
        Your task is to answer the user's question using ONLY the explicitly stated facts in the provided context.

        CRITICAL CONSTRAINTS:
        1. ENTITY VERIFICATION: Before answering, verify if the exact unique entities, subjects, proper nouns, specific versions, or terminology mentioned in the question are explicitly present and active in the provided context.
        2. NO CROSS-APPLICATION: If the context discusses a different version, a related but distinct subject, or lacks the specific facts requested, you must not infer, cross-apply facts, or substitute information.
        3. STRICT REFUSAL: If the exact subject and answer cannot be explicitly determined from the provided context, you MUST respond exactly with this phrase and nothing else:
        "I cannot determine the answer from the provided context."
        """)

        mlflow.log_param("grader_prompt", """You are a ruthless, zero-tolerance hallucination auditor. 
        Your ONLY job is to verify if the Generated Answer is strictly derived from the Context.
        CRITICAL INSTRUCTIONS:
        1. If the Generated Answer contains EVEN ONE word, number, or claim that is not explicitly written in the Context, you MUST reply NO.
        2. If the Generated Answer relies on outside knowledge, you MUST reply NO.
        3. Only if every single fact in the answer is proven by the context, reply YES.

        Reply ONLY with YES or NO:
        """
        )


        mlflow.log_param("num_question",len(result_df))

        mlflow.log_param("embedding_model","all-MiniLM-L6-v2")

        mlflow.log_metric("faithfulness",result_df["faithfulness"].dropna().mean())
        
        mlflow.log_metric("answer_relevancy",result_df["answer_relevancy"].dropna().mean())
      
        mlflow.log_metric("context_recall",result_df["context_recall"].dropna().mean())

        mlflow.log_metric("context_precision",result_df["context_precision"].dropna().mean())

        mlflow.log_metric("latency", result_df["latency"].dropna().mean())

        mlflow.log_text(csv_string , "evaluation_results.csv")
    
    
    print(result_df[
    ["faithfulness",
     "answer_relevancy",
     "context_precision",
     "context_recall"]
].notna().sum())

    print("\nSaved results")

    metrics = ["faithfulness","answer_relevancy","context_precision","context_recall","latency"]

    print(result_df[metrics].mean().round(4))

    summary = (result_df.groupby("question_type")[metrics].mean().round(4))

    print(summary)

    print("\nCompleted Evaluation")
