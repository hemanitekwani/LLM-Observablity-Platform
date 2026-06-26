import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="RAG MLOps Observability Platform", layout="wide")

st.title("📊 Enterprise RAG Observability & Evaluation Platform")
st.markdown("Dynamic MLflow run tracking, parameter inspection, and cross-experiment comparative analytics.")
st.markdown("---")

st.sidebar.header("📁 MLflow Experiment Scanner")

experiment_dir = st.sidebar.text_input(
    "MLflow Experiment Directory",
    value=r"mlruns\530252935482749885" 
)
def scan_mlflow_runs(exp_dir):
    """Scans the experiment directory and dynamically loads all runs."""
    all_runs = {}
    
    if not os.path.exists(exp_dir):
        return all_runs
        
    for run_id in os.listdir(exp_dir):
        run_path = os.path.join(exp_dir, run_id)
        if not os.path.isdir(run_path) or run_id == "meta.yaml":
            continue
            
        # 1. Get Run Name from tags (or default to run_id)
        run_name = run_id
        name_path = os.path.join(run_path, "tags", "mlflow.runName")
        if os.path.exists(name_path):
            try:
                with open(name_path, "r") as f:
                    run_name = f.read().strip()
            except Exception:
                pass

        metrics = {}
        met_dir = os.path.join(run_path, "metrics")
        if os.path.exists(met_dir):
            for m_file in os.listdir(met_dir):
                try:
                    with open(os.path.join(met_dir, m_file), "r") as f:
                        parts = f.read().strip().split()
                        if len(parts) >= 2:
                            metrics[m_file] = float(parts[1])
                except Exception:
                    pass

        params = {}
        par_dir = os.path.join(run_path, "params")
        if os.path.exists(par_dir):
            for p_file in os.listdir(par_dir):
                try:
                    with open(os.path.join(par_dir, p_file), "r") as f:
                        params[p_file] = f.read().strip()
                except Exception:
                    pass

        df = pd.DataFrame()
        csv_path = os.path.join(run_path, "artifacts", "evaluation_results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "question_type" not in df.columns and "type" in df.columns:
                    df["question_type"] = df["type"]
            except Exception:
                pass

        all_runs[run_name] = {
            "run_id": run_id,
            "metrics": metrics,
            "params": params,
            "data": df
        }
        
    return all_runs


scanned_runs = scan_mlflow_runs(experiment_dir)

if not scanned_runs:
    st.error("No runs found. Please check the MLflow Experiment Directory path in the sidebar.")
    st.stop()

## TABS

tab1, tab2 = st.tabs([
    "🔍 Individual Run Inspector", 
    "📈 Cross-Run Comparison Engine"
])

# TAB 1: INDIVIDUAL RUN INSPECTOR
with tab1:
    st.subheader("Single Run Diagnostics")
    st.markdown("Select a specific experiment run to view its parameters, macro metrics, and hallucination log.")
    
    selected_run_name = st.selectbox("Select Target Run", list(scanned_runs.keys()))
    run_data = scanned_runs[selected_run_name]
    
    col_metrics, col_params = st.columns([2, 1])
    
    df = run_data.get("data",pd.DataFrame())

    adjusted_faithfulness = 0.0
    safety_rate = 0.0
    safe_refusal_text = "I cannot determine the answer from the provided context"

    if not df.empty and "faithfulness" in df.columns:
        df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce")
        df["is_refusal"] = df["answer"].fillna("").str.contains(safe_refusal_text,case=False,na=False)
            
        attempted_df = df[~df["is_refusal"]]

        if not attempted_df.empty:
            adjusted_faithfulness = attempted_df["faithfulness"].mean()

        if "question_type" in df.columns:
            adv_df = df[df["question_type"].str.lower() == "adversarial"]

        if not adv_df.empty:
            safety_rate = (adv_df["is_refusal"].mean() * 100) if not adv_df.empty else 0.0

    
    with col_metrics:
        st.markdown("#### 🎯 Macro Metrics")
        m = run_data["metrics"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Raw Faithfulness", f"{m.get('faithfulness', 0.0):.4f}")
        m2.metric("Adjusted Faithfulness", f"{adjusted_faithfulness:.4f}")
        m3.metric("Safety Rate", f"{safety_rate:.1f}%")

        m4, m5, m6,m7 = st.columns(4)
        m4.metric("Answer Relevancy", f"{m.get('answer_relevancy', 0.0):.4f}")
        m5.metric("Context Precision", f"{m.get('context_precision', 0.0):.4f}")
        m6.metric("Latency", f"{m.get('latency', 0.0):.2f}s")
        m7.metric("Context Recall", f"{m.get('context_recall', 0.0):.4f}")
    
    with col_params:
        st.markdown("#### ⚙️ Hyperparameters")
        if run_data["params"]:
            st.json(run_data["params"])
        else:
            st.info("No parameters logged for this run.")

    st.markdown("---")
    st.markdown("#### 🚨 Hallucination Audit Log")
    df = run_data["data"]


    if not df.empty and "faithfulness" in df.columns:
        c1 , c2 = st.columns(2)

        c1.metric("Adjusted Faithfulness (Attempted Answer)" , f"{adjusted_faithfulness:.4f}")

        c2.metric("Adversarial Safety Rate", f"{safety_rate:.1f}%")
        
        st.write("")
        
        failed_df = df[(df["faithfulness"] < 0.85) & (~df["is_refusal"])]

        safeguarded_df = df[df["is_refusal"]]

        if not safeguarded_df.empty:
            st.success(f"🛡️ **Guardrail Active:** Successfully blocked {len(safeguarded_df)} out-of-scope questions.")
            
        if failed_df.empty:
            st.success("No critical hallucinations detected in this run.")
        else:
            st.error(f"⚠️ Detected {len(failed_df)} true hallucinations (Low Faithfulness & Unsafe).")

            fail_q = st.selectbox("Select True Hallucination to Audit", failed_df["question"].unique())

            fail_row = failed_df[failed_df["question"] == fail_q].iloc[0]
                
            st.text_area("User Question", value=fail_row["question"], disabled=True)

            final_query = fail_row.get("final_query_used", fail_row["question"])

            if final_query != fail_row["question"]:
                st.info(f"🔄 **Agentic Rewrite Triggered:** The system rewrote the search query to: `{final_query}`")

            st.text_area("Generated Answer", value=fail_row.get("answer", ""), height=100, disabled=True)


    else:
        st.info("No CSV artifact found for this run.")



# TAB 2: CROSS-RUN COMPARISON ENGINE


with tab2:
    st.subheader("Multi-Experiment Comparative Analysis")
    st.markdown("Select multiple runs to benchmark system optimizations against each other.")
    
    runs_to_compare = st.multiselect(
        "Select Runs to Compare", 
        list(scanned_runs.keys()), 
        default=list(scanned_runs.keys())[:2] if len(scanned_runs) >= 2 else list(scanned_runs.keys())
    )
    
    if runs_to_compare:
        leaderboard = []
        combined_dfs = []

        metrics_to_plot = ["faithfulness", "adjusted_faithfulness", "answer_relevancy", "context_precision", "context_recall", "latency"]
        safe_text = "I cannot determine the answer from the provided context"

        for r_name in runs_to_compare:
            metrics = scanned_runs[r_name]["metrics"]
            params = scanned_runs[r_name]["params"]
            df = scanned_runs[r_name].get("data", pd.DataFrame()).copy()

            if not df.empty:
               
                df["adjusted_faithfulness"] = df.apply(
                    lambda row: 1.0 if safe_text.lower() in str(row.get("answer", "")).lower() 
                    else pd.to_numeric(row.get("faithfulness", 0.0), errors="coerce"), axis=1
                )
                
                for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "latency"]:
                    if m in df.columns:
                        df[m] = pd.to_numeric(df[m], errors="coerce")

                leaderboard.append({
                    "Run Name": r_name,
                    "Raw Faithfulness": df["faithfulness"].mean(),
                    "Adj. Faithfulness": df["adjusted_faithfulness"].mean(),
                    "Answer Relevancy": df["answer_relevancy"].mean(),
                    "Context Precision": df["context_precision"].mean(),
                    "Context Recall": df["context_recall"].mean(),
                    "Latency (s)": df["latency"].mean(),
                    "Chunk Size": params.get("chunk_size", "N/A"),
                    "vector_Top_K": params.get("vector_fetch_k", params.get("top_k", "NAN")),
                    "reranker_top_k": params.get("reranker_top_k", "None"),
                })
                
                df["Run Name"] = r_name
                combined_dfs.append(df)

        # DISPLAY LEADERBOARD
        if leaderboard:
            leaderboard_df = pd.DataFrame(leaderboard).set_index("Run Name")
            st.dataframe(leaderboard_df.style.highlight_max(
                axis=0, 
                subset=["Raw Faithfulness", "Adj. Faithfulness", "Answer Relevancy"], 
                color="#eafaf1"
            ).format(precision=4), use_container_width=True)

        # DISPLAY SUBSPACE PLOTS
        st.markdown("#### Subspace Performance Benchmarks")
        if combined_dfs:
            combined_data = pd.concat(combined_dfs)
            subspace_summary = combined_data.groupby(["Run Name", "question_type"]).mean(numeric_only=True).reset_index()
            
            cols = st.columns(2)
            for i, metric in enumerate(metrics_to_plot):
                title = metric.replace('_', ' ').title()
                fig = px.bar(
                    subspace_summary, 
                    x="question_type", 
                    y=metric, 
                    color="Run Name", 
                    barmode="group", 
                    title=f"{title} by Domain", 
                    template="plotly_white"
                )
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough row-level artifact data to generate subspace charts.")