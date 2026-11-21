"""
ranking_evaluation.py
=====================
ResumeMatch AI: Candidate Ranking Quality Evaluation

This script evaluates how effectively the system ranks relevant candidates near the top:
1. Selects 5 diverse job queries (Data Scientist, Java Developer, Python Developer, DevOps, HR).
2. Evaluates each job against the entire pool of 166 candidate resumes.
3. Computes multi-component match scores (Semantic, Skills, Experience, Education, Overall).
4. Sorts and ranks candidates from highest to lowest match score.
5. Evaluates Information Retrieval (IR) ranking metrics:
   - Precision@5 (P@5)
   - Precision@10 (P@10)
   - NDCG@5 (Normalized Discounted Cumulative Gain at 5)
   - NDCG@10 (Normalized Discounted Cumulative Gain at 10)
   - Mean Reciprocal Rank (MRR)
6. Compares the ResumeMatch AI composite model against a Simple Keyword-Overlap Baseline.
7. Generates publication-ready visualizations and CSV tables.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.services.matching_service import MatchingService
except ImportError:
    from services.matching_service import MatchingService

RESULTS_DIR = CURRENT_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_dcg(relevance_scores: list, k: int) -> float:
    """
    Computes Discounted Cumulative Gain at rank K.
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    """
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        rel = relevance_scores[i]
        # Rank index is i + 1, so denominator is log2((i+1) + 1) = log2(i + 2)
        dcg += rel / np.log2(i + 2)
    return dcg


def calculate_ndcg(relevance_scores: list, k: int, total_relevant: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K.
    NDCG@K = DCG@K / IDCG@K
    """
    dcg = calculate_dcg(relevance_scores, k)
    # Ideal DCG: top ranks all have rel = 1
    ideal_relevance = [1] * min(k, total_relevant)
    idcg = calculate_dcg(ideal_relevance, k)
    if idcg == 0.0:
        return 0.0
    return round(float(dcg / idcg), 4)


def calculate_reciprocal_rank(relevance_scores: list) -> float:
    """
    Computes Reciprocal Rank (1 / rank of first relevant item).
    """
    for i, rel in enumerate(relevance_scores):
        if rel == 1:
            return round(1.0 / (i + 1), 4)
    return 0.0


def evaluate_ranking_for_query(
    job_row: pd.Series,
    target_category: str,
    resumes_df: pd.DataFrame,
    service: MatchingService,
) -> dict:
    """
    Ranks candidates for a single job query using both:
    1. ResumeMatch AI Multi-component Model
    2. Simple Keyword Overlap Baseline
    Computes P@5, P@10, NDCG@5, NDCG@10, and RR for both methods.
    """
    # Total relevant resumes in the dataset for this category
    total_relevant = (resumes_df["Category"].str.lower() == target_category.lower()).sum()

    # -------------------------------------------------------------------------
    # Method A: ResumeMatch AI Model
    # -------------------------------------------------------------------------
    model_ranked = service.rank_candidates(job_row, resumes_df)
    # Binary relevance: candidate category matches target domain
    model_rel = [
        1 if cat.lower() == target_category.lower() else 0
        for cat in model_ranked["Category"]
    ]

    model_p5 = round(sum(model_rel[:5]) / 5.0, 4)
    model_p10 = round(sum(model_rel[:10]) / 10.0, 4)
    model_ndcg5 = calculate_ndcg(model_rel, 5, total_relevant)
    model_ndcg10 = calculate_ndcg(model_rel, 10, total_relevant)
    model_rr = calculate_reciprocal_rank(model_rel)

    # -------------------------------------------------------------------------
    # Method B: Keyword Overlap Baseline
    # -------------------------------------------------------------------------
    base_ranked = service.rank_candidates_keyword_baseline(job_row, resumes_df)
    base_rel = [
        1 if cat.lower() == target_category.lower() else 0
        for cat in base_ranked["Category"]
    ]

    base_p5 = round(sum(base_rel[:5]) / 5.0, 4)
    base_p10 = round(sum(base_rel[:10]) / 10.0, 4)
    base_ndcg5 = calculate_ndcg(base_rel, 5, total_relevant)
    base_ndcg10 = calculate_ndcg(base_rel, 10, total_relevant)
    base_rr = calculate_reciprocal_rank(base_rel)

    return {
        "Job_Title": job_row.get("Cleaned_Job_Title", job_row.get("Job Title")),
        "Target_Domain": target_category,
        "Total_Pool_Size": len(resumes_df),
        "Total_Relevant_In_Pool": int(total_relevant),
        # Model metrics
        "Model_P@5": model_p5,
        "Model_P@10": model_p10,
        "Model_NDCG@5": model_ndcg5,
        "Model_NDCG@10": model_ndcg10,
        "Model_MRR": model_rr,
        # Baseline metrics
        "Baseline_P@5": base_p5,
        "Baseline_P@10": base_p10,
        "Baseline_NDCG@5": base_ndcg5,
        "Baseline_NDCG@10": base_ndcg10,
        "Baseline_MRR": base_rr,
        "sample_ranked_df": model_ranked,
    }


def run_ranking_evaluation():
    print("=" * 80)
    print("RESUMEMATCH AI - CANDIDATE RANKING QUALITY EVALUATION")
    print("=" * 80)

    resumes_path = BACKEND_DIR / "data" / "processed" / "cleaned_resumes.csv"
    jobs_path = BACKEND_DIR / "data" / "processed" / "cleaned_jobs.csv"

    resumes_df = pd.read_csv(resumes_path)
    jobs_df = pd.read_csv(jobs_path)

    print(f"Loaded {len(resumes_df)} resumes and {len(jobs_df)} jobs.")

    service = MatchingService()

    # Define 5 diverse test queries
    test_queries = [
        {"domain": "Data Science", "title_search": "Data Scientist", "alt_title": "Machine Learning"},
        {"domain": "Java Developer", "title_search": "Java Developer", "alt_title": "Java"},
        {"domain": "Python Developer", "title_search": "Python Developer", "alt_title": "Python"},
        {"domain": "DevOps Engineer", "title_search": "DevOps", "alt_title": "Cloud"},
        {"domain": "HR", "title_search": "HR Executive", "alt_title": "Human Resources"},
    ]

    all_query_results = []
    first_query_ranked_df = None

    for q in test_queries:
        dom = q["domain"]
        matched_jobs = jobs_df[
            jobs_df["Cleaned_Job_Title"].str.contains(q["title_search"], case=False, na=False)
        ]
        if len(matched_jobs) == 0:
            matched_jobs = jobs_df[
                jobs_df["Cleaned_Job_Title"].str.contains(q["alt_title"], case=False, na=False)
            ]
        selected_job = matched_jobs.iloc[0]

        q_res = evaluate_ranking_for_query(selected_job, dom, resumes_df, service)
        
        if first_query_ranked_df is None:
            first_query_ranked_df = q_res["sample_ranked_df"].copy()
            first_query_ranked_df.insert(0, "Target_Job", q_res["Job_Title"])

        # Omit DataFrame from tabular metrics
        del q_res["sample_ranked_df"]
        all_query_results.append(q_res)

    results_df = pd.DataFrame(all_query_results)

    # Compute Macro-Averages
    avg_row = {
        "Job_Title": "MACRO-AVERAGE",
        "Target_Domain": "ALL 5 DOMAINS",
        "Total_Pool_Size": len(resumes_df),
        "Total_Relevant_In_Pool": int(results_df["Total_Relevant_In_Pool"].mean()),
        "Model_P@5": round(results_df["Model_P@5"].mean(), 4),
        "Model_P@10": round(results_df["Model_P@10"].mean(), 4),
        "Model_NDCG@5": round(results_df["Model_NDCG@5"].mean(), 4),
        "Model_NDCG@10": round(results_df["Model_NDCG@10"].mean(), 4),
        "Model_MRR": round(results_df["Model_MRR"].mean(), 4),
        "Baseline_P@5": round(results_df["Baseline_P@5"].mean(), 4),
        "Baseline_P@10": round(results_df["Baseline_P@10"].mean(), 4),
        "Baseline_NDCG@5": round(results_df["Baseline_NDCG@5"].mean(), 4),
        "Baseline_NDCG@10": round(results_df["Baseline_NDCG@10"].mean(), 4),
        "Baseline_MRR": round(results_df["Baseline_MRR"].mean(), 4),
    }

    results_with_avg = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

    metrics_csv_path = RESULTS_DIR / "ranking_quality_metrics.csv"
    results_with_avg.to_csv(metrics_csv_path, index=False)
    print(f"\n[Saved] Ranking metrics table saved to: {metrics_csv_path}")

    # Save Top 10 candidate rankings sample
    sample_csv_path = RESULTS_DIR / "candidate_rankings_sample.csv"
    first_query_ranked_df.head(10).to_csv(sample_csv_path, index=False)
    print(f"[Saved] Sample candidate rankings table (Top 10) saved to: {sample_csv_path}")

    # Display Macro Results
    print("\n" + "=" * 80)
    print("RANKING QUALITY EVALUATION SUMMARY (SYSTEM VS BASELINE)")
    print("=" * 80)
    print(f"Metric       | ResumeMatch AI Model | Simple Keyword Baseline | Absolute Gain")
    print("-" * 75)
    print(f"Precision@5  | {avg_row['Model_P@5']*100:>18.2f}% | {avg_row['Baseline_P@5']*100:>21.2f}% | {(avg_row['Model_P@5']-avg_row['Baseline_P@5'])*100:>+12.2f}%")
    print(f"Precision@10 | {avg_row['Model_P@10']*100:>18.2f}% | {avg_row['Baseline_P@10']*100:>21.2f}% | {(avg_row['Model_P@10']-avg_row['Baseline_P@10'])*100:>+12.2f}%")
    print(f"NDCG@5       | {avg_row['Model_NDCG@5']:>19.4f}  | {avg_row['Baseline_NDCG@5']:>22.4f}  | {avg_row['Model_NDCG@5']-avg_row['Baseline_NDCG@5']:>+13.4f}")
    print(f"NDCG@10      | {avg_row['Model_NDCG@10']:>19.4f}  | {avg_row['Baseline_NDCG@10']:>22.4f}  | {avg_row['Model_NDCG@10']-avg_row['Baseline_NDCG@10']:>+13.4f}")
    print(f"MRR          | {avg_row['Model_MRR']:>19.4f}  | {avg_row['Baseline_MRR']:>22.4f}  | {avg_row['Model_MRR']-avg_row['Baseline_MRR']:>+13.4f}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    # Plot 1: Comparison of Model vs Baseline on Macro Metrics
    metrics_names = ["Precision@5", "Precision@10", "NDCG@5", "NDCG@10", "MRR"]
    model_vals = [
        avg_row["Model_P@5"],
        avg_row["Model_P@10"],
        avg_row["Model_NDCG@5"],
        avg_row["Model_NDCG@10"],
        avg_row["Model_MRR"],
    ]
    base_vals = [
        avg_row["Baseline_P@5"],
        avg_row["Baseline_P@10"],
        avg_row["Baseline_NDCG@5"],
        avg_row["Baseline_NDCG@10"],
        avg_row["Baseline_MRR"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width / 2, model_vals, width, label="ResumeMatch AI (Composite)", color="#3730a3")
    plt.bar(x + width / 2, base_vals, width, label="Keyword-Overlap Baseline", color="#94a3b8")

    plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
    plt.title("Candidate Ranking Quality: ResumeMatch AI vs. Keyword Baseline", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, metrics_names, fontsize=10, fontweight="semibold")
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add data value labels on top of bars
    for i in range(len(x)):
        plt.text(x[i] - width / 2, model_vals[i] + 0.02, f"{model_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold", color="#3730a3")
        plt.text(x[i] + width / 2, base_vals[i] + 0.02, f"{base_vals[i]:.2f}", ha="center", fontsize=9, color="#475569")

    plt.tight_layout()
    chart1_path = RESULTS_DIR / "ranking_system_vs_baseline.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Ranking System vs Baseline Chart: {chart1_path}")

    # Plot 2: Per-Domain NDCG Comparison
    domains = results_df["Target_Domain"].tolist()
    m_ndcg5 = results_df["Model_NDCG@5"].tolist()
    b_ndcg5 = results_df["Baseline_NDCG@5"].tolist()

    x2 = np.arange(len(domains))
    plt.figure(figsize=(9, 5))
    plt.plot(x2, m_ndcg5, marker="o", linewidth=2.5, color="#16a34a", label="ResumeMatch AI (NDCG@5)")
    plt.plot(x2, b_ndcg5, marker="s", linewidth=2.0, linestyle="--", color="#dc2626", label="Baseline (NDCG@5)")
    plt.xticks(x2, domains, fontsize=10)
    plt.ylabel("NDCG@5 Score", fontsize=11)
    plt.ylim(0, 1.1)
    plt.title("Ranking Quality (NDCG@5) across Evaluation Job Domains", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart2_path = RESULTS_DIR / "ranking_metrics_ndcg_precision.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Domain NDCG Breakdown Chart: {chart2_path}")


if __name__ == "__main__":
    run_ranking_evaluation()
