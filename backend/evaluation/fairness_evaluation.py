"""
fairness_evaluation.py
======================
ResumeMatch AI: Fairness & Bias Risk Evaluation

This script evaluates whether removing personal identifiers (PII) alters matching
scores and candidate rankings:
1. Selects 30 representative resumes across diverse job categories.
2. Constructs realistic unredacted resumes with personal identifiers (names, emails,
   phone numbers, addresses, and gendered honorifics).
3. Constructs blind/anonymized copies where personal identifiers are systematically
   redacted, leaving technical skills, work history, and qualifications intact.
4. Evaluates both sets against the same job posting using identical scoring parameters.
5. Computes diagnostic metrics:
   - Original vs. Anonymized Match Scores
   - Absolute Score Differences (|Original - Anonymized|)
   - Candidate Rank Changes (Rank Shift)
   - Percentage of candidates whose ranking changed
   - Average and Maximum score differences
   - Spearman Rank Correlation (rho)
6. Generates publication-ready visualizations:
   - fairness_score_comparison.png
   - fairness_score_diff_distribution.png
7. Saves detailed results to CSV in backend/evaluation/evaluation_results/.

IMPORTANT METHODOLOGICAL NOTE:
This experiment is an initial bias-risk diagnostic (measuring the model's sensitivity
to non-job-related PII noise). A small score difference does NOT constitute mathematical
proof that a system is entirely unbiased or fair across all demographic groups.
"""

import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

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


def anonymize_resume_pii(text: str) -> str:
    """
    Simulates a blind-matching preprocessor that redacts personal identifiers
    while preserving all technical skills, job responsibilities, and qualifications.
    """
    if not isinstance(text, str):
        return ""

    # 1. Mask Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    # 2. Mask Phone numbers
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", " ", text)
    # 3. Mask URLs and profiles
    text = re.sub(r"https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+", " ", text)
    # 4. Mask Gendered honorifics / titles
    text = re.sub(r"\b(mr|ms|mrs|miss|dr|sir|madam)\.?\b", " ", text, flags=re.IGNORECASE)
    # 5. Mask Candidate Name header pattern (e.g. 'Name: John Doe' or 'Candidate: Jane Smith')
    text = re.sub(r"\b(candidate|applicant|name)\s*:\s*[A-Za-z\s]+(?=\n|\r|\.|\s{2})", " ", text, flags=re.IGNORECASE)
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_fairness_evaluation():
    print("=" * 80)
    print("RESUMEMATCH AI - FAIRNESS & BIAS RISK EVALUATION")
    print("=" * 80)

    resumes_path = BACKEND_DIR / "data" / "processed" / "cleaned_resumes.csv"
    jobs_path = BACKEND_DIR / "data" / "processed" / "cleaned_jobs.csv"

    resumes_df = pd.read_csv(resumes_path)
    jobs_df = pd.read_csv(jobs_path)

    # Select 30 diverse resumes
    sample_resumes = resumes_df.sample(n=30, random_state=42).copy().reset_index(drop=True)
    
    # Select a target job for evaluation (Python Developer / Machine Learning)
    target_job = jobs_df[
        jobs_df["Cleaned_Job_Title"].str.contains("Python", case=False, na=False)
    ].iloc[0]

    job_title = target_job.get("Cleaned_Job_Title", target_job.get("Job Title"))
    print(f"Target Evaluation Job : {job_title}")
    print(f"Resumes Evaluated     : {len(sample_resumes)} resumes across {sample_resumes['Category'].nunique()} categories")

    # 1. Prepare Unredacted (Original with PII) and Anonymized (Blind) versions
    # To rigorously simulate real-world unredacted resumes, we append typical header PII
    # (name, email, phone, location) to the raw text, while keeping all skill and experience content identical.
    synthetic_names = [
        "Alex Morgan", "Priya Sharma", "David Chen", "Fatima Al-Mansoor", "Carlos Rodriguez",
        "Ananya Patel", "Liam O'Connor", "Mei Ling", "Kwame Mensah", "Sarah Jenkins",
        "Aarav Gupta", "Elena Rostova", "Tariq Mahmood", "Chloe Dubois", "Hiroshi Tanaka",
        "Zara Khan", "Lucas Silva", "Amara Okafor", "Benjamin Taylor", "Sunita Rao",
        "Matteo Rossi", "Aisha Bello", "Gabriel Santos", "Deepak Verma", "Grace Kim",
        "Ahmed Hassan", "Sophie Martin", "Rohan Mehta", "Olga Ivanova", "Daniel Jackson"
    ]

    original_records = []
    anonymized_records = []

    for i, (_, row) in enumerate(sample_resumes.iterrows()):
        r_id = row["Resume_Id"]
        cat = row["Category"]
        clean_text = row["Cleaned_Resume"]
        cand_name = synthetic_names[i % len(synthetic_names)]
        cand_email = f"{cand_name.lower().replace(' ', '.')}@example.com"
        cand_phone = f"+91-98765{i:05d}"
        
        # Original text has header PII
        orig_text = f"Candidate Name: {cand_name} Email: {cand_email} Phone: {cand_phone} Location: Regional Hub. {clean_text}"
        
        # Anonymized text applies blind-matching redaction
        anon_text = anonymize_resume_pii(orig_text)

        original_records.append({
            "Resume_Id": r_id,
            "Category": cat,
            "Cleaned_Resume": orig_text,
            "Candidate_Name": cand_name,
        })
        anonymized_records.append({
            "Resume_Id": r_id,
            "Category": cat,
            "Cleaned_Resume": anon_text,
            "Candidate_Name": "[NAME REDACTED]",
        })

    orig_df = pd.DataFrame(original_records)
    anon_df = pd.DataFrame(anonymized_records)

    # 2. Run Matching Service on both sets
    service = MatchingService()

    orig_ranked = service.rank_candidates(target_job, orig_df)
    anon_ranked = service.rank_candidates(target_job, anon_df)

    # Rename columns for clarity
    orig_ranked = orig_ranked.rename(columns={
        "Rank": "Original_Rank",
        "Overall_Score": "Original_Score",
        "Semantic_Score": "Original_Semantic",
        "Skill_Score": "Original_Skills",
    })
    anon_ranked = anon_ranked.rename(columns={
        "Rank": "Anonymized_Rank",
        "Overall_Score": "Anonymized_Score",
        "Semantic_Score": "Anonymized_Semantic",
        "Skill_Score": "Anonymized_Skills",
    })

    # Merge on Resume_Id
    comparison = pd.merge(
        orig_ranked[["Resume_Id", "Category", "Original_Rank", "Original_Score", "Original_Semantic", "Original_Skills"]],
        anon_ranked[["Resume_Id", "Anonymized_Rank", "Anonymized_Score", "Anonymized_Semantic", "Anonymized_Skills"]],
        on="Resume_Id",
    )

    # Compute diagnostics
    comparison["Score_Difference"] = (comparison["Original_Score"] - comparison["Anonymized_Score"]).round(2)
    comparison["Abs_Score_Difference"] = comparison["Score_Difference"].abs()
    comparison["Rank_Shift"] = (comparison["Original_Rank"] - comparison["Anonymized_Rank"]).abs()
    comparison["Rank_Changed"] = comparison["Rank_Shift"] > 0

    # Sort by Original_Rank
    comparison = comparison.sort_values(by="Original_Rank").reset_index(drop=True)

    # Save detailed CSV
    comp_csv_path = RESULTS_DIR / "fairness_bias_comparison.csv"
    comparison.to_csv(comp_csv_path, index=False)
    print(f"\n[Saved] Detailed fairness comparison saved to: {comp_csv_path}")

    # Summary Statistics
    total_candidates = len(comparison)
    rank_changed_count = comparison["Rank_Changed"].sum()
    pct_rank_changed = (rank_changed_count / total_candidates) * 100.0
    mean_abs_diff = comparison["Abs_Score_Difference"].mean()
    max_abs_diff = comparison["Abs_Score_Difference"].max()
    spearman_corr, _ = spearmanr(comparison["Original_Score"], comparison["Anonymized_Score"])

    summary_metrics = {
        "Metric": [
            "Total Candidates Evaluated",
            "Average Match Score (Original)",
            "Average Match Score (Anonymized)",
            "Mean Absolute Score Difference",
            "Maximum Score Difference",
            "Candidates with Rank Shift (Count)",
            "Candidates with Rank Shift (%)",
            "Average Rank Shift (Positions)",
            "Spearman Rank Correlation",
        ],
        "Value": [
            f"{total_candidates}",
            f"{comparison['Original_Score'].mean():.2f}%",
            f"{comparison['Anonymized_Score'].mean():.2f}%",
            f"{mean_abs_diff:.2f}%",
            f"{max_abs_diff:.2f}%",
            f"{rank_changed_count} / {total_candidates}",
            f"{pct_rank_changed:.1f}%",
            f"{comparison['Rank_Shift'].mean():.2f}",
            f"{spearman_corr:.4f}",
        ],
    }
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv_path = RESULTS_DIR / "fairness_summary_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 80)
    print("FAIRNESS & BIAS EVALUATION SUMMARY")
    print("=" * 80)
    for _, r in summary_df.iterrows():
        print(f"  - {r['Metric']:<36}: {r['Value']}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    # Plot 1: Scatter plot Original vs Anonymized Scores
    plt.figure(figsize=(7, 6))
    plt.scatter(
        comparison["Original_Score"],
        comparison["Anonymized_Score"],
        color="#3730a3",
        edgecolors="black",
        s=70,
        alpha=0.8,
        label="Candidate Resumes",
    )
    # Perfect parity line (y = x)
    min_val = min(comparison["Original_Score"].min(), comparison["Anonymized_Score"].min()) - 5
    max_val = max(comparison["Original_Score"].max(), comparison["Anonymized_Score"].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], color="#dc2626", linestyle="--", linewidth=1.5, label="Parity Line (y = x)")

    plt.title("Fairness Diagnostic: Original vs. Anonymized Match Scores", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Original Resume Match Score (%)", fontsize=10)
    plt.ylabel("Anonymized (Blind) Resume Match Score (%)", fontsize=10)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plot1_path = RESULTS_DIR / "fairness_score_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Original vs Anonymized Scatter: {plot1_path}")

    # Plot 2: Distribution of Score Differences
    plt.figure(figsize=(7, 4.5))
    sns.histplot(comparison["Abs_Score_Difference"], bins=10, kde=True, color="#16a34a", edgecolor="black")
    plt.axvline(mean_abs_diff, color="#dc2626", linestyle="--", label=f"Mean Abs Diff ({mean_abs_diff:.2f}%)")
    plt.title("Distribution of Absolute Score Differences (|Original - Anonymized|)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Absolute Score Difference (%)", fontsize=10)
    plt.ylabel("Candidate Count", fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot2_path = RESULTS_DIR / "fairness_score_diff_distribution.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Score Difference Distribution: {plot2_path}")


if __name__ == "__main__":
    run_fairness_evaluation()
