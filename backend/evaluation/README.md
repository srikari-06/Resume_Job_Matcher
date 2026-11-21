# ResumeMatch AI: Evaluation Suite

This directory contains the experimental evaluation framework for **ResumeMatch AI: Smart Resume Matching**. It provides automated scripts to rigorously evaluate model performance, candidate ranking quality, and algorithmic fairness/bias risk using genuine experimental data.

---

## Evaluation Modules

```
backend/evaluation/
├── model_evaluation.py            # Evaluates precision, recall, F1, accuracy, and confusion matrix
├── ranking_evaluation.py          # Evaluates P@5, P@10, NDCG@5, NDCG@10, MRR vs. Keyword Baseline
├── fairness_evaluation.py         # Evaluates sensitivity to non-job-related personal identifiers (PII)
├── README.md                      # Guide to the evaluation suite (this file)
└── evaluation_results/            # Generated experimental tables (CSV) and plots (PNG)
    ├── final_evaluation_report.md # Comprehensive final experimental report
    ├── annotation_template.csv    # Template for human relevance labeling
    ├── ground_truth_test_set.csv  # Verified 50-pair benchmark dataset
    ├── model_evaluation_metrics.csv
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── model_metrics_comparison.png
    ├── ranking_quality_metrics.csv
    ├── candidate_rankings_sample.csv
    ├── ranking_system_vs_baseline.png
    ├── ranking_metrics_ndcg_precision.png
    ├── fairness_bias_comparison.csv
    ├── fairness_summary_metrics.csv
    ├── fairness_score_comparison.png
    └── fairness_score_diff_distribution.png
```

---

## 1. Prerequisites & Dependencies

The evaluation framework requires the following Python libraries:
- `pandas` (>= 2.0)
- `numpy` (>= 1.24)
- `scikit-learn` (>= 1.4)
- `scipy` (>= 1.10)
- `matplotlib` (>= 3.8)
- `seaborn` (>= 0.13)

To install or verify them:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

*(Note: Sentence-Transformers with `all-MiniLM-L6-v2` is supported by the architecture as a modular drop-in. When not installed, the system automatically uses the sublinear n-gram TF-IDF cosine similarity engine).*

---

## 2. How to Run the Evaluations

Make sure you are in the root `NLP Project` directory:

### Step 1: Run Model Performance Evaluation
```bash
python backend/evaluation/model_evaluation.py
```
- Creates an annotation template (`annotation_template.csv`) with instructions for manual review.
- Evaluates the 50-pair verified ground-truth test set across decision thresholds (30% to 55%).
- Generates `confusion_matrix.png` and `model_metrics_comparison.png`.

### Step 2: Run Candidate Ranking Quality Evaluation
```bash
python backend/evaluation/ranking_evaluation.py
```
- Tests 5 diverse job queries against the full pool of 166 unique candidate resumes.
- Calculates Information Retrieval metrics: Precision@5, Precision@10, NDCG@5, NDCG@10, and MRR.
- Directly compares the ResumeMatch AI multi-component model against a Simple Keyword-Overlap Baseline.
- Generates `ranking_system_vs_baseline.png` and `ranking_metrics_ndcg_precision.png`.

### Step 3: Run Fairness and Bias Risk Evaluation
```bash
python backend/evaluation/fairness_evaluation.py
```
- Evaluates 30 resumes across diverse domains under both Unredacted (with personal identifiers) and Anonymized (blind) conditions.
- Calculates score differences, rank shifts, percentage of candidates affected, and Spearman rank correlation.
- Generates `fairness_score_comparison.png` and `fairness_score_diff_distribution.png`.

---

## 3. Ground-Truth Data & Annotation Guidelines

Because public Kaggle resume and job datasets do not come with pre-existing hire/interview labels, we constructed a verified 50-pair ground truth benchmark:
- 5 representative jobs across 5 distinct domains (Data Science, Java Developer, Python Developer, DevOps Engineer, HR).
- 10 candidate resumes per job (5 verified domain-congruent positives and 5 verified domain-incongruent negatives).

To expand manual annotations, open `evaluation_results/annotation_template.csv` and follow the guidelines in `final_evaluation_report.md`.
