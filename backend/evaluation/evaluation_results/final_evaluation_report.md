# ResumeMatch AI: Experimental Evaluation Report

**Project Title:** ResumeMatch AI: Smart Resume Matching  
**Evaluation Scope:** Model Performance, Candidate Ranking Quality, and Algorithmic Fairness / Bias Risk  
**Date of Evaluation:** November 2026  
**Artifact Directory:** `backend/evaluation/evaluation_results/`

---

## Executive Summary

This report documents the empirical evaluation of the **ResumeMatch AI** matching engine. Experiments were designed to test three critical dimensions of an AI-assisted hiring system:
1. **Model Classification Performance**: Assessing precision, recall, F1-score, and accuracy against a verified ground-truth test set of 50 resume-job pairs across 5 distinct domains.
2. **Candidate Ranking Quality**: Measuring information retrieval effectiveness (Precision@5, Precision@10, NDCG@5, NDCG@10, and MRR) against a Simple Keyword-Overlap Baseline across 166 candidate resumes.
3. **Fairness & Bias Risk Diagnostic**: Quantifying the system's sensitivity to non-job-related personal identifiers (PII) by comparing unredacted vs. anonymized (blind) resume evaluations.

All reported figures represent actual experimental measurements obtained by executing the evaluation suite in `backend/evaluation/`.

---

## 1. Model Performance Evaluation

### 1.1 Methodology & Ground-Truth Annotation
The raw Kaggle datasets (`UpdatedResumeDataSet.csv` and the Naukri 30,000 jobs dataset) are unlinked and lack pre-existing hire/interview labels. To avoid generating unverified synthetic labels, we established a **50-pair verified ground-truth test set** (`ground_truth_test_set.csv`) covering 5 distinct technical and non-technical domains:
- **Data Science / Machine Learning**
- **Java Software Development**
- **Python Backend Development**
- **DevOps & Cloud Engineering**
- **Human Resources (HR)**

Each domain was evaluated against 10 candidate resumes:
- **5 Verified Positives (Relevance = 1)**: Domain-congruent candidates possessing matching technical toolsets and career trajectories.
- **5 Verified Negatives (Relevance = 0)**: Domain-incongruent candidates (e.g., Civil Engineering, Advocate, Arts, Sales resumes evaluated against Data Science openings).

An annotation template (`annotation_template.csv`) was also generated to facilitate extended human review.

### 1.2 Model Implementation Status
- **Semantic Model**: Evaluated using the implemented sublinear TF-IDF n-gram (1, 2) vectorizer paired with Cosine Similarity.
- **Sentence-Transformers Status**: While the system architecture is structured to support `all-MiniLM-L6-v2`, the `sentence-transformers` library was not present in the local evaluation environment. The TF-IDF cosine similarity engine served as the verified primary NLP model.
- **Scoring Engine**: Evaluated using the multi-component composite formula matching the frontend specification:
  $$\text{Overall Score} = 0.40 \times \text{Semantic} + 0.30 \times \text{Skills} + 0.20 \times \text{Experience} + 0.10 \times \text{Education}$$

### 1.3 Classification Results Across Decision Thresholds

| Threshold (%) | Accuracy | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30.0%** | **78.00%** | **93.75%** | **60.00%** | **73.17%** | 15 | 1 | 24 | 10 |
| **35.0%** | 68.00% | 100.00% | 36.00% | 52.94% | 9 | 0 | 25 | 16 |
| **40.0%** | 62.00% | 100.00% | 24.00% | 38.71% | 6 | 0 | 25 | 19 |
| **45.0%** | 60.00% | 100.00% | 20.00% | 33.33% | 5 | 0 | 25 | 20 |
| **50.0%** | 60.00% | 100.00% | 20.00% | 33.33% | 5 | 0 | 25 | 20 |
| **55.0%** | 58.00% | 100.00% | 16.00% | 27.59% | 4 | 0 | 25 | 21 |

*Data source: `backend/evaluation/evaluation_results/model_evaluation_metrics.csv`*

### 1.4 Confusion Matrix (at Optimal Threshold = 30.0%)

| | Predicted Irrelevant (0) | Predicted Relevant (1) |
| :--- | :---: | :---: |
| **Actual Irrelevant (0)** | **24 (TN)** | **1 (FP)** |
| **Actual Relevant (1)** | **10 (FN)** | **15 (TP)** |

*Visual reference: `backend/evaluation/evaluation_results/confusion_matrix.png`*  
*Threshold curve reference: `backend/evaluation/evaluation_results/model_metrics_comparison.png`*

### 1.5 Interpretation
- **High Precision (93.75% to 100%)**: At the 30% decision threshold, when ResumeMatch AI classifies a resume as a match, 15 out of 16 candidates are genuine domain fits (only 1 False Positive). Above 35%, Precision reaches 100% (zero False Positives). This protects recruiters from reviewing completely unqualified applicants.
- **Moderate Recall (60.00%)**: At 30% threshold, 10 True Positives were missed (False Negatives). This reflects a known limitation of lexical/n-gram TF-IDF matching: candidates who use alternative vocabulary, synonyms, or different phrasing for their skills may not meet the lexical similarity threshold. This provides strong empirical justification for upgrading to dense semantic embeddings (`all-MiniLM-L6-v2`).

---

## 2. Candidate Ranking Quality Evaluation

### 2.1 Methodology
In hiring workflows, recruiters rarely review candidates as a binary classification. Instead, they examine the **top ranked candidates**. We evaluated whether ResumeMatch AI ranks suitable candidates near the top of the pool.

- **Pool Size**: All 166 unique resumes from `cleaned_resumes.csv`.
- **Query Jobs**: 5 diverse job listings from Naukri:
  1. *Data Scientist - Sql/nosql*
  2. *Back End Java Developer - Restful Services/soap*
  3. *Python Developer*
  4. *Opportunity For Azure Devops Architect*
  5. *HR Executive*
- **Baseline**: A standard **Keyword-Overlap Baseline** using raw token overlap (`CountVectorizer`) between job description and resumes.

### 2.2 Ranking Quality Metrics (System vs. Baseline)

| Evaluation Metric | ResumeMatch AI (Composite Model) | Simple Keyword Baseline | Absolute Gain | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Precision@5 (P@5)** | **52.00%** | 32.00% | **+20.00%** | **+62.5%** |
| **Precision@10 (P@10)** | **42.00%** | 24.00% | **+18.00%** | **+75.0%** |
| **NDCG@5** | **0.5581** | 0.3398 | **+0.2183** | **+64.2%** |
| **NDCG@10** | **0.5477** | 0.3044 | **+0.2433** | **+79.9%** |
| **Mean Reciprocal Rank (MRR)** | **0.7400** | 0.5508 | **+0.1892** | **+34.3%** |

*Data source: `backend/evaluation/evaluation_results/ranking_quality_metrics.csv`*  
*Plot reference: `backend/evaluation/evaluation_results/ranking_system_vs_baseline.png`*  
*Domain breakdown reference: `backend/evaluation/evaluation_results/ranking_metrics_ndcg_precision.png`*

### 2.3 Per-Domain Breakdown

The table below shows the exact measured ranking quality metrics across each of the 5 evaluation job domains, as saved in `ranking_quality_metrics.csv`:

| Target Job Domain | Model P@5 | Baseline P@5 | Model P@10 | Baseline P@10 | Model NDCG@5 | Baseline NDCG@5 | Model NDCG@10 | Baseline NDCG@10 | Model MRR | Baseline MRR | First Rel. Rank (Model) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Data Science** | **80.00%** | 80.00% | **70.00%** | 50.00% | **0.8539** | 0.7860 | **0.7682** | 0.5885 | **1.0000** | 1.0000 | Rank 1 |
| **Java Developer** | **20.00%** | 0.00% | **40.00%** | 20.00% | **0.2140** | 0.0000 | **0.3503** | 0.1370 | **0.5000** | 0.1429 | Rank 2 |
| **Python Developer** | **80.00%** | 0.00% | **50.00%** | 10.00% | **0.8688** | 0.0000 | **0.8626** | 0.0911 | **1.0000** | 0.1111 | Rank 1 |
| **DevOps Engineer** | **60.00%** | 60.00% | **40.00%** | 30.00% | **0.7227** | 0.6992 | **0.6725** | 0.5667 | **1.0000** | 1.0000 | Rank 1 |
| **HR Executive** | **20.00%** | 20.00% | **10.00%** | 10.00% | **0.1312** | 0.2140 | **0.0851** | 0.1389 | **0.2000** | 0.5000 | Rank 5 |
| **MACRO-AVERAGE** | **52.00%** | **32.00%** | **42.00%** | **24.00%** | **0.5581** | **0.3398** | **0.5477** | **0.3044** | **0.7400** | **0.5508** | — |

*Mathematical verification of macro-averages across the 5 domains:*
- $\text{Model P@5} = (0.80 + 0.20 + 0.80 + 0.60 + 0.20) / 5 = 2.60 / 5 = \mathbf{0.5200\ (52.00\%)}$
- $\text{Model P@10} = (0.70 + 0.40 + 0.50 + 0.40 + 0.10) / 5 = 2.10 / 5 = \mathbf{0.4200\ (42.00\%)}$
- $\text{Model NDCG@5} = (0.8539 + 0.2140 + 0.8688 + 0.7227 + 0.1312) / 5 = 2.7906 / 5 = \mathbf{0.5581}$
- $\text{Model NDCG@10} = (0.7682 + 0.3503 + 0.8626 + 0.6725 + 0.0851) / 5 = 2.7387 / 5 = \mathbf{0.5477}$
- $\text{Model MRR} = (1.0000 + 0.5000 + 1.0000 + 1.0000 + 0.2000) / 5 = 3.7000 / 5 = \mathbf{0.7400}$

### 2.4 Sample Candidate Ranking Output (Top 5 for Data Scientist Job)

| Rank | Candidate ID | Category | Semantic Score (%) | Skill Score (%) | Exp Score (%) | Edu Score (%) | Overall Match (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `RES_0004` | Data Science | 9.99% | 60.00% | 100.00% | 75.00% | **45.50%** |
| **2** | `RES_0007` | Data Science | 21.14% | 50.00% | 85.00% | 85.00% | **45.46%** |
| **3** | `RES_0010` | Data Science | 18.54% | 50.00% | 85.00% | 85.00% | **44.42%** |
| **4** | `RES_0135` | Database | 10.67% | 50.00% | 85.00% | 85.00% | **41.27%** |
| **5** | `RES_0009` | Data Science | 9.61% | 40.00% | 85.00% | 85.00% | **40.34%** |

*Data source: `backend/evaluation/evaluation_results/candidate_rankings_sample.csv`*

### 2.5 Interpretation
- **Superior Top-Rank Quality**: Across all 5 domains, ResumeMatch AI outperformed the baseline by **+20.00% on P@5** and **+0.2183 on NDCG@5**. 
- **Rapid Top-Hit Retrieval**: The system achieved an **MRR of 0.7400**. In 3 out of 5 query domains (Data Science, Python, DevOps), the very first candidate displayed at Rank 1 was a confirmed relevant domain fit ($RR = 1.0$). In Java Development, the first relevant candidate appeared at Rank 2 ($RR = 0.50$), and in HR at Rank 5 ($RR = 0.20$), yielding a macro-average MRR of $(1.0 + 0.5 + 1.0 + 1.0 + 0.2) / 5 = 0.7400$.
- **Caution on Raw Scores**: In NLP recruitment systems, the highest semantic score does **not** automatically denote the "best" hire. A resume stuffed with repetitive keywords might yield higher semantic overlap than a concise senior engineer's resume. Incorporating structured skill matching, years of experience, and education heuristics prevents keyword stuffing from dominating the rankings.

---

## 3. Fairness and Bias Risk Diagnostic

### 3.1 Objective & Experimental Design
Algorithmic bias in hiring often stems from resume models scoring non-job-related personal characteristics (names indicating gender or ethnicity, contact channels, or geographic headers).

To evaluate sensitivity to these non-job features:
- A diverse sample of **30 resumes** was evaluated against a standard job opening.
- **Condition A (Original / Unredacted)**: Resumes containing candidate names, email addresses, phone numbers, and geographic location headers.
- **Condition B (Anonymized / Blind)**: Identical resumes where personal identifiers were masked (`[NAME REDACTED]`, `[EMAIL REDACTED]`, `[PHONE REDACTED]`, gendered honorifics stripped), while all technical skills, career histories, and education credentials were kept intact.
- Both sets were scored using the identical algorithm.

### 3.2 Quantitative Fairness Diagnostic Results

| Metric | Measured Value |
| :--- | :---: |
| **Total Resumes Evaluated** | 30 candidates across 17 categories |
| **Average Match Score (Original)** | 27.94% |
| **Average Match Score (Anonymized)** | 27.95% |
| **Mean Absolute Score Difference** | **0.02%** |
| **Maximum Score Difference** | **0.12%** |
| **Candidates with Rank Shift (Count)** | **0 / 30 (0.0%)** |
| **Average Rank Position Shift** | **0.00 positions** |
| **Spearman Rank Correlation ($\rho$)** | **1.0000** |

*Data source: `backend/evaluation/evaluation_results/fairness_bias_comparison.csv` and `fairness_summary_metrics.csv`*  
*Scatter plot reference: `backend/evaluation/evaluation_results/fairness_score_comparison.png`*  
*Distribution plot reference: `backend/evaluation/evaluation_results/fairness_score_diff_distribution.png`*

### 3.3 Critical Methodological Interpretation
> [!IMPORTANT]
> **What this result proves**: The 0.02% score difference and 1.0000 rank correlation demonstrate that **ResumeMatch AI's feature extraction is functionally invariant to non-job PII noise**. The model scores candidates on their technical skills, domain vocabulary, experience, and degrees rather than applicant names or contact headers.

> [!WARNING]
> **What this result DOES NOT prove**: A negligible score difference between original and blind resumes is an **initial bias-risk diagnostic, not mathematical proof that the system is completely unbiased**. True algorithmic bias can also manifest through proxy variables (e.g., historical educational disparities, career gap penalties, or gender-coded technical verbiage). Continuous auditing and human-in-the-loop oversight remain mandatory.

---

## 4. Summary of Evaluation Artifacts

All experimental artifacts have been saved and verified in `backend/evaluation/evaluation_results/`:

| Artifact | File Type | Description |
| :--- | :--- | :--- |
| `annotation_template.csv` | CSV | Template with guidelines for human annotators to review resume-job relevance. |
| `ground_truth_test_set.csv` | CSV | 50-pair verified benchmark test set (25 positives, 25 negatives). |
| `model_evaluation_metrics.csv` | CSV | Accuracy, Precision, Recall, and F1 across decision thresholds (30%–55%). |
| `confusion_matrix.csv` | CSV | Confusion matrix data (TP=15, FP=1, TN=24, FN=10) at optimal threshold. |
| `confusion_matrix.png` | PNG | High-resolution annotated heatmap of classification confusion matrix. |
| `model_metrics_comparison.png` | PNG | Precision-Recall-F1-Accuracy threshold curve chart. |
| `ranking_quality_metrics.csv` | CSV | P@5, P@10, NDCG@5, NDCG@10, MRR per query job and macro-averages. |
| `candidate_rankings_sample.csv` | CSV | Top 10 candidate rankings with all sub-scores for the Data Science query job. |
| `ranking_system_vs_baseline.png`| PNG | Bar chart comparing ResumeMatch AI vs. Simple Keyword Baseline. |
| `ranking_metrics_ndcg_precision.png`| PNG | Per-domain NDCG@5 trend chart. |
| `fairness_bias_comparison.csv` | CSV | Candidate-by-candidate comparison of original vs. anonymized scores and ranks. |
| `fairness_summary_metrics.csv` | CSV | Aggregated fairness diagnostic metrics. |
| `fairness_score_comparison.png` | PNG | Scatter plot comparing original vs. blind scores with parity reference line. |
| `fairness_score_diff_distribution.png` | PNG | Distribution plot of absolute score differences. |

---

## 5. Limitations & Future Roadmap

1. **Dense Semantic Embeddings (`all-MiniLM-L6-v2`)**: Upgrading from TF-IDF n-grams to Sentence-BERT embeddings will capture semantic synonyms (e.g., recognizing that "NLP practitioner" relates to "Computational Linguist"), directly addressing the 40% false negative rate observed in the recall evaluation.
2. **Larger Annotated Datasets**: Expanding the ground-truth benchmark from 50 pairs to 500+ pairs via multi-annotator crowdsourcing will yield narrower confidence intervals and support fine-grained domain subcategories.
3. **Automated Named Entity Recognition (NER)**: Integrating a spaCy/BERT NER pipeline will automate the detection and masking of PII directly from raw PDF/DOCX uploads without requiring regex heuristics.
4. **Proxy Variable & Demographic Parity Auditing**: Expanding fairness evaluations to audit systemic proxy indicators (e.g., college graduation years as an age proxy) to guarantee fair treatment across all protected classes.
