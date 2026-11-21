"""
matching_service.py
===================
ResumeMatch AI: Resume-Job Matching & Scoring Service

This service provides:
1. Semantic Similarity Matching using TF-IDF vectorization and Cosine Similarity
   (with architecture ready to plug in Sentence-BERT / all-MiniLM-L6-v2).
2. Skill Matching: Regex-based keyword matching between candidate resumes and job skills.
3. Experience Scoring: Heuristic evaluation of candidate experience vs job requirements.
4. Education Scoring: Recognition of degree keywords.
5. Composite Overall Score: Configurable weighted formula matching the frontend
   specifications (40% Semantic, 30% Skills, 20% Experience, 10% Education).
6. Baseline Keyword Matcher: Raw token overlap baseline for ranking quality comparison.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MatchingService:
    """
    Multi-component matching engine that compares candidate resumes against job postings.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ):
        """
        Initialize the matching service.

        Parameters:
        -----------
        weights : dict, optional
            Weights for composite scoring. Defaults to:
            {'semantic': 0.40, 'skills': 0.30, 'experience': 0.20, 'education': 0.10}
        """
        self.weights = weights or {
            "semantic": 0.40,
            "skills": 0.30,
            "experience": 0.20,
            "education": 0.10,
        }
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf

    # -------------------------------------------------------------------------
    # 1. Semantic Similarity
    # -------------------------------------------------------------------------
    def compute_semantic_similarity(
        self,
        job_text: str,
        resume_texts: List[str],
    ) -> np.ndarray:
        """
        Computes cosine similarity between a single job description and multiple resume texts.

        Uses TF-IDF representation with sublinear term-frequency scaling and n-grams.
        Returns array of similarity scores scaled to 0 - 100%.
        """
        if not job_text or not resume_texts:
            return np.zeros(len(resume_texts))

        corpus = [job_text] + list(resume_texts)
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
            max_features=10000,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            job_vector = tfidf_matrix[0:1]
            resume_vectors = tfidf_matrix[1:]
            sim_scores = cosine_similarity(job_vector, resume_vectors).flatten()
            # Scale to 0 - 100 percentage
            return np.clip(sim_scores * 100.0, 0.0, 100.0)
        except ValueError:
            # Empty vocabulary fallback
            return np.zeros(len(resume_texts))

    # -------------------------------------------------------------------------
    # 2. Skill Matching
    # -------------------------------------------------------------------------
    def match_skills(
        self,
        resume_text: str,
        job_skills: Union[str, List[str]],
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Matches required job skills against candidate resume text using word-boundary regex.

        Parameters:
        -----------
        resume_text : str
            Full clean text of the resume.
        job_skills : str or list of str
            Comma or pipe-separated string or list of skill names.

        Returns:
        --------
        dict:
            - 'matched': list of detected skills
            - 'missing': list of missing skills
            - 'skill_score': percentage (0 - 100%)
        """
        if isinstance(job_skills, str):
            skills_list = [s.strip() for s in re.split(r"[,|]", job_skills) if s.strip()]
        else:
            skills_list = [str(s).strip() for s in job_skills if str(s).strip()]

        # Filter out generic placeholders
        skills_list = [s for s in skills_list if s.lower() not in {"not specified", "na", "n/a", ""}]

        if not skills_list:
            return {"matched": [], "missing": [], "skill_score": 100.0}

        text_lower = resume_text.lower() if isinstance(resume_text, str) else ""
        matched = []
        missing = []

        for skill in skills_list:
            s_clean = skill.lower()
            # Construct a safe word-boundary pattern
            # Escape regex special characters in the skill name
            escaped_skill = re.escape(s_clean)
            pattern = rf"\b{escaped_skill}\b"
            if re.search(pattern, text_lower):
                matched.append(skill)
            else:
                # Also check common hyphen/space variations (e.g., 'scikit-learn' vs 'scikit learn')
                var_pattern = rf"\b{re.escape(s_clean.replace('-', ' '))}\b"
                if re.search(var_pattern, text_lower):
                    matched.append(skill)
                else:
                    missing.append(skill)

        score = (len(matched) / len(skills_list)) * 100.0
        return {
            "matched": matched,
            "missing": missing,
            "skill_score": round(score, 2),
        }

    # -------------------------------------------------------------------------
    # 3. Experience Scoring
    # -------------------------------------------------------------------------
    def extract_resume_experience_years(self, resume_text: str) -> Optional[float]:
        """
        Extracts candidate's stated years of experience from resume text using regex patterns.
        """
        if not isinstance(resume_text, str):
            return None

        # Pattern 1: 'X+ years of experience', 'X years experience'
        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?\s+experience",
            resume_text,
            re.IGNORECASE,
        )
        if matches:
            try:
                vals = [float(m) for m in matches if float(m) < 45]  # Cap realistic work years
                if vals:
                    return max(vals)
            except ValueError:
                pass

        # Pattern 2: 'Experience ... X years' or 'Exp: X yrs'
        matches_alt = re.findall(
            r"(?:exp|experience)\s*[-:]?\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            resume_text,
            re.IGNORECASE,
        )
        if matches_alt:
            try:
                vals = [float(m) for m in matches_alt if float(m) < 45]
                if vals:
                    return max(vals)
            except ValueError:
                pass

        return None

    def compute_experience_score(
        self,
        candidate_years: Optional[float],
        min_required: Optional[float],
        max_required: Optional[float],
    ) -> float:
        """
        Computes experience fit score (0 - 100%) based on required range.
        """
        # If job does not specify experience requirement, give neutral high score
        if min_required is None or np.isnan(min_required):
            return 85.0

        # If candidate experience could not be explicitly extracted, give neutral baseline
        if candidate_years is None:
            return 75.0

        if candidate_years >= min_required:
            if max_required is not None and not np.isnan(max_required) and candidate_years > (max_required + 5):
                # Slightly over-qualified
                return 90.0
            return 100.0
        else:
            # Under required experience: scale proportionally
            ratio = candidate_years / max(1.0, min_required)
            return round(max(30.0, ratio * 90.0), 2)

    # -------------------------------------------------------------------------
    # 4. Education Scoring
    # -------------------------------------------------------------------------
    def compute_education_score(self, resume_text: str) -> float:
        """
        Heuristic check for recognized academic degrees in resume text.
        """
        if not isinstance(resume_text, str):
            return 60.0

        text_lower = resume_text.lower()
        # High tier: Doctorate / Masters
        if re.search(r"\b(ph\.?d|doctorate|m\.?tech|m\.?e|mca|m\.?sc|mba|master|post\s+graduate)\b", text_lower):
            return 95.0
        # Standard tier: Bachelors / Engineering
        elif re.search(r"\b(b\.?tech|b\.?e|bca|b\.?sc|bachelor|degree|b\.?com|bba)\b", text_lower):
            return 85.0
        # Diploma tier
        elif re.search(r"\b(diploma|higher\s+secondary|polytechnic)\b", text_lower):
            return 75.0
        else:
            return 70.0

    # -------------------------------------------------------------------------
    # 5. Composite Overall Score
    # -------------------------------------------------------------------------
    def compute_overall_score(
        self,
        semantic_score: float,
        skill_score: float,
        experience_score: float,
        education_score: float,
    ) -> float:
        """
        Computes the weighted composite score (0 - 100%).
        """
        w = self.weights
        overall = (
            w.get("semantic", 0.40) * semantic_score
            + w.get("skills", 0.30) * skill_score
            + w.get("experience", 0.20) * experience_score
            + w.get("education", 0.10) * education_score
        )
        return round(float(np.clip(overall, 0.0, 100.0)), 2)

    # -------------------------------------------------------------------------
    # 6. Rank Candidates Against a Single Job
    # -------------------------------------------------------------------------
    def rank_candidates(
        self,
        job_row: Union[pd.Series, dict],
        resumes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evaluates and ranks all candidates in resumes_df against a single job posting.

        Returns a sorted DataFrame containing candidate rankings and all score breakdowns.
        """
        job_desc = str(job_row.get("Cleaned_Job_Description", job_row.get("Job Title", "")))
        job_skills = str(job_row.get("Cleaned_Key_Skills", job_row.get("Key Skills", "")))
        min_exp = job_row.get("Min_Experience_Yrs", None)
        max_exp = job_row.get("Max_Experience_Yrs", None)

        resume_texts = resumes_df["Cleaned_Resume"].tolist()

        # 1. Semantic similarities
        semantic_scores = self.compute_semantic_similarity(job_desc, resume_texts)

        results = []
        for idx, (_, r_row) in enumerate(resumes_df.iterrows()):
            r_id = r_row.get("Resume_Id", f"RES_{idx+1:04d}")
            cat = r_row.get("Category", "Unknown")
            r_text = str(r_row.get("Cleaned_Resume", ""))
            sem_s = round(float(semantic_scores[idx]), 2)

            # 2. Skill match
            skill_res = self.match_skills(r_text, job_skills)
            skill_s = skill_res["skill_score"]

            # 3. Experience match
            cand_exp = self.extract_resume_experience_years(r_text)
            exp_s = self.compute_experience_score(cand_exp, min_exp, max_exp)

            # 4. Education match
            edu_s = self.compute_education_score(r_text)

            # 5. Composite overall
            overall_s = self.compute_overall_score(sem_s, skill_s, exp_s, edu_s)

            results.append({
                "Resume_Id": r_id,
                "Category": cat,
                "Semantic_Score": sem_s,
                "Skill_Score": skill_s,
                "Experience_Score": exp_s,
                "Education_Score": edu_s,
                "Overall_Score": overall_s,
                "Matched_Skills": ", ".join(skill_res["matched"][:5]),
                "Missing_Skills": ", ".join(skill_res["missing"][:5]),
            })

        ranked_df = pd.DataFrame(results)
        # Sort by Overall_Score descending, then Semantic_Score descending
        ranked_df = ranked_df.sort_values(by=["Overall_Score", "Semantic_Score"], ascending=[False, False])
        ranked_df.reset_index(drop=True, inplace=True)
        ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))

        return ranked_df

    # -------------------------------------------------------------------------
    # 7. Baseline Keyword Matcher (For Comparative Evaluation)
    # -------------------------------------------------------------------------
    def rank_candidates_keyword_baseline(
        self,
        job_row: Union[pd.Series, dict],
        resumes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simple baseline: Ranks candidates using raw CountVectorizer term overlap.
        """
        job_desc = str(job_row.get("Cleaned_Job_Description", job_row.get("Job Title", "")))
        resume_texts = resumes_df["Cleaned_Resume"].tolist()

        vectorizer = CountVectorizer(stop_words="english", binary=True)
        try:
            matrix = vectorizer.fit_transform([job_desc] + resume_texts)
            job_vec = matrix[0:1]
            res_vecs = matrix[1:]
            overlap_scores = (res_vecs * job_vec.T).toarray().flatten()
        except ValueError:
            overlap_scores = np.zeros(len(resume_texts))

        results = []
        for idx, (_, r_row) in enumerate(resumes_df.iterrows()):
            results.append({
                "Resume_Id": r_row.get("Resume_Id", f"RES_{idx+1:04d}"),
                "Category": r_row.get("Category", "Unknown"),
                "Keyword_Overlap_Count": int(overlap_scores[idx]),
            })

        df_base = pd.DataFrame(results)
        df_base = df_base.sort_values(by="Keyword_Overlap_Count", ascending=False).reset_index(drop=True)
        df_base.insert(0, "Rank", range(1, len(df_base) + 1))
        return df_base
