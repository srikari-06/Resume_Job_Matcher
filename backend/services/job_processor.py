"""
job_processor.py
================
ResumeMatch AI: Job Postings Preprocessing Service

This module provides reusable, beginner-friendly functions to:
1. Clean job titles, skills, experience, and location strings.
2. Handle missing values appropriately across all fields.
3. Remove duplicate job postings.
4. Synthesize a unified NLP job description text combining title, role, skills,
   and experience for matching against candidate resumes.
5. Preserve all original raw columns for filtering and display in the recruiter workspace.
"""

import re
from typing import Optional, Tuple, Union
import pandas as pd


def clean_text_field(text: str) -> str:
    """
    Cleans general text strings (removes excess whitespace, non-ascii characters, etc.).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove non-ascii characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_skills(skills_text: str) -> str:
    """
    Cleans and standardizes the pipe-separated 'Key Skills' string from Naukri.

    Example:
        Input  : "  Media Planning| Digital Media | Digital Media "
        Output : "Media Planning, Digital Media"
    """
    if not isinstance(skills_text, str) or not skills_text.strip():
        return "Not Specified"

    # Split on pipe '|' or comma ','
    raw_tokens = re.split(r"[|,]", str(skills_text))
    cleaned_tokens = []
    seen = set()

    for token in raw_tokens:
        clean_tok = clean_text_field(token)
        # Skip empty tokens or single punctuation
        if clean_tok and clean_tok.lower() not in seen:
            seen.add(clean_tok.lower())
            cleaned_tokens.append(clean_tok)

    if not cleaned_tokens:
        return "Not Specified"

    return ", ".join(cleaned_tokens)


def parse_experience_years(exp_text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses minimum and maximum required years of experience from strings like:
    - '5 - 10 yrs' -> (5.0, 10.0)
    - '2 yrs'      -> (2.0, 2.0)
    - '0 - 1 yrs'  -> (0.0, 1.0)
    - 'Not Specified' -> (None, None)
    """
    if not isinstance(exp_text, str):
        return None, None

    # Pattern for range: '5 - 10 yrs'
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:yrs|yr|years)?", exp_text, re.IGNORECASE)
    if range_match:
        try:
            return float(range_match.group(1)), float(range_match.group(2))
        except ValueError:
            pass

    # Pattern for single number: '3+ yrs' or '3 yrs'
    single_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:yrs|yr|years)", exp_text, re.IGNORECASE)
    if single_match:
        try:
            val = float(single_match.group(1))
            return val, val
        except ValueError:
            pass

    return None, None


def clean_location(loc_text: str) -> str:
    """
    Cleans repeated location strings (e.g. 'Pune,Pune' -> 'Pune').
    """
    if not isinstance(loc_text, str) or not loc_text.strip():
        return "Not Specified"

    tokens = [t.strip() for t in loc_text.split(",") if t.strip()]
    unique_tokens = []
    for t in tokens:
        if t not in unique_tokens:
            unique_tokens.append(t)

    return ", ".join(unique_tokens) if unique_tokens else "Not Specified"


def build_job_nlp_text(row: pd.Series) -> str:
    """
    Synthesizes a rich, unified NLP text representation from job posting fields.
    Since the Naukri 30k dataset has no single full 'Job Description' column,
    we combine:
    - Job Title
    - Role & Role Category
    - Functional Area & Industry
    - Job Experience Required
    - Key Skills

    This provides a comprehensive text block for semantic resume matching.
    """
    title = clean_text_field(str(row.get("Job Title", "")))
    role = clean_text_field(str(row.get("Role", "")))
    role_cat = clean_text_field(str(row.get("Role Category", "")))
    func_area = clean_text_field(str(row.get("Functional Area", "")))
    industry = clean_text_field(str(row.get("Industry", "")))
    skills = clean_text_field(str(row.get("Cleaned_Key_Skills", "")))
    exp = clean_text_field(str(row.get("Job Experience Required", "")))

    parts = []
    if title and title != "Not Specified":
        parts.append(f"Job Title: {title}.")
    if role and role != "Not Specified":
        parts.append(f"Role: {role}.")
    if role_cat and role_cat != "Not Specified":
        parts.append(f"Category: {role_cat}.")
    if func_area and func_area != "Not Specified":
        parts.append(f"Functional Area: {func_area}.")
    if industry and industry != "Not Specified":
        parts.append(f"Industry: {industry}.")
    if exp and exp != "Not Specified":
        parts.append(f"Experience Required: {exp}.")
    if skills and skills != "Not Specified":
        parts.append(f"Required Skills: {skills}.")

    combined = " ".join(parts)
    return clean_text_field(combined)


def process_jobs(
    data: Union[pd.DataFrame, str],
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Loads, cleans, and standardizes the Naukri job postings dataset.

    Parameters:
    -----------
    data : pd.DataFrame or str
        Either a Pandas DataFrame or a file path to the Naukri job CSV.
    drop_duplicates : bool, default=True
        Whether to remove duplicate job postings.

    Returns:
    --------
    pd.DataFrame : Cleaned job postings with preserved original fields,
                   structured experience columns, standardized skills,
                   and synthesized Cleaned_Job_Description for NLP.
    """
    # Step 1: Load data
    if isinstance(data, str):
        print(f"[Job Processor] Reading jobs CSV from: {data}")
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input 'data' must be a file path string or a pandas DataFrame.")

    initial_count = len(df)
    print(f"[Job Processor] Initial records: {initial_count:,}")

    # Step 2: Handle missing values
    # A job posting without a Job Title cannot be matched or displayed
    if "Job Title" in df.columns:
        valid_title_mask = df["Job Title"].notna() & (df["Job Title"].astype(str).str.strip().str.len() > 0)
        df = df[valid_title_mask].copy()
        dropped_titles = initial_count - len(df)
        print(f"[Job Processor] Dropped {dropped_titles:,} records with missing/empty Job Title.")

    # Fill non-critical missing columns with 'Not Specified'
    fill_cols = [
        "Key Skills",
        "Job Experience Required",
        "Role Category",
        "Location",
        "Functional Area",
        "Industry",
        "Role",
        "Job Salary",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Not Specified").astype(str)

    # Step 3: Handle duplicate job postings
    # Content-based duplicate removal: postings with identical title, skills, experience, and location
    if drop_duplicates:
        dedup_subset = [c for c in ["Job Title", "Key Skills", "Job Experience Required", "Location"] if c in df.columns]
        before_dedup = len(df)
        df = df.drop_duplicates(subset=dedup_subset, keep="first").copy()
        dups_removed = before_dedup - len(df)
        print(f"[Job Processor] Removed {dups_removed:,} duplicate job postings.")
    else:
        print("[Job Processor] Duplicate removal skipped as requested.")

    # Reset index
    df = df.reset_index(drop=True)

    # Step 4: Clean individual columns
    print("[Job Processor] Standardizing job titles, skills, location, and experience...")
    df["Cleaned_Job_Title"] = df["Job Title"].apply(clean_text_field)
    df["Cleaned_Key_Skills"] = df["Key Skills"].apply(clean_skills)
    df["Cleaned_Location"] = df["Location"].apply(clean_location)

    # Parse experience into min/max numeric values
    exp_parsed = df["Job Experience Required"].apply(parse_experience_years)
    df["Min_Experience_Yrs"] = [p[0] for p in exp_parsed]
    df["Max_Experience_Yrs"] = [p[1] for p in exp_parsed]

    # Step 5: Synthesize Cleaned_Job_Description for NLP
    print("[Job Processor] Generating unified NLP job descriptions...")
    df["Cleaned_Job_Description"] = df.apply(build_job_nlp_text, axis=1)

    # Compute word count of the job description
    df["Job_Word_Count"] = df["Cleaned_Job_Description"].apply(lambda t: len(t.split()))

    print(f"[Job Processor] Successfully processed {len(df):,} job postings.")
    return df
