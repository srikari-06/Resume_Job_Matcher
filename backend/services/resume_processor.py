"""
resume_processor.py
===================
ResumeMatch AI: Resume Preprocessing Service

This module provides reusable, beginner-friendly functions to:
1. Clean messy resume text (remove URLs, emails, special characters, and formatting artifacts).
2. Handle missing or empty resume values.
3. Remove duplicate resume records.
4. Prepare clean, NLP-ready text while preserving all original data and categories.
"""

import re
from typing import Optional, Union
import pandas as pd


def clean_resume_text(text: str, lowercase: bool = False) -> str:
    """
    Cleans raw resume text for Natural Language Processing (NLP).

    Steps performed:
    1. Removes URLs and web links (http, https, www).
    2. Removes email addresses.
    3. Removes Twitter/social media mentions (@username) and hashtags (#topic).
    4. Fixes common encoding/mojibake artifacts (e.g., 'NaÃ¯ve' -> 'Naive', unicode bullets).
    5. Replaces punctuation and bullet symbols with clean single spaces.
    6. Collapses multiple spaces, tabs, and newlines into a single space.
    7. Optionally converts the result to lowercase.

    Parameters:
    -----------
    text : str
        The raw text extracted from a resume.
    lowercase : bool, default=False
        Whether to convert the cleaned text to lowercase.

    Returns:
    --------
    str : Cleaned text ready for tokenization, vectorization, or matching.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 2. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # 3. Remove social handles (@mention) and hashtags (#hashtag)
    text = re.sub(r"[@#]\S+", " ", text)

    # 4. Remove 'RT' and 'cc' tags commonly found in text datasets
    text = re.sub(r"\b(RT|cc)\b", " ", text)

    # 5. Fix common encoding artifacts (mojibake) like 'NaÃ¯ve' -> 'Naive'
    text = text.replace("NaÃ¯ve", "Naive").replace("naÃ¯ve", "naive")

    # 6. Remove non-ASCII characters (e.g. unicode bullet symbols, strange characters)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 7. Replace special symbols and punctuation with a single space
    # (Keeps alphanumeric characters; separates words cleanly)
    text = re.sub(r"[^\w\s]", " ", text)

    # 8. Collapse multiple whitespace characters (spaces, tabs, newlines) into one
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Optional lowercase
    if lowercase:
        text = text.lower()

    return text


def process_resumes(
    data: Union[pd.DataFrame, str],
    drop_duplicates: bool = True,
    lowercase: bool = False,
) -> pd.DataFrame:
    """
    Loads, cleans, and standardizes the resume dataset.

    Parameters:
    -----------
    data : pd.DataFrame or str
        Either a Pandas DataFrame or a file path to the resume CSV.
    drop_duplicates : bool, default=True
        Whether to remove duplicate resume rows.
    lowercase : bool, default=False
        Whether to lowercase the cleaned NLP resume text.

    Returns:
    --------
    pd.DataFrame : The cleaned resume dataset with preserved original columns,
                   new unique Resume IDs, and NLP-ready cleaned text.
    """
    # Step 1: Load data if file path provided
    if isinstance(data, str):
        print(f"[Resume Processor] Reading resume CSV from: {data}")
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input 'data' must be a file path string or a pandas DataFrame.")

    initial_count = len(df)
    print(f"[Resume Processor] Initial records: {initial_count:,}")

    # Step 2: Handle missing values
    # Ensure critical columns exist
    if "Resume" not in df.columns:
        raise KeyError("Dataset is missing required column: 'Resume'")

    # Drop rows where 'Resume' text is null or just whitespace
    df = df.dropna(subset=["Resume"]).copy()
    df = df[df["Resume"].astype(str).str.strip().str.len() > 0]
    after_null_count = len(df)
    dropped_nulls = initial_count - after_null_count
    if dropped_nulls > 0:
        print(f"[Resume Processor] Dropped {dropped_nulls} record(s) with missing resume text.")

    # Fill missing Category if any with 'Uncategorized'
    if "Category" in df.columns:
        df["Category"] = df["Category"].fillna("Uncategorized").astype(str).str.strip()

    # Step 3: Handle duplicate records
    if drop_duplicates:
        # Check duplicates based on Resume text (or Category + Resume)
        df = df.drop_duplicates(subset=["Resume"], keep="first").copy()
        duplicates_removed = after_null_count - len(df)
        print(f"[Resume Processor] Removed {duplicates_removed:,} duplicate resume(s).")
    else:
        print("[Resume Processor] Duplicate removal skipped as requested.")

    # Reset index cleanly
    df = df.reset_index(drop=True)

    # Step 4: Create Clean Resume ID
    df.insert(0, "Resume_Id", [f"RES_{i+1:04d}" for i in range(len(df))])

    # Step 5: Clean Resume Text for NLP
    print("[Resume Processor] Cleaning resume text for NLP...")
    df["Cleaned_Resume"] = df["Resume"].apply(lambda t: clean_resume_text(t, lowercase=lowercase))

    # Step 6: Compute useful metrics (word count and character count)
    df["Resume_Word_Count"] = df["Cleaned_Resume"].apply(lambda t: len(t.split()))
    df["Resume_Char_Count"] = df["Cleaned_Resume"].apply(len)

    print(f"[Resume Processor] Successfully processed {len(df):,} resumes.")
    return df
