"""
inspect_dataset.py
==================
ResumeMatch AI: Dataset Inspection Script

This script:
1. Automatically finds both raw CSV datasets in backend/data/raw/ (even if nested).
2. Loads both datasets with Pandas.
3. Displays row and column counts.
4. Displays all column names.
5. Displays 5 sample records from each dataset.
6. Checks missing values and duplicate rows.
7. Identifies columns containing resume text, job titles, descriptions, skills,
   education, and experience.

Run this script using:
    python backend/inspect_dataset.py
"""

import os
import sys
from pathlib import Path
import pandas as pd


def find_dataset_file(base_dir: Path, filename_pattern: str) -> Path:
    """
    Search recursively for a file matching a pattern within base_dir.
    Returns the Path if found, otherwise raises FileNotFoundError.
    """
    matches = list(base_dir.rglob(filename_pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file matching '{filename_pattern}' inside '{base_dir}'"
        )
    return matches[0]


def inspect_resumes(file_path: Path) -> pd.DataFrame:
    """
    Inspect the UpdatedResumeDataSet CSV file.
    """
    print("\n" + "=" * 80)
    print("1. RESUME DATASET INSPECTION")
    print("=" * 80)
    print(f"File location: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # 1. Row and column counts
    rows, cols = df.shape
    print(f"\n[Shape] Total Rows: {rows:,} | Total Columns: {cols}")

    # 2. Column names and types
    print("\n[Columns & Data Types]:")
    for col in df.columns:
        print(f"  - '{col}' ({df[col].dtype})")

    # 3. Missing values check
    print("\n[Missing Values Analysis]:")
    missing = df.isnull().sum()
    for col in df.columns:
        pct = (missing[col] / rows) * 100
        print(f"  - {col}: {missing[col]} missing ({pct:.2f}%)")

    # 4. Duplicate rows check
    exact_dups = df.duplicated().sum()
    text_dups = df.duplicated(subset=['Resume']).sum() if 'Resume' in df.columns else 0
    print("\n[Duplicate Records Analysis]:")
    print(f"  - Full exact duplicate rows: {exact_dups:,} (out of {rows:,})")
    print(f"  - Duplicate resume texts: {text_dups:,}")
    print(f"  - Unique resumes: {rows - text_dups:,}")

    # 5. Category breakdown
    if 'Category' in df.columns:
        print(f"\n[Job Categories] Total unique categories: {df['Category'].nunique()}")
        print("Category distribution across unique resumes:")
        unique_df = df.drop_duplicates(subset=['Resume'])
        for cat, cnt in unique_df['Category'].value_counts().items():
            print(f"  - {cat:<26}: {cnt} unique resume(s)")

    # 6. Column Mapping identification
    print("\n[Identified Semantic Columns]:")
    print("  - Resume Text : 'Resume' (contains full text with skills, experience, education, projects)")
    print("  - Domain/Role : 'Category' (target domain label for classification/matching)")
    print("  - Skills      : Contained within 'Resume' text (e.g., 'Programming Languages: Python...')")
    print("  - Education   : Contained within 'Resume' text (e.g., 'Education Details May 2013 to May 2017 B.E...')")
    print("  - Experience  : Contained within 'Resume' text (e.g., work history, company experience)")

    # 7. Five sample records
    print("\n[Sample Records (5 samples - truncated for readability)]:\n")
    samples = df.head(5)
    for idx, row in samples.iterrows():
        cat = row.get('Category', 'N/A')
        text = str(row.get('Resume', ''))[:180].replace('\n', ' ').replace('\r', ' ')
        print(f"Sample #{idx + 1} | Category: [{cat}]")
        print(f"  Preview: {text}...\n")

    return df


def inspect_jobs(file_path: Path) -> pd.DataFrame:
    """
    Inspect the Naukri jobs dataset CSV file.
    """
    print("\n" + "=" * 80)
    print("2. NAUKRI JOB POSTINGS DATASET INSPECTION")
    print("=" * 80)
    print(f"File location: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # 1. Row and column counts
    rows, cols = df.shape
    print(f"\n[Shape] Total Rows: {rows:,} | Total Columns: {cols}")

    # 2. Column names and types
    print("\n[Columns & Data Types]:")
    for col in df.columns:
        print(f"  - '{col}' ({df[col].dtype})")

    # 3. Missing values check
    print("\n[Missing Values Analysis]:")
    missing = df.isnull().sum()
    for col in df.columns:
        pct = (missing[col] / rows) * 100
        print(f"  - {col:<24}: {missing[col]:>5} missing ({pct:>5.2f}%)")

    # 4. Duplicate rows check
    exact_dups = df.duplicated().sum()
    uniq_id_dups = df.duplicated(subset=['Uniq Id']).sum() if 'Uniq Id' in df.columns else 0
    content_dups = df.duplicated(subset=['Job Title', 'Key Skills', 'Job Experience Required']).sum()
    print("\n[Duplicate Records Analysis]:")
    print(f"  - Exact full-row duplicates     : {exact_dups}")
    print(f"  - Duplicate 'Uniq Id' values     : {uniq_id_dups} (every row has a unique crawl ID)")
    print(f"  - Content duplicates (Title+Skills+Exp): {content_dups:,} re-postings")

    # 5. Column Mapping identification
    print("\n[Identified Semantic Columns]:")
    print("  - Job Title       : 'Job Title' (primary title of the posting)")
    print("  - Job Role        : 'Role' (specific role classification, e.g. Software Developer)")
    print("  - Skills Required : 'Key Skills' (pipe-delimited string of key skills, e.g. 'Java| SQL| Spring')")
    print("  - Experience Req. : 'Job Experience Required' (e.g. '2 - 5 yrs', '5 - 10 yrs')")
    print("  - Domain/Industry : 'Industry' & 'Role Category' & 'Functional Area'")
    print("  - Location        : 'Location' (city / region of posting)")
    print("  - Salary Range    : 'Job Salary' (salary bracket if disclosed)")
    print("  - Job Description : NOTE -> This Kaggle Naukri 30k dataset has NO separate 'Job Description' column.")
    print("                      Instead, the job profile is described by combining 'Job Title', 'Role',")
    print("                      'Role Category', 'Functional Area', 'Industry', and 'Key Skills'.")
    print("  - Education       : NOTE -> There is NO dedicated education column in this raw dataset.")
    print("                      Qualifications are reflected via 'Role Category' and 'Key Skills'.")

    # 6. Five sample records
    print("\n[Sample Records (5 samples)]:\n")
    sample_cols = [c for c in ['Job Title', 'Key Skills', 'Job Experience Required', 'Location', 'Role'] if c in df.columns]
    samples = df[sample_cols].head(5)
    for idx, row in samples.iterrows():
        print(f"Sample #{idx + 1}:")
        print(f"  Job Title   : {row.get('Job Title', 'N/A')}")
        print(f"  Role        : {row.get('Role', 'N/A')}")
        print(f"  Key Skills  : {str(row.get('Key Skills', 'N/A'))[:100]}")
        print(f"  Experience  : {row.get('Job Experience Required', 'N/A')}")
        print(f"  Location    : {row.get('Location', 'N/A')}\n")

    return df


def main():
    print("=" * 80)
    print("RESUMEMATCH AI - DATASET INSPECTION")
    print("=" * 80)

    # Determine base directory (NLP Project root or script directory)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "backend" else script_dir
    raw_dir = project_root / "backend" / "data" / "raw"

    print(f"Project Root Directory : {project_root}")
    print(f"Raw Data Directory     : {raw_dir}")

    # Find datasets
    try:
        resume_file = find_dataset_file(raw_dir, "*UpdatedResumeDataSet*.csv")
        jobs_file = find_dataset_file(raw_dir, "*marketing_sample_for_naukri_com-job*.csv")
    except FileNotFoundError as err:
        print(f"\n[ERROR] File discovery failed: {err}")
        sys.exit(1)

    print(f"\n[Found Datasets]:")
    print(f"  1. Resume Dataset : {resume_file.name}")
    print(f"  2. Jobs Dataset   : {jobs_file.name}")

    # Inspect both datasets
    resumes_df = inspect_resumes(resume_file)
    jobs_df = inspect_jobs(jobs_file)

    print("\n" + "=" * 80)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 80)
    print(f"- Resumes: {len(resumes_df):,} raw records ({len(resumes_df.drop_duplicates(subset=['Resume'])):,} unique)")
    print(f"- Jobs   : {len(jobs_df):,} raw postings across {jobs_df['Location'].nunique() if 'Location' in jobs_df.columns else 'N/A'} locations")
    print("- Ready for preprocessing step.")
    print("=" * 80)


if __name__ == "__main__":
    main()
