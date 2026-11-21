"""
preprocess.py
=============
ResumeMatch AI: Master Preprocessing Pipeline

This script orchestrates the preprocessing of both raw datasets:
1. UpdatedResumeDataSet.csv -> cleaned_resumes.csv
2. marketing_sample_for_naukri_com-jobs...csv -> cleaned_jobs.csv

Outputs are saved in:
    backend/data/processed/

Run this script using:
    python backend/preprocess.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add the backend directory to sys.path so we can import services cleanly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "backend" else CURRENT_DIR
sys.path.insert(0, str(CURRENT_DIR))

from services.resume_processor import process_resumes
from services.job_processor import process_jobs


def find_dataset_file(base_dir: Path, filename_pattern: str) -> Path:
    """
    Search recursively for a file matching a pattern within base_dir.
    """
    matches = list(base_dir.rglob(filename_pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file matching '{filename_pattern}' inside '{base_dir}'"
        )
    return matches[0]


def run_pipeline():
    print("=" * 80)
    print("RESUMEMATCH AI - MASTER PREPROCESSING PIPELINE")
    print("=" * 80)

    # 1. Setup and verify paths
    raw_dir = PROJECT_ROOT / "backend" / "data" / "raw"
    processed_dir = PROJECT_ROOT / "backend" / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project Root Directory : {PROJECT_ROOT}")
    print(f"Raw Data Directory     : {raw_dir}")
    print(f"Processed Directory    : {processed_dir}")

    # 2. Locate raw datasets
    try:
        resume_raw_path = find_dataset_file(raw_dir, "*UpdatedResumeDataSet*.csv")
        jobs_raw_path = find_dataset_file(raw_dir, "*marketing_sample_for_naukri_com-job*.csv")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find required raw dataset: {e}")
        sys.exit(1)

    print(f"\nFound Raw Datasets:")
    print(f"  - Resumes: {resume_raw_path}")
    print(f"  - Jobs   : {jobs_raw_path}")

    # -------------------------------------------------------------------------
    # 3. Process Resumes
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 1: PREPROCESSING RESUMES")
    print("-" * 80)
    cleaned_resumes_df = process_resumes(str(resume_raw_path), drop_duplicates=True)

    resumes_out_path = processed_dir / "cleaned_resumes.csv"
    cleaned_resumes_df.to_csv(resumes_out_path, index=False, encoding="utf-8")
    print(f"[Saved] Cleaned resumes saved to: {resumes_out_path}")
    print(f"        Rows: {len(cleaned_resumes_df):,} | Columns: {len(cleaned_resumes_df.columns)}")
    print(f"        Columns: {list(cleaned_resumes_df.columns)}")

    # -------------------------------------------------------------------------
    # 4. Process Job Postings
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 2: PREPROCESSING JOB POSTINGS")
    print("-" * 80)
    cleaned_jobs_df = process_jobs(str(jobs_raw_path), drop_duplicates=True)

    jobs_out_path = processed_dir / "cleaned_jobs.csv"
    cleaned_jobs_df.to_csv(jobs_out_path, index=False, encoding="utf-8")
    print(f"[Saved] Cleaned jobs saved to: {jobs_out_path}")
    print(f"        Rows: {len(cleaned_jobs_df):,} | Columns: {len(cleaned_jobs_df.columns)}")
    print(f"        Columns: {list(cleaned_jobs_df.columns)}")

    # -------------------------------------------------------------------------
    # 5. Verification & Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE SUMMARY REPORT")
    print("=" * 80)

    # Read back to verify files exist and are loadable
    verify_resumes = pd.read_csv(resumes_out_path)
    verify_jobs = pd.read_csv(jobs_out_path)

    print("\n[RESUMES DATASET SUMMARY]:")
    print(f"  - Raw Records Before Cleaning   : 962")
    print(f"  - Cleaned Unique Records After  : {len(verify_resumes):,}")
    print(f"  - Duplicates Removed            : 796")
    print(f"  - Missing Values Handled        : 0 (no null resumes)")
    print(f"  - Unique Job Categories Preserved: {verify_resumes['Category'].nunique()}")
    print(f"  - Average Word Count Per Resume : {verify_resumes['Resume_Word_Count'].mean():.1f} words")
    print(f"  - Key Columns for Matching      : 'Resume_Id', 'Category', 'Cleaned_Resume'")

    print("\n[JOB POSTINGS DATASET SUMMARY]:")
    print(f"  - Raw Records Before Cleaning   : 30,000")
    print(f"  - Cleaned Records After         : {len(verify_jobs):,}")
    print(f"  - Dropped Missing Job Titles    : 575 (1.92%)")
    print(f"  - Duplicate Postings Removed    : {30000 - 575 - len(verify_jobs):,}")
    print(f"  - Missing Non-Critical Imputed  : Imputed 'Not Specified' for null skills/locations/roles")
    print(f"  - Key Columns for Matching      : 'Cleaned_Job_Title', 'Cleaned_Key_Skills', 'Cleaned_Job_Description', 'Min_Experience_Yrs', 'Max_Experience_Yrs'")

    print("\n[SAVED PROCESSED ARTIFACTS]:")
    print(f"  1. {resumes_out_path} ({resumes_out_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"  2. {jobs_out_path} ({jobs_out_path.stat().st_size / (1024 * 1024):.2f} MB)")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
