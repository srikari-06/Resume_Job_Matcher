# ResumeMatch AI: Smart Resume Matching

Welcome to **ResumeMatch AI**, an intelligent, transparent, and fair recruitment platform that uses Natural Language Processing (NLP) to match candidate resumes with job openings, identify skill gaps, and minimize hiring bias.

---

## 1. Project Overview

Finding the right candidate among hundreds of applications is time-consuming and often inconsistent. Traditional keyword searches frequently miss qualified candidates who describe their experience using different terms, while manual reviews can introduce unconscious bias.

**ResumeMatch AI** solves this by:
1. **Cleaning and standardizing unstructured resume text**: Removing formatting noise, emails, URLs, and encoding artifacts while preserving skills, education, and career history.
2. **Structuring job postings**: Extracting required skills, min/max experience ranges, and combining role attributes into unified NLP job descriptions.
3. **Comparing candidates semantically**: Evaluating candidate profiles against job requirements using NLP representations (semantic similarity, skill overlap, and experience fit).
4. **Promoting fairness**: Providing explainable match breakdowns and simulating blind evaluations to reduce demographic bias during initial screening.

---

## 2. Main Features

### Implemented Features (Working Now)
- **Resume Inspection & Preprocessing** (`backend/services/resume_processor.py`):
  - Automated detection of URLs, email addresses, social handles (`@`), hashtags (`#`), and non-ASCII mojibake artifacts (e.g., repairing corrupted text like `NaÃ¯ve` to `Naive`).
  - Removal of bullet points, formatting symbols, carriage returns, and duplicate whitespace.
  - Full dataset deduplication (reduced 962 raw rows down to 166 unique, high-quality resumes across 25 job domains).
  - Word count and character count calculation for each resume.
- **Job Posting Processing** (`backend/services/job_processor.py`):
  - Cleans and standardizes pipe-separated skills into clean, deduplicated skill lists.
  - Automatically parses text experience requirements (e.g., `"5 - 10 yrs"`) into numeric minimum and maximum values (`5.0` and `10.0`).
  - Cleans repeated location strings (e.g., `"Pune,Pune"` $\rightarrow$ `"Pune"`).
  - Synthesizes a unified `Cleaned_Job_Description` column combining Title, Role, Category, Functional Area, Industry, Experience, and Skills for NLP vectorization.
  - Drops unmatchable postings missing job titles and removes duplicate listings.
- **Inspection Utilities** (`backend/inspect_dataset.py`):
  - Recursive search for raw datasets.
  - Displays shapes, column types, 5-sample previews, missing value counts, and duplicate statistics.
- **Interactive Recruiter Workspace Prototype** (`index.html`):
  - Single-page application UI featuring a Recruiter Dashboard, Job Manager, Resume Matcher, Candidate Rankings, Skill Gap Analysis, Blind Matching, and Explainability views.

### Planned Features (In Development)
- **PDF & DOCX Resume Parser**: Live text extraction from uploaded candidate document files.
- **Semantic Vectorizer & Embedding Engine**: TF-IDF, Cosine Similarity, and Sentence-BERT representations saved inside `backend/models/`.
- **Automated Named Entity Recognition (NER)**: Machine-learning-based skill extraction and PII redaction (names, gender indicators, locations).
- **Backend REST API**: Flask or FastAPI backend to dynamically connect `index.html` with the Python NLP services.

---

## 3. Technology Stack

### Current Technologies
- **Python 3.12+**: Core programming language for data engineering and NLP.
- **Pandas (v2.3+)**: Data manipulation, missing value handling, and dataset deduplication.
- **NumPy**: Numerical operations and data structures.
- **Regular Expressions (`re`)**: Pattern matching, text sanitization, and experience parsing.
- **Frontend Prototype**: HTML5, Tailwind CSS (via CDN), Alpine.js (via CDN).

### Planned Libraries (For Next Phases)
- **Scikit-Learn**: TF-IDF vectorization and cosine similarity calculations.
- **Sentence-Transformers**: Deep semantic embeddings.
- **FastAPI / Flask**: RESTful backend server.

---

## 4. Project Folder Structure

```
NLP Project/
├── index.html                           # Recruiter UI prototype (Tailwind CSS + Alpine.js)
├── README.md                            # Complete project guide and documentation
└── backend/
    ├── inspect_dataset.py               # Inspects raw datasets and displays statistics
    ├── preprocess.py                    # Master pipeline orchestrator
    ├── services/                        # Modular processing logic
    │   ├── resume_processor.py          # Resume cleaning, deduplication, and metrics
    │   └── job_processor.py             # Job cleaning, experience parsing, and NLP synthesis
    ├── models/                          # Directory for saved vectorizers & model weights
    └── data/
        ├── raw/                         # Raw Kaggle CSV files (unchanged)
        │   ├── UpdatedResumeDataSet.csv
        │   └── marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv
        └── processed/                   # Cleaned datasets ready for NLP
            ├── cleaned_resumes.csv
            └── cleaned_jobs.csv
```

### Folder and File Roles
- **`backend/data/raw/`**: Holds the original, untouched CSV files downloaded from Kaggle.
- **`backend/data/processed/`**: Holds the output CSV files produced by `backend/preprocess.py`.
- **`backend/services/`**: Reusable Python modules containing cleaning functions for resumes and jobs.
- **`backend/models/`**: Reserved directory to store vectorizer models, tokenizers, or weights.
- **`backend/inspect_dataset.py`**: Reads raw data and displays complete diagnostics.
- **`backend/preprocess.py`**: Executes the end-to-end cleaning and saves processed CSVs.
- **`index.html`**: Interactive frontend dashboard prototype.

---

## 5. Dataset Setup

The project uses two public Kaggle datasets:

1. **UpdatedResumeDataSet.csv**: A collection of resumes categorized into 25 domains (Data Science, Java Developer, HR, Testing, etc.).
2. **marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv**: A collection of 30,000 job listings scraped from Naukri.com.

### Where to Place the Datasets
Make sure your original CSV files are placed inside:
```
backend/data/raw/
```

### Running Inspection and Preprocessing
Open your terminal inside the root `NLP Project` directory and run:

1. **Inspect raw data**:
   ```bash
   python backend/inspect_dataset.py
   ```

2. **Run cleaning & preprocessing**:
   ```bash
   python backend/preprocess.py
   ```

### Output Files
After running `preprocess.py`, the cleaned datasets are automatically saved to:
- `backend/data/processed/cleaned_resumes.csv` (166 unique cleaned resumes, ~0.92 MB)
- `backend/data/processed/cleaned_jobs.csv` (29,364 cleaned job postings, ~25.25 MB)

---

## 6. Installation and Setup

### Prerequisites
- Python 3.8 or newer (tested with Python 3.12).
- Modern web browser (Chrome, Edge, Firefox, or Safari).

### Step 1: Clone or Open the Project
Open your terminal (PowerShell, Command Prompt, or bash) and navigate to the project directory:
```bash
cd "NLP Project"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies
Install the required packages using `pip`:
```bash
pip install pandas numpy
```

*(Optional future dependencies for upcoming modeling steps: `pip install scikit-learn`)*

### Step 4: Frontend Dependencies
**No `npm install` is required.**
The frontend (`index.html`) uses Tailwind CSS and Alpine.js loaded directly from official CDNs. You do not need Node.js or npm to run the current UI.

---

## 7. How to Run the Project

### Running Backend Data Pipeline
To inspect and preprocess the data, run:
```bash
# Step 1: Inspect the raw datasets
python backend/inspect_dataset.py

# Step 2: Generate the cleaned CSVs
python backend/preprocess.py
```
*Note: These are batch processing scripts. They execute, print a summary report, and exit once finished.*

### Running the Frontend
You have two options to view the recruiter workspace:

#### Option A: Direct Browser Launch (Simplest)
Simply double-click `index.html` in your file explorer, or right-click `index.html` and select **Open with > Google Chrome** (or your preferred browser).

#### Option B: Local HTTP Server (Recommended)
From the `NLP Project` root directory, start a local server:
```bash
python -m http.server 8000
```
Then open your browser and navigate to:
```
http://localhost:8000/
```

#### Demo Recruiter Login Credentials
- **Email**: `admin@resumematch.ai`
- **Password**: `admin123`
*(Or click the **"Continue as Demo Recruiter"** button on the login screen).*

> **Note on Terminal Windows**:
> If you start the local web server with `python -m http.server 8000`, that terminal window **must remain open** while you browse the application. Press `Ctrl + C` in that terminal when you wish to stop the server.

---

## 8. How to Test and Verify

You can verify that everything is configured and functioning correctly with the following checks:

### 1. Verify Dataset Inspection
Run:
```bash
python backend/inspect_dataset.py
```
**Expected Result**:
- Detects `UpdatedResumeDataSet.csv` (962 rows, 2 columns, 0 nulls, 796 duplicates).
- Detects the Naukri CSV (30,000 rows, 11 columns, 575 missing titles).
- Exits with return code 0.

### 2. Verify Preprocessing Output
Run:
```bash
python backend/preprocess.py
```
**Expected Result**:
- Creates `backend/data/processed/cleaned_resumes.csv` with exactly 166 rows.
- Creates `backend/data/processed/cleaned_jobs.csv` with 29,364 rows.
- Displays a final summary banner confirming success.

### 3. Quick Data Verification One-Liner
To quickly inspect the generated files using Python:
```bash
python -c "import pandas as pd; print('Resumes:', pd.read_csv('backend/data/processed/cleaned_resumes.csv').shape); print('Jobs:', pd.read_csv('backend/data/processed/cleaned_jobs.csv').shape)"
```
**Expected Output**:
```
Resumes: (166, 6)
Jobs: (29364, 18)
```

---

## 9. Troubleshooting

### 1. `ModuleNotFoundError: No module named 'pandas'`
- **Cause**: Pandas is not installed in your active Python environment.
- **Solution**: Run `pip install pandas numpy` or make sure your virtual environment is activated.

### 2. `FileNotFoundError: Could not find any file matching...`
- **Cause**: The raw CSV files are not located in `backend/data/raw/` or have different filenames.
- **Solution**: Confirm that `UpdatedResumeDataSet.csv` and the Naukri CSV are placed inside `backend/data/raw/`.

### 3. PowerShell Script Execution Error (`Activate.ps1 cannot be loaded`)
- **Cause**: Windows PowerShell restricts script execution by default.
- **Solution**: Run this in PowerShell before activating your virtual environment:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

### 4. Frontend Styling or Buttons Not Working
- **Cause**: Your computer is offline or blocking CDN requests.
- **Solution**: `index.html` loads Tailwind CSS and Alpine.js over the internet. Ensure you have an active internet connection when opening `index.html`.

### 5. Corrupted Resume Text (Mojibake characters like `NaÃ¯ve`)
- **Cause**: Original CSV was saved with mixed encoding.
- **Solution**: Already handled! `backend/services/resume_processor.py` includes automatic string replacements and regex cleaning to fix this issue.

---

## 10. Future Improvements

The current version completes the dataset inspection and preprocessing foundation. Planned next steps include:

1. **TF-IDF & Cosine Similarity Matcher**: Implement a lightweight vectorizer to compute mathematical similarity scores between resumes and job postings.
2. **Dense Embeddings with Sentence-BERT**: Integrate pretrained transformer embeddings for semantic matching beyond exact keyword overlaps.
3. **Automated PII Redactor**: A blind-matching module that automatically detects and masks candidate names, phone numbers, genders, and locations.
4. **Interactive File Upload Backend**: Connect the upload dropzone in `index.html` to a FastAPI or Flask endpoint to parse and evaluate live `.pdf` and `.docx` resumes.
5. **Skill Gap Recommendations**: Algorithmically extract missing skills from candidate profiles and output actionable training recommendations.

---

## License & Credits
- **Datasets**: Sourced from public Kaggle datasets (Updated Resume Dataset & Naukri.com Job Postings Dataset).
- **Project**: Developed as part of the NLP Project: **ResumeMatch AI**.
