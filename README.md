# 🤖 Resume Screening Automation — Internee.pk

> **Task 2 of 6** | Virtual Data Science Internship @ [Internee.pk](https://internee.pk)

An NLP-powered pipeline that extracts skills from resume text and matches candidates to job postings using TF-IDF + cosine similarity — automating the first pass of intern resume screening.

---

## 📌 Objective

Automate resume filtering to match candidates with job openings by extracting skills/experience from unstructured resume text and ranking candidates against job requirements.

---

## 🗂️ Project Structure

```
resume-screening-automation/
│
├── data/
│   ├── resumes.json              # 50 mock resumes (structured + free-text)
│   ├── resumes.csv                # Same data, flattened for spreadsheet view
│   ├── jobs.json                  # 8 job postings with required skills
│   ├── jobs.csv
│   ├── match_scores.csv           # Full 50x8 resume-to-job score matrix
│   ├── top_candidates.json        # Top 5 ranked candidates per job
│   └── extracted_skills.json      # Skills extracted per resume via NLP
│
├── charts/
│   ├── 01_match_score_heatmap.png       # Match scores, sample of 15 candidates
│   ├── 02_top_candidates_per_job.png    # Top 5 ranked candidates per role
│   ├── 03_skill_frequency.png           # Most common skills in candidate pool
│   └── 04_avg_score_per_job.png         # Which roles have the strongest pool
│
├── generate_data.py               # Mock resume/job data generator
├── match.py                        # NLP extraction + TF-IDF matching pipeline
└── README.md
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| spaCy | Tokenization + PhraseMatcher for skill extraction |
| scikit-learn | TF-IDF vectorization + cosine similarity |
| pandas | Data handling and aggregation |
| matplotlib / seaborn | Visualizations |

---

## 📡 Data Source & Approach

**Note on data access:** This project does not have access to Internee.pk's live intern resume database. To demonstrate the full pipeline, 50 synthetic resumes and 8 job postings were generated with realistic free-text structure (the same kind of unstructured paragraphs a real resume parser would need to handle). The NLP extraction and matching logic is identical to what would run against real resume text.

### Pipeline

**1. Skill Extraction — spaCy PhraseMatcher**
Rather than relying on a pretrained NER model (a black box), a custom **skills taxonomy** of 47 skills across 6 domains (data, web dev, marketing, design, business, soft skills) was built. spaCy's `PhraseMatcher` scans each resume's free text and extracts exact skill mentions — case-insensitive, fast, and fully explainable.

> Why rule-based over BERT/NER here? Resume screening benefits from **transparency** — recruiters need to know *why* a candidate was matched. A custom taxonomy is auditable and easy to extend with new skills as needed.

**2. Resume-to-Job Matching — TF-IDF + Cosine Similarity**
Each resume and job description is vectorized using TF-IDF (term frequency–inverse document frequency), which weights words by how distinctive they are. Cosine similarity then measures how closely a resume's content aligns with a job description.

**Final Match Score** = `0.6 × skill overlap` + `0.4 × text similarity`, with a soft penalty if the candidate doesn't meet minimum experience requirements.

Skill overlap is weighted higher because it's the more direct signal of job fit — text similarity alone can be noisy (e.g. two unrelated resumes might share filler language).

---

## 📈 Key Findings

| Metric | Result |
|--------|--------|
| Resumes processed | 50 |
| Job postings | 8 |
| Skill extraction accuracy (vs ground truth) | 100% |
| Most in-demand skill in pool | Leadership (22/50 candidates) |
| Strongest applicant pool | Digital Marketing Intern (avg score 0.147) |
| Weakest applicant pool | Frontend Developer Intern (avg score 0.073) |
| Overall avg match score | 0.114 |

### Top Candidate per Role
| Job | Top Candidate | Score | Experience |
|-----|--------------|-------|------------|
| Data Analyst Intern | Ayesha Sheikh | 0.452 | 1 yr |
| Machine Learning Intern | Tariq Iqbal | 0.561 | 1 yr |
| Frontend Developer Intern | Ayesha Sheikh | 0.447 | 0 yrs |
| Full Stack Developer Intern | Ali Sheikh | 0.373 | 0 yrs |
| Digital Marketing Intern | Ahmed Chaudhry | 0.605 | 2 yrs |
| Performance Marketing Intern | Ayesha Raza | 0.592 | 2 yrs |
| UI/UX Design Intern | Fatima Iqbal | 0.606 | 1 yr |
| Business Analyst Intern | Bilal Khan | 0.793 | 1 yr |

---

## 💡 Recommendations

**1. Frontend Developer Intern needs sourcing attention**
This role had the weakest applicant pool — either job descriptions aren't reaching the right candidates, or there's a genuine skills gap in the pipeline for React/JS-specific roles.

**2. Soft skills are oversaturated, not differentiating**
"Leadership" appeared in 22/50 resumes — it's no longer a distinguishing factor. Screening should weight technical/role-specific skills more heavily when ranking, which the current 60/40 split already does.

**3. Bilal Khan's 0.793 score for Business Analyst is an outlier**
Worth a manual review — a score this much higher than the rest of the pool (avg 0.114) suggests either an exceptionally strong match or a resume that closely echoes the job description's exact phrasing (worth checking for keyword-stuffing in real-world use).

**4. Extend the skills taxonomy over time**
100% extraction accuracy here reflects a closed taxonomy matching generated data. In production, the taxonomy should be reviewed periodically against real resumes to catch emerging skills (new frameworks, tools) not yet in the dictionary.

---

## ▶️ How to Run

```bash
git clone https://github.com/YOUR_USERNAME/internee-resume-screening
cd internee-resume-screening

pip install spacy scikit-learn pandas matplotlib seaborn

python generate_data.py   # generates mock resumes + job postings
python match.py            # runs extraction + matching, outputs charts
```

---

## 🧠 What I Learned

- Building a rule-based NLP extraction pipeline with spaCy's `PhraseMatcher`
- Vectorizing text with `TfidfVectorizer` and computing `cosine_similarity`
- Designing a composite scoring function (skill overlap + text similarity + experience penalty)
- Why explainability matters in screening pipelines — and when rule-based NLP beats black-box models

---

## 👤 Author

**Abdullah** — Data Science Student @ NIT Lahore (Powered by ASU)
Virtual Intern @ Internee.pk | Task 2/6

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)
