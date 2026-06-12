"""
STEP 2: SKILL EXTRACTION + RESUME-TO-JOB MATCHING
====================================================

Two-part pipeline:

PART A — SKILL EXTRACTION (spaCy PhraseMatcher)
  Real resumes are unstructured text. We need to pull structured info
  (a list of skills) OUT of that text. PhraseMatcher scans text for exact
  matches against our skills taxonomy — fast, explainable, and doesn't
  require a heavy pretrained model.

PART B — RESUME-TO-JOB MATCHING (TF-IDF + Cosine Similarity)
  TF-IDF (Term Frequency–Inverse Document Frequency) converts text into
  numerical vectors based on word importance. Cosine similarity then
  measures how "close" two vectors are (0 = unrelated, 1 = identical).

  This is the SAME core technique used by real resume-screening tools
  and search engines for relevance ranking.
"""

import pandas as pd
import numpy as np
import json
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('charts', exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
resumes_df = pd.read_json('data/resumes.json')
jobs_df = pd.read_json('data/jobs.json')

print(f"Loaded {len(resumes_df)} resumes and {len(jobs_df)} job postings\n")


# ════════════════════════════════════════════════════════════════
# PART A: SKILL EXTRACTION WITH SPACY PHRASEMATCHER
# ════════════════════════════════════════════════════════════════
"""
nlp = spacy.blank("en") loads spaCy WITHOUT a pretrained model —
just the tokenizer (splits text into words). This is lightweight and
needs no downloads, but means no pretrained NER/POS tagging.

PhraseMatcher then lets us define our OWN vocabulary (the skills
taxonomy) to scan for in any text. Think of it as a smart, case-insensitive
"find all of these phrases" tool.
"""

nlp = spacy.blank("en")

# Build the full skill vocabulary from our taxonomy (same one used in generate_data.py)
ALL_SKILLS = [
    'Python', 'SQL', 'Pandas', 'NumPy', 'Excel', 'Tableau', 'Power BI',
    'Machine Learning', 'Data Visualization', 'Statistics', 'R',
    'JavaScript', 'React', 'Node.js', 'HTML', 'CSS', 'TypeScript',
    'Next.js', 'REST APIs', 'Git', 'MongoDB',
    'SEO', 'Content Writing', 'Social Media Marketing', 'Google Analytics',
    'Email Marketing', 'Meta Ads', 'Canva', 'Copywriting',
    'Figma', 'Adobe Photoshop', 'UI/UX Design', 'Adobe Illustrator',
    'Wireframing', 'Prototyping',
    'Financial Modeling', 'Market Research', 'PowerPoint',
    'Business Analysis', 'Project Management',
    'Communication', 'Teamwork', 'Leadership', 'Problem Solving',
    'Time Management', 'Adaptability',
]

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
# attr="LOWER" makes matching case-insensitive — "python" matches "Python"

# nlp.pipe() processes multiple texts efficiently in a batch
patterns = [nlp.make_doc(skill) for skill in ALL_SKILLS]
matcher.add("SKILLS", patterns)


def extract_skills(text):
    """
    Run the matcher on a piece of text and return the unique skills found.

    doc = nlp(text)        → tokenize the text
    matcher(doc)           → returns list of (match_id, start, end) tuples
    doc[start:end].text    → the actual matched phrase
    """
    doc = nlp(text)
    matches = matcher(doc)
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        # Normalize back to our canonical casing by matching against ALL_SKILLS
        for skill in ALL_SKILLS:
            if skill.lower() == span.text.lower():
                found_skills.add(skill)
                break
    return sorted(found_skills)


# Apply extraction to every resume's free text
resumes_df['extracted_skills'] = resumes_df['resume_text'].apply(extract_skills)

# Sanity check: does extraction match the "ground truth" skills list we generated?
resumes_df['extraction_match'] = resumes_df.apply(
    lambda row: set(row['extracted_skills']) == set(row['skills']), axis=1
)

print("PART A — SKILL EXTRACTION")
print(f"  Resumes processed: {len(resumes_df)}")
print(f"  Extraction accuracy vs ground truth: {resumes_df['extraction_match'].mean()*100:.1f}%")
print(f"\n  Example — {resumes_df.iloc[0]['name']}:")
print(f"    Extracted skills: {resumes_df.iloc[0]['extracted_skills']}")


# ════════════════════════════════════════════════════════════════
# PART B: TF-IDF + COSINE SIMILARITY MATCHING
# ════════════════════════════════════════════════════════════════
"""
How TF-IDF works (intuition):
  - Common words across ALL documents (e.g. "the", "and", "intern")
    get LOW weight — they don't help distinguish documents.
  - Words that are frequent in ONE document but rare across others
    get HIGH weight — they're distinctive/important.

TfidfVectorizer converts each text into a vector of these weighted scores.
cosine_similarity then measures the angle between two vectors:
  - 1.0 = identical direction (very similar content)
  - 0.0 = completely different content

We combine TWO signals for the final match score:
  1. TEXT similarity (TF-IDF on resume_text vs job_description)
  2. SKILL OVERLAP (% of required skills the candidate actually has)

Combining both is more robust than text similarity alone — a resume
could be textually similar but miss critical required skills.
"""

# Build a combined corpus: all resume texts + all job descriptions together.
# TF-IDF needs to see the FULL vocabulary across both sets to compute
# meaningful weights (otherwise resume-only and job-only words wouldn't
# be comparable).
all_texts = list(resumes_df['resume_text']) + list(jobs_df['job_description'])

vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
tfidf_matrix = vectorizer.fit_transform(all_texts)
# tfidf_matrix shape: (n_resumes + n_jobs, n_features)

n_resumes = len(resumes_df)
resume_vectors = tfidf_matrix[:n_resumes]
job_vectors = tfidf_matrix[n_resumes:]

# cosine_similarity(A, B) returns a matrix where entry [i,j] is the
# similarity between resume i and job j.
text_similarity = cosine_similarity(resume_vectors, job_vectors)
# Shape: (50 resumes, 8 jobs) — a full similarity grid


def skill_overlap_score(resume_skills, required_skills):
    """
    What % of the job's required skills does this candidate have?
    set intersection (&) finds common elements between two sets.
    """
    resume_set = set(resume_skills)
    required_set = set(required_skills)
    if not required_set:
        return 0.0
    overlap = resume_set & required_set
    return len(overlap) / len(required_set)


# Build the final combined score matrix
# Final score = 60% skill overlap + 40% text similarity
# Skill overlap is weighted higher because it's the more DIRECT signal
# for "can this person do the job" — text similarity can be noisy.
final_scores = np.zeros((len(resumes_df), len(jobs_df)))

for i, resume in resumes_df.iterrows():
    for j, job in jobs_df.iterrows():
        skill_score = skill_overlap_score(resume['extracted_skills'], job['required_skills'])

        # Experience penalty: if candidate doesn't meet min experience,
        # reduce score by 20% (soft penalty, not disqualification)
        exp_penalty = 1.0 if resume['experience_years'] >= job['min_experience'] else 0.8

        final_scores[i, j] = (0.6 * skill_score + 0.4 * text_similarity[i, j]) * exp_penalty

# Convert to DataFrame for easy handling — rows=resumes, columns=job titles
scores_df = pd.DataFrame(
    final_scores,
    index=resumes_df['resume_id'],
    columns=jobs_df['title']
)

print(f"\nPART B — MATCHING")
print(f"  Score matrix shape: {scores_df.shape} (resumes x jobs)")
print(f"  Score range: {final_scores.min():.3f} - {final_scores.max():.3f}")


# ════════════════════════════════════════════════════════════════
# TOP CANDIDATES PER JOB
# ════════════════════════════════════════════════════════════════
"""
For each job, rank all 50 resumes by their match score and take the top 5.
This is the actual "deliverable" — a ranked shortlist per role.
"""
top_candidates = {}
for job_title in jobs_df['title']:
    # nlargest(5) gets the top 5 highest scores — like SQL ORDER BY ... LIMIT 5
    top5 = scores_df[job_title].nlargest(5)
    top_candidates[job_title] = [
        {
            'resume_id': rid,
            'name': resumes_df.loc[resumes_df['resume_id']==rid, 'name'].values[0],
            'score': round(score, 3),
            'experience_years': int(resumes_df.loc[resumes_df['resume_id']==rid, 'experience_years'].values[0]),
        }
        for rid, score in top5.items()
    ]

print("\nTOP CANDIDATE FOR EACH ROLE:")
for job_title, candidates in top_candidates.items():
    top = candidates[0]
    print(f"  {job_title:<32} → {top['name']:<20} (score: {top['score']:.3f}, {top['experience_years']} yrs exp)")


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
scores_df.to_csv('data/match_scores.csv')

with open('data/top_candidates.json', 'w') as f:
    json.dump(top_candidates, f, indent=2)

resumes_df[['resume_id', 'name', 'domain', 'experience_years', 'extracted_skills']].to_json(
    'data/extracted_skills.json', orient='records', indent=2
)


# ════════════════════════════════════════════════════════════════
# VISUALIZATION 1: MATCH SCORE HEATMAP (sample of 15 resumes)
# ════════════════════════════════════════════════════════════════
sample_resumes = resumes_df.sample(15, random_state=1)['resume_id']
sample_scores = scores_df.loc[sample_resumes]

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    sample_scores,
    annot=True, fmt='.2f', cmap='RdYlGn',
    linewidths=0.5, ax=ax,
    cbar_kws={'label': 'Match Score'}
)
ax.set_title("Resume-to-Job Match Scores (sample of 15 candidates)", fontsize=13, fontweight='bold')
ax.set_xlabel("Job Posting")
ax.set_ylabel("Resume ID")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('charts/01_match_score_heatmap.png', bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Chart 1 saved — match score heatmap")


# ════════════════════════════════════════════════════════════════
# VISUALIZATION 2: TOP 5 CANDIDATES PER JOB (grouped bar)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, (job_title, candidates) in enumerate(top_candidates.items()):
    ax = axes[idx]
    names = [c['name'].split()[0] for c in candidates]  # first names only for readability
    scores = [c['score'] for c in candidates]
    colors = sns.color_palette("Blues_r", len(candidates))

    bars = ax.barh(names, scores, color=colors, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}', va='center', fontsize=8)

    ax.set_title(job_title, fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    sns.despine(ax=ax)

fig.suptitle("Top 5 Candidate Matches per Job Posting", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/02_top_candidates_per_job.png', bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Chart 2 saved — top candidates per job")


# ════════════════════════════════════════════════════════════════
# VISUALIZATION 3: SKILL FREQUENCY ACROSS ALL RESUMES
# ════════════════════════════════════════════════════════════════
"""
Which skills are most common in the candidate pool overall?
Useful for understanding the talent pool's strengths/gaps.
"""
from collections import Counter

all_extracted = [skill for skills in resumes_df['extracted_skills'] for skill in skills]
skill_counts = Counter(all_extracted)
skill_freq = pd.Series(skill_counts).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(skill_freq.index, skill_freq.values, color=sns.color_palette("mako", len(skill_freq)))
for bar in bars:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{int(bar.get_width())}', va='center', fontsize=9)
ax.set_title("Top 15 Most Common Skills in Candidate Pool (n=50 resumes)", fontsize=13, fontweight='bold')
ax.set_xlabel("Number of Candidates")
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig('charts/03_skill_frequency.png', bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Chart 3 saved — skill frequency distribution")


# ════════════════════════════════════════════════════════════════
# VISUALIZATION 4: AVG MATCH SCORE PER JOB (which roles have strongest pool)
# ════════════════════════════════════════════════════════════════
avg_scores = scores_df.mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(avg_scores.index, avg_scores.values, color=sns.color_palette("crest", len(avg_scores)))
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{bar.get_height():.2f}', ha='center', fontsize=9, fontweight='bold')
ax.set_title("Average Candidate Match Score by Job Posting\n(Higher = stronger applicant pool for this role)",
              fontsize=12, fontweight='bold')
ax.set_ylabel("Avg Match Score")
plt.xticks(rotation=30, ha='right')
sns.despine()
plt.tight_layout()
plt.savefig('charts/04_avg_score_per_job.png', bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Chart 4 saved — avg match score per job")


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY FINDINGS SUMMARY")
print("="*60)
print(f"  Total resumes processed:     {len(resumes_df)}")
print(f"  Total job postings:          {len(jobs_df)}")
print(f"  Skill extraction accuracy:   {resumes_df['extraction_match'].mean()*100:.1f}%")
print(f"  Most in-demand skill in pool: {skill_freq.index[0]} ({skill_freq.iloc[0]} candidates)")
print(f"  Strongest applicant pool:    {avg_scores.index[0]} (avg score {avg_scores.iloc[0]:.3f})")
print(f"  Weakest applicant pool:      {avg_scores.index[-1]} (avg score {avg_scores.iloc[-1]:.3f})")
print(f"  Avg match score overall:     {final_scores.mean():.3f}")
print("\n✅ All 4 charts saved to /charts/")
