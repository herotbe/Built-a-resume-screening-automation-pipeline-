"""
STEP 1: GENERATING MOCK RESUME + JOB POSTING DATA
====================================================
In a real project, this data would come from "Internee's intern resume database".
We don't have access to that, so we generate realistic resumes and job
descriptions that mirror the structure of real-world data.

Each resume is a block of free text (like a real resume would be) —
this is IMPORTANT because our NLP pipeline needs to extract structured
info (skills, experience) FROM unstructured text. That's the whole point
of using NLP here instead of just reading a spreadsheet.
"""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)
os.makedirs('data', exist_ok=True)

# ── SKILL POOL ────────────────────────────────────────────────────────────────
# A realistic taxonomy of skills across different intern role categories.
# This taxonomy is ALSO used later by our extraction engine — in a real
# pipeline this would be a maintained "skills ontology".

SKILL_POOL = {
    'data': ['Python', 'SQL', 'Pandas', 'NumPy', 'Excel', 'Tableau', 'Power BI',
             'Machine Learning', 'Data Visualization', 'Statistics', 'R'],
    'web_dev': ['JavaScript', 'React', 'Node.js', 'HTML', 'CSS', 'TypeScript',
                'Next.js', 'REST APIs', 'Git', 'MongoDB'],
    'marketing': ['SEO', 'Content Writing', 'Social Media Marketing', 'Google Analytics',
                  'Email Marketing', 'Meta Ads', 'Canva', 'Copywriting'],
    'design': ['Figma', 'Adobe Photoshop', 'UI/UX Design', 'Adobe Illustrator',
               'Wireframing', 'Prototyping'],
    'business': ['Financial Modeling', 'Market Research', 'Excel', 'PowerPoint',
                 'Business Analysis', 'Project Management'],
    'soft': ['Communication', 'Teamwork', 'Leadership', 'Problem Solving',
             'Time Management', 'Adaptability'],
}

ALL_SKILLS = sorted(set(skill for skills in SKILL_POOL.values() for skill in skills))

EDUCATION = [
    'BSc Computer Science', 'BSc Data Science', 'BBA', 'BSc Software Engineering',
    'BSc Marketing', 'BS Economics', 'BSc Information Technology', 'BSc Graphic Design'
]

UNIVERSITIES = [
    'NIT Lahore', 'LUMS', 'FAST-NUCES', 'Punjab University', 'COMSATS',
    'UET Lahore', 'IBA Karachi', 'GIKI'
]

NAMES_FIRST = ['Ahmed', 'Ayesha', 'Hassan', 'Fatima', 'Ali', 'Sana', 'Bilal', 'Maria',
               'Usman', 'Zainab', 'Omar', 'Hira', 'Hamza', 'Amna', 'Faisal', 'Mahnoor',
               'Saad', 'Iqra', 'Tariq', 'Noor']
NAMES_LAST = ['Khan', 'Malik', 'Raza', 'Sheikh', 'Butt', 'Chaudhry', 'Iqbal', 'Hussain',
              'Ahmad', 'Farooq']


# ── RESUME GENERATION ─────────────────────────────────────────────────────────
def generate_resume(resume_id):
    """
    Each resume gets:
      - 1 primary domain (drives most skills)
      - A few cross-domain skills (realistic — people aren't one-dimensional)
      - 0-3 years of experience
      - Education info
      - A free-text "summary" paragraph (this is what NLP will parse)
    """
    domain = np.random.choice(list(SKILL_POOL.keys() - {'soft'}))
    primary_skills = list(np.random.choice(
        SKILL_POOL[domain],
        size=min(4, len(SKILL_POOL[domain])),
        replace=False
    ))

    # Add 1-3 cross-domain skills for realism
    other_domains = [d for d in SKILL_POOL if d != domain]
    cross_skills = []
    for _ in range(np.random.randint(1, 3)):
        d = np.random.choice(other_domains)
        cross_skills.append(np.random.choice(SKILL_POOL[d]))

    # Always add 1-2 soft skills
    soft_skills = list(np.random.choice(SKILL_POOL['soft'], size=2, replace=False))

    skills = list(set(primary_skills + cross_skills + soft_skills))

    experience_years = np.random.choice([0, 0, 1, 1, 2, 2, 3], p=[0.25,0.2,0.2,0.15,0.1,0.05,0.05])
    education = np.random.choice(EDUCATION)
    university = np.random.choice(UNIVERSITIES)
    name = f"{np.random.choice(NAMES_FIRST)} {np.random.choice(NAMES_LAST)}"

    # Build the free-text resume summary — this is the "raw text" our NLP
    # pipeline will process, mimicking a real resume's summary section.
    exp_phrase = (
        "with no formal work experience yet, eager to start a career"
        if experience_years == 0 else
        f"with {experience_years} year(s) of hands-on experience"
    )

    summary = (
        f"{name} is a {education} student at {university} {exp_phrase}. "
        f"Proficient in {', '.join(skills[:-1])} and {skills[-1]}. "
        f"Strong background in {domain.replace('_', ' ')} with a passion for "
        f"continuous learning and delivering quality results. "
        f"Demonstrates {soft_skills[0].lower()} and {soft_skills[1].lower()} "
        f"in team-based environments."
    )

    return {
        'resume_id': f"R{resume_id:03d}",
        'name': name,
        'domain': domain,
        'education': education,
        'university': university,
        'experience_years': int(experience_years),
        'skills': skills,
        'resume_text': summary,
    }


# ── JOB POSTING GENERATION ──────────────────────────────────────────────────────
JOB_TEMPLATES = [
    {
        'title': 'Data Analyst Intern',
        'domain': 'data',
        'required_skills': ['Python', 'SQL', 'Pandas', 'Data Visualization', 'Statistics'],
        'min_experience': 0,
    },
    {
        'title': 'Machine Learning Intern',
        'domain': 'data',
        'required_skills': ['Python', 'Machine Learning', 'NumPy', 'Statistics'],
        'min_experience': 1,
    },
    {
        'title': 'Frontend Developer Intern',
        'domain': 'web_dev',
        'required_skills': ['JavaScript', 'React', 'HTML', 'CSS', 'Git'],
        'min_experience': 0,
    },
    {
        'title': 'Full Stack Developer Intern',
        'domain': 'web_dev',
        'required_skills': ['JavaScript', 'Node.js', 'React', 'MongoDB', 'REST APIs'],
        'min_experience': 1,
    },
    {
        'title': 'Digital Marketing Intern',
        'domain': 'marketing',
        'required_skills': ['SEO', 'Social Media Marketing', 'Content Writing', 'Google Analytics'],
        'min_experience': 0,
    },
    {
        'title': 'Performance Marketing Intern',
        'domain': 'marketing',
        'required_skills': ['Meta Ads', 'Google Analytics', 'Email Marketing', 'Copywriting'],
        'min_experience': 1,
    },
    {
        'title': 'UI/UX Design Intern',
        'domain': 'design',
        'required_skills': ['Figma', 'UI/UX Design', 'Wireframing', 'Prototyping'],
        'min_experience': 0,
    },
    {
        'title': 'Business Analyst Intern',
        'domain': 'business',
        'required_skills': ['Excel', 'Financial Modeling', 'Market Research', 'Business Analysis'],
        'min_experience': 0,
    },
]


def generate_job(job_id, template):
    description = (
        f"We are looking for a {template['title']} to join our team. "
        f"The ideal candidate should be proficient in {', '.join(template['required_skills'][:-1])} "
        f"and {template['required_skills'][-1]}. "
        f"Minimum {template['min_experience']} year(s) of experience required. "
        f"This role focuses on {template['domain'].replace('_', ' ')} tasks and "
        f"offers hands-on learning in a fast-paced environment."
    )
    return {
        'job_id': f"J{job_id:02d}",
        'title': template['title'],
        'domain': template['domain'],
        'required_skills': template['required_skills'],
        'min_experience': template['min_experience'],
        'job_description': description,
    }


# ── GENERATE ALL DATA ─────────────────────────────────────────────────────────
resumes = [generate_resume(i) for i in range(1, 51)]
jobs = [generate_job(i, t) for i, t in enumerate(JOB_TEMPLATES, start=1)]

resumes_df = pd.DataFrame(resumes)
jobs_df = pd.DataFrame(jobs)

# Save — skills/required_skills are lists, so JSON preserves structure best.
# We also save a CSV version with skills joined as strings for easy viewing.
resumes_df.to_json('data/resumes.json', orient='records', indent=2)
jobs_df.to_json('data/jobs.json', orient='records', indent=2)

resumes_csv = resumes_df.copy()
resumes_csv['skills'] = resumes_csv['skills'].apply(lambda x: ', '.join(x))
resumes_csv.to_csv('data/resumes.csv', index=False)

jobs_csv = jobs_df.copy()
jobs_csv['required_skills'] = jobs_csv['required_skills'].apply(lambda x: ', '.join(x))
jobs_csv.to_csv('data/jobs.csv', index=False)

print(f"✅ Generated {len(resumes_df)} resumes across {resumes_df['domain'].nunique()} domains")
print(f"✅ Generated {len(jobs_df)} job postings")
print(f"\nSample resume text:\n{resumes[0]['resume_text']}")
print(f"\nSample job description:\n{jobs[0]['job_description']}")
print(f"\nDomain distribution:\n{resumes_df['domain'].value_counts()}")
