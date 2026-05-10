"""
ats_scoring.py
ATS (Applicant Tracking System) score calculation for AI Resume Optimizer.

Weighted formula:
  Skill Match      = 40%
  Experience       = 25%
  Keyword Density  = 20%
  Tools/Frameworks = 15%
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Scoring weights
WEIGHTS = {
    "skill_match": 0.40,
    "experience": 0.25,
    "keyword_density": 0.20,
    "tools_frameworks": 0.15,
}

# Experience patterns for parsing years
EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*years?\s+of\s+experience',
    r'(\d+)\+?\s*years?\s+experience',
    r'(\d+)\+?\s*yrs?\s+experience',
    r'(\d+)\+?\s*years?\s+in',
    r'over\s+(\d+)\s+years?',
    r'(\d+)\s*-\s*(\d+)\s+years?',  # range like "3-5 years"
]


def parse_experience_years(text: str) -> float:
    """
    Parse years of experience from a text string.

    Args:
        text: Text describing experience requirement.

    Returns:
        Number of years (float). Returns 0.0 if not parseable.
    """
    if not text:
        return 0.0

    text_lower = text.lower()

    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2 and groups[1]:
                # Range: use average
                return (float(groups[0]) + float(groups[1])) / 2
            return float(groups[0])

    # Try to extract any number
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return float(numbers[0])

    return 0.0


def calculate_experience_score(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any]
) -> float:
    """
    Calculate experience relevance score.

    Args:
        resume_data: Structured resume data.
        jd_data: Structured JD data.

    Returns:
        Score between 0.0 and 100.0
    """
    jd_exp_text = jd_data.get("experience_required", "")
    resume_exp_text = resume_data.get("experience_years", "")

    jd_years = parse_experience_years(jd_exp_text)
    resume_years = parse_experience_years(resume_exp_text)

    # If JD doesn't specify experience, give full score
    if jd_years == 0:
        # Check if candidate has any experience
        has_experience = bool(resume_data.get("experience"))
        return 85.0 if has_experience else 50.0

    if resume_years == 0:
        # Estimate from number of jobs
        job_count = len(resume_data.get("experience", []))
        if job_count >= 3:
            resume_years = 5.0
        elif job_count >= 2:
            resume_years = 3.0
        elif job_count >= 1:
            resume_years = 1.5
        else:
            resume_years = 0.0

    if resume_years == 0:
        return 10.0

    # Score based on how well experience matches requirement
    ratio = resume_years / jd_years

    if ratio >= 1.0:
        # Meets or exceeds requirement
        if ratio <= 2.0:
            score = 90.0 + (ratio - 1.0) * 5  # bonus for exceeding
        else:
            score = 95.0  # overqualified but still good
    elif ratio >= 0.75:
        score = 70.0 + (ratio - 0.75) * 80  # 70-90
    elif ratio >= 0.5:
        score = 40.0 + (ratio - 0.5) * 120  # 40-70
    else:
        score = ratio * 80  # 0-40

    return round(min(100.0, score), 1)


def calculate_keyword_density_score(
    resume_text: str,
    jd_data: dict[str, Any]
) -> float:
    """
    Calculate keyword density/frequency score.

    Args:
        resume_text: Raw resume text.
        jd_data: Structured JD data.

    Returns:
        Score between 0.0 and 100.0
    """
    if not resume_text:
        return 0.0

    resume_lower = resume_text.lower()

    # Gather keywords from JD
    keywords = []
    keywords.extend(jd_data.get("required_skills", []))
    keywords.extend(jd_data.get("preferred_skills", []))
    keywords.extend(jd_data.get("keywords", []))
    keywords.extend(jd_data.get("tools", []))
    keywords.extend(jd_data.get("frameworks", []))

    # Add job title words
    job_title = jd_data.get("job_title", "")
    if job_title:
        keywords.extend(job_title.split())

    # Also extract keywords from responsibilities
    responsibilities = jd_data.get("responsibilities", [])
    resp_text = " ".join(responsibilities).lower()
    # Common tech/action keywords
    action_words = re.findall(r'\b[a-z]{4,}\b', resp_text)
    keywords.extend(action_words[:20])

    if not keywords:
        return 50.0

    # Count how many keywords appear in resume
    keyword_hits = 0
    unique_keywords = list({k.lower().strip() for k in keywords if len(k) > 2})

    for keyword in unique_keywords:
        if keyword in resume_lower:
            keyword_hits += 1

    score = (keyword_hits / len(unique_keywords)) * 100
    return round(min(100.0, score), 1)


def calculate_tools_frameworks_score(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any]
) -> float:
    """
    Calculate tools and frameworks match score.

    Args:
        resume_data: Structured resume data.
        jd_data: Structured JD data.

    Returns:
        Score between 0.0 and 100.0
    """
    resume_tools = set(t.lower().strip() for t in resume_data.get("tools", []))
    resume_frameworks = set(f.lower().strip() for f in resume_data.get("frameworks", []))
    resume_combined = resume_tools | resume_frameworks

    jd_tools = set(t.lower().strip() for t in jd_data.get("tools", []))
    jd_frameworks = set(f.lower().strip() for f in jd_data.get("frameworks", []))
    jd_combined = jd_tools | jd_frameworks

    if not jd_combined:
        return 75.0  # No specific tools required

    matched = 0
    for jd_item in jd_combined:
        for resume_item in resume_combined:
            if jd_item in resume_item or resume_item in jd_item:
                matched += 1
                break

    score = (matched / len(jd_combined)) * 100
    return round(min(100.0, score), 1)


def calculate_ats_score(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    resume_text: str,
    skill_match_score: float
) -> dict[str, Any]:
    """
    Calculate comprehensive ATS score with all components.

    Args:
        resume_data: Structured resume data.
        jd_data: Structured JD data.
        resume_text: Raw resume text.
        skill_match_score: Pre-calculated skill match score (0-100).

    Returns:
        Dictionary with all score components and final ATS score.
    """
    # Calculate individual components
    experience_score = calculate_experience_score(resume_data, jd_data)
    keyword_score = calculate_keyword_density_score(resume_text, jd_data)
    tools_score = calculate_tools_frameworks_score(resume_data, jd_data)

    # Calculate weighted final score
    final_score = (
        skill_match_score * WEIGHTS["skill_match"] +
        experience_score * WEIGHTS["experience"] +
        keyword_score * WEIGHTS["keyword_density"] +
        tools_score * WEIGHTS["tools_frameworks"]
    )

    return {
        "final_score": round(final_score, 1),
        "skill_match_score": skill_match_score,
        "experience_score": experience_score,
        "keyword_score": keyword_score,
        "tools_score": tools_score,
        "weights": WEIGHTS,
        "grade": _score_to_grade(final_score),
    }


def _score_to_grade(score: float) -> str:
    """Convert numerical score to letter grade with label."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Fair"
    elif score >= 40:
        return "Poor"
    else:
        return "Very Poor"


def format_ats_score_display(score_data: dict[str, Any]) -> str:
    """
    Format ATS score for markdown display.

    Args:
        score_data: Result from calculate_ats_score().

    Returns:
        Formatted markdown string.
    """
    final = score_data["final_score"]
    grade = score_data["grade"]

    # Visual progress bar
    filled = int(final / 5)
    bar = "█" * filled + "░" * (20 - filled)

    lines = [
        f"## ATS Compatibility Score",
        f"",
        f"### {final}% — {grade}",
        f"`{bar}` {final}%",
        f"",
        f"### Score Breakdown",
        f"",
        f"| Component | Score | Weight | Contribution |",
        f"|-----------|-------|--------|-------------|",
        f"| 🎯 Skill Match | {score_data['skill_match_score']}% | 40% | {score_data['skill_match_score']*0.4:.1f} pts |",
        f"| 💼 Experience | {score_data['experience_score']}% | 25% | {score_data['experience_score']*0.25:.1f} pts |",
        f"| 🔍 Keyword Density | {score_data['keyword_score']}% | 20% | {score_data['keyword_score']*0.2:.1f} pts |",
        f"| 🛠️ Tools & Frameworks | {score_data['tools_score']}% | 15% | {score_data['tools_score']*0.15:.1f} pts |",
    ]

    return "\n".join(lines)
