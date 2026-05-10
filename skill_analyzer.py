"""
skill_analyzer.py
Skill gap analysis between resume and job description.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Common skill aliases for partial matching
SKILL_ALIASES = {
    "javascript": ["js", "node.js", "nodejs", "ecmascript", "es6", "es2015"],
    "typescript": ["ts"],
    "python": ["py", "python3"],
    "react": ["reactjs", "react.js"],
    "angular": ["angularjs", "angular.js"],
    "vue": ["vuejs", "vue.js"],
    "kubernetes": ["k8s"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp"],
    "microsoft azure": ["azure"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "continuous integration": ["ci", "ci/cd"],
    "continuous deployment": ["cd", "ci/cd"],
    "postgresql": ["postgres"],
    "mongodb": ["mongo"],
    "graphql": ["graph ql"],
    "restful api": ["rest api", "rest", "restful"],
    "docker": ["containerization"],
    "git": ["github", "gitlab", "version control"],
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison."""
    return skill.lower().strip().replace("-", " ").replace("_", " ")


def skills_match(skill1: str, skill2: str) -> bool:
    """
    Check if two skills match (exact or via alias).

    Args:
        skill1: First skill string.
        skill2: Second skill string.

    Returns:
        True if skills match or are aliases of each other.
    """
    s1 = normalize_skill(skill1)
    s2 = normalize_skill(skill2)

    if s1 == s2:
        return True

    # Check if one contains the other (for compound skills)
    if s1 in s2 or s2 in s1:
        return True

    # Check alias mappings
    for canonical, aliases in SKILL_ALIASES.items():
        all_forms = [canonical] + aliases
        if s1 in all_forms and s2 in all_forms:
            return True

    return False


def find_partial_match(skill: str, skill_list: list[str]) -> str | None:
    """
    Find a partial/semantic match for a skill in a list.

    Args:
        skill: Skill to search for.
        skill_list: List of skills to search in.

    Returns:
        Matched skill string if found, None otherwise.
    """
    skill_normalized = normalize_skill(skill)

    for candidate in skill_list:
        candidate_normalized = normalize_skill(candidate)

        # Check word overlap (2+ word match)
        skill_words = set(skill_normalized.split())
        candidate_words = set(candidate_normalized.split())

        if len(skill_words) > 1 or len(candidate_words) > 1:
            overlap = skill_words & candidate_words
            # Meaningful words (longer than 2 chars)
            meaningful_overlap = {w for w in overlap if len(w) > 2}
            if meaningful_overlap:
                return candidate

        # Check if one is a substring of the other
        if skill_normalized in candidate_normalized or candidate_normalized in skill_normalized:
            return candidate

    return None


def perform_skill_gap_analysis(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    llm_client=None
) -> dict[str, Any]:
    """
    Perform comprehensive skill gap analysis between resume and JD.
    Uses rule-based matching first, optionally enhanced by LLM.

    Args:
        resume_data: Structured resume data (from LLM extraction).
        jd_data: Structured JD data (from LLM extraction).
        llm_client: Optional LLM client for semantic analysis.

    Returns:
        Dictionary with matched, missing, partial, and additional skills.
    """
    # Gather all resume skills
    resume_skills_raw = (
        resume_data.get("skills", []) +
        resume_data.get("tools", []) +
        resume_data.get("frameworks", [])
    )
    resume_skills = [s for s in resume_skills_raw if s]

    # Gather all JD requirements
    jd_required = jd_data.get("required_skills", [])
    jd_preferred = jd_data.get("preferred_skills", [])
    jd_tools = jd_data.get("tools", [])
    jd_frameworks = jd_data.get("frameworks", [])
    jd_all = jd_required + jd_preferred + jd_tools + jd_frameworks
    jd_all = list({s for s in jd_all if s})  # deduplicate

    matched_skills = []
    missing_skills = []
    partial_matches = []
    matched_jd_skills = set()

    # Check each JD skill against resume
    for jd_skill in jd_all:
        found_exact = False
        found_partial = False

        # Exact / alias match
        for resume_skill in resume_skills:
            if skills_match(jd_skill, resume_skill):
                if jd_skill not in matched_skills:
                    matched_skills.append(jd_skill)
                matched_jd_skills.add(jd_skill)
                found_exact = True
                break

        if not found_exact:
            # Partial match
            partial = find_partial_match(jd_skill, resume_skills)
            if partial:
                partial_matches.append({
                    "jd_skill": jd_skill,
                    "resume_skill": partial,
                    "similarity": "partial"
                })
                matched_jd_skills.add(jd_skill)
                found_partial = True

        if not found_exact and not found_partial:
            missing_skills.append(jd_skill)

    # Additional skills (resume has but JD doesn't require)
    additional_skills = []
    for resume_skill in resume_skills:
        found = False
        for jd_skill in jd_all:
            if skills_match(resume_skill, jd_skill):
                found = True
                break
        if not found:
            additional_skills.append(resume_skill)

    # Separate required vs preferred missing skills
    missing_required = [s for s in missing_skills if s in jd_required]
    missing_preferred = [s for s in missing_skills if s in jd_preferred + jd_tools + jd_frameworks]

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "missing_required": missing_required,
        "missing_preferred": missing_preferred,
        "partial_matches": partial_matches,
        "additional_skills": additional_skills,
        "resume_skills_total": len(resume_skills),
        "jd_skills_total": len(jd_all),
        "match_count": len(matched_skills),
        "partial_count": len(partial_matches),
    }


def calculate_skill_match_score(gap_analysis: dict[str, Any]) -> float:
    """
    Calculate skill match percentage score.

    Args:
        gap_analysis: Result from perform_skill_gap_analysis().

    Returns:
        Score between 0.0 and 100.0
    """
    total_jd = gap_analysis.get("jd_skills_total", 0)
    if total_jd == 0:
        return 0.0

    matched = gap_analysis.get("match_count", 0)
    partial = gap_analysis.get("partial_count", 0)

    # Partial matches count as 0.5
    effective_matches = matched + (partial * 0.5)
    score = min(100.0, (effective_matches / total_jd) * 100)
    return round(score, 1)


def format_gap_analysis_display(gap_analysis: dict[str, Any]) -> str:
    """
    Format gap analysis results for display.

    Args:
        gap_analysis: Result from perform_skill_gap_analysis().

    Returns:
        Formatted markdown string.
    """
    lines = []

    matched = gap_analysis.get("matched_skills", [])
    missing_req = gap_analysis.get("missing_required", [])
    missing_pref = gap_analysis.get("missing_preferred", [])
    partial = gap_analysis.get("partial_matches", [])
    additional = gap_analysis.get("additional_skills", [])

    if matched:
        lines.append("### ✅ Matched Skills")
        lines.append(", ".join(f"`{s}`" for s in matched))
        lines.append("")

    if missing_req:
        lines.append("### ❌ Missing Required Skills")
        lines.append(", ".join(f"`{s}`" for s in missing_req))
        lines.append("")

    if missing_pref:
        lines.append("### ⚠️ Missing Preferred Skills")
        lines.append(", ".join(f"`{s}`" for s in missing_pref))
        lines.append("")

    if partial:
        lines.append("### 🔶 Partial Matches")
        for p in partial:
            lines.append(f"- **{p['jd_skill']}** ← similar to `{p['resume_skill']}` in your resume")
        lines.append("")

    if additional:
        lines.append("### ➕ Additional Skills (Not in JD)")
        lines.append(", ".join(f"`{s}`" for s in additional[:15]))
        lines.append("")

    return "\n".join(lines)
