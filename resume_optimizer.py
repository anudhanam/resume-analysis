"""
resume_optimizer.py
LLM-based resume optimization for AI Resume Optimizer.
"""

import json
import logging
from typing import Any

from llm_prompts import (
    RESUME_OPTIMIZATION_PROMPT,
    SECTION_OPTIMIZATION_PROMPTS,
    ATS_SCORE_EXPLANATION_PROMPT,
)

logger = logging.getLogger(__name__)


def optimize_resume(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    selected_skills: list[str],
    sections_to_optimize: list[str],
    llm_client,
    selected_exp_bullets: list[str] = None,
) -> dict[str, Any]:
    """
    Optimize each selected section individually to avoid token limits.
    Merges results back into a copy of the original resume data.
    selected_exp_bullets: checkbox labels from the UI identifying which
                          experience bullets the user wants rewritten.
    """
    optimized = resume_data.copy()

    for section in sections_to_optimize:
        logger.info(f"Optimizing section: {section}")
        try:
            result = optimize_section(
                section, resume_data, jd_data, selected_skills, llm_client,
                selected_exp_bullets=selected_exp_bullets if section == "experience" else None,
            )
            if result:
                optimized[section] = result
            else:
                logger.warning(f"Section '{section}' returned empty — keeping original")
        except Exception as e:
            logger.error(f"Failed to optimize section '{section}': {e}", exc_info=True)

    if selected_skills:
        current = list(optimized.get("skills", []))
        for skill in selected_skills:
            if skill not in current:
                current.append(skill)
        optimized["skills"] = current

    return optimized


def optimize_section(
    section: str,
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    selected_skills: list[str],
    llm_client,
    selected_exp_bullets: list[str] = None,
) -> Any:
    """
    Optimize a specific resume section.
    For 'experience', only rewrites bullets selected by the user via checkboxes.
    All other bullets and all unselected jobs are returned unchanged.
    """
    section_lower = section.lower()
    prompt_template = SECTION_OPTIMIZATION_PROMPTS.get(section_lower)

    if not prompt_template:
        logger.warning(f"No prompt template for section: {section}")
        return resume_data.get(section_lower)

    if section_lower == "summary":
        prompt = prompt_template.format(
            current_summary=resume_data.get("summary", ""),
            job_title=jd_data.get("job_title", ""),
            required_skills=", ".join(jd_data.get("required_skills", [])[:8]),
            experience_years=resume_data.get("experience_years", ""),
        )

    elif section_lower == "skills":
        prompt = prompt_template.format(
            current_skills=", ".join(
                resume_data.get("skills", []) +
                resume_data.get("tools", []) +
                resume_data.get("frameworks", [])
            ),
            selected_skills=", ".join(selected_skills),
            required_skills=", ".join(jd_data.get("required_skills", [])),
        )

    elif section_lower == "experience":
        return _optimize_experience_selective(
            resume_data.get("experience", []),
            selected_exp_bullets or [],
            jd_data,
            llm_client,
            prompt_template,
        )

    elif section_lower == "projects":
        exp_summary = f"{resume_data.get('experience_years', 'some')} experience as "
        if resume_data.get("experience"):
            exp_summary += resume_data["experience"][0].get("role", "professional")
        prompt = prompt_template.format(
            current_projects=json.dumps(resume_data.get("projects", []), indent=2),
            selected_skills=", ".join(selected_skills),
            candidate_background=exp_summary,
            job_title=jd_data.get("job_title", ""),
        )

    else:
        return resume_data.get(section_lower)

    response = llm_client.complete(prompt)
    return _parse_section_response(section_lower, response)


def get_ats_explanation(
    score_data: dict[str, Any],
    gap_analysis: dict[str, Any],
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    llm_client,
) -> dict[str, Any]:
    """
    Get LLM-generated explanation for the ATS score.

    Args:
        score_data: ATS score data from ats_scoring module.
        gap_analysis: Skill gap analysis data.
        resume_data: Structured resume data.
        jd_data: Structured JD data.
        llm_client: LLM client instance.

    Returns:
        Dictionary with score explanation and suggestions.
    """
    prompt = ATS_SCORE_EXPLANATION_PROMPT.format(
        ats_score=score_data["final_score"],
        skill_score=score_data["skill_match_score"],
        experience_score=score_data["experience_score"],
        keyword_score=score_data["keyword_score"],
        tools_score=score_data["tools_score"],
        matched_skills=", ".join(gap_analysis.get("matched_skills", [])[:10]),
        missing_skills=", ".join(gap_analysis.get("missing_skills", [])[:10]),
        job_title=jd_data.get("job_title", ""),
        experience_required=jd_data.get("experience_required", ""),
        candidate_experience=resume_data.get("experience_years", ""),
    )

    response = llm_client.complete(prompt)
    explanation = _parse_json_response(response)

    if not explanation:
        # Fallback explanation
        explanation = _generate_fallback_explanation(score_data, gap_analysis)

    return explanation


def format_explanation_display(explanation: dict[str, Any]) -> str:
    """
    Format score explanation for markdown display.

    Args:
        explanation: Result from get_ats_explanation().

    Returns:
        Formatted markdown string.
    """
    lines = ["## 📊 Score Explanation", ""]

    overall = explanation.get("overall_feedback", "")
    if overall:
        lines.extend([overall, ""])

    strengths = explanation.get("strengths", [])
    if strengths:
        lines.append("### ✅ Strengths")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    weaknesses = explanation.get("weaknesses", [])
    if weaknesses:
        lines.append("### ⚠️ Areas for Improvement")
        for w in weaknesses:
            lines.append(f"- {w}")
        lines.append("")

    suggestions = explanation.get("improvement_suggestions", [])
    if suggestions:
        lines.append("### 💡 Recommendations")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")
        lines.append("")

    breakdown = explanation.get("score_breakdown_explanation", {})
    if breakdown:
        lines.append("### 📈 Component Details")
        for key, val in breakdown.items():
            label = key.replace("_", " ").title()
            lines.append(f"**{label}:** {val}")
        lines.append("")

    return "\n".join(lines)


def _optimize_experience_selective(
    experience: list[dict],
    selected_labels: list[str],
    jd_data: dict[str, Any],
    llm_client,
    prompt_template: str,
) -> list[dict]:
    """
    Rewrite only the bullets the user selected; leave everything else untouched.

    selected_labels come from the UI checkboxes in the format:
        '[Role @ Company] Improve: "bullet snippet…"'

    Strategy:
      1. Parse each label to extract role, company, and bullet snippet.
      2. For each job, collect only the bullets that were selected.
      3. Send those bullets to the LLM in one call per job (only jobs that
         have at least one selected bullet get an LLM call).
      4. Stitch the rewritten bullets back into their original positions;
         unselected bullets stay exactly as they were.
    """
    import re

    # Parse label → (role, company, snippet)
    def parse_label(label: str):
        m = re.match(r'\[(.+?) @ (.+?)\] Improve: "(.+?)…?"', label)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        return None, None, None

    # Group selected snippets by (role, company) key
    selections: dict[tuple, list[str]] = {}
    for label in selected_labels:
        role, company, snippet = parse_label(label)
        if role and company and snippet:
            key = (role, company)
            selections.setdefault(key, []).append(snippet)

    jd_keywords      = ", ".join(jd_data.get("keywords", []) + jd_data.get("required_skills", [])[:5])
    responsibilities = ", ".join(jd_data.get("responsibilities", [])[:5])

    result_experience = []

    for job in experience:
        role    = job.get("role", "")
        company = job.get("company", "")
        key     = (role, company)
        snippets_to_rewrite = selections.get(key, [])

        if not snippets_to_rewrite:
            # No bullets selected for this job — keep entirely unchanged
            result_experience.append(job)
            continue

        # Identify which bullet indices were selected by matching snippets
        original_bullets = job.get("bullets", [])
        selected_indices = set()
        for i, bullet in enumerate(original_bullets):
            for snippet in snippets_to_rewrite:
                if bullet.startswith(snippet.rstrip("…")):
                    selected_indices.add(i)
                    break

        if not selected_indices:
            result_experience.append(job)
            continue

        # Send only selected bullets to the LLM
        bullets_to_rewrite = [original_bullets[i] for i in sorted(selected_indices)]
        logger.info(
            f"Rewriting {len(bullets_to_rewrite)} bullet(s) for [{role} @ {company}]"
        )

        prompt = prompt_template.format(
            current_experience=json.dumps(
                [{"role": role, "company": company, "bullets": bullets_to_rewrite}],
                indent=2,
            ),
            jd_keywords=jd_keywords,
            responsibilities=responsibilities,
        )

        response  = llm_client.complete(prompt)
        parsed    = _parse_json_response(response)

        # Extract rewritten bullets from response
        rewritten_bullets = []
        if isinstance(parsed, list) and parsed:
            rewritten_bullets = parsed[0].get("bullets", []) if isinstance(parsed[0], dict) else []
        elif isinstance(parsed, dict):
            rewritten_bullets = parsed.get("bullets", [])

        if not rewritten_bullets:
            logger.warning(f"LLM returned no bullets for [{role} @ {company}] — keeping originals")
            result_experience.append(job)
            continue

        # Stitch rewritten bullets back; preserve original order and untouched bullets
        new_bullets = list(original_bullets)  # copy
        for list_pos, orig_idx in enumerate(sorted(selected_indices)):
            if list_pos < len(rewritten_bullets):
                new_bullets[orig_idx] = rewritten_bullets[list_pos]

        result_experience.append({**job, "bullets": new_bullets})

    return result_experience


def _parse_json_response(response: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not response:
        return None

    # Remove markdown code blocks
    import re
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()

    # Try to find JSON object
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(response[start:end+1])
        except json.JSONDecodeError:
            pass

    # Try array
    start = response.find('[')
    end = response.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(response[start:end+1])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM response")
    return None


def _parse_section_response(section: str, response: str) -> Any:
    """Parse section-specific LLM response."""
    if not response:
        return None

    if section == "summary":
        # Return as plain text
        import re
        clean = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        return clean.strip()

    elif section == "skills":
        # Parse as JSON array
        parsed = _parse_json_response(response)
        if isinstance(parsed, list):
            return parsed
        # Try to extract skills from text
        import re
        skills = re.findall(r'"([^"]+)"', response)
        return skills if skills else []

    else:
        # Parse as JSON
        return _parse_json_response(response)


def _apply_basic_optimization(
    resume_data: dict[str, Any],
    selected_skills: list[str]
) -> dict[str, Any]:
    """Apply basic optimization without LLM (fallback)."""
    optimized = resume_data.copy()

    # Add selected skills to skills list
    current_skills = optimized.get("skills", [])
    for skill in selected_skills:
        if skill not in current_skills:
            current_skills.append(skill)
    optimized["skills"] = current_skills

    return optimized


def _generate_fallback_explanation(
    score_data: dict[str, Any],
    gap_analysis: dict[str, Any]
) -> dict[str, Any]:
    """Generate a rule-based explanation when LLM fails."""
    weaknesses = []
    suggestions = []

    if score_data["skill_match_score"] < 60:
        missing = gap_analysis.get("missing_skills", [])[:5]
        weaknesses.append(f"Missing key skills: {', '.join(missing)}" if missing else "Low skill match with job requirements")
        suggestions.append("Add the missing required skills to your resume if you have them")

    if score_data["experience_score"] < 60:
        weaknesses.append("Experience level may not fully meet job requirements")
        suggestions.append("Emphasize relevant projects and transferable skills")

    if score_data["keyword_score"] < 60:
        weaknesses.append("Resume lacks important keywords from the job description")
        suggestions.append("Mirror language from the job description in your resume")

    if score_data["tools_score"] < 60:
        weaknesses.append("Tools and frameworks don't fully align with JD requirements")
        suggestions.append("Add matching tools and frameworks to your skills section")

    return {
        "overall_feedback": f"Your resume scores {score_data['final_score']}% ({score_data['grade']}) against this job description.",
        "strengths": [f"Matched {len(gap_analysis.get('matched_skills', []))} skills from the job description"],
        "weaknesses": weaknesses,
        "improvement_suggestions": suggestions,
        "score_breakdown_explanation": {
            "skill_match": f"{score_data['skill_match_score']}% of required skills found",
            "experience": f"{score_data['experience_score']}% experience relevance",
            "keyword_density": f"{score_data['keyword_score']}% keyword coverage",
            "tools_frameworks": f"{score_data['tools_score']}% tools/frameworks match",
        }
    }
