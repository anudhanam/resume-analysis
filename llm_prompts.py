"""
llm_prompts.py
All LLM prompt templates for AI Resume Optimizer.
"""

RESUME_EXTRACTION_PROMPT = """
You are a professional resume parser. Extract structured information from the resume text below.

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "name": "",
  "email": "",
  "phone": "",
  "summary": "",
  "skills": [],
  "tools": [],
  "frameworks": [],
  "experience_years": "",
  "experience": [
    {{
      "company": "",
      "role": "",
      "duration": "",
      "bullets": []
    }}
  ],
  "projects": [
    {{
      "name": "",
      "description": "",
      "technologies": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": []
}}

Resume Text:
{resume_text}
"""

JD_EXTRACTION_PROMPT = """
You are a job description parser. Extract structured information from the job description below.

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "job_title": "",
  "company": "",
  "required_skills": [],
  "preferred_skills": [],
  "tools": [],
  "frameworks": [],
  "experience_required": "",
  "responsibilities": [],
  "keywords": []
}}

Job Description Text:
{jd_text}
"""

SKILL_GAP_ANALYSIS_PROMPT = """
You are an expert technical recruiter performing a skill gap analysis.

Given the resume skills and job description requirements below, identify:
1. Matched skills (skills present in both resume and JD)
2. Missing skills (required/preferred JD skills not in resume)
3. Partial matches (semantically similar but not exact)
4. Additional skills (resume skills not mentioned in JD)

Return ONLY valid JSON (no markdown, no extra text):
{{
  "matched_skills": [],
  "missing_skills": [],
  "partial_matches": [
    {{"resume_skill": "", "jd_skill": "", "similarity": ""}}
  ],
  "additional_skills": [],
  "analysis_summary": ""
}}

Resume Skills: {resume_skills}
Resume Tools: {resume_tools}
Resume Frameworks: {resume_frameworks}

JD Required Skills: {required_skills}
JD Preferred Skills: {preferred_skills}
JD Tools: {jd_tools}
JD Frameworks: {jd_frameworks}
"""

ATS_SCORE_EXPLANATION_PROMPT = """
You are an ATS (Applicant Tracking System) expert. Based on the analysis below, explain WHY the candidate scored this score and what they should improve.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "overall_feedback": "",
  "strengths": [],
  "weaknesses": [],
  "improvement_suggestions": [],
  "score_breakdown_explanation": {{
    "skill_match": "",
    "experience": "",
    "keyword_density": "",
    "tools_frameworks": ""
  }}
}}

ATS Score: {ats_score}%
Skill Match Score: {skill_score}%
Experience Score: {experience_score}%
Keyword Score: {keyword_score}%
Tools Score: {tools_score}%

Matched Skills: {matched_skills}
Missing Skills: {missing_skills}
Job Title: {job_title}
Required Experience: {experience_required}
Candidate Experience: {candidate_experience}
"""

RESUME_OPTIMIZATION_PROMPT = """
You are an expert resume writer and ATS optimization specialist.

Given the original resume and job description, create an optimized version of the resume that:
1. Naturally incorporates the selected new skills: {selected_skills}
2. Improves the professional summary to align with the JD
3. Adds or enhances experience bullet points with impact metrics
4. Adds 1-2 realistic projects that demonstrate the selected skills
5. Uses strong action verbs and ATS-friendly keywords
6. Keeps all information realistic and believable based on existing experience
7. Maintains the candidate's actual experience level

IMPORTANT RULES:
- Do NOT fabricate job titles, companies, or dates
- Keep all additions realistic and proportional to existing experience
- Preserve the candidate's real accomplishments
- Only enhance descriptions, never invent fake history

Return ONLY valid JSON (no markdown, no extra text):
{{
  "name": "",
  "email": "",
  "phone": "",
  "summary": "",
  "skills": [],
  "tools": [],
  "frameworks": [],
  "experience": [
    {{
      "company": "",
      "role": "",
      "duration": "",
      "bullets": []
    }}
  ],
  "projects": [
    {{
      "name": "",
      "description": "",
      "technologies": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": []
}}

Original Resume (JSON):
{resume_json}

Job Description (JSON):
{jd_json}

Sections to optimize: {sections_to_optimize}
"""

SECTION_OPTIMIZATION_PROMPTS = {
    "summary": """
Rewrite this professional summary to better align with the job description.
Use strong opening, mention key relevant skills, and quantify impact where possible.
Keep it to 3-4 sentences maximum.

Current Summary: {current_summary}
Job Title: {job_title}
Key Required Skills: {required_skills}
Candidate Experience: {experience_years}

Return ONLY the new summary text, no JSON, no labels.
""",

    "skills": """
Reorganize and optimize this skills section for ATS.
Add the selected new skills naturally. Group related skills.
Remove irrelevant skills. Prioritize skills matching the JD.

Current Skills: {current_skills}
Selected New Skills: {selected_skills}
JD Required Skills: {required_skills}

Return ONLY a JSON array of skills: ["skill1", "skill2", ...]
""",

    "experience": """
Enhance these experience bullet points to be more impactful and ATS-friendly.
Add quantified metrics where reasonable. Use strong action verbs.
Naturally weave in relevant keywords from the JD.

Current Experience:
{current_experience}

JD Keywords: {jd_keywords}
JD Responsibilities: {responsibilities}

Return ONLY valid JSON array matching the input structure.
""",

    "projects": """
Enhance existing projects and add 1-2 new realistic projects that demonstrate the selected skills.
Projects should be believable given the candidate's background.

Current Projects: {current_projects}
Selected New Skills: {selected_skills}
Candidate Background: {candidate_background}
Job Title Target: {job_title}

Return ONLY valid JSON array of projects.
"""
}
