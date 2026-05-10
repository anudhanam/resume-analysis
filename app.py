"""
app.py
AI Resume Optimizer — Gradio Application (Gemini 2.5 Flash)

Run with: python app.py
"""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from pdf_parser import extract_text_from_pdf
from skill_analyzer import (
    perform_skill_gap_analysis,
    calculate_skill_match_score,
    format_gap_analysis_display,
)
from ats_scoring import calculate_ats_score
from resume_optimizer import optimize_resume, get_ats_explanation, format_explanation_display
from llm_prompts import RESUME_EXTRACTION_PROMPT, JD_EXTRACTION_PROMPT

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Initial app state ─────────────────────────────────────────────────────────
INITIAL_STATE = {
    "resume_data":      None,
    "jd_data":          None,
    "resume_text":      "",
    "gap_analysis":     None,
    "score_data":       None,
    "optimized_resume": None,
    "llm_client":       None,
}

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
footer { display: none !important; }
.status-box {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 12px 16px; border-radius: 8px;
    font-size: 0.92rem; line-height: 1.5; margin: 8px 0;
}
.status-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.status-error   { background: #2d1515; border: 1px solid #7f1d1d; color: #fca5a5; }
.status-success { background: #0f2d1a; border: 1px solid #14532d; color: #86efac; }
.status-info    { background: #0f1e2d; border: 1px solid #1e3a5f; color: #93c5fd; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers — one shared builder, three named shortcuts
# ─────────────────────────────────────────────────────────────────────────────

def _status(kind: str, icon: str, msg: str) -> str:
    return (
        f'<div class="status-box status-{kind}">'
        f'<span class="status-icon">{icon}</span> {msg}</div>'
    )

def status_error(msg):   return _status("error",   "✗", msg)
def status_info(msg):    return _status("info",    "⋯", msg)
def status_success(msg): return _status("success", "✓", msg)


# ─────────────────────────────────────────────────────────────────────────────
# LLM client — flat factory, no class needed
# ─────────────────────────────────────────────────────────────────────────────

def create_client():
    """
    Load GOOGLE_API_KEY and return (client_namespace, "ok") or (None, error_msg).
    The returned namespace exposes .complete() and .extract_json().
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return None, "GOOGLE_API_KEY not found. Add it to your .env file."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except ImportError:
        return None, "google-generativeai not installed. Run: pip install google-generativeai"
    except Exception as exc:
        return None, str(exc)

    import json
    import re as _re

    def complete(prompt, system=None, max_tokens=4096, temperature=0.2):
        full = f"{system}\n\n{prompt}" if system else prompt
        cfg  = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        try:
            resp = model.generate_content(
                full, generation_config=cfg,
                request_options={"timeout": 120},
            )
            return resp.text
        except Exception as exc:
            logger.error(f"Gemini error: {exc}")
            raise

    def extract_json(prompt):
        system = (
            "You are an expert resume analyst. "
            "Always respond with ONLY valid JSON — no markdown fences, no preamble."
        )
        text = complete(prompt, system=system)
        if not text:
            return None
        text = _re.sub(r"```json\s*|```\s*", "", text).strip()
        for s, e in [("{", "}"), ("[", "]")]:
            i, j = text.find(s), text.rfind(e)
            if i != -1 and j > i:
                try:
                    return json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    pass
        logger.warning("Could not parse JSON from Gemini response")
        return None

    return SimpleNamespace(complete=complete, extract_json=extract_json), "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def build_score_chart(score_data: dict) -> go.Figure:
    final      = score_data.get("final_score", 0)
    skill      = score_data.get("skill_match_score", 0)
    experience = score_data.get("experience_score", 0)
    keyword    = score_data.get("keyword_score", 0)
    tools      = score_data.get("tools_score", 0)
    grade      = score_data.get("grade", "")

    BG, TRACK, TEXT_COL, MUTED = "#0f172a", "#1e293b", "#f1f5f9", "#94a3b8"

    def bar_color(s):
        if s >= 80: return "#4ade80"
        if s >= 60: return "#facc15"
        if s >= 40: return "#fb923c"
        return "#f87171"

    sub_rows = [
        ("🎯  Skill Match",        skill,      "40 %"),
        ("💼  Experience",         experience, "25 %"),
        ("🔍  Keyword Density",    keyword,    "20 %"),
        ("🛠️  Tools & Frameworks", tools,      "15 %"),
    ]
    labels = ["<b>⚡ Overall ATS Score</b>", ""] + [r[0] for r in sub_rows]
    values = [final, 0, *[r[1] for r in sub_rows]]
    colors = [bar_color(final), BG, *[bar_color(r[1]) for r in sub_rows]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[100] * len(labels), y=labels, orientation="h",
        marker_color=TRACK, showlegend=False, hoverinfo="skip",
        width=[0.7, 0, *[0.5] * len(sub_rows)],
    ))
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0, showlegend=False,
        width=[0.7, 0, *[0.5] * len(sub_rows)],
        hovertemplate="%{y}: <b>%{x}%</b><extra></extra>",
        text=[f"<b>{v}%</b>" if v else "" for v in values],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=13, color="#0f172a", family="Inter, sans-serif"),
    ))
    fig.add_annotation(
        x=final + 1.5, y="<b>⚡ Overall ATS Score</b>",
        text=f"  <b>{grade}</b>", showarrow=False,
        font=dict(size=13, color=bar_color(final), family="Inter, sans-serif"),
        xanchor="left", yanchor="middle",
    )
    for lbl, w in zip([r[0] for r in sub_rows], [r[2] for r in sub_rows]):
        fig.add_annotation(
            x=103, y=lbl,
            text=f"<span style='color:{MUTED};font-size:10px'>{w}</span>",
            showarrow=False,
            font=dict(size=10, color=MUTED, family="Inter, sans-serif"),
            xanchor="left", yanchor="middle",
        )
    fig.update_layout(
        barmode="overlay", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=50, b=20, l=10, r=60), height=340,
        xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False,
                   zeroline=False, showline=False),
        yaxis=dict(autorange="reversed",
                   tickfont=dict(size=12, color=TEXT_COL, family="Inter, sans-serif"),
                   showgrid=False, zeroline=False, showline=False),
        font=dict(family="Inter, sans-serif", color=TEXT_COL),
        title=dict(
            text=(
                f"ATS Compatibility Score  ·  "
                f"<span style='color:{bar_color(final)}'>{final}%</span>  "
                f"<span style='color:{MUTED}'>({grade})</span>"
            ),
            font=dict(size=15, color=TEXT_COL, family="Inter, sans-serif"),
            x=0.01, xanchor="left",
        ),
    )
    return fig


def build_empty_chart() -> go.Figure:
    BG = "#0f172a"
    fig = go.Figure()
    fig.add_annotation(
        text="Run analysis to see your ATS score chart",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color="#94a3b8", family="Inter, sans-serif"),
        xanchor="center", yanchor="middle",
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=20, b=20, l=20, r=20), height=340,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_skill_tags(skills: list) -> str:
    """Render a list of skills as HTML pill badges."""
    if not skills:
        return '<p style="color:#94a3b8;font-size:0.85rem;margin:4px 0">No skills added yet.</p>'
    pills = "".join(
        f'<span style="display:inline-block;background:#1e3a5f;color:#93c5fd;'
        f'border:1px solid #2563eb;border-radius:999px;padding:2px 12px;'
        f'margin:3px 4px;font-size:0.82rem">{s}</span>'
        for s in skills
    )
    return f'<div style="margin:6px 0">{pills}</div>'


def _format_resume_preview(d: dict) -> str:
    """Render structured resume data as a markdown string."""
    lines = []
    if d.get("name"):
        lines.append(f"# {d['name']}")
    contact = "  |  ".join(filter(None, [d.get("email", ""), d.get("phone", "")]))
    if contact:
        lines += [f"**{contact}**", ""]
    if d.get("summary"):
        lines += ["## Professional Summary", d["summary"], ""]
    skills = list(dict.fromkeys(
        s for s in d.get("skills", []) + d.get("tools", []) + d.get("frameworks", []) if s
    ))
    if skills:
        lines += ["## Technical Skills", " • ".join(skills), ""]
    if d.get("experience"):
        lines.append("## Experience")
    for job in d.get("experience", []):
        lines.append(
            f"**{job.get('role', '')}** — {job.get('company', '')}  |  "
            f"_{job.get('duration', '')}_"
        )
        for b in job.get("bullets", [])[:4]:
            if b:
                lines.append(f"- {b}")
        lines.append("")
    if d.get("projects"):
        lines.append("## Projects")
    for p in d.get("projects", []):
        lines.append(f"**{p.get('name', '')}**")
        if p.get("description"):
            lines.append(p["description"])
        if p.get("technologies"):
            lines.append(f"*Tech: {', '.join(p['technologies'])}*")
        lines.append("")
    for edu in d.get("education", []):
        lines.append(
            f"**{edu.get('degree', '')}** — {edu.get('institution', '')}  |  {edu.get('year', '')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core action: run_analysis
# Single function — returns all 14 UI outputs at once, no chained .then() steps
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(resume_file, jd_file, state: dict):
    """
    Full pipeline: PDF extraction → LLM parsing → skill gap → ATS score → explanation.

    Returns 14 values mapped directly to UI outputs:
        analyze_btn, status_output,
        state, score_chart, gap_analysis_output, explanation_output,
        tabs,
        req_dropdown, pref_dropdown, exp_checkboxes,
        req_added_state, pref_added_state,
        req_tags_html, pref_tags_html
    """
    empty_tags = _render_skill_tags([])

    def err(msg):
        """Return a uniform error tuple that resets all outputs."""
        return (
            gr.update(value="🔍 Analyze Resume", interactive=True),
            status_error(msg),
            state, build_empty_chart(), "", "",
            gr.update(),                          # tabs — no navigation on error
            gr.update(choices=[], value=None),    # req_dropdown
            gr.update(choices=[], value=None),    # pref_dropdown
            gr.update(choices=[], value=[]),      # exp_checkboxes
            [], [],                               # req/pref added state
            empty_tags, empty_tags,               # tag html
        )

    if not resume_file:
        return err("Please upload a Resume PDF.")
    if not jd_file:
        return err("Please upload a Job Description PDF.")

    client, msg = create_client()
    if not client:
        return err(msg)

    state = state.copy()
    state["llm_client"] = client

    try:
        resume_path = resume_file.name if hasattr(resume_file, "name") else resume_file
        jd_path     = jd_file.name     if hasattr(jd_file,     "name") else jd_file

        resume_text = extract_text_from_pdf(resume_path)
        jd_text     = extract_text_from_pdf(jd_path)

        if not resume_text.strip():
            return err("Could not extract text from Resume PDF. Is it image-based?")
        if not jd_text.strip():
            return err("Could not extract text from Job Description PDF.")

        state["resume_text"] = resume_text

        resume_data = client.extract_json(
            RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
        ) or {}
        jd_data = client.extract_json(
            JD_EXTRACTION_PROMPT.format(jd_text=jd_text)
        ) or {}

        state["resume_data"] = resume_data
        state["jd_data"]     = jd_data

        gap_analysis = perform_skill_gap_analysis(resume_data, jd_data)
        state["gap_analysis"] = gap_analysis

        score_data = calculate_ats_score(
            resume_data, jd_data, resume_text,
            calculate_skill_match_score(gap_analysis),
        )
        state["score_data"] = score_data

        explanation  = get_ats_explanation(score_data, gap_analysis, resume_data, jd_data, client)
        gap_display  = format_gap_analysis_display(gap_analysis)
        expl_display = format_explanation_display(explanation)
        chart        = build_score_chart(score_data)

        # Build Optimize-tab option lists directly here — no separate function needed
        req_choices  = gap_analysis.get("missing_required",  [])
        pref_choices = gap_analysis.get("missing_preferred", [])
        exp_options  = [
            f'[{job.get("role", "")} @ {job.get("company", "")}] '
            f'Improve: "{b[:60]}{"…" if len(b) > 60 else ""}"'
            for job in resume_data.get("experience", [])[:3]
            for b in job.get("bullets", [])[:3]
            if b
        ]

        status = status_success(
            f"Analysis complete &nbsp;·&nbsp; "
            f"<b>{resume_data.get('name', 'Candidate')}</b> → "
            f"<b>{jd_data.get('job_title', 'the role')}</b> &nbsp;·&nbsp; "
            f"ATS Score: <b>{score_data['final_score']}%</b> ({score_data['grade']})"
        )

        return (
            gr.update(value="🔍 Analyze Resume", interactive=True),
            status,
            state, chart, gap_display, expl_display,
            gr.update(selected=1),                        # auto-switch to ATS Score tab
            gr.update(choices=req_choices,  value=None),
            gr.update(choices=pref_choices, value=None),
            gr.update(choices=exp_options,  value=[]),
            [], [],
            _render_skill_tags([]), _render_skill_tags([]),
        )

    except Exception as exc:
        logger.error(f"Analysis failed: {exc}", exc_info=True)
        return err(f"Analysis failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Core action: optimize_resume_action
# ─────────────────────────────────────────────────────────────────────────────

def optimize_resume_action(req_added, pref_added, exp_improvements, sections_selected, state):
    """Merge inputs, call LLM optimization, return updated state and preview."""
    if not state.get("resume_data"):
        return state, status_error("Run Analyze first before optimizing."), gr.update()
    if not state.get("llm_client"):
        return state, status_error("LLM client not ready. Re-run Analyze."), gr.update()

    selected_skills   = list(dict.fromkeys((req_added or []) + (pref_added or [])))
    sections_selected = list(sections_selected or [])
    if exp_improvements and "Experience" not in sections_selected:
        sections_selected.append("Experience")
    sections = [s.lower() for s in sections_selected] or ["summary", "skills", "experience", "projects"]

    try:
        optimized = optimize_resume(
            resume_data            = state["resume_data"],
            jd_data                = state["jd_data"],
            selected_skills        = selected_skills,
            sections_to_optimize   = sections,
            llm_client             = state["llm_client"],
            selected_exp_bullets   = list(exp_improvements or []),
        )
        state = {**state, "optimized_resume": optimized}

        status = status_success(
            f"Resume optimized &nbsp;·&nbsp; <b>{len(selected_skills)}</b> skill(s) added "
            f"&nbsp;·&nbsp; <b>{len(sections)}</b> section(s) rewritten"
        )
        return state, status, gr.update(value=_format_resume_preview(optimized), visible=True)

    except Exception as exc:
        logger.error(f"Optimization failed: {exc}", exc_info=True)
        return state, status_error(f"Optimization failed: {exc}"), gr.update()


# ─────────────────────────────────────────────────────────────────────────────
# Skill add handler — one function shared by both dropdowns
# ─────────────────────────────────────────────────────────────────────────────

def add_skill(dropdown_val, current_list: list):
    """Append dropdown_val to current_list if not already present."""
    updated = list(current_list or [])
    if dropdown_val and dropdown_val not in updated:
        updated.append(dropdown_val)
    return updated, _render_skill_tags(updated)


# ─────────────────────────────────────────────────────────────────────────────
# UI layout
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        css=CSS,
        title="AI Resume Optimizer",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate"),
    ) as app:

        state = gr.State(INITIAL_STATE.copy())
        gr.Markdown("# 🎯 AI Resume Optimizer")

        with gr.Tabs() as tabs:

            # ── Tab 1: Upload & Analyze ───────────────────────────────────────
            with gr.Tab("📤 Upload & Analyze"):
                gr.Markdown("### 📂 Upload both PDFs and click Analyze.")
                status_output = gr.HTML("")
                with gr.Row():
                    resume_upload = gr.File(label="📄 Resume (PDF)",          file_types=[".pdf"])
                    jd_upload     = gr.File(label="📋 Job Description (PDF)", file_types=[".pdf"])
                analyze_btn = gr.Button("🔍 Analyze Resume", variant="primary", size="lg")

            # ── Tab 2: ATS Score ──────────────────────────────────────────────
            with gr.Tab("📊 ATS Score"):
                score_chart = gr.Plot(value=build_empty_chart(), show_label=False)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔎 Skill Gap")
                        gap_analysis_output = gr.Markdown("*Run analysis to see results.*")
                    with gr.Column():
                        gr.Markdown("### 💡 Explanation")
                        explanation_output = gr.Markdown("*Run analysis to see explanation.*")

            # ── Tab 3: Optimize ───────────────────────────────────────────────
            with gr.Tab("⚡ Optimize"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("*Run Analyze first, then configure improvements below.*")

                        gr.Markdown("### ❌ Missing Required Skills")
                        with gr.Row():
                            req_dropdown = gr.Dropdown(
                                choices=[], value=None, label="Select a required skill",
                                interactive=True, scale=3,
                            )
                            req_add_btn = gr.Button("➕ Add", variant="primary", size="sm", scale=1)
                        req_tags_html   = gr.HTML(_render_skill_tags([]))
                        req_added_state = gr.State([])

                        gr.Markdown("---")

                        gr.Markdown("### ⚠️ Missing Preferred Skills")
                        with gr.Row():
                            pref_dropdown = gr.Dropdown(
                                choices=[], value=None, label="Select a preferred skill",
                                interactive=True, scale=3,
                            )
                            pref_add_btn = gr.Button("➕ Add", variant="primary", size="sm", scale=1)
                        pref_tags_html   = gr.HTML(_render_skill_tags([]))
                        pref_added_state = gr.State([])

                        gr.Markdown("---")

                        gr.Markdown("### 💼 Experience Improvements")
                        exp_checkboxes = gr.CheckboxGroup(
                            choices=[], value=[],
                            label="Select bullet points to improve",
                            info="Tick bullets you want Gemini to rewrite with stronger impact.",
                        )

                        gr.Markdown("---")

                        sections_checkboxes = gr.CheckboxGroup(
                            choices=["Summary", "Skills", "Projects", "Experience"],
                            value=["Summary", "Skills", "Projects", "Experience"],
                            label="Sections to optimize",
                        )
                        optimize_btn    = gr.Button("⚡ Apply Improvements", variant="primary", size="lg")
                        optimize_status = gr.HTML("")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📄 Optimized Resume Preview")
                        resume_preview = gr.Markdown(
                            "*Configure improvements on the left and click Apply.*"
                        )

        # ── Event handlers ────────────────────────────────────────────────────

        # Analyze:
        #   Step 1 — instant UI feedback (queue=False so it fires immediately)
        #   Step 2 — full pipeline, re-enables button and updates all outputs
        analyze_btn.click(
            fn=lambda: (
                gr.update(value="⏳ Analyzing…", interactive=False),
                status_info("Extracting text, running Gemini analysis… please wait."),
            ),
            outputs=[analyze_btn, status_output],
            queue=False,
        ).then(
            fn=run_analysis,
            inputs=[resume_upload, jd_upload, state],
            outputs=[
                analyze_btn, status_output,
                state, score_chart, gap_analysis_output, explanation_output,
                tabs,
                req_dropdown, pref_dropdown, exp_checkboxes,
                req_added_state, pref_added_state,
                req_tags_html, pref_tags_html,
            ],
        )

        # Optimize:
        #   Step 1 — instant UI feedback
        #   Step 2 — LLM rewrite
        #   Step 3 — re-enable button
        optimize_btn.click(
            fn=lambda: (
                gr.update(value="⏳ Optimizing…", interactive=False),
                status_info("Rewriting resume sections with Gemini… please wait."),
            ),
            outputs=[optimize_btn, optimize_status],
            queue=False,
        ).then(
            fn=optimize_resume_action,
            inputs=[req_added_state, pref_added_state, exp_checkboxes, sections_checkboxes, state],
            outputs=[state, optimize_status, resume_preview],
        ).then(
            fn=lambda: gr.update(value="⚡ Apply Improvements", interactive=True),
            outputs=[optimize_btn],
            queue=False,
        )

        # Add-skill buttons — both use the same handler
        req_add_btn.click(
            fn=add_skill,
            inputs=[req_dropdown, req_added_state],
            outputs=[req_added_state, req_tags_html],
            queue=False,
        )
        pref_add_btn.click(
            fn=add_skill,
            inputs=[pref_dropdown, pref_added_state],
            outputs=[pref_added_state, pref_tags_html],
            queue=False,
        )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False)
