"""Markdown Renderer — JSON → Markdown.

Reads only from the final JSON. Image paths are resolved relative to the JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_markdown(json_path: str | Path, config: dict[str, Any]) -> str:
    """Render a Markdown document from the exam JSON."""
    json_path = Path(json_path)
    render_cfg = config.get("render", {})
    math_delim = render_cfg.get("md_math_delimiter", "$$")
    include_warnings = render_cfg.get("include_review_warnings", True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines: list[str] = []
    exam = data.get("exam", {})
    title = exam.get("exam_title") or exam.get("exam_code") or "Exam"
    lines.append(f"# {title}\n")

    for k in ("subject", "exam_code", "exam_date", "level"):
        if exam.get(k):
            lines.append(f"**{k.replace('_',' ').title()}:** {exam[k]}")
    lines.append("")

    # Sections table
    sections = data.get("sections", [])
    if sections:
        lines.append("## Sections\n")
        lines.append("| # | Section | Questions | Marks |")
        lines.append("|---|---------|-----------|-------|")
        for s in sections:
            name = s.get("section_name") or s.get("section_id", "")
            lines.append(f"| {s.get('section_number','')} | {name} | {s.get('num_questions',0)} | {s.get('section_marks',0)} |")
        lines.append("")

    lines.append("## Questions\n")
    current_section = None

    for q in data.get("questions", []):
        sid = q.get("section_id")
        if sid and sid != current_section:
            current_section = sid
            lines.append(f"### Section: {sid}\n")

        qnum = q.get("question_number", "")
        qtype = q.get("question_type", "")
        lines.append(f"#### Q{qnum} [{qtype}] (Marks: +{q.get('correct_marks','')} / -{q.get('negative_marks',0)})")
        lines.append(f"*ID: {q.get('question_id','')}*\n")

        ext_meta = q.get("extraction_metadata", {})
        if include_warnings and ext_meta.get("needs_review"):
            lines.append(f"> ⚠️ **Needs review:** {ext_meta.get('review_reason','Unknown')}\n")

        # Comprehension
        comp = q.get("comprehension")
        if comp:
            if comp.get("comprehension_text"):
                lines.append(f"**Comprehension Passage:**\n\n{comp['comprehension_text']}\n")
            if comp.get("comprehension_text_latex"):
                lines.append(f"{math_delim}\n{comp['comprehension_text_latex']}\n{math_delim}\n")
            for fig in comp.get("figures", []):
                _fig_md(fig, lines)

        if q.get("question_text"):
            lines.append(f"{q['question_text']}\n")
        if q.get("question_text_latex"):
            lines.append(f"{math_delim}\n{q['question_text_latex']}\n{math_delim}\n")
        elif q.get("math_crop_path"):
            lines.append(f"![math expression]({q['math_crop_path']})\n")

        for fig in q.get("figures", []):
            _fig_md(fig, lines)

        # MCQ
        mcq = q.get("mcq")
        if mcq:
            lines.append("**Options:**\n")
            for o in mcq.get("options", []):
                c = " ✓" if o.get("is_correct") else ""
                t = o.get("option_text", "")
                if o.get("option_text_latex"):
                    t += f" $${o['option_text_latex']}$$"
                elif o.get("option_image_path") and not t:
                    t = f"![option]({o['option_image_path']})"
                lines.append(f"- **{o['option_id']}.** {t}{c}")
            lines.append("")
            if mcq.get("correct_option_id"):
                lines.append(f"**Correct Answer:** {mcq['correct_option_id']}\n")

        # MSQ
        msq = q.get("msq")
        if msq:
            lines.append("**Options (Multiple Select):**\n")
            for o in msq.get("options", []):
                c = " ✓" if o.get("is_correct") else ""
                lines.append(f"- **{o['option_id']}.** {o.get('option_text','')}{c}")
            lines.append("")
            if msq.get("correct_option_ids"):
                lines.append(f"**Correct Answers:** {', '.join(msq['correct_option_ids'])}\n")

        # SA
        sa = q.get("sa")
        if sa:
            for k in ("response_type", "answers_type"):
                if sa.get(k):
                    lines.append(f"**{k.replace('_',' ').title()}:** {sa[k]}")
            ar = sa.get("answer_range")
            if ar:
                lines.append(f"**Accepted range:** {ar['min']} to {ar['max']}")
            if sa.get("possible_answers"):
                lines.append("**Possible Answers:**\n")
                for a in sa["possible_answers"]:
                    lines.append(f"- {a}")
            if sa.get("answer_truncated_across_page"):
                lines.append("\n> ⚠️ Answer may be truncated across page boundary")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


def _fig_md(fig: dict, lines: list[str]) -> None:
    alt = fig.get("alt_text") or fig.get("figure_description") or "figure"
    lines.append(f"![{alt}]({fig.get('image_asset_path','')})")
    if fig.get("figure_description"):
        lines.append(f"*{fig['figure_description']}*")
    lines.append("")
