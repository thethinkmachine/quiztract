"""HTML Renderer — JSON → HTML.

Uses Jinja2 template. Reads only from the final JSON.
KaTeX loaded from CDN with local fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_html(json_path: str | Path, config: dict[str, Any]) -> str:
    """Render an HTML document from the exam JSON."""
    json_path = Path(json_path)
    render_cfg = config.get("render", {})
    katex_cdn = render_cfg.get("katex_cdn", "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist")
    include_warnings = render_cfg.get("include_review_warnings", True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try Jinja2 template first
    template_dir = Path(__file__).parent / "templates"
    if (template_dir / "exam.html.j2").exists():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("exam.html.j2")
        return template.render(
            data=data,
            katex_cdn=katex_cdn,
            include_warnings=include_warnings,
        )

    # Fallback: inline HTML generation
    return _generate_html_inline(data, katex_cdn, include_warnings)


def _generate_html_inline(data: dict, katex_cdn: str, include_warnings: bool) -> str:
    """Generate HTML inline when template is not available."""
    exam = data.get("exam", {})
    title = exam.get("exam_title") or exam.get("exam_code") or "Exam"

    questions_html = []
    for q in data.get("questions", []):
        questions_html.append(_render_question(q, include_warnings))

    nav_items = []
    for q in data.get("questions", []):
        qnum = q.get("question_number", "")
        qid = q.get("question_id", "")
        review = " ⚠️" if q.get("extraction_metadata", {}).get("needs_review") else ""
        nav_items.append(f'<a href="#q-{qid}">Q{qnum}{review}</a>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet" href="{katex_cdn}/katex.min.css">
<script defer src="{katex_cdn}/katex.min.js"></script>
<script defer src="{katex_cdn}/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters:[
        {{left:'$$',right:'$$',display:true}},
        {{left:'$',right:'$',display:false}}
    ]}});"></script>
{_get_css()}
</head>
<body>
<nav class="sidebar">
<h3>Questions</h3>
{''.join(nav_items)}
</nav>
<main>
<h1>{title}</h1>
{_render_exam_meta(exam)}
{_render_sections(data.get("sections", []))}
<h2>Questions</h2>
{''.join(questions_html)}
</main>
</body>
</html>"""


def _get_css() -> str:
    return """<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;line-height:1.6;color:#1a1a2e;background:#f8f9fa;display:flex}
.sidebar{position:fixed;left:0;top:0;width:220px;height:100vh;background:#1a1a2e;color:#fff;padding:20px;overflow-y:auto}
.sidebar h3{margin-bottom:12px;font-size:14px;text-transform:uppercase;letter-spacing:1px;color:#8892b0}
.sidebar a{display:block;color:#ccd6f6;text-decoration:none;padding:4px 8px;font-size:13px;border-radius:4px;margin-bottom:2px}
.sidebar a:hover{background:rgba(255,255,255,.1)}
main{margin-left:240px;max-width:900px;padding:40px}
h1{font-size:28px;margin-bottom:16px;color:#1a1a2e}
h2{font-size:22px;margin:32px 0 16px;color:#1a1a2e;border-bottom:2px solid #e8e8e8;padding-bottom:8px}
h3{font-size:18px;margin:24px 0 12px;color:#495057}
.question{background:#fff;border-radius:8px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.question.needs-review{border-left:4px solid #ffc107}
.q-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.q-type{background:#e3f2fd;color:#1565c0;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.q-marks{color:#666;font-size:13px}
.warning{background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:12px;margin:12px 0;font-size:14px}
.warning::before{content:'⚠️ '}
.options{list-style:none;padding:0;margin:12px 0}
.options li{padding:8px 12px;margin:4px 0;background:#f8f9fa;border-radius:6px;font-size:14px}
.options li.correct{background:#d4edda;border:1px solid #28a745}
.answer-info{background:#f0f4f8;padding:12px;border-radius:6px;margin:12px 0;font-size:14px}
figure{margin:16px 0;text-align:center}
figure img{max-width:100%;border-radius:4px;border:1px solid #dee2e6}
figcaption{color:#666;font-size:13px;margin-top:8px;font-style:italic}
.meta-table{width:100%;border-collapse:collapse;margin:16px 0}
.meta-table th,.meta-table td{padding:8px 12px;border:1px solid #dee2e6;text-align:left;font-size:14px}
.meta-table th{background:#f8f9fa;font-weight:600}
@media print{.sidebar{display:none}main{margin-left:0}body{background:#fff}}
</style>"""


def _render_exam_meta(exam: dict) -> str:
    parts = []
    for k, label in [("subject","Subject"),("exam_code","Code"),("exam_date","Date"),("level","Level")]:
        if exam.get(k):
            parts.append(f"<p><strong>{label}:</strong> {exam[k]}</p>")
    return "\n".join(parts)


def _render_sections(sections: list) -> str:
    if not sections:
        return ""
    rows = ""
    for s in sections:
        name = s.get("section_name") or s.get("section_id","")
        rows += f"<tr><td>{s.get('section_number','')}</td><td>{name}</td><td>{s.get('num_questions',0)}</td><td>{s.get('section_marks',0)}</td></tr>"
    return f"""<h2>Sections</h2>
<table class="meta-table"><thead><tr><th>#</th><th>Section</th><th>Questions</th><th>Marks</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def _render_question(q: dict, include_warnings: bool) -> str:
    qid = q.get("question_id","")
    qnum = q.get("question_number","")
    qtype = q.get("question_type","")
    needs_review = q.get("extraction_metadata",{}).get("needs_review", False)
    cls = "question needs-review" if needs_review else "question"

    html = f'<div class="{cls}" id="q-{qid}">'
    html += f'<div class="q-header"><span>Q{qnum} <span class="q-type">{qtype}</span></span>'
    html += f'<span class="q-marks">+{q.get("correct_marks","")} / -{q.get("negative_marks",0)}</span></div>'

    if include_warnings and needs_review:
        reason = q.get("extraction_metadata",{}).get("review_reason","")
        html += f'<div class="warning">Needs review: {reason}</div>'

    # Comprehension
    comp = q.get("comprehension")
    if comp:
        if comp.get("comprehension_text"):
            html += f'<div class="answer-info"><strong>Passage:</strong><br>{comp["comprehension_text"]}</div>'
        if comp.get("comprehension_text_latex"):
            html += f'<p>$${comp["comprehension_text_latex"]}$$</p>'
        for fig in comp.get("figures",[]):
            html += _render_figure_html(fig)

    if q.get("question_text"):
        html += f'<p>{q["question_text"]}</p>'
    if q.get("question_text_latex"):
        html += f'<p>$${q["question_text_latex"]}$$</p>'
    elif q.get("math_crop_path"):
        html += f'<img src="{q["math_crop_path"]}" alt="math expression">'

    for fig in q.get("figures",[]):
        html += _render_figure_html(fig)

    # MCQ / MSQ options
    for key in ("mcq","msq"):
        block = q.get(key)
        if not block:
            continue
        label = "Options" if key == "mcq" else "Options (Multiple Select)"
        html += f'<p><strong>{label}:</strong></p><ul class="options">'
        for o in block.get("options",[]):
            c = ' class="correct"' if o.get("is_correct") else ""
            t = o.get("option_text","")
            if o.get("option_text_latex"):
                t += f" ${o['option_text_latex']}$"
            elif o.get("option_image_path") and not t:
                t = f'<img src="{o["option_image_path"]}" alt="option">'
            html += f'<li{c}><strong>{o["option_id"]}.</strong> {t}</li>'
        html += "</ul>"
        ans_key = "correct_option_id" if key == "mcq" else "correct_option_ids"
        ans = block.get(ans_key)
        if ans:
            val = ans if isinstance(ans, str) else ", ".join(ans)
            html += f'<p><strong>Answer:</strong> {val}</p>'

    # SA
    sa = q.get("sa")
    if sa:
        html += '<div class="answer-info">'
        for k, l in [("response_type","Response"),("answers_type","Type")]:
            if sa.get(k):
                html += f"<p><strong>{l}:</strong> {sa[k]}</p>"
        ar = sa.get("answer_range")
        if ar:
            html += f"<p><strong>Range:</strong> {ar['min']} to {ar['max']}</p>"
        if sa.get("possible_answers"):
            html += "<p><strong>Answers:</strong></p><ul>"
            for a in sa["possible_answers"]:
                html += f"<li>{a}</li>"
            html += "</ul>"
        if sa.get("answer_truncated_across_page"):
            html += '<div class="warning">Answer may be truncated</div>'
        html += "</div>"

    html += "</div>"
    return html


def _render_figure_html(fig: dict) -> str:
    alt = fig.get("alt_text") or fig.get("figure_description") or ""
    path = fig.get("image_asset_path","")
    desc = fig.get("figure_description","")
    cap = f"<figcaption>{desc}</figcaption>" if desc else ""
    return f'<figure><img src="{path}" alt="{alt}">{cap}</figure>'
