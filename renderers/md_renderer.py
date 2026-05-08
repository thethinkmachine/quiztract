import json
from pathlib import Path
from typing import Any

def render_markdown(json_path: Path, config: dict[str, Any]) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    lines.append(f"# Exam Questions ({json_path.stem})\n")

    questions = data.get("questions", [])
    if not questions:
        lines.append("No questions found.\n")
        return "\n".join(lines)

    for i, q in enumerate(questions, 1):
        q_id = q.get("question_id", f"UNK-{i}")
        q_type = q.get("question_type", "MCQ")
        lines.append(f"## Question {i} (ID: {q_id} | Type: {q_type})\n")
        
        ca = q.get("correct_answer")
        if ca:
            lines.append(f"**Correct Answer:** {ca}\n")
            
        # Meta
        meta = q.get("metadata") or {}
        if meta:
            lines.append("**Metadata:**")
            for k, v in meta.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")

        # Text
        text = q.get("question_text", "")
        if text:
            lines.append(text)
            lines.append("")

        # Images
        for img in q.get("images") or []:
            path = img.get("local_path")
            desc = img.get("description", "Image")
            if path:
                lines.append(f"![{desc}]({path})")
                lines.append("")

        # Tables
        for table in q.get("tables") or []:
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if headers:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in rows:
                    lines.append("| " + " | ".join(map(str, row)) + " |")
            lines.append("")

        # Options
        options = q.get("options") or []
        for j, opt in enumerate(options):
            label = chr(65 + j)  # A, B, C, D
            lines.append(f"- **{label}.** {opt}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
