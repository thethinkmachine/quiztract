"""Stage 6 — Derived Output Rendering.

Orchestrates rendering of Markdown and HTML from the final JSON.
Renderers read only from the JSON file — never from intermediate representations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def render_outputs(
    json_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Render Markdown and HTML from the final JSON.

    Args:
        json_path: Path to the {exam_code}.json file.
        config: Pipeline configuration dict.
        output_dir: Override output directory.

    Returns:
        Tuple of (md_path, html_path).
    """
    json_path = Path(json_path)

    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    exam_code = json_path.stem
    md_path = output_dir / f"{exam_code}.md"
    html_path = output_dir / f"{exam_code}.html"

    logger.info("Rendering outputs from {}", json_path.name)

    # Render Markdown
    from renderers.md_renderer import render_markdown

    md_content = render_markdown(json_path, config)
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Wrote Markdown: {}", md_path)

    # Render HTML
    from renderers.html_renderer import render_html

    html_content = render_html(json_path, config)
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("Wrote HTML: {}", html_path)

    return md_path, html_path
