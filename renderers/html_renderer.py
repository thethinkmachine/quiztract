import json
from pathlib import Path
from typing import Any
import jinja2

def render_html(json_path: Path, config: dict[str, Any]) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=jinja2.select_autoescape()
    )
    
    template = env.get_template("exam.html.j2")
    render_cfg = config.get("render", {})
    katex_cdn = render_cfg.get("katex_cdn", "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist")
    
    return template.render(
        exam_code=json_path.stem,
        questions=data.get("questions", []),
        katex_cdn=katex_cdn
    )
