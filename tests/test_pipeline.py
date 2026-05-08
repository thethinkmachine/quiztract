import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.schema import PageDocument
from pipeline.process import process_document
from pipeline.render import render_outputs

def test_process_document_success(tmp_path):
    # Setup mock pages
    from PIL import Image
    real_image = Image.new("RGB", (1000, 1000))
    page1 = PageDocument(
        page_number=1,
        raster_image=real_image,
        text_blocks=[],
        page_width=1000.0,
        page_height=1000.0,
        width_px=1000,
        height_px=1000,
    )
    
    # Mock VLM function that returns standard JSON
    def mock_vlm_fn(image, prompt):
        return json.dumps({
            "questions": [
                {
                    "question_id": "test_1",
                    "question_type": "MCQ",
                    "question_text": "What is 2+2?",
                    "options": ["3", "4", "5", "6"],
                    "correct_answer": "4",
                    "metadata": {"Correct Marks": "1"}
                }
            ]
        })

    config = {}
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = process_document([page1], config, mock_vlm_fn, output_dir)
    
    assert "questions" in result
    assert len(result["questions"]) == 1
    assert result["questions"][0]["question_id"] == "test_1"
    assert result["questions"][0]["page_number"] == 1

def test_render_outputs(tmp_path):
    # Setup mock JSON data
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    exam_code = "TEST01"
    json_path = output_dir / f"{exam_code}.json"
    
    with open(json_path, "w") as f:
        json.dump({
            "questions": [
                {
                    "question_id": "test_1",
                    "question_type": "MCQ",
                    "question_text": "Sample text",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A"
                }
            ]
        }, f)
        
    config = {
        "render": {
            "katex_cdn": "https://cdn.jsdelivr.net/npm/katex"
        }
    }
    
    md_path, html_path = render_outputs(json_path, config, output_dir)
    
    assert md_path.exists()
    assert html_path.exists()
    
    with open(html_path, "r") as f:
        html_content = f.read()
        assert "Sample text" in html_content
        assert "TEST01" in html_content
