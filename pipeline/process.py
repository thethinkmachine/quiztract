import json
import re
from pathlib import Path
from typing import Any
from PIL import Image

from loguru import logger
from models.schema import PageDocument

def process_document(pages: list[PageDocument], config: dict[str, Any], vlm_fn: Any, output_dir: Path) -> dict:
    """Process rasterized pages directly with VLM to extract questions and images."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = output_dir / "assets/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    all_questions = []
    
    prompt = """You are analyzing a page from a test paper. Your task is to extract all questions, tables, inline latex math, and metadata.
    Carefully distinguish between different question types: 'MCQ' (Multiple Choice), 'MSQ' (Multiple Select), 'NAT' (Numerical Answer), and 'COMPREHENSION' (A parent passage with no direct options).
    Also extract the correct answer(s) if indicated in the text or metadata.
    Output the data as a JSON object containing a list of questions.
    For any image (graph, network, picture), output its bounding box as [ymin, xmin, ymax, xmax] where coordinates are integers from 0 to 1000 representing normalized relative positions of the page width and height. Do NOT use percentages.
    
    Format:
    {
      "questions": [
        {
          "question_id": "string or null",
          "question_type": "MCQ | MSQ | NAT | COMPREHENSION",
          "question_text": "text including latex math",
          "options": ["A", "B", "C", "D"], // Leave empty if NAT or COMPREHENSION
          "correct_answer": "string, list of strings, or null",
          "metadata": {},
          "tables": [{"headers": [], "rows": []}],
          "images": [{"type": "graph/network/picture", "description": "...", "bbox": [ymin, xmin, ymax, xmax]}]
        }
      ]
    }
    
    Ensure you return valid JSON. Do not include markdown codeblocks around the JSON.
    """
    
    for page in pages:
        logger.info(f"Processing page {page.page_number} with VLM...")
        
        try:
            result_text = vlm_fn(page.raster_image, prompt)
            
            # Clean JSON if it's wrapped in markers
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*$', '', result_text).strip()
            
            try:
                page_data = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON for page {page.page_number}:\n{result_text}")
                continue
                
            for q in page_data.get("questions", []):
                q["page_number"] = page.page_number
                
                # Crop and save extracted images
                for i, img_info in enumerate(q.get("images", [])):
                    bbox = img_info.get("bbox")
                    if bbox and len(bbox) == 4:
                        width, height = page.raster_image.size
                        ymin, xmin, ymax, xmax = bbox
                        
                        # Convert 0-1000 normalized to pixels
                        left = int((xmin / 1000.0) * width)
                        top = int((ymin / 1000.0) * height)
                        right = int((xmax / 1000.0) * width)
                        bottom = int((ymax / 1000.0) * height)
                        
                        try:
                            crop = page.raster_image.crop((left, top, right, bottom))
                            img_filename = f"page_{page.page_number}_q_{q.get('question_id', 'unk')}_img_{i}.png"
                            img_path = figures_dir / img_filename
                            crop.save(img_path)
                            img_info["local_path"] = str(img_path.relative_to(output_dir))
                        except Exception as e:
                            logger.error(f"Failed to crop image on page {page.page_number}: {e}")
                
                all_questions.append(q)
                
        except Exception as e:
            logger.error(f"Error processing page {page.page_number}: {e}")
            
    return {"questions": all_questions}
