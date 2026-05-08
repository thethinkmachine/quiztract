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
    seen_questions = set()
    
    prompt = """You are analyzing a page from a test paper. Your task is to extract all questions, tables, inline latex math, and metadata.
    Carefully distinguish between different question types: 'MCQ' (Multiple Choice), 'MSQ' (Multiple Select), 'NAT' (Numerical Answer), and 'COMPREHENSION' (A parent passage with no direct options).
    Also extract the correct answer(s) if indicated in the text or metadata.
    Output the data as a JSON object containing a list of questions.
    For any image (graph, network, picture), output its bounding box as [ymin, xmin, ymax, xmax] where coordinates are integers from 0 to 1000 representing normalized relative positions of the page width and height. Do NOT use percentages.
    
    CRITICAL RULES:
    1. Do NOT solve the questions.
    2. Do NOT hallucinate options. Only extract options that are explicitly printed on the page.
    3. If there are no options printed, leave the options list empty. Never generate your own combinations or lists.
    4. For 'COMPREHENSION', 'NAT', and 'SA' question types, the 'options' list MUST be strictly empty [] under all circumstances.
    5. Do NOT output the same question multiple times. Eliminate duplicates.
    6. Ensure the 'question_text' actually contains the text of the question, not just metadata headers.
    7. Strip any 'Option ID' numeric prefixes from the extracted options.
    
    EXAMPLE OUTPUT:
    {
      "questions": [
        {
          "question_type": "MCQ",
          "question_text": "What is the capital of France?",
          "options": ["Paris", "London", "Berlin", "Madrid"],
          "correct_answer": "Paris",
          "metadata": {"Correct Marks": "1", "Negative Marks": "0.33"},
          "tables": [],
          "images": []
        },
        {
          "question_type": "MSQ",
          "question_text": "Which of the following are prime numbers?",
          "options": ["2", "4", "5", "9"],
          "correct_answer": ["2", "5"],
          "metadata": {"Correct Marks": "2", "Negative Marks": "0"},
          "tables": [],
          "images": []
        },
        {
          "question_type": "NAT",
          "question_text": "Calculate the area of a circle with radius 3. Use pi = 3.14.",
          "options": [],
          "correct_answer": "28.26",
          "metadata": {"Correct Marks": "1", "Negative Marks": "0"},
          "tables": [],
          "images": []
        },
        {
          "question_type": "COMPREHENSION",
          "question_text": "Read the following passage carefully: The nodes A, B, and C form a network...",
          "options": [],
          "correct_answer": null,
          "metadata": {"Question Pattern Type": "NonMatrix"},
          "tables": [],
          "images": [{"type": "network", "description": "Graph of nodes A, B, C", "bbox": [100, 200, 300, 400]}]
        }
      ]
    }
    
    Format:
    {
      "questions": [
        {
          "question_type": "MCQ | MSQ | NAT | COMPREHENSION | SA",
          "question_text": "text including latex math",
          "options": ["A", "B", "C", "D"], // Leave empty if NAT, SA or COMPREHENSION
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
                
            for q_index, q in enumerate(page_data.get("questions", [])):
                q["page_number"] = page.page_number
                
                # 1. Clean Question Text and remove metadata prefixes
                raw_text = q.get("question_text", "")
                if not raw_text:
                    continue
                # Remove prefixes like "Question Number : 2 Question Id : 6406531862692 Question Type : MSQ"
                clean_text = re.sub(r'(?i)^(Question Number\s*:\s*\d+\s*)?(Question Id\s*:\s*\d+\s*)?(Question Type\s*:\s*[A-Za-z]+\s*)?', '', raw_text).strip()
                # Remove standalone digit prefixes like "6406536034580. "
                clean_text = re.sub(r'^\d+\.\s*(?:\u2026\s*)?', '', clean_text).strip()
                q["question_text"] = clean_text
                
                # 2. Deduplicate
                if clean_text in seen_questions:
                    continue
                seen_questions.add(clean_text)

                # 3. Clean Options
                q_type = q.get("question_type", "MCQ").upper()
                if q_type in ["COMPREHENSION", "NAT", "SA"]:
                    q["options"] = []
                else:
                    opts = q.get("options") or []
                    # If more than 8 options, it is almost certainly a hallucination
                    if len(opts) > 8:
                        q["options"] = []
                    else:
                        # Clean ID prefixes from options (e.g. "6406536034572. \u2192 Printed...")
                        cleaned_opts = []
                        for opt in opts:
                            cleaned_opts.append(re.sub(r'^\d+\.\s*(?:\u2192\s*|\u2026\s*)?', '', str(opt)).strip())
                        q["options"] = cleaned_opts
                
                # Crop and save extracted images
                for i, img_info in enumerate(q.get("images", [])):
                    bbox = img_info.get("bbox")
                    if bbox and len(bbox) == 4:
                        width, height = page.raster_image.size
                        # Detect if VLM is using 0-1 ratios instead of 0-1000 integers
                        # If all values are <= 1.0 and at least one is > 0, it's likely a 0-1 ratio.
                        is_ratio = all(0 <= v <= 1.0 for v in bbox)
                        scale = 1.0 if is_ratio else 1000.0
                        
                        ymin, xmin, ymax, xmax = bbox
                        
                        # Convert to pixels
                        left = int((xmin / scale) * width)
                        top = int((ymin / scale) * height)
                        right = int((xmax / scale) * width)
                        bottom = int((ymax / scale) * height)
                        
                        try:
                            crop = page.raster_image.crop((left, top, right, bottom))
                            img_filename = f"page_{page.page_number}_q_{q_index}_img_{i}.png"
                            img_path = figures_dir / img_filename
                            crop.save(img_path)
                            img_info["local_path"] = str(img_path.relative_to(output_dir))
                        except Exception as e:
                            logger.error(f"Failed to crop image on page {page.page_number}: {e}")
                
                all_questions.append(q)
                
        except Exception as e:
            logger.error(f"Error processing page {page.page_number}: {e}")
            
    return {"questions": all_questions}
