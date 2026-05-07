"""Stage 2 — Block Classification.

Processes each page's docling blocks and raster to:
- Assign a block type to every block
- Detect question boundaries
- Detect comprehension boundaries
- Detect metadata field lines
- Flag blocks whose classification is uncertain
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from models.enums import BlockType, ClassificationSource
from models.schema import (
    ClassifiedBlock,
    ClassifiedDocument,
    PageDocument,
    QuestionGroup,
)


# ---------------------------------------------------------------------------
# Question header regex patterns
# ---------------------------------------------------------------------------

# Matches lines like: "Question Id : 12345  Question Type : MCQ  ..."
QUESTION_HEADER_PATTERN = re.compile(
    r"Question\s+Id\s*:\s*(\S+)",
    re.IGNORECASE,
)

# Extract question type from header
QUESTION_TYPE_PATTERN = re.compile(
    r"Question\s+Type\s*:\s*(\w+)",
    re.IGNORECASE,
)

# Extract question number
QUESTION_NUMBER_PATTERN = re.compile(
    r"Question\s+Number\s*:\s*(\d+)",
    re.IGNORECASE,
)

# Extract section info
SECTION_PATTERN = re.compile(
    r"Section\s*:\s*(.+?)(?:\s{2,}|$)",
    re.IGNORECASE,
)

# Metadata field pattern: "Key : Value" with common CBT keys
METADATA_KV_PATTERN = re.compile(
    r"([\w\s]+?)\s*:\s*(.+?)(?:\s{2,}|$)"
)

# Section header patterns
SECTION_HEADER_PATTERN = re.compile(
    r"^(?:Section\s+\d+|Part\s+[A-Z]|SECTION\s+[A-Z])",
    re.IGNORECASE,
)

# Page artifact patterns (headers, footers, page numbers)
PAGE_ARTIFACT_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*$"),  # Standalone page number
]

# Option patterns (A), B), 1., 2., etc.)
OPTION_PATTERN = re.compile(
    r"^\s*(?:\(?[A-Da-d1-4]\)?[\.\):]?\s)",
)

# Known metadata field keys for classification
KNOWN_METADATA_KEYS = {
    "question id", "question number", "question type", "question label",
    "correct marks", "negative marks", "response type", "answers type",
    "possible answers", "answer range", "evaluation required for sa",
    "sub question shuffling allowed", "group comprehension questions",
    "question numbers", "question pattern type", "section",
    "show word count", "text area type", "display number panel",
    "group all questions", "enable mark review", "question shuffling allowed",
}

# Answer-related patterns
ANSWER_BLOCK_PATTERN = re.compile(
    r"(?:Possible\s+Answers|Answer\s+Range|Correct\s+Answer)\s*:",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_document(
    pages: list[PageDocument],
    exam_code: str,
    config: dict[str, Any],
    vlm_fn: Any | None = None,
) -> ClassifiedDocument:
    """Classify all blocks and detect question boundaries.

    Args:
        pages: List of PageDocument from Stage 1.
        exam_code: Exam identifier string.
        config: Pipeline configuration dict.
        vlm_fn: Optional VLM inference function for fallback classification.

    Returns:
        ClassifiedDocument with all blocks classified and question groups detected.
    """
    classify_cfg = config.get("classify", {})
    confidence_threshold = classify_cfg.get("confidence_threshold", 0.75)
    metadata_keys = {k.lower() for k in classify_cfg.get("metadata_field_patterns", [])}
    metadata_keys.update(KNOWN_METADATA_KEYS)

    logger.info("Classifying blocks for exam: {}", exam_code)

    all_blocks: list[ClassifiedBlock] = []
    question_groups: list[QuestionGroup] = []
    current_group: Optional[QuestionGroup] = None

    for page in pages:
        if page.parse_failed:
            logger.warning("Skipping failed page {}", page.page_number)
            continue

        page_blocks = _classify_page(
            page=page,
            metadata_keys=metadata_keys,
            vlm_fn=vlm_fn,
            confidence_threshold=confidence_threshold,
        )

        for block in page_blocks:
            block_idx = len(all_blocks)
            all_blocks.append(block)

            # Detect question boundaries
            if block.block_type == BlockType.QUESTION_HEADER:
                # Close previous group
                if current_group is not None:
                    current_group.end_page = block.page_number
                    question_groups.append(current_group)

                # Parse header metadata
                header_meta = _parse_question_header(block.raw_text or "")

                # Start new group
                q_type = header_meta.get("question_type", "").upper()
                current_group = QuestionGroup(
                    question_id=header_meta.get("question_id"),
                    question_type=q_type if q_type else None,
                    question_number=_safe_int(header_meta.get("question_number")),
                    section_id=header_meta.get("section"),
                    is_comprehension=(q_type == "COMPREHENSION"),
                    block_indices=[block_idx],
                    metadata_fields=header_meta,
                    start_page=block.page_number,
                    end_page=block.page_number,
                )
            elif current_group is not None:
                current_group.block_indices.append(block_idx)
                current_group.end_page = block.page_number

                # Accumulate metadata from metadata_field blocks
                if block.block_type == BlockType.METADATA_FIELD and block.raw_text:
                    kv = _parse_metadata_line(block.raw_text)
                    current_group.metadata_fields.update(kv)

    # Close final group
    if current_group is not None:
        question_groups.append(current_group)

    # Post-process: detect comprehension sub-questions
    question_groups = _detect_comprehension_children(question_groups)

    logger.info(
        "Classification complete: {} blocks, {} question groups",
        len(all_blocks),
        len(question_groups),
    )

    return ClassifiedDocument(
        exam_code=exam_code,
        pages=pages,
        blocks=all_blocks,
        question_groups=question_groups,
    )


# ---------------------------------------------------------------------------
# Page-level classification
# ---------------------------------------------------------------------------


def _classify_page(
    page: PageDocument,
    metadata_keys: set[str],
    vlm_fn: Any | None,
    confidence_threshold: float,
) -> list[ClassifiedBlock]:
    """Classify all blocks on a single page."""
    blocks: list[ClassifiedBlock] = []

    # If we have docling blocks, classify each one
    if page.docling_blocks:
        for db in page.docling_blocks:
            block = _classify_docling_block(db, page, metadata_keys)
            blocks.append(block)
    else:
        # Fall back to text-line-based classification
        blocks = _classify_from_text(page, metadata_keys)

    # VLM fallback for unknown blocks
    if vlm_fn is not None:
        for i, block in enumerate(blocks):
            if block.block_type == BlockType.UNKNOWN:
                vlm_block = _vlm_classify_block(block, page, vlm_fn, confidence_threshold)
                if vlm_block is not None:
                    blocks[i] = vlm_block

    return blocks


def _classify_docling_block(
    db: Any,
    page: PageDocument,
    metadata_keys: set[str],
) -> ClassifiedBlock:
    """Classify a single docling block using rule-based heuristics."""
    text = getattr(db, "text", "") or ""
    label = getattr(db, "label", "") or ""
    label_lower = label.lower()

    # Determine bounding box from provenance
    bbox = _extract_bbox(db, page)

    # Rule-based classification
    block_type = BlockType.TEXT
    confidence = "high"
    source = ClassificationSource.RULE

    # Check for question header
    if QUESTION_HEADER_PATTERN.search(text):
        block_type = BlockType.QUESTION_HEADER

    # Check for page artifacts
    elif _is_page_artifact(text):
        block_type = BlockType.PAGE_ARTIFACT

    # Check for section header
    elif SECTION_HEADER_PATTERN.match(text.strip()):
        block_type = BlockType.SECTION_HEADER

    # Check for metadata fields
    elif _is_metadata_line(text, metadata_keys):
        block_type = BlockType.METADATA_FIELD

    # Check for answer block
    elif ANSWER_BLOCK_PATTERN.search(text):
        block_type = BlockType.ANSWER_BLOCK

    # Check for option block
    elif OPTION_PATTERN.match(text):
        block_type = BlockType.OPTION_BLOCK

    # Check docling label for hints
    elif "table" in label_lower:
        block_type = BlockType.TABLE

    elif "figure" in label_lower or "image" in label_lower or "picture" in label_lower:
        block_type = BlockType.IMAGE

    elif "formula" in label_lower or "equation" in label_lower:
        # Check for matrix
        if _text_contains_matrix(text):
            block_type = BlockType.MATRIX
        else:
            block_type = BlockType.MATH_BLOCK

    # Check text for inline math markers
    elif _has_inline_math(text):
        block_type = BlockType.MATH_INLINE

    # If nothing matched and text is very short or empty, mark unknown
    elif not text.strip():
        block_type = BlockType.UNKNOWN
        confidence = "low"
        source = ClassificationSource.UNKNOWN

    # Crop the block from the page raster
    crop = None
    if bbox and page.raster_image:
        try:
            crop = page.raster_image.crop(bbox)
        except Exception:
            pass

    return ClassifiedBlock(
        block_type=block_type.value if isinstance(block_type, BlockType) else block_type,
        raw_text=text if text else None,
        bounding_box=bbox,
        page_number=page.page_number,
        classification_source=source.value if isinstance(source, ClassificationSource) else source,
        classification_confidence=confidence,
        raster_crop=crop,
    )


def _classify_from_text(
    page: PageDocument,
    metadata_keys: set[str],
) -> list[ClassifiedBlock]:
    """Classify blocks from the raw text layer when docling blocks are unavailable."""
    blocks: list[ClassifiedBlock] = []
    lines = page.raw_text.split("\n")

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        block_type = BlockType.TEXT
        confidence = "medium"

        if QUESTION_HEADER_PATTERN.search(line_stripped):
            block_type = BlockType.QUESTION_HEADER
            confidence = "high"
        elif _is_page_artifact(line_stripped):
            block_type = BlockType.PAGE_ARTIFACT
            confidence = "high"
        elif SECTION_HEADER_PATTERN.match(line_stripped):
            block_type = BlockType.SECTION_HEADER
            confidence = "high"
        elif _is_metadata_line(line_stripped, metadata_keys):
            block_type = BlockType.METADATA_FIELD
            confidence = "high"
        elif ANSWER_BLOCK_PATTERN.search(line_stripped):
            block_type = BlockType.ANSWER_BLOCK
            confidence = "high"
        elif OPTION_PATTERN.match(line_stripped):
            block_type = BlockType.OPTION_BLOCK
            confidence = "high"

        blocks.append(
            ClassifiedBlock(
                block_type=block_type.value,
                raw_text=line_stripped,
                bounding_box=None,
                page_number=page.page_number,
                classification_source=ClassificationSource.RULE.value,
                classification_confidence=confidence,
            )
        )

    return blocks


# ---------------------------------------------------------------------------
# VLM fallback classification
# ---------------------------------------------------------------------------


def _vlm_classify_block(
    block: ClassifiedBlock,
    page: PageDocument,
    vlm_fn: Any,
    confidence_threshold: float,
) -> Optional[ClassifiedBlock]:
    """Classify a block using VLM when rule-based classification failed."""
    if block.raster_crop is None and block.bounding_box is not None:
        try:
            block.raster_crop = page.raster_image.crop(block.bounding_box)
        except Exception:
            return None

    if block.raster_crop is None:
        return None

    try:
        # Load prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "classify_block.txt"
        prompt = prompt_path.read_text(encoding="utf-8")

        # Call VLM
        response = vlm_fn(image=block.raster_crop, prompt=prompt)

        # Parse response
        result = json.loads(response)
        predicted_type = result.get("block_type", "unknown")
        confidence = result.get("confidence", "low")

        # Check confidence threshold
        conf_map = {"high": 1.0, "medium": 0.5, "low": 0.25}
        if conf_map.get(confidence, 0) < confidence_threshold:
            predicted_type = "unknown"

        return ClassifiedBlock(
            block_type=predicted_type,
            raw_text=block.raw_text,
            bounding_box=block.bounding_box,
            page_number=block.page_number,
            classification_source=ClassificationSource.VLM.value,
            classification_confidence=confidence,
            raster_crop=block.raster_crop,
        )

    except Exception as exc:
        logger.warning("VLM classification failed: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_question_header(text: str) -> dict[str, str]:
    """Parse a question header line into key-value pairs."""
    result: dict[str, str] = {}

    qid = QUESTION_HEADER_PATTERN.search(text)
    if qid:
        result["question_id"] = qid.group(1)

    qtype = QUESTION_TYPE_PATTERN.search(text)
    if qtype:
        result["question_type"] = qtype.group(1)

    qnum = QUESTION_NUMBER_PATTERN.search(text)
    if qnum:
        result["question_number"] = qnum.group(1)

    section = SECTION_PATTERN.search(text)
    if section:
        result["section"] = section.group(1).strip()

    # Extract marks
    marks_pattern = re.compile(r"Correct\s+Marks\s*:\s*([\d.]+)", re.IGNORECASE)
    neg_pattern = re.compile(r"Negative\s+Marks\s*:\s*([\d.]+)", re.IGNORECASE)

    marks = marks_pattern.search(text)
    if marks:
        result["correct_marks"] = marks.group(1)

    neg = neg_pattern.search(text)
    if neg:
        result["negative_marks"] = neg.group(1)

    # Also parse all key-value pairs generically
    kv = _parse_metadata_line(text)
    for k, v in kv.items():
        if k not in result:
            result[k] = v

    return result


def _parse_metadata_line(text: str) -> dict[str, str]:
    """Parse a metadata line of form 'Key : Value  Key : Value' into dict."""
    result: dict[str, str] = {}

    # Split on double-space separated key:value pairs
    # Pattern: "Key : Value" possibly followed by more pairs
    pairs = re.findall(r"([\w\s]+?)\s*:\s*([^:]+?)(?:\s{2,}|$)", text)
    for key, value in pairs:
        clean_key = key.strip().lower().replace(" ", "_")
        clean_value = value.strip()
        if clean_key and clean_value:
            result[clean_key] = clean_value

    return result


def _is_metadata_line(text: str, metadata_keys: set[str]) -> bool:
    """Check if a line is a platform metadata field."""
    text_lower = text.lower().strip()

    for key in metadata_keys:
        if key in text_lower and ":" in text:
            return True

    return False


def _is_page_artifact(text: str) -> bool:
    """Check if text is a page artifact (header, footer, page number)."""
    for pattern in PAGE_ARTIFACT_PATTERNS:
        if pattern.match(text.strip()):
            return True
    return False


def _has_inline_math(text: str) -> bool:
    """Check if text likely contains inline math expressions."""
    math_indicators = [
        r"\frac", r"\sqrt", r"\sum", r"\int", r"\alpha", r"\beta",
        r"\gamma", r"\delta", r"\theta", r"\pi", r"\infty",
        r"\in", r"\subset", r"\cup", r"\cap", r"\times",
        "^{", "_{", r"\mathbb", r"\mathcal", r"\vec",
    ]
    return any(ind in text for ind in math_indicators)


def _text_contains_matrix(text: str) -> bool:
    """Check if text contains matrix-related LaTeX constructs."""
    matrix_indicators = [
        r"\begin{bmatrix}", r"\begin{pmatrix}", r"\begin{vmatrix}",
        r"\begin{matrix}", r"\begin{Bmatrix}", r"\begin{Vmatrix}",
        r"\begin{array}",
    ]
    return any(ind in text for ind in matrix_indicators)


def _extract_bbox(
    db: Any, page: PageDocument
) -> Optional[tuple[int, int, int, int]]:
    """Extract bounding box from a docling block, converting to pixel coordinates."""
    try:
        prov = getattr(db, "prov", None)
        if not prov or len(prov) == 0:
            return None

        bbox_obj = getattr(prov[0], "bbox", None)
        if bbox_obj is None:
            return None

        # Docling bbox is in document coordinates — convert to pixel coordinates
        # Docling uses (l, t, r, b) format typically
        l = getattr(bbox_obj, "l", None) or getattr(bbox_obj, "x0", 0)
        t = getattr(bbox_obj, "t", None) or getattr(bbox_obj, "y0", 0)
        r = getattr(bbox_obj, "r", None) or getattr(bbox_obj, "x1", 0)
        b = getattr(bbox_obj, "b", None) or getattr(bbox_obj, "y1", 0)

        # Get page dimensions from docling provenance
        page_w = getattr(prov[0], "page_width", None) or getattr(bbox_obj, "coord_origin_width", None)
        page_h = getattr(prov[0], "page_height", None) or getattr(bbox_obj, "coord_origin_height", None)

        if page_w and page_h and page_w > 0 and page_h > 0:
            # Scale to pixel coordinates
            scale_x = page.width_px / page_w
            scale_y = page.height_px / page_h
            return (
                int(l * scale_x),
                int(t * scale_y),
                int(r * scale_x),
                int(b * scale_y),
            )
        else:
            # Assume already in reasonable coordinates
            return (int(l), int(t), int(r), int(b))

    except Exception as exc:
        logger.debug("Failed to extract bbox: {}", exc)
        return None


def _detect_comprehension_children(
    groups: list[QuestionGroup],
) -> list[QuestionGroup]:
    """Post-process question groups to link comprehension sub-questions."""
    comp_groups: dict[str, QuestionGroup] = {}

    # First pass: identify comprehension parents
    for group in groups:
        if group.is_comprehension and group.question_id:
            comp_groups[group.question_id] = group

            # Parse question numbers range if available
            qnums = group.metadata_fields.get("question_numbers", "")
            if qnums:
                # E.g. "1 - 5" or "1,2,3,4,5"
                group.metadata_fields["_sub_question_numbers"] = qnums

    # Second pass: link sub-questions to their comprehension parent
    for group in groups:
        if group.is_comprehension:
            continue

        # Check if this question follows a comprehension block
        # by looking at question number ranges
        for comp_id, comp_group in comp_groups.items():
            qnums_raw = comp_group.metadata_fields.get("_sub_question_numbers", "")
            if not qnums_raw:
                continue

            # Parse range
            try:
                if "-" in qnums_raw:
                    parts = qnums_raw.split("-")
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    sub_range = range(start, end + 1)
                elif "," in qnums_raw:
                    sub_range = [int(x.strip()) for x in qnums_raw.split(",")]
                else:
                    continue

                if group.question_number and group.question_number in sub_range:
                    group.parent_comprehension_id = comp_id
                    break

            except (ValueError, IndexError):
                continue

    return groups


def _safe_int(value: Any) -> Optional[int]:
    """Safely convert a value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
