"""Page Boundary Stitching Utility.

Handles content that is split across page boundaries:
- Question stems truncated at page breaks
- Comprehension preambles spanning multiple pages
- Answer blocks (Possible Answers) split from their values
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from models.schema import ExtractedDocument, ExtractedQuestionGroup


# Sentinel keys that indicate potentially truncated fields
TRUNCATION_SENTINELS = {
    "possible_answers",
    "answer_range",
}


def stitch_pages(
    doc: ExtractedDocument,
    config: dict[str, Any],
) -> ExtractedDocument:
    """Apply page boundary stitching to an ExtractedDocument.

    Scans all question groups for incomplete fields caused by page breaks
    and attempts to stitch content from subsequent pages.

    Args:
        doc: ExtractedDocument from Stage 3.
        config: Pipeline configuration dict.

    Returns:
        Modified ExtractedDocument with stitched content.
    """
    stitch_cfg = config.get("stitch", {})
    max_lookahead = stitch_cfg.get("max_lookahead_pages", 2)
    sentinel_keys = set(
        k.lower().replace(" ", "_")
        for k in stitch_cfg.get("truncation_sentinel_keys", ["Possible Answers", "Answer Range"])
    )
    sentinel_keys.update(TRUNCATION_SENTINELS)

    logger.info("Running page boundary stitching (max lookahead: {} pages)", max_lookahead)

    stitched_fields: list[str] = []
    groups = doc.question_groups

    for i, group in enumerate(groups):
        # Check for truncated SA answers
        if group.sa_data is not None:
            if not group.sa_data.possible_answers and _has_sentinel(group, "possible_answers"):
                # Look ahead in subsequent groups for continuation
                found = _stitch_sa_answers(group, groups, i, max_lookahead)
                if found:
                    field_id = f"{group.question_id}.sa.possible_answers"
                    stitched_fields.append(field_id)
                    logger.info("Stitched SA possible_answers for {}", group.question_id)
                else:
                    group.sa_data.answer_truncated_across_page = True
                    logger.warning(
                        "Could not stitch SA possible_answers for {} — marking as truncated",
                        group.question_id,
                    )

            # Check for truncated answer range
            if group.sa_data.answer_range_min is None and _has_sentinel(group, "answer_range"):
                found = _stitch_answer_range(group, groups, i, max_lookahead)
                if found:
                    field_id = f"{group.question_id}.sa.answer_range"
                    stitched_fields.append(field_id)
                    logger.info("Stitched SA answer_range for {}", group.question_id)
                else:
                    group.sa_data.answer_truncated_across_page = True

        # Check for truncated question text
        if not group.question_text.strip():
            found = _stitch_question_text(group, groups, i, max_lookahead)
            if found:
                field_id = f"{group.question_id}.question_text"
                stitched_fields.append(field_id)
                logger.info("Stitched question_text for {}", group.question_id)

        # Check for truncated comprehension text
        if group.comprehension_data is not None and not (group.comprehension_data.comprehension_text or "").strip():
            found = _stitch_comprehension_text(group, groups, i, max_lookahead)
            if found:
                field_id = f"{group.question_id}.comprehension.comprehension_text"
                stitched_fields.append(field_id)
                logger.info("Stitched comprehension_text for {}", group.question_id)

    # Update extraction stats
    if stitched_fields:
        doc.extraction_stats["page_boundary_stitching_applied"] = True
        doc.extraction_stats["stitched_fields"] = stitched_fields
    else:
        doc.extraction_stats["page_boundary_stitching_applied"] = False

    logger.info("Stitching complete: {} fields stitched", len(stitched_fields))
    return doc


# ---------------------------------------------------------------------------
# Internal stitching methods
# ---------------------------------------------------------------------------


def _has_sentinel(group: ExtractedQuestionGroup, key: str) -> bool:
    """Check if a question group has a sentinel key in its blocks' metadata."""
    for block in group.blocks:
        if block.metadata_kv:
            for k in block.metadata_kv:
                if key in k.lower().replace(" ", "_"):
                    return True
        if block.text and key.replace("_", " ") in block.text.lower():
            return True
    return False


def _stitch_sa_answers(
    group: ExtractedQuestionGroup,
    all_groups: list[ExtractedQuestionGroup],
    current_idx: int,
    max_lookahead: int,
) -> bool:
    """Try to stitch SA possible_answers from subsequent question groups."""
    # Look at blocks following this group for continuation content
    for look_idx in range(current_idx + 1, min(current_idx + 1 + max_lookahead, len(all_groups))):
        next_group = all_groups[look_idx]
        for block in next_group.blocks:
            if block.text:
                # Look for answer-like content (numbers, comma-separated values)
                answers = _parse_possible_answers(block.text)
                if answers:
                    group.sa_data.possible_answers = answers
                    return True
    return False


def _stitch_answer_range(
    group: ExtractedQuestionGroup,
    all_groups: list[ExtractedQuestionGroup],
    current_idx: int,
    max_lookahead: int,
) -> bool:
    """Try to stitch answer_range from subsequent content."""
    for look_idx in range(current_idx + 1, min(current_idx + 1 + max_lookahead, len(all_groups))):
        next_group = all_groups[look_idx]
        for block in next_group.blocks:
            if block.text:
                match = re.search(r"([\d.e+-]+)\s*(?:to|-)\s*([\d.e+-]+)", block.text, re.IGNORECASE)
                if match:
                    try:
                        group.sa_data.answer_range_min = float(match.group(1))
                        group.sa_data.answer_range_max = float(match.group(2))
                        return True
                    except ValueError:
                        pass
    return False


def _stitch_question_text(
    group: ExtractedQuestionGroup,
    all_groups: list[ExtractedQuestionGroup],
    current_idx: int,
    max_lookahead: int,
) -> bool:
    """Try to stitch question_text from subsequent blocks."""
    # Look for text content in the blocks of subsequent groups
    # that doesn't look like a new question header
    for look_idx in range(current_idx + 1, min(current_idx + 1 + max_lookahead, len(all_groups))):
        next_group = all_groups[look_idx]
        # Don't steal text from a properly formed question
        if next_group.question_text.strip():
            break
        for block in next_group.blocks:
            if block.text and block.block_type == "text":
                group.question_text = block.text
                return True
    return False


def _stitch_comprehension_text(
    group: ExtractedQuestionGroup,
    all_groups: list[ExtractedQuestionGroup],
    current_idx: int,
    max_lookahead: int,
) -> bool:
    """Try to stitch comprehension_text from subsequent pages."""
    parts: list[str] = []
    for look_idx in range(current_idx + 1, min(current_idx + 1 + max_lookahead, len(all_groups))):
        next_group = all_groups[look_idx]
        if next_group.question_type and next_group.question_type != "COMPREHENSION":
            # Reached a non-comprehension question, stop
            break
        for block in next_group.blocks:
            if block.text and block.block_type == "text":
                parts.append(block.text)

    if parts:
        existing = group.comprehension_data.comprehension_text or ""
        group.comprehension_data.comprehension_text = (existing + "\n" + "\n".join(parts)).strip()
        return True
    return False


def _parse_possible_answers(text: str) -> list[str]:
    """Parse possible answer values from text."""
    # Try comma-separated values
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        # Check if they look like answer values
        if all(_looks_like_answer(p) for p in parts):
            return parts

    # Try semicolon-separated
    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if all(_looks_like_answer(p) for p in parts):
            return parts

    # Single value
    text = text.strip()
    if _looks_like_answer(text):
        return [text]

    return []


def _looks_like_answer(text: str) -> bool:
    """Check if text looks like an answer value (numeric or short alphanumeric)."""
    text = text.strip()
    if not text:
        return False
    # Numeric (possibly with decimal, sign, or scientific notation)
    if re.match(r"^[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?$", text):
        return True
    # Short alphanumeric (up to ~50 chars)
    if len(text) <= 50:
        return True
    return False
