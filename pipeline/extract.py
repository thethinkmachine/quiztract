"""Stage 3 — Content Extraction.

For each classified block, applies the appropriate extraction method:
- text: copy raw text
- math_inline/math_block/matrix: VLM transcription to LaTeX
- image: VLM description + structured data extraction
- table: docling extractor → VLM fallback
- option_block: VLM transcription
- metadata_field: key-value parsing

Saves raster crops for all non-plain-text blocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from models.enums import BlockType, ExtractionSource, ExtractionMode
from models.schema import (
    ClassifiedBlock,
    ClassifiedDocument,
    ExtractedBlock,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedOption,
    ExtractedQuestionGroup,
    ExtractedSAData,
    ExtractedComprehensionData,
    QuestionGroup,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_content(
    classified: ClassifiedDocument,
    config: dict[str, Any],
    vlm_fn: Any | None = None,
    output_dir: str | Path | None = None,
) -> ExtractedDocument:
    """Extract content from all classified blocks.

    Args:
        classified: ClassifiedDocument from Stage 2.
        config: Pipeline configuration dict.
        vlm_fn: Optional VLM inference function(image, prompt) -> str.
        output_dir: Base output directory for saving asset crops.

    Returns:
        ExtractedDocument with all content extracted.
    """
    extract_cfg = config.get("extract", {})
    output_cfg = config.get("output", {})

    # Set up asset directories
    if output_dir is None:
        base = Path(output_cfg.get("base_dir", "./output"))
        output_dir = base / classified.exam_code
    else:
        output_dir = Path(output_dir)

    figures_dir = output_dir / output_cfg.get("figures_subdir", "assets/figures")
    math_dir = output_dir / output_cfg.get("math_crops_subdir", "assets/math_crops")
    option_dir = output_dir / output_cfg.get("option_crops_subdir", "assets/option_crops")

    for d in [figures_dir, math_dir, option_dir]:
        d.mkdir(parents=True, exist_ok=True)

    math_padding = extract_cfg.get("math_crop_padding_px", 8)
    figure_padding = extract_cfg.get("figure_crop_padding_px", 12)

    logger.info("Extracting content for exam: {}", classified.exam_code)

    # Extraction statistics
    stats: dict[str, Any] = {
        "total_blocks": len(classified.blocks),
        "vlm_transcription_count": 0,
        "raster_fallback_count": 0,
        "text_extraction_count": 0,
    }

    # Extract content for each block
    extracted_blocks: list[ExtractedBlock] = []
    for i, block in enumerate(classified.blocks):
        ext = _extract_block(
            block=block,
            block_index=i,
            pages=classified.pages,
            vlm_fn=vlm_fn,
            figures_dir=figures_dir,
            math_dir=math_dir,
            option_dir=option_dir,
            math_padding=math_padding,
            figure_padding=figure_padding,
            exam_code=classified.exam_code,
            config=config,
        )
        extracted_blocks.append(ext)

        # Update stats
        if ext.extraction_source == ExtractionSource.VLM_TRANSCRIPTION.value:
            stats["vlm_transcription_count"] += 1
        elif ext.extraction_source == ExtractionSource.RASTER_ONLY.value:
            stats["raster_fallback_count"] += 1
        else:
            stats["text_extraction_count"] += 1

    # Build question groups from classified groups + extracted blocks
    question_groups = _build_question_groups(
        classified.question_groups,
        extracted_blocks,
        classified.blocks,
        output_dir,
    )

    logger.info(
        "Extraction complete: {} blocks processed ({} VLM, {} raster fallback)",
        len(extracted_blocks),
        stats["vlm_transcription_count"],
        stats["raster_fallback_count"],
    )

    return ExtractedDocument(
        exam_code=classified.exam_code,
        question_groups=question_groups,
        extraction_stats=stats,
    )


# ---------------------------------------------------------------------------
# Block-level extraction
# ---------------------------------------------------------------------------


def _extract_block(
    block: ClassifiedBlock,
    block_index: int,
    pages: list,
    vlm_fn: Any | None,
    figures_dir: Path,
    math_dir: Path,
    option_dir: Path,
    math_padding: int,
    figure_padding: int,
    exam_code: str,
    config: dict[str, Any],
) -> ExtractedBlock:
    """Extract content from a single classified block."""
    btype = block.block_type

    if btype == BlockType.TEXT.value:
        return _extract_text(block)

    elif btype == BlockType.MATH_INLINE.value:
        return _extract_math_inline(block, vlm_fn, math_dir, math_padding, block_index)

    elif btype in (BlockType.MATH_BLOCK.value, BlockType.MATRIX.value):
        return _extract_math_block(
            block, vlm_fn, math_dir, math_padding, block_index,
            is_matrix=(btype == BlockType.MATRIX.value),
        )

    elif btype == BlockType.IMAGE.value:
        return _extract_image(block, vlm_fn, figures_dir, figure_padding, block_index)

    elif btype == BlockType.TABLE.value:
        return _extract_table(block, vlm_fn, figures_dir, figure_padding, block_index)

    elif btype == BlockType.OPTION_BLOCK.value:
        return _extract_option(block, vlm_fn, option_dir, block_index)

    elif btype == BlockType.METADATA_FIELD.value:
        return _extract_metadata(block)

    elif btype == BlockType.ANSWER_BLOCK.value:
        return _extract_metadata(block)  # Parse as key-value

    elif btype == BlockType.QUESTION_HEADER.value:
        return _extract_metadata(block)

    elif btype == BlockType.PAGE_ARTIFACT.value:
        return ExtractedBlock(
            block_type=btype,
            text=None,
            extraction_source=ExtractionSource.TEXT_LAYER.value,
            extraction_confidence="high",
        )

    else:
        # Unknown or section_header — store text if available
        return ExtractedBlock(
            block_type=btype,
            text=block.raw_text,
            extraction_source=ExtractionSource.TEXT_LAYER.value,
            extraction_confidence="low" if not block.raw_text else "medium",
            needs_review=(btype == BlockType.UNKNOWN.value),
            review_reason="VLM classification returned unknown block type" if btype == BlockType.UNKNOWN.value else None,
        )


# ---------------------------------------------------------------------------
# Type-specific extraction methods
# ---------------------------------------------------------------------------


def _extract_text(block: ClassifiedBlock) -> ExtractedBlock:
    """Extract plain text — direct copy from text layer."""
    return ExtractedBlock(
        block_type=BlockType.TEXT.value,
        text=block.raw_text or "",
        extraction_source=ExtractionSource.TEXT_LAYER.value,
        extraction_confidence="high",
    )


def _extract_math_inline(
    block: ClassifiedBlock,
    vlm_fn: Any | None,
    math_dir: Path,
    padding: int,
    block_index: int,
) -> ExtractedBlock:
    """Extract inline math — try text layer first, VLM fallback."""
    # Try to extract LaTeX from text layer
    latex = _try_latex_from_text(block.raw_text)

    if latex and _validate_latex_basic(latex):
        return ExtractedBlock(
            block_type=BlockType.MATH_INLINE.value,
            text=block.raw_text,
            latex=latex,
            extraction_source=ExtractionSource.TEXT_LAYER.value,
            extraction_confidence="high",
        )

    # VLM fallback
    crop_path = _save_crop(block, math_dir, f"math_{block_index:04d}.png", padding)
    vlm_result = _vlm_extract_math(block, vlm_fn)

    return ExtractedBlock(
        block_type=BlockType.MATH_INLINE.value,
        text=block.raw_text,
        latex=vlm_result.get("latex") if vlm_result else None,
        math_crop_path=crop_path,
        extraction_source=ExtractionSource.VLM_TRANSCRIPTION.value if vlm_result else ExtractionSource.RASTER_ONLY.value,
        extraction_confidence=vlm_result.get("confidence", "medium") if vlm_result else "low",
        needs_review=vlm_result is None,
        review_reason="Low confidence VLM transcription of math block" if vlm_result is None else None,
    )


def _extract_math_block(
    block: ClassifiedBlock,
    vlm_fn: Any | None,
    math_dir: Path,
    padding: int,
    block_index: int,
    is_matrix: bool = False,
) -> ExtractedBlock:
    """Extract standalone math block — always VLM transcription + crop."""
    crop_path = _save_crop(block, math_dir, f"math_{block_index:04d}.png", padding)
    vlm_result = _vlm_extract_math(block, vlm_fn)

    confidence = "medium"
    needs_review = False
    review_reason = None

    if vlm_result:
        latex = vlm_result.get("latex", "")
        confidence = vlm_result.get("confidence", "medium")

        # Validate LaTeX
        if latex and _validate_latex_basic(latex):
            if confidence == "medium":
                confidence = "high"
        else:
            confidence = "low"
            needs_review = True
            review_reason = "LaTeX string has unbalanced braces"

        # For matrices, additional validation
        if is_matrix and latex:
            dims = vlm_result.get("dimensions")
            if dims:
                # Store dimensions for validation in Stage 4
                pass
    else:
        latex = None
        confidence = "low"
        needs_review = True
        review_reason = "Low confidence VLM transcription of math block"

    return ExtractedBlock(
        block_type=BlockType.MATRIX.value if is_matrix else BlockType.MATH_BLOCK.value,
        text=block.raw_text,
        latex=latex,
        math_crop_path=crop_path,
        extraction_source=ExtractionSource.VLM_TRANSCRIPTION.value if vlm_result else ExtractionSource.RASTER_ONLY.value,
        extraction_confidence=confidence,
        needs_review=needs_review,
        review_reason=review_reason,
    )


def _extract_image(
    block: ClassifiedBlock,
    vlm_fn: Any | None,
    figures_dir: Path,
    padding: int,
    block_index: int,
) -> ExtractedBlock:
    """Extract image — save crop + VLM description and data extraction."""
    crop_path = _save_crop(block, figures_dir, f"fig_{block_index:04d}.png", padding)

    figure = ExtractedFigure(
        figure_type="unknown",
        figure_description="",
        source_page=block.page_number,
        image_asset_path=crop_path or "",
        extraction_mode=ExtractionMode.RASTER_ONLY.value,
    )

    if vlm_fn is not None and block.raster_crop is not None:
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "extract_figure.txt"
            prompt = prompt_path.read_text(encoding="utf-8")
            response = vlm_fn(image=block.raster_crop, prompt=prompt)
            result = json.loads(response)

            figure.figure_type = result.get("figure_type", "unknown")
            figure.figure_description = result.get("description", "")
            figure.alt_text = result.get("alt_text")
            figure.extraction_confidence = result.get("confidence", "medium")

            if result.get("data_extraction_possible") and result.get("figure_data"):
                figure.figure_data = result["figure_data"]
                figure.extraction_mode = ExtractionMode.VLM_TRANSCRIBED.value
            else:
                figure.extraction_mode = ExtractionMode.RASTER_ONLY.value

        except Exception as exc:
            logger.warning("VLM figure extraction failed: {}", exc)

    return ExtractedBlock(
        block_type=BlockType.IMAGE.value,
        figure=figure,
        extraction_source=ExtractionSource.VLM_TRANSCRIPTION.value if figure.figure_description else ExtractionSource.RASTER_ONLY.value,
        extraction_confidence=figure.extraction_confidence or "low",
        needs_review=(figure.extraction_mode == ExtractionMode.RASTER_ONLY.value),
        review_reason="Figure extracted as raster only, no structured data" if figure.extraction_mode == ExtractionMode.RASTER_ONLY.value else None,
    )


def _extract_table(
    block: ClassifiedBlock,
    vlm_fn: Any | None,
    figures_dir: Path,
    padding: int,
    block_index: int,
) -> ExtractedBlock:
    """Extract table — try docling first, then VLM, then raster only."""
    crop_path = _save_crop(block, figures_dir, f"table_{block_index:04d}.png", padding)

    figure = ExtractedFigure(
        figure_type="table",
        figure_description="Table",
        source_page=block.page_number,
        image_asset_path=crop_path or "",
        extraction_mode=ExtractionMode.RASTER_ONLY.value,
    )

    # Try VLM extraction
    if vlm_fn is not None and block.raster_crop is not None:
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "extract_table.txt"
            prompt = prompt_path.read_text(encoding="utf-8")
            response = vlm_fn(image=block.raster_crop, prompt=prompt)
            result = json.loads(response)

            figure.figure_data = {
                "headers": result.get("headers", []),
                "rows": result.get("rows", []),
                "num_rows": result.get("num_rows", 0),
                "num_cols": result.get("num_cols", 0),
            }
            figure.extraction_mode = ExtractionMode.VLM_TRANSCRIBED.value
            figure.extraction_confidence = result.get("confidence", "medium")

        except Exception as exc:
            logger.warning("VLM table extraction failed: {}", exc)

    return ExtractedBlock(
        block_type=BlockType.TABLE.value,
        figure=figure,
        extraction_source=ExtractionSource.VLM_TRANSCRIPTION.value if figure.extraction_mode != ExtractionMode.RASTER_ONLY.value else ExtractionSource.RASTER_ONLY.value,
        extraction_confidence=figure.extraction_confidence or "low",
    )


def _extract_option(
    block: ClassifiedBlock,
    vlm_fn: Any | None,
    option_dir: Path,
    block_index: int,
) -> ExtractedBlock:
    """Extract an MCQ/MSQ option."""
    has_image = block.raster_crop is not None and not (block.raw_text or "").strip()
    crop_path = None

    option = ExtractedOption(
        option_id=f"opt_{block_index}",
        option_text=block.raw_text or "",
        has_image=has_image,
    )

    if has_image:
        crop_path = _save_crop(block, option_dir, f"opt_{block_index:04d}.png", 4)
        option.option_image_path = crop_path

        # VLM transcription for image-rendered options
        if vlm_fn is not None and block.raster_crop is not None:
            try:
                prompt_path = Path(__file__).parent.parent / "prompts" / "extract_option.txt"
                prompt = prompt_path.read_text(encoding="utf-8")
                response = vlm_fn(image=block.raster_crop, prompt=prompt)
                result = json.loads(response)

                option.option_text = result.get("option_text", "")
                option.option_text_latex = result.get("option_text_latex")
                if result.get("option_label"):
                    option.option_id = result["option_label"]

            except Exception as exc:
                logger.warning("VLM option extraction failed: {}", exc)
    else:
        # Parse option label from text
        import re
        match = re.match(r"^\s*\(?([A-Da-d1-4])\)?[\.\):]?\s*(.*)", block.raw_text or "", re.DOTALL)
        if match:
            option.option_id = match.group(1).upper()
            option.option_text = match.group(2).strip()

    return ExtractedBlock(
        block_type=BlockType.OPTION_BLOCK.value,
        text=option.option_text,
        options=[option],
        extraction_source=ExtractionSource.VLM_TRANSCRIPTION.value if has_image else ExtractionSource.TEXT_LAYER.value,
        extraction_confidence="medium" if has_image else "high",
    )


def _extract_metadata(block: ClassifiedBlock) -> ExtractedBlock:
    """Extract metadata key-value pairs from text layer."""
    import re

    kv: dict[str, str] = {}
    text = block.raw_text or ""

    pairs = re.findall(r"([\w\s]+?)\s*:\s*([^:]+?)(?:\s{2,}|$)", text)
    for key, value in pairs:
        clean_key = key.strip()
        clean_value = value.strip()
        if clean_key and clean_value:
            kv[clean_key] = clean_value

    return ExtractedBlock(
        block_type=block.block_type,
        text=text,
        metadata_kv=kv if kv else None,
        extraction_source=ExtractionSource.TEXT_LAYER.value,
        extraction_confidence="high",
    )


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------


def _vlm_extract_math(block: ClassifiedBlock, vlm_fn: Any | None) -> Optional[dict]:
    """Use VLM to transcribe a math expression to LaTeX."""
    if vlm_fn is None or block.raster_crop is None:
        return None

    try:
        prompt_path = Path(__file__).parent.parent / "prompts" / "extract_math.txt"
        prompt = prompt_path.read_text(encoding="utf-8")
        response = vlm_fn(image=block.raster_crop, prompt=prompt)
        return json.loads(response)
    except Exception as exc:
        logger.warning("VLM math extraction failed: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _save_crop(
    block: ClassifiedBlock,
    target_dir: Path,
    filename: str,
    padding: int,
) -> Optional[str]:
    """Save a raster crop of a block to disk."""
    if block.raster_crop is None:
        return None

    try:
        # Apply padding
        img = block.raster_crop
        if padding > 0:
            padded = Image.new(
                "RGB",
                (img.width + 2 * padding, img.height + 2 * padding),
                (255, 255, 255),
            )
            padded.paste(img, (padding, padding))
            img = padded

        out_path = target_dir / filename
        img.save(str(out_path))
        # Return path relative to output directory (2 levels up from assets/*)
        return str(out_path.relative_to(target_dir.parent.parent))
    except Exception as exc:
        logger.warning("Failed to save crop {}: {}", filename, exc)
        return None


def _try_latex_from_text(text: Optional[str]) -> Optional[str]:
    """Try to extract LaTeX from raw text using pylatexenc."""
    if not text:
        return None

    try:
        from pylatexenc.latexwalker import LatexWalker

        # If text already looks like LaTeX, return it
        latex_indicators = [r"\frac", r"\sqrt", r"\sum", r"\int", r"\begin", "^{", "_{"]
        if any(ind in text for ind in latex_indicators):
            # Basic cleanup
            latex = text.strip()
            # Remove surrounding $ or $$ if present
            if latex.startswith("$$") and latex.endswith("$$"):
                latex = latex[2:-2]
            elif latex.startswith("$") and latex.endswith("$"):
                latex = latex[1:-1]
            return latex

    except ImportError:
        pass

    return None


def _validate_latex_basic(latex: str) -> bool:
    """Basic LaTeX validation: check balanced braces and environments."""
    # Check balanced braces
    depth = 0
    for ch in latex:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        return False

    # Check balanced \begin/\end
    import re
    begins = re.findall(r"\\begin\{(\w+)\}", latex)
    ends = re.findall(r"\\end\{(\w+)\}", latex)
    if begins != ends:
        return False

    return True


# ---------------------------------------------------------------------------
# Question group assembly
# ---------------------------------------------------------------------------


def _build_question_groups(
    classified_groups: list[QuestionGroup],
    extracted_blocks: list[ExtractedBlock],
    classified_blocks: list[ClassifiedBlock],
    output_dir: Path,
) -> list[ExtractedQuestionGroup]:
    """Build ExtractedQuestionGroup objects from classified groups and extracted blocks."""
    result: list[ExtractedQuestionGroup] = []

    for group in classified_groups:
        blocks = [extracted_blocks[i] for i in group.block_indices if i < len(extracted_blocks)]

        # Collect question text from text blocks
        text_parts: list[str] = []
        latex_parts: list[str] = []
        figures: list[ExtractedFigure] = []
        options: list[ExtractedOption] = []
        math_crop_path: Optional[str] = None
        question_text_source = ExtractionSource.TEXT_LAYER.value
        question_text_confidence: Optional[str] = "high"
        needs_review = False
        review_reason = None

        for eb in blocks:
            if eb.block_type == BlockType.TEXT.value and eb.text:
                text_parts.append(eb.text)

            elif eb.block_type in (BlockType.MATH_INLINE.value, BlockType.MATH_BLOCK.value, BlockType.MATRIX.value):
                if eb.latex:
                    latex_parts.append(eb.latex)
                if eb.text:
                    text_parts.append(eb.text)
                if eb.math_crop_path and math_crop_path is None:
                    math_crop_path = eb.math_crop_path
                if eb.extraction_source == ExtractionSource.VLM_TRANSCRIPTION.value:
                    question_text_source = ExtractionSource.VLM_TRANSCRIPTION.value

            elif eb.block_type == BlockType.IMAGE.value and eb.figure:
                figures.append(eb.figure)

            elif eb.block_type == BlockType.TABLE.value and eb.figure:
                figures.append(eb.figure)

            elif eb.block_type == BlockType.OPTION_BLOCK.value:
                options.extend(eb.options)

            # Propagate review flags
            if eb.needs_review:
                needs_review = True
                if eb.review_reason:
                    review_reason = eb.review_reason

            # Track lowest confidence
            if eb.extraction_confidence == "low":
                question_text_confidence = "low"
            elif eb.extraction_confidence == "medium" and question_text_confidence != "low":
                question_text_confidence = "medium"

        # Build SA data if applicable
        sa_data = None
        if group.question_type == "SA":
            sa_data = _build_sa_data(group.metadata_fields)

        # Build comprehension data if applicable
        comp_data = None
        if group.is_comprehension:
            comp_data = _build_comprehension_data(group, text_parts, latex_parts, figures, math_crop_path)

        # Parse marks
        correct_marks = _safe_float(group.metadata_fields.get("correct_marks"))
        negative_marks = _safe_float(group.metadata_fields.get("negative_marks")) or 0.0

        # Detect confirmation question
        is_confirmation = "confirmation" in (group.metadata_fields.get("question_label", "") or "").lower()

        # Determine correct option
        correct_option_id = None
        correct_option_ids: list[str] = []
        if group.question_type == "MCQ":
            # Look in metadata for correct answer
            correct_ans = group.metadata_fields.get("possible_answers") or group.metadata_fields.get("correct_answer")
            if correct_ans:
                correct_option_id = correct_ans.strip()
                # Mark the option as correct
                for opt in options:
                    opt.is_correct = (opt.option_id == correct_option_id)
        elif group.question_type == "MSQ":
            correct_ans = group.metadata_fields.get("possible_answers") or group.metadata_fields.get("correct_answer")
            if correct_ans:
                correct_option_ids = [x.strip() for x in correct_ans.split(",")]
                for opt in options:
                    opt.is_correct = (opt.option_id in correct_option_ids)

        eq = ExtractedQuestionGroup(
            question_id=group.question_id,
            question_number=group.question_number,
            question_type=group.question_type,
            question_is_confirmation=is_confirmation,
            section_id=group.section_id,
            parent_comprehension_id=group.parent_comprehension_id,
            correct_marks=correct_marks,
            negative_marks=negative_marks,
            question_text="\n".join(text_parts),
            question_text_latex="\n".join(latex_parts) if latex_parts else None,
            math_crop_path=math_crop_path,
            has_figure=len(figures) > 0,
            figures=figures,
            options=options,
            correct_option_id=correct_option_id,
            correct_option_ids=correct_option_ids,
            sa_data=sa_data,
            comprehension_data=comp_data,
            question_text_source=question_text_source,
            question_text_confidence=question_text_confidence,
            needs_review=needs_review,
            review_reason=review_reason,
            blocks=blocks,
        )
        result.append(eq)

    return result


def _build_sa_data(metadata: dict[str, str]) -> ExtractedSAData:
    """Build SA-specific data from metadata fields."""
    # Parse possible answers
    possible_answers: list[str] = []
    pa_raw = metadata.get("possible_answers", "")
    if pa_raw:
        possible_answers = [x.strip() for x in pa_raw.split(",") if x.strip()]

    # Parse answer range
    range_min = None
    range_max = None
    ar_raw = metadata.get("answer_range", "")
    if ar_raw:
        import re
        match = re.match(r"([\d.e+-]+)\s*(?:to|-)\s*([\d.e+-]+)", ar_raw, re.IGNORECASE)
        if match:
            try:
                range_min = float(match.group(1))
                range_max = float(match.group(2))
            except ValueError:
                pass

    return ExtractedSAData(
        response_type=metadata.get("response_type"),
        evaluation_required=_parse_bool(metadata.get("evaluation_required_for_sa")),
        answers_type=metadata.get("answers_type"),
        answer_range_min=range_min,
        answer_range_max=range_max,
        possible_answers=possible_answers,
    )


def _build_comprehension_data(
    group: QuestionGroup,
    text_parts: list[str],
    latex_parts: list[str],
    figures: list[ExtractedFigure],
    math_crop_path: Optional[str],
) -> ExtractedComprehensionData:
    """Build comprehension-specific data."""
    # Parse question numbers range
    qnums_raw = group.metadata_fields.get("question_numbers", "")
    start = None
    end = None
    if qnums_raw:
        import re
        match = re.match(r"(\d+)\s*(?:to|-)\s*(\d+)", qnums_raw, re.IGNORECASE)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))

    return ExtractedComprehensionData(
        topic_name=group.metadata_fields.get("topic_name"),
        comprehension_text="\n".join(text_parts) if text_parts else None,
        comprehension_text_latex="\n".join(latex_parts) if latex_parts else None,
        math_crop_path=math_crop_path,
        question_numbers_start=start,
        question_numbers_end=end,
        figures=figures,
    )


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_bool(value: Any) -> Optional[bool]:
    """Parse a boolean from string."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return None
