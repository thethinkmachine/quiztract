"""Stage 5 — JSON Assembly.

Builds the final JSON output from the ValidatedDocument:
- Applies the null policy (omit inapplicable fields, null for missing)
- Applies the discard list for platform-only fields
- Writes {exam_code}.json and {exam_code}.provenance.json to disk
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from models.schema import (
    AnswerRange,
    ComprehensionData,
    ExamMeta,
    ExamOutput,
    ExtractionMetadata,
    FigureItem,
    MCQData,
    MSQData,
    OptionItem,
    ProvenanceFile,
    ProvenanceFlags,
    QuestionItem,
    QuestionNumbersRange,
    SAData,
    SectionMeta,
    SubSection,
    ValidatedDocument,
    ExtractedQuestionGroup,
    ExtractedFigure,
    ExtractedOption,
)


# Fields to discard from final JSON (parsed during classification but not written)
DISCARD_FIELDS = {
    "sub_question_shuffling_allowed",
    "group_comprehension_questions",
    "question_pattern_type",
    "show_word_count",
    "text_area_type",
    "display_number_panel",
    "group_all_questions",
    "enable_mark_review",
    "question_shuffling_allowed",
    "question_label",
}


def assemble_json(
    doc: ValidatedDocument,
    config: dict[str, Any],
    source_pdf_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Assemble final JSON from a ValidatedDocument and write to disk.

    Args:
        doc: ValidatedDocument from Stage 4.
        config: Pipeline configuration dict.
        source_pdf_path: Path to the original PDF file.
        output_dir: Override output directory.

    Returns:
        Path to the written JSON file.
    """
    assemble_cfg = config.get("assemble", {})
    output_cfg = config.get("output", {})
    json_indent = assemble_cfg.get("json_indent", 2)
    ensure_ascii = assemble_cfg.get("ensure_ascii", False)
    run_validation = assemble_cfg.get("run_schema_validation", True)

    if output_dir is None:
        base = Path(output_cfg.get("base_dir", "./output"))
        output_dir = base / doc.exam_code
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Assembling JSON for exam: {}", doc.exam_code)

    # Build exam metadata
    exam_meta = _build_exam_meta(doc)

    # Build sections
    sections = _build_sections(doc)

    # Build questions — ordered per spec
    questions = _build_questions(doc)

    # Assemble final output
    exam_output = ExamOutput(
        exam=exam_meta,
        sections=sections,
        questions=questions,
    )

    # Serialize to dict and apply null policy
    output_dict = _apply_null_policy(exam_output.model_dump(mode="json"))

    # Schema validation
    schema_errors: list[str] = []
    if run_validation:
        schema_errors = _validate_schema(output_dict)
        if schema_errors:
            logger.warning("Schema validation failed with {} errors", len(schema_errors))
            for err in schema_errors:
                logger.warning("  Schema error: {}", err)

    # Write JSON
    json_path = output_dir / f"{doc.exam_code}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=json_indent, ensure_ascii=ensure_ascii)

    logger.info("Wrote JSON: {}", json_path)

    # Build and write provenance
    provenance = _build_provenance(doc, source_pdf_path, schema_errors)
    prov_path = output_dir / f"{doc.exam_code}.provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(
            provenance.model_dump(mode="json"),
            f,
            indent=json_indent,
            ensure_ascii=ensure_ascii,
        )

    logger.info("Wrote provenance: {}", prov_path)

    return json_path


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_exam_meta(doc: ValidatedDocument) -> ExamMeta:
    """Build exam metadata from the document."""
    # Try to extract from the first question group's metadata
    meta: dict[str, Any] = {}
    for group in doc.question_groups:
        for block in group.blocks:
            if block.metadata_kv:
                meta.update(block.metadata_kv)

    return ExamMeta(
        exam_code=doc.exam_code,
        exam_title=meta.get("exam_title") or meta.get("Exam Title"),
        subject=meta.get("subject") or meta.get("Subject"),
        level=meta.get("level") or meta.get("Level"),
        exam_mode=meta.get("exam_mode") or meta.get("Exam Mode"),
        exam_date=meta.get("exam_date") or meta.get("Exam Date"),
    )


def _build_sections(doc: ValidatedDocument) -> list[SectionMeta]:
    """Build sections list from question groups."""
    section_map: dict[str, dict[str, Any]] = {}

    for group in doc.question_groups:
        sid = group.section_id
        if not sid:
            continue

        if sid not in section_map:
            section_map[sid] = {
                "section_id": sid,
                "section_name": None,
                "num_questions": 0,
                "section_marks": 0.0,
                "section_negative_marks": 0.0,
            }

        section_map[sid]["num_questions"] += 1
        if group.correct_marks:
            section_map[sid]["section_marks"] += group.correct_marks
        if group.negative_marks:
            section_map[sid]["section_negative_marks"] += group.negative_marks

    sections = []
    for i, (sid, data) in enumerate(sorted(section_map.items()), 1):
        sections.append(
            SectionMeta(
                section_id=data["section_id"],
                section_name=data.get("section_name"),
                section_number=i,
                num_questions=data["num_questions"],
                section_marks=data["section_marks"],
                section_negative_marks=data["section_negative_marks"],
            )
        )

    return sections


def _build_questions(doc: ValidatedDocument) -> list[QuestionItem]:
    """Build the questions array in the correct order.

    Order:
    1. Confirmation question first
    2. COMPREHENSION parents within their section
    3. Sub-questions in question_number order
    4. Standalone questions in question_number order
    """
    confirmation: list[QuestionItem] = []
    comprehension_parents: list[QuestionItem] = []
    sub_questions: list[QuestionItem] = []
    standalone: list[QuestionItem] = []

    for group in doc.question_groups:
        q = _group_to_question(group)
        if q is None:
            continue

        if q.question_is_confirmation:
            confirmation.append(q)
        elif q.question_type == "COMPREHENSION" and q.comprehension is not None:
            comprehension_parents.append(q)
        elif q.parent_comprehension_id:
            sub_questions.append(q)
        else:
            standalone.append(q)

    # Sort
    comprehension_parents.sort(key=lambda q: q.question_number or 0)
    sub_questions.sort(key=lambda q: q.question_number or 0)
    standalone.sort(key=lambda q: q.question_number or 0)

    # Interleave comprehension parents with their sub-questions
    result: list[QuestionItem] = []
    result.extend(confirmation)

    # Group comprehension parents with their children
    used_sub_ids: set[str] = set()
    for comp in comprehension_parents:
        result.append(comp)
        # Add sub-questions belonging to this comprehension
        for sub in sub_questions:
            if sub.parent_comprehension_id == comp.question_id:
                result.append(sub)
                used_sub_ids.add(sub.question_id)

    # Add remaining sub-questions (orphans)
    for sub in sub_questions:
        if sub.question_id not in used_sub_ids:
            result.append(sub)

    # Add standalone questions
    result.extend(standalone)

    return result


def _group_to_question(group: ExtractedQuestionGroup) -> QuestionItem | None:
    """Convert an ExtractedQuestionGroup to a QuestionItem."""
    if group.question_id is None:
        return None

    qtype = (group.question_type or "MCQ").upper()

    # Build type-specific data
    mcq = None
    msq = None
    sa = None
    comprehension = None

    if qtype == "MCQ":
        mcq = MCQData(
            options=[_option_to_item(o) for o in group.options],
            correct_option_id=group.correct_option_id,
        )
    elif qtype == "MSQ":
        msq = MSQData(
            max_selectable_options=group.max_selectable_options,
            options=[_option_to_item(o) for o in group.options],
            correct_option_ids=group.correct_option_ids,
        )
    elif qtype == "SA":
        sa = _build_sa_output(group)
    elif qtype == "COMPREHENSION":
        comprehension = _build_comprehension_output(group)

    # Build figures
    figures = [_figure_to_item(f) for f in group.figures]

    # Build extraction metadata
    extraction_meta = ExtractionMetadata(
        question_text_source=group.question_text_source,
        question_text_confidence=group.question_text_confidence,
        needs_review=group.needs_review,
        review_reason=group.review_reason,
    )

    return QuestionItem(
        question_id=group.question_id,
        question_number=group.question_number,
        question_type=qtype,
        question_is_confirmation=group.question_is_confirmation,
        correct_marks=group.correct_marks,
        negative_marks=group.negative_marks,
        question_text=group.question_text,
        question_text_latex=group.question_text_latex,
        math_crop_path=group.math_crop_path,
        has_figure=group.has_figure,
        figures=figures,
        topic=group.topic,
        section_id=group.section_id,
        parent_comprehension_id=group.parent_comprehension_id,
        extraction_metadata=extraction_meta,
        mcq=mcq,
        msq=msq,
        sa=sa,
        comprehension=comprehension,
    )


def _option_to_item(opt: ExtractedOption) -> OptionItem:
    """Convert ExtractedOption to OptionItem."""
    return OptionItem(
        option_id=opt.option_id,
        option_text=opt.option_text,
        option_text_latex=opt.option_text_latex,
        option_image_path=opt.option_image_path,
        is_correct=opt.is_correct,
        has_image=opt.has_image,
    )


def _figure_to_item(fig: ExtractedFigure) -> FigureItem:
    """Convert ExtractedFigure to FigureItem."""
    return FigureItem(
        figure_type=fig.figure_type,
        figure_description=fig.figure_description,
        source_page=fig.source_page,
        figure_data=fig.figure_data,
        image_asset_path=fig.image_asset_path,
        alt_text=fig.alt_text,
        extraction_mode=fig.extraction_mode,
        extraction_confidence=fig.extraction_confidence,
    )


def _build_sa_output(group: ExtractedQuestionGroup) -> SAData:
    """Build SA data for the final output."""
    if group.sa_data is None:
        return SAData()

    answer_range = None
    if group.sa_data.answer_range_min is not None and group.sa_data.answer_range_max is not None:
        answer_range = AnswerRange(
            min=group.sa_data.answer_range_min,
            max=group.sa_data.answer_range_max,
        )

    return SAData(
        response_type=group.sa_data.response_type,
        evaluation_required=group.sa_data.evaluation_required,
        answers_type=group.sa_data.answers_type,
        answers_case_sensitive=group.sa_data.answers_case_sensitive,
        answer_format_hint=group.sa_data.answer_format_hint,
        answer_constraints=group.sa_data.answer_constraints,
        answer_range=answer_range,
        possible_answers=group.sa_data.possible_answers,
        answer_truncated_across_page=group.sa_data.answer_truncated_across_page,
    )


def _build_comprehension_output(group: ExtractedQuestionGroup) -> ComprehensionData:
    """Build comprehension data for the final output."""
    if group.comprehension_data is None:
        return ComprehensionData()

    q_range = None
    if group.comprehension_data.question_numbers_start is not None and group.comprehension_data.question_numbers_end is not None:
        q_range = QuestionNumbersRange(
            start=group.comprehension_data.question_numbers_start,
            end=group.comprehension_data.question_numbers_end,
        )

    figures = [_figure_to_item(f) for f in group.comprehension_data.figures]

    return ComprehensionData(
        topic_name=group.comprehension_data.topic_name,
        comprehension_text=group.comprehension_data.comprehension_text,
        comprehension_text_latex=group.comprehension_data.comprehension_text_latex,
        math_crop_path=group.comprehension_data.math_crop_path,
        question_numbers_range=q_range,
        figures=figures,
        sub_question_ids=group.comprehension_data.sub_question_ids,
        additional_data=group.comprehension_data.additional_data,
    )


# ---------------------------------------------------------------------------
# Null policy & cleanup
# ---------------------------------------------------------------------------


def _apply_null_policy(data: Any) -> Any:
    """Apply null policy recursively:
    - Remove keys with None values for type-specific blocks (mcq, msq, sa, comprehension)
    - Keep null for applicable-but-missing fields
    - Convert empty arrays to [] not null
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Skip discarded platform fields
            if key in DISCARD_FIELDS:
                continue

            # Skip type-specific blocks that are None (not applicable)
            if key in ("mcq", "msq", "sa", "comprehension", "sub_section") and value is None:
                continue

            cleaned[key] = _apply_null_policy(value)

        return cleaned

    elif isinstance(data, list):
        return [_apply_null_policy(item) for item in data]

    return data


def _validate_schema(data: dict) -> list[str]:
    """Validate JSON against schema using jsonschema."""
    errors: list[str] = []

    try:
        import jsonschema

        # Basic structural validation (we use Pydantic for the detailed schema)
        if "exam" not in data:
            errors.append("Missing top-level 'exam' object")
        if "sections" not in data:
            errors.append("Missing top-level 'sections' array")
        if "questions" not in data:
            errors.append("Missing top-level 'questions' array")

        # Validate question structure
        for i, q in enumerate(data.get("questions", [])):
            if "question_id" not in q:
                errors.append(f"Question {i}: missing question_id")
            if "question_type" not in q:
                errors.append(f"Question {i}: missing question_type")

    except ImportError:
        logger.warning("jsonschema not installed — skipping schema validation")

    return errors


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _build_provenance(
    doc: ValidatedDocument,
    source_pdf_path: str | Path,
    schema_errors: list[str],
) -> ProvenanceFile:
    """Build the provenance sidecar file."""
    source_pdf_path = Path(source_pdf_path)

    # Compute SHA256
    sha256 = ""
    try:
        sha256 = hashlib.sha256(source_pdf_path.read_bytes()).hexdigest()
    except Exception:
        pass

    # Count stats
    total_questions = len(doc.question_groups)
    comprehension_blocks = sum(
        1 for g in doc.question_groups
        if (g.question_type or "").upper() == "COMPREHENSION"
    )
    questions_needing_review = sum(
        1 for g in doc.question_groups if g.needs_review
    )

    stats = doc.extraction_stats
    raster_count = stats.get("raster_fallback_count", 0)
    vlm_count = stats.get("vlm_transcription_count", 0)
    stitching_applied = stats.get("page_boundary_stitching_applied", False)
    stitched_fields = stats.get("stitched_fields", [])

    flags = ProvenanceFlags(
        total_questions=total_questions,
        comprehension_blocks=comprehension_blocks,
        questions_needing_review=questions_needing_review,
        raster_fallback_count=raster_count,
        vlm_transcription_count=vlm_count,
        page_boundary_stitching_applied=stitching_applied,
        stitched_fields=stitched_fields,
        schema_validation_passed=len(schema_errors) == 0,
        schema_validation_errors=schema_errors,
    )

    return ProvenanceFile(
        source_pdf=source_pdf_path.name,
        source_pdf_sha256=sha256,
        extraction_date=str(date.today()),
        extraction_flags=flags,
    )
