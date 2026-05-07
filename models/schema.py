"""Pydantic models mirroring the JSON schema and pipeline intermediate representations.

Models are organized in three groups:
1. Pipeline intermediate representations (PageDocument → ClassifiedDocument → ExtractedDocument → ValidatedDocument)
2. Final JSON output schema (ExamOutput containing ExamMeta, SectionMeta, QuestionItem)
3. Provenance sidecar schema (ProvenanceFile)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from PIL import Image as PILImage
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Pipeline Intermediate Representations
# ---------------------------------------------------------------------------


class PageDocument(BaseModel):
    """Stage 1 output: one entry per PDF page."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_number: int = Field(..., description="1-based page number")
    raw_text: str = Field(default="", description="Full text layer of the page, may be empty")
    raster_image: PILImage.Image = Field(..., description="PIL Image at configured DPI")
    width_px: int = Field(..., description="Raster width in pixels")
    height_px: int = Field(..., description="Raster height in pixels")
    docling_blocks: list[Any] = Field(default_factory=list, description="Raw docling block objects")
    parse_failed: bool = Field(default=False, description="True if docling failed to parse this page")


class ClassifiedBlock(BaseModel):
    """A single block with its classification from Stage 2."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    block_type: str = Field(..., description="Block type tag")
    raw_text: Optional[str] = Field(default=None, description="Text content if available")
    bounding_box: Optional[tuple[int, int, int, int]] = Field(
        default=None, description="x0, y0, x1, y1 in pixels"
    )
    page_number: int = Field(..., description="Source page number (1-based)")
    classification_source: str = Field(
        default="rule", description="How this block was classified: rule | vlm | unknown"
    )
    classification_confidence: str = Field(
        default="high", description="Confidence: high | medium | low"
    )
    raster_crop: Optional[PILImage.Image] = Field(
        default=None, description="Cropped raster of this block"
    )


class QuestionGroup(BaseModel):
    """A group of blocks belonging to a single question or comprehension block."""

    question_id: Optional[str] = Field(default=None)
    question_type: Optional[str] = Field(default=None)
    question_number: Optional[int] = Field(default=None)
    section_id: Optional[str] = Field(default=None)
    is_comprehension: bool = Field(default=False)
    parent_comprehension_id: Optional[str] = Field(default=None)
    block_indices: list[int] = Field(default_factory=list, description="Indices into ClassifiedDocument.blocks")
    metadata_fields: dict[str, str] = Field(default_factory=dict, description="Parsed key-value metadata fields")
    start_page: int = Field(default=0)
    end_page: int = Field(default=0)


class ClassifiedDocument(BaseModel):
    """Stage 2 output: all blocks classified, question boundaries detected."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    exam_code: str
    pages: list[PageDocument] = Field(default_factory=list)
    blocks: list[ClassifiedBlock] = Field(default_factory=list)
    question_groups: list[QuestionGroup] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extracted representations (Stage 3 output)
# ---------------------------------------------------------------------------


class ExtractedFigure(BaseModel):
    """A figure extracted from the document."""

    figure_type: str = Field(default="unknown", description="E.g. graph, diagram, chart, table, matrix, tree")
    figure_description: str = Field(default="", description="VLM-generated description")
    source_page: int = Field(..., description="Page number where figure was found")
    figure_data: Optional[dict[str, Any]] = Field(
        default=None, description="Structured data if extraction succeeded"
    )
    image_asset_path: str = Field(..., description="Relative path to saved raster crop")
    alt_text: Optional[str] = Field(default=None)
    extraction_mode: str = Field(default="raster_only")
    extraction_confidence: Optional[str] = Field(default=None)


class ExtractedOption(BaseModel):
    """An extracted MCQ/MSQ option."""

    option_id: str
    option_text: str = Field(default="")
    option_text_latex: Optional[str] = Field(default=None)
    option_image_path: Optional[str] = Field(default=None)
    has_image: bool = Field(default=False)
    is_correct: Optional[bool] = Field(default=None)


class ExtractedBlock(BaseModel):
    """Stage 3 output per block: content extracted via appropriate method."""

    block_type: str
    text: Optional[str] = Field(default=None)
    latex: Optional[str] = Field(default=None)
    math_crop_path: Optional[str] = Field(default=None)
    figure: Optional[ExtractedFigure] = Field(default=None)
    metadata_kv: Optional[dict[str, str]] = Field(default=None)
    options: list[ExtractedOption] = Field(default_factory=list)
    extraction_source: str = Field(default="text_layer")
    extraction_confidence: str = Field(default="high")
    needs_review: bool = Field(default=False)
    review_reason: Optional[str] = Field(default=None)


class ExtractedSAData(BaseModel):
    """SA-specific extracted data."""

    response_type: Optional[str] = Field(default=None)
    evaluation_required: Optional[bool] = Field(default=None)
    answers_type: Optional[str] = Field(default=None)
    answers_case_sensitive: Optional[bool] = Field(default=None)
    answer_format_hint: Optional[str] = Field(default=None)
    answer_constraints: Optional[str] = Field(default=None)
    answer_range_min: Optional[float] = Field(default=None)
    answer_range_max: Optional[float] = Field(default=None)
    possible_answers: list[str] = Field(default_factory=list)
    answer_truncated_across_page: bool = Field(default=False)


class ExtractedComprehensionData(BaseModel):
    """Comprehension-specific extracted data."""

    topic_name: Optional[str] = Field(default=None)
    comprehension_text: Optional[str] = Field(default=None)
    comprehension_text_latex: Optional[str] = Field(default=None)
    math_crop_path: Optional[str] = Field(default=None)
    question_numbers_start: Optional[int] = Field(default=None)
    question_numbers_end: Optional[int] = Field(default=None)
    figures: list[ExtractedFigure] = Field(default_factory=list)
    sub_question_ids: list[str] = Field(default_factory=list)
    additional_data: Optional[dict[str, Any]] = Field(default=None)


class ExtractedQuestionGroup(BaseModel):
    """All extracted content for a single question or comprehension block."""

    question_id: Optional[str] = Field(default=None)
    question_number: Optional[int] = Field(default=None)
    question_type: Optional[str] = Field(default=None)
    question_is_confirmation: bool = Field(default=False)
    section_id: Optional[str] = Field(default=None)
    sub_section: Optional[dict[str, Any]] = Field(default=None)
    parent_comprehension_id: Optional[str] = Field(default=None)
    correct_marks: Optional[float] = Field(default=None)
    negative_marks: Optional[float] = Field(default=0.0)

    # Stem content
    question_text: str = Field(default="")
    question_text_latex: Optional[str] = Field(default=None)
    math_crop_path: Optional[str] = Field(default=None)

    # Figures
    has_figure: bool = Field(default=False)
    figures: list[ExtractedFigure] = Field(default_factory=list)

    # Type-specific data
    options: list[ExtractedOption] = Field(default_factory=list)
    correct_option_id: Optional[str] = Field(default=None)
    correct_option_ids: list[str] = Field(default_factory=list)
    max_selectable_options: Optional[int] = Field(default=None)
    sa_data: Optional[ExtractedSAData] = Field(default=None)
    comprehension_data: Optional[ExtractedComprehensionData] = Field(default=None)

    # Extraction metadata
    question_text_source: str = Field(default="text_layer")
    question_text_confidence: Optional[str] = Field(default=None)
    needs_review: bool = Field(default=False)
    review_reason: Optional[str] = Field(default=None)

    # All blocks belonging to this question
    blocks: list[ExtractedBlock] = Field(default_factory=list)

    # Topic
    topic: Optional[str] = Field(default=None)


class ExtractedDocument(BaseModel):
    """Stage 3 output: all content extracted and organized by question."""

    exam_code: str
    question_groups: list[ExtractedQuestionGroup] = Field(default_factory=list)
    extraction_stats: dict[str, Any] = Field(default_factory=dict)


class ValidatedDocument(BaseModel):
    """Stage 4 output: identical to ExtractedDocument with review flags populated."""

    exam_code: str
    question_groups: list[ExtractedQuestionGroup] = Field(default_factory=list)
    extraction_stats: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final JSON Output Schema
# ---------------------------------------------------------------------------


class ExamMeta(BaseModel):
    """Top-level exam metadata."""

    exam_title: Optional[str] = Field(default=None)
    exam_code: Optional[str] = Field(default=None)
    subject: Optional[str] = Field(default=None)
    level: Optional[str] = Field(default=None)
    exam_mode: Optional[str] = Field(default=None)
    exam_date: Optional[str] = Field(default=None, description="YYYY-MM-DD format or null")


class SectionMeta(BaseModel):
    """Section metadata."""

    section_id: str
    section_name: Optional[str] = Field(default=None)
    section_number: Optional[int] = Field(default=None)
    mandatory_or_optional: Optional[str] = Field(default=None)
    num_questions: int = Field(default=0)
    num_questions_to_attempt: Optional[int] = Field(default=None)
    section_marks: float = Field(default=0.0)
    section_negative_marks: Optional[float] = Field(default=None)
    section_max_duration_minutes: Optional[float] = Field(default=None)


class FigureItem(BaseModel):
    """A figure within a question or comprehension block."""

    figure_type: str = Field(default="unknown")
    figure_description: str = Field(default="")
    source_page: int
    figure_data: Optional[dict[str, Any]] = Field(default=None)
    image_asset_path: str
    alt_text: Optional[str] = Field(default=None)
    extraction_mode: str = Field(default="raster_only")
    extraction_confidence: Optional[str] = Field(default=None)


class OptionItem(BaseModel):
    """An MCQ or MSQ option."""

    option_id: str
    option_text: str = Field(default="")
    option_text_latex: Optional[str] = Field(default=None)
    option_image_path: Optional[str] = Field(default=None)
    is_correct: Optional[bool] = Field(default=None)
    has_image: bool = Field(default=False)


class MCQData(BaseModel):
    """MCQ-specific question data."""

    options: list[OptionItem] = Field(default_factory=list)
    correct_option_id: Optional[str] = Field(default=None)


class MSQData(BaseModel):
    """MSQ-specific question data."""

    max_selectable_options: Optional[int] = Field(default=None)
    options: list[OptionItem] = Field(default_factory=list)
    correct_option_ids: list[str] = Field(default_factory=list)


class AnswerRange(BaseModel):
    """Numeric range for SA answers."""

    min: float
    max: float


class SAData(BaseModel):
    """SA-specific question data."""

    response_type: Optional[str] = Field(default=None)
    evaluation_required: Optional[bool] = Field(default=None)
    answers_type: Optional[str] = Field(default=None)
    answers_case_sensitive: Optional[bool] = Field(default=None)
    answer_format_hint: Optional[str] = Field(default=None)
    answer_constraints: Optional[str] = Field(default=None)
    answer_range: Optional[AnswerRange] = Field(default=None)
    possible_answers: list[str] = Field(default_factory=list)
    answer_truncated_across_page: bool = Field(default=False)


class QuestionNumbersRange(BaseModel):
    """Range of sub-question numbers in a comprehension block."""

    start: int
    end: int


class ComprehensionData(BaseModel):
    """Comprehension-specific question data."""

    topic_name: Optional[str] = Field(default=None)
    comprehension_text: Optional[str] = Field(default=None)
    comprehension_text_latex: Optional[str] = Field(default=None)
    math_crop_path: Optional[str] = Field(default=None)
    question_numbers_range: Optional[QuestionNumbersRange] = Field(default=None)
    figures: list[FigureItem] = Field(default_factory=list)
    sub_question_ids: list[str] = Field(default_factory=list)
    additional_data: Optional[dict[str, Any]] = Field(default=None)


class ExtractionMetadata(BaseModel):
    """Extraction provenance for a question."""

    question_text_source: str = Field(default="text_layer")
    question_text_confidence: Optional[str] = Field(default=None)
    needs_review: bool = Field(default=False)
    review_reason: Optional[str] = Field(default=None)


class SubSection(BaseModel):
    """Sub-section information."""

    sub_section_id: Optional[str] = Field(default=None)
    sub_section_name: Optional[str] = Field(default=None)
    sub_section_number: Optional[int] = Field(default=None)


class QuestionItem(BaseModel):
    """A single question in the final JSON output."""

    question_id: str
    question_number: Optional[int] = Field(default=None)
    question_type: str
    question_is_confirmation: bool = Field(default=False)
    correct_marks: Optional[float] = Field(default=None)
    negative_marks: Optional[float] = Field(default=0.0)
    question_text: str = Field(default="")
    question_text_latex: Optional[str] = Field(default=None)
    math_crop_path: Optional[str] = Field(default=None)
    has_figure: bool = Field(default=False)
    figures: list[FigureItem] = Field(default_factory=list)
    topic: Optional[str] = Field(default=None)
    section_id: Optional[str] = Field(default=None)
    sub_section: Optional[SubSection] = Field(default=None)
    parent_comprehension_id: Optional[str] = Field(default=None)
    extraction_metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)

    # Type-specific data — only the applicable key is present
    mcq: Optional[MCQData] = Field(default=None)
    msq: Optional[MSQData] = Field(default=None)
    sa: Optional[SAData] = Field(default=None)
    comprehension: Optional[ComprehensionData] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Conditional fields: mcq, msq, sa, comprehension are mutually exclusive and only the one matching question_type is present."
        }
    )


class ExamOutput(BaseModel):
    """Top-level JSON output structure."""

    exam: ExamMeta
    sections: list[SectionMeta] = Field(default_factory=list)
    questions: list[QuestionItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Provenance Sidecar
# ---------------------------------------------------------------------------


class ProvenanceFlags(BaseModel):
    """Extraction statistics for the provenance file."""

    total_questions: int = Field(default=0)
    comprehension_blocks: int = Field(default=0)
    questions_needing_review: int = Field(default=0)
    raster_fallback_count: int = Field(default=0)
    vlm_transcription_count: int = Field(default=0)
    page_boundary_stitching_applied: bool = Field(default=False)
    stitched_fields: list[str] = Field(default_factory=list)
    schema_validation_passed: bool = Field(default=True)
    schema_validation_errors: list[str] = Field(default_factory=list)


class ProvenanceFile(BaseModel):
    """Provenance sidecar JSON structure."""

    source_pdf: str
    source_pdf_sha256: str = Field(default="")
    extraction_model: str = Field(default="ibm-granite/granite-vision-4.1-4b")
    extraction_date: str = Field(default="")
    pipeline_version: str = Field(default="1.0.0")
    extraction_flags: ProvenanceFlags = Field(default_factory=ProvenanceFlags)
