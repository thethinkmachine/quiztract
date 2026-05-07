"""Enum definitions for the Quiztract pipeline.

All enumerations used across pipeline stages, schema models,
and configuration are defined here.
"""

from enum import Enum


class BlockType(str, Enum):
    """Block type tags assigned during classification (Stage 2)."""

    TEXT = "text"
    MATH_INLINE = "math_inline"
    MATH_BLOCK = "math_block"
    MATRIX = "matrix"
    IMAGE = "image"
    TABLE = "table"
    METADATA_FIELD = "metadata_field"
    QUESTION_HEADER = "question_header"
    OPTION_BLOCK = "option_block"
    ANSWER_BLOCK = "answer_block"
    SECTION_HEADER = "section_header"
    PAGE_ARTIFACT = "page_artifact"
    UNKNOWN = "unknown"


class QuestionType(str, Enum):
    """Supported question types."""

    MCQ = "MCQ"
    MSQ = "MSQ"
    SA = "SA"
    COMPREHENSION = "COMPREHENSION"


class ExtractionSource(str, Enum):
    """How content was extracted for a block or question."""

    TEXT_LAYER = "text_layer"
    VLM_TRANSCRIPTION = "vlm_transcription"
    RASTER_ONLY = "raster_only"


class ExtractionConfidence(str, Enum):
    """Confidence level of an extraction result."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExtractionMode(str, Enum):
    """Extraction mode for figures and tables."""

    STRUCTURED = "structured"
    VLM_TRANSCRIBED = "vlm_transcribed"
    RASTER_ONLY = "raster_only"


class ClassificationSource(str, Enum):
    """How a block was classified."""

    RULE = "rule"
    VLM = "vlm"
    UNKNOWN = "unknown"


class ResponseType(str, Enum):
    """SA question response type."""

    ALPHANUMERIC = "Alphanumeric"
    NUMERIC = "Numeric"


class AnswersType(str, Enum):
    """SA question answers type."""

    EQUAL = "Equal"
    SET = "Set"
    RANGE = "Range"


class SectionMandatory(str, Enum):
    """Whether a section is mandatory or optional."""

    MANDATORY = "Mandatory"
    OPTIONAL = "Optional"


class ErrorLevel(str, Enum):
    """Pipeline error severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    REVIEW = "REVIEW"
    ERROR = "ERROR"
    FATAL = "FATAL"
