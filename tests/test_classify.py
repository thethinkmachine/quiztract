"""Tests for Stage 2 — Block Classification."""

import pytest
from PIL import Image

from models.schema import PageDocument, ClassifiedBlock, QuestionGroup
from pipeline.classify import (
    classify_document,
    _parse_question_header,
    _parse_metadata_line,
    _is_metadata_line,
    _is_page_artifact,
    _has_inline_math,
)


@pytest.fixture
def default_config():
    return {
        "classify": {
            "confidence_threshold": 0.75,
            "metadata_field_patterns": [
                "Question Id", "Question Type", "Correct Marks",
                "Negative Marks", "Section",
            ],
        }
    }


@pytest.fixture
def mock_raster():
    return Image.new("RGB", (100, 140), (255, 255, 255))


class TestParseQuestionHeader:
    def test_basic_header(self):
        text = "Question Id : 123456  Question Type : MCQ  Correct Marks : 2  Negative Marks : 0.67"
        result = _parse_question_header(text)
        assert result["question_id"] == "123456"
        assert result["question_type"] == "MCQ"
        assert result["correct_marks"] == "2"
        assert result["negative_marks"] == "0.67"

    def test_comprehension_header(self):
        text = "Question Id : 789  Question Type : COMPREHENSION  Question Numbers : 1 - 5"
        result = _parse_question_header(text)
        assert result["question_id"] == "789"
        assert result["question_type"] == "COMPREHENSION"


class TestMetadataLine:
    def test_parse_kv(self):
        text = "Response Type : Numeric  Answers Type : Range"
        result = _parse_metadata_line(text)
        assert "response_type" in result
        assert "answers_type" in result

    def test_is_metadata(self):
        keys = {"question id", "question type", "correct marks"}
        assert _is_metadata_line("Question Id : 123", keys) is True
        assert _is_metadata_line("Hello world", keys) is False


class TestPageArtifact:
    def test_page_number(self):
        assert _is_page_artifact("Page 1 of 10") is True
        assert _is_page_artifact("- 5 -") is True
        assert _is_page_artifact("This is normal text") is False


class TestInlineMath:
    def test_detect(self):
        assert _has_inline_math(r"The value of \frac{a}{b}") is True
        assert _has_inline_math("No math here") is False
        assert _has_inline_math(r"\sqrt{x^2 + 1}") is True


class TestClassifyDocument:
    def test_empty_document(self, default_config, mock_raster):
        page = PageDocument(
            page_number=1,
            raw_text="",
            raster_image=mock_raster,
            width_px=100,
            height_px=140,
            docling_blocks=[],
        )
        result = classify_document([page], "TEST001", default_config)
        assert result.exam_code == "TEST001"
        assert len(result.question_groups) == 0

    def test_single_question(self, default_config, mock_raster):
        text = "Question Id : Q001  Question Type : MCQ  Correct Marks : 2\nWhat is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6"
        page = PageDocument(
            page_number=1,
            raw_text=text,
            raster_image=mock_raster,
            width_px=100,
            height_px=140,
            docling_blocks=[],
        )
        result = classify_document([page], "TEST001", default_config)
        assert len(result.question_groups) >= 1
        assert result.question_groups[0].question_id == "Q001"
