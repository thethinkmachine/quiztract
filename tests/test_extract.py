"""Tests for Stage 3 — Content Extraction."""

import pytest

from models.schema import (
    ClassifiedBlock,
    ClassifiedDocument,
    ExtractedBlock,
    ExtractedOption,
    PageDocument,
    QuestionGroup,
)
from pipeline.extract import (
    _try_latex_from_text,
    _validate_latex_basic,
    _extract_text,
    _extract_metadata,
)
from PIL import Image


@pytest.fixture
def mock_raster():
    return Image.new("RGB", (100, 140), (255, 255, 255))


class TestLatexUtils:
    def test_try_latex_from_text(self):
        assert _try_latex_from_text(r"\frac{a}{b}") is not None
        assert _try_latex_from_text("plain text") is None
        assert _try_latex_from_text(r"$$\sqrt{x}$$") == r"\sqrt{x}"
        assert _try_latex_from_text(None) is None

    def test_validate_latex_basic(self):
        assert _validate_latex_basic(r"\frac{a}{b}") is True
        assert _validate_latex_basic(r"\frac{a}{b") is False  # Unbalanced
        assert _validate_latex_basic(r"\begin{matrix}1\\2\end{matrix}") is True
        assert _validate_latex_basic(r"\begin{matrix}1\\2") is False  # Missing \end


class TestExtractText:
    def test_plain_text(self):
        block = ClassifiedBlock(
            block_type="text",
            raw_text="Hello world",
            page_number=1,
        )
        result = _extract_text(block)
        assert result.text == "Hello world"
        assert result.extraction_source == "text_layer"
        assert result.extraction_confidence == "high"


class TestExtractMetadata:
    def test_metadata_kv(self):
        block = ClassifiedBlock(
            block_type="metadata_field",
            raw_text="Response Type : Numeric  Answers Type : Range",
            page_number=1,
        )
        result = _extract_metadata(block)
        assert result.metadata_kv is not None
        assert "Response Type" in result.metadata_kv


class TestExtractedOption:
    def test_option_creation(self):
        opt = ExtractedOption(
            option_id="A",
            option_text="42",
            has_image=False,
        )
        assert opt.option_id == "A"
        assert opt.option_text == "42"
        assert opt.is_correct is None
