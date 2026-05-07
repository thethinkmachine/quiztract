"""Tests for Stage 4 — Validation."""

import pytest

from models.schema import (
    ExtractedDocument,
    ExtractedQuestionGroup,
    ExtractedSAData,
    ExtractedComprehensionData,
    ExtractedBlock,
    ExtractedOption,
)
from pipeline.validate import (
    validate_document,
    _check_latex_balance,
    _parse_matrix_dimensions,
    _find_dimension_references,
)


@pytest.fixture
def default_config():
    return {
        "validate": {
            "warn_on_missing_correct_option": True,
            "warn_on_empty_sub_questions": True,
            "latex_balance_check": True,
            "matrix_dimension_check": True,
            "min_mcq_options": 2,
            "max_mcq_options": 6,
        }
    }


class TestLatexBalance:
    def test_balanced(self):
        assert _check_latex_balance(r"\frac{a}{b}") == []

    def test_unbalanced_brace(self):
        issues = _check_latex_balance(r"\frac{a}{b")
        assert len(issues) > 0

    def test_unbalanced_left_right(self):
        issues = _check_latex_balance(r"\left( x \right)")
        assert len(issues) == 0

    def test_unbalanced_env(self):
        issues = _check_latex_balance(r"\begin{matrix} 1 & 2")
        assert len(issues) > 0


class TestMatrixDimensions:
    def test_2x2_bmatrix(self):
        latex = r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}"
        result = _parse_matrix_dimensions(latex)
        assert result == (2, 2)

    def test_3x3(self):
        latex = r"\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}"
        result = _parse_matrix_dimensions(latex)
        assert result == (3, 3)

    def test_no_matrix(self):
        assert _parse_matrix_dimensions(r"\frac{a}{b}") is None


class TestDimensionReferences:
    def test_find_3x3(self):
        text = "Consider the 3x3 matrix A"
        refs = _find_dimension_references(text)
        assert (3, 3) in refs

    def test_find_none(self):
        refs = _find_dimension_references("No dimensions here")
        assert len(refs) == 0


class TestValidateDocument:
    def test_empty_question_text(self, default_config):
        group = ExtractedQuestionGroup(
            question_id="Q1",
            question_type="MCQ",
            question_text="",
            options=[
                ExtractedOption(option_id="A", option_text="Yes"),
                ExtractedOption(option_id="B", option_text="No"),
            ],
        )
        doc = ExtractedDocument(
            exam_code="TEST",
            question_groups=[group],
        )
        result = validate_document(doc, default_config)
        assert any("question_text is empty" in e for e in result.validation_errors)

    def test_answer_range_invalid(self, default_config):
        group = ExtractedQuestionGroup(
            question_id="Q2",
            question_type="SA",
            question_text="Compute the value",
            sa_data=ExtractedSAData(
                response_type="Numeric",
                answers_type="Range",
                answer_range_min=10.0,
                answer_range_max=5.0,  # min > max
            ),
        )
        doc = ExtractedDocument(
            exam_code="TEST",
            question_groups=[group],
        )
        result = validate_document(doc, default_config)
        assert result.question_groups[0].needs_review is True

    def test_valid_mcq(self, default_config):
        group = ExtractedQuestionGroup(
            question_id="Q3",
            question_type="MCQ",
            question_text="What is 2+2?",
            correct_marks=2.0,
            options=[
                ExtractedOption(option_id="A", option_text="3"),
                ExtractedOption(option_id="B", option_text="4", is_correct=True),
                ExtractedOption(option_id="C", option_text="5"),
                ExtractedOption(option_id="D", option_text="6"),
            ],
            correct_option_id="B",
        )
        doc = ExtractedDocument(
            exam_code="TEST",
            question_groups=[group],
        )
        result = validate_document(doc, default_config)
        assert len(result.validation_errors) == 0
