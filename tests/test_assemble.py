"""Tests for Stage 5 — JSON Assembly."""

import json
import pytest
from pathlib import Path

from models.schema import (
    ValidatedDocument,
    ExtractedQuestionGroup,
    ExtractedOption,
    ExtractedSAData,
    ExamOutput,
    ExamMeta,
    QuestionItem,
    MCQData,
    OptionItem,
    ExtractionMetadata,
)
from pipeline.assemble import (
    assemble_json,
    _apply_null_policy,
    _build_questions,
    _group_to_question,
)


@pytest.fixture
def default_config():
    return {
        "assemble": {
            "json_indent": 2,
            "ensure_ascii": False,
            "run_schema_validation": True,
        },
        "output": {
            "base_dir": "./output",
        },
    }


class TestNullPolicy:
    def test_remove_none_type_specific(self):
        data = {
            "question_id": "Q1",
            "mcq": {"options": []},
            "msq": None,
            "sa": None,
            "comprehension": None,
        }
        result = _apply_null_policy(data)
        assert "mcq" in result
        assert "msq" not in result
        assert "sa" not in result
        assert "comprehension" not in result

    def test_keep_null_applicable_field(self):
        data = {"question_text": None, "topic": None}
        result = _apply_null_policy(data)
        assert result["question_text"] is None
        assert result["topic"] is None

    def test_empty_array_stays(self):
        data = {"figures": [], "options": []}
        result = _apply_null_policy(data)
        assert result["figures"] == []
        assert result["options"] == []


class TestGroupToQuestion:
    def test_mcq_conversion(self):
        group = ExtractedQuestionGroup(
            question_id="Q1",
            question_type="MCQ",
            question_text="What is 2+2?",
            correct_marks=2.0,
            negative_marks=0.67,
            options=[
                ExtractedOption(option_id="A", option_text="3"),
                ExtractedOption(option_id="B", option_text="4", is_correct=True),
            ],
            correct_option_id="B",
        )
        q = _group_to_question(group)
        assert q is not None
        assert q.question_type == "MCQ"
        assert q.mcq is not None
        assert q.mcq.correct_option_id == "B"
        assert len(q.mcq.options) == 2
        assert q.sa is None
        assert q.comprehension is None

    def test_sa_conversion(self):
        group = ExtractedQuestionGroup(
            question_id="Q2",
            question_type="SA",
            question_text="Enter value",
            sa_data=ExtractedSAData(
                response_type="Numeric",
                answers_type="Equal",
                possible_answers=["42"],
            ),
        )
        q = _group_to_question(group)
        assert q is not None
        assert q.sa is not None
        assert q.sa.response_type == "Numeric"
        assert q.mcq is None


class TestAssembleJson:
    def test_write_json(self, default_config, tmp_path):
        group = ExtractedQuestionGroup(
            question_id="Q1",
            question_type="MCQ",
            question_text="Test question",
            correct_marks=1.0,
            options=[
                ExtractedOption(option_id="A", option_text="Yes"),
                ExtractedOption(option_id="B", option_text="No"),
            ],
        )
        doc = ValidatedDocument(
            exam_code="TEST",
            question_groups=[group],
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        output_dir = tmp_path / "output" / "TEST"
        json_path = assemble_json(doc, default_config, pdf_path, output_dir=output_dir)

        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "exam" in data
        assert "questions" in data
        assert len(data["questions"]) == 1

        # Provenance should also exist
        prov_path = output_dir / "TEST.provenance.json"
        assert prov_path.exists()
