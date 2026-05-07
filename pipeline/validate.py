"""Stage 4 — Validation.

Runs all validation checks on the ExtractedDocument:
- Matrix dimension checks
- Answer range consistency
- Required field presence
- Confidence thresholding
- LaTeX balance checks
- Option count sanity

Sets needs_review flags and review_reason strings.
Does NOT modify or discard content — only annotates.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from models.schema import (
    ExtractedDocument,
    ExtractedQuestionGroup,
    ValidatedDocument,
)


def validate_document(
    doc: ExtractedDocument,
    config: dict[str, Any],
) -> ValidatedDocument:
    """Run all validation checks on an ExtractedDocument.

    Args:
        doc: ExtractedDocument from Stage 3 (post-stitching).
        config: Pipeline configuration dict.

    Returns:
        ValidatedDocument with review flags populated throughout.
    """
    validate_cfg = config.get("validate", {})
    warn_missing_correct = validate_cfg.get("warn_on_missing_correct_option", True)
    warn_empty_sub_qs = validate_cfg.get("warn_on_empty_sub_questions", True)
    do_latex_check = validate_cfg.get("latex_balance_check", True)
    do_matrix_check = validate_cfg.get("matrix_dimension_check", True)
    min_mcq_opts = validate_cfg.get("min_mcq_options", 2)
    max_mcq_opts = validate_cfg.get("max_mcq_options", 6)

    logger.info("Validating document: {}", doc.exam_code)

    errors: list[str] = []
    warnings: list[str] = []

    for group in doc.question_groups:
        # 4.1 Matrix dimension validation
        if do_matrix_check:
            _validate_matrix_dimensions(group, errors, warnings)

        # 4.2 Answer range validation
        _validate_answer_range(group, errors, warnings)

        # 4.3 Required field presence
        _validate_required_fields(
            group, errors, warnings,
            warn_missing_correct=warn_missing_correct,
            warn_empty_sub_qs=warn_empty_sub_qs,
        )

        # 4.4 Confidence thresholding
        _validate_confidence(group, errors, warnings)

        # 4.5 LaTeX balance check
        if do_latex_check:
            _validate_latex(group, errors, warnings)

        # 4.6 Option count sanity
        _validate_option_count(group, errors, warnings, min_mcq_opts, max_mcq_opts)

    logger.info(
        "Validation complete: {} errors, {} warnings",
        len(errors),
        len(warnings),
    )

    return ValidatedDocument(
        exam_code=doc.exam_code,
        question_groups=doc.question_groups,
        extraction_stats=doc.extraction_stats,
        validation_errors=errors,
        validation_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Validation check implementations
# ---------------------------------------------------------------------------


def _validate_matrix_dimensions(
    group: ExtractedQuestionGroup,
    errors: list[str],
    warnings: list[str],
) -> None:
    """4.1 — Validate matrix/vector dimensions for consistency."""
    for block in group.blocks:
        if block.block_type not in ("matrix", "math_block"):
            continue
        if not block.latex:
            continue

        # Parse matrix dimensions from LaTeX
        dims = _parse_matrix_dimensions(block.latex)
        if dims is None:
            continue

        rows, cols = dims

        # Check if the question stem references specific dimensions
        stem = group.question_text or ""
        stem_dims = _find_dimension_references(stem)

        for ref_rows, ref_cols in stem_dims:
            if (ref_rows, ref_cols) != (rows, cols):
                reason = (
                    f"Matrix dimension mismatch: extracted {rows}x{cols}, "
                    f"usage implies {ref_rows}x{ref_cols}"
                )
                group.needs_review = True
                group.review_reason = reason
                warnings.append(f"Q{group.question_id}: {reason}")


def _validate_answer_range(
    group: ExtractedQuestionGroup,
    errors: list[str],
    warnings: list[str],
) -> None:
    """4.2 — Validate SA answer range consistency."""
    if group.sa_data is None:
        return

    if group.sa_data.answers_type and group.sa_data.answers_type.lower() == "range":
        rmin = group.sa_data.answer_range_min
        rmax = group.sa_data.answer_range_max

        if rmin is None or rmax is None:
            reason = "Answer range min or max is missing"
            group.needs_review = True
            group.review_reason = reason
            warnings.append(f"Q{group.question_id}: {reason}")
            return

        if rmin >= rmax:
            reason = f"Answer range min >= max: {rmin} >= {rmax}"
            group.needs_review = True
            group.review_reason = reason
            errors.append(f"Q{group.question_id}: {reason}")

        # Check numeric consistency with response type
        if group.sa_data.response_type and group.sa_data.response_type.lower() == "numeric":
            try:
                float(rmin)
                float(rmax)
            except (ValueError, TypeError):
                reason = "Answer range values are not numeric but response_type is Numeric"
                group.needs_review = True
                group.review_reason = reason
                errors.append(f"Q{group.question_id}: {reason}")


def _validate_required_fields(
    group: ExtractedQuestionGroup,
    errors: list[str],
    warnings: list[str],
    warn_missing_correct: bool,
    warn_empty_sub_qs: bool,
) -> None:
    """4.3 — Validate required field presence."""
    qid = group.question_id or "unknown"

    # question_text is always required
    if not (group.question_text or "").strip():
        reason = "Required field question_text is empty"
        group.needs_review = True
        group.review_reason = reason
        errors.append(f"Q{qid}: {reason}")

    # correct_marks is always required
    if group.correct_marks is None:
        reason = "Required field correct_marks is null"
        group.needs_review = True
        group.review_reason = reason
        warnings.append(f"Q{qid}: {reason}")

    qtype = (group.question_type or "").upper()

    if qtype == "MCQ":
        # Must have >= 2 options
        if len(group.options) < 2:
            reason = f"MCQ has fewer than 2 options ({len(group.options)} found)"
            group.needs_review = True
            group.review_reason = reason
            errors.append(f"Q{qid}: {reason}")

        # Warn if correct_option_id is missing
        if warn_missing_correct and not group.correct_option_id:
            warnings.append(f"Q{qid}: correct_option_id is null for MCQ")

    elif qtype == "MSQ":
        if len(group.options) < 2:
            reason = f"MSQ has fewer than 2 options ({len(group.options)} found)"
            group.needs_review = True
            group.review_reason = reason
            errors.append(f"Q{qid}: {reason}")

        if warn_missing_correct and not group.correct_option_ids:
            warnings.append(f"Q{qid}: correct_option_ids is empty for MSQ")

    elif qtype == "SA":
        if group.sa_data and not group.sa_data.possible_answers:
            reason = "Possible answers field empty for SA question"
            if group.sa_data.answer_truncated_across_page:
                reason = "Possible answers field empty after page boundary stitching"
            warnings.append(f"Q{qid}: {reason}")

    elif qtype == "COMPREHENSION":
        if warn_empty_sub_qs and group.comprehension_data:
            if not group.comprehension_data.sub_question_ids:
                reason = "Sub-questions list empty for COMPREHENSION block"
                group.needs_review = True
                group.review_reason = reason
                warnings.append(f"Q{qid}: {reason}")


def _validate_confidence(
    group: ExtractedQuestionGroup,
    errors: list[str],
    warnings: list[str],
) -> None:
    """4.4 — Flag low-confidence extractions for review."""
    for block in group.blocks:
        if block.extraction_confidence == "low":
            group.needs_review = True
            if not group.review_reason:
                group.review_reason = "Low confidence extraction detected"
            warnings.append(
                f"Q{group.question_id}: Low confidence {block.block_type} block"
            )


def _validate_latex(
    group: ExtractedQuestionGroup,
    errors: list[str],
    warnings: list[str],
) -> None:
    """4.5 — Validate LaTeX balance for all extracted LaTeX strings."""
    latex_strings: list[tuple[str, str]] = []  # (source, latex)

    if group.question_text_latex:
        latex_strings.append(("question_text_latex", group.question_text_latex))

    for block in group.blocks:
        if block.latex:
            latex_strings.append((f"block_{block.block_type}", block.latex))

    for opt in group.options:
        if opt.option_text_latex:
            latex_strings.append((f"option_{opt.option_id}", opt.option_text_latex))

    for source, latex in latex_strings:
        issues = _check_latex_balance(latex)
        if issues:
            reason = f"LaTeX string has unbalanced braces in {source}: {'; '.join(issues)}"
            group.needs_review = True
            group.review_reason = reason

            # Downgrade confidence to low
            group.question_text_confidence = "low"

            warnings.append(f"Q{group.question_id}: {reason}")


def _validate_option_count(
    group: ExtractedQuestionGroup,
    errors: list[str],
    warnings: list[str],
    min_opts: int,
    max_opts: int,
) -> None:
    """4.6 — Validate option count for MCQ/MSQ."""
    qtype = (group.question_type or "").upper()
    n = len(group.options)

    if qtype == "MCQ":
        if n < min_opts or n > max_opts:
            reason = f"Option count outside expected range: {n} options detected (expected {min_opts}-{max_opts})"
            group.needs_review = True
            group.review_reason = reason
            warnings.append(f"Q{group.question_id}: {reason}")

    elif qtype == "MSQ":
        if n < min_opts:
            reason = f"Option count outside expected range: {n} options detected (expected >= {min_opts})"
            group.needs_review = True
            group.review_reason = reason
            warnings.append(f"Q{group.question_id}: {reason}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_matrix_dimensions(latex: str) -> tuple[int, int] | None:
    """Parse matrix dimensions from LaTeX string."""
    # Look for \begin{...matrix} environments
    env_match = re.search(
        r"\\begin\{[a-zA-Z]*matrix\}(.*?)\\end\{[a-zA-Z]*matrix\}",
        latex,
        re.DOTALL,
    )
    if not env_match:
        # Try \begin{array}
        env_match = re.search(
            r"\\begin\{array\}.*?\}(.*?)\\end\{array\}",
            latex,
            re.DOTALL,
        )

    if not env_match:
        return None

    content = env_match.group(1).strip()

    # Count rows (separated by \\)
    rows = [r.strip() for r in re.split(r"\\\\", content) if r.strip()]
    if not rows:
        return None

    # Count cols in first row (separated by &)
    cols = len(rows[0].split("&"))

    return (len(rows), cols)


def _find_dimension_references(text: str) -> list[tuple[int, int]]:
    """Find matrix dimension references in text (e.g., '3x3 matrix', '3 × 3')."""
    dims = []

    patterns = [
        r"(\d+)\s*[x×]\s*(\d+)\s*(?:matrix|matrices|vector|array)",
        r"(\d+)\s*[x×]\s*(\d+)",
        r"order\s+(\d+)\s*[x×]\s*(\d+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                r = int(match.group(1))
                c = int(match.group(2))
                dims.append((r, c))
            except ValueError:
                pass

    return dims


def _check_latex_balance(latex: str) -> list[str]:
    """Check LaTeX string for balance issues. Returns list of issue descriptions."""
    issues: list[str] = []

    # Check balanced braces
    depth = 0
    for ch in latex:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            issues.append("Unmatched closing brace '}'")
            break
    if depth > 0:
        issues.append(f"Unmatched opening braces: {depth} unclosed")

    # Check balanced \left / \right
    lefts = len(re.findall(r"\\left[\(\[\{|.]", latex))
    rights = len(re.findall(r"\\right[\)\]\}|.]", latex))
    if lefts != rights:
        issues.append(f"Unbalanced \\left/\\right: {lefts} left, {rights} right")

    # Check balanced \begin / \end
    begins = re.findall(r"\\begin\{(\w+)\}", latex)
    ends = re.findall(r"\\end\{(\w+)\}", latex)
    if begins != ends:
        issues.append(f"Unbalanced \\begin/\\end environments")

    return issues
