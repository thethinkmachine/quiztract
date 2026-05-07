"""Quiztract pipeline stages.

Stage 1: ingest   — PDF ingestion & page segmentation
Stage 2: classify — Block classification & boundary detection
Stage 3: extract  — Content extraction per block type
Stage 4: validate — Validation & review flagging
Stage 5: assemble — JSON assembly & provenance
Stage 6: render   — Markdown & HTML rendering
"""

from pipeline.ingest import ingest_pdf
from pipeline.classify import classify_document
from pipeline.extract import extract_content
from pipeline.stitch import stitch_pages
from pipeline.validate import validate_document
from pipeline.assemble import assemble_json
from pipeline.render import render_outputs

__all__ = [
    "ingest_pdf",
    "classify_document",
    "extract_content",
    "stitch_pages",
    "validate_document",
    "assemble_json",
    "render_outputs",
]
