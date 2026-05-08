"""Quiztract pipeline stages.

Stage 1: ingest  — PDF ingestion & page segmentation
Stage 2: process — VLM block classification & extraction
Stage 3: render  — Markdown & HTML rendering
"""

from pipeline.ingest import ingest_pdf
from pipeline.process import process_document
from pipeline.render import render_outputs

__all__ = [
    "ingest_pdf",
    "process_document",
    "render_outputs",
]
