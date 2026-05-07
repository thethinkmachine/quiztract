#!/usr/bin/env python3
"""Quiztract — CBT Question Paper PDF → Structured JSON Pipeline.

CLI entry point for running the extraction pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import click
import yaml
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

# Load environment variables
load_dotenv()

console = Console()


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "pipeline.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning("Config file not found: {} — using defaults", config_path)
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _setup_logging(debug: bool) -> None:
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


def _load_vlm(config: dict[str, Any]) -> Any:
    """Load the VLM model for extraction. Returns a callable(image, prompt) -> str."""
    extract_cfg = config.get("extract", {})
    model_id = extract_cfg.get("vlm_model_id") or os.getenv("VLM_MODEL_ID", "ibm-granite/granite-vision-4.1-4b")
    device = extract_cfg.get("vlm_device") or os.getenv("VLM_DEVICE", "cpu")
    max_tokens = extract_cfg.get("vlm_max_new_tokens") or int(os.getenv("VLM_MAX_NEW_TOKENS", "1024"))

    console.print(f"[bold blue]Loading VLM:[/] {model_id} on {device}")

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        def vlm_fn(image, prompt: str) -> str:
            """Run VLM inference on an image with a text prompt."""
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_tokens)
            result = processor.decode(outputs[0], skip_special_tokens=True)
            # Strip the prompt from the output if echoed
            if result.startswith(prompt):
                result = result[len(prompt):].strip()
            return result

        console.print("[bold green]✓ VLM loaded successfully[/]")
        return vlm_fn

    except Exception as exc:
        console.print(f"[bold yellow]⚠ VLM loading failed: {exc}[/]")
        console.print("[dim]Pipeline will run without VLM — raster-only fallback for non-text blocks[/]")
        return None


def _run_pipeline(
    pdf_path: Path,
    exam_code: str,
    config: dict[str, Any],
    vlm_fn: Any | None,
    skip_render: bool,
    skip_validation: bool,
    debug: bool,
) -> None:
    """Run the full pipeline on a single PDF."""
    from pipeline.ingest import ingest_pdf
    from pipeline.classify import classify_document
    from pipeline.extract import extract_content
    from pipeline.stitch import stitch_pages
    from pipeline.validate import validate_document
    from pipeline.assemble import assemble_json
    from pipeline.render import render_outputs

    output_cfg = config.get("output", {})
    base_dir = Path(output_cfg.get("base_dir", "./output"))
    output_dir = base_dir / exam_code

    console.print(Panel(
        f"[bold]PDF:[/] {pdf_path.name}\n[bold]Exam Code:[/] {exam_code}\n[bold]Output:[/] {output_dir}",
        title="Quiztract Pipeline",
        border_style="blue",
    ))

    # Stage 1: Ingest
    console.print("\n[bold cyan]Stage 1/6:[/] PDF Ingestion & Segmentation")
    pages = ingest_pdf(pdf_path, config)
    console.print(f"  → {len(pages)} pages ingested")

    # Stage 2: Classify
    console.print("\n[bold cyan]Stage 2/6:[/] Block Classification")
    classified = classify_document(pages, exam_code, config, vlm_fn=vlm_fn)
    console.print(f"  → {len(classified.blocks)} blocks, {len(classified.question_groups)} question groups")

    # Stage 3: Extract
    console.print("\n[bold cyan]Stage 3/6:[/] Content Extraction")
    extracted = extract_content(classified, config, vlm_fn=vlm_fn, output_dir=output_dir)
    stats = extracted.extraction_stats
    console.print(f"  → {stats.get('vlm_transcription_count', 0)} VLM transcriptions, {stats.get('raster_fallback_count', 0)} raster fallbacks")

    # Stitching
    console.print("\n[dim]  Running page boundary stitching...[/]")
    extracted = stitch_pages(extracted, config)

    # Stage 4: Validate
    if not skip_validation:
        console.print("\n[bold cyan]Stage 4/6:[/] Validation")
        validated = validate_document(extracted, config)
        console.print(f"  → {len(validated.validation_errors)} errors, {len(validated.validation_warnings)} warnings")
    else:
        console.print("\n[bold yellow]Stage 4/6:[/] Validation [dim](skipped)[/]")
        from models.schema import ValidatedDocument
        validated = ValidatedDocument(
            exam_code=extracted.exam_code,
            question_groups=extracted.question_groups,
            extraction_stats=extracted.extraction_stats,
        )

    # Stage 5: Assemble
    console.print("\n[bold cyan]Stage 5/6:[/] JSON Assembly")
    json_path = assemble_json(validated, config, pdf_path, output_dir=output_dir)
    console.print(f"  → {json_path}")

    # Stage 6: Render
    if not skip_render:
        console.print("\n[bold cyan]Stage 6/6:[/] Rendering MD & HTML")
        md_path, html_path = render_outputs(json_path, config, output_dir=output_dir)
        console.print(f"  → {md_path}")
        console.print(f"  → {html_path}")
    else:
        console.print("\n[bold yellow]Stage 6/6:[/] Rendering [dim](skipped)[/]")

    # Summary
    review_count = sum(1 for g in validated.question_groups if g.needs_review)
    console.print(Panel(
        f"[bold green]✓ Pipeline complete[/]\n"
        f"Questions: {len(validated.question_groups)}\n"
        f"Needing review: {review_count}\n"
        f"Output: {output_dir}",
        title="Done",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True), help="Path to a single PDF file")
@click.option("--input-dir", type=click.Path(exists=True), help="Path to directory of PDFs (use with --bulk)")
@click.option("--bulk", is_flag=True, help="Process all PDFs in input-dir")
@click.option("--exam-code", type=str, help="Override exam code (single file mode only)")
@click.option("--skip-render", is_flag=True, help="Skip Stage 6, produce JSON only")
@click.option("--skip-validation", is_flag=True, help="Skip Stage 4 (not recommended)")
@click.option("--debug", is_flag=True, help="Save intermediate representations and page rasters")
@click.option("--review-only", is_flag=True, help="Re-render MD/HTML for flagged questions")
@click.option("--config", "config_path", type=click.Path(), help="Path to alternate pipeline.yaml")
def main(
    input_path: Optional[str],
    input_dir: Optional[str],
    bulk: bool,
    exam_code: Optional[str],
    skip_render: bool,
    skip_validation: bool,
    debug: bool,
    review_only: bool,
    config_path: Optional[str],
) -> None:
    """Quiztract — Convert CBT question paper PDFs to structured JSON."""
    _setup_logging(debug)
    config = _load_config(config_path)

    if debug:
        config.setdefault("ingest", {})["debug_save_page_rasters"] = True

    console.print("[bold]Quiztract[/] v1.0.0\n", style="blue")

    # Load VLM
    vlm_fn = _load_vlm(config)

    if bulk and input_dir:
        # Bulk mode
        pdf_dir = Path(input_dir)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            console.print(f"[red]No PDFs found in {pdf_dir}[/]")
            sys.exit(1)

        console.print(f"[bold]Bulk processing {len(pdfs)} PDFs[/]\n")
        for pdf in pdfs:
            code = exam_code or pdf.stem
            try:
                _run_pipeline(pdf, code, config, vlm_fn, skip_render, skip_validation, debug)
            except Exception as exc:
                console.print(f"[red]Failed: {pdf.name} — {exc}[/]")
                if debug:
                    import traceback
                    traceback.print_exc()

    elif input_path:
        # Single file mode
        pdf = Path(input_path)
        code = exam_code or pdf.stem
        _run_pipeline(pdf, code, config, vlm_fn, skip_render, skip_validation, debug)

    else:
        console.print("[red]Error: Provide --input or --input-dir with --bulk[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
