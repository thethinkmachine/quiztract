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
    max_tokens = extract_cfg.get("vlm_max_new_tokens") or int(os.getenv("VLM_MAX_NEW_TOKENS", "4096"))
    hf_cache_dir_value = extract_cfg.get("hf_cache_dir") or os.getenv("HF_HUB_CACHE") or os.getenv("HF_HOME")
    hf_cache_dir = Path(hf_cache_dir_value).expanduser() if hf_cache_dir_value else None
    if hf_cache_dir is not None:
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

    offload_dir_value = extract_cfg.get("vlm_offload_dir") or os.getenv(
        "VLM_OFFLOAD_DIR",
        str(Path.home() / ".cache" / "quiztract" / "vlm-offload"),
    )
    offload_dir = Path(offload_dir_value).expanduser()
    offload_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Loading VLM:[/] {model_id} on {device}")

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        def _load_components(trust_remote_code: bool) -> tuple[Any, Any]:
            pretrained_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
            if hf_cache_dir is not None:
                pretrained_kwargs["cache_dir"] = str(hf_cache_dir)

            processor = AutoProcessor.from_pretrained(
                model_id,
                **pretrained_kwargs,
            )
            if getattr(processor, "tokenizer", None) is not None:
                processor.tokenizer.padding_side = "left"  # Required for correct generation

            if device == "cpu":
                dtype = torch.float32
                model_kwargs: dict[str, Any] = {
                    "dtype": dtype,
                    "device_map": None,
                    "low_cpu_mem_usage": True,
                }
            else:
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                gpu_total_gib = 0
                if torch.cuda.is_available():
                    gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                default_gpu_budget_gib = max(2, int(gpu_total_gib * 0.75) if gpu_total_gib else 4)
                configured_gpu_budget = extract_cfg.get("vlm_gpu_max_memory_gb")
                gpu_budget_gib = int(
                    configured_gpu_budget
                    if configured_gpu_budget is not None
                    else os.getenv("VLM_GPU_MAX_MEMORY_GB", str(default_gpu_budget_gib))
                )
                configured_cpu_budget = extract_cfg.get("vlm_cpu_max_memory_gb")
                cpu_budget_gib = int(
                    configured_cpu_budget
                    if configured_cpu_budget is not None
                    else os.getenv("VLM_CPU_MAX_MEMORY_GB", "32")
                )
                model_kwargs = {
                    "dtype": dtype,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "max_memory": {0: f"{gpu_budget_gib}GiB", "cpu": f"{cpu_budget_gib}GiB"},
                    "offload_folder": str(offload_dir),
                    "offload_state_dict": True,
                }

                use_4bit = extract_cfg.get("vlm_4bit_quantization", False)
                if use_4bit:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )

            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **pretrained_kwargs,
                **model_kwargs,
            ).eval()
            return processor, model

        last_error: Exception | None = None
        for trust_remote_code in (False, True):
            try:
                processor, model = _load_components(trust_remote_code)

                def vlm_fn(image, prompt: str) -> str:
                    """Run VLM inference using official Granite Vision chat template style."""
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            use_cache=True,
                        )

                    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
                    result = processor.decode(generated_ids, skip_special_tokens=True)

                    return result.strip()

                if trust_remote_code:
                    console.print("[dim]Granite Vision loaded via trust_remote_code compatibility path[/]")
                console.print("[bold green]✓ VLM loaded successfully[/]")
                return vlm_fn
            except Exception as exc:
                last_error = exc
                if not trust_remote_code:
                    console.print("[yellow]Native Granite Vision load failed; retrying with trust_remote_code=True[/]")
                    continue
                break

        if last_error is not None:
            console.print(f"[bold yellow]⚠ VLM loading failed: {last_error}[/]")
        return None

    except ImportError as e:
        error_text = str(e)
        if "AutoModelForImageTextToText" in error_text or "AutoModelForVision2Seq" in error_text:
            console.print("[bold red]⚠ Error: Granite Vision requires a recent 'transformers' install.[/]")
            console.print("[yellow]Please run: pip install -r requirements.txt[/]")
        else:
            console.print(f"[bold yellow]⚠ ImportError: {e}[/]")
        return None


def _run_pipeline(
    pdf_path: Path,
    exam_code: str,
    config: dict[str, Any],
    vlm_fn: Any | None,
    skip_render: bool,
    debug: bool,
) -> None:
    """Run the full pipeline on a single PDF."""
    from pipeline.ingest import ingest_pdf
    from pipeline.process import process_document
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
    console.print("\n[bold cyan]Stage 1/3:[/] PDF Ingestion & Segmentation")
    pages = ingest_pdf(pdf_path, config)
    console.print(f"  → {len(pages)} pages ingested")

    # Stage 2: Process
    console.print("\n[bold cyan]Stage 2/3:[/] VLM Processing")
    extracted_json = process_document(pages, config, vlm_fn, output_dir)
    import json
    with open(output_dir / f"{exam_code}.json", "w") as f:
        json.dump(extracted_json, f, indent=2)
    console.print(f"  → Saved to {output_dir / f'{exam_code}.json'}")
    
    # Stage 3: Render
    if not skip_render:
        console.print("\n[bold cyan]Stage 3/3:[/] Rendering MD & HTML")
        md_path, html_path = render_outputs(output_dir / f"{exam_code}.json", config, output_dir=output_dir)
        console.print(f"  → {md_path}")
        console.print(f"  → {html_path}")
    else:
        console.print("\n[bold yellow]Stage 3/3:[/] Rendering [dim](skipped)[/]")

    console.print(Panel(f"[bold green]✓ Processing complete[/]\nQuestions Extracted: {len(extracted_json.get('questions', []))}\nOutput: {output_dir}", title="Done", border_style="green"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True), help="Path to a single PDF file")
@click.option("--input-dir", type=click.Path(exists=True), help="Path to directory of PDFs (use with --bulk)")
@click.option("--bulk", is_flag=True, help="Process all PDFs in input-dir")
@click.option("--exam-code", type=str, help="Override exam code (single file mode only)")
@click.option("--skip-render", is_flag=True, help="Skip Stage 3, produce JSON only")
@click.option("--debug", is_flag=True, help="Save intermediate representations and page rasters")
@click.option("--review-only", is_flag=True, help="Re-render MD/HTML for flagged questions")
@click.option("--config", "config_path", type=click.Path(), help="Path to alternate pipeline.yaml")
def main(
    input_path: Optional[str],
    input_dir: Optional[str],
    bulk: bool,
    exam_code: Optional[str],
    skip_render: bool,
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
                _run_pipeline(pdf, code, config, vlm_fn, skip_render, debug)
            except Exception as exc:
                console.print(f"[red]Failed: {pdf.name} — {exc}[/]")
                if debug:
                    import traceback
                    traceback.print_exc()

    elif input_path:
        # Single file mode
        pdf = Path(input_path)
        code = exam_code or pdf.stem
        _run_pipeline(pdf, code, config, vlm_fn, skip_render, debug)

    else:
        console.print("[red]Error: Provide --input or --input-dir with --bulk[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
