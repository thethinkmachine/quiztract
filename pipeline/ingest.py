"""Stage 1 — PDF Ingestion & Segmentation.

Accepts a PDF file path, parses it with docling, renders every page as a
raster image at the configured DPI, extracts the raw text layer, and returns
a list of PageDocument objects — one per page.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image
from tqdm import tqdm

from models.schema import PageDocument


# ---------------------------------------------------------------------------
# Timeout helper for per-page processing
# ---------------------------------------------------------------------------

class _PageTimeout(Exception):
    """Raised when a single page exceeds the configured timeout."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _PageTimeout("Page processing timed out")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_pdf(
    pdf_path: str | Path,
    config: dict[str, Any],
) -> list[PageDocument]:
    """Ingest a PDF and return a list of PageDocument objects.

    Args:
        pdf_path: Path to the input PDF file.
        config: Pipeline configuration dict (parsed from pipeline.yaml).

    Returns:
        List of PageDocument, one per successfully parsed page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If zero pages could be parsed and fail_on_parse_error is True.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ingest_cfg = config.get("ingest", {})
    dpi = ingest_cfg.get("dpi", 300)
    page_timeout = ingest_cfg.get("page_timeout_seconds", 30)
    fail_on_error = ingest_cfg.get("fail_on_parse_error", False)
    debug_rasters = ingest_cfg.get("debug_save_page_rasters", False)

    logger.info("Ingesting PDF: {}", pdf_path.name)
    logger.info("DPI: {} | Timeout per page: {}s", dpi, page_timeout)

    # ------------------------------------------------------------------
    # Step 1: Render pages as raster images using pdf2image
    # ------------------------------------------------------------------
    page_images = _render_pages(pdf_path, dpi)
    total_pages = len(page_images)
    logger.info("Rendered {} page(s) at {} DPI", total_pages, dpi)

    # ------------------------------------------------------------------
    # Step 2: Parse document structure with docling
    # ------------------------------------------------------------------
    docling_result = _parse_with_docling(pdf_path)

    # ------------------------------------------------------------------
    # Step 3: Build PageDocument list
    # ------------------------------------------------------------------
    pages: list[PageDocument] = []
    failed_pages: list[int] = []

    for page_num in tqdm(range(1, total_pages + 1), desc="Processing pages"):
        try:
            # Apply per-page timeout (Unix only; on Windows this is a no-op)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(page_timeout)

            page_doc = _build_page_document(
                page_num=page_num,
                raster=page_images[page_num - 1],
                docling_result=docling_result,
            )
            pages.append(page_doc)

        except _PageTimeout:
            logger.warning("Page {} timed out after {}s — marking as failed", page_num, page_timeout)
            pages.append(_failed_page(page_num, page_images[page_num - 1]))
            failed_pages.append(page_num)

        except Exception as exc:
            logger.error("Page {} failed: {} — marking as failed", page_num, exc)
            pages.append(_failed_page(page_num, page_images[page_num - 1]))
            failed_pages.append(page_num)

        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    if failed_pages:
        logger.warning("Failed pages: {}", failed_pages)

    if not pages or all(p.parse_failed for p in pages):
        msg = f"Zero pages successfully parsed from {pdf_path.name}"
        if fail_on_error:
            raise RuntimeError(msg)
        logger.error(msg)

    # Optionally save debug rasters
    if debug_rasters:
        _save_debug_rasters(pages, pdf_path)

    logger.info(
        "Ingestion complete: {}/{} pages OK, {} failed",
        len(pages) - len(failed_pages),
        total_pages,
        len(failed_pages),
    )
    return pages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Render all pages of a PDF as PIL Images using pdf2image."""
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")
        return images
    except ImportError:
        logger.warning("pdf2image not installed — falling back to pymupdf for rasterisation")
        return _render_pages_pymupdf(pdf_path, dpi)


def _render_pages_pymupdf(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Fallback: render pages using PyMuPDF (fitz)."""
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def _parse_with_docling(pdf_path: Path) -> Any:
    """Parse PDF with docling and return the result object.

    Returns None if docling is not available or fails.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        logger.info("Docling parse successful")
        return result
    except ImportError:
        logger.warning(
            "docling not installed — text extraction will rely on pymupdf fallback"
        )
        return None
    except Exception as exc:
        logger.error("Docling parse failed: {} — continuing with raster only", exc)
        return None


def _extract_text_pymupdf(pdf_path: Path, page_num: int) -> str:
    """Extract text from a single page using PyMuPDF as fallback."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        if page_num - 1 < len(doc):
            text = doc[page_num - 1].get_text()
            doc.close()
            return text
        doc.close()
    except Exception as exc:
        logger.debug("PyMuPDF text extraction failed for page {}: {}", page_num, exc)
    return ""


def _extract_page_text(page_num: int, docling_result: Any) -> str:
    """Extract the raw text layer for a specific page from the docling result."""
    if docling_result is None:
        return ""

    try:
        # Docling v2 API: iterate document elements and collect text by page
        doc = docling_result.document
        page_texts: list[str] = []

        for element in doc.iterate_items():
            item = element
            # Handle tuple returns from iterate_items
            if isinstance(element, tuple):
                item = element[1] if len(element) > 1 else element[0]

            # Check if this element belongs to our page
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                item_page = getattr(prov[0], "page_no", None) or getattr(prov[0], "page", None)
                if item_page == page_num:
                    text = getattr(item, "text", None)
                    if text:
                        page_texts.append(text)

        return "\n".join(page_texts)

    except Exception as exc:
        logger.debug("Failed to extract text from docling for page {}: {}", page_num, exc)
        return ""


def _extract_page_blocks(page_num: int, docling_result: Any) -> list[Any]:
    """Extract docling block objects for a specific page."""
    if docling_result is None:
        return []

    try:
        doc = docling_result.document
        blocks = []

        for element in doc.iterate_items():
            item = element
            if isinstance(element, tuple):
                item = element[1] if len(element) > 1 else element[0]

            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                item_page = getattr(prov[0], "page_no", None) or getattr(prov[0], "page", None)
                if item_page == page_num:
                    blocks.append(item)

        return blocks

    except Exception as exc:
        logger.debug("Failed to extract blocks from docling for page {}: {}", page_num, exc)
        return []


def _build_page_document(
    page_num: int,
    raster: Image.Image,
    docling_result: Any,
) -> PageDocument:
    """Build a PageDocument from a rendered page and docling results."""
    raw_text = _extract_page_text(page_num, docling_result)
    docling_blocks = _extract_page_blocks(page_num, docling_result)

    return PageDocument(
        page_number=page_num,
        raw_text=raw_text,
        raster_image=raster,
        width_px=raster.width,
        height_px=raster.height,
        docling_blocks=docling_blocks,
        parse_failed=False,
    )


def _failed_page(page_num: int, raster: Image.Image) -> PageDocument:
    """Create a PageDocument for a page that failed to parse."""
    return PageDocument(
        page_number=page_num,
        raw_text="",
        raster_image=raster,
        width_px=raster.width,
        height_px=raster.height,
        docling_blocks=[],
        parse_failed=True,
    )


def _save_debug_rasters(pages: list[PageDocument], pdf_path: Path) -> None:
    """Save page raster images to disk for debugging."""
    debug_dir = pdf_path.parent / "debug_rasters" / pdf_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    for page in pages:
        out_path = debug_dir / f"page_{page.page_number:04d}.png"
        page.raster_image.save(str(out_path))
        logger.debug("Saved debug raster: {}", out_path)
