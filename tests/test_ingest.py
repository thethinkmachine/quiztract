"""Tests for Stage 1 — PDF Ingestion."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

from models.schema import PageDocument


@pytest.fixture
def default_config():
    return {
        "ingest": {
            "dpi": 150,  # Lower DPI for tests
            "page_timeout_seconds": 10,
            "fail_on_parse_error": False,
            "debug_save_page_rasters": False,
        }
    }


@pytest.fixture
def mock_raster():
    """A small test image."""
    return Image.new("RGB", (100, 140), (255, 255, 255))


class TestIngestModule:
    """Test the ingest module functions."""

    def test_page_document_creation(self, mock_raster):
        """Test that PageDocument can be created with all fields."""
        page = PageDocument(
            page_number=1,
            raw_text="Sample text",
            raster_image=mock_raster,
            width_px=100,
            height_px=140,
            docling_blocks=[],
        )
        assert page.page_number == 1
        assert page.raw_text == "Sample text"
        assert page.width_px == 100
        assert page.height_px == 140
        assert page.parse_failed is False

    def test_failed_page_document(self, mock_raster):
        """Test creating a failed page document."""
        page = PageDocument(
            page_number=3,
            raw_text="",
            raster_image=mock_raster,
            width_px=100,
            height_px=140,
            docling_blocks=[],
            parse_failed=True,
        )
        assert page.parse_failed is True
        assert page.raw_text == ""

    def test_ingest_file_not_found(self, default_config):
        """Test that ingest raises FileNotFoundError for missing PDF."""
        from pipeline.ingest import ingest_pdf

        with pytest.raises(FileNotFoundError):
            ingest_pdf("/nonexistent/path.pdf", default_config)

    @patch("pipeline.ingest._render_pages")
    @patch("pipeline.ingest._parse_with_docling")
    def test_ingest_basic(self, mock_docling, mock_render, default_config, mock_raster, tmp_path):
        """Test basic ingestion with mocked PDF processing."""
        # Create a dummy PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        mock_render.return_value = [mock_raster, mock_raster]
        mock_docling.return_value = None

        from pipeline.ingest import ingest_pdf

        pages = ingest_pdf(pdf_path, default_config)
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
