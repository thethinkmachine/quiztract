# Quiztract

**CBT Question Paper PDF → Structured JSON Pipeline**

Quiztract converts Computer Based Test (CBT) question paper PDFs into structured JSON documents, with derived Markdown and HTML views.

## Features

- **6-stage pipeline**: Ingest → Classify → Extract → Validate → Assemble → Render
- **VLM-powered extraction**: Uses `granite-vision-4.1-4b` for math transcription, figure analysis, and option OCR
- **Raster ground truth**: Every non-plain-text element gets a raster crop alongside its extracted text/LaTeX
- **Review flagging**: Low-confidence extractions are flagged for human review — never silently dropped
- **Multiple question types**: MCQ, MSQ, SA, COMPREHENSION (extensible)
- **Dual output**: Structured JSON + derived Markdown and HTML with KaTeX rendering

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your VLM settings

# 3. Run on a single PDF
python run.py --input input/QIB3_2024.pdf --exam-code QIB3

# 4. Bulk processing
python run.py --input-dir input/ --bulk
```

## Output Structure

```
output/
└── QIB3/
    ├── QIB3.json              # Primary structured output
    ├── QIB3.md                # Derived Markdown view
    ├── QIB3.html              # Derived HTML view (with KaTeX)
    ├── QIB3.provenance.json   # Pipeline execution details
    └── assets/
        ├── figures/           # Figure raster crops
        ├── math_crops/        # Math expression crops
        └── option_crops/      # Image-rendered option crops
```

## CLI Options

```
--input PATH           Path to a single PDF file
--input-dir PATH       Path to directory of PDFs (use with --bulk)
--bulk                 Process all PDFs in input-dir
--exam-code CODE       Override exam code (single file mode only)
--skip-render          Skip Stage 6, produce JSON only
--skip-validation      Skip Stage 4 (not recommended)
--debug                Save intermediate representations and page rasters
--review-only          Re-render MD/HTML for flagged questions
--config PATH          Path to alternate pipeline.yaml
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `pipeline/ingest.py` | PDF parsing with docling, page rasterization at 300 DPI |
| 2 | `pipeline/classify.py` | Block classification, question boundary detection |
| 3 | `pipeline/extract.py` | Content extraction (text/LaTeX/VLM), raster crop saving |
| 4 | `pipeline/validate.py` | Matrix dimension checks, answer range validation, LaTeX balance |
| 5 | `pipeline/assemble.py` | JSON assembly with null policy, provenance generation |
| 6 | `pipeline/render.py` | Markdown and HTML rendering from JSON |

## Configuration

All tunable parameters are in `config/pipeline.yaml`. Key settings:

- **DPI**: Page rendering resolution (default: 300)
- **VLM model**: Which vision-language model to use
- **Confidence threshold**: Below which blocks are flagged for review
- **Option count limits**: Expected MCQ option range for validation

## Running Tests

```bash
pytest tests/ -v
```

## Adding a New Question Type

1. Add extraction logic in `pipeline/extract.py`
2. Add Pydantic model in `models/schema.py`
3. Add validation checks in `pipeline/validate.py`
4. Add renderer handling in `renderers/`
5. Existing types are never modified — new types are additive only

## License

See [LICENSE](LICENSE) for details.