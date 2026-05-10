"""
pdf_parser.py
Handles PDF text extraction and cleaning for AI Resume Optimizer.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract and clean text from a PDF file.
    Tries PyMuPDF first, falls back to pdfplumber.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Cleaned text string extracted from the PDF.
    """
    if not file_path or not Path(file_path).exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    text = ""

    # Try PyMuPDF (fitz) first
    try:
        import fitz  # PyMuPDF
        text = _extract_with_pymupdf(file_path)
        logger.info("Extracted PDF text using PyMuPDF")
    except ImportError:
        logger.warning("PyMuPDF not available, trying pdfplumber...")
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}, trying pdfplumber...")

    # Fallback to pdfplumber
    if not text.strip():
        try:
            import pdfplumber
            text = _extract_with_pdfplumber(file_path)
            logger.info("Extracted PDF text using pdfplumber")
        except ImportError:
            raise ImportError(
                "Neither PyMuPDF nor pdfplumber is installed. "
                "Run: pip install PyMuPDF pdfplumber"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF text: {e}")

    if not text.strip():
        raise ValueError("Could not extract any text from the PDF. The file may be image-based or corrupted.")

    return clean_text(text)


def _extract_with_pymupdf(file_path: str) -> str:
    """Extract text using PyMuPDF (fitz)."""
    import fitz
    doc = fitz.open(file_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        pages_text.append(page_text)

    doc.close()
    return "\n".join(pages_text)


def _extract_with_pdfplumber(file_path: str) -> str:
    """Extract text using pdfplumber."""
    import pdfplumber
    pages_text = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

    return "\n".join(pages_text)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    - Remove excessive whitespace
    - Normalize line breaks
    - Remove special control characters
    - Preserve meaningful structure

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode dashes and quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '•').replace('\u25cf', '•')

    # Remove excessive blank lines (keep max 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split('\n')]

    # Remove lines that are just whitespace or single characters (artifacts)
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped or (cleaned_lines and cleaned_lines[-1] != ''):
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Normalize spaces within lines
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def get_pdf_metadata(file_path: str) -> dict:
    """
    Get basic metadata from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Dictionary with PDF metadata.
    """
    metadata = {"pages": 0, "file_size_kb": 0, "title": ""}

    try:
        file_size = Path(file_path).stat().st_size / 1024
        metadata["file_size_kb"] = round(file_size, 1)

        try:
            import fitz
            doc = fitz.open(file_path)
            metadata["pages"] = len(doc)
            meta = doc.metadata
            metadata["title"] = meta.get("title", "")
            doc.close()
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Could not get PDF metadata: {e}")

    return metadata
