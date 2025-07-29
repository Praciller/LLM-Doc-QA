"""Tests for PDF processing functionality."""

import pytest
from unittest.mock import patch, MagicMock
from backend.pdf_processor import PDFProcessor, PDFProcessingError


def test_pdf_processor_initialization():
    """Test PDF processor initialization."""
    processor = PDFProcessor()
    assert processor is not None


@patch('backend.pdf_processor.pypdf')
def test_extract_text_from_bytes_success(mock_pypdf):
    """Test successful text extraction from PDF bytes."""
    # Mock PDF reader
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Sample PDF text content"
    mock_reader.pages = [mock_page]
    mock_reader.metadata = {
        "/Title": "Test Document",
        "/Author": "Test Author"
    }
    mock_pypdf.PdfReader.return_value = mock_reader
    
    processor = PDFProcessor()
    pdf_bytes = b"fake pdf content"
    
    text, metadata = processor.extract_text_from_bytes(pdf_bytes)
    
    assert "Sample PDF text content" in text
    assert metadata["title"] == "Test Document"
    assert metadata["author"] == "Test Author"
    assert metadata["pages_processed"] == 1
    assert metadata["total_pages"] == 1


@patch('backend.pdf_processor.pypdf')
def test_extract_text_from_bytes_empty_text(mock_pypdf):
    """Test handling of PDF with no extractable text."""
    # Mock PDF reader with empty text
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader.pages = [mock_page]
    mock_reader.metadata = {}
    mock_pypdf.PdfReader.return_value = mock_reader
    
    processor = PDFProcessor()
    pdf_bytes = b"fake pdf content"
    
    with pytest.raises(PDFProcessingError, match="No text could be extracted"):
        processor.extract_text_from_bytes(pdf_bytes)


@patch('backend.pdf_processor.pypdf')
def test_validate_pdf_success(mock_pypdf):
    """Test successful PDF validation."""
    # Mock PDF reader
    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock()]
    mock_pypdf.PdfReader.return_value = mock_reader
    
    processor = PDFProcessor()
    pdf_bytes = b"fake pdf content"
    
    result = processor.validate_pdf(pdf_bytes)
    
    assert result is True


@patch('backend.pdf_processor.pypdf')
def test_validate_pdf_failure(mock_pypdf):
    """Test PDF validation failure."""
    # Mock PDF reader that raises exception
    mock_pypdf.PdfReader.side_effect = Exception("Invalid PDF")
    
    processor = PDFProcessor()
    pdf_bytes = b"invalid content"
    
    result = processor.validate_pdf(pdf_bytes)
    
    assert result is False


@patch('backend.pdf_processor.pypdf', None)
def test_pdf_processor_no_library():
    """Test PDF processor when no PDF library is available."""
    with pytest.raises(PDFProcessingError, match="No PDF library available"):
        PDFProcessor()
