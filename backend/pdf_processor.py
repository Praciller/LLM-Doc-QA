"""PDF processing utilities for extracting text from PDF files."""

import io
import logging
from typing import Optional, Tuple

try:
    import pypdf
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        import PyPDF2 as pypdf
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        pypdf = None
        PDF_LIBRARY = None

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor:
    """Utility class for processing PDF files."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        if pypdf is None:
            raise PDFProcessingError(
                "No PDF library available. Please install pypdf or PyPDF2: "
                "pip install pypdf"
            )
        logger.info(f"PDF processor initialized with {PDF_LIBRARY}")
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Tuple[str, dict]:
        """
        Extract text from PDF bytes.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Tuple of (extracted_text, metadata)
            
        Raises:
            PDFProcessingError: If text extraction fails
        """
        try:
            # Create a file-like object from bytes
            pdf_file = io.BytesIO(pdf_bytes)
            
            if PDF_LIBRARY == "pypdf":
                reader = pypdf.PdfReader(pdf_file)
            else:  # PyPDF2
                reader = pypdf.PdfFileReader(pdf_file)
            
            # Extract metadata
            metadata = self._extract_metadata(reader)
            
            # Extract text from all pages
            text_content = []
            num_pages = len(reader.pages) if PDF_LIBRARY == "pypdf" else reader.numPages
            
            for page_num in range(num_pages):
                try:
                    if PDF_LIBRARY == "pypdf":
                        page = reader.pages[page_num]
                        text = page.extract_text()
                    else:  # PyPDF2
                        page = reader.getPage(page_num)
                        text = page.extractText()
                    
                    if text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{text.strip()}")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            if not text_content:
                raise PDFProcessingError("No text could be extracted from the PDF")
            
            full_text = "\n\n".join(text_content)
            metadata["pages_processed"] = len(text_content)
            metadata["total_pages"] = num_pages
            metadata["text_length"] = len(full_text)
            
            logger.info(f"Successfully extracted text from {num_pages} pages, "
                       f"total length: {len(full_text)} characters")
            
            return full_text, metadata
            
        except PDFProcessingError:
            raise
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise PDFProcessingError(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_metadata(self, reader) -> dict:
        """Extract metadata from PDF reader."""
        metadata = {
            "title": None,
            "author": None,
            "subject": None,
            "creator": None,
            "producer": None,
            "creation_date": None,
            "modification_date": None,
        }
        
        try:
            if PDF_LIBRARY == "pypdf":
                pdf_metadata = reader.metadata
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("/Title"),
                        "author": pdf_metadata.get("/Author"),
                        "subject": pdf_metadata.get("/Subject"),
                        "creator": pdf_metadata.get("/Creator"),
                        "producer": pdf_metadata.get("/Producer"),
                        "creation_date": str(pdf_metadata.get("/CreationDate", "")),
                        "modification_date": str(pdf_metadata.get("/ModDate", "")),
                    })
            else:  # PyPDF2
                pdf_metadata = reader.getDocumentInfo()
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("/Title"),
                        "author": pdf_metadata.get("/Author"),
                        "subject": pdf_metadata.get("/Subject"),
                        "creator": pdf_metadata.get("/Creator"),
                        "producer": pdf_metadata.get("/Producer"),
                        "creation_date": str(pdf_metadata.get("/CreationDate", "")),
                        "modification_date": str(pdf_metadata.get("/ModDate", "")),
                    })
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
        
        return metadata
    
    def validate_pdf(self, pdf_bytes: bytes) -> bool:
        """
        Validate if the provided bytes represent a valid PDF file.
        
        Args:
            pdf_bytes: File content as bytes
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            
            if PDF_LIBRARY == "pypdf":
                reader = pypdf.PdfReader(pdf_file)
                # Try to access pages to validate
                _ = len(reader.pages)
            else:  # PyPDF2
                reader = pypdf.PdfFileReader(pdf_file)
                # Try to access pages to validate
                _ = reader.numPages
            
            return True
            
        except Exception as e:
            logger.warning(f"PDF validation failed: {e}")
            return False


# Global PDF processor instance
def get_pdf_processor() -> PDFProcessor:
    """Get a PDF processor instance."""
    try:
        return PDFProcessor()
    except PDFProcessingError as e:
        logger.error(f"Failed to initialize PDF processor: {e}")
        raise
