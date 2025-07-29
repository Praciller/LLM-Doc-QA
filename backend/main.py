"""FastAPI main application."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .models import (
    SummarizeRequest,
    SummarizeResponse,
    QueryRequest,
    QueryResponse,
    ErrorResponse
)
from .gemini_client import gemini_client
from .pdf_processor import get_pdf_processor, PDFProcessingError

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting AI Document Q&A System")
    yield
    logger.info("Shutting down AI Document Q&A System")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered document Q&A system with summarization and query capabilities",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else "An unexpected error occurred"
        ).model_dump()
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "summarize": "/summarize",
            "summarize_pdf": "/summarize-pdf",
            "query": "/query",
            "query_pdf": "/query-pdf",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-document-qa-system"}


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Summarize the provided text using Google Gemini Pro.
    
    Args:
        request: Summarization request containing text and options
        
    Returns:
        Summary response with generated summary and metadata
        
    Raises:
        HTTPException: If summarization fails
    """
    try:
        logger.info(f"Summarization request received for text of length: {len(request.text)}")
        
        # Generate summary using Gemini
        summary = await gemini_client.summarize_text(
            text=request.text,
            max_length=request.max_length,
            style=request.style or "concise"
        )
        
        # Calculate metrics
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        response = SummarizeResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            source_type="text"
        )
        
        logger.info(f"Summarization completed. Compression ratio: {compression_ratio:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def query_document(request: QueryRequest) -> QueryResponse:
    """
    Answer a question based on the provided context using RAG approach.
    
    Args:
        request: Query request containing question and context
        
    Returns:
        Query response with generated answer
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Query request received: '{request.question[:100]}...'")
        
        # Generate answer using RAG approach
        answer = await gemini_client.answer_question(
            question=request.question,
            context=request.context,
            include_sources=request.include_sources or False
        )
        
        response = QueryResponse(
            answer=answer,
            confidence=None,  # Could be implemented with additional logic
            sources=None if not request.include_sources else [],  # Could extract sources
            source_type="text"
        )
        
        logger.info("Query processing completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post(
    "/summarize-pdf",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def summarize_pdf(
    file: UploadFile = File(...),
    max_length: Optional[int] = Form(None),
    style: str = Form("concise")
) -> SummarizeResponse:
    """
    Summarize a PDF file using Google Gemini 2.0 Flash.

    Args:
        file: PDF file to summarize
        max_length: Maximum length of the summary in words
        style: Summary style ('concise', 'detailed', 'bullet_points')

    Returns:
        Summary response with generated summary and metadata

    Raises:
        HTTPException: If PDF processing or summarization fails
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        # Read file content
        pdf_content = await file.read()
        logger.info(f"PDF upload received: {file.filename}, size: {len(pdf_content)} bytes")

        # Process PDF
        pdf_processor = get_pdf_processor()

        if not pdf_processor.validate_pdf(pdf_content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file"
            )

        text_content, pdf_metadata = pdf_processor.extract_text_from_bytes(pdf_content)

        # Generate summary using Gemini
        summary = await gemini_client.summarize_text(
            text=text_content,
            max_length=max_length,
            style=style or "concise"
        )

        # Calculate metrics
        original_length = len(text_content)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        response = SummarizeResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            source_type="pdf",
            pdf_metadata=pdf_metadata
        )

        logger.info(f"PDF summarization completed for {file.filename}. "
                   f"Compression ratio: {compression_ratio:.2f}")
        return response

    except PDFProcessingError as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF processing failed: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF summarization failed: {str(e)}"
        )


@app.post(
    "/query-pdf",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def query_pdf(
    file: UploadFile = File(...),
    question: str = Form(...),
    include_sources: bool = Form(False)
) -> QueryResponse:
    """
    Answer a question about a PDF document using RAG approach.

    Args:
        file: PDF file to query
        question: The question to ask about the PDF content
        include_sources: Whether to include source references

    Returns:
        Query response with generated answer

    Raises:
        HTTPException: If PDF processing or query processing fails
    """
    try:
        # Validate inputs
        if not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        # Read file content
        pdf_content = await file.read()
        logger.info(f"PDF query received: {file.filename}, question: '{question[:100]}...'")

        # Process PDF
        pdf_processor = get_pdf_processor()

        if not pdf_processor.validate_pdf(pdf_content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file"
            )

        text_content, pdf_metadata = pdf_processor.extract_text_from_bytes(pdf_content)

        # Generate answer using RAG approach
        answer = await gemini_client.answer_question(
            question=question,
            context=text_content,
            include_sources=include_sources
        )

        response = QueryResponse(
            answer=answer,
            confidence=None,  # Could be implemented with additional logic
            sources=None if not include_sources else [],  # Could extract sources
            source_type="pdf",
            pdf_metadata=pdf_metadata
        )

        logger.info(f"PDF query processing completed for {file.filename}")
        return response

    except PDFProcessingError as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF processing failed: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF query processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload,
        log_level=settings.log_level.lower()
    )
