"""Pydantic models for request/response validation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """Request model for the summarize endpoint."""
    
    text: str = Field(
        ...,
        description="The text content to summarize",
        min_length=1,
        max_length=50000
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length of the summary in words",
        ge=10,
        le=1000
    )
    style: Optional[str] = Field(
        default="concise",
        description="Summary style: 'concise', 'detailed', or 'bullet_points'"
    )


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    
    question: str = Field(
        ...,
        description="The question to ask about the context",
        min_length=1,
        max_length=1000
    )
    context: str = Field(
        ...,
        description="The context/document content to query against",
        min_length=1,
        max_length=50000
    )
    include_sources: Optional[bool] = Field(
        default=False,
        description="Whether to include source references in the response"
    )


class SummarizeResponse(BaseModel):
    """Response model for the summarize endpoint."""

    summary: str = Field(
        ...,
        description="The generated summary"
    )
    original_length: int = Field(
        ...,
        description="Length of the original text in characters"
    )
    summary_length: int = Field(
        ...,
        description="Length of the summary in characters"
    )
    compression_ratio: float = Field(
        ...,
        description="Ratio of summary length to original length"
    )
    source_type: str = Field(
        default="text",
        description="Source type: 'text' or 'pdf'"
    )
    pdf_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="PDF metadata if source was a PDF file"
    )


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""

    answer: str = Field(
        ...,
        description="The answer to the question based on the context"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the answer (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Source references if requested"
    )
    source_type: str = Field(
        default="text",
        description="Source type: 'text' or 'pdf'"
    )
    pdf_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="PDF metadata if source was a PDF file"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )
