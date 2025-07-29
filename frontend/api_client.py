"""API client for communicating with the FastAPI backend."""

import asyncio
import httpx
import streamlit as st
from typing import Dict, Any, Optional

from .config import config


class APIClient:
    """Client for making requests to the FastAPI backend."""
    
    def __init__(self, base_url: str = None):
        """Initialize the API client."""
        self.base_url = base_url or config.API_BASE_URL
        self.timeout = 60.0  # 60 seconds timeout
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST requests
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            raise Exception("Request timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                error_detail = e.response.json().get("detail", "Validation error")
                raise Exception(f"Validation error: {error_detail}")
            elif e.response.status_code == 500:
                error_detail = e.response.json().get("detail", "Internal server error")
                raise Exception(f"Server error: {error_detail}")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        return await self._make_request("GET", "/health")
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Get API information."""
        return await self._make_request("GET", "/")
    
    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> Dict[str, Any]:
        """
        Summarize text using the API.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            style: Summary style
            
        Returns:
            Summarization response
        """
        data = {
            "text": text,
            "style": style
        }
        if max_length:
            data["max_length"] = max_length
            
        return await self._make_request("POST", "/summarize", data)
    
    async def query_document(
        self,
        question: str,
        context: str,
        include_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Query a document using the API.
        
        Args:
            question: Question to ask
            context: Document context
            include_sources: Whether to include sources
            
        Returns:
            Query response
        """
        data = {
            "question": question,
            "context": context,
            "include_sources": include_sources
        }
        
        return await self._make_request("POST", "/query", data)

    async def summarize_pdf(
        self,
        file_content: bytes,
        filename: str,
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> Dict[str, Any]:
        """
        Summarize a PDF file using the API.

        Args:
            file_content: PDF file content as bytes
            filename: Name of the PDF file
            max_length: Maximum length of summary
            style: Summary style

        Returns:
            Summarization response
        """
        url = f"{self.base_url}/summarize-pdf"

        try:
            files = {"file": (filename, file_content, "application/pdf")}
            data = {"style": style}
            if max_length:
                data["max_length"] = str(max_length)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise Exception("Request timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                error_detail = e.response.json().get("detail", "Validation error")
                raise Exception(f"Validation error: {error_detail}")
            elif e.response.status_code == 500:
                error_detail = e.response.json().get("detail", "Internal server error")
                raise Exception(f"Server error: {error_detail}")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    async def query_pdf(
        self,
        file_content: bytes,
        filename: str,
        question: str,
        include_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Query a PDF file using the API.

        Args:
            file_content: PDF file content as bytes
            filename: Name of the PDF file
            question: Question to ask
            include_sources: Whether to include sources

        Returns:
            Query response
        """
        url = f"{self.base_url}/query-pdf"

        try:
            files = {"file": (filename, file_content, "application/pdf")}
            data = {
                "question": question,
                "include_sources": str(include_sources).lower()
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise Exception("Request timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                error_detail = e.response.json().get("detail", "Validation error")
                raise Exception(f"Validation error: {error_detail}")
            elif e.response.status_code == 500:
                error_detail = e.response.json().get("detail", "Internal server error")
                raise Exception(f"Server error: {error_detail}")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")


# Global API client instance
@st.cache_resource
def get_api_client() -> APIClient:
    """Get a cached API client instance."""
    return APIClient()
