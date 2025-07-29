"""Configuration for the Streamlit frontend."""

import os
from typing import Optional


class FrontendConfig:
    """Configuration settings for the Streamlit frontend."""
    
    # API Configuration
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    
    # Streamlit Configuration
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "127.0.0.1")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # UI Configuration
    PAGE_TITLE: str = "AI Document Q&A System"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"
    
    # Default values
    DEFAULT_SUMMARY_STYLE: str = "concise"
    DEFAULT_MAX_SUMMARY_LENGTH: int = 200
    
    # Text limits
    MAX_TEXT_LENGTH: int = 50000
    MAX_QUESTION_LENGTH: int = 1000

    # File upload limits
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: list = ["pdf"]


# Global config instance
config = FrontendConfig()
