"""Configuration settings for the FastAPI backend."""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google Gemini API Configuration
    google_api_key: str
    
    # FastAPI Configuration
    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    fastapi_reload: bool = True
    
    # Application Configuration
    app_name: str = "AI Document Q&A System"
    app_version: str = "0.1.0"
    debug: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    
    # Gemini Model Configuration
    gemini_model: str = "gemini-2.0-flash"
    max_tokens: int = 2048
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file


# Global settings instance
settings = Settings()
