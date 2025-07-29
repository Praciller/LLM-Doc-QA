"""Google Gemini API client for AI operations."""

import logging
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google Gemini 2.0 Flash API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self._configure_api()
        self.model = genai.GenerativeModel(settings.gemini_model)
        
    def _configure_api(self) -> None:
        """Configure the Google Generative AI API."""
        try:
            genai.configure(api_key=settings.google_api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using Gemini 2.0 Flash.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If text generation fails
        """
        try:
            # Use provided parameters or fall back to settings
            temp = temperature if temperature is not None else settings.temperature
            max_tok = max_tokens if max_tokens is not None else settings.max_tokens
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tok,
                candidate_count=1,
            )
            
            # Configure safety settings to be less restrictive for document Q&A
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Generate content
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if not response.text:
                raise Exception("Empty response from Gemini API")
                
            logger.info(f"Successfully generated text of length: {len(response.text)}")
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise Exception(f"Failed to generate text: {str(e)}")
    
    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> str:
        """
        Summarize text using a specialized prompt.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            style: Summary style ('concise', 'detailed', 'bullet_points')
            
        Returns:
            Generated summary
        """
        from .prompt_templates import get_summarization_prompt
        
        prompt = get_summarization_prompt(text, max_length, style)
        return await self.generate_text(prompt, temperature=0.3)
    
    async def answer_question(
        self,
        question: str,
        context: str,
        include_sources: bool = False
    ) -> str:
        """
        Answer a question based on provided context.
        
        Args:
            question: The question to answer
            context: The context/document content
            include_sources: Whether to include source references
            
        Returns:
            Generated answer
        """
        from .prompt_templates import get_rag_prompt
        
        prompt = get_rag_prompt(question, context, include_sources)
        return await self.generate_text(prompt, temperature=0.5)


# Global client instance
gemini_client = GeminiClient()
