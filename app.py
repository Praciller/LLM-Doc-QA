"""Standalone Streamlit application for AI Document Q&A System."""

import asyncio
import io
import os
import logging
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import pypdf
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        import PyPDF2 as pypdf
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        st.error("No PDF library found. Please install pypdf or PyPDF2.")
        st.stop()


# Configuration
class Config:
    """Application configuration."""

    # UI Configuration
    PAGE_TITLE = "AI Document Q&A System"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"

    # Default values
    DEFAULT_SUMMARY_STYLE = "concise"
    DEFAULT_MAX_SUMMARY_LENGTH = 200

    # Text limits
    MAX_TEXT_LENGTH = 50000
    MAX_QUESTION_LENGTH = 1000

    # File upload limits
    MAX_FILE_SIZE_MB = 10
    ALLOWED_FILE_TYPES = ["pdf"]

    # Gemini Configuration
    GEMINI_MODEL = "gemini-2.0-flash"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7


config = Config()


# Prompt Templates
SUMMARIZATION_TEMPLATES = {
    "concise": """
You are an expert text summarizer. Your task is to create a concise, accurate summary of the provided text.

Instructions:
- Focus on the main ideas and key points
- Maintain the original meaning and context
- Use clear, professional language
- Avoid unnecessary details or repetition
{% if max_length %}
- Keep the summary to approximately {{ max_length }} words or less
{% endif %}

Text to summarize:
{{ text }}

Summary:""",

    "detailed": """
You are an expert text summarizer. Your task is to create a comprehensive, detailed summary of the provided text.

Instructions:
- Include all important points and supporting details
- Maintain the structure and flow of the original text
- Use clear, professional language
- Provide context for key concepts
{% if max_length %}
- Keep the summary to approximately {{ max_length }} words or less
{% endif %}

Text to summarize:
{{ text }}

Detailed Summary:""",

    "bullet_points": """
You are an expert text summarizer. Your task is to create a bullet-point summary of the provided text.

Instructions:
- Extract the main ideas and present them as clear bullet points
- Use concise, actionable language
- Organize points logically
- Include the most important information first
{% if max_length %}
- Keep the summary to approximately {{ max_length }} words or less
{% endif %}

Text to summarize:
{{ text }}

Bullet Point Summary:"""
}

RAG_TEMPLATE = """
You are an expert AI assistant specializing in document analysis and question answering. Your task is to answer questions based on the provided context.

Instructions:
- Answer the question using ONLY the information provided in the context
- If the answer is not in the context, clearly state that the information is not available
- Be accurate, concise, and helpful
- Maintain a professional tone
{% if include_sources %}
- Include references to specific parts of the document when possible
{% endif %}

Context:
{{ context }}

Question: {{ question }}

Answer:"""


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor:
    """Handles PDF file processing and text extraction."""

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Tuple[str, dict]:
        """Extract text from PDF bytes."""
        try:
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

            logger.info(f"Successfully extracted {len(full_text)} characters from {len(text_content)} pages")
            return full_text, metadata

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise PDFProcessingError(f"Failed to process PDF: {str(e)}")

    def _extract_metadata(self, reader) -> dict:
        """Extract metadata from PDF reader."""
        metadata = {
            "total_pages": len(reader.pages) if PDF_LIBRARY == "pypdf" else reader.numPages,
            "title": None,
            "author": None,
            "creation_date": None,
            "pages_processed": 0
        }

        try:
            if PDF_LIBRARY == "pypdf":
                pdf_metadata = reader.metadata
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("/Title"),
                        "author": pdf_metadata.get("/Author"),
                        "creation_date": str(pdf_metadata.get("/CreationDate", ""))
                    })
            else:  # PyPDF2
                pdf_metadata = reader.getDocumentInfo()
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("/Title"),
                        "author": pdf_metadata.get("/Author"),
                        "creation_date": str(pdf_metadata.get("/CreationDate", ""))
                    })
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")

        return metadata


class GeminiClient:
    """Client for interacting with Google Gemini 2.0 Flash API."""

    def __init__(self, api_key: str):
        """Initialize the Gemini client."""
        self.api_key = api_key
        self._configure_api()
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)

    def _configure_api(self) -> None:
        """Configure the Google Generative AI API."""
        try:
            genai.configure(api_key=self.api_key)
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
        """Generate text using Gemini 2.0 Flash."""
        try:
            temp = temperature if temperature is not None else config.TEMPERATURE
            max_tok = max_tokens if max_tokens is not None else config.MAX_TOKENS

            generation_config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tok,
                candidate_count=1,
            )

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

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


def get_summarization_prompt(text: str, max_length: Optional[int] = None, style: str = "concise") -> str:
    """Generate a summarization prompt based on the specified style."""
    if style not in SUMMARIZATION_TEMPLATES:
        style = "concise"

    template = Template(SUMMARIZATION_TEMPLATES[style])
    return template.render(text=text, max_length=max_length)


def get_rag_prompt(question: str, context: str, include_sources: bool = False) -> str:
    """Generate a RAG (Retrieval-Augmented Generation) prompt."""
    template = Template(RAG_TEMPLATE)
    return template.render(
        question=question,
        context=context,
        include_sources=include_sources
    )


# Initialize global instances
@st.cache_resource
def get_pdf_processor():
    """Get a cached PDF processor instance."""
    return PDFProcessor()


@st.cache_resource
def get_gemini_client():
    """Get a cached Gemini client instance."""
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found. Please set GOOGLE_API_KEY environment variable or add it to Streamlit secrets.")
        st.stop()
    return GeminiClient(api_key)


def configure_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
        initial_sidebar_state="expanded"
    )


def display_header():
    """Display the application header."""
    st.title("🤖 AI Document Q&A System")
    st.markdown(
        """
        Welcome to the AI Document Q&A System! This application provides two main features:
        - **📝 Text Summarization**: Generate concise summaries of your documents
        - **❓ Document Q&A**: Ask questions about your documents and get AI-powered answers

        *Powered by Google Gemini 2.0 Flash*
        """
    )


def check_api_status():
    """Check and display API status."""
    try:
        client = get_gemini_client()
        st.sidebar.success("✅ Gemini API Connected")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Gemini API Error: {str(e)}")
        return False


async def summarize_text_async(text: str, max_length: Optional[int] = None, style: str = "concise") -> Dict[str, Any]:
    """Summarize text using Gemini."""
    client = get_gemini_client()
    prompt = get_summarization_prompt(text, max_length, style)
    summary = await client.generate_text(prompt, temperature=0.3)

    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": len(summary) / len(text) if len(text) > 0 else 0,
        "source_type": "text"
    }


async def query_text_async(question: str, context: str, include_sources: bool = False) -> Dict[str, Any]:
    """Answer question based on text context."""
    client = get_gemini_client()
    prompt = get_rag_prompt(question, context, include_sources)
    answer = await client.generate_text(prompt, temperature=0.5)

    return {
        "answer": answer,
        "source_type": "text",
        "sources": [] if include_sources else None
    }


async def process_pdf_async(pdf_bytes: bytes, operation: str, **kwargs) -> Dict[str, Any]:
    """Process PDF for summarization or Q&A."""
    processor = get_pdf_processor()
    text_content, pdf_metadata = processor.extract_text_from_bytes(pdf_bytes)

    if operation == "summarize":
        result = await summarize_text_async(
            text_content,
            kwargs.get("max_length"),
            kwargs.get("style", "concise")
        )
    elif operation == "query":
        result = await query_text_async(
            kwargs.get("question"),
            text_content,
            kwargs.get("include_sources", False)
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")

    result["source_type"] = "pdf"
    result["pdf_metadata"] = pdf_metadata
    return result


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


def summarization_tab():
    """Display the text summarization interface."""
    st.header("📝 Text Summarization")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "📄 PDF Upload"],
        horizontal=True
    )

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        if input_method == "📝 Text Input":
            text_input = st.text_area(
                "Enter text to summarize:",
                height=200,
                max_chars=config.MAX_TEXT_LENGTH,
                help=f"Maximum {config.MAX_TEXT_LENGTH:,} characters"
            )
            uploaded_file = None
        else:
            text_input = ""
            uploaded_file = st.file_uploader(
                "Upload a PDF file to summarize:",
                type=config.ALLOWED_FILE_TYPES,
                help=f"Maximum file size: {config.MAX_FILE_SIZE_MB}MB"
            )

            if uploaded_file:
                # Check file size
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > config.MAX_FILE_SIZE_MB:
                    st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum limit of {config.MAX_FILE_SIZE_MB}MB")
                    uploaded_file = None
                else:
                    st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f}MB)")

    with col2:
        st.subheader("Options")

        summary_style = st.selectbox(
            "Summary Style:",
            options=["concise", "detailed", "bullet_points"],
            index=0,
            help="Choose the style of summary you prefer"
        )

        max_length = st.number_input(
            "Max Length (words):",
            min_value=10,
            max_value=1000,
            value=config.DEFAULT_MAX_SUMMARY_LENGTH,
            help="Maximum number of words in the summary"
        )

        summarize_button = st.button(
            "🔄 Generate Summary",
            type="primary",
            use_container_width=True
        )

    # Processing and results
    if summarize_button:
        # Validate input
        if input_method == "📝 Text Input":
            if not text_input.strip():
                st.error("Please enter some text to summarize.")
                return
            if len(text_input) < 50:
                st.warning("Text is quite short. Consider entering more content for better summarization.")
        else:
            if not uploaded_file:
                st.error("Please upload a PDF file to summarize.")
                return

        with st.spinner("Generating summary..."):
            try:
                if input_method == "📝 Text Input":
                    result = run_async(summarize_text_async(
                        text=text_input,
                        max_length=max_length,
                        style=summary_style
                    ))
                else:
                    # PDF upload
                    file_content = uploaded_file.getvalue()
                    result = run_async(process_pdf_async(
                        pdf_bytes=file_content,
                        operation="summarize",
                        max_length=max_length,
                        style=summary_style
                    ))

                # Display results
                st.success("Summary generated successfully!")

                # Summary content
                st.subheader("📄 Summary")
                st.write(result["summary"])

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Original Length",
                        f"{result['original_length']:,} chars"
                    )
                with col2:
                    st.metric(
                        "Summary Length",
                        f"{result['summary_length']:,} chars"
                    )
                with col3:
                    st.metric(
                        "Compression Ratio",
                        f"{result['compression_ratio']:.1%}"
                    )

                # PDF metadata if available
                if result.get("source_type") == "pdf" and result.get("pdf_metadata"):
                    with st.expander("📋 PDF Information"):
                        metadata = result["pdf_metadata"]
                        col1, col2 = st.columns(2)
                        with col1:
                            if metadata.get("title"):
                                st.write(f"**Title:** {metadata['title']}")
                            if metadata.get("author"):
                                st.write(f"**Author:** {metadata['author']}")
                            if metadata.get("pages_processed"):
                                st.write(f"**Pages Processed:** {metadata['pages_processed']}")
                        with col2:
                            if metadata.get("total_pages"):
                                st.write(f"**Total Pages:** {metadata['total_pages']}")
                            if metadata.get("creation_date"):
                                st.write(f"**Created:** {metadata['creation_date'][:10]}")

            except Exception as e:
                st.error(f"Summarization failed: {str(e)}")


def query_tab():
    """Display the document Q&A interface."""
    st.header("❓ Document Q&A")

    # Input method selection
    input_method = st.radio(
        "Choose document input method:",
        ["📝 Text Input", "📄 PDF Upload"],
        horizontal=True,
        key="query_input_method"
    )

    # Context input
    if input_method == "📝 Text Input":
        st.subheader("📄 Document Context")
        context_input = st.text_area(
            "Paste your document content here:",
            height=200,
            max_chars=config.MAX_TEXT_LENGTH,
            help=f"Maximum {config.MAX_TEXT_LENGTH:,} characters"
        )
        uploaded_file = None
    else:
        st.subheader("📄 Upload PDF Document")
        context_input = ""
        uploaded_file = st.file_uploader(
            "Upload a PDF file to query:",
            type=config.ALLOWED_FILE_TYPES,
            help=f"Maximum file size: {config.MAX_FILE_SIZE_MB}MB",
            key="query_pdf_upload"
        )

        if uploaded_file:
            # Check file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum limit of {config.MAX_FILE_SIZE_MB}MB")
                uploaded_file = None
            else:
                st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f}MB)")

    # Question input
    col1, col2 = st.columns([3, 1])

    with col1:
        question_input = st.text_input(
            "Ask a question about the document:",
            max_chars=config.MAX_QUESTION_LENGTH,
            help=f"Maximum {config.MAX_QUESTION_LENGTH:,} characters"
        )

    with col2:
        st.subheader("Options")
        include_sources = st.checkbox(
            "Include Sources",
            value=False,
            help="Include source references in the answer"
        )

        query_button = st.button(
            "🔍 Get Answer",
            type="primary",
            use_container_width=True
        )

    # Processing and results
    if query_button:
        # Validate input
        if input_method == "📝 Text Input":
            if not context_input.strip():
                st.error("Please provide document context.")
                return
        else:
            if not uploaded_file:
                st.error("Please upload a PDF file to query.")
                return

        if not question_input.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Processing your question..."):
            try:
                if input_method == "📝 Text Input":
                    result = run_async(query_text_async(
                        question=question_input,
                        context=context_input,
                        include_sources=include_sources
                    ))
                else:
                    # PDF upload
                    file_content = uploaded_file.getvalue()
                    result = run_async(process_pdf_async(
                        pdf_bytes=file_content,
                        operation="query",
                        question=question_input,
                        include_sources=include_sources
                    ))

                # Display results
                st.success("Answer generated successfully!")

                st.subheader("💡 Answer")
                st.write(result["answer"])

                if result.get("sources") and include_sources:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        st.write(f"{i}. {source}")

                # PDF metadata if available
                if result.get("source_type") == "pdf" and result.get("pdf_metadata"):
                    with st.expander("📋 PDF Information"):
                        metadata = result["pdf_metadata"]
                        col1, col2 = st.columns(2)
                        with col1:
                            if metadata.get("title"):
                                st.write(f"**Title:** {metadata['title']}")
                            if metadata.get("author"):
                                st.write(f"**Author:** {metadata['author']}")
                            if metadata.get("pages_processed"):
                                st.write(f"**Pages Processed:** {metadata['pages_processed']}")
                        with col2:
                            if metadata.get("total_pages"):
                                st.write(f"**Total Pages:** {metadata['total_pages']}")
                            if metadata.get("creation_date"):
                                st.write(f"**Created:** {metadata['creation_date'][:10]}")

            except Exception as e:
                st.error(f"Query processing failed: {str(e)}")


def main():
    """Main application function."""
    configure_page()
    display_header()

    # Sidebar
    st.sidebar.title("🔧 System Status")
    api_connected = check_api_status()

    if not api_connected:
        st.error(
            """
            ⚠️ **Gemini API Connection Failed**

            Please ensure you have set your Google API key:
            - As an environment variable: `GOOGLE_API_KEY`
            - Or in Streamlit secrets: Add `GOOGLE_API_KEY` to your secrets

            You can get an API key from: https://makersuite.google.com/app/apikey
            """
        )
        return

    # Main tabs
    tab1, tab2 = st.tabs(["📝 Summarization", "❓ Q&A"])

    with tab1:
        summarization_tab()

    with tab2:
        query_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            AI Document Q&A System | Powered by Google Gemini 2.0 Flash
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
