"""Main Streamlit application for the AI Document Q&A System."""

import asyncio
import streamlit as st
from typing import Optional

from .config import config
from .api_client import get_api_client


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
        """
    )


def check_api_connection():
    """Check if the API is accessible."""
    try:
        client = get_api_client()
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        health = loop.run_until_complete(client.health_check())
        loop.close()
        
        if health.get("status") == "healthy":
            st.sidebar.success("✅ API Connected")
            return True
        else:
            st.sidebar.error("❌ API Unhealthy")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ API Connection Failed: {str(e)}")
        return False


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
                client = get_api_client()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                if input_method == "📝 Text Input":
                    result = loop.run_until_complete(
                        client.summarize_text(
                            text=text_input,
                            max_length=max_length,
                            style=summary_style
                        )
                    )
                else:
                    # PDF upload
                    file_content = uploaded_file.getvalue()
                    result = loop.run_until_complete(
                        client.summarize_pdf(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            max_length=max_length,
                            style=summary_style
                        )
                    )

                loop.close()

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
                client = get_api_client()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                if input_method == "📝 Text Input":
                    result = loop.run_until_complete(
                        client.query_document(
                            question=question_input,
                            context=context_input,
                            include_sources=include_sources
                        )
                    )
                else:
                    # PDF upload
                    file_content = uploaded_file.getvalue()
                    result = loop.run_until_complete(
                        client.query_pdf(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            question=question_input,
                            include_sources=include_sources
                        )
                    )

                loop.close()

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
    api_connected = check_api_connection()
    
    if not api_connected:
        st.error(
            """
            ⚠️ **API Connection Failed**
            
            Please ensure the FastAPI backend is running:
            ```bash
            python -m backend.main
            ```
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
            AI Document Q&A System | Powered by Google Gemini 2.0 Flash & FastAPI
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
