# AI Document Q&A System

A powerful AI-driven document question-answering and summarization system built with FastAPI and Streamlit, powered by Google Gemini 2.0 Flash.

## Features

- **📝 Text Summarization**: Generate concise, detailed, or bullet-point summaries of your documents
- **📄 PDF Upload Support**: Upload PDF files directly for summarization and Q&A
- **❓ Document Q&A**: Ask questions about your documents and get AI-powered answers using RAG (Retrieval-Augmented Generation)
- **🚀 FastAPI Backend**: RESTful API with automatic documentation and validation
- **🎨 Streamlit Frontend**: User-friendly web interface for easy interaction
- **🔧 Modern Architecture**: Built with modern Python tools and best practices
- **📊 PDF Metadata Extraction**: Automatic extraction of PDF metadata (title, author, pages, etc.)

## Architecture

```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐    API Calls    ┌─────────────────┐
│                 │ ──────────────► │                 │ ──────────────► │                 │
│ Streamlit       │                 │ FastAPI         │                 │ Google Gemini   │
│ Frontend        │ ◄────────────── │ Backend         │ ◄────────────── │ 2.0 Flash API   │
│                 │                 │                 │                 │                 │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Google API key for Gemini 2.0 Flash
- uv package manager (recommended) or pip

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ai-document-qa-system
   ```

2. **Install dependencies using uv**:

   ```bash
   # Install uv if you haven't already
   pip install uv

   # Install project dependencies
   uv pip install -e .
   ```

   Or using pip:

   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Google API key:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

1. **Start the FastAPI backend**:

   ```bash
   python run_backend.py
   ```

   The API will be available at `http://127.0.0.1:8000`

   - API Documentation: `http://127.0.0.1:8000/docs`
   - Alternative docs: `http://127.0.0.1:8000/redoc`

2. **Start the Streamlit frontend** (in a new terminal):

   ```bash
   streamlit run app.py
   ```

   The web interface will be available at `http://127.0.0.1:8501`

## API Endpoints

### POST /summarize

Summarize text content using Google Gemini 2.0 Flash.

**Request Body**:

```json
{
  "text": "Your text content here...",
  "max_length": 200,
  "style": "concise"
}
```

**Response**:

```json
{
  "summary": "Generated summary...",
  "original_length": 1500,
  "summary_length": 180,
  "compression_ratio": 0.12,
  "source_type": "text"
}
```

### POST /summarize-pdf

Summarize PDF files using Google Gemini 2.0 Flash.

**Request**: Multipart form data with PDF file upload

- `file`: PDF file (required)
- `max_length`: Maximum summary length in words (optional)
- `style`: Summary style - "concise", "detailed", or "bullet_points" (optional)

**Response**:

```json
{
  "summary": "Generated summary...",
  "original_length": 1500,
  "summary_length": 180,
  "compression_ratio": 0.12,
  "source_type": "pdf",
  "pdf_metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "pages_processed": 5,
    "total_pages": 5
  }
}
```

### POST /query

Answer questions about documents using RAG approach.

**Request Body**:

```json
{
  "question": "What is this document about?",
  "context": "Document content here...",
  "include_sources": false
}
```

**Response**:

```json
{
  "answer": "Generated answer...",
  "confidence": null,
  "sources": null,
  "source_type": "text"
}
```

### POST /query-pdf

Answer questions about PDF documents using RAG approach.

**Request**: Multipart form data with PDF file upload

- `file`: PDF file (required)
- `question`: Question to ask about the PDF (required)
- `include_sources`: Whether to include source references (optional)

**Response**:

```json
{
  "answer": "Generated answer...",
  "confidence": null,
  "sources": null,
  "source_type": "pdf",
  "pdf_metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "pages_processed": 5,
    "total_pages": 5
  }
}
```

### GET /health

Health check endpoint.

**Response**:

```json
{
  "status": "healthy",
  "service": "ai-document-qa-system"
}
```

## Configuration

### Environment Variables

| Variable         | Description                       | Default     |
| ---------------- | --------------------------------- | ----------- |
| `GOOGLE_API_KEY` | Google Gemini 2.0 Flash API key   | Required    |
| `FASTAPI_HOST`   | FastAPI server host               | `127.0.0.1` |
| `FASTAPI_PORT`   | FastAPI server port               | `8000`      |
| `FASTAPI_RELOAD` | Enable auto-reload in development | `true`      |
| `LOG_LEVEL`      | Logging level                     | `INFO`      |

### Customization

- **Prompt Templates**: Modify templates in `backend/prompt_templates.py`
- **Model Settings**: Adjust Gemini 2.0 Flash parameters in `backend/config.py`
- **UI Configuration**: Customize the interface in `frontend/config.py`

## Development

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=backend --cov=frontend
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8

# Type checking
mypy backend/ frontend/
```

### Project Structure

```
ai-document-qa-system/
├── backend/                 # FastAPI backend
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── config.py           # Configuration settings
│   ├── models.py           # Pydantic models
│   ├── gemini_client.py    # Google Gemini 2.0 Flash client
│   └── prompt_templates.py # AI prompt templates
├── frontend/               # Streamlit frontend
│   ├── __init__.py
│   ├── app.py              # Main Streamlit app
│   ├── config.py           # Frontend configuration
│   └── api_client.py       # API client for backend
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_api.py         # API tests
│   └── test_prompt_templates.py
├── app.py                  # Streamlit entry point
├── run_backend.py          # Backend server script
├── pyproject.toml          # Project configuration
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Usage Guide

### 📝 Text Summarization

1. Go to the "📝 Summarization" tab
2. Choose input method:
   - **Text Input**: Paste your text directly
   - **PDF Upload**: Upload a PDF file (max 10MB)
3. Select summary style and length options
4. Click "Generate Summary"

### ❓ Document Q&A

1. Go to the "❓ Q&A" tab
2. Choose document input method:
   - **Text Input**: Paste your document content
   - **PDF Upload**: Upload a PDF file (max 10MB)
3. Enter your question about the document
4. Optionally enable "Include Sources" for source references
5. Click "Get Answer"

### 📄 PDF Support

- **Supported Files**: All standard PDF files with extractable text
- **File Size Limit**: 10MB maximum
- **Text Extraction**: Automatic text extraction with metadata
- **Metadata Display**: View PDF title, author, page count, and creation date

## Deployment

### Docker (Coming Soon)

A Docker configuration will be added for easy deployment.

### Production Considerations

- Set `FASTAPI_RELOAD=false` in production
- Use a production WSGI server like Gunicorn
- Configure proper logging and monitoring
- Set up rate limiting and authentication as needed
- Use environment-specific configuration files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.
