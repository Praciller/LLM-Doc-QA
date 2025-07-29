"""Tests for prompt templates."""

import pytest
from backend.prompt_templates import (
    get_summarization_prompt,
    get_rag_prompt,
    get_qa_prompt,
    create_custom_prompt
)


def test_get_summarization_prompt_concise():
    """Test concise summarization prompt generation."""
    text = "This is a test document."
    prompt = get_summarization_prompt(text, max_length=50, style="concise")
    
    assert text in prompt
    assert "concise" in prompt.lower()
    assert "50" in prompt


def test_get_summarization_prompt_detailed():
    """Test detailed summarization prompt generation."""
    text = "This is a test document."
    prompt = get_summarization_prompt(text, style="detailed")
    
    assert text in prompt
    assert "detailed" in prompt.lower()


def test_get_summarization_prompt_bullet_points():
    """Test bullet points summarization prompt generation."""
    text = "This is a test document."
    prompt = get_summarization_prompt(text, style="bullet_points")
    
    assert text in prompt
    assert "bullet" in prompt.lower()


def test_get_rag_prompt():
    """Test RAG prompt generation."""
    question = "What is this about?"
    context = "This is a test document about AI."
    
    prompt = get_rag_prompt(question, context, include_sources=False)
    
    assert question in prompt
    assert context in prompt
    assert "Context:" in prompt
    assert "Question:" in prompt


def test_get_rag_prompt_with_sources():
    """Test RAG prompt generation with sources."""
    question = "What is this about?"
    context = "This is a test document about AI."
    
    prompt = get_rag_prompt(question, context, include_sources=True)
    
    assert question in prompt
    assert context in prompt
    assert "source" in prompt.lower()


def test_get_qa_prompt():
    """Test simple QA prompt generation."""
    question = "What is artificial intelligence?"
    prompt = get_qa_prompt(question)
    
    assert question in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt


def test_create_custom_prompt():
    """Test custom prompt creation."""
    template = "Hello {{ name }}, you are {{ age }} years old."
    prompt = create_custom_prompt(template, name="Alice", age=30)
    
    assert prompt == "Hello Alice, you are 30 years old."


def test_invalid_summarization_style():
    """Test invalid summarization style defaults to concise."""
    text = "This is a test document."
    prompt = get_summarization_prompt(text, style="invalid_style")
    
    # Should default to concise style
    assert text in prompt
    assert "concise" in prompt.lower()
