"""Prompt templates for different AI operations."""

from typing import Optional
from jinja2 import Template


# Summarization prompt templates
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
- Include all major points and important details
- Organize information logically
- Maintain the structure and flow of the original text
- Use clear, professional language
- Include relevant examples or data points mentioned
{% if max_length %}
- Keep the summary to approximately {{ max_length }} words or less
{% endif %}

Text to summarize:
{{ text }}

Detailed Summary:""",

    "bullet_points": """
You are an expert text summarizer. Your task is to create a bullet-point summary of the provided text.

Instructions:
- Extract key points and present them as bullet points
- Use clear, concise language for each point
- Organize points logically (chronologically or by importance)
- Each bullet point should be self-contained and meaningful
{% if max_length %}
- Limit to approximately {{ max_length }} words total across all bullet points
{% endif %}

Text to summarize:
{{ text }}

Bullet Point Summary:
•""",
}


# RAG (Retrieval-Augmented Generation) prompt template
RAG_TEMPLATE = """
You are an AI assistant specialized in answering questions based on provided context. Your task is to analyze the given context and provide accurate, helpful answers to user questions.

Instructions:
- Base your answer strictly on the information provided in the context
- If the context doesn't contain enough information to answer the question, clearly state this
- Be precise and factual in your responses
- Use direct quotes from the context when appropriate
- If asked about something not covered in the context, explain that the information is not available
{% if include_sources %}
- Include references to specific parts of the context that support your answer
{% endif %}

Context:
{{ context }}

Question: {{ question }}

{% if include_sources %}
Please provide your answer and include source references where applicable.

Answer:
{% else %}
Answer:
{% endif %}"""


# Question-answering without context template
QA_TEMPLATE = """
You are a helpful AI assistant. Please answer the following question to the best of your ability.

Question: {{ question }}

Answer:"""


def get_summarization_prompt(
    text: str,
    max_length: Optional[int] = None,
    style: str = "concise"
) -> str:
    """
    Generate a summarization prompt based on the specified style.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary in words
        style: Summary style ('concise', 'detailed', 'bullet_points')
        
    Returns:
        Formatted prompt string
    """
    if style not in SUMMARIZATION_TEMPLATES:
        style = "concise"
    
    template = Template(SUMMARIZATION_TEMPLATES[style])
    return template.render(text=text, max_length=max_length)


def get_rag_prompt(
    question: str,
    context: str,
    include_sources: bool = False
) -> str:
    """
    Generate a RAG (Retrieval-Augmented Generation) prompt.
    
    Args:
        question: The question to answer
        context: The context/document content
        include_sources: Whether to include source references
        
    Returns:
        Formatted prompt string
    """
    template = Template(RAG_TEMPLATE)
    return template.render(
        question=question,
        context=context,
        include_sources=include_sources
    )


def get_qa_prompt(question: str) -> str:
    """
    Generate a simple question-answering prompt.
    
    Args:
        question: The question to answer
        
    Returns:
        Formatted prompt string
    """
    template = Template(QA_TEMPLATE)
    return template.render(question=question)


# Custom prompt template function for advanced use cases
def create_custom_prompt(
    template_string: str,
    **kwargs
) -> str:
    """
    Create a custom prompt from a template string.
    
    Args:
        template_string: Jinja2 template string
        **kwargs: Variables to substitute in the template
        
    Returns:
        Formatted prompt string
    """
    template = Template(template_string)
    return template.render(**kwargs)
