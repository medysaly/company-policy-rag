from langchain_anthropic import ChatAnthropic

from app.config import settings


def get_llm():
    """Create and return the LLM."""
    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
    )
