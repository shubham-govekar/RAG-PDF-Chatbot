import os

from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

GROQ_FAST_MODEL = "openai/gpt-oss-20b"
GROQ_BIG_MODEL = "openai/gpt-oss-120b"

NVIDIA_FAST_MODEL = "openai/gpt-oss-20b"
NVIDIA_BIG_MODEL = "openai/gpt-oss-120b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _nvidia_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Initializes the NVIDIA NIM fallback client."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Required as the fallback provider."
        )
    return ChatOpenAI(
        model=model,
        base_url=NVIDIA_BASE_URL,
        api_key=SecretStr(api_key),
        temperature=temperature,
        max_retries=0,
    )


def _resilient(groq_model: str, nvidia_model: str, temperature: float = 0.0):
    """
    Creates a resilient LLM client with Groq as primary and NVIDIA as fallback.
    Disables internal retries to ensure immediate fail-over during rate limits.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set."
        )

    groq_llm = ChatGroq(model=groq_model, temperature=temperature, max_retries=0)
    nvidia_llm = _nvidia_llm(nvidia_model, temperature=temperature)

    return groq_llm.with_fallbacks([nvidia_llm])


def get_fast_llm(temperature: float = 0.0):
    """Returns the fast-tier LLM for lightweight routing and query reformulation."""
    return _resilient(GROQ_FAST_MODEL, NVIDIA_FAST_MODEL, temperature)


def get_big_llm(temperature: float = 0.0):
    """Returns the big-tier LLM for complex reasoning, generation, and summarization."""
    return _resilient(GROQ_BIG_MODEL, NVIDIA_BIG_MODEL, temperature)