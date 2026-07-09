"""
Cloud LLM client factory for the RAG pipeline.

Two tiers:
  - FAST : cheap/low-latency model for lightweight tasks (detect_intent, reformulate_query)
  - BIG  : stronger model for tasks that need more reasoning (generate_answer, summarize_document)

Resilience: Groq is primary (fast, generous free tier). Each call retries with
exponential backoff on Groq first; if that's exhausted (e.g. free-tier rate
limit), it falls over to NVIDIA NIM once. This uses LangChain's built-in
Runnable.with_retry() / .with_fallbacks() rather than custom retry code, per
your "keep it simple" preference.

Model choice note (checked July 9, 2026): Groq deprecated
`llama-3.1-8b-instant` and `llama-3.3-70b-versatile`, shutdown date
2026-08-16 (https://console.groq.com/docs/deprecations). This file uses
their recommended replacements directly so the pipeline doesn't break
mid-project:
    openai/gpt-oss-20b   (replaces llama-3.1-8b-instant)
    openai/gpt-oss-120b  (replaces llama-3.3-70b-versatile)
Both are also present in NVIDIA NIM's free catalog under the same IDs, which
keeps the fallback pair model-consistent. Re-check
https://console.groq.com/docs/models and https://build.nvidia.com/models
before your final eval run in case the free-tier catalog has shifted again.

Env vars required (.env):
    GROQ_API_KEY=...
    NVIDIA_API_KEY=...   # nvapi-... from build.nvidia.com
"""
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
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set — required as the fallback provider. "
            "Get a free key at https://build.nvidia.com/models"
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
    Groq primary, NVIDIA NIM as fallback — fail fast, no internal retries.

    FIX: this previously stacked two retry layers — LangChain's
    .with_retry(stop_after_attempt=3, ...) on top of the Groq SDK's own
    automatic retry-on-429 (which honors the API's Retry-After header).
    On sustained rate limiting those compounded into long waits (visible
    as repeated "groq._base_client: Retrying request..." log lines with
    growing delays) before ever reaching the NVIDIA fallback. Disabling
    both retry layers (max_retries=0) means a 429 fails immediately and
    falls over to NVIDIA right away instead of sitting through backoff —
    much better for a synchronous Streamlit request under a tight free-tier
    quota.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
        )

    groq_llm = ChatGroq(model=groq_model, temperature=temperature, max_retries=0)
    nvidia_llm = _nvidia_llm(nvidia_model, temperature=temperature)

    return groq_llm.with_fallbacks([nvidia_llm])


def get_fast_llm(temperature: float = 0.0):
    """Small/fast model — use for detect_intent and reformulate_query."""
    return _resilient(GROQ_FAST_MODEL, NVIDIA_FAST_MODEL, temperature)


def get_big_llm(temperature: float = 0.0):
    """Larger model — use for generate_answer and summarize_document."""
    return _resilient(GROQ_BIG_MODEL, NVIDIA_BIG_MODEL, temperature)