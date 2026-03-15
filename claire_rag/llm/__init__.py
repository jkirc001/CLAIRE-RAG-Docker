"""LLM integration for RAG."""

from claire_rag.llm.client import LLMClient, get_llm_client
from claire_rag.llm.prompts import build_context_prompt
from claire_rag.llm.service import answer_question, generate_answer

__all__ = [
    "LLMClient",
    "get_llm_client",
    "build_context_prompt",
    "generate_answer",
    "answer_question",
]
