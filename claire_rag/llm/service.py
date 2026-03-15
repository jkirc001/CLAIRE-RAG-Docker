"""High-level LLM service for RAG."""

import logging
from pathlib import Path
from typing import Any

import yaml

from claire_rag.llm.client import LLMClient, get_llm_client
from claire_rag.llm.prompts import build_context_prompt
from claire_rag.ranking import Ranker
from claire_rag.retrieval import retrieve, retrieve_and_rank

logger = logging.getLogger(__name__)


def _load_ranker_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load ranker configuration from settings.yaml."""
    if config_dir is None:
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"

    settings_path = Path(config_dir) / "settings.yaml"
    if not settings_path.exists():
        # Return defaults if config file doesn't exist
        return {"enabled": True, "retrieve_k": 50, "rank_k": 10}

    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f) or {}

    ranker_config = settings.get("ranker", {})
    return {
        "enabled": ranker_config.get("enabled", True),
        "retrieve_k": ranker_config.get("retrieve_k", 50),
        "rank_k": ranker_config.get("rank_k", 10),
    }


def generate_answer(
    prompt: str,
    llm_client: LLMClient | None = None,
) -> str:
    """
    Generate an answer from a prompt using the configured LLM.

    Args:
        prompt: Input prompt
        llm_client: Optional LLM client instance. If None, creates a new one.

    Returns:
        Generated answer text
    """
    if llm_client is None:
        llm_client = get_llm_client()

    return llm_client.generate(prompt)


def answer_question(
    query: str,
    k: int = 10,
    dataset: str | None = None,
    llm_client: LLMClient | None = None,
    vector_store=None,
    use_ranker: bool | None = None,
    retrieve_k: int | None = None,
    rank_k: int | None = None,
    ranker: Ranker | None = None,
    config_dir: Path | None = None,
    debug: bool = False,
) -> dict:
    """
    Answer a question using RAG pipeline (3-component: Retriever, Ranker, Generator).

    Args:
        query: User question
        k: Number of chunks to retrieve (used when use_ranker=False, deprecated)
        dataset: Optional dataset filter
        llm_client: Optional LLM client instance. If None, creates a new one.
        vector_store: Optional vector store instance. If None, uses default.
        use_ranker: If True, use two-stage retrieval with re-ranking. If None, uses config default.
        retrieve_k: Number of chunks to retrieve before ranking. If None, uses config default.
        rank_k: Number of chunks to return after ranking. If None, uses config default.
        ranker: Optional Ranker instance. If None and use_ranker=True, uses default.
        config_dir: Optional config directory path. If None, uses default.
        debug: If True, enable verbose logging for component execution.

    Returns:
        Dictionary with:
        - "question": The original query
        - "answer": Generated answer
        - "chunks": List of retrieved Chunk objects (for provenance)
        - "pipeline_info": Dictionary with component execution info (for debug)
    """
    # Load ranker configuration
    ranker_config = _load_ranker_config(config_dir)

    # Determine if ranker should be used
    if use_ranker is None:
        use_ranker = ranker_config["enabled"]

    # Use config defaults if not explicitly provided
    if retrieve_k is None:
        retrieve_k = ranker_config["retrieve_k"]
    if rank_k is None:
        # If ranker is enabled and k was provided (but rank_k wasn't), use k as rank_k
        # This ensures --k flag works correctly when ranker is enabled
        rank_k = k if use_ranker else ranker_config["rank_k"]
    # Ensure retrieve_k is at least as large as rank_k
    if use_ranker and retrieve_k < rank_k:
        retrieve_k = rank_k

    # Track pipeline components for debug output
    pipeline_info = {
        "components_used": [],
        "retriever": {},
        "ranker": {},
        "generator": {},
    }

    # Component 1: RETRIEVER
    if debug:
        logger.info("🔍 [RETRIEVER] Starting vector search...")
    pipeline_info["components_used"].append("Retriever")

    if use_ranker:
        # Two-stage: Retrieve candidates first
        from claire_rag.retrieval import retrieve as retrieve_candidates

        candidate_chunks = retrieve_candidates(
            query=query,
            k=retrieve_k,
            dataset=dataset,
            vector_store=vector_store,
        )
        pipeline_info["retriever"] = {
            "enabled": True,
            "candidates_retrieved": len(candidate_chunks),
            "method": "vector_search",
        }
        if debug:
            logger.info(
                f"🔍 [RETRIEVER] Retrieved {len(candidate_chunks)} candidate chunks"
            )

        # Component 2: RANKER
        if debug:
            logger.info("📊 [RANKER] Re-ranking candidates...")
        pipeline_info["components_used"].append("Ranker")
        chunks = retrieve_and_rank(
            query=query,
            retrieve_k=retrieve_k,
            rank_k=rank_k,
            dataset=dataset,
            vector_store=vector_store,
            ranker=ranker,
            debug=debug,
        )
        pipeline_info["ranker"] = {
            "enabled": True,
            "candidates_ranked": len(candidate_chunks),
            "results_returned": len(chunks),
            "method": "cross_encoder",
        }
        if debug:
            logger.info(f"📊 [RANKER] Ranked to top {len(chunks)} chunks")
    else:
        chunks = retrieve(query, k=k, dataset=dataset, vector_store=vector_store)
        pipeline_info["retriever"] = {
            "enabled": True,
            "candidates_retrieved": len(chunks),
            "method": "vector_search",
        }
        pipeline_info["ranker"] = {"enabled": False}
        if debug:
            logger.info(f"🔍 [RETRIEVER] Retrieved {len(chunks)} chunks (no ranking)")

    # Build context prompt
    prompt = build_context_prompt(query, chunks)

    # Component 3: GENERATOR
    if debug:
        logger.info("🤖 [GENERATOR] Generating answer with LLM...")
    pipeline_info["components_used"].append("Generator")

    # Generate answer
    if llm_client is None:
        llm_client = get_llm_client()

    # Check if we need usage info (for debug mode)
    # We'll detect this by checking if generate supports return_usage
    try:
        answer, usage = llm_client.generate(prompt, return_usage=True)
        pipeline_info["generator"] = {
            "enabled": True,
            "model": llm_client.model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if debug:
            logger.info(f"🤖 [GENERATOR] Generated answer using {llm_client.model}")
        return {
            "question": query,
            "answer": answer,
            "chunks": chunks,
            "prompt": prompt,
            "usage": usage,
            "pipeline_info": pipeline_info,
        }
    except TypeError:
        # Fallback for stub mode or older clients
        answer = llm_client.generate(prompt)
        pipeline_info["generator"] = {
            "enabled": True,
            "model": llm_client.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if debug:
            logger.info(
                f"🤖 [GENERATOR] Generated answer using {llm_client.model} (stub mode)"
            )
        return {
            "question": query,
            "answer": answer,
            "chunks": chunks,
            "prompt": prompt,
            "usage": None,
            "pipeline_info": pipeline_info,
        }
