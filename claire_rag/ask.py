"""CLI for asking questions to the RAG system."""

import logging
import os
import sys
import warnings
from pathlib import Path

import typer

from claire_rag.llm import answer_question, get_llm_client

app = typer.Typer()
logger = logging.getLogger(__name__)


def _configure_logging(debug: bool = False) -> None:
    """
    Configure logging levels for the application and external libraries.

    Args:
        debug: If True, show INFO logs. If False, suppress external library INFO logs.
    """
    if not debug:
        # Suppress INFO logs from external libraries in normal mode
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(
            logging.WARNING
        )
        logging.getLogger("sentence_transformers.cross_encoder").setLevel(
            logging.WARNING
        )
        logging.getLogger("sentence_transformers.cross_encoder.CrossEncoder").setLevel(
            logging.WARNING
        )
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        # Suppress tokenizers warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Suppress Python warnings (FutureWarning, etc.)
        warnings.filterwarnings("ignore")
    else:
        # In debug mode, show INFO logs
        logging.getLogger("sentence_transformers").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
        # Show warnings in debug mode
        warnings.filterwarnings("default")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask"),
    k: int = typer.Option(
        10,
        "--k",
        help="Number of chunks to return (maps to rank_k when ranker is enabled, or k when ranker is disabled)",
    ),
    dataset: str = typer.Option(
        None,
        "--dataset",
        help="Filter by dataset. Available: CVE, CWE, CAPEC, ATTACK, NICE, DCWF",
    ),
    eval_mode: bool = typer.Option(
        False,
        "--eval",
        help="Enable evaluation mode (forces gpt-4o for LLM)",
    ),
    show_chunks: bool = typer.Option(
        False,
        "--show-chunks",
        help="Show retrieved chunks for debugging",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Use stub LLM mode (no API calls, free testing)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show debug information (chunks, token usage, cost estimates)",
    ),
    no_ranker: bool = typer.Option(
        False,
        "--no-ranker",
        help="Disable cross-encoder re-ranking (ranker is enabled by default)",
    ),
    retrieve_k: int = typer.Option(
        None,
        "--retrieve-k",
        help="Number of chunks to retrieve before ranking (default: 50 from config)",
    ),
    rank_k: int = typer.Option(
        None,
        "--rank-k",
        help="Number of chunks to return after ranking (default: 10 from config)",
    ),
) -> None:
    """
    Ask a question to the RAG system using 3-component architecture:
    1. Retriever - Fast vector search for candidate chunks
    2. Ranker - Cross-encoder re-ranking for improved relevance
    3. Generator - LLM answer generation

    The ranker is enabled by default. Use --no-ranker to disable it.
    """
    # Configure logging based on debug mode
    _configure_logging(debug=debug)

    if eval_mode:
        os.environ["CLAIRE_ENV"] = "evaluation"
        logger.info("Evaluation mode enabled. LLM model will be set to gpt-4o.")

    # Determine ranker usage (default: enabled, unless --no-ranker is set)
    use_ranker = not no_ranker

    # Map --k to rank_k when ranker is enabled, or use as-is when disabled
    if use_ranker:
        # When ranker is enabled, --k should set rank_k (unless --rank-k is explicitly set)
        if rank_k is None:
            rank_k = k
        # Ensure retrieve_k is at least as large as rank_k
        if retrieve_k is None:
            # Default retrieve_k from config, but ensure it's >= rank_k
            from claire_rag.llm.service import _load_ranker_config

            ranker_config = _load_ranker_config()
            default_retrieve_k = ranker_config["retrieve_k"]
            retrieve_k = max(default_retrieve_k, rank_k)
        elif retrieve_k < rank_k:
            # If user explicitly set retrieve_k but it's less than rank_k, increase it
            retrieve_k = rank_k

    # Only show verbose logs in debug mode
    if debug:
        logger.info(f"Question: {question}")
        if use_ranker:
            logger.info(
                f"Using ranker (3-component RAG): retrieve_k={retrieve_k}, rank_k={rank_k}"
            )
        else:
            logger.info("Ranker disabled (2-component RAG: Retriever + Generator)")

    # Check if vector store exists
    vectorstore_path = Path("./vectorstore")
    if not vectorstore_path.exists():
        typer.echo(
            "Error: Vector store not found. Please run 'python -m claire_rag.index build' first.",
            err=True,
        )
        sys.exit(1)

    # Get LLM client
    try:
        if stub:
            if debug:
                logger.info("Using stub mode - no API calls will be made")
            llm_client = get_llm_client(use_stub=True)
        else:
            llm_client = get_llm_client()
    except Exception as e:
        typer.echo(f"Error initializing LLM client: {e}", err=True)
        if not stub:
            typer.echo(
                "Make sure you have set OPENAI_API_KEY in your .env file.", err=True
            )
        sys.exit(1)

    # Answer the question
    try:
        result = answer_question(
            query=question,
            k=k,
            dataset=dataset,
            llm_client=llm_client,
            use_ranker=use_ranker,
            retrieve_k=retrieve_k,
            rank_k=rank_k,
            debug=debug,
        )

        # Display answer
        typer.echo("\n" + "=" * 80)
        typer.echo("ANSWER")
        typer.echo("=" * 80)
        typer.echo(result["answer"])
        typer.echo("=" * 80)

        # Show debug info if requested
        if debug:
            # Show 3-component RAG pipeline information
            pipeline_info = result.get("pipeline_info", {})

            typer.echo("\n" + "=" * 80)
            typer.echo("3-COMPONENT RAG PIPELINE")
            typer.echo("=" * 80)

            # Component 1: Retriever
            retriever_info = pipeline_info.get("retriever", {})
            typer.echo("\n[1] 🔍 RETRIEVER")
            typer.echo("    Status: Enabled")
            typer.echo("    Method: Vector search (embedding similarity)")
            if retriever_info.get("candidates_retrieved"):
                typer.echo(
                    f"    Candidates retrieved: {retriever_info['candidates_retrieved']}"
                )

            # Component 2: Ranker
            ranker_info = pipeline_info.get("ranker", {})
            if ranker_info.get("enabled"):
                typer.echo("\n[2] 📊 RANKER")
                typer.echo("    Status: Enabled")
                typer.echo("    Method: Cross-encoder re-ranking")
                if ranker_info.get("candidates_ranked"):
                    typer.echo(
                        f"    Candidates ranked: {ranker_info['candidates_ranked']}"
                    )
                if ranker_info.get("results_returned"):
                    typer.echo(
                        f"    Results returned: {ranker_info['results_returned']}"
                    )
            else:
                typer.echo("\n[2] 📊 RANKER")
                typer.echo("    Status: Disabled (skipped)")

            # Component 3: Generator
            generator_info = pipeline_info.get("generator", {})
            typer.echo("\n[3] 🤖 GENERATOR")
            typer.echo("    Status: Enabled")
            if generator_info.get("model"):
                typer.echo(f"    Model: {generator_info['model']}")
            if generator_info.get("total_tokens"):
                typer.echo(f"    Tokens used: {generator_info['total_tokens']:,}")

            typer.echo("\n" + "=" * 80)

            # Show prompt if available
            if result.get("prompt"):
                typer.echo("\n" + "=" * 80)
                typer.echo("PROMPT SENT TO LLM")
                typer.echo("=" * 80)
                typer.echo(result["prompt"])
                typer.echo("=" * 80)

            # Show cost information
            usage = result.get("usage")
            if usage and not stub:
                typer.echo("\n" + "=" * 80)
                typer.echo("COST INFORMATION")
                typer.echo("=" * 80)

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                typer.echo(f"Model: {llm_client.model}")
                typer.echo(f"Prompt tokens: {prompt_tokens:,}")
                typer.echo(f"Completion tokens: {completion_tokens:,}")
                typer.echo(f"Total tokens: {total_tokens:,}")

                # Calculate cost based on model (pricing as of 2024)
                # gpt-4o-mini: $0.15/$0.60 per 1M tokens (input/output)
                # gpt-4o: $2.50/$10.00 per 1M tokens (input/output)
                if "gpt-4o-mini" in llm_client.model:
                    input_cost_per_1m = 0.15
                    output_cost_per_1m = 0.60
                elif "gpt-4o" in llm_client.model:
                    input_cost_per_1m = 2.50
                    output_cost_per_1m = 10.00
                else:
                    # Default to gpt-4o-mini pricing
                    input_cost_per_1m = 0.15
                    output_cost_per_1m = 0.60

                input_cost = (prompt_tokens / 1_000_000) * input_cost_per_1m
                output_cost = (completion_tokens / 1_000_000) * output_cost_per_1m
                total_cost = input_cost + output_cost

                typer.echo("\nEstimated cost:")
                typer.echo(f"  Input:  ${input_cost:.6f}")
                typer.echo(f"  Output: ${output_cost:.6f}")
                typer.echo(f"  Total:  ${total_cost:.6f}")
            elif stub:
                typer.echo("\n" + "=" * 80)
                typer.echo("COST INFORMATION")
                typer.echo("=" * 80)
                typer.echo("Stub mode - No API calls made (Cost: $0.00)")
            else:
                typer.echo("\n" + "=" * 80)
                typer.echo("COST INFORMATION")
                typer.echo("=" * 80)
                typer.echo("Usage information not available")

        # Show chunks if requested or in debug mode
        if show_chunks or debug:
            typer.echo("\n" + "=" * 80)
            typer.echo(f"RETRIEVED CHUNKS ({len(result['chunks'])} chunks)")
            typer.echo("=" * 80)
            for idx, chunk in enumerate(result["chunks"], start=1):
                typer.echo(f"\n[{idx}] {chunk.dataset} {chunk.source_id}")
                typer.echo(f"Document: {chunk.document_id}")
                typer.echo(f"Text: {chunk.text[:200]}...")
                if chunk.metadata.get("distance"):
                    typer.echo(f"Distance: {chunk.metadata['distance']:.4f}")

    except Exception as e:
        logger.exception("Error answering question")
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app()
