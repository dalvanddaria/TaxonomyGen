"""Shared helpers: LLM client, taxonomy/cursor persistence, parsing, and retry logic."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Literal, Optional

import re

import mysql.connector
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from configuration import Configuration
from prompts import TAXONOMY_EVALUATION_PROMPT
from state import Mention

# --- Database connection ---


def get_connection():
    """Open a new MySQL connection using the DB_* environment variables."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
    )


# --- LLM client ---


def load_chat_model(
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Build a chat model routed through OpenRouter. model_name is "provider/model-name"."""
    kwargs = {
        "model": model_name,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)


# --- Building mention content from the SQL columns ---


def build_mention_content(
    title: Optional[str],
    text_before: Optional[str],
    text_keyword: Optional[str],
    text_after: Optional[str],
) -> str:
    """Combine the title and surrounding-text columns into one text block."""
    fragment = " ".join(
        part.strip() for part in [text_before, text_keyword, text_after] if part
    )
    return f"{(title or '').strip()}\n{fragment}".strip()


def row_to_mention(row: dict) -> Mention:
    """Convert a MySQL result row into a Mention."""
    content = build_mention_content(
        title=row.get("title"),
        text_before=row.get("text_before"),
        text_keyword=row.get("text_keyword"),
        text_after=row.get("text_after"),
    )
    return Mention(
        id=str(row.get("id")),
        content=content,
        url=row.get("url"),
    )


def save_mention_classifications(mentions: List[Mention]) -> None:
    """Write classification results back to the mentions table, keyed by id.

    Stores categories as JSON text (the column is plain text, not a native
    JSON type), matching the format used everywhere else in the pipeline.
    Raises on a connection/query failure; callers decide whether that
    should block a cursor advance.
    """
    if not mentions:
        return

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        UPDATE mentions
        SET primary_category = %s, categories = %s, confidence = %s, explanation = %s
        WHERE id = %s
    """
    params = [
        (
            m.primary_category,
            json.dumps(m.category, ensure_ascii=False) if m.category else None,
            m.confidence,
            m.explanation,
            int(m.id),
        )
        for m in mentions
    ]

    cursor.executemany(query, params)
    conn.commit()
    cursor.close()
    conn.close()


# --- Mode detection (cold_start vs incremental) ---

# Resolved relative to this file so it's stable regardless of the working directory.
TAXONOMY_STORE_PATH = Path(__file__).resolve().parent / "data" / "taxonomies.json"


def _load_taxonomy_store() -> dict:
    """Read the taxonomy store, returning an empty dict if it doesn't exist yet."""
    if not TAXONOMY_STORE_PATH.exists():
        return {}
    with open(TAXONOMY_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_taxonomy(project_id: str) -> Optional[List[str]]:
    """Return the saved taxonomy for a project, or None if it has none yet."""
    store = _load_taxonomy_store()
    return store.get(project_id)


def save_taxonomy(project_id: str, taxonomy: List[str]) -> None:
    """Persist a project's taxonomy, overwriting any previous version."""
    TAXONOMY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = _load_taxonomy_store()
    store[project_id] = taxonomy
    with open(TAXONOMY_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def determine_mode(project_id: str) -> Literal["cold_start", "incremental"]:
    """Return incremental if a taxonomy already exists for this project, else cold_start."""
    existing = load_existing_taxonomy(project_id)
    if existing:
        return "incremental"
    return "cold_start"


# --- Incremental fetch cursor (per project) ---

CURSOR_STORE_PATH = Path(__file__).resolve().parent / "data" / "cursors.json"


def _load_cursor_store() -> dict:
    """Read the cursor store, returning an empty dict if it doesn't exist yet."""
    if not CURSOR_STORE_PATH.exists():
        return {}
    with open(CURSOR_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_last_processed_id(project_id: str) -> int:
    """Return the highest mention id already processed for a project, or -1 if none yet."""
    store = _load_cursor_store()
    return int(store.get(project_id, -1))


def save_last_processed_id(project_id: str, last_id: int) -> None:
    """Advance a project's cursor to last_id. Never moves it backwards."""
    CURSOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = _load_cursor_store()
    store[project_id] = max(int(store.get(project_id, -1)), int(last_id))
    with open(CURSOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def reset_cursor(project_id: str) -> None:
    """Force a project's cursor back to -1, bypassing the normal never-backwards rule."""
    CURSOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = _load_cursor_store()
    store[project_id] = -1
    with open(CURSOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


# --- Taxonomy generation/refinement support ---


def format_mentions(mentions: List[Mention]) -> str:
    """Render a list of mentions as prompt-ready text, one id-tagged block per mention."""
    return "\n\n".join(f"[id={mention.id}] {mention.content}" for mention in mentions)


def parse_classifications(json_string: str) -> List[dict]:
    """Extract the classification JSON array from a model response, tolerating surrounding text."""
    start = json_string.find("[")
    end = json_string.rfind("]")

    if start == -1 or end == -1 or end < start:
        return []

    candidate = json_string[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return []


def parse_taxa(xml_string: str) -> List[str]:
    """Extract topic names from a <topics><topic>...</topic></topics> response."""
    matches = re.findall(r"<topic>(.*?)</topic>", xml_string, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def parse_evaluation_choice(xml_string: str) -> Optional[int]:
    """Extract the chosen variant (1 or 2) from an evaluation response."""
    match = re.search(r"<better_variant>\s*([12])\s*</better_variant>", xml_string)
    if not match:
        return None
    return int(match.group(1))


# Retry budget for malformed or over-limit taxonomy responses, and the
# temperature increment applied on each retry.
MAX_TAXONOMY_RETRIES = 5
RETRY_TEMPERATURE_STEP = 0.1


async def invoke_taxonomy_chain(
    chain_factory,
    mentions: List[Mention],
    *,
    max_topics: Optional[int] = None,
    allow_empty: bool = True,
    **extra_vars,
) -> dict:
    """Run a taxonomy chain, retrying at a higher temperature if the response is invalid.

    chain_factory(temperature_bump) builds a fresh chain for each attempt.
    allow_empty is False for generate/update/review (a taxonomy is required)
    and True for new-topic proposals, where finding nothing is a valid result.
    """
    documents_text = format_mentions(mentions)
    input_vars = {"documents": documents_text, **extra_vars}

    taxonomy: List[str] = []
    for attempt in range(MAX_TAXONOMY_RETRIES):
        chain = chain_factory(attempt * RETRY_TEMPERATURE_STEP)
        taxonomy = await chain.ainvoke(input_vars)

        format_ok = allow_empty or len(taxonomy) > 0
        size_ok = max_topics is None or len(taxonomy) <= max_topics
        if format_ok and size_ok:
            retry_note = f" (dupa {attempt} reincercari)" if attempt else ""
            return {
                "taxonomy": taxonomy,
                "status": [
                    f"Taxonomy step completed: {len(taxonomy)} topics, "
                    f"from {len(mentions)} mentions.{retry_note}"
                ],
            }

    # Retries exhausted: truncate to the cap rather than failing the run.
    if max_topics is not None and len(taxonomy) > max_topics:
        taxonomy = taxonomy[:max_topics]

    return {
        "taxonomy": taxonomy,
        "status": [
            f"Taxonomy step: raspuns LLM invalid/peste limita dupa "
            f"{MAX_TAXONOMY_RETRIES} incercari - folosim ultimul rezultat "
            f"({len(taxonomy)} topics, posibil trunchiat)."
        ],
    }


async def select_better_taxonomy(
    configuration: Configuration,
    taxonomy_a: List[str],
    taxonomy_b: List[str],
    validation_mentions: List[Mention],
    use_case: str,
) -> List[str]:
    """Return the better of two taxonomy candidates, evaluated on a held-out sample.

    Candidate order is randomized per call to cancel out position bias in
    the evaluating model. Falls back to taxonomy_b if there's no validation
    sample to evaluate against, or if the model's response can't be parsed.
    """
    if taxonomy_a == taxonomy_b:
        return taxonomy_a
    if not validation_mentions:
        return taxonomy_b

    swap = random.random() < 0.5
    variant_1, variant_2 = (taxonomy_b, taxonomy_a) if swap else (taxonomy_a, taxonomy_b)

    prompt = TAXONOMY_EVALUATION_PROMPT.partial(
        use_case=use_case,
        max_num_clusters=configuration.max_num_clusters,
        taxonomy_a="\n".join(f"- {t}" for t in variant_1),
        taxonomy_b="\n".join(f"- {t}" for t in variant_2),
    )
    model = load_chat_model(configuration.fast_llm, temperature=0)
    chain = (
        prompt | model | StrOutputParser() | parse_evaluation_choice
    ).with_config(run_name="EvaluateTaxonomy")

    documents_text = format_mentions(validation_mentions)
    choice = await chain.ainvoke({"documents": documents_text})

    if choice is None:
        return taxonomy_b

    mapping = (taxonomy_a, taxonomy_b) if not swap else (taxonomy_b, taxonomy_a)
    return mapping[0] if choice == 1 else mapping[1]
