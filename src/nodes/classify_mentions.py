"""Node for classifying mentions against a taxonomy.

Used in both modes:
- cold_start: after review_taxonomy, with the freshly generated taxonomy
- incremental: with the existing taxonomy
"""

from __future__ import annotations

from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from prompts import CLASSIFICATION_PROMPT
from state import Mention, State
from utils import (
    format_mentions,
    load_chat_model,
    parse_classifications,
    save_last_processed_id,
    save_mention_classifications,
)


def _setup_classification_chain(configuration: Configuration, taxonomy: list[str]):
    """Build the classification prompt -> model -> parser chain for a fixed taxonomy."""
    prompt = CLASSIFICATION_PROMPT.partial(
        taxonomy="\n".join(f"- {t}" for t in taxonomy)
    )
    # max_tokens: response is one JSON object per mention in the batch.
    # temperature=0: this is a decision, consistency is more important than exploration. The model should be deterministic.
    model = load_chat_model(configuration.fast_llm, max_tokens=4096, temperature=0)

    return (prompt | model | StrOutputParser() | parse_classifications).with_config(
        run_name="ClassifyMentions"
    )


def _coerce_confidence(value) -> Optional[float]:
    """Coerce a confidence value to float, tolerating models that return it as a string."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_classification(mention: Mention, result: dict) -> Mention:
    """Return a new Mention with a classification result applied."""
    return Mention(
        id=mention.id,
        content=mention.content,
        url=mention.url,
        category=result.get("categories", []),
        primary_category=result.get("primary_category"),
        explanation=result.get("explanation"),
        confidence=_coerce_confidence(result.get("confidence")),
    )


def _mark_as_unclassified(mention: Mention, reason: str) -> Mention:
    """Return a copy of the mention marked unclassified, so it's tracked instead of dropped."""
    return Mention(
        id=mention.id,
        content=mention.content,
        url=mention.url,
        category=[],
        primary_category=None,
        explanation=reason,
        confidence=0.0,
    )


async def run_classification(
    mentions: list[Mention],
    taxonomy: list[str],
    configuration: Configuration,
) -> tuple[list[Mention], list[Mention]]:
    """Classify a list of mentions against a taxonomy.

    Returns (classified, low_confidence). A failed batch or a missing
    per-mention result doesn't drop the mention - it's marked unclassified
    and returned in low_confidence for review.
    """
    if not mentions:
        return [], []

    chain = _setup_classification_chain(configuration, taxonomy)

    batch_size = configuration.classification_batch_size
    batches = [
        mentions[i : i + batch_size] for i in range(0, len(mentions), batch_size)
    ]
    batch_inputs = [{"documents": format_mentions(b)} for b in batches]

    batch_results = await chain.abatch(
        batch_inputs,
        config={"max_concurrency": configuration.max_concurrency},
        return_exceptions=True,
    )

    classified: list[Mention] = []
    low_confidence: list[Mention] = []

    for batch, results in zip(batches, batch_results):
        if isinstance(results, Exception):
            for mention in batch:
                low_confidence.append(
                    _mark_as_unclassified(mention, f"Eroare la clasificare: {results}")
                )
            continue

        results_by_id = {str(r.get("id")): r for r in results}

        for mention in batch:
            result = results_by_id.get(mention.id)
            if result is None:
                low_confidence.append(
                    _mark_as_unclassified(mention, "Lipsa din raspunsul LLM.")
                )
                continue

            updated = _apply_classification(mention, result)
            confidence = updated.confidence or 0.0

            if confidence >= configuration.confidence_threshold:
                classified.append(updated)
            else:
                low_confidence.append(updated)

    return classified, low_confidence


async def classify_mentions(state: State, config: RunnableConfig) -> dict:
    """Classify every mention in the corpus against the taxonomy currently available."""
    configuration = Configuration.from_runnable_config(config)
    taxonomy = state.taxonomy or state.existing_taxonomy or []

    if not taxonomy:
        results = [
            _mark_as_unclassified(m, "Nicio taxonomie disponibila la clasificare.")
            for m in state.all_mentions
        ]
        low_confidence = results
        status = [
            "Clasificare sarita: nu exista nicio taxonomie "
            f"(nici existenta, nici generata) pentru {len(state.all_mentions)} mentiuni."
        ]
    else:
        classified, low_confidence = await run_classification(
            state.all_mentions, taxonomy, configuration
        )
        results = classified + low_confidence
        status = [
            f"Classified {len(classified)} mentions with confidence >= "
            f"{configuration.confidence_threshold}; {len(low_confidence)} need review "
            f"(total: {len(results)} of {len(state.all_mentions)} fetched)."
        ]

    # Persist results to the mentions table before advancing the cursor, so
    # a DB write failure leaves these mentions eligible to be refetched and
    # retried next run instead of being skipped forever.
    if state.next_cursor:
        try:
            save_mention_classifications(results)
            save_last_processed_id(state.project_id, state.next_cursor)
        except Exception as exc:
            status.append(
                f"Nu am putut salva clasificarile in baza de date - cursorul "
                f"NU a avansat, se reincearca la urmatoarea rulare: {exc}"
            )

    return {
        "mentions": results,
        "low_confidence_mentions": low_confidence,
        "taxonomy": taxonomy,
        "status": status,
    }
