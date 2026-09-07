"""Node for analyzing low-confidence mentions and proposing new topics."""

from __future__ import annotations

import random

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from nodes.classify_mentions import run_classification
from prompts import NEW_TOPIC_PROPOSAL_PROMPT
from state import State
from utils import (
    invoke_taxonomy_chain,
    load_chat_model,
    parse_taxa,
    save_mention_classifications,
    save_taxonomy,
)

# Below this many low-confidence mentions, a new topic is unlikely to be a real pattern.
MIN_MENTIONS_FOR_NEW_TOPIC = 5
# Sample size used to detect a new-topic pattern, not the full ambiguous pile.
MAX_MENTIONS_FOR_PROPOSAL = 100

# Zero: this is a judgment call on an existing pattern, not a creative task.
BASE_TEMPERATURE = 0.0


def _setup_proposal_chain(
    configuration: Configuration, taxonomy: list[str], temperature: float = BASE_TEMPERATURE
):
    """Build the new-topic proposal prompt -> model -> parser chain."""
    prompt = NEW_TOPIC_PROPOSAL_PROMPT.partial(
        taxonomy="\n".join(f"- {t}" for t in taxonomy)
    )
    # Uses the stronger model (configuration.model): this call is rare enough
    # that the extra quality is worth the cost.
    model = load_chat_model(configuration.model, temperature=temperature)

    return (prompt | model | StrOutputParser() | parse_taxa).with_config(
        run_name="ProposeNewTopics"
    )


async def propose_new_topics(state: State, config: RunnableConfig) -> dict:
    """Look for a coherent pattern in low-confidence mentions and propose new topics.

    If new topics are found, every low-confidence mention is reclassified
    against the updated taxonomy, not just the sample used to detect it.
    """
    if len(state.low_confidence_mentions) < MIN_MENTIONS_FOR_NEW_TOPIC:
        return {
            "status": [
                f"Doar {len(state.low_confidence_mentions)} mentiuni cu incredere "
                "scazuta - insuficient pentru a propune topicuri noi."
            ],
        }

    configuration = Configuration.from_runnable_config(config)
    taxonomy = state.taxonomy or state.existing_taxonomy or []

    remaining_capacity = configuration.max_num_clusters - len(taxonomy)
    if remaining_capacity <= 0:
        return {
            "status": [
                f"Taxonomia e deja la limita de {configuration.max_num_clusters} "
                "topicuri - nu se propun altele noi."
            ],
        }

    sample_size = min(MAX_MENTIONS_FOR_PROPOSAL, len(state.low_confidence_mentions))
    sample = random.sample(state.low_confidence_mentions, sample_size)

    def chain_factory(temp_bump: float):
        return _setup_proposal_chain(configuration, taxonomy, temperature=BASE_TEMPERATURE + temp_bump)

    proposal_result = await invoke_taxonomy_chain(
        chain_factory,
        sample,
        max_topics=remaining_capacity,
        allow_empty=True,
    )
    new_topics = proposal_result["taxonomy"]

    if not new_topics:
        return {
            "status": [
                f"Niciun topic nou coerent gasit (analizat un esantion de "
                f"{sample_size} din {len(state.low_confidence_mentions)} mentiuni ambigue)."
            ],
        }

    updated_taxonomy = taxonomy + new_topics
    save_taxonomy(state.project_id, updated_taxonomy)

    reclassified, still_low = await run_classification(
        state.low_confidence_mentions, updated_taxonomy, configuration
    )

    status = [
        f"Propuse {len(new_topics)} topic(uri) noi: {', '.join(new_topics)}",
        f"Reclasificate {len(reclassified)} din {len(state.low_confidence_mentions)} "
        "mentiuni ambigue, folosind taxonomia actualizata.",
    ]

    # Overwrite the earlier (lower-confidence) DB row for each reclassified
    # mention. The cursor already advanced in classify_mentions, so a
    # failure here is reported but doesn't block anything further.
    try:
        save_mention_classifications(reclassified + still_low)
    except Exception as exc:
        status.append(f"Nu am putut actualiza clasificarile in baza de date: {exc}")

    # Swap in only the reclassified entries; leave everything else as-is.
    reclassified_by_id = {m.id: m for m in reclassified + still_low}
    final_mentions = [reclassified_by_id.get(m.id, m) for m in state.mentions]

    return {
        "mentions": final_mentions,
        "taxonomy": updated_taxonomy,
        "new_topics_proposed": new_topics,
        "low_confidence_mentions": still_low,
        "status": status,
    }
