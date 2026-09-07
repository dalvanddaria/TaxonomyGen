"""Node for reviewing and finalizing the taxonomy on a larger, fresh sample."""

from __future__ import annotations

import random

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from prompts import TAXONOMY_REVIEW_PROMPT
from state import State
from utils import invoke_taxonomy_chain, load_chat_model, parse_taxa, save_taxonomy

# Zero: this is the final pass, and should be consistent rather than exploratory.
BASE_TEMPERATURE = 0.0


def _setup_review_chain(configuration: Configuration, temperature: float = BASE_TEMPERATURE):
    """Build the review prompt -> model -> parser chain."""
    review_prompt = TAXONOMY_REVIEW_PROMPT.partial(
        max_num_clusters=configuration.max_num_clusters,
    )
    model = load_chat_model(configuration.fast_llm, temperature=temperature)

    return (review_prompt | model | StrOutputParser() | parse_taxa).with_config(
        run_name="ReviewTaxonomy"
    )


async def review_taxonomy(state: State, config: RunnableConfig) -> dict:
    """Check the taxonomy against a fresh sample of the whole corpus, then save it."""
    configuration = Configuration.from_runnable_config(config)

    indices = list(range(len(state.all_mentions)))
    random.shuffle(indices)
    sample_indices = indices[: configuration.taxonomy_batch_size]
    sample_mentions = [state.all_mentions[i] for i in sample_indices]

    # Review the best candidate found during refinement, not just the latest one.
    taxonomy_to_review = state.best_taxonomy or state.taxonomy
    current_taxonomy_text = "\n".join(f"- {t}" for t in taxonomy_to_review)

    def chain_factory(temp_bump: float):
        return _setup_review_chain(configuration, temperature=BASE_TEMPERATURE + temp_bump)

    result = await invoke_taxonomy_chain(
        chain_factory,
        sample_mentions,
        max_topics=configuration.max_num_clusters,
        allow_empty=False,
        current_taxonomy=current_taxonomy_text,
    )

    save_taxonomy(state.project_id, result["taxonomy"])
    result["status"].append(f"Taxonomy saved for project '{state.project_id}'.")

    return result
