"""Node for updating taxonomies based on new minibatches (iterative refinement)."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from prompts import TAXONOMY_UPDATE_PROMPT
from state import State
from utils import (
    invoke_taxonomy_chain,
    load_chat_model,
    parse_taxa,
    select_better_taxonomy,
)

# Lower than generation: refining an existing taxonomy, not inventing one.
BASE_TEMPERATURE = 0.2


def _setup_update_chain(configuration: Configuration, temperature: float = BASE_TEMPERATURE):
    """Build the update prompt -> model -> parser chain."""
    update_prompt = TAXONOMY_UPDATE_PROMPT.partial(
        max_num_clusters=configuration.max_num_clusters,
    )
    model = load_chat_model(configuration.fast_llm, temperature=temperature)

    return (update_prompt | model | StrOutputParser() | parse_taxa).with_config(
        run_name="UpdateTaxonomy"
    )


async def update_taxonomy(state: State, config: RunnableConfig) -> dict:
    """Refine the taxonomy with the next unprocessed minibatch. Used in both modes."""
    configuration = Configuration.from_runnable_config(config)

    # taxonomy_updates accumulates by 1 per call (see state.py); cold_start
    # starts at 0 (minibatch 0 already consumed by generate_taxonomy) and
    # incremental starts at -1 (see mode_detector.py), so this always lands
    # on the next minibatch neither path has processed yet.
    which_mb = (state.taxonomy_updates + 1) % len(state.minibatches)
    batch_mentions = [state.mentions[i] for i in state.minibatches[which_mb]]

    current_taxonomy_text = "\n".join(f"- {t}" for t in state.taxonomy)

    def chain_factory(temp_bump: float):
        return _setup_update_chain(configuration, temperature=BASE_TEMPERATURE + temp_bump)

    result = await invoke_taxonomy_chain(
        chain_factory,
        batch_mentions,
        max_topics=configuration.max_num_clusters,
        allow_empty=False,
        current_taxonomy=current_taxonomy_text,
    )
    result["taxonomy_updates"] = 1

    # Evaluate the candidate against the best taxonomy found so far instead
    # of accepting it unconditionally.
    candidate_taxonomy = result["taxonomy"]
    best_so_far = state.best_taxonomy or state.taxonomy
    best_taxonomy = await select_better_taxonomy(
        configuration,
        best_so_far,
        candidate_taxonomy,
        state.validation_mentions,
        state.use_case,
    )
    result["best_taxonomy"] = best_taxonomy
    result["status"] = result["status"] + [
        "Model selection: pastram taxonomia anterioara (candidatul nou nu a "
        "castigat pe esantionul de validare)."
        if best_taxonomy == best_so_far and best_taxonomy != candidate_taxonomy
        else "Model selection: candidatul nou devine cea mai buna taxonomie cunoscuta."
    ]
    return result
