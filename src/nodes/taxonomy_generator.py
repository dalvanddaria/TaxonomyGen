"""Node for generating the initial taxonomy from the first minibatch."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from prompts import TAXONOMY_GENERATION_PROMPT
from state import State
from utils import invoke_taxonomy_chain, load_chat_model, parse_taxa

# Higher than the refinement/review passes: generating from scratch benefits
# from more exploration.
BASE_TEMPERATURE = 0.5


def _setup_taxonomy_chain(configuration: Configuration, temperature: float = BASE_TEMPERATURE):
    """Build the generation prompt -> model -> parser chain."""
    taxonomy_prompt = TAXONOMY_GENERATION_PROMPT.partial(
        use_case=(
            "Identifica principalele topicuri discutate in mentiunile "
            "unui proiect de brand monitoring."
        ),
        max_num_clusters=configuration.max_num_clusters,
    )
    model = load_chat_model(configuration.fast_llm, temperature=temperature)

    return (taxonomy_prompt | model | StrOutputParser() | parse_taxa).with_config(
        run_name="GenerateTaxonomy"
    )


async def generate_taxonomy(state: State, config: RunnableConfig) -> dict:
    """Generate the initial taxonomy from the first minibatch. cold_start only."""
    configuration = Configuration.from_runnable_config(config)

    first_batch_indices = state.minibatches[0]
    batch_mentions = [state.mentions[i] for i in first_batch_indices]

    def chain_factory(temp_bump: float):
        return _setup_taxonomy_chain(configuration, temperature=BASE_TEMPERATURE + temp_bump)

    result = await invoke_taxonomy_chain(
        chain_factory,
        batch_mentions,
        max_topics=configuration.max_num_clusters,
        allow_empty=False,
    )
    # First candidate of the run: nothing to compare it against yet.
    result["best_taxonomy"] = result["taxonomy"]
    return result
