"""Node for generating minibatches from mentions."""

from __future__ import annotations

import random

from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from state import State


def _create_batches(indices: list[int], batch_size: int) -> list[list[int]]:
    """Split indices into fixed-size batches, padding the final short batch by resampling."""
    if len(indices) < batch_size:
        return [indices]

    num_full_batches = len(indices) // batch_size
    batches = [
        indices[i * batch_size : (i + 1) * batch_size] for i in range(num_full_batches)
    ]

    leftovers = len(indices) % batch_size
    if leftovers:
        # Pad the last batch up to batch_size with a resample rather than
        # shipping an undersized final batch.
        last_batch = indices[num_full_batches * batch_size :]
        elements_to_add = batch_size - leftovers
        last_batch += random.sample(indices, elements_to_add)
        batches.append(last_batch)

    return batches


async def generate_minibatches(state: State, config: RunnableConfig) -> dict:
    """Shuffle the sampled mentions and split them into fixed-size minibatches."""
    configuration = Configuration.from_runnable_config(config)

    indices = list(range(len(state.mentions)))
    random.shuffle(indices)

    batches = _create_batches(indices, configuration.taxonomy_batch_size)

    return {
        "minibatches": batches,
        "status": [
            f"Generated {len(batches)} minibatch(es) from {len(state.mentions)} mentions."
        ],
    }
