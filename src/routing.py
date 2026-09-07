"""Conditional-edge functions used by graph.py."""

from __future__ import annotations

from typing import Literal

from state import State
from nodes.new_topics import MIN_MENTIONS_FOR_NEW_TOPIC


def should_process_mentions(state: State) -> Literal["get_minibatches", "__end__"]:
    """End the run if fetch_mentions returned no new mentions."""
    if not state.all_mentions:
        return "__end__"
    return "get_minibatches"


def route_by_mode(state: State) -> Literal["cold_start", "incremental"]:
    """Route to the cold_start or incremental branch based on state.mode."""
    return state.mode


def should_continue_taxonomy_loop(
    state: State,
) -> Literal["update_taxonomy", "review_taxonomy"]:
    """Continue the refinement loop until every minibatch has been processed once."""
    if len(state.minibatches) <= 1:
        return "review_taxonomy"
    if state.taxonomy_updates >= len(state.minibatches) - 1:
        return "review_taxonomy"
    return "update_taxonomy"


def should_propose_new_topics(state: State) -> Literal["propose_new_topics", "__end__"]:
    """Route to propose_new_topics only if enough mentions were low-confidence to suggest a pattern."""
    if len(state.low_confidence_mentions) >= MIN_MENTIONS_FOR_NEW_TOPIC:
        return "propose_new_topics"
    return "__end__"
