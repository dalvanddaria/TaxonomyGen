"""LangGraph definition for the taxonomy classification pipeline.

Builds or refines a taxonomy, classifies mentions against it, and proposes
new topics for mentions that didn't fit.
"""

from langgraph.graph import END, START, StateGraph

from configuration import Configuration
from routing import (
    route_by_mode,
    should_continue_taxonomy_loop,
    should_process_mentions,
    should_propose_new_topics,
)
from state import InputState, OutputState, State

from nodes.mode_detector import detect_mode
from nodes.data_loader import fetch_mentions
from nodes.minibatches_generator import generate_minibatches
from nodes.taxonomy_generator import generate_taxonomy
from nodes.taxonomy_updater import update_taxonomy
from nodes.taxonomy_reviewer import review_taxonomy
from nodes.classify_mentions import classify_mentions
from nodes.new_topics import propose_new_topics

builder = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)

# --- Nodes ---
builder.add_node("detect_mode", detect_mode)
builder.add_node("get_mentions", fetch_mentions)
builder.add_node("get_minibatches", generate_minibatches)
builder.add_node("generate_taxonomy", generate_taxonomy)
builder.add_node("update_taxonomy", update_taxonomy)
builder.add_node("review_taxonomy", review_taxonomy)
builder.add_node("classify_mentions", classify_mentions)
builder.add_node("propose_new_topics", propose_new_topics)

# --- Edges ---
builder.add_edge(START, "detect_mode")
builder.add_edge("detect_mode", "get_mentions")

# End the run here if fetch_mentions found nothing new.
builder.add_conditional_edges(
    "get_mentions",
    should_process_mentions,
    {
        "get_minibatches": "get_minibatches",
        "__end__": END,
    },
)

# cold_start generates a taxonomy from scratch; incremental refines the existing one.
builder.add_conditional_edges(
    "get_minibatches",
    route_by_mode,
    {
        "cold_start": "generate_taxonomy",
        "incremental": "update_taxonomy",
    },
)

# --- Taxonomy refinement loop ---
builder.add_conditional_edges(
    "generate_taxonomy",
    should_continue_taxonomy_loop,
    {
        "update_taxonomy": "update_taxonomy",
        "review_taxonomy": "review_taxonomy",
    },
)
builder.add_conditional_edges(
    "update_taxonomy",
    should_continue_taxonomy_loop,
    {
        "update_taxonomy": "update_taxonomy",
        "review_taxonomy": "review_taxonomy",
    },
)
builder.add_edge("review_taxonomy", "classify_mentions")

# --- Classification + new topic proposal ---
builder.add_conditional_edges(
    "classify_mentions",
    should_propose_new_topics,
    {
        "propose_new_topics": "propose_new_topics",
        "__end__": END,
    },
)
builder.add_edge("propose_new_topics", END)

graph = builder.compile()
graph.name = "Taxonomy Generation & Classification"
