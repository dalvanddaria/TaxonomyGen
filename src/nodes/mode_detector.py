"""Node for detecting whether this is a cold-start or incremental run."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from state import State
from utils import determine_mode, load_existing_taxonomy, reset_cursor


async def detect_mode(state: State, config: RunnableConfig) -> dict:
    """Determine cold_start vs incremental for this project and seed state accordingly."""
    mode = determine_mode(state.project_id)

    existing_taxonomy = None
    result: dict = {"mode": mode}

    if mode == "cold_start":
        # No taxonomy exists yet - a stale cursor from a previous, now-deleted
        # taxonomy would otherwise skip mentions the new taxonomy never saw.
        reset_cursor(state.project_id)

    if mode == "incremental":
        existing_taxonomy = load_existing_taxonomy(state.project_id)
        result["taxonomy"] = existing_taxonomy or []
        result["best_taxonomy"] = existing_taxonomy or []
        # -1 so the first update_taxonomy call resolves to minibatch 0, not 1.
        result["taxonomy_updates"] = -1

    status_message = f"Mode detected: {mode} for project '{state.project_id}'" + (
        f" ({len(existing_taxonomy)} topics existing)" if existing_taxonomy else ""
    )

    result["existing_taxonomy"] = existing_taxonomy
    result["status"] = [status_message]
    return result
