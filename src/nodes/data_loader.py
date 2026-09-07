"""Node for loading mentions from the MySQL database."""

from __future__ import annotations

import random

from langchain_core.runnables import RunnableConfig

from configuration import Configuration
from state import State
from utils import get_connection, get_last_processed_id, row_to_mention


async def fetch_mentions(state: State, config: RunnableConfig) -> dict:
    """Fetch new mentions for a project using an id-based cursor, and split them into a refinement sample and a validation sample."""
    if not state.project_id:
        raise ValueError("project_id not set in state")

    configuration = Configuration.from_runnable_config(config)
    last_id = get_last_processed_id(state.project_id)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT id, url, title, text_before, text_keyword, text_after
        FROM mentions
        WHERE project_id = %s AND id > %s
        ORDER BY id ASC
        LIMIT %s
    """
    cursor.execute(query, (state.project_id, last_id, configuration.max_mentions))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    all_mentions = [row_to_mention(row) for row in rows]
    next_cursor = max((int(row["id"]) for row in rows), default=last_id)

    # sample is drawn first and capped at len(all_mentions), so it can only be
    # empty if no mentions were fetched at all. validation_mentions is drawn
    # from the remainder, keeping the two sets disjoint. max_mentions is sized
    # to give both their full size in the common case; on a small tail batch,
    # validation_mentions shrinks instead of sample.
    sample_size = min(configuration.sample_size, len(all_mentions))
    sample = random.sample(all_mentions, sample_size)

    sample_ids = {m.id for m in sample}
    remaining = [m for m in all_mentions if m.id not in sample_ids]
    validation_size = min(configuration.taxonomy_batch_size, len(remaining))
    validation_mentions = random.sample(remaining, validation_size)

    status_message = (
        f"Fetched {len(all_mentions)} new mentions for project '{state.project_id}' "
        f"(cursor was id > {last_id})"
    )

    return {
        "all_mentions": all_mentions,
        "mentions": sample,
        "validation_mentions": validation_mentions,
        "next_cursor": next_cursor,
        "status": [status_message],
    }
