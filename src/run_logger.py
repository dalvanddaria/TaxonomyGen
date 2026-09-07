"""Append-only run history, per project.

Writes one JSON line per run to data/run_history.jsonl instead of one big
JSON file, so it stays safe to append to across a long loop of runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

RUN_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "run_history.jsonl"


def log_run(
    project_id: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Append one run's outcome to the history file."""
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "project_id": project_id,
        "error": error,
    }

    if result is not None:
        mentions = result.get("mentions", [])
        low_confidence = result.get("low_confidence_mentions", [])
        record.update(
            {
                "mentions_total": len(mentions),
                "classified": len(mentions) - len(low_confidence),
                "low_confidence": len(low_confidence),
                "taxonomy_size": len(result.get("taxonomy", [])),
                "new_topics_proposed": result.get("new_topics_proposed", []),
                "status": result.get("status", []),
            }
        )

    with open(RUN_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
