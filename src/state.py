"""State schema for the taxonomy classification graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Literal
import operator


@dataclass
class Mention:
    """A single web mention as it moves through fetching, classification, and review."""

    id: str
    content: str
    url: Optional[str] = None
    category: Optional[List[str]] = None
    primary_category: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class InputState:
    """Input schema: the only value required to start a graph run."""

    project_id: str = ""


@dataclass
class OutputState:
    """Output schema: the fields returned when the graph finishes."""

    mentions: List[Mention] = field(default_factory=list)
    taxonomy: List[str] = field(default_factory=list)
    new_topics_proposed: List[str] = field(default_factory=list)
    # Replaced, not accumulated, by propose_new_topics after reclassification.
    low_confidence_mentions: List[Mention] = field(default_factory=list)
    status: Annotated[List[str], operator.add] = field(default_factory=list)


@dataclass
class State(InputState, OutputState):
    """Full graph state, threaded through every node."""

    mode: Literal["cold_start", "incremental"] = field(default="cold_start")
    existing_taxonomy: Optional[List[str]] = field(default=None)
    all_mentions: List[Mention] = field(default_factory=list)
    minibatches: List[List[int]] = field(default_factory=list)
    # Count of update_taxonomy calls so far. Accumulates via operator.add
    # across the refinement loop; used to pick the next minibatch.
    taxonomy_updates: Annotated[int, operator.add] = field(default=0)
    use_case: str = field(default="topic classification for web mentions")
    # Highest mention id fetched this run, persisted as the cursor once classified.
    next_cursor: int = field(default=0)
    # Held-out mentions used only to compare taxonomy candidates, never for refinement.
    validation_mentions: List[Mention] = field(default_factory=list)
    # Best-scoring taxonomy found so far, not necessarily the most recent candidate.
    best_taxonomy: List[str] = field(default_factory=list)
