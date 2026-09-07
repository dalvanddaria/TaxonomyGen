"""Runtime configuration for the taxonomy classification graph: models, batch sizes, and thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class Configuration:
    """Tunable parameters for the taxonomy classification graph, sourced from RunnableConfig."""

    # Higher-quality, higher-cost model. Used only by propose_new_topics,
    # a low-frequency call where the extra quality is worth the price.
    model: str = field(
        default="anthropic/claude-sonnet-5",
        metadata={
            "description": "Modelul principal, folosit pentru generarea/rafinarea taxonomiei. "
        },
    )

    # Cheaper, faster model. Used for every high-volume call: classification
    # and all taxonomy generation/update/review prompts.
    fast_llm: str = field(
        default="anthropic/claude-haiku-4.5",
        metadata={"description": "Model mai ieftin/rapid, pentru sarcini simple "},
    )

    # Must cover sample_size + taxonomy_batch_size, so a full run has enough
    # fetched mentions to give the validation set its full size without
    # shrinking the refinement sample.
    max_mentions: int = field(
        default=650,
        metadata={"description": "Numar maxim de mentiuni preluate per rulare."},
    )

    # Number of fetched mentions used to build/refine the taxonomy each run.
    sample_size: int = field(
        default=500,
        metadata={
            "description": "Esantion pentru generarea initiala a taxonomiei (Faza 1)."
        },
    )

    # Minibatch size for taxonomy generation/update/review. Kept separate
    # from classification_batch_size: output here is a short topic list,
    # not one object per mention, so it tolerates a much larger batch.
    taxonomy_batch_size: int = field(
        default=150,
        metadata={
            "description": "Marimea mini-batch-urilor pentru rafinarea taxonomiei."
        },
    )

    # Minibatch size for classification. Output is one JSON object per
    # mention, so this stays small enough to avoid response truncation.
    classification_batch_size: int = field(
        default=25,
        metadata={
            "description": "Marimea mini-batch-urilor pentru clasificarea mentiunilor."
        },
    )

    # Hard ceiling on the number of topics a taxonomy can contain.
    max_num_clusters: int = field(
        default=20,
        metadata={"description": "Numar maxim de topicuri/clustere permise."},
    )

    # Classifications scoring below this confidence are treated as ambiguous
    # and routed to low_confidence_mentions instead of being accepted.
    confidence_threshold: float = field(
        default=0.7,
        metadata={
            "description": "Sub acest scor de incredere, se escaladeaza la LLM "
            "pentru posibil topic nou."
        },
    )

    # Concurrent LLM calls allowed during batched classification. Lower this
    # on a rate-limited model tier.
    max_concurrency: int = field(
        default=5,
        metadata={"description": "Cate cereri LLM simultane trimitem la clasificare. "},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Build a Configuration from a RunnableConfig's configurable dict, ignoring unknown keys."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
