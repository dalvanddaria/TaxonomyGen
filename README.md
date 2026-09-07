# Taxonomy Classification

## What this is

This project automatically builds a topic taxonomy from web mentions of a
brand or product, and classifies each mention against it, using an LLM
instead of manual tagging or keyword matching. It exists because
brand-monitoring platforms collect far more mentions than anyone can read
by hand, someone still needs to know what topics are actually being
discussed (pricing complaints, competitor comparisons, technical issues)
without opening every article, post, and comment one at a time.

The pipeline runs in two modes: `cold_start`, which builds a taxonomy from
scratch for a new project, and `incremental`, which refines an existing
taxonomy against newly arrived mentions and classifies them against it,
so a project's taxonomy keeps improving over time instead of being
regenerated from zero on every run.

## Prerequisites

- **Python 3.10+**
- **A MySQL server** with a `mentions` table containing at least:
  `id` (int, primary key), `project_id` (string), `url`, `title` (nullable),
  `text_before`, `text_keyword`, `text_after`, plus `primary_category`,
  `categories`, `confidence`, and `explanation` (all nullable) for the
  pipeline to write classification results back into
- **An OpenRouter API key** - all LLM calls go through OpenRouter's OpenAI-compatible endpoint, not OpenAI's or Anthropic's directly
- Python packages in `requirements.txt`: `langgraph`, `langchain-core`,
  `langchain-openai`, `python-dotenv`, `mysql-connector-python`

## How it works

```
START
  |
detect_mode          -> cold_start vs incremental
  |
get_mentions         -> fetch from DB, per-project cursor
  |
(no new mentions?) --> END
  |
get_minibatches
  |
  +-- cold_start  --> generate_taxonomy (from scratch, first minibatch)
  +-- incremental --> update_taxonomy   (from the existing taxonomy)
              |
      update_taxonomy (self-loop: one pass per remaining minibatch)
              |
       review_taxonomy
              |
      classify_mentions
              |
(enough low-confidence mentions?)
       /                \
propose_new_topics      END
       |
      END
```

Incremental improves the existing taxonomy using minibatches from the new mentions, same mechanism as cold_start, just not starting from empty.

Classification isn't single-label: each mention gets a `primary_category`
(its main topic) plus a full `category` list of every topic that applies -
a mention can legitimately belong to more than one, so it isn't forced
into a single bucket.

Every result gets written back to the `mentions` row it came from -
`primary_category`, `categories` (as JSON text), `confidence`, and
`explanation` - so the classification isn't just printed to a console and
discarded. A mention that couldn't be classified (missing taxonomy, a
failed batch, no result from the model) still gets a row update: `NULL`
category and confidence, with the reason recorded in `explanation`.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: OPENROUTER_API_KEY, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
cd src
python run_once.py
```

`run_once.py` runs the pipeline once against a hardcoded project. To point
it at your own data, open `src/run_once.py` and change this line near the
top of the file:

```python
project_id = "soundiiz"  # a project_id that actually exists in your DB
```
Every project gets its own taxonomy and cursor, keyed by `project_id` in
`taxonomies.json`/`cursors.json` and filtered by `project_id` in the DB
query - so soundiiz and mangools (or any other project sharing the same
`mentions` table) are built and processed completely independently, even
though they live in the same table and files. Current projects: soundiiz,
mangools.

## Running it

- **`run_once.py`** - one invocation, production defaults, full detail
  printed (status, taxonomy, sample mentions, low-confidence list,
  proposed topics). Good for inspecting a single run closely.
- **`run_until_caught_up.py <project_id>`** - runs the graph repeatedly
  until the cursor catches up. A large project (~11.7k mentions) takes
  ~18 runs at the default `max_mentions=650`; the script stops on its own
  once there's nothing left. A persistent (non-transient) error stops the
  whole loop rather than retrying forever.

Every run saves its progress to src/data/: the current taxonomy goes in
taxonomies.json, the last mention id already processed for each project
goes in cursors.json (so the next run knows where to pick up), and a
record of the run (what got classified, what didn't, any errors) gets
appended to run_history.jsonl. `run_until_caught_up.py` guarantees that
history entry even on a persistent failure; `run_once.py` only logs it if
the graph actually finishes - an unhandled crash skips the log entry.

## Project structure

```
src/
├── graph.py                    # the LangGraph pipeline definition - start here
├── state.py                    # State / InputState / OutputState / Mention
├── configuration.py            # runtime config: models, batch sizes, thresholds
├── prompts.py                  # every LLM prompt template
├── routing.py                  # conditional-edge functions used by graph.py
├── utils.py                    # LLM client, DB connection, taxonomy/cursor persistence, parsing, retry logic
├── run_once.py                 # CLI: one invocation, full output printed
├── run_until_caught_up.py      # CLI: loop a project until its backlog is fully processed
├── run_logger.py               # append-only run history
├── nodes/
│   ├── mode_detector.py        # detect_mode: cold_start vs incremental
│   ├── data_loader.py          # fetch_mentions: MySQL fetch with cursor
│   ├── minibatches_generator.py
│   ├── taxonomy_generator.py   # generate_taxonomy (cold_start only)
│   ├── taxonomy_updater.py     # update_taxonomy (refinement loop)
│   ├── taxonomy_reviewer.py    # review_taxonomy (final pass + save)
│   ├── classify_mentions.py    # classify_mentions + run_classification
│   └── new_topics.py           # propose_new_topics
└── data/
    ├── taxonomies.json         # {project_id: [topics]}
    ├── cursors.json            # {project_id: last_processed_mention_id}
    └── run_history.jsonl       # one JSON line per run
```

## Model selection

Every time `update_taxonomy` produces a new version of the taxonomy, it
doesn't just get accepted automatically - it's tested against the best
version found so far. The test uses a separate set of mentions
(`state.validation_mentions`, drawn in `fetch_mentions`) that isn't used
to build or refine the taxonomy, so it's a fair check rather than the
model grading its own work.

Two versions are tracked separately: `taxonomy`, which is just whatever
came out of the latest refinement step, and `best_taxonomy`, which only
changes when a new version actually wins that comparison. Only
`best_taxonomy` gets passed on to the final review step - so if one batch
of mentions happens to be unrepresentative and produces a worse taxonomy,
that worse version gets discarded instead of becoming the new baseline.

When the model compares two versions, which one is labeled "1" and which
is "2" in the prompt is randomized on every call. Without that, models
tend to favor whichever option they see first, regardless of which one is
actually better - randomizing the order cancels that bias out.

## Guardrails

The pipeline is built to keep running - and keep data intact - even when
an LLM call misbehaves or fails outright, instead of crashing or silently
dropping mentions:

- **Retry on malformed output.** Every taxonomy-producing call (generate,
  update, review, propose) retries up to `MAX_TAXONOMY_RETRIES` (5) times if
  the response is malformed or goes over `max_num_clusters`, raising the
  temperature slightly each attempt to break the model out of a bad
  pattern. If it's still bad after 5 tries, the result is truncated to the
  cap and the run continues instead of failing outright.
- **No mention is silently lost during classification.** If a batch call
  errors out, or the model's response is missing an entry for a mention,
  that mention isn't dropped - it's marked unclassified with an explanation
  and sent to `low_confidence_mentions` for review, the same as a genuinely
  ambiguous one. Every mention fetched gets an outcome.
- **The cursor only advances once mentions are safely delivered.**
  `cursors.json` is updated after classification results have actually
  been written back to the `mentions` table, not as soon as they're
  fetched or classified in memory. If the process crashes, or the DB write
  itself fails, the cursor doesn't move - those mentions get fetched and
  retried on the next run instead of being skipped.

## Production configuration

Calibrated for the real project sizes in use (8k-11.7k mentions/project,
paid model):

- **`fast_llm` = Haiku 4.5** handles all the high-volume calls
  (classification + every taxonomy prompt) at roughly half of Sonnet's
  price. **`model` = Sonnet 5** is only used by `propose_new_topics`, rare
  enough that the extra quality is worth it.
- **`sample_size` = 500, `max_mentions` = 650** per run (650 = 500 +
  `taxonomy_batch_size`, so there's always enough fetched to give the
  validation set its full size on top of the refinement sample - see
  Model selection). Override `max_mentions` for a one-off full backfill
  rather than raising the default.
- **`taxonomy_batch_size` = 150 / `classification_batch_size` = 25** - kept
  separate: classification output is capped by tokens-per-response,
  taxonomy refinement isn't, so it can safely use a much bigger batch.
- **`max_num_clusters` = 20**, **`max_concurrency` = 5** (drop to 1-2 on a
  rate-limited free model).

**Topics vs. entity names.** Early runs drifted toward naming competing
products as topics (`Spotify`, `Tidal`, `Deezer`...), and once even the
monitored brand itself showed up as its own topic - both are the same
failure, sliding back into keyword matching instead of describing what's
discussed. All four topic-producing prompts now say explicitly, with
examples: a topic names a theme, not an entity; competitor mentions share
one topic ("competitor comparison"); the monitored brand is never a valid
topic, since every mention is already about it by definition.

## Proposing new topics

Only runs once `low_confidence_mentions` passes `MIN_MENTIONS_FOR_NEW_TOPIC`
(5) - below that it's likely just noise. Samples up to
`MAX_MENTIONS_FOR_PROPOSAL` (100) mentions to check for a pattern, and caps
new topics at whatever room is left under `max_num_clusters`. If a new
topic is found, every low-confidence mention gets reclassified against it -
not just the ones sampled to detect it, since others may fit it too.

## Known limitations

- **No lightweight-classifier distillation** - every mention goes through
  the LLM directly to get classified, so cost scales linearly with volume.
  Fine at the current scale (~10k mentions/project); would need a
  distilled classifier to stay cheap at an order of magnitude more.
- **No scheduler** - `run_until_caught_up.py` has to be run manually;
  nothing triggers automatically when new mentions come in.
- **No prompt caching** - the classification prompt repeats the same
  taxonomy/instructions text on every batch in a run, which is exactly
  what caching is for, but the current client library doesn't support it
  cleanly and the prompt is smaller than the minimum size providers
  require to cache it anyway.
