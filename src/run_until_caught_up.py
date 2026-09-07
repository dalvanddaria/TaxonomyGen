"""Run the graph repeatedly for one project until it's caught up (no new mentions left).

Usage: python run_until_caught_up.py <project_id> [--max-iterations N]
                                                    [--sleep-seconds N]
                                                    [--max-retries N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from graph import graph  # noqa: E402
from run_logger import log_run  # noqa: E402


async def run_until_caught_up(
    project_id: str,
    max_iterations: int = 50,
    sleep_seconds: float = 2.0,
    max_retries: int = 1,
) -> None:
    """Invoke the graph in a loop until a run reports zero new mentions or the iteration cap is hit."""
    total_mentions = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 60}\nRUN #{iteration} for '{project_id}'\n{'=' * 60}")

        attempt = 0
        result = None
        last_error: Optional[BaseException] = None

        while attempt <= max_retries:
            try:
                result = await graph.ainvoke({"project_id": project_id})
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                attempt += 1
                if attempt <= max_retries:
                    print(f"  Error on attempt {attempt}: {exc}")
                    print(f"  Retrying in {sleep_seconds * 2}s...")
                    await asyncio.sleep(sleep_seconds * 2)

        if last_error is not None:
            # Retries exhausted on a single run: stop the whole loop instead of
            # retrying an error that won't resolve on its own.
            print(f"\nPERSISTENT ERROR after {max_retries + 1} attempts: {last_error}")
            print("Stopping - check your connection/API key and rerun.")
            log_run(project_id, result=None, error=str(last_error))
            return

        log_run(project_id, result=result)

        mentions_count = len(result["mentions"])
        total_mentions += mentions_count

        print("Status:")
        for line in result["status"]:
            print(f"  - {line}")
        print(f"Mentions processed: {mentions_count}")

        if mentions_count == 0:
            print(f"\n{'=' * 60}")
            print(f"DONE - '{project_id}' is caught up, no new mentions left.")
            print(f"Total: {iteration} runs, {total_mentions} mentions.")
            print(f"{'=' * 60}")
            return

        if iteration < max_iterations:
            await asyncio.sleep(sleep_seconds)

    print(f"\nHit the safety cap of {max_iterations} runs - stopping.")
    print(f"Total: {total_mentions} mentions.")
    print("Run the script again if there's more backlog to cover.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id", help="Project to process (e.g. soundiiz)")
    parser.add_argument(
        "--max-iterations", type=int, default=50,
        help="Safety cap - stop after this many runs regardless (default 50).",
    )
    parser.add_argument(
        "--sleep-seconds", type=float, default=2.0,
        help="Pause between successful runs (default 2s).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=1,
        help="Retries per run on a transient error (default 1).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_until_caught_up(
            args.project_id,
            max_iterations=args.max_iterations,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
    )
