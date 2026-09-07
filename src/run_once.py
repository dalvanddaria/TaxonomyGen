"""Run the graph once with production defaults and print the full result."""

import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    # Without this, printing Romanian diacritics crashes on Windows' default console encoding.
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from graph import graph
from run_logger import log_run


async def main():
    project_id = "soundiiz"  # a project_id that actually exists in your DB

    result = await graph.ainvoke({"project_id": project_id})
    log_run(project_id, result=result)

    print("=== STATUS ===")
    for line in result["status"]:
        print(" -", line)

    print("\n=== TAXONOMY ===")
    for t in result["taxonomy"]:
        print(" -", t)

    print(f"\n=== MENTIONS ({len(result['mentions'])}) ===")
    for m in result["mentions"][:10]:
        print(f"  [{m.id}] conf={m.confidence} primary={m.primary_category} cats={m.category}")
    if len(result["mentions"]) > 10:
        print(f"  ... and {len(result['mentions']) - 10} more")

    print(f"\n=== LOW CONFIDENCE ({len(result['low_confidence_mentions'])}) ===")
    for m in result["low_confidence_mentions"][:10]:
        print(f"  [{m.id}] conf={m.confidence} - {m.explanation}")
    if len(result["low_confidence_mentions"]) > 10:
        print(f"  ... and {len(result['low_confidence_mentions']) - 10} more")

    print("\n=== NEW TOPICS PROPOSED ===", result.get("new_topics_proposed"))


asyncio.run(main())
