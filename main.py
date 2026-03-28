#!/usr/bin/env python3
"""Browser UI-Agent - Entry Point

Example usage:
    python main.py --task "Search for Python tutorials on Google"
    python main.py --task "Go to github.com and star the first trending repo"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from agent.agent import UIAgent, AgentConfig
from agent.browser import BrowserConfig
from agent.llm import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser UI-Agent: Automate web tasks with LLM+browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task "Search Google for Python async tutorials"
  python main.py --task "Book a flight" --model claude-3-5-sonnet-20241022
  python main.py --task "Extract top 10 HackerNews titles" --headless
    """,
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Natural language task for the agent to perform",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of agent steps (default: 30)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save task result to JSON file",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Disable confirmation prompts for irreversible actions",
    )
    return parser.parse_args()


async def run_task(args: argparse.Namespace) -> dict:
    """Run the agent task."""
    # Validate API keys
    if args.model.startswith(("gpt", "o1", "o3")):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)

    # Build config
    config = AgentConfig(
        model=args.model,
        max_steps=args.max_steps,
        confirm_irreversible=not args.no_confirm,
        browser=BrowserConfig(
            headless=args.headless,
            viewport_width=1280,
            viewport_height=800,
        ),
    )

    logger.info("Task: %s", args.task)
    logger.info("Model: %s | Headless: %s | Max steps: %d", args.model, args.headless, args.max_steps)

    agent = UIAgent(config=config)
    result = await agent.run(args.task)

    # Display result
    print("\n" + "=" * 60)
    print(f"STATUS: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"STEPS:  {result['steps']}")
    print(f"RESULT: {result['result']}")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        serializable = {k: v for k, v in result.items() if k != "history"}
        serializable["history"] = [
            {
                "step": r.step,
                "observation": r.observation,
                "thought": r.thought,
                "action": r.action,
                "verified": r.verified,
            }
            for r in result.get("history", [])
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info("Result saved to %s", args.output)

    return result


def main():
    args = parse_args()
    try:
        result = asyncio.run(run_task(args))
        sys.exit(0 if result["success"] else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
