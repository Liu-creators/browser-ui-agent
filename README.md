# Browser UI-Agent

A Python framework for building **browser-based UI Agents** that autonomously control web browsers to complete tasks using LLM vision and reasoning capabilities.

## Architecture

```
browser-ui-agent/
├── agent/
│   ├── __init__.py      # Package exports
│   ├── agent.py         # UIAgent core (ReAct loop)
│   ├── browser.py       # BrowserController (Playwright)
│   ├── llm.py           # LLMClient (OpenAI / Anthropic)
│   └── prompts.py       # Optimized system prompts
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
└── README.md
```

## How It Works

The agent uses the **ReAct (Reason + Act)** pattern:

1. **Observe** - Take a screenshot of the current browser state
2. **Think** - Send screenshot + task context to LLM for reasoning
3. **Act** - Execute the LLM-decided browser action (click, type, navigate, scroll, etc.)
4. **Verify** - Confirm the action succeeded with another LLM vision check
5. **Recover** - On failure, ask LLM for a recovery strategy
6. **Repeat** until task is complete or max steps reached

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Set API key

```bash
# For OpenAI (GPT-4o)
export OPENAI_API_KEY=sk-...

# For Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run a task

```bash
# Basic usage
python main.py --task "Search for Python tutorials on Google"

# Use Claude instead of GPT-4o
python main.py --task "Find the top trending repos on GitHub" --model claude-3-5-sonnet-20241022

# Headless mode + save results
python main.py --task "Extract top 10 HackerNews titles" --headless --output result.json

# More steps for complex tasks
python main.py --task "Book a table at a restaurant" --max-steps 50
```

## Python API

```python
import asyncio
from agent import UIAgent, AgentConfig, BrowserConfig

async def main():
    config = AgentConfig(
        model="gpt-4o",
        max_steps=30,
        browser=BrowserConfig(headless=False)
    )
    agent = UIAgent(config=config)
    result = await agent.run("Go to news.ycombinator.com and list the top 5 posts")
    print(result["result"])

asyncio.run(main())
```

## Supported Models

| Provider | Models |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini` |
| Anthropic | `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229` |

## Prompt Architecture

The `agent/prompts.py` module contains 6 specialized prompts:

| Prompt | Purpose |
|---|---|
| `SYSTEM_PROMPT` | Core agent identity, ReAct format, safety rules |
| `VISION_PROMPT` | Screenshot analysis & element extraction |
| `PLANNING_PROMPT` | Task decomposition into atomic steps |
| `VERIFICATION_PROMPT` | Post-action success verification |
| `RECOVERY_PROMPT` | Error diagnosis & recovery strategy |
| `EXTRACTION_PROMPT` | Structured data extraction from pages |

## Key Design Decisions

- **Screenshot-based perception**: No DOM scraping - the agent sees the page exactly as a human would
- **JSON action schema**: LLM outputs structured JSON for reliable action parsing
- **Step history context**: Last 5 steps are provided to LLM to prevent loops
- **Configurable timeouts**: All Playwright waits are configurable via `BrowserConfig`
- **Async-first**: Built on `asyncio` + `playwright.async_api` for performance

## CLI Options

```
--task, -t      Task description (required)
--model, -m     LLM model (default: gpt-4o)
--headless      Run browser without GUI
--max-steps     Max agent steps (default: 30)
--output, -o    Save JSON result to file
--no-confirm    Skip confirmation prompts
```

## Requirements

- Python 3.11+
- Chromium (installed via `playwright install chromium`)
- OpenAI API key OR Anthropic API key
