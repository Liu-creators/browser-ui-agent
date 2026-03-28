"""UI-Agent Core

Orchestrates the LLM + browser to execute tasks autonomously.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.browser import ActionResult, BrowserController, BrowserConfig
from agent.llm import LLMClient
from agent.prompts import (
    SYSTEM_PROMPT,
    VISION_PROMPT,
    PLANNING_PROMPT,
    VERIFICATION_PROMPT,
    RECOVERY_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the UI-Agent."""

    model: str = "gpt-4o"             # LLM model to use
    max_steps: int = 30               # max actions per task
    max_retries: int = 3              # retries on action failure
    screenshot_on_every_step: bool = True
    confirm_irreversible: bool = True  # ask before destructive actions
    verbose: bool = True
    browser: BrowserConfig = field(default_factory=BrowserConfig)


@dataclass
class StepRecord:
    """A single agent step log."""

    step: int
    observation: str
    thought: str
    action: str
    result: Optional[ActionResult] = None
    verified: bool = False


class UIAgent:
    """The main UI-Agent that combines LLM reasoning with browser control.

    Usage::

        agent = UIAgent(AgentConfig())
        result = await agent.run("Search for Python tutorials on Google")
    """

    def __init__(self, config: Optional[AgentConfig] = None, llm: Optional[LLMClient] = None):
        self.config = config or AgentConfig()
        self.llm = llm or LLMClient(model=self.config.model)
        self.browser = BrowserController(self.config.browser)
        self.history: list[StepRecord] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> dict[str, Any]:
        """Execute a task end-to-end.

        Args:
            task: Natural language task description.

        Returns:
            Dict with keys: success, result, steps, history.
        """
        logger.info("Starting task: %s", task)
        self.history.clear()
        self._step = 0

        async with self.browser.session():
            # Step 1: Initial screenshot & planning
            screenshot_b64 = await self._capture_screenshot()
            plan = await self._plan(task, screenshot_b64)
            if self.config.verbose:
                logger.info("Execution plan:\n%s", plan)

            # Step 2: ReAct loop
            while self._step < self.config.max_steps:
                self._step += 1
                logger.info("--- Step %d/%d ---", self._step, self.config.max_steps)

                screenshot_b64 = await self._capture_screenshot()
                step_result = await self._reason_and_act(task, screenshot_b64, plan)

                if step_result.get("done"):
                    return {
                        "success": True,
                        "result": step_result.get("summary", "Task completed"),
                        "steps": self._step,
                        "history": self.history,
                    }

                if step_result.get("failed"):
                    return {
                        "success": False,
                        "result": step_result.get("error", "Task failed"),
                        "steps": self._step,
                        "history": self.history,
                    }

            return {
                "success": False,
                "result": f"Reached max steps ({self.config.max_steps}) without completing task",
                "steps": self._step,
                "history": self.history,
            }

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    async def _plan(self, task: str, screenshot_b64: str) -> str:
        """Ask LLM to generate an execution plan."""
        page_info = await self.browser.get_page_info()
        prompt = PLANNING_PROMPT.format(
            task=task,
            current_url=page_info["url"],
            page_state="Browser just opened, no page loaded yet.",
        )
        return await self.llm.vision_chat(
            system=SYSTEM_PROMPT,
            user_text=prompt,
            image_b64=screenshot_b64,
        )

    async def _reason_and_act(
        self, task: str, screenshot_b64: str, plan: str
    ) -> dict[str, Any]:
        """One ReAct cycle: observe -> think -> act -> verify."""
        page_info = await self.browser.get_page_info()
        history_summary = self._format_history_summary()

        user_prompt = f"""Current task: {task}

Execution plan:
{plan}

Step history:
{history_summary}

Current URL: {page_info['url']}
Current page title: {page_info['title']}
Step: {self._step}/{self.config.max_steps}

Analyze the screenshot and decide the next action.
Respond in JSON with this schema:
{{
  "observation": "...",
  "thought": "...",
  "action_type": "navigate|click|type|scroll|wait|extract|done|failed",
  "action_args": {{...}},
  "expected_outcome": "...",
  "done": false,
  "failed": false,
  "summary": "(only if done=true) final result summary",
  "error": "(only if failed=true) error description"
}}"""

        response_text = await self.llm.vision_chat(
            system=SYSTEM_PROMPT,
            user_text=user_prompt,
            image_b64=screenshot_b64,
            response_format="json",
        )

        try:
            step_data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response, retrying...")
            return {"done": False, "failed": False}

        # Log step
        record = StepRecord(
            step=self._step,
            observation=step_data.get("observation", ""),
            thought=step_data.get("thought", ""),
            action=f"{step_data.get('action_type')}({step_data.get('action_args', {})})",
        )

        if self.config.verbose:
            logger.info("OBSERVATION: %s", record.observation)
            logger.info("THOUGHT: %s", record.thought)
            logger.info("ACTION: %s", record.action)

        # Execute action
        if not step_data.get("done") and not step_data.get("failed"):
            result = await self._execute_action(
                step_data.get("action_type", "wait"),
                step_data.get("action_args", {}),
            )
            record.result = result

            # Verify
            if result.success:
                record.verified = await self._verify_step(
                    action_desc=record.action,
                    expected=step_data.get("expected_outcome", ""),
                )
            else:
                # Recovery attempt
                recovered = await self._attempt_recovery(
                    failed_action=record.action,
                    error_details=result.error or "",
                    task_goal=task,
                )
                if not recovered:
                    step_data["failed"] = True
                    step_data["error"] = result.error

        self.history.append(record)
        return step_data

    async def _execute_action(self, action_type: str, args: dict) -> ActionResult:
        """Dispatch an LLM-decided action to the browser."""
        dispatch = {
            "navigate": lambda: self.browser.navigate(args.get("url", "")),
            "click": lambda: self.browser.click(
                selector=args.get("selector"),
                x=args.get("x"),
                y=args.get("y"),
                double=args.get("double", False),
            ),
            "type": lambda: self.browser.type_text(
                selector=args.get("selector", ""),
                text=args.get("text", ""),
                press_enter=args.get("press_enter", False),
            ),
            "scroll": lambda: self.browser.scroll(
                delta_x=args.get("delta_x", 0),
                delta_y=args.get("delta_y", 300),
            ),
            "wait": lambda: self._wait_action(args),
            "press_key": lambda: self.browser.press_key(args.get("key", "Enter")),
            "go_back": lambda: self.browser.go_back(),
            "extract": lambda: self.browser.get_text(args.get("selector", "body")),
        }
        handler = dispatch.get(action_type)
        if handler:
            return await handler()
        return ActionResult(success=True, message=f"No-op for action: {action_type}")

    async def _wait_action(self, args: dict) -> ActionResult:
        selector = args.get("selector")
        seconds = args.get("seconds", 1)
        if selector:
            return await self.browser.wait_for_selector(selector)
        await self.browser.sleep(seconds)
        return ActionResult(success=True, message=f"Waited {seconds}s")

    async def _verify_step(self, action_desc: str, expected: str) -> bool:
        """Ask LLM to verify if the action had the expected result."""
        screenshot_b64 = await self._capture_screenshot()
        prompt = VERIFICATION_PROMPT.format(
            action_description=action_desc,
            expected_outcome=expected,
        )
        response = await self.llm.vision_chat(
            system=SYSTEM_PROMPT,
            user_text=prompt,
            image_b64=screenshot_b64,
        )
        return "SUCCESS" in response.upper() or "PARTIAL" in response.upper()

    async def _attempt_recovery(self, failed_action: str, error_details: str, task_goal: str) -> bool:
        """Ask LLM for a recovery strategy and attempt to execute it."""
        screenshot_b64 = await self._capture_screenshot()
        prompt = RECOVERY_PROMPT.format(
            failed_action=failed_action,
            error_details=error_details,
            task_goal=task_goal,
        )
        recovery_plan = await self.llm.vision_chat(
            system=SYSTEM_PROMPT,
            user_text=prompt,
            image_b64=screenshot_b64,
        )
        logger.info("Recovery plan: %s", recovery_plan)
        # For now just log; real implementation would parse and retry
        return False

    async def _capture_screenshot(self) -> str:
        """Take screenshot and return base64 string."""
        result = await self.browser.screenshot()
        return result.screenshot or ""

    def _format_history_summary(self) -> str:
        if not self.history:
            return "No steps taken yet."
        lines = []
        for record in self.history[-5:]:  # last 5 steps
            status = "OK" if record.verified else "UNVERIFIED"
            lines.append(f"Step {record.step} [{status}]: {record.action}")
        return "\n".join(lines)
