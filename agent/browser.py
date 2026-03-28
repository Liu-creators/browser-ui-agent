"""Browser Controller

Manages Playwright browser instances for UI-Agent automation.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger(__name__)


@dataclass
class BrowserConfig:
    """Browser launch configuration."""

    headless: bool = False
    slow_mo: int = 50          # ms between actions (helps reliability)
    viewport_width: int = 1280
    viewport_height: int = 800
    timeout: int = 30_000       # default timeout in ms
    screenshot_full_page: bool = False
    user_agent: Optional[str] = None
    proxy: Optional[dict] = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of a browser action."""

    success: bool
    message: str = ""
    data: Optional[dict] = None
    screenshot: Optional[str] = None  # base64-encoded PNG
    error: Optional[str] = None


class BrowserController:
    """High-level browser controller for UI-Agent.

    Provides a clean API for common browser automation actions:
    navigate, click, type, scroll, extract, screenshot.
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the browser and create a fresh context."""
        self._playwright = await async_playwright().start()
        launch_kwargs = {
            "headless": self.config.headless,
            "slow_mo": self.config.slow_mo,
            "args": self.config.extra_args,
        }
        if self.config.proxy:
            launch_kwargs["proxy"] = self.config.proxy

        self._browser = await self._playwright.chromium.launch(**launch_kwargs)

        context_kwargs = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        }
        if self.config.user_agent:
            context_kwargs["user_agent"] = self.config.user_agent

        self._context = await self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self.config.timeout)
        self._page = await self._context.new_page()
        logger.info("Browser started (headless=%s)", self.config.headless)

    async def close(self) -> None:
        """Gracefully close browser resources."""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")

    @asynccontextmanager
    async def session(self):
        """Async context manager for a browser session."""
        await self.start()
        try:
            yield self
        finally:
            await self.close()

    # ------------------------------------------------------------------
    # Page accessors
    # ------------------------------------------------------------------

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    @property
    def current_url(self) -> str:
        return self._page.url if self._page else ""

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> ActionResult:
        """Navigate to a URL."""
        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            response = await self.page.goto(url, wait_until=wait_until)
            status = response.status if response else 0
            return ActionResult(
                success=status < 400,
                message=f"Navigated to {url} (status {status})",
                data={"url": self.current_url, "status": status},
            )
        except Exception as exc:
            return self._error("navigate", exc)

    async def go_back(self) -> ActionResult:
        try:
            await self.page.go_back()
            return ActionResult(success=True, message="Navigated back", data={"url": self.current_url})
        except Exception as exc:
            return self._error("go_back", exc)

    async def go_forward(self) -> ActionResult:
        try:
            await self.page.go_forward()
            return ActionResult(success=True, message="Navigated forward", data={"url": self.current_url})
        except Exception as exc:
            return self._error("go_forward", exc)

    async def reload(self) -> ActionResult:
        try:
            await self.page.reload()
            return ActionResult(success=True, message="Page reloaded")
        except Exception as exc:
            return self._error("reload", exc)

    # ------------------------------------------------------------------
    # Element Interaction
    # ------------------------------------------------------------------

    async def click(
        self,
        selector: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
        double: bool = False,
    ) -> ActionResult:
        """Click on a selector or coordinates."""
        try:
            if x is not None and y is not None:
                if double:
                    await self.page.mouse.dblclick(x, y)
                else:
                    await self.page.mouse.click(x, y, button=button)
                return ActionResult(success=True, message=f"Clicked at ({x}, {y})")
            elif selector:
                element = self.page.locator(selector).first
                if double:
                    await element.dblclick()
                else:
                    await element.click(button=button)
                return ActionResult(success=True, message=f"Clicked '{selector}'")
            else:
                return ActionResult(success=False, error="Provide selector or coordinates")
        except Exception as exc:
            return self._error("click", exc)

    async def type_text(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = False,
    ) -> ActionResult:
        """Type into an input field."""
        try:
            element = self.page.locator(selector).first
            await element.wait_for(state="visible")
            if clear_first:
                await element.clear()
            await element.type(text, delay=30)
            if press_enter:
                await element.press("Enter")
            return ActionResult(success=True, message=f"Typed into '{selector}'")
        except Exception as exc:
            return self._error("type_text", exc)

    async def press_key(self, key: str) -> ActionResult:
        """Press a keyboard key."""
        try:
            await self.page.keyboard.press(key)
            return ActionResult(success=True, message=f"Pressed key '{key}'")
        except Exception as exc:
            return self._error("press_key", exc)

    async def scroll(
        self,
        x: float = 0,
        y: float = 0,
        delta_x: float = 0,
        delta_y: float = 300,
    ) -> ActionResult:
        """Scroll the page."""
        try:
            await self.page.mouse.wheel(delta_x, delta_y)
            return ActionResult(success=True, message=f"Scrolled by ({delta_x}, {delta_y})")
        except Exception as exc:
            return self._error("scroll", exc)

    async def hover(self, selector: str) -> ActionResult:
        """Hover over an element."""
        try:
            await self.page.locator(selector).first.hover()
            return ActionResult(success=True, message=f"Hovered over '{selector}'")
        except Exception as exc:
            return self._error("hover", exc)

    async def select_option(self, selector: str, value: str) -> ActionResult:
        """Select an option from a dropdown."""
        try:
            await self.page.locator(selector).first.select_option(value)
            return ActionResult(success=True, message=f"Selected '{value}' in '{selector}'")
        except Exception as exc:
            return self._error("select_option", exc)

    # ------------------------------------------------------------------
    # Wait Utilities
    # ------------------------------------------------------------------

    async def wait_for_selector(
        self, selector: str, state: str = "visible", timeout: Optional[int] = None
    ) -> ActionResult:
        """Wait until an element matches the desired state."""
        try:
            await self.page.wait_for_selector(
                selector, state=state, timeout=timeout or self.config.timeout
            )
            return ActionResult(success=True, message=f"Element '{selector}' is {state}")
        except Exception as exc:
            return self._error("wait_for_selector", exc)

    async def wait_for_url(self, url_pattern: str, timeout: Optional[int] = None) -> ActionResult:
        try:
            await self.page.wait_for_url(url_pattern, timeout=timeout or self.config.timeout)
            return ActionResult(success=True, message=f"URL matches '{url_pattern}'")
        except Exception as exc:
            return self._error("wait_for_url", exc)

    async def wait_for_load(self, state: str = "networkidle") -> ActionResult:
        try:
            await self.page.wait_for_load_state(state)
            return ActionResult(success=True, message=f"Page reached state: {state}")
        except Exception as exc:
            return self._error("wait_for_load", exc)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    # ------------------------------------------------------------------
    # Screenshot & Page Info
    # ------------------------------------------------------------------

    async def screenshot(self, full_page: Optional[bool] = None) -> ActionResult:
        """Capture screenshot and return base64-encoded PNG."""
        try:
            raw = await self.page.screenshot(
                full_page=full_page if full_page is not None else self.config.screenshot_full_page
            )
            encoded = base64.b64encode(raw).decode("utf-8")
            return ActionResult(
                success=True,
                message="Screenshot captured",
                screenshot=encoded,
                data={"size_bytes": len(raw)},
            )
        except Exception as exc:
            return self._error("screenshot", exc)

    async def get_page_info(self) -> dict:
        """Return basic info about the current page."""
        return {
            "url": self.current_url,
            "title": await self.page.title(),
        }

    async def get_text(self, selector: str) -> ActionResult:
        """Get visible text of an element."""
        try:
            text = await self.page.locator(selector).first.inner_text()
            return ActionResult(success=True, message="Text extracted", data={"text": text})
        except Exception as exc:
            return self._error("get_text", exc)

    async def get_attribute(self, selector: str, attribute: str) -> ActionResult:
        try:
            value = await self.page.locator(selector).first.get_attribute(attribute)
            return ActionResult(success=True, data={"value": value})
        except Exception as exc:
            return self._error("get_attribute", exc)

    async def evaluate_js(self, script: str) -> ActionResult:
        """Execute JavaScript in the page context."""
        try:
            result = await self.page.evaluate(script)
            return ActionResult(success=True, data={"result": result})
        except Exception as exc:
            return self._error("evaluate_js", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(action: str, exc: Exception) -> ActionResult:
        msg = f"{action} failed: {exc}"
        logger.error(msg)
        return ActionResult(success=False, error=msg)
