"""Browser tools — interactive web browsing via Playwright."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sentinel.core.types import PermissionTier
from sentinel.tools.base import Tool

logger = logging.getLogger(__name__)

# Lazy-loaded browser instance shared across tool calls within a session.
_browser = None
_playwright = None


async def _get_browser():
    """Lazily start Playwright and return a shared browser instance."""
    global _browser, _playwright
    if _browser is not None:
        return _browser

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is not installed. Run: pip install playwright && playwright install chromium"
        )

    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    return _browser


async def _shutdown_browser():
    """Cleanly close the browser and Playwright (call on app shutdown)."""
    global _browser, _playwright
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None


class BrowseWebTool(Tool):
    name = "browse_web"
    description = (
        "Navigate to a URL in a headless browser and return the visible text content. "
        "Unlike web_fetch, this renders JavaScript so it works with SPAs, dynamic pages, "
        "and pages that require client-side rendering. "
        "Use for reading modern web pages, documentation sites, and JS-heavy applications."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
            "wait_for": {
                "type": "string",
                "description": (
                    "CSS selector to wait for before extracting content. "
                    "Useful for SPAs that load content dynamically."
                ),
                "default": "",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Navigation timeout in milliseconds.",
                "default": 30000,
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return from page text.",
                "default": 20000,
            },
        },
        "required": ["url"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(
        self,
        url: str,
        wait_for: str = "",
        timeout_ms: int = 30000,
        max_chars: int = 20000,
    ) -> str:
        browser = await _get_browser()
        page = await browser.new_page()
        try:
            page.set_default_timeout(timeout_ms)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout_ms)
            else:
                # Give JS a moment to render
                await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10000))

            title = await page.title()
            text = await page.inner_text("body")
            final_url = page.url

            # Clean up excessive whitespace
            lines = [ln.strip() for ln in text.splitlines()]
            text = "\n".join(ln for ln in lines if ln)

            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            header = f"Title: {title}\nURL: {final_url}\n---\n"
            return header + text
        except Exception as e:
            return f"Browse failed: {type(e).__name__}: {e}"
        finally:
            await page.close()


class BrowserClickTool(Tool):
    name = "browser_click"
    description = (
        "Navigate to a URL, click an element matching a CSS selector, "
        "then return the resulting page text. Use for interacting with buttons, "
        "links, tabs, and other clickable elements on web pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
            "selector": {
                "type": "string",
                "description": "CSS selector of the element to click.",
            },
            "wait_after_ms": {
                "type": "integer",
                "description": "Milliseconds to wait after clicking for content to update.",
                "default": 2000,
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Navigation timeout in milliseconds.",
                "default": 30000,
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return.",
                "default": 20000,
            },
        },
        "required": ["url", "selector"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(
        self,
        url: str,
        selector: str,
        wait_after_ms: int = 2000,
        timeout_ms: int = 30000,
        max_chars: int = 20000,
    ) -> str:
        browser = await _get_browser()
        page = await browser.new_page()
        try:
            page.set_default_timeout(timeout_ms)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_selector(selector, timeout=timeout_ms)
            await page.click(selector)
            await page.wait_for_timeout(wait_after_ms)

            title = await page.title()
            text = await page.inner_text("body")
            final_url = page.url

            lines = [ln.strip() for ln in text.splitlines()]
            text = "\n".join(ln for ln in lines if ln)

            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            return f"Title: {title}\nURL: {final_url}\n---\n{text}"
        except Exception as e:
            return f"Click failed: {type(e).__name__}: {e}"
        finally:
            await page.close()


class BrowserFormTool(Tool):
    name = "browser_form"
    description = (
        "Navigate to a URL, fill in form fields, and optionally submit. "
        "Use for login pages, search forms, and any page requiring text input. "
        "Fields is a list of {selector, value} objects."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector of the input."},
                        "value": {"type": "string", "description": "Value to type into the input."},
                    },
                    "required": ["selector", "value"],
                },
                "description": "List of form fields to fill.",
            },
            "submit_selector": {
                "type": "string",
                "description": "CSS selector of the submit button. If empty, presses Enter on the last field.",
                "default": "",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Navigation timeout in milliseconds.",
                "default": 30000,
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return.",
                "default": 20000,
            },
        },
        "required": ["url", "fields"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(
        self,
        url: str,
        fields: list[dict[str, str]],
        submit_selector: str = "",
        timeout_ms: int = 30000,
        max_chars: int = 20000,
    ) -> str:
        browser = await _get_browser()
        page = await browser.new_page()
        try:
            page.set_default_timeout(timeout_ms)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            for field in fields:
                sel, val = field["selector"], field["value"]
                await page.wait_for_selector(sel, timeout=timeout_ms)
                await page.fill(sel, val)

            if submit_selector:
                await page.click(submit_selector)
            elif fields:
                last_sel = fields[-1]["selector"]
                await page.press(last_sel, "Enter")

            # Wait for navigation or content update
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10000))

            title = await page.title()
            text = await page.inner_text("body")
            final_url = page.url

            lines = [ln.strip() for ln in text.splitlines()]
            text = "\n".join(ln for ln in lines if ln)

            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            return f"Title: {title}\nURL: {final_url}\n---\n{text}"
        except Exception as e:
            return f"Form fill failed: {type(e).__name__}: {e}"
        finally:
            await page.close()


class BrowserScreenshotTool(Tool):
    name = "browser_screenshot"
    description = (
        "Navigate to a URL and take a screenshot. Returns the screenshot as a "
        "base64-encoded PNG. Use for visual inspection of web pages, checking "
        "layouts, or capturing page state."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
            "selector": {
                "type": "string",
                "description": "CSS selector to screenshot a specific element. Empty = full page.",
                "default": "",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture the full scrollable page, not just the viewport.",
                "default": False,
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Navigation timeout in milliseconds.",
                "default": 30000,
            },
        },
        "required": ["url"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(
        self,
        url: str,
        selector: str = "",
        full_page: bool = False,
        timeout_ms: int = 30000,
    ) -> str:
        import base64

        browser = await _get_browser()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        try:
            page.set_default_timeout(timeout_ms)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10000))

            if selector:
                element = await page.wait_for_selector(selector, timeout=timeout_ms)
                screenshot_bytes = await element.screenshot()
            else:
                screenshot_bytes = await page.screenshot(full_page=full_page)

            b64 = base64.b64encode(screenshot_bytes).decode()
            size_kb = len(screenshot_bytes) / 1024
            title = await page.title()
            return (
                f"Title: {title}\nURL: {page.url}\n"
                f"Screenshot: {size_kb:.1f} KB PNG (base64 encoded)\n"
                f"data:image/png;base64,{b64}"
            )
        except Exception as e:
            return f"Screenshot failed: {type(e).__name__}: {e}"
        finally:
            await page.close()


class BrowserExtractTool(Tool):
    name = "browser_extract"
    description = (
        "Navigate to a URL and extract structured data using CSS selectors. "
        "Returns a JSON array of extracted elements. "
        "Use for scraping lists, tables, product data, or any repeated elements."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
            "selector": {
                "type": "string",
                "description": "CSS selector for the repeated elements to extract.",
            },
            "fields": {
                "type": "object",
                "description": (
                    "Map of field names to CSS sub-selectors (relative to each matched element). "
                    "Use 'text' as selector to get the element's own text. "
                    "Prefix with '@attr:' to extract an attribute (e.g., '@attr:href')."
                ),
            },
            "max_items": {
                "type": "integer",
                "description": "Maximum number of items to extract.",
                "default": 50,
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Navigation timeout in milliseconds.",
                "default": 30000,
            },
        },
        "required": ["url", "selector", "fields"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(
        self,
        url: str,
        selector: str,
        fields: dict[str, str],
        max_items: int = 50,
        timeout_ms: int = 30000,
    ) -> str:
        browser = await _get_browser()
        page = await browser.new_page()
        try:
            page.set_default_timeout(timeout_ms)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_selector(selector, timeout=timeout_ms)

            elements = await page.query_selector_all(selector)
            results: list[dict[str, Any]] = []

            for el in elements[:max_items]:
                item: dict[str, Any] = {}
                for name, sub_sel in fields.items():
                    if sub_sel == "text":
                        item[name] = (await el.inner_text()).strip()
                    elif sub_sel.startswith("@attr:"):
                        attr = sub_sel[6:]
                        item[name] = await el.get_attribute(attr)
                    else:
                        sub_el = await el.query_selector(sub_sel)
                        if sub_el:
                            item[name] = (await sub_el.inner_text()).strip()
                        else:
                            item[name] = None
                results.append(item)

            return json.dumps(results, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Extract failed: {type(e).__name__}: {e}"
        finally:
            await page.close()
