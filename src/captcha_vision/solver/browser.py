"""
Browser lifecycle, stealth patches, and fast human-like mouse movement.

Key differences from aplesner/Breaking-reCAPTCHAv2's Selenium approach:
  - Chromium via Playwright (faster CDP, no geckodriver dependency).
  - Stealth JS injected on every page to hide the ``navigator.webdriver``
    flag and automation-related properties that Google checks.
  - Mouse moves use a compact Bezier curve (25 steps vs 100) so we spend
    ~200ms per move instead of ~800ms — critical when classifying 16 tiles.
"""
from __future__ import annotations

import random
import time

from playwright.sync_api import BrowserContext, Frame, Page, Playwright

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# JS snippet injected before any page loads to mask Playwright automation
# signals.  This is the single biggest factor (after cookies) in whether
# Google serves an easy checkbox-only pass or a series of hard challenges.
_STEALTH_JS = """
() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
    });
    window.chrome = { runtime: {} };
    const origQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (params) =>
        params.name === 'notifications'
            ? Promise.resolve({ state: Notification.permission })
            : origQuery(params);
}
"""


# ---------------------------------------------------------------------------
# Mouse movement — fast Bezier
# ---------------------------------------------------------------------------

def _bezier_path(
    start: tuple[float, float],
    end: tuple[float, float],
    n_steps: int = 25,
) -> list[tuple[float, float]]:
    """Cubic Bezier with two random control points (compact 25-step)."""
    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    c1 = (mx + random.uniform(-50, 50), my + random.uniform(-50, 50))
    c2 = (mx + random.uniform(-50, 50), my + random.uniform(-50, 50))
    points: list[tuple[float, float]] = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        u = 1.0 - t
        x = u**3*start[0] + 3*u**2*t*c1[0] + 3*u*t**2*c2[0] + t**3*end[0]
        y = u**3*start[1] + 3*u**2*t*c1[1] + 3*u*t**2*c2[1] + t**3*end[1]
        points.append((x, y))
    return points


def human_move(page: Page, x: float, y: float) -> None:
    """Move mouse along a fast Bezier curve (~150-250ms total)."""
    raw = page.evaluate("() => [window._mx ?? 200, window._my ?? 300]")
    start: tuple[float, float] = (float(raw[0]), float(raw[1]))
    for px, py in _bezier_path(start, (x, y)):
        page.mouse.move(px, py)
        time.sleep(random.uniform(0.003, 0.008))
    page.evaluate(f"() => {{ window._mx = {x}; window._my = {y}; }}")


def human_click(page: Page, x: float, y: float) -> None:
    """Move to (x, y) then left-click with realistic micro-delays."""
    human_move(page, x, y)
    time.sleep(random.uniform(0.03, 0.09))
    page.mouse.click(x, y)
    time.sleep(random.uniform(0.05, 0.15))


# ---------------------------------------------------------------------------
# Browser launch
# ---------------------------------------------------------------------------

def launch(
    pw: Playwright,
    *,
    headless: bool = False,
    user_data_dir: str | None = None,
) -> tuple[BrowserContext, Page]:
    """
    Launch Chromium with stealth patches and return (context, page).

    Using ``user_data_dir`` with a real Chrome profile is the #1 factor
    in getting easy checkbox-only passes from Google's risk engine.
    """
    launch_kwargs: dict = {
        "headless": headless,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
        ],
    }
    if user_data_dir:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir,
            **launch_kwargs,
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
    else:
        browser = pw.chromium.launch(**launch_kwargs)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=_USER_AGENT,
        )
        page = ctx.new_page()

    # Inject stealth JS before any navigation
    ctx.add_init_script(_STEALTH_JS)
    return ctx, page


# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------

def navigate_and_open(page: Page, url: str) -> bool:
    """
    Navigate to *url*, click the reCAPTCHA checkbox, and wait for the
    challenge iframe.

    Returns True  → challenge appeared (need to solve it).
    Returns False → checkbox passed without a challenge (lucky pass).
    """
    page.goto(url, wait_until="networkidle")
    time.sleep(random.uniform(0.8, 1.5))

    checkbox_fl = page.frame_locator('iframe[title="reCAPTCHA"]')
    checkbox_fl.locator(".recaptcha-checkbox-border").click()
    time.sleep(random.uniform(2.5, 4.0))

    try:
        page.wait_for_selector(
            'iframe[src*="bframe"]',
            timeout=8_000,
        )
        time.sleep(random.uniform(0.5, 1.0))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def get_challenge_frame(page: Page) -> Frame | None:
    """Return the live challenge Frame (bframe), or None if not visible."""
    for frame in page.frames:
        if "bframe" in (frame.url or ""):
            return frame
    return None


def get_checkbox_frame(page: Page) -> Frame | None:
    """Return the reCAPTCHA checkbox Frame (anchor)."""
    for frame in page.frames:
        if "anchor" in (frame.url or ""):
            return frame
    return None
