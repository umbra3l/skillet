"""Visit the hosted app in a headless browser so Streamlit Cloud keeps it awake.

Streamlit Community Cloud puts an app to sleep after a period without
visitors, and serves its page shell even while the app is asleep, so a plain
HTTP request does not wake it. This opens the app in Chromium, clicks the
"get this app back up" button if the app is asleep, and waits until the app
has rendered its plan panel. It exits non-zero if the app never loads, so a
scheduled run that fails sends a notification.

Run:  python scripts/keep_awake.py [url]
Needs: pip install playwright && playwright install --with-deps chromium
"""
from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://skillet.streamlit.app/"
READY_TEXT = "All five, estimated together"
WAKE_BUTTON = re.compile(r"get this app back up", re.I)
TIMEOUT_S = 300


def app_text(page) -> str:
    """Visible text of the page and every frame (the app renders in an iframe)."""
    texts = []
    for frame in page.frames:
        try:
            texts.append(frame.locator("body").inner_text(timeout=2000))
        except Exception:
            pass
    return "\n".join(texts)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        start, woke = time.time(), False
        while time.time() - start < TIMEOUT_S:
            for frame in page.frames:
                button = frame.get_by_role("button", name=WAKE_BUTTON)
                if button.count():
                    button.first.click()
                    woke = True
                    print("app was asleep: clicked the wake-up button")
            if READY_TEXT in app_text(page):
                print(f"app is up after {time.time() - start:.0f}s"
                      + (" (woken from sleep)" if woke else ""))
                browser.close()
                return 0
            time.sleep(5)
        print(f"app did not load within {TIMEOUT_S}s", file=sys.stderr)
        print(app_text(page)[:1500], file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
