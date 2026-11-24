"""Capture high-resolution dashboard screenshots for the README.

Start the app first (`streamlit run app.py`), then run:
    python scripts/capture_screenshots.py [--url http://localhost:8501]

Drives headless Chrome over the DevTools protocol and writes JPEGs to docs/screenshots/.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PORT = 9333
WIDTH, HEIGHT, SCALE = 1440, 960, 2

# (file name, tab label, text of the element to scroll to or None for the top)
SHOTS = [
    ("01_overview.jpg", "Overview", None),
    ("02_portfolio_insights.jpg", "Portfolio Insights", None),
    ("03_diagnostics.jpg", "Model Comparison", "Credit-risk diagnostics"),
    ("04_credit_strategy.jpg", "Credit Strategy", "Approval-rate trade-off"),
    ("05_policy_simulator.jpg", "Policy Simulator", "Policy simulator"),
    ("06_swap_set.jpg", "Policy Simulator", "Swap-set analysis"),
    ("07_applicant_explorer.jpg", "Applicant Explorer", "Browse"),
    ("08_validation_scorecard.jpg", "Model Validation", "Validation scorecard"),
    ("09_capital_pricing.jpg", "Capital & Stress", "Capital, expected loss"),
    ("10_stress_test.jpg", "Capital & Stress", "What happens in a downturn?"),
    ("11_remediation.jpg", "Model Validation", "From v1.0 to v1.1"),
    ("12_woe_scorecard.jpg", "Model Comparison", "WoE / IV application scorecard"),
]


def find_chrome():
    for candidate in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Google Chrome / Chromium not found")


class Page:
    def __init__(self, ws_url):
        self.ws = connect(ws_url, max_size=None, legacy=True)
        self.ids = itertools.count(1)

    def call(self, method, **params):
        msg_id = next(self.ids)
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expression):
        res = self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        return res.get("result", {}).get("value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    profile = tempfile.mkdtemp(prefix="cri-chrome-")
    chrome = subprocess.Popen([
        find_chrome(), "--headless=new", f"--remote-debugging-port={PORT}",
        f"--user-data-dir={profile}", "--hide-scrollbars",
        f"--window-size={WIDTH},{HEIGHT}", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        page = Page(ws_url)
        page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT,
                  deviceScaleFactor=SCALE, mobile=False)
        page.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        page.call("Page.navigate", url=args.url)
        for _ in range(60):
            time.sleep(1)
            if page.js("document.querySelectorAll('[role=\"tab\"]').length") and page.js(
                "document.querySelectorAll('.js-plotly-plot').length > 10"
            ):
                break
        time.sleep(3)
        page.js("document.querySelector('[data-testid=\"stHeader\"]')?.style.setProperty('display','none')")

        for name, tab, anchor in SHOTS:
            show_sidebar = name.startswith("01_")
            page.js(f"""(() => {{
                const sb = document.querySelector('[data-testid="stSidebar"]');
                if (sb) sb.style.display = {json.dumps("" if show_sidebar else "none")};
            }})()""")
            page.js(f"""[...document.querySelectorAll('[role="tab"]')]
                .find(t => t.innerText.includes({json.dumps(tab)}))?.click()""")
            time.sleep(2.5)
            if anchor:
                page.js(f"""(() => {{
                    const el = [...document.querySelectorAll('h3, p, label, div')]
                      .filter(e => e.offsetParent !== null && e.children.length < 3)
                      .find(e => e.innerText && e.innerText.trim().startsWith({json.dumps(anchor)}));
                    if (el) el.scrollIntoView({{block: 'start'}});
                    const main = document.querySelector('[data-testid="stMain"]');
                    if (main) main.scrollBy(0, -24);
                }})()""")
            else:
                page.js("document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0, 0); window.scrollTo(0, 0)")
            time.sleep(1.5)
            shot = page.call("Page.captureScreenshot", format="jpeg", quality=88)
            (OUT / name).write_bytes(base64.b64decode(shot["data"]))
            print("saved", (OUT / name).relative_to(ROOT))
    finally:
        chrome.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
