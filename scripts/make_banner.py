"""Render docs/banner.png (README hero and GitHub social preview, 1280x640 @2x).

Run after scripts/capture_screenshots.py:
    python scripts/make_banner.py
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT / "docs"
SHOT = DOCS / "screenshots" / "01_overview.jpg"
OUT = DOCS / "banner.png"

shot_b64 = base64.b64encode(SHOT.read_bytes()).decode()

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
html, body {{ width: 1280px; height: 640px; overflow: hidden; }}
body {{ font-family: Inter, -apple-system, Helvetica, sans-serif; color: #E7EEF8;
  background: radial-gradient(1200px 600px at 85% 10%, #173352 0%, #0E1929 55%) #0E1929; position: relative; }}
.grid {{ position: absolute; inset: 0; opacity: .25;
  background-image: linear-gradient(#253B54 1px, transparent 1px), linear-gradient(90deg, #253B54 1px, transparent 1px);
  background-size: 40px 40px; mask-image: linear-gradient(90deg, #000 0%, transparent 55%); }}
.left {{ position: absolute; left: 72px; top: 92px; width: 600px; }}
.brand {{ display: flex; align-items: center; gap: 14px; }}
.eyebrow {{ color: #31C6B0; font-weight: 700; letter-spacing: 3px; font-size: 14px; }}
h1 {{ font-size: 62px; line-height: 1.02; letter-spacing: -2px; font-weight: 800; margin: 26px 0 18px; }}
h1 span {{ background: linear-gradient(90deg, #69A8FF, #31C6B0); -webkit-background-clip: text; color: transparent; }}
p {{ color: #A8B7CA; font-size: 20px; line-height: 1.45; }}
.stats {{ display: flex; gap: 12px; margin-top: 34px; flex-wrap: wrap; }}
.stat {{ background: #142238; border: 1px solid #253B54; border-radius: 12px; padding: 12px 16px; }}
.stat b {{ display: block; font-size: 24px; font-weight: 800; color: #F3F7FC; }}
.stat span {{ color: #A8B7CA; font-size: 13px; }}
.shot {{ position: absolute; right: -150px; top: 70px; width: 760px; border-radius: 16px;
  border: 1px solid #253B54; box-shadow: 0 30px 80px rgba(0,0,0,.55);
  transform: perspective(1600px) rotateY(-14deg) rotateX(4deg); }}
.tech {{ position: absolute; left: 72px; bottom: 44px; color: #6f8099; font-size: 14px; letter-spacing: .4px; }}
</style></head><body>
<div class="grid"></div>
<img class="shot" src="data:image/jpeg;base64,{shot_b64}">
<div class="left">
  <div class="brand">
    <svg width="48" height="48" viewBox="0 0 40 40"><defs><linearGradient id="g" x1="0" y1="1" x2="1" y2="0">
    <stop offset="0" stop-color="#69A8FF"/><stop offset="1" stop-color="#31C6B0"/></linearGradient></defs>
    <rect x="1.5" y="1.5" width="37" height="37" rx="10" fill="#142238" stroke="#253B54"/>
    <rect x="9" y="22" width="5" height="9" rx="1.5" fill="url(#g)" opacity=".55"/>
    <rect x="17.5" y="16" width="5" height="15" rx="1.5" fill="url(#g)" opacity=".8"/>
    <rect x="26" y="10" width="5" height="21" rx="1.5" fill="url(#g)"/>
    <path d="M8 17 L18 11 L24 14 L32 7" fill="none" stroke="#FFB36B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <div class="eyebrow">MODEL RISK · CREDIT POLICY</div>
  </div>
  <h1>Credit Risk<br><span>Intelligence</span></h1>
  <p>PD modelling, independent validation, remediation and credit-policy decisioning on 307,511 consumer-loan applicants.</p>
  <div class="stats">
    <div class="stat"><b>0.406</b><span>Gini</span></div>
    <div class="stat"><b>0.304</b><span>KS</span></div>
    <div class="stat"><b>0.061 → 0.016</b><span>overfitting gap fixed</span></div>
    <div class="stat"><b>Basel IRB</b><span>capital & pricing</span></div>
  </div>
</div>
<div class="tech">PostgreSQL · Python · LightGBM · Optuna · WoE scorecard · Basel IRB · Streamlit</div>
</body></html>"""

html_path = DOCS / "_banner.html"
html_path.write_text(HTML, encoding="utf-8")
chrome = next(c for c in [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome"), shutil.which("chromium"),
] if c and Path(c).exists())
subprocess.run([
    chrome, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=2",
    "--window-size=1280,640", "--virtual-time-budget=4000",
    f"--screenshot={OUT}", html_path.as_uri(),
], check=True, capture_output=True)
html_path.unlink()
print("saved", OUT.relative_to(ROOT))
