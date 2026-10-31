"""take_screenshot.py - Captures crisp screenshots of the dashboard."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import websockets


async def wait_for_element(ws, selector: str, max_attempts: int = 30):
    """Polls CDP until the specified DOM selector exists."""
    check_script = f"Boolean(document.querySelector('{selector}'))"
    for _ in range(max_attempts):
        msg_id = 99
        await ws.send(json.dumps({
            "id": msg_id,
            "method": "Runtime.evaluate",
            "params": {"expression": check_script, "returnByValue": True}
        }))
        while True:
            resp = json.loads(await ws.recv())
            if resp.get("id") == msg_id:
                val = resp.get("result", {}).get("result", {}).get("value", False)
                if val:
                    return True
                break
        await asyncio.sleep(0.5)
    return False


async def capture_tab_screenshot(ws, tab_name: str, output_path: str, click_script: str = ""):
    if click_script:
        await ws.send(json.dumps({
            "id": 10,
            "method": "Runtime.evaluate",
            "params": {"expression": click_script}
        }))
        await ws.recv()
        await asyncio.sleep(1.5)

    msg_id = 20
    await ws.send(json.dumps({
        "id": msg_id,
        "method": "Page.captureScreenshot",
        "params": {"format": "png", "captureBeyondViewport": False}
    }))

    while True:
        raw_resp = await ws.recv()
        data = json.loads(raw_resp)
        if data.get("id") == msg_id:
            img_b64 = data["result"]["data"]
            img_bytes = base64.b64decode(img_b64)
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            print(f"[{tab_name}] Screenshot saved: {output_path} ({len(img_bytes)} bytes)")
            break


async def run_cdp_session(url: str):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    user_data = os.path.join(os.environ.get("TEMP", "."), "chrome_cdp_profile_final")

    chrome_cmd = [
        chrome_path,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--no-sandbox",
        "--disable-gpu",
        f"--user-data-dir={user_data}",
        "--window-size=1920,1080",
    ]

    chrome_proc = subprocess.Popen(chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)

    try:
        tabs_req = urllib.request.urlopen("http://127.0.0.1:9222/json")
        tabs = json.loads(tabs_req.read().decode("utf-8"))
        page_tabs = [t for t in tabs if t.get("type") == "page"]

        if page_tabs:
            tab_info = page_tabs[0]
        else:
            put_req = urllib.request.Request("http://127.0.0.1:9222/json/new", data=b"", method="PUT")
            tab_info = json.loads(urllib.request.urlopen(put_req).read().decode("utf-8"))

        ws_url = tab_info.get("webSocketDebuggerUrl")
        print(f"Connected to CDP at: {ws_url}")

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await ws.recv()

            print(f"Navigating to {url}...")
            await ws.send(json.dumps({"id": 2, "method": "Page.navigate", "params": {"url": url}}))
            await ws.recv()

            print("Waiting for .metrics-container to mount...")
            ready = await wait_for_element(ws, ".metrics-container", max_attempts=40)
            if not ready:
                print("Warning: .metrics-container did not appear, waiting fixed 5s...")
                await asyncio.sleep(5.0)
            else:
                print(".metrics-container mounted! Waiting 2s for plotly charts to complete rendering...")
                await asyncio.sleep(2.0)

            # 1. Search tree tab (default active tab)
            await capture_tab_screenshot(ws, "Search tree", "results/after_dashboard.png")

            # 2. Comparison tab
            click_comp = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[data-baseweb="tab"]'));
                const t = tabs.find(x => x.innerText.trim() === 'Comparison');
                if (t) t.click();
            })()
            """
            await capture_tab_screenshot(ws, "Comparison", "results/after_dashboard_comparison.png", click_comp)

            # 3. Candidates tab
            click_cand = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[data-baseweb="tab"]'));
                const t = tabs.find(x => x.innerText.trim() === 'Candidates');
                if (t) t.click();
            })()
            """
            await capture_tab_screenshot(ws, "Candidates", "results/after_dashboard_candidates.png", click_cand)

            # 4. Background tab
            click_bg = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[data-baseweb="tab"]'));
                const t = tabs.find(x => x.innerText.trim() === 'Background');
                if (t) t.click();
            })()
            """
            await capture_tab_screenshot(ws, "Background", "results/after_dashboard_background.png", click_bg)

    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
        except Exception:
            chrome_proc.kill()


def main():
    port = "8510"
    st_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/dashboard.py",
        f"--server.port={port}",
        "--server.headless=true",
    ]
    print(f"Starting Streamlit on port {port}...")
    st_proc = subprocess.Popen(st_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait until HTTP port is active
    server_ready = False
    for _ in range(25):
        try:
            urllib.request.urlopen(f"http://localhost:{port}")
            server_ready = True
            break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        print("Failed to reach Streamlit server on port", port)
        st_proc.kill()
        return

    print("Streamlit server is ready!")
    try:
        asyncio.run(run_cdp_session(f"http://localhost:{port}"))
    finally:
        st_proc.terminate()
        try:
            st_proc.wait(timeout=3)
        except Exception:
            st_proc.kill()


if __name__ == "__main__":
    main()
