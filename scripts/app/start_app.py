"""Start the FireGuard website and API as one process, and open it in the browser.

    python scripts/app/start_app.py            (or double-click "Start FireGuard.bat")

What it does, each step skipped when there is nothing to do:

    1 database   apply the backend's database migrations (alembic upgrade head)
    2 model      load the best model's registry bundle into the database, once per bundle
                 (results/best_model.json names it; before training finishes this is skipped
                 and the Replay tab is empty, the other tabs still work)
    3 website    npm install (first time only) and npm run build, whenever the website code or
                 results/ changed since the last build (the Results tab is built from results/)
    4 serve      uvicorn on http://127.0.0.1:<port>, serving the API and the built website

Stop it with Ctrl+C (or by closing the window).

Options
    --port 8000         port to serve on
    --no-browser        do not open the browser
    --rebuild           build the website even if nothing changed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"
LOADED_MARK = BACKEND / "artifacts" / ".loaded_bundles.json"

# Inputs of the website build: its own code plus the report files the Results tab imports.
BUILD_INPUTS = [
    FRONTEND / "src",
    FRONTEND / "index.html",
    FRONTEND / "package.json",
    FRONTEND / "vite.config.ts",
    ROOT / "results",
]


def say(msg: str) -> None:
    print(f"[fireguard] {msg}", flush=True)


def run(cmd: list[str], cwd: Path) -> None:
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        sys.exit(f"[fireguard] failed ({r.returncode}): {' '.join(map(str, cmd))}")


def migrate() -> None:
    say("database: applying migrations")
    run([sys.executable, "-m", "alembic", "upgrade", "head"], BACKEND)


def load_best_model() -> None:
    best_path = ROOT / "results" / "best_model.json"
    if not best_path.exists():
        say(
            "model: results/best_model.json not found yet (training not finished); "
            "the Replay tab stays empty until it is"
        )
        return
    run_id = json.loads(best_path.read_text())["published_run_id"]
    bundle = ROOT / "artifacts" / "registry" / run_id / "registry_bundle.json"
    if not bundle.exists():
        say(
            f"model: {bundle.relative_to(ROOT)} not found; finish "
            "'python scripts/train_all.py ... --only publish' first"
        )
        return
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    try:
        loaded = json.loads(LOADED_MARK.read_text())
    except (OSError, ValueError):
        loaded = {}
    if loaded.get(run_id) == digest:
        say(f"model: {run_id} already loaded")
        return
    mode = json.loads(bundle.read_text()).get("mode", "real")
    cmd = [sys.executable, "scripts/load_registry_bundle.py", "--bundle", str(bundle)]
    if mode == "real":
        cmd.append("--publish")
    say(f"model: loading {run_id} ({mode}) into the database")
    run(cmd, BACKEND)
    loaded[run_id] = digest
    LOADED_MARK.parent.mkdir(parents=True, exist_ok=True)
    LOADED_MARK.write_text(json.dumps(loaded, indent=2))


def newest_mtime(paths: list[Path]) -> float:
    newest = 0.0
    for p in paths:
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    newest = max(newest, f.stat().st_mtime)
    return newest


def build_website(force: bool) -> None:
    npm = shutil.which("npm")
    index = DIST / "index.html"
    if npm is None:
        if index.exists():
            say("website: npm not found, serving the existing build")
            return
        sys.exit(
            "[fireguard] npm not found: install Node.js (https://nodejs.org) and retry"
        )
    if not (FRONTEND / "node_modules").exists():
        say("website: installing packages (first time only)")
        run([npm, "ci"], FRONTEND)
    if (
        not force
        and index.exists()
        and index.stat().st_mtime >= newest_mtime(BUILD_INPUTS)
    ):
        say("website: build is up to date")
        return
    say("website: building")
    run([npm, "run", "build"], FRONTEND)


def open_when_ready(url: str) -> None:
    for _ in range(120):
        try:
            with urllib.request.urlopen(url + "/api/v1/health", timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        return
    say(f"open {url}  (Ctrl+C to stop)")
    webbrowser.open(url)


def serve(port: int, browser: bool) -> None:
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, FIREGUARD_FRONTEND_DIST=str(DIST))
    if browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    say(f"serving on {url}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fireguard.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=BACKEND,
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    migrate()
    load_best_model()
    build_website(a.rebuild)
    serve(a.port, not a.no_browser)


if __name__ == "__main__":
    main()
