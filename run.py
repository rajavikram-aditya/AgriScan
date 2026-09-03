"""
run.py — One-command launcher for the AgriScan project.

Starts:
  1. Flask ML API        →  http://localhost:5000
  2. Vite React UI       →  http://localhost:3000
  3. Streamlit app       →  http://localhost:8501   (optional, see STREAMLIT below)

Usage:
  python run.py            # starts Flask + Vite, opens browser
  python run.py --all      # also starts Streamlit main app
  python run.py --pipeline # also starts image_processing_view.py Streamlit viewer
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
AGRISCAN_DIR  = os.path.join(ROOT_DIR, "AgriScan")
PYTHON        = sys.executable

# Detect pnpm / npm
def _find_npm():
    for cmd in ("pnpm", "npm"):
        try:
            subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            return cmd
        except Exception:
            continue
    return "npm"

NPM = _find_npm()

# ─── ANSI colours (Windows 10+ supports them) ─────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
R  = "\033[91m"   # red
B  = "\033[1m"    # bold
RE = "\033[0m"    # reset

# ─── Stream reader: prefix every output line with a tag ───────────────────────
def _stream(proc, tag, colour):
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                print(f"{colour}[{tag}]{RE} {line}")
    except Exception:
        pass

# ─── Launch helper ────────────────────────────────────────────────────────────
def _start(label, colour, cmd, cwd):
    print(f"{colour}{B}  →  Starting {label}{RE}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # On Windows we need shell=True for npm/pnpm batch shims
        shell=(sys.platform == "win32" and cmd[0] in (NPM, "streamlit")),
    )
    t = threading.Thread(target=_stream, args=(proc, label, colour), daemon=True)
    t.start()
    return proc

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    run_streamlit_main     = "--all"      in args
    run_streamlit_pipeline = "--pipeline" in args or "--all" in args

    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("color")

    print(f"\n{G}{B}{'═' * 58}")
    print("        AgriScan — Plant Disease Intelligence")
    print(f"{'═' * 58}{RE}\n")

    procs = []

    # 1. Flask ML Backend
    api_proc = _start(
        "Flask API  :5000", C,
        [PYTHON, os.path.join(AGRISCAN_DIR, "api.py")],
        AGRISCAN_DIR,
    )
    procs.append(("Flask API", api_proc))
    time.sleep(1.5)   # give Flask a head start before Vite hits it

    # 2. Vite React Frontend
    vite_proc = _start(
        "Vite UI   :3000", G,
        [NPM, "run", "dev"],
        ROOT_DIR,
    )
    procs.append(("Vite UI", vite_proc))

    # 3. Streamlit main app (optional)
    if run_streamlit_main:
        st_proc = _start(
            "Streamlit  :8501", Y,
            ["streamlit", "run", os.path.join(AGRISCAN_DIR, "app.py"),
             "--server.port", "8501", "--server.headless", "true"],
            AGRISCAN_DIR,
        )
        procs.append(("Streamlit", st_proc))

    # 4. Streamlit pipeline viewer (optional)
    if run_streamlit_pipeline:
        pl_proc = _start(
            "Pipeline   :8502", Y,
            ["streamlit", "run",
             os.path.join(AGRISCAN_DIR, "image_processing_view.py"),
             "--server.port", "8502", "--server.headless", "true"],
            AGRISCAN_DIR,
        )
        procs.append(("Pipeline viewer", pl_proc))

    # Summary
    print(f"\n{G}{'─' * 58}")
    print(f"  {B}AgriScan is running!{RE}{G}")
    print(f"  React UI  →  {B}http://localhost:3000{RE}{G}")
    print(f"  Flask API →  http://localhost:5000")
    if run_streamlit_main:
        print(f"  Streamlit →  http://localhost:8501")
    if run_streamlit_pipeline:
        print(f"  Pipeline  →  http://localhost:8502")
    print(f"\n  Press {B}Ctrl + C{RE}{G} to stop all servers.")
    print(f"{'─' * 58}{RE}\n")

    # Open browser after a short delay
    def _open():
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:3000")
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()

    # ── Keep alive — watch for early exits ───────────────────────────────────
    try:
        while True:
            time.sleep(1)
            for name, proc in procs:
                if proc.poll() is not None:
                    print(f"\n{R}[{name}] exited unexpectedly (code {proc.returncode}).{RE}")
                    # Don't exit — let user see what happened
    except KeyboardInterrupt:
        print(f"\n{Y}Stopping all servers…{RE}")
    finally:
        for name, proc in procs:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                else:
                    proc.terminate()
        print(f"{G}All servers stopped.{RE}\n")


if __name__ == "__main__":
    main()
