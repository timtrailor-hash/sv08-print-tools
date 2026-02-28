#!/usr/bin/env python3
"""Long-running printer monitoring daemon with adaptive polling.

Polls every 30s when either printer is printing, every 5min when idle.
Writes to /tmp/printer_status/status.json on each cycle.
Designed to run as a launchd KeepAlive service.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

OUTPUT_DIR = "/tmp/printer_status"
STATUS_FILE = os.path.join(OUTPUT_DIR, "status.json")
POLL_PRINTING = 15    # seconds between polls when printing
POLL_IDLE = 300       # seconds between polls when idle
ERROR_BACKOFF = 60    # seconds to wait after consecutive errors
MAX_BACKOFF = 300     # max backoff

# Auto-profiling state
_last_profiled_file = None
_profiling_in_progress = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Graceful shutdown
_running = True

def _shutdown(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def is_printing(data):
    """Check if either printer is actively printing."""
    printers = data.get("printers", {})
    for name, info in printers.items():
        if not info.get("online"):
            continue
        state = (info.get("state") or "").lower()
        if state in ("printing", "running"):
            return True
    return False


def _auto_profile(filename):
    """Run gcode_profile.py in background for a new print file."""
    global _profiling_in_progress
    _profiling_in_progress = True
    try:
        print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: analyzing {filename}")
        sys.stdout.flush()
        result = subprocess.run(
            [sys.executable, "gcode_profile.py", "--file", filename],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: done for {filename}")
            # Auto-enable autospeed
            try:
                from gcode_profile import load_auto_speed, save_auto_speed
                cfg = load_auto_speed()
                if not cfg.get("enabled"):
                    cfg["enabled"] = True
                    cfg.setdefault("mode", "optimal")
                    save_auto_speed(cfg)
                    print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: autospeed enabled")
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: autospeed enable failed: {e}")
        else:
            print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: failed: {result.stderr[:200]}")
        sys.stdout.flush()
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: timed out (>300s)")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] AUTO-PROFILE: error: {e}")
        sys.stdout.flush()
    finally:
        _profiling_in_progress = False


def _check_auto_profile(data, force_check=False):
    """Trigger auto-profiling when a Sovol print is running without a profile.

    Called on every poll. On layer changes, force_check=True bypasses the
    _last_profiled_file cache so we verify the profile still exists.
    """
    global _last_profiled_file
    if _profiling_in_progress:
        return
    sovol = data.get("printers", {}).get("sovol", {})
    if not sovol.get("online"):
        return
    state = (sovol.get("state") or "").lower()
    filename = sovol.get("filename")
    if state != "printing" or not filename:
        return
    # Skip quick check if we already profiled this file (unless layer changed)
    if filename == _last_profiled_file and not force_check:
        return
    # Verify profile actually exists and is valid for this file
    try:
        from gcode_profile import load_profile
        existing = load_profile(filename)
        if existing:
            _last_profiled_file = filename
            return
    except Exception:
        pass
    # No valid profile — auto-generate
    _last_profiled_file = filename  # prevent duplicate triggers
    thread = threading.Thread(target=_auto_profile, args=(filename,), daemon=True)
    thread.start()


def run_fetch():
    """Run one fetch cycle. Returns the data dict."""
    import printer_status_fetch as fetch
    data = {"timestamp": datetime.now().isoformat(), "printers": {}}
    data["printers"]["sovol"] = fetch.fetch_sovol()
    data["printers"]["bambu"] = fetch.fetch_bambu()

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return data


def main():
    consecutive_errors = 0
    last_sovol_layer = None

    print(f"[{datetime.now().isoformat()}] Printer daemon starting (PID {os.getpid()})")
    print(f"  Polling: {POLL_PRINTING}s printing / {POLL_IDLE}s idle")
    print(f"  Auto-profiling: enabled (triggers on new print or missing profile)")
    sys.stdout.flush()

    while _running:
        try:
            data = run_fetch()
            consecutive_errors = 0

            # Auto-profile check — on new print AND on each layer change
            sovol = data.get("printers", {}).get("sovol", {})
            cur_layer = sovol.get("current_layer")
            layer_changed = (cur_layer != last_sovol_layer)
            if layer_changed:
                last_sovol_layer = cur_layer
            _check_auto_profile(data, force_check=layer_changed)

            printing = is_printing(data)
            interval = POLL_PRINTING if printing else POLL_IDLE

            sovol_state = sovol.get("state", "offline")
            bambu_state = data["printers"].get("bambu", {}).get("state", "offline")
            print(f"[{datetime.now().isoformat()}] "
                  f"SV08={sovol_state} A1={bambu_state} "
                  f"next={interval}s")
            sys.stdout.flush()

        except Exception as e:
            consecutive_errors += 1
            interval = min(ERROR_BACKOFF * consecutive_errors, MAX_BACKOFF)
            print(f"[{datetime.now().isoformat()}] ERROR ({consecutive_errors}): {e} "
                  f"retry in {interval}s")
            sys.stdout.flush()

        # Sleep in small increments so SIGTERM is responsive
        deadline = time.time() + interval
        while _running and time.time() < deadline:
            time.sleep(1)

    print(f"[{datetime.now().isoformat()}] Daemon shutting down gracefully")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
