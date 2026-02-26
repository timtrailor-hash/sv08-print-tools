#!/usr/bin/env python3
"""Long-running printer monitoring daemon with adaptive polling.

Polls every 30s when either printer is printing, every 5min when idle.
Writes to /tmp/printer_status/status.json on each cycle.
Designed to run as a launchd KeepAlive service.
"""

import json
import os
import signal
import sys
import time
from datetime import datetime

OUTPUT_DIR = "/tmp/printer_status"
STATUS_FILE = os.path.join(OUTPUT_DIR, "status.json")
POLL_PRINTING = 15    # seconds between polls when printing
POLL_IDLE = 300       # seconds between polls when idle
ERROR_BACKOFF = 60    # seconds to wait after consecutive errors
MAX_BACKOFF = 300     # max backoff

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

    print(f"[{datetime.now().isoformat()}] Printer daemon starting (PID {os.getpid()})")
    print(f"  Polling: {POLL_PRINTING}s printing / {POLL_IDLE}s idle")
    sys.stdout.flush()

    while _running:
        try:
            data = run_fetch()
            consecutive_errors = 0

            printing = is_printing(data)
            interval = POLL_PRINTING if printing else POLL_IDLE

            sovol_state = data["printers"].get("sovol", {}).get("state", "offline")
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
