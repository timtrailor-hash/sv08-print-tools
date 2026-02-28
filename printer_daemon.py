#!/usr/bin/env python3
"""Long-running printer monitoring daemon with adaptive polling.

Polls every 30s when either printer is printing, every 5min when idle.
Writes to /tmp/printer_status/status.json on each cycle.
Designed to run as a launchd KeepAlive service.
"""

import configparser
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

OUTPUT_DIR = "/tmp/printer_status"
STATUS_FILE = os.path.join(OUTPUT_DIR, "status.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "print_checkpoint.json")
ALERT_FILE = os.path.join(OUTPUT_DIR, "printer_alerts.jsonl")
POLL_PRINTING = 15    # seconds between polls when printing
POLL_IDLE = 300       # seconds between polls when idle
ERROR_BACKOFF = 60    # seconds to wait after consecutive errors
MAX_BACKOFF = 300     # max backoff

# Auto-recovery constants
MAX_RECOVERY_ATTEMPTS = 3       # per print job
RECOVERY_SETTLE_S = 10          # wait for Klipper to settle after crash
RECOVERY_RECONNECT_S = 30       # wait for Klipper to reconnect after FIRMWARE_RESTART

# CPU/thermal alert thresholds
CPU_LOAD_WARN = 1.2
CPU_TEMP_WARN = 70.0  # °C

# Auto-profiling state
_last_profiled_file = None
_profiling_in_progress = False

# saved_variables.cfg validation
_SAVED_VAR_CHECK_INTERVAL = 300  # 5 minutes
_last_saved_var_check = 0

# Error alerting state
_last_alerted_state = {"sv08": None, "a1": None}

# Auto-recovery state
_recovery_attempts = 0
_last_print_file = None           # reset recovery counter on new print
_recovery_in_progress = False

# CPU/thermal monitoring state
_last_cpu_alert_time = 0
_CPU_ALERT_COOLDOWN = 300  # don't re-alert within 5 min

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


def _save_checkpoint(data):
    """Save print position checkpoint for crash recovery.

    Writes layer, progress, Z height, filename, speed, and timestamp
    every poll cycle while printing. Used to determine where a print
    was when a crash/shutdown occurred.
    """
    sovol = data.get("printers", {}).get("sovol", {})
    state = (sovol.get("state") or "").lower()
    if state != "printing":
        return

    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "filename": sovol.get("filename", ""),
        "state": state,
        "progress_pct": sovol.get("progress", 0),
        "current_layer": sovol.get("current_layer", 0),
        "total_layers": sovol.get("total_layers", 0),
        "z_position": sovol.get("z_position", 0),
        "print_duration_s": sovol.get("print_duration", 0),
        "speed_factor": sovol.get("speed_factor", 1.0),
        "bed_temp": sovol.get("bed_temp", 0),
        "nozzle_temp": sovol.get("nozzle_temp", 0),
        "filament_used_mm": sovol.get("filament_used_mm", 0),
    }

    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] CHECKPOINT: write error: {e}")
        sys.stdout.flush()


def _write_alert(alert_dict):
    """Append an alert to the JSONL file for conversation_server to broadcast."""
    try:
        with open(ALERT_FILE, "a") as f:
            f.write(json.dumps(alert_dict) + "\n")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ALERT: write error: {e}")
        sys.stdout.flush()


def _get_moonraker_base():
    """Return the Moonraker base URL for the SV08."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from printer_config import SOVOL_IP, MOONRAKER_PORT
        return f"http://{SOVOL_IP}:{MOONRAKER_PORT}"
    except ImportError:
        return None


def _moonraker_post(path, data=None, timeout=10):
    """Send a POST request to Moonraker. Returns response dict or None."""
    base = _get_moonraker_base()
    if not base:
        return None
    url = f"{base}{path}"
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] MOONRAKER POST {path} failed: {e}")
        sys.stdout.flush()
        return None


def _moonraker_get(path, timeout=5):
    """Send a GET request to Moonraker. Returns response dict or None."""
    base = _get_moonraker_base()
    if not base:
        return None
    url = f"{base}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _check_cpu_thermal():
    """Query Moonraker for host CPU load and temperature, alert if high.

    Uses /machine/proc_stats which returns cpu_temp and moonraker_stats
    with per-sample cpu_usage percentages. We derive sysload from the
    latest sample's cpu_usage (0-100%) and the cpu count.
    """
    global _last_cpu_alert_time
    now = time.time()

    proc = _moonraker_get("/machine/proc_stats")
    if not proc:
        return
    result = proc.get("result", {})
    cpu_temp = result.get("cpu_temp", 0)

    # Derive sysload from moonraker_stats cpu_usage percentage
    sysload = 0
    cpu_count = result.get("system_info", {}).get("cpu_info", {}).get("cpu_count", 4)
    ms = result.get("moonraker_stats", [])
    if ms:
        # cpu_usage is 0-100% across all cores; convert to load-average-like number
        sysload = ms[-1].get("cpu_usage", 0) / 100.0 * cpu_count

    alerts = []
    if sysload > CPU_LOAD_WARN:
        alerts.append(f"sysload={sysload:.2f} (>{CPU_LOAD_WARN})")
    if cpu_temp > CPU_TEMP_WARN:
        alerts.append(f"cpu_temp={cpu_temp:.1f}°C (>{CPU_TEMP_WARN}°C)")

    if alerts and (now - _last_cpu_alert_time > _CPU_ALERT_COOLDOWN):
        _last_cpu_alert_time = now
        msg = f"SV08 HOST WARNING: {', '.join(alerts)}"
        print(f"[{datetime.now().isoformat()}] {msg}")
        sys.stdout.flush()
        _write_alert({
            "type": "printer_alert",
            "printer": "sv08",
            "event": "cpu_thermal_warning",
            "message": msg,
            "sysload": sysload,
            "cpu_temp": cpu_temp,
        })


def _validate_saved_variables():
    """Check saved_variables.cfg for corruption before Klipper does.

    Fetches the file from Moonraker, parses with ConfigParser, and
    auto-fixes duplicate keys by keeping the last value. Alerts on
    any corruption detected.
    """
    global _last_saved_var_check
    now = time.time()
    if now - _last_saved_var_check < _SAVED_VAR_CHECK_INTERVAL:
        return
    _last_saved_var_check = now

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from printer_config import SOVOL_IP, MOONRAKER_PORT
    except ImportError:
        return

    url = f"http://{SOVOL_IP}:{MOONRAKER_PORT}/server/files/config/saved_variables.cfg"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            content = r.read().decode("utf-8", errors="replace")
    except Exception:
        return  # Printer offline — skip

    # Try parsing — ConfigParser raises on duplicates by default
    cp = configparser.ConfigParser(strict=True)
    try:
        cp.read_string(content)
        return  # File is valid
    except (configparser.DuplicateOptionError, configparser.Error) as e:
        print(f"[{datetime.now().isoformat()}] CORRUPTION DETECTED in saved_variables.cfg: {e}")
        sys.stdout.flush()

    # Auto-fix: parse non-strictly, rewrite without duplicates
    cp_fix = configparser.ConfigParser(strict=False)
    try:
        cp_fix.read_string(content)
        fixed = io.StringIO()
        cp_fix.write(fixed)
        fixed_content = fixed.getvalue()

        # Upload the fixed file back via Moonraker
        boundary = "----SavedVarFixBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="saved_variables.cfg"\r\n'
            f"Content-Type: text/plain\r\n"
            f'\r\n{fixed_content}\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="root"; \r\n'
            f"\r\nconfig\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        upload_url = f"http://{SOVOL_IP}:{MOONRAKER_PORT}/server/files/upload"
        req = urllib.request.Request(
            upload_url, data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[{datetime.now().isoformat()}] AUTO-FIX: saved_variables.cfg rewritten (duplicates removed)")
        sys.stdout.flush()
    except Exception as fix_err:
        print(f"[{datetime.now().isoformat()}] AUTO-FIX FAILED: {fix_err}")
        sys.stdout.flush()

    # Write alert for conversation_server to broadcast
    alert = {
        "type": "printer_alert",
        "printer": "sv08",
        "event": "config_corruption",
        "message": f"SV08 saved_variables.cfg was corrupted (duplicate keys). Auto-fixed.",
    }
    _write_alert(alert)


def _attempt_recovery():
    """Attempt auto-recovery after a Klipper crash/shutdown.

    Sequence: wait → FIRMWARE_RESTART → verify ready → reheat bed → alert.
    Returns True if recovery succeeded, False otherwise.
    """
    global _recovery_attempts, _recovery_in_progress
    _recovery_in_progress = True

    try:
        _recovery_attempts += 1
        attempt = _recovery_attempts
        print(f"[{datetime.now().isoformat()}] RECOVERY: attempt {attempt}/{MAX_RECOVERY_ATTEMPTS}")
        sys.stdout.flush()

        # Step 1: Wait for Klipper to settle
        print(f"[{datetime.now().isoformat()}] RECOVERY: waiting {RECOVERY_SETTLE_S}s for Klipper to settle")
        sys.stdout.flush()
        time.sleep(RECOVERY_SETTLE_S)

        # Step 2: Send FIRMWARE_RESTART
        print(f"[{datetime.now().isoformat()}] RECOVERY: sending FIRMWARE_RESTART")
        sys.stdout.flush()
        result = _moonraker_post("/printer/firmware_restart")
        if result is None:
            print(f"[{datetime.now().isoformat()}] RECOVERY: FIRMWARE_RESTART failed — Moonraker unreachable")
            sys.stdout.flush()
            return False

        # Step 3: Wait for Klipper to reconnect
        print(f"[{datetime.now().isoformat()}] RECOVERY: waiting {RECOVERY_RECONNECT_S}s for reconnect")
        sys.stdout.flush()
        time.sleep(RECOVERY_RECONNECT_S)

        # Step 4: Verify Klipper state is 'ready'
        info = _moonraker_get("/printer/info")
        if not info:
            print(f"[{datetime.now().isoformat()}] RECOVERY: cannot reach Moonraker after restart")
            sys.stdout.flush()
            return False

        state = info.get("result", {}).get("state", "unknown")
        if state != "ready":
            print(f"[{datetime.now().isoformat()}] RECOVERY: Klipper state is '{state}', not 'ready'")
            sys.stdout.flush()
            return False

        print(f"[{datetime.now().isoformat()}] RECOVERY: Klipper is ready!")
        sys.stdout.flush()

        # Step 5: Reheat bed from checkpoint
        bed_temp = 60  # default
        checkpoint_info = ""
        try:
            with open(CHECKPOINT_FILE) as f:
                cp = json.load(f)
            bed_temp = int(cp.get("bed_temp", 60)) or 60
            checkpoint_info = (f"Last checkpoint: {cp.get('progress_pct', 0):.1f}% "
                             f"(layer {cp.get('current_layer', '?')}/{cp.get('total_layers', '?')}, "
                             f"Z={cp.get('z_position', '?')}mm, "
                             f"file={cp.get('filename', '?')})")
        except Exception:
            checkpoint_info = "No checkpoint available"

        print(f"[{datetime.now().isoformat()}] RECOVERY: reheating bed to {bed_temp}°C")
        sys.stdout.flush()
        _moonraker_post("/printer/gcode/script",
                        {"script": f"M140 S{bed_temp}"})

        # Step 6: Send alert with recovery details
        msg = (f"SV08 Max RECOVERED (attempt {attempt}). "
               f"Bed reheating to {bed_temp}°C. {checkpoint_info}")
        print(f"[{datetime.now().isoformat()}] RECOVERY: {msg}")
        sys.stdout.flush()
        _write_alert({
            "type": "printer_alert",
            "printer": "sv08",
            "event": "auto_recovery_success",
            "message": msg,
        })
        return True

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] RECOVERY: unexpected error: {e}")
        sys.stdout.flush()
        return False
    finally:
        _recovery_in_progress = False


def _check_error_states(data):
    """Detect printer error/shutdown states and attempt auto-recovery.

    Checks for: error, shutdown, klippy_shutdown, disconnect.
    Only alerts once per error state transition (not every poll).
    On SV08 crash: attempts FIRMWARE_RESTART + reheat up to MAX_RECOVERY_ATTEMPTS times.
    """
    global _recovery_attempts
    error_states = {"error", "shutdown", "klippy_shutdown", "klippy_disconnect"}

    # Check SV08
    sovol = data.get("printers", {}).get("sovol", {})
    sv08_state = (sovol.get("state") or "unknown").lower()

    if sv08_state in error_states and _last_alerted_state["sv08"] != sv08_state:
        _last_alerted_state["sv08"] = sv08_state

        # Try to get more detail from Moonraker
        error_msg = ""
        try:
            import urllib.request
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from printer_config import SOVOL_IP, MOONRAKER_PORT
            url = f"http://{SOVOL_IP}:{MOONRAKER_PORT}/printer/info"
            with urllib.request.urlopen(url, timeout=5) as r:
                info = json.loads(r.read())
            error_msg = info.get("result", {}).get("state_message", "")
        except Exception:
            pass

        # Read last checkpoint for context
        checkpoint_info = ""
        try:
            with open(CHECKPOINT_FILE) as f:
                cp = json.load(f)
            checkpoint_info = (f" Last checkpoint: {cp.get('progress_pct', 0):.1f}% "
                             f"(layer {cp.get('current_layer', '?')}/{cp.get('total_layers', '?')}, "
                             f"Z={cp.get('z_position', '?')}mm)")
        except Exception:
            pass

        short_error = error_msg.split("\n")[0][:100] if error_msg else sv08_state
        alert = {
            "type": "printer_alert",
            "printer": "sv08",
            "event": "firmware_error",
            "message": f"SV08 Max FIRMWARE ERROR: {short_error}.{checkpoint_info}",
        }
        _write_alert(alert)
        print(f"[{datetime.now().isoformat()}] ALERT: SV08 error state '{sv08_state}': {short_error}")
        sys.stdout.flush()

        # Auto-recovery: attempt FIRMWARE_RESTART + reheat
        if not _recovery_in_progress and _recovery_attempts < MAX_RECOVERY_ATTEMPTS:
            print(f"[{datetime.now().isoformat()}] RECOVERY: initiating auto-recovery "
                  f"({_recovery_attempts}/{MAX_RECOVERY_ATTEMPTS} attempts used)")
            sys.stdout.flush()
            recovered = _attempt_recovery()
            if recovered:
                _last_alerted_state["sv08"] = None  # allow re-detection if it crashes again
            else:
                msg = (f"SV08 auto-recovery FAILED (attempt {_recovery_attempts}/"
                       f"{MAX_RECOVERY_ATTEMPTS}). Manual intervention may be required.")
                print(f"[{datetime.now().isoformat()}] {msg}")
                sys.stdout.flush()
                _write_alert({
                    "type": "printer_alert",
                    "printer": "sv08",
                    "event": "recovery_failed",
                    "message": msg,
                })
        elif _recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
            msg = (f"SV08 Max: all {MAX_RECOVERY_ATTEMPTS} recovery attempts exhausted. "
                   f"Manual intervention required.")
            print(f"[{datetime.now().isoformat()}] {msg}")
            sys.stdout.flush()
            _write_alert({
                "type": "printer_alert",
                "printer": "sv08",
                "event": "recovery_exhausted",
                "message": msg,
            })
    elif sv08_state not in error_states:
        _last_alerted_state["sv08"] = None

    # Check A1
    bambu = data.get("printers", {}).get("bambu", {})
    a1_state = (bambu.get("state") or "unknown").lower()

    if a1_state in error_states and _last_alerted_state["a1"] != a1_state:
        _last_alerted_state["a1"] = a1_state
        alert = {
            "type": "printer_alert",
            "printer": "a1",
            "event": "firmware_error",
            "message": f"Bambu A1 ERROR: printer in {a1_state} state",
        }
        _write_alert(alert)
        print(f"[{datetime.now().isoformat()}] ALERT: A1 error state '{a1_state}'")
        sys.stdout.flush()
    elif a1_state not in error_states:
        _last_alerted_state["a1"] = None


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
    global _recovery_attempts, _last_print_file
    consecutive_errors = 0
    last_sovol_layer = None

    print(f"[{datetime.now().isoformat()}] Printer daemon starting (PID {os.getpid()})")
    print(f"  Polling: {POLL_PRINTING}s printing / {POLL_IDLE}s idle")
    print(f"  Auto-profiling: enabled (triggers on new print or missing profile)")
    print(f"  Auto-recovery: enabled (max {MAX_RECOVERY_ATTEMPTS} attempts per print)")
    print(f"  CPU monitoring: sysload>{CPU_LOAD_WARN}, temp>{CPU_TEMP_WARN}°C")
    sys.stdout.flush()

    while _running:
        try:
            data = run_fetch()
            consecutive_errors = 0

            # Save position checkpoint while printing
            _save_checkpoint(data)

            # Check for firmware errors/shutdowns — triggers auto-recovery
            _check_error_states(data)

            # Reset recovery counter when a new print starts
            sovol = data.get("printers", {}).get("sovol", {})
            cur_file = sovol.get("filename", "")
            if cur_file and cur_file != _last_print_file:
                _last_print_file = cur_file
                if _recovery_attempts > 0:
                    print(f"[{datetime.now().isoformat()}] RECOVERY: counter reset (new print: {cur_file})")
                    sys.stdout.flush()
                _recovery_attempts = 0

            # Validate saved_variables.cfg periodically (prevent race condition corruption)
            _validate_saved_variables()

            # CPU/thermal monitoring while printing
            if is_printing(data):
                _check_cpu_thermal()

            # Auto-profile check — on new print AND on each layer change
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
