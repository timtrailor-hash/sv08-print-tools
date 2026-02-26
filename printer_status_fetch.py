#!/usr/bin/env python3
"""Fetch live status from both printers and save images/data for inline display."""

import json
import logging
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

# Structured logging — replaces silent except:pass blocks
_log_path = os.path.join("/tmp/printer_status", "printer.log")
_handler = RotatingFileHandler(_log_path, maxBytes=2_000_000, backupCount=3)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
log = logging.getLogger("printer_fetch")
log.addHandler(_handler)
log.setLevel(logging.WARNING)

def _log_exc(context):
    """Log exception at WARNING level with context. Use in except blocks."""
    log.warning("%s failed", context, exc_info=True)


def _fetch_url(url, timeout=5, retries=3, backoff=1.0):
    """Fetch URL with exponential backoff. Returns response bytes or raises."""
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)  # 1s, 2s, 4s
                log.info("Retry %d/%d for %s after %.1fs (%s)",
                         attempt + 1, retries - 1, url[:80], wait, e)
                time.sleep(wait)
    raise last_err


OUTPUT_DIR = "/tmp/printer_status"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# When run through a symlink, Python resolves the real path and sets sys.path[0]
# to the repo directory. printer_config.py lives in the parent dir (alongside the
# symlink), so we need to add that to sys.path for the import to succeed.
_real_dir = os.path.dirname(os.path.realpath(__file__))
_parent_dir = os.path.dirname(_real_dir)
if (os.path.isfile(os.path.join(_parent_dir, "printer_config.py"))
        and _parent_dir not in sys.path):
    sys.path.insert(0, _parent_dir)

try:
    from printer_config import (SOVOL_IP, SOVOL_WIFI_IP, SOVOL_CAMERA_PORT,
                                MOONRAKER_PORT, BAMBU_IP, BAMBU_SERIAL,
                                BAMBU_ACCESS_CODE, BAMBU_MQTT_PORT)
except ImportError:
    SOVOL_IP = "[REDACTED — see printer_config.example.py]"
    SOVOL_WIFI_IP = "[REDACTED — see printer_config.example.py]"
    SOVOL_CAMERA_PORT = 8081
    MOONRAKER_PORT = 7125
    BAMBU_IP = "[REDACTED — see printer_config.example.py]"
    BAMBU_SERIAL = "[REDACTED — see printer_config.example.py]"
    BAMBU_ACCESS_CODE = "[REDACTED — see printer_config.example.py]"
    BAMBU_MQTT_PORT = 8883

ETA_SNAPSHOTS_FILE = os.path.join(OUTPUT_DIR, "eta_snapshots.jsonl")
_SNAPSHOT_INTERVAL_S = 300  # 5 minutes

# ── ETA method change / connection failure alerting ──
# Tracks last known method per printer so we can detect transitions
# (e.g. profile → pace+slicer) and write an alert file for the
# conversation_server to pick up and push to the iOS app.
ALERT_FILE = os.path.join(OUTPUT_DIR, "printer_alerts.jsonl")
_prev_eta_method = {}  # {printer_key: "profile(...)" or "pace+slicer(...)"}
_prev_fetch_ok = {"sovol": True, "bambu": True}  # track connection health


def _method_family(method_str):
    """Extract the method family from a full method string.

    'profile(cal,layers=247,α=0.36)' → 'profile'
    'pace+slicer(α=0.35)'           → 'pace+slicer'
    'physics(α=0.40)'               → 'physics'
    'bambu_mqtt'                     → 'bambu_mqtt'
    """
    if not method_str:
        return ""
    paren = method_str.find("(")
    return method_str[:paren] if paren > 0 else method_str


def _write_alert(alert_dict):
    """Append a printer alert to the alert file for conversation_server."""
    alert_dict["ts"] = datetime.now().isoformat()
    alert_dict["ts_epoch"] = time.time()
    try:
        with open(ALERT_FILE, "a") as f:
            f.write(json.dumps(alert_dict) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        _log_exc("write_alert")


def check_eta_method_change(printer_key, result):
    """Detect ETA method family changes and write an alert.

    Called after ETA calculation. If the method switches (e.g. profile →
    pace+slicer), writes an alert that conversation_server broadcasts.
    """
    method = result.get("eta_method", "")
    if not method:
        return
    family = _method_family(method)
    prev = _prev_eta_method.get(printer_key)

    if prev and prev != family:
        if family in ("pace+slicer", "physics"):
            # Degraded — profile dropped out
            msg = (f"{'SV08 Max' if 'sovol' in printer_key else 'Bambu A1'}: "
                   f"ETA switched from {prev} to {family} — "
                   f"profile calibration lost, predictions less accurate")
            event = "eta_method_degraded"
        else:
            # Recovered — back to profile
            msg = (f"{'SV08 Max' if 'sovol' in printer_key else 'Bambu A1'}: "
                   f"ETA recovered to {family} (was {prev})")
            event = "eta_method_recovered"

        _write_alert({
            "type": "printer_alert",
            "printer": "sv08" if "sovol" in printer_key else "a1",
            "event": event,
            "message": msg,
            "old_method": prev,
            "new_method": family,
        })

    _prev_eta_method[printer_key] = family


def check_connection_health(printer_key, success, error_msg=""):
    """Track connection health and alert on failure/recovery."""
    was_ok = _prev_fetch_ok.get(printer_key, True)
    name = "SV08 Max" if printer_key == "sovol" else "Bambu A1"

    if was_ok and not success:
        _write_alert({
            "type": "printer_alert",
            "printer": "sv08" if printer_key == "sovol" else "a1",
            "event": "connection_lost",
            "message": f"{name}: Connection lost — {error_msg[:100]}",
        })
    elif not was_ok and success:
        _write_alert({
            "type": "printer_alert",
            "printer": "sv08" if printer_key == "sovol" else "a1",
            "event": "connection_restored",
            "message": f"{name}: Connection restored",
        })

    _prev_fetch_ok[printer_key] = success


def _save_eta_snapshot(result):
    """Save an immutable ETA prediction snapshot every 5 minutes.

    Once written, these lines are never modified. The ETA graph reads
    them directly so past predictions can't be retroactively adjusted.
    """
    now = datetime.now()
    now_ts = now.timestamp()
    cur_file = result.get("filename", "")
    progress = result.get("progress", 0)

    if not cur_file or progress <= 0:
        return

    # Check if 5 minutes have passed since the last snapshot for THIS file
    # (scan backwards through recent entries to find the right printer)
    if os.path.exists(ETA_SNAPSHOTS_FILE):
        try:
            with open(ETA_SNAPSHOTS_FILE, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size > 0:
                    # Read last 4KB to find recent entries for this file
                    f.seek(max(0, size - 4096))
                    lines = f.read().decode().strip().split("\n")
                    for line in reversed(lines):
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue
                        if entry.get("filename") == cur_file:
                            if (now_ts - entry.get("snapshot_ts", 0)
                                    < _SNAPSHOT_INTERVAL_S):
                                return  # Too soon for this printer
                            break  # Found last entry for this file, old enough
                    # If no entry found for cur_file, fall through to save
        except Exception:
            log.debug("suppressed", exc_info=True)

    # Parse remaining time from the final remaining_str (e.g. "12h 34m")
    remaining_str = result.get("remaining_str", "")
    remaining_s = 0
    try:
        parts = remaining_str.replace("h", "").replace("m", "").split()
        if len(parts) == 2:
            remaining_s = int(parts[0]) * 3600 + int(parts[1]) * 60
        elif len(parts) == 1:
            remaining_s = int(parts[0]) * 60
    except (ValueError, IndexError):
        pass

    if remaining_s <= 0:
        return

    elapsed_s = result.get("print_duration", 0)
    finish_ts = now_ts + remaining_s

    snapshot = {
        "snapshot_ts": round(now_ts),
        "snapshot_str": now.strftime("%a %d %b %H:%M"),
        "filename": cur_file,
        "progress": round(progress, 1),
        "elapsed_s": round(elapsed_s),
        "remaining_s": round(remaining_s),
        "predicted_finish_ts": round(finish_ts),
        "predicted_finish_str": datetime.fromtimestamp(finish_ts).strftime(
            "%a %d %b %H:%M"),
        "predicted_total_s": round(elapsed_s + remaining_s),
        "eta_method": result.get("eta_method", ""),
        "eta_confidence": result.get("eta_confidence", ""),
        "alpha": result.get("alpha"),
        "speed_factor": result.get("speed_factor", 1.0),
    }

    try:
        with open(ETA_SNAPSHOTS_FILE, "a") as f:
            f.write(json.dumps(snapshot) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        log.debug("suppressed", exc_info=True)


def fetch_sovol():
    """Fetch Sovol SV08 Max status via Moonraker API."""
    result = {"printer": "Sovol SV08 Max", "online": False}

    # Try Ethernet first, fall back to WiFi if unreachable
    for ip in [SOVOL_IP, SOVOL_WIFI_IP]:
        base = f"http://{ip}:{MOONRAKER_PORT}"
        url = f"{base}/printer/objects/query?print_stats&display_status&heater_bed&extruder&fan&gcode_move&motion_report"
        try:
            data = json.loads(_fetch_url(url, timeout=5))["result"]["status"]
            result["online"] = True
            result["connection"] = "ethernet" if ip == SOVOL_IP else "wifi_fallback"
            break
        except Exception:
            continue
    else:
        result["error"] = "unreachable on both ethernet (.52) and wifi (.38)"
        check_connection_health("sovol", False,
                                "unreachable on ethernet and wifi")
        return result

    ps = data.get("print_stats", {})
    ds = data.get("display_status", {})
    bed = data.get("heater_bed", {})
    ext = data.get("extruder", {})
    gm = data.get("gcode_move", {})
    mr = data.get("motion_report", {})

    result["state"] = ps.get("state", "unknown")
    result["filename"] = ps.get("filename", "")
    result["progress"] = round((ds.get("progress") or 0) * 100, 1)
    result["print_duration"] = ps.get("print_duration") or 0
    result["total_duration"] = ps.get("total_duration") or 0
    result["filament_used_mm"] = ps.get("filament_used") or 0
    result["bed_temp"] = round(bed.get("temperature") or 0, 1)
    result["bed_target"] = round(bed.get("target") or 0, 1)
    result["nozzle_temp"] = round(ext.get("temperature") or 0, 1)
    result["nozzle_target"] = round(ext.get("target") or 0, 1)

    # Layer progress
    info = ps.get("info", {})
    result["current_layer"] = info.get("current_layer") or 0
    result["total_layers"] = info.get("total_layer") or 0
    if result["total_layers"] > 0:
        result["layer_progress"] = round(result["current_layer"] / result["total_layers"] * 100, 1)

    # Speed factor and live velocity
    result["speed_factor"] = gm.get("speed_factor") or 1.0
    result["commanded_speed"] = round((gm.get("speed") or 0) / 60, 1)  # mm/min -> mm/s
    result["live_velocity"] = round(mr.get("live_velocity") or 0, 1)

    # Metadata for current file
    if result["filename"]:
        enc = urllib.parse.quote(result["filename"])
        try:
            meta = json.loads(_fetch_url(
                f"{base}/server/files/metadata?filename={enc}", timeout=5))["result"]
            result["estimated_time"] = meta.get("estimated_time") or 0
            result["filament_total_mm"] = meta.get("filament_total") or 0
            result["filament_weight_g"] = meta.get("filament_weight_total") or 0
            result["slicer"] = meta.get("slicer", "")
            result["layer_height"] = meta.get("layer_height") or 0
            result["nozzle_diameter"] = meta.get("nozzle_diameter") or 0
            result["filament_name"] = meta.get("filament_name", "")
            result["filament_type"] = meta.get("filament_type", "")
            result["object_height"] = meta.get("object_height") or 0

            # Thumbnail
            thumbs = meta.get("thumbnails", [])
            if thumbs:
                biggest = max(thumbs, key=lambda t: t.get("width", 0) * t.get("height", 0))
                tp = biggest.get("relative_path", "")
                if tp:
                    thumb_url = f"{base}/server/files/gcodes/{urllib.parse.quote(tp)}"
                    thumb_path = os.path.join(OUTPUT_DIR, "sovol_thumbnail.png")
                    with open(thumb_path, "wb") as f:
                        f.write(_fetch_url(thumb_url, timeout=5, retries=2))
                    result["thumbnail"] = thumb_path
        except Exception as e:
            result["meta_error"] = str(e)

        # Print start time from history
        try:
            hist = json.loads(_fetch_url(
                f"{base}/server/history/list?limit=1&order=desc", timeout=5))["result"]
            if hist.get("jobs"):
                result["start_time"] = hist["jobs"][0].get("start_time", 0)
        except Exception:
            _log_exc("fetch_sovol_history")

    # Camera snapshot (with timeout to prevent hanging)
    try:
        cam_path = os.path.join(OUTPUT_DIR, "sovol_camera.jpg")
        with open(cam_path, "wb") as f:
            f.write(_fetch_url(
                f"http://{SOVOL_IP}:{SOVOL_CAMERA_PORT}/webcam/?action=snapshot",
                timeout=3, retries=2))
        result["camera"] = cam_path
    except Exception as e:
        result["camera_error"] = str(e)

    # Calculate times — smart ETA
    if result.get("start_time"):
        result["start_str"] = datetime.fromtimestamp(result["start_time"]).strftime("%a %d %b %H:%M")
    dur = result.get("print_duration", 0)
    result["duration_str"] = f"{int(dur // 3600)}h {int((dur % 3600) // 60)}m"

    try:
        from print_eta import calculate_eta, log_snapshot
        eta_result = calculate_eta(
            progress_pct=result["progress"],
            print_duration_s=dur,
            estimated_time_s=result.get("estimated_time", 0),
            speed_factor=result.get("speed_factor", 1.0),
            live_velocity=result.get("live_velocity", 0),
            commanded_speed=result.get("commanded_speed", 0),
            current_layer=result.get("current_layer", 0),
            total_layers=result.get("total_layers", 0),
            filename=result.get("filename", ""),
            printer="sovol",
        )
        if "error" not in eta_result:
            result["remaining_str"] = eta_result["remaining_str"]
            result["eta_str"] = eta_result["eta_str"]
            result["eta_confidence"] = eta_result["confidence"]
            result["eta_method"] = eta_result["method"]
            if "effective_speed" in eta_result:
                result["effective_speed"] = eta_result["effective_speed"]
            if "alpha" in eta_result:
                result["alpha"] = eta_result["alpha"]
            if "optimal_speed_pct" in eta_result:
                result["optimal_speed_pct"] = eta_result["optimal_speed_pct"]
        # Log snapshot for tracking
        if result.get("state") == "printing":
            log_snapshot(result)
    except Exception:
        # Fallback to naive calculation
        if (result.get("progress") or 0) > 0:
            actual_total = dur / (result["progress"] / 100)
            remaining = actual_total - dur
            eta = datetime.now() + timedelta(seconds=remaining)
            result["remaining_str"] = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
            result["eta_str"] = eta.strftime("%a %d %b %H:%M")

    # Speed analysis using physics model
    est = result.get("estimated_time", 0)
    prog = result.get("progress", 0)
    spd = result.get("speed_factor", 1.0)
    if est > 0 and dur > 0 and prog > 1:
        effective = (est * (prog / 100)) / dur
        result["effective_speed"] = round(effective, 2)
        result["speed_efficiency"] = round((effective / spd) * 100) if spd > 0 else 0

        # Use alpha from ETA model for physics-based speed analysis
        alpha = result.get("alpha", 0.35)
        try:
            from print_eta import speed_time_ratio, optimal_speed_factor, speed_factor_benefit
            opt = optimal_speed_factor(alpha)
            result["max_useful_speed"] = round(opt, 2)
            result["max_useful_pct"] = round(opt * 100)

            # Speed warning using physics model
            if spd > 1.1:
                time_pct, desc, opt_pct = speed_factor_benefit(spd, alpha)
                if time_pct >= 98:  # Less than 2% benefit
                    result["speed_warning"] = (
                        f"Speed {round(spd*100)}% has no meaningful benefit "
                        f"(α={alpha:.2f}, geometry limited). "
                        f"Optimal: {opt_pct}%."
                    )
                elif time_pct > 100:  # Actually slower
                    result["speed_warning"] = (
                        f"Speed {round(spd*100)}% is making this print SLOWER! "
                        f"α={alpha:.2f} means accel overhead > cruise savings. "
                        f"Optimal: {opt_pct}%. Drop speed to save time."
                    )
        except Exception:
            # Fallback to old heuristic
            max_useful = min(spd, effective * 1.1)
            result["max_useful_speed"] = round(max_useful, 2)
            result["max_useful_pct"] = round(max_useful * 100)
            if spd > 1.2 and effective < spd * 0.6:
                result["speed_warning"] = (
                    f"Speed set to {spd:.0f}x but only achieving {effective:.1f}x."
                )
    # ── Per-layer gcode profile integration ──
    # Load profile if available, calibrate with live alpha, add layer-level
    # speed recommendations and profile-based ETA
    try:
        from gcode_profile import (load_profile, get_layer_info,
                                   calibrate_profile, calibrated_eta_remaining,
                                   profile_eta_remaining, load_auto_speed,
                                   save_auto_speed, set_printer_speed)

        profile = load_profile(result.get("filename"))
        if profile and result.get("state") == "printing":
            cur_layer = result.get("current_layer", 0)
            total_layers = result.get("total_layers", 0)
            spd = result.get("speed_factor", 1.0)

            # Get measured alpha — prefer live measurement from ETA model,
            # but fall back to persisted alpha file (which may have been
            # measured at a different speed factor)
            measured_alpha = result.get("alpha")
            if (not measured_alpha or measured_alpha == 0.4):
                # 0.4 is DEFAULT_ALPHA — means no real measurement.
                # Try the alpha persistence file directly.
                try:
                    alpha_file = "/tmp/printer_status/current_alpha.json"
                    if os.path.exists(alpha_file):
                        with open(alpha_file) as af:
                            alpha_data = json.load(af)
                        stored = alpha_data.get("alpha")
                        if stored and stored > 0.01:
                            measured_alpha = stored
                except Exception:
                    log.debug("suppressed", exc_info=True)

            result["profile_available"] = True

            # Get raw layer info
            layer_info = get_layer_info(profile, cur_layer)
            if layer_info:
                result["layer_raw_alpha"] = layer_info.get("raw_alpha",
                                                           layer_info.get("alpha"))
                result["layer_avg_segment"] = layer_info.get("avg_segment_mm")

            # Calibrate if we have a live measurement
            if measured_alpha and measured_alpha > 0.01 and cur_layer > 2:
                cal = calibrate_profile(profile, measured_alpha, cur_layer,
                                        spd, elapsed_time_s=dur)
                if cal:
                    result["profile_calibrated"] = True

                    # Current layer calibrated info
                    for cl in cal:
                        if cl["layer"] == cur_layer:
                            result["layer_alpha"] = cl["calibrated_alpha"]
                            result["layer_optimal_speed"] = cl["optimal_speed_pct"]
                            # Per-layer effective speed: 1/R(S) using this layer's alpha
                            try:
                                from print_eta import speed_time_ratio
                                la = cl["calibrated_alpha"]
                                R = speed_time_ratio(spd, la)
                                result["effective_speed"] = round(1.0 / R, 2)
                                result["effective_speed_scope"] = "layer"
                                result["speed_efficiency"] = round((1.0 / R / spd) * 100) if spd > 0 else 0
                            except Exception:
                                log.debug("suppressed", exc_info=True)
                            # Update speed warning to use layer alpha
                            try:
                                from print_eta import speed_factor_benefit, optimal_speed_factor
                                la = cl["calibrated_alpha"]
                                opt_l = optimal_speed_factor(la)
                                result["max_useful_speed"] = round(opt_l, 2)
                                result["max_useful_pct"] = round(opt_l * 100)
                                if spd > 1.1:
                                    time_pct, desc, opt_pct = speed_factor_benefit(spd, la)
                                    if time_pct >= 98:
                                        result["speed_warning"] = (
                                            f"Speed {round(spd*100)}% has no meaningful benefit "
                                            f"on this layer (α={la:.2f}). "
                                            f"Optimal: {opt_pct}%.")
                                    elif time_pct > 100:
                                        result["speed_warning"] = (
                                            f"Speed {round(spd*100)}% is SLOWER on this layer! "
                                            f"α={la:.2f}, optimal: {opt_pct}%. Drop speed.")
                                    else:
                                        result.pop("speed_warning", None)
                            except Exception:
                                log.debug("suppressed", exc_info=True)
                            break

                    # Next layer preview
                    for cl in cal:
                        if cl["layer"] == cur_layer + 1:
                            result["next_layer_alpha"] = cl["calibrated_alpha"]
                            result["next_layer_optimal"] = cl["optimal_speed_pct"]
                            break

                    # Profile-based ETA (calibrated)
                    progress_in_layer = 0.5
                    if total_layers > 0 and cur_layer > 0:
                        # Estimate progress within current layer from
                        # overall progress vs layer progress
                        layer_pct = result.get("layer_progress", 50)
                        overall_pct = result.get("progress", 0)
                        # rough: how far through this layer based on
                        # overall progress vs layer boundaries
                        progress_in_layer = max(0.1, min(0.9,
                            (overall_pct / 100 - cur_layer / total_layers)
                            / (1.0 / total_layers) if total_layers > 0 else 0.5
                        ))

                    cal_remaining = calibrated_eta_remaining(
                        cal, cur_layer, progress_in_layer,
                        speed_factor=spd)
                    if cal_remaining > 0:
                        from datetime import timedelta as td
                        cal_eta = datetime.now() + td(seconds=cal_remaining)
                        cal_remaining_str = (
                            f"{int(cal_remaining // 3600)}h "
                            f"{int((cal_remaining % 3600) // 60)}m")
                        cal_eta_str = cal_eta.strftime(
                            "%a %d %b %H:%M")
                        result["profile_remaining_str"] = cal_remaining_str
                        result["profile_eta_str"] = cal_eta_str
                        # Promote profile ETA as primary when calibrated
                        result["remaining_str"] = cal_remaining_str
                        result["eta_str"] = cal_eta_str
                        result["eta_method"] = (
                            f"profile(cal,layers={len(cal)},"
                            f"α={measured_alpha:.2f})")
                        result["eta_confidence"] = "high"

                    # ── Auto-speed adjustment ──
                    auto_cfg = load_auto_speed()
                    if auto_cfg.get("enabled"):
                        result["auto_speed_enabled"] = True
                        result["auto_speed_mode"] = auto_cfg.get("mode",
                                                                 "optimal")

                        # Find target speed for current layer
                        target_pct = None
                        for cl in cal:
                            if cl["layer"] == cur_layer:
                                target_pct = cl["optimal_speed_pct"]
                                break

                        if target_pct and cur_layer >= auto_cfg.get(
                                "skip_first_layers", 2):
                            # Apply mode
                            if auto_cfg.get("mode") == "conservative":
                                target_pct = max(round(target_pct * 0.95),
                                    auto_cfg.get("min_speed_pct", 80))

                            # Hard cap: never exceed global optimal from
                            # measured alpha. Per-layer calibration can
                            # underestimate complexity; the global
                            # measurement is ground truth.
                            try:
                                from print_eta import optimal_speed_factor
                                global_opt_pct = round(
                                    optimal_speed_factor(measured_alpha) * 100)
                                target_pct = min(target_pct, global_opt_pct)
                            except Exception:
                                log.debug("suppressed", exc_info=True)

                            # Clamp
                            target_pct = max(
                                auto_cfg.get("min_speed_pct", 80),
                                min(target_pct,
                                    auto_cfg.get("max_speed_pct", 200)))

                            current_pct = round(spd * 100)
                            last = auto_cfg.get("last_adjustment", {})

                            # If current speed is above optimal (actively
                            # hurting the print), jump immediately — don't
                            # wait for layer change or apply smoothing.
                            try:
                                from print_eta import speed_time_ratio
                                hurting = speed_time_ratio(
                                    spd, measured_alpha) > 1.0
                            except Exception:
                                hurting = False

                            should_adjust = False
                            if hurting and current_pct > target_pct:
                                # Speed is making print slower — force
                                # immediate drop
                                should_adjust = True
                            elif (last.get("layer") != cur_layer and
                                    abs(target_pct - current_pct) > 5):
                                # Normal: adjust on layer change
                                should_adjust = True

                            if should_adjust:
                                success = set_printer_speed(target_pct)
                                if success:
                                    auto_cfg["last_adjustment"] = {
                                        "layer": cur_layer,
                                        "speed": target_pct,
                                        "from_speed": current_pct,
                                        "alpha": result.get("layer_alpha"),
                                        "forced": hurting,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    save_auto_speed(auto_cfg)
                                    result["speed_adjusted"] = True
                                    result["speed_adjusted_to"] = target_pct
                                    result["speed_adjusted_from"] = current_pct
    except Exception:
        log.debug("suppressed", exc_info=True)  # Profile not available or error — non-critical

    # ── Immutable ETA snapshots — save every 5 minutes ──
    # These are frozen predictions that never get recalculated.
    # The ETA graph plots these directly so the algorithm can't
    # retroactively make itself look more accurate.
    if result.get("state") == "printing" and result.get("remaining_str"):
        _save_eta_snapshot(result)

    # ── Frontend display fields (consumed by wrapper read-only route) ──
    # Camera/thumbnail availability
    if os.path.exists(os.path.join(OUTPUT_DIR, "sovol_camera.jpg")):
        result["has_camera"] = True
    if os.path.exists(os.path.join(OUTPUT_DIR, "sovol_thumbnail.png")):
        result["has_thumbnail"] = True
    result["current_speed_pct"] = round(
        result.get("speed_factor", 1.0) * 100)

    # ETA history from immutable snapshots (for drift graph)
    try:
        snap_path = os.path.join(OUTPUT_DIR, "eta_snapshots.jsonl")
        cur_file = result.get("filename", "")
        if os.path.exists(snap_path) and cur_file:
            eta_history = []
            with open(snap_path) as sf:
                for line in sf:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("filename") != cur_file:
                            continue
                        finish_ts = entry.get("predicted_finish_ts", 0)
                        prog = entry.get("progress", 0)
                        if finish_ts > 0 and prog > 0:
                            eta_history.append({
                                "progress": prog,
                                "elapsed_h": round(
                                    entry.get("elapsed_s", 0) / 3600, 2),
                                "remaining_h": round(
                                    entry.get("remaining_s", 0) / 3600, 2),
                                "finish_ts": finish_ts,
                                "finish_str": entry.get(
                                    "predicted_finish_str", ""),
                            })
                    except Exception:
                        continue
            # Detect reprints of same filename: if progress drops
            # significantly, discard entries from the previous run.
            # A drop of >5 points catches both full completions (99→1)
            # and short/cancelled runs (e.g. 9→1).
            if len(eta_history) >= 2:
                last_reset = 0
                for i in range(1, len(eta_history)):
                    prev_prog = eta_history[i - 1]["progress"]
                    curr_prog = eta_history[i]["progress"]
                    if curr_prog < prev_prog - 5:
                        last_reset = i
                if last_reset > 0:
                    eta_history = eta_history[last_reset:]

            if len(eta_history) >= 2:
                result["eta_history"] = eta_history
    except Exception:
        log.debug("suppressed", exc_info=True)

    # Speed graph from gcode profile (per-layer optimal speeds)
    try:
        from gcode_profile import load_profile
        cur_layer = result.get("current_layer", 0)
        profile = load_profile(result.get("filename", ""))
        if profile and profile.get("layers"):
            speed_graph = []
            for layer in profile["layers"]:
                entry = {
                    "layer": layer["layer"],
                    "optimal_pct": layer.get("optimal_speed_pct", 100),
                    "alpha": round(layer.get("alpha", 1.0), 3),
                }
                if layer["layer"] < cur_layer:
                    entry["status"] = "past"
                elif layer["layer"] == cur_layer:
                    entry["status"] = "current"
                else:
                    entry["status"] = "future"
                speed_graph.append(entry)
            result["speed_graph"] = speed_graph
    except Exception:
        log.debug("suppressed", exc_info=True)

    fil_used_m = result.get("filament_used_mm", 0) / 1000
    fil_total_m = result.get("filament_total_mm", 0) / 1000
    if fil_total_m > 0:
        result["filament_used_m"] = round(fil_used_m, 1)
        result["filament_total_m"] = round(fil_total_m, 1)
        result["filament_pct"] = round(fil_used_m / fil_total_m * 100)
        wt = result.get("filament_weight_g", 0)
        if wt:
            result["filament_used_g"] = round(wt * (fil_used_m / fil_total_m))
            result["filament_total_g"] = round(wt)

    # Filament feed monitor — detect stalled extrusion
    # Compare expected filament usage (from progress) vs actual usage
    if fil_total_m > 0 and (result.get("progress") or 0) > 5 and result.get("state") == "printing":
        expected_m = fil_total_m * (result["progress"] / 100)
        if expected_m > 0:
            feed_ratio = fil_used_m / expected_m
            result["feed_ratio"] = round(feed_ratio, 2)
            # If we've used less than 50% of expected filament, something's wrong
            if feed_ratio < 0.5:
                result["filament_warning"] = (
                    f"Filament feed issue! Used {fil_used_m:.1f}m but expected ~{expected_m:.1f}m "
                    f"({feed_ratio:.0%} of expected). Check filament path."
                )
                # macOS notification
                try:
                    import subprocess
                    subprocess.Popen([
                        "osascript", "-e",
                        f'display notification "Used {fil_used_m:.1f}m but expected {expected_m:.1f}m — check filament!" '
                        f'with title "SV08: Filament Feed Issue!" sound name "Basso"'
                    ])
                except Exception:
                    log.debug("suppressed", exc_info=True)

    # Extract model name from filename
    fn = result.get("filename", "")
    name = fn.replace(".gcode", "").replace(".3mf", "")
    # Strip slicer suffixes like _PLA_0.2_1d3h35m
    for sep in ["__PLA", "__ABS", "__PETG", "__TPU"]:
        if sep in name:
            name = name[:name.index(sep)]
    # Strip printer prefix
    for prefix in ["Sovol SV08 MAX_", "Sovol SV08_"]:
        if prefix in name:
            name = name[name.index(prefix) + len(prefix):]
    # Strip scale prefix like 250pct_
    if "pct_" in name:
        parts = name.split("pct_", 1)
        scale = parts[0].replace("_", "")
        name = parts[1] if len(parts) > 1 else name
        result["scale"] = f"{scale}%"
    result["model_name"] = name.replace("_", " ").strip()

    # ── Completion detection: score ETA predictions when print finishes ──
    try:
        state_file = os.path.join(OUTPUT_DIR, "last_print_state.json")
        last_state = {}
        if os.path.exists(state_file):
            with open(state_file) as f:
                last_state = json.load(f)

        cur_state = result.get("state", "")
        cur_file = result.get("filename", "")

        # Detect transition: was printing → now complete/standby
        if (last_state.get("state") == "printing" and
                cur_state in ("complete", "standby", "cancelled") and
                last_state.get("filename")):
            actual_time = last_state.get("print_duration", 0)
            if actual_time > 60:  # only score prints longer than 1 min
                from eta_learner import score_completed_job
                from print_eta import save_completed_job
                scored = score_completed_job(
                    last_state["filename"], actual_time,
                    last_state.get("start_time"))
                if scored:
                    result["learning_scored"] = True
                    result["learning_snapshots"] = scored["snapshot_count"]
                # Also save to completed jobs for correction factor
                save_completed_job(last_state)

        # Save current state for next comparison
        with open(state_file, "w") as f:
            json.dump({
                "state": cur_state,
                "filename": cur_file,
                "print_duration": dur,
                "start_time": result.get("start_time"),
            }, f)
    except Exception:
        log.debug("suppressed", exc_info=True)

    # ── Connection health + ETA method change alerts ──
    check_connection_health("sovol", result.get("online", False),
                            result.get("error", ""))
    if result.get("state") == "printing":
        check_eta_method_change("sovol", result)

    return result


def fetch_bambu():
    """Fetch Bambu A1 status via local MQTT."""
    result = {"printer": "Bambu A1", "online": False}

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        result["error"] = "paho-mqtt not installed"
        return result

    done = [False]

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(f"device/{BAMBU_SERIAL}/report")
        else:
            result["error"] = f"Connect failed: rc={rc}"
            done[0] = True

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            if "print" not in data:
                return
            p = data["print"]
            result["online"] = True
            result["state"] = p.get("gcode_state", "unknown")
            result["progress"] = p.get("mc_percent") or 0
            result["remaining_min"] = p.get("mc_remaining_time") or 0
            result["filename"] = p.get("gcode_file", "")
            result["subtask_name"] = p.get("subtask_name", "")
            result["layer_num"] = p.get("layer_num") or 0
            result["total_layer_num"] = p.get("total_layer_num") or 0
            result["nozzle_temp"] = round(p.get("nozzle_temper") or 0, 1)
            result["nozzle_target"] = round(p.get("nozzle_target_temper") or 0, 1)
            result["bed_temp"] = round(p.get("bed_temper") or 0, 1)
            result["bed_target"] = round(p.get("bed_target_temper") or 0, 1)
            result["speed_level"] = p.get("spd_lvl", 0)
            result["wifi_signal"] = p.get("wifi_signal", "")
            result["print_type"] = p.get("print_type", "")

            # AMS info
            ams = p.get("ams", {})
            if ams and "ams" in ams:
                trays = []
                for unit in ams["ams"]:
                    for tray in unit.get("tray", []):
                        if tray.get("tray_type"):
                            trays.append({
                                "slot": tray.get("id", ""),
                                "type": tray.get("tray_type", ""),
                                "color": tray.get("tray_color", ""),
                                "name": tray.get("tray_sub_brands", "")
                            })
                result["ams_trays"] = trays

            # Calculate times
            rem = result.get("remaining_min") or 0
            prog = result.get("progress") or 0
            if prog > 0 and prog < 100:
                total_min = rem / (1 - prog / 100)
                elapsed_min = total_min - rem
                result["duration_str"] = f"{int(elapsed_min // 60)}h {int(elapsed_min % 60)}m"
            result["remaining_str"] = f"{rem}m" if rem < 60 else f"{rem // 60}h {rem % 60}m"
            eta = datetime.now() + timedelta(minutes=rem)
            result["eta_str"] = eta.strftime("%a %d %b %H:%M")

            # Model name
            name = result.get("subtask_name", "") or result.get("filename", "")
            name = name.replace(".gcode", "").replace(".3mf", "")
            result["model_name"] = name

            done[0] = True
        except Exception as e:
            result["parse_error"] = str(e)
            done[0] = True

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set("bblp", BAMBU_ACCESS_CODE)
    client.tls_set_context(ctx)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BAMBU_IP, BAMBU_MQTT_PORT, 60)
        client.loop_start()
        # Push status request
        import threading
        def push_req():
            time.sleep(0.5)
            client.publish(
                f"device/{BAMBU_SERIAL}/request",
                json.dumps({"pushing": {"sequence_id": "0", "command": "pushall"}})
            )
        threading.Thread(target=push_req, daemon=True).start()

        timeout = time.time() + 8
        while not done[0] and time.time() < timeout:
            time.sleep(0.2)
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        result["error"] = str(e)

    # ── Immutable ETA snapshots for Bambu ──
    if (result.get("state", "").upper() == "RUNNING" and
            result.get("remaining_str") and result.get("filename")):
        # Build elapsed_s from progress + remaining
        rem_min = result.get("remaining_min") or 0
        prog = result.get("progress") or 0
        elapsed_s = 0
        if prog > 0 and prog < 100:
            total_min = rem_min / (1 - prog / 100)
            elapsed_s = (total_min - rem_min) * 60
        snap_result = {
            "filename": result["filename"],
            "progress": prog,
            "state": "printing",
            "remaining_str": result["remaining_str"],
            "print_duration": elapsed_s,
            "eta_method": "bambu_mqtt",
            "eta_confidence": "",
            "alpha": None,
            "speed_factor": result.get("speed_level", 1),
        }
        _save_eta_snapshot(snap_result)

    # ── Bambu A1 camera snapshot via FTPS ──
    # Only capture when printing — idle printer has no new recordings
    bambu_printing = result.get("state", "").lower() in ("printing", "running", "prepare")
    try:
        import ftplib
        import socket
        import subprocess
        import io

        cam_path = os.path.join(OUTPUT_DIR, "a1_camera.jpg")
        if not bambu_printing:
            if os.path.exists(cam_path) and os.path.getsize(cam_path) > 0:
                result["has_camera"] = True
            raise StopIteration("not printing")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Implicit FTPS on port 990
        raw_sock = socket.create_connection((BAMBU_IP, 990), timeout=10)
        tls_sock = ctx.wrap_socket(raw_sock, server_hostname=BAMBU_IP)
        ftp = ftplib.FTP()
        ftp.af = socket.AF_INET
        ftp.sock = tls_sock
        ftp.file = tls_sock.makefile('r', encoding='utf-8', errors='replace')
        ftp.welcome = ftp.getresp()
        ftp.login('bblp', BAMBU_ACCESS_CODE)

        # Wrap data connections in TLS too
        _orig_ntransfercmd = ftp.ntransfercmd
        def _tls_ntransfercmd(cmd, rest=None):
            conn, size = _orig_ntransfercmd(cmd, rest)
            conn = ctx.wrap_socket(conn, server_hostname=BAMBU_IP)
            return conn, size
        ftp.ntransfercmd = _tls_ntransfercmd
        ftp.voidcmd('PBSZ 0')
        ftp.voidcmd('PROT P')

        ftp.cwd('/ipcam')
        listing = []
        ftp.retrlines('NLST', listing.append)
        avis = sorted([f for f in listing if f.endswith('.avi')])

        if avis:
            latest_avi = avis[-1]
            buf = io.BytesIO()
            max_dl = 128 * 1024  # 128KB — enough for one frame, ~4s download
            dl_count = [0]
            def _cb(data):
                if dl_count[0] < max_dl:
                    buf.write(data)
                    dl_count[0] += len(data)
                    if dl_count[0] >= max_dl:
                        raise EOFError
            try:
                ftp.retrbinary('RETR ' + latest_avi, _cb)
            except (EOFError, Exception):
                pass

            if buf.tell() > 0:
                tmp = os.path.join(OUTPUT_DIR, ".a1_partial.avi")
                with open(tmp, 'wb') as tf:
                    tf.write(buf.getvalue())
                subprocess.run(
                    ['/opt/homebrew/bin/ffmpeg', '-y', '-i', tmp,
                     '-vf', 'select=gte(n\\,0)',
                     '-vframes', '1', '-q:v', '2', cam_path],
                    capture_output=True, timeout=10)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        try:
            ftp.close()
        except Exception:
            log.debug("suppressed", exc_info=True)

        if os.path.exists(cam_path) and os.path.getsize(cam_path) > 0:
            result["has_camera"] = True
    except StopIteration:
        # Not printing — skip FTPS, use cached image
        cam_path = os.path.join(OUTPUT_DIR, "a1_camera.jpg")
        if os.path.exists(cam_path) and os.path.getsize(cam_path) > 0:
            result["has_camera"] = True
    except Exception as e:
        # Camera capture failed — log it so we can debug
        log.warning("Bambu camera capture failed: %s: %s",
                     type(e).__name__, e)
        cam_path = os.path.join(OUTPUT_DIR, "a1_camera.jpg")
        if os.path.exists(cam_path) and os.path.getsize(cam_path) > 0:
            result["has_camera"] = True

    # Bambu ETA history — use snapshot_ts for elapsed_h (monotonic)
    try:
        snap_path = os.path.join(OUTPUT_DIR, "eta_snapshots.jsonl")
        cur_file = result.get("filename", "")
        if os.path.exists(snap_path) and cur_file:
            eta_history = []
            with open(snap_path) as sf:
                for line in sf:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("filename") != cur_file:
                            continue
                        finish_ts = entry.get("predicted_finish_ts", 0)
                        prog_snap = entry.get("progress", 0)
                        snap_ts = entry.get("snapshot_ts", 0)
                        if finish_ts > 0 and prog_snap > 0 and snap_ts > 0:
                            eta_history.append({
                                "progress": prog_snap,
                                "snapshot_ts": snap_ts,
                                "remaining_h": round(
                                    entry.get("remaining_s", 0) / 3600, 2),
                                "finish_ts": finish_ts,
                                "finish_str": entry.get(
                                    "predicted_finish_str", ""),
                            })
                    except Exception:
                        continue
            # Detect reprints of same filename: if progress drops
            # significantly, discard entries from the previous run.
            # A drop of >5 points catches both full completions (99→1)
            # and short/cancelled runs (e.g. 9→1).
            if len(eta_history) >= 2:
                last_reset = 0
                for i in range(1, len(eta_history)):
                    prev_prog = eta_history[i - 1]["progress"]
                    curr_prog = eta_history[i]["progress"]
                    if curr_prog < prev_prog - 5:
                        last_reset = i  # New print started here
                if last_reset > 0:
                    eta_history = eta_history[last_reset:]

            if len(eta_history) >= 2:
                first_ts = eta_history[0]["snapshot_ts"]
                for entry in eta_history:
                    entry["elapsed_h"] = round(
                        (entry["snapshot_ts"] - first_ts) / 3600, 2)
                    del entry["snapshot_ts"]
                result["eta_history"] = eta_history
    except Exception:
        log.debug("suppressed", exc_info=True)

    # ── Connection health alerts for Bambu ──
    check_connection_health("bambu", result.get("online", False),
                            result.get("error", ""))

    return result


if __name__ == "__main__":
    data = {"timestamp": datetime.now().isoformat(), "printers": {}}

    data["printers"]["sovol"] = fetch_sovol()
    data["printers"]["bambu"] = fetch_bambu()

    out_path = os.path.join(OUTPUT_DIR, "status.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(json.dumps(data, indent=2))
