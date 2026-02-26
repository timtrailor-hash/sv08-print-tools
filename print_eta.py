#!/usr/bin/env python3
"""Smart print ETA calculator for SV08 Max — physics-based model.

Uses trapezoid move kinematics to model how speed factor interacts with
acceleration limits and geometry complexity:

    Move time: T(S) = d/(v·S) + (v·S)/a
                      ↑ cruise    ↑ accel overhead

    Time ratio: R(S) = (1/S + α·S) / (1 + α)

Alpha (α) captures geometry complexity — the fraction of move time spent
accelerating/decelerating vs cruising:
    α = v² / (a · d_avg)

Typical alpha values:
    0.02–0.08  simple     (cubes, vases, large flat surfaces)
    0.08–0.20  moderate   (benchies, functional parts)
    0.20–0.40  complex    (detailed figurines, architectural)
    0.40–0.60  very complex (dragons, articulated, organic)
    0.60–1.0+  extreme    (miniatures, fine detail)

The model self-calibrates: α is derived from the measured effective speed
once enough progress data is available (>3%).

Validated against:
    - SV08 Max: Zephyros dragon 250% (α≈0.50), articulated snake (α≈0.55)
    - Community data: Ender 3 benchies at 50/150mm/s (α≈0.14)
    - Square corner velocity tests (α≈0.15 for moderate geometry)
    - Klipper kinematics docs (klipper3d.org/Kinematics.html)
    - Dyze Design speed calculations (dyzedesign.com/2016/11/printing-300-mm-s-part-2-calculations/)

Also logs snapshots to a tracker file for future analysis.
"""

import json
import logging
import math
import os
from datetime import datetime, timedelta

log = logging.getLogger("print_eta")

TRACKER_DIR = "/tmp/printer_status"
TRACKER_FILE = os.path.join(TRACKER_DIR, "print_tracker.jsonl")
HISTORY_FILE = os.path.join(TRACKER_DIR, "completed_jobs.json")
ALPHA_FILE_LEGACY = os.path.join(TRACKER_DIR, "current_alpha.json")
os.makedirs(TRACKER_DIR, exist_ok=True)

# Slicer correction factor. OrcaSlicer machine limits were corrected
# 2026-02-20 (~13% reduction). Set to 1.0 to avoid double-correcting.
# Auto-calibrates from completed_jobs.json once enough data exists.
DEFAULT_SLICER_CORRECTION = 1.0

# Default alpha — conservative estimate for unknown geometry.
# 0.40 means we expect ~40% of move time is accel/decel overhead.
# This is deliberately slightly high so initial ETAs are pessimistic
# (an ETA that improves is better than one that slips).
DEFAULT_ALPHA = 0.40


# ── Physics model ──────────────────────────────────────────────
# Canonical implementations live in printer_physics.py.
# Imported here for backward compatibility (all existing callers
# do `from print_eta import speed_time_ratio` etc.)

try:
    from printer_physics import (
        speed_time_ratio,
        optimal_speed_factor,
        alpha_from_measurement,
    )
except ImportError:
    # Standalone fallback (e.g. running from sv08-print-tools repo)
    def speed_time_ratio(speed_factor, alpha):
        S = max(speed_factor, 0.1)
        a = max(alpha, 0.001)
        return (1.0 / S + a * S) / (1.0 + a)

    def optimal_speed_factor(alpha):
        if alpha <= 0.001:
            return 10.0
        return min(1.0 / math.sqrt(alpha), 10.0)

    def alpha_from_measurement(effective_speed, speed_factor):
        S = speed_factor
        if S <= 1.05 or effective_speed <= 0:
            return None
        denom = effective_speed * S - 1.0
        if denom <= 0.01:
            return 1.5
        numer = 1.0 - effective_speed / S
        return max(0.0, min(numer / denom, 5.0))


def speed_factor_benefit(speed_factor, alpha):
    """Human-readable summary of speed factor impact.

    Returns (time_pct, description, optimal_pct).
    """
    ratio = speed_time_ratio(speed_factor, alpha)
    pct = round(ratio * 100)
    opt = optimal_speed_factor(alpha)
    opt_pct = round(opt * 100)

    if ratio > 1.02:
        desc = f"SLOWER at {round(speed_factor*100)}% — accel overhead exceeds cruise savings"
    elif ratio > 0.98:
        desc = f"No meaningful benefit at {round(speed_factor*100)}%"
    else:
        saving = round((1 - ratio) * 100)
        desc = f"{saving}% faster at {round(speed_factor*100)}%"

    return pct, desc, opt_pct


# ── Alpha persistence ──────────────────────────────────────────

def _alpha_file(printer="sovol"):
    """Return per-printer alpha file path."""
    return os.path.join(TRACKER_DIR, f"current_alpha_{printer}.json")


def _load_alpha_state(printer="sovol"):
    """Load persisted alpha measurement for the current print.

    Uses per-printer file (current_alpha_{printer}.json). Falls back to
    the legacy current_alpha.json if per-printer file doesn't exist yet
    (one-time migration).
    """
    pfile = _alpha_file(printer)
    if os.path.exists(pfile):
        try:
            with open(pfile) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    # Migrate from legacy single-file if it exists
    if os.path.exists(ALPHA_FILE_LEGACY):
        try:
            with open(ALPHA_FILE_LEGACY) as f:
                data = json.load(f)
            # Save to per-printer file so we don't migrate again
            _save_alpha_state(data, printer)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_alpha_state(state, printer="sovol"):
    """Persist alpha measurement for the current print (per-printer).

    Uses write-to-temp + rename for crash safety — a crash mid-write
    won't corrupt the existing file.
    """
    path = _alpha_file(printer)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


# ── Profile calibration cache (survive transient failures) ────
_last_valid_cal = {}  # {printer: {"cal": [...], "layer": N, "alpha": X, "speed": S}}

# ── Main ETA calculator ───────────────────────────────────────

def calculate_eta(progress_pct, print_duration_s, estimated_time_s,
                  speed_factor, live_velocity=0, commanded_speed=0,
                  current_layer=0, total_layers=0, filename="",
                  printer="sovol"):
    """Calculate smart ETA for a running print.

    Returns dict with:
        remaining_s, remaining_str: estimated time remaining
        eta, eta_str: estimated completion time
        confidence: 'low' | 'medium' | 'high'
        method: description of calculation method used
        effective_speed: actual speed multiplier vs slicer baseline
        alpha: geometry complexity parameter (0=simple, 1=very complex)
        optimal_speed_pct: best speed factor % for this geometry
    """
    result = {}
    S = max(speed_factor, 0.1)
    correction = get_correction_factor()

    # Load persisted alpha — prefer same filename, but use any recent stored
    # alpha if no filename match (e.g. old data before filename was saved)
    alpha_state = _load_alpha_state(printer)
    stored_alpha = None
    if alpha_state.get("filename") == filename and filename:
        stored_alpha = alpha_state.get("alpha")
    elif alpha_state.get("alpha") and not alpha_state.get("filename"):
        # Legacy: stored alpha without filename — still use it
        stored_alpha = alpha_state.get("alpha")

    if progress_pct <= 0 or print_duration_s <= 0:
        # ── No progress yet: use physics model ──
        if estimated_time_s > 0:
            alpha = stored_alpha if stored_alpha else DEFAULT_ALPHA
            ratio = speed_time_ratio(S, alpha)
            remaining = estimated_time_s * ratio * correction

            result["alpha"] = round(alpha, 3)
            result["optimal_speed_pct"] = round(optimal_speed_factor(alpha) * 100)
            result["method"] = f"physics(α={alpha:.2f})"
            result["confidence"] = "low"
        else:
            return {"error": "no data available"}
    else:
        # ── Measure effective speed from actual progress ──
        expected_at_1x = estimated_time_s * (progress_pct / 100)
        effective_speed = expected_at_1x / print_duration_s
        result["effective_speed"] = round(effective_speed, 2)

        # ── Derive alpha from measurement ──
        measured_alpha = None
        if S > 1.05 and progress_pct >= 3:
            measured_alpha = alpha_from_measurement(effective_speed, S)

        # Pick best available alpha
        if measured_alpha is not None:
            alpha = measured_alpha
            _save_alpha_state({
                "alpha": round(alpha, 4),
                "speed_factor": S,
                "progress_pct": round(progress_pct, 1),
                "effective_speed": round(effective_speed, 3),
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
            }, printer)
        elif stored_alpha is not None:
            alpha = stored_alpha
        else:
            alpha = DEFAULT_ALPHA

        result["alpha"] = round(alpha, 3)
        result["optimal_speed_pct"] = round(optimal_speed_factor(alpha) * 100)

        # ── Predict effective speed at current speed factor ──
        predicted_eff = 1.0 / speed_time_ratio(S, alpha)

        # ── Blend physics prediction with measurement ──
        # Early layers are unreliable (first-layer speed, base plate geometry).
        # Gradually shift from physics model to direct measurement.
        if progress_pct < 10:
            meas_weight = progress_pct / 10  # 0→1 over 0–10%
            blended_eff = (predicted_eff * (1 - meas_weight) +
                           max(effective_speed, 0.3) * meas_weight)
        else:
            # After 10%, trust measurement fully
            blended_eff = effective_speed if effective_speed > 0.1 else predicted_eff

        # ── Calculate remaining time ──
        # Strategy: use the gcode profile as single source of truth when
        # available.  It knows every layer's geometry and self-corrects
        # by comparing expected vs actual elapsed time (time_cal_factor).
        # Only fall back to pace/slicer when no profile exists.

        profile_remaining = None
        try:
            from gcode_profile import (load_profile, calibrate_profile,
                                       calibrated_eta_remaining)
            profile = load_profile(filename) if filename else None
            if profile and alpha and current_layer > 0:
                cal = None
                # Try calibration with one retry on transient failure
                for _attempt in range(2):
                    try:
                        cal = calibrate_profile(
                            profile, alpha, current_layer, S,
                            elapsed_time_s=print_duration_s)
                        if cal:
                            # Cache this valid calibration for fallback
                            _last_valid_cal[printer] = {
                                "cal": cal, "layer": current_layer,
                                "alpha": alpha, "speed": S,
                            }
                            break
                    except Exception:
                        if _attempt == 0:
                            import time as _time
                            _time.sleep(0.3)
                        else:
                            log.debug("profile calibration retry failed",
                                      exc_info=True)

                # Fall back to cached calibration on transient failure
                if not cal and printer in _last_valid_cal:
                    cached = _last_valid_cal[printer]
                    # Only reuse if same print and not too stale (< 50 layers)
                    if (cached.get("layer", 0) > 0 and
                            abs(current_layer - cached["layer"]) < 50):
                        cal = cached["cal"]
                        log.info("Using cached profile calibration "
                                 "(layer %d→%d)", cached["layer"],
                                 current_layer)

                if cal:
                    pil = 0.5
                    if total_layers > 0:
                        expected_layer_frac = current_layer / total_layers
                        actual_frac = progress_pct / 100
                        layer_span = 1.0 / total_layers if total_layers > 0 else 1.0
                        pil = max(0.1, min(0.9,
                            (actual_frac - expected_layer_frac) / layer_span))
                    profile_remaining = calibrated_eta_remaining(
                        cal, current_layer, pil,
                        speed_factor=speed_factor)
        except Exception:
            log.debug("suppressed", exc_info=True)

        if profile_remaining and profile_remaining > 0:
            # Profile available — use it directly, no blending
            remaining = profile_remaining
            result["method"] = f"profile(α={alpha:.2f})"
            result["profile_remaining_s"] = round(profile_remaining)
        else:
            # No profile — fall back to pace/slicer blend
            slicer_left = estimated_time_s * (1 - progress_pct / 100)
            slicer_remaining = (slicer_left / max(blended_eff, 0.1)) * correction
            pace_total = print_duration_s / (progress_pct / 100)
            pace_remaining = pace_total - print_duration_s

            if progress_pct < 20:
                # Early: trust slicer more
                w_pace = progress_pct / 40  # 0→0.5 over 0-20%
                remaining = pace_remaining * w_pace + slicer_remaining * (1 - w_pace)
            else:
                # Later: pace is more reliable than slicer
                remaining = pace_remaining * 0.6 + slicer_remaining * 0.4

            result["method"] = f"pace+slicer(α={alpha:.2f})"

        # Log prediction for the learner
        try:
            from eta_learner import log_prediction
            slicer_left = estimated_time_s * (1 - progress_pct / 100)
            slicer_rem = (slicer_left / max(blended_eff, 0.1)) * correction
            pace_rem = print_duration_s / (progress_pct / 100) - print_duration_s
            log_prediction(
                filename=filename,
                progress_pct=progress_pct,
                alpha=alpha,
                elapsed_s=print_duration_s,
                pace_remaining_s=pace_rem,
                slicer_remaining_s=slicer_rem,
                profile_remaining_s=profile_remaining,
                blended_remaining_s=remaining,
                speed_factor=speed_factor,
                current_layer=current_layer,
            )
        except Exception:
            log.debug("suppressed", exc_info=True)

        # Confidence
        if profile_remaining and profile_remaining > 0:
            result["confidence"] = "high" if progress_pct >= 5 else "medium"
        elif progress_pct < 5:
            result["confidence"] = "low"
        elif progress_pct < 15:
            result["confidence"] = "medium"
        else:
            result["confidence"] = "high"

    # ── Format output ──
    remaining = max(0, min(remaining, 7 * 24 * 3600))
    eta = datetime.now() + timedelta(seconds=remaining)

    result["remaining_s"] = round(remaining)
    result["remaining_str"] = (f"{int(remaining // 3600)}h "
                                f"{int((remaining % 3600) // 60)}m")
    result["eta"] = eta
    result["eta_str"] = eta.strftime("%a %d %b %H:%M")
    result["progress_pct"] = round(progress_pct, 1)

    return result


# ── Logging & history ──────────────────────────────────────────

LAYER_LOG_FILE = os.path.join(TRACKER_DIR, "layer_log.jsonl")


def log_snapshot(print_data):
    """Append a progress snapshot to the tracker file for future analysis."""
    if not print_data.get("filename") or print_data.get("state") != "printing":
        return

    snapshot = {
        "ts": datetime.now().isoformat(),
        "file": print_data.get("filename", ""),
        "progress": print_data.get("progress", 0),
        "print_duration": print_data.get("print_duration", 0),
        "estimated_time": print_data.get("estimated_time", 0),
        "speed_factor": print_data.get("speed_factor", 1.0),
        "live_velocity": print_data.get("live_velocity", 0),
        "commanded_speed": print_data.get("commanded_speed", 0),
        "current_layer": print_data.get("current_layer", 0),
        "total_layers": print_data.get("total_layers", 0),
        "bed_temp": print_data.get("bed_temp", 0),
        "nozzle_temp": print_data.get("nozzle_temp", 0),
        "filament_used_mm": print_data.get("filament_used_mm", 0),
        # ETA and profile data for optimization
        "alpha": print_data.get("alpha"),
        "eta_method": print_data.get("eta_method"),
        "remaining_s": print_data.get("remaining_s"),
        "effective_speed": print_data.get("effective_speed"),
        "optimal_speed_pct": print_data.get("optimal_speed_pct"),
        "layer_alpha": print_data.get("layer_alpha"),
        "speed_adjusted": print_data.get("speed_adjusted", False),
        "speed_adjusted_to": print_data.get("speed_adjusted_to"),
        # Calibration data for tracking profile accuracy
        "time_cal_factor": print_data.get("time_cal_factor"),
        "profile_remaining_s": print_data.get("profile_remaining_s"),
        "confidence": print_data.get("confidence"),
    }

    with open(TRACKER_FILE, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
        f.flush()
        os.fsync(f.fileno())

    # Per-layer log: track layer transitions for actual vs predicted analysis
    _log_layer_transition(print_data)


# Track last logged layer to detect transitions
_last_logged_layer = {"layer": -1, "elapsed": 0, "filename": ""}


def _get_profile_layer_data(filename, layer_num, alpha, speed_factor, elapsed_s):
    """Get profile predicted time for a specific layer (for logging)."""
    try:
        from gcode_profile import (load_profile, calibrate_profile,
                                   get_layer_info)
        profile = load_profile(filename) if filename else None
        if not profile:
            return {}

        raw_layer = get_layer_info(profile, layer_num)
        raw_time = raw_layer.get("time_1x_s", 0) if raw_layer else 0
        raw_alpha = raw_layer.get("raw_alpha", 0) if raw_layer else 0
        move_time = raw_layer.get("move_time_s", 0) if raw_layer else 0
        fixed_overhead = raw_layer.get("fixed_overhead_s", 0) if raw_layer else 0
        retractions = raw_layer.get("retractions", 0) if raw_layer else 0
        segments = raw_layer.get("segments", 0) if raw_layer else 0

        # Get calibrated data
        cal_data = {}
        if alpha and layer_num > 0:
            cal = calibrate_profile(profile, alpha, layer_num,
                                    speed_factor, elapsed_s)
            for l in cal:
                if l["layer"] == layer_num:
                    cal_data = l
                    break

        return {
            "profile_time_1x_s": round(raw_time, 1),
            "profile_raw_alpha": round(raw_alpha, 4),
            "profile_move_time_s": round(move_time, 1),
            "profile_fixed_overhead_s": round(fixed_overhead, 1),
            "profile_retractions": retractions,
            "profile_segments": segments,
            "profile_calibrated_time_s": cal_data.get("time_calibrated_s", 0),
            "profile_calibrated_alpha": cal_data.get("calibrated_alpha", 0),
            "profile_optimal_speed_pct": cal_data.get("optimal_speed_pct", 0),
        }
    except Exception:
        return {}


def _log_layer_transition(print_data):
    """Log when a new layer starts, recording actual time for the previous layer.

    Captures comprehensive per-layer data for optimization:
    - Actual time vs profile predicted time
    - Speed factor, alpha (global + per-layer)
    - Profile calibration data (cal factor, optimal speed)
    - Auto-speed adjustments
    """
    global _last_logged_layer
    cur = print_data.get("current_layer", 0)
    elapsed = print_data.get("print_duration", 0)
    filename = print_data.get("filename", "")

    # Reset on new file
    if filename != _last_logged_layer["filename"]:
        _last_logged_layer = {"layer": -1, "elapsed": 0, "filename": filename}

    if cur <= _last_logged_layer["layer"]:
        return  # same layer, skip

    if _last_logged_layer["layer"] >= 0:
        # Log completed layer with full detail
        actual_time = elapsed - _last_logged_layer["elapsed"]
        alpha = print_data.get("alpha")
        speed_factor = print_data.get("speed_factor", 1.0)

        entry = {
            "ts": datetime.now().isoformat(),
            "file": filename,
            "layer": _last_logged_layer["layer"],
            "actual_time_s": round(actual_time, 1),
            "elapsed_total_s": round(elapsed, 1),
            "speed_factor": speed_factor,
            "alpha": alpha,
            "layer_alpha": print_data.get("layer_alpha"),
            "effective_speed": print_data.get("effective_speed"),
            "optimal_speed_pct": print_data.get("optimal_speed_pct"),
            "remaining_s": print_data.get("remaining_s"),
            "eta_method": print_data.get("eta_method"),
            "speed_adjusted": print_data.get("speed_adjusted", False),
            "speed_adjusted_to": print_data.get("speed_adjusted_to"),
            "live_velocity": print_data.get("live_velocity", 0),
            "bed_temp": print_data.get("bed_temp", 0),
            "nozzle_temp": print_data.get("nozzle_temp", 0),
        }

        # Add profile predicted data for this layer
        profile_data = _get_profile_layer_data(
            filename, _last_logged_layer["layer"],
            alpha, speed_factor, _last_logged_layer["elapsed"])
        entry.update(profile_data)

        # Calculate prediction error if we have both actual and predicted
        cal_time = profile_data.get("profile_calibrated_time_s", 0)
        if cal_time > 0 and actual_time > 0:
            entry["prediction_error_pct"] = round(
                (cal_time - actual_time) / actual_time * 100, 1)

        try:
            with open(LAYER_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            log.debug("suppressed", exc_info=True)

    _last_logged_layer = {"layer": cur, "elapsed": elapsed, "filename": filename}


PRINT_SUMMARY_DIR = os.path.join(TRACKER_DIR, "print_summaries")
os.makedirs(PRINT_SUMMARY_DIR, exist_ok=True)


def save_completed_job(job_data):
    """Save a completed job for future correction factor tuning.

    Also generates a comprehensive per-layer summary for optimization.
    """
    jobs = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            jobs = json.load(f)

    actual_time = job_data.get("print_duration", 0)
    estimated_time = job_data.get("estimated_time", 0)

    jobs.append({
        "ts": datetime.now().isoformat(),
        "file": job_data.get("filename", ""),
        "estimated_time": estimated_time,
        "actual_time": actual_time,
        "speed_factor": job_data.get("speed_factor", 1.0),
        "ratio": (actual_time / estimated_time) if estimated_time else None,
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

    # Generate comprehensive print summary
    _generate_print_summary(job_data)


def _generate_print_summary(job_data):
    """Generate a detailed per-layer summary comparing predicted vs actual.

    Saved as JSON in /tmp/printer_status/print_summaries/<filename>_<timestamp>.json
    for future analysis and optimization.
    """
    filename = job_data.get("filename", "unknown")
    actual_total = job_data.get("print_duration", 0)
    estimated_total = job_data.get("estimated_time", 0)

    summary = {
        "generated": datetime.now().isoformat(),
        "filename": filename,
        "actual_total_s": round(actual_total, 1),
        "actual_total_h": round(actual_total / 3600, 2) if actual_total else 0,
        "slicer_estimated_s": round(estimated_total, 1),
        "slicer_accuracy_pct": round(
            estimated_total / actual_total * 100, 1) if actual_total else 0,
        "final_speed_factor": job_data.get("speed_factor", 1.0),
        "final_alpha": job_data.get("alpha"),
    }

    # Load profile for per-layer analysis
    try:
        from gcode_profile import load_profile
        profile = load_profile(filename) if filename else None
        if profile:
            summary["profile_total_1x_s"] = profile.get("total_time_1x_s", 0)
            summary["profile_total_1x_h"] = round(
                profile.get("total_time_1x_s", 0) / 3600, 2)
            summary["profile_total_optimal_s"] = profile.get(
                "total_time_optimal_s", 0)
            summary["profile_layers"] = profile.get("total_layers", 0)
            summary["profile_generated"] = profile.get("generated", "")

            # Per-layer profile data
            layer_profiles = []
            for l in profile.get("layers", []):
                layer_profiles.append({
                    "layer": l["layer"],
                    "z": l.get("z", 0),
                    "segments": l.get("segments", 0),
                    "distance_mm": l.get("distance_mm", 0),
                    "raw_alpha": l.get("raw_alpha", l.get("alpha", 0)),
                    "optimal_speed_pct": l.get("optimal_speed_pct", 100),
                    "time_1x_s": l.get("time_1x_s", 0),
                    "move_time_s": l.get("move_time_s", 0),
                    "fixed_overhead_s": l.get("fixed_overhead_s", 0),
                    "retractions": l.get("retractions", 0),
                    "zhop_count": l.get("zhop_count", 0),
                })
            summary["profile_per_layer"] = layer_profiles
    except Exception:
        log.debug("suppressed", exc_info=True)

    # Load layer log for actual per-layer times
    try:
        if os.path.exists(LAYER_LOG_FILE):
            actual_layers = []
            with open(LAYER_LOG_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("file") == filename:
                            actual_layers.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
            summary["actual_per_layer"] = actual_layers
            summary["actual_layers_logged"] = len(actual_layers)

            # Calculate per-layer accuracy stats
            if actual_layers and summary.get("profile_per_layer"):
                profile_by_layer = {
                    l["layer"]: l for l in summary["profile_per_layer"]}
                errors = []
                for al in actual_layers:
                    pl = profile_by_layer.get(al["layer"])
                    if pl and al.get("actual_time_s", 0) > 0:
                        err = al.get("prediction_error_pct")
                        if err is not None:
                            errors.append(err)
                if errors:
                    summary["layer_prediction_stats"] = {
                        "count": len(errors),
                        "mean_error_pct": round(sum(errors) / len(errors), 1),
                        "median_error_pct": round(
                            sorted(errors)[len(errors) // 2], 1),
                        "max_overestimate_pct": round(max(errors), 1),
                        "max_underestimate_pct": round(min(errors), 1),
                    }
    except Exception:
        log.debug("suppressed", exc_info=True)

    # Load tracker data for speed history
    try:
        if os.path.exists(TRACKER_FILE):
            speed_history = []
            with open(TRACKER_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("file") == filename:
                            speed_history.append({
                                "ts": entry.get("ts"),
                                "progress": entry.get("progress"),
                                "speed_factor": entry.get("speed_factor"),
                                "layer": entry.get("current_layer"),
                                "remaining_s": entry.get("remaining_s"),
                                "alpha": entry.get("alpha"),
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
            summary["speed_history_count"] = len(speed_history)
            summary["speed_history"] = speed_history
    except Exception:
        log.debug("suppressed", exc_info=True)

    # Save summary
    safe_name = filename.replace("/", "_").replace(" ", "_")[:80]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(PRINT_SUMMARY_DIR, f"{safe_name}_{ts}.json")
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        log.debug("suppressed", exc_info=True)


def get_correction_factor():
    """Calculate slicer correction from completed job history.

    Returns average actual/estimated ratio from completed prints at 1x speed.
    Falls back to DEFAULT_SLICER_CORRECTION if insufficient data.
    """
    if not os.path.exists(HISTORY_FILE):
        return DEFAULT_SLICER_CORRECTION

    with open(HISTORY_FILE) as f:
        jobs = json.load(f)

    ratios = [j["ratio"] for j in jobs
              if j.get("ratio") and 0.5 < j["ratio"] < 3.0
              and j.get("speed_factor", 1.0) <= 1.05]

    if len(ratios) < 3:
        return DEFAULT_SLICER_CORRECTION

    return sum(ratios) / len(ratios)
