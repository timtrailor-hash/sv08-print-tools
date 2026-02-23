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
import math
import os
from datetime import datetime, timedelta

TRACKER_DIR = "/tmp/printer_status"
TRACKER_FILE = os.path.join(TRACKER_DIR, "print_tracker.jsonl")
HISTORY_FILE = os.path.join(TRACKER_DIR, "completed_jobs.json")
ALPHA_FILE = os.path.join(TRACKER_DIR, "current_alpha.json")
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

def speed_time_ratio(speed_factor, alpha):
    """Predict time ratio T(S)/T(1) using trapezoid move physics.

    For a print with geometry complexity alpha at speed factor S:
        R(S) = (1/S + α·S) / (1 + α)

    Returns < 1.0 if speed factor helps, > 1.0 if it hurts.
    When α > 1/S, increasing speed makes the print SLOWER because the
    extra acceleration overhead exceeds the cruise time savings.
    """
    S = max(speed_factor, 0.1)
    a = max(alpha, 0.001)
    return (1.0 / S + a * S) / (1.0 + a)


def alpha_from_measurement(effective_speed, speed_factor):
    """Back-calculate geometry complexity α from measured effective speed.

    Given effective_speed = T_slicer(1x) / T_actual at speed factor S:
        α = (1 - eff/S) / (eff·S - 1)

    Only works when speed_factor > 1 (at S=1, α cancels out of the
    time ratio formula, so it can't be determined).
    Returns None if speed_factor ≈ 1 or data is insufficient.
    """
    S = speed_factor
    if S <= 1.05 or effective_speed <= 0:
        return None

    denom = effective_speed * S - 1.0
    if denom <= 0.01:
        # effective_speed * S ≈ 1 means print is heavily accel-limited
        return 1.5

    numer = 1.0 - effective_speed / S
    alpha = numer / denom
    return max(0.0, min(alpha, 2.0))


def optimal_speed_factor(alpha):
    """Speed factor that minimises print time for this geometry.

    S_optimal = 1/√α. Beyond this, extra speed makes the print slower.
    Returns the optimal multiplier (e.g. 1.35 means 135%).
    """
    if alpha <= 0.001:
        return 10.0
    return min(1.0 / math.sqrt(alpha), 10.0)


def speed_factor_benefit(speed_factor, alpha):
    """Human-readable summary of speed factor impact.

    Returns (time_pct, description) where time_pct is the predicted
    percentage of baseline time (e.g. 85 means 15% faster).
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

def _load_alpha_state():
    """Load persisted alpha measurement for the current print."""
    if os.path.exists(ALPHA_FILE):
        try:
            with open(ALPHA_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_alpha_state(state):
    """Persist alpha measurement for the current print."""
    with open(ALPHA_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Main ETA calculator ───────────────────────────────────────

def calculate_eta(progress_pct, print_duration_s, estimated_time_s,
                  speed_factor, live_velocity=0, commanded_speed=0,
                  current_layer=0, total_layers=0, filename=""):
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
    alpha_state = _load_alpha_state()
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
            })
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
        slicer_left = estimated_time_s * (1 - progress_pct / 100)

        # Method 1: Slicer-based (remaining slicer time / effective speed)
        slicer_remaining = (slicer_left / max(blended_eff, 0.1)) * correction

        # Method 2: Pace-based (linear extrapolation from progress)
        pace_total = print_duration_s / (progress_pct / 100)
        pace_remaining = pace_total - print_duration_s

        # ── Method 3: Profile-based (per-layer gcode analysis) ──
        profile_remaining = None
        try:
            from gcode_profile import (load_profile, calibrate_profile,
                                       calibrated_eta_remaining)
            profile = load_profile(filename) if filename else None
            if profile and alpha and current_layer > 2:
                cal = calibrate_profile(profile, alpha, current_layer, S,
                                               elapsed_time_s=print_duration_s)
                if cal:
                    # Estimate progress within current layer
                    pil = 0.5
                    if total_layers > 0:
                        expected_layer_frac = current_layer / total_layers
                        actual_frac = progress_pct / 100
                        pil = max(0.1, min(0.9,
                            (actual_frac - expected_layer_frac)
                            / (1.0 / total_layers)))
                    profile_remaining = calibrated_eta_remaining(
                        cal, current_layer, pil)
        except Exception:
            pass

        # ── Adaptive blend weights (learned from completed prints) ──
        try:
            from eta_learner import get_blend_weights, log_prediction
            weights = get_blend_weights(progress_pct, alpha)
            w_pace = weights.get("pace", 0.5)
            w_slicer = weights.get("slicer", 0.3)
            w_profile = weights.get("profile", 0.2)
            weights_source = "learned"
        except Exception:
            # Fallback to hardcoded if learner unavailable
            if progress_pct < 5:
                w_pace, w_slicer, w_profile = 0.0, 0.8, 0.2
            elif progress_pct < 20:
                f = (progress_pct - 5) / 15
                w_pace, w_slicer, w_profile = f * 0.5, (1-f) * 0.8, 0.2
            else:
                w_pace = min(0.7, 0.5 + (progress_pct - 20) / 200)
                w_slicer = (1 - w_pace) * 0.4
                w_profile = (1 - w_pace) * 0.6
            weights_source = "fallback"

        # If no profile available, redistribute its weight
        if not profile_remaining or profile_remaining <= 0:
            total_non_profile = w_pace + w_slicer
            if total_non_profile > 0:
                w_pace = w_pace / total_non_profile
                w_slicer = w_slicer / total_non_profile
            else:
                w_pace, w_slicer = 0.5, 0.5
            w_profile = 0.0

        # Normalize weights
        total_w = w_pace + w_slicer + w_profile
        if total_w > 0:
            w_pace /= total_w
            w_slicer /= total_w
            w_profile /= total_w

        remaining = (pace_remaining * w_pace +
                     slicer_remaining * w_slicer +
                     (profile_remaining or 0) * w_profile)

        result["method"] = (
            f"blend(pace={w_pace:.0%},slicer={w_slicer:.0%},"
            f"profile={w_profile:.0%},α={alpha:.2f},"
            f"{weights_source})")
        if profile_remaining and profile_remaining > 0:
            result["profile_remaining_s"] = round(profile_remaining)

        # Log individual method predictions for the learner
        try:
            log_prediction(
                filename=filename,
                progress_pct=progress_pct,
                alpha=alpha,
                elapsed_s=print_duration_s,
                pace_remaining_s=pace_remaining,
                slicer_remaining_s=slicer_remaining,
                profile_remaining_s=profile_remaining,
                blended_remaining_s=remaining,
                speed_factor=speed_factor,
                current_layer=current_layer,
            )
        except Exception:
            pass  # logging failure is non-critical

        # Confidence
        if progress_pct < 5:
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
    }

    with open(TRACKER_FILE, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def save_completed_job(job_data):
    """Save a completed job for future correction factor tuning."""
    jobs = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            jobs = json.load(f)

    jobs.append({
        "ts": datetime.now().isoformat(),
        "file": job_data.get("filename", ""),
        "estimated_time": job_data.get("estimated_time", 0),
        "actual_time": job_data.get("print_duration", 0),
        "speed_factor": job_data.get("speed_factor", 1.0),
        "ratio": (job_data.get("print_duration", 0) /
                  job_data.get("estimated_time", 1)) if job_data.get("estimated_time") else None,
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


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
