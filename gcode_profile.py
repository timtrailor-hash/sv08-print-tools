#!/usr/bin/env python3
"""Per-layer gcode complexity profiler for SV08 Max.

Streams a gcode file from Moonraker, analyzes each layer's geometry
using trapezoid move kinematics, and outputs a per-layer speed/time
profile. Enables:
  - Dynamic speed adjustment per layer (auto M220)
  - Accurate per-layer ETA predictions
  - Speed recommendations based on actual geometry

Usage:
    python3 gcode_profile.py                    # analyze current print
    python3 gcode_profile.py --file <name>      # analyze specific file
    python3 gcode_profile.py --auto-speed on    # enable dynamic speed
    python3 gcode_profile.py --auto-speed off   # disable dynamic speed
    python3 gcode_profile.py --status           # show profile summary
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

from config import MOONRAKER_URL as MOONRAKER
PROFILE_DIR = "/tmp/printer_status"
PROFILE_FILE = os.path.join(PROFILE_DIR, "gcode_profile.json")
AUTO_SPEED_FILE = os.path.join(PROFILE_DIR, "auto_speed.json")
os.makedirs(PROFILE_DIR, exist_ok=True)


# ── Printer kinematics (fetched live, these are fallbacks) ────────

DEFAULT_MAX_VELOCITY = 700.0      # mm/s
DEFAULT_MAX_ACCEL = 40000.0       # mm/s² (from printer.cfg, not OrcaSlicer)
DEFAULT_SCV = 5.0                 # square corner velocity mm/s


# ── Fetch printer config from Moonraker ───────────────────────────

def fetch_printer_kinematics():
    """Get kinematics from the printer's config file (not live state).

    The live toolhead values can be lowered by slicer SET_VELOCITY_LIMIT
    commands (e.g. OrcaSlicer sets ACCEL=5000-6000 per feature). We need
    the printer.cfg values for correct alpha calculations.
    """
    config = {
        "max_velocity": DEFAULT_MAX_VELOCITY,
        "max_accel": DEFAULT_MAX_ACCEL,
        "scv": DEFAULT_SCV,
    }

    # First try: get config file values (the real hardware limits)
    try:
        url = (f"{MOONRAKER}/printer/objects/query?"
               "configfile=settings")
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        settings = data["result"]["status"]["configfile"]["settings"]
        printer = settings.get("printer", {})
        if printer.get("max_velocity"):
            config["max_velocity"] = float(printer["max_velocity"])
        if printer.get("max_accel"):
            config["max_accel"] = float(printer["max_accel"])
        if printer.get("square_corner_velocity"):
            config["scv"] = float(printer["square_corner_velocity"])
        return config
    except Exception:
        pass

    # Fallback: live toolhead values (may be lowered by slicer)
    try:
        url = (f"{MOONRAKER}/printer/objects/query?"
               "toolhead=max_velocity,max_accel,square_corner_velocity")
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        th = data["result"]["status"]["toolhead"]
        config["max_velocity"] = th.get("max_velocity", DEFAULT_MAX_VELOCITY)
        config["max_accel"] = th.get("max_accel", DEFAULT_MAX_ACCEL)
        config["scv"] = th.get("square_corner_velocity", DEFAULT_SCV)
    except Exception as e:
        print(f"Warning: couldn't fetch kinematics ({e}), using defaults",
              file=sys.stderr)

    return config


def get_current_filename():
    """Get the filename of the currently printing gcode."""
    url = (f"{MOONRAKER}/printer/objects/query?"
           "print_stats=filename,state")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        ps = data["result"]["status"]["print_stats"]
        return ps.get("filename", ""), ps.get("state", "")
    except Exception:
        return "", ""


# ── Trapezoid move kinematics ─────────────────────────────────────

def move_time(distance, feedrate, accel, junction_v=5.0):
    """Calculate time for a single move using trapezoid kinematics.

    Assumes the move starts and ends at junction_v.

    Returns (total_time, cruise_time, accel_time).
    """
    if distance < 0.001:
        return 0.0, 0.0, 0.0

    v = min(feedrate, 700.0)  # cap at max_velocity
    jv = min(junction_v, v)

    # Distance needed to accelerate from junction_v to v
    d_accel = (v * v - jv * jv) / (2.0 * accel)
    d_decel = d_accel  # symmetric

    if d_accel + d_decel > distance:
        # Triangle profile — never reaches target feedrate
        v_peak_sq = jv * jv + accel * distance
        v_peak = math.sqrt(v_peak_sq)
        t_accel = (v_peak - jv) / accel
        t_decel = (v_peak - jv) / accel
        return t_accel + t_decel, 0.0, t_accel + t_decel
    else:
        # Trapezoid profile — reaches cruise speed
        t_accel = (v - jv) / accel
        t_decel = t_accel
        d_cruise = distance - d_accel - d_decel
        t_cruise = d_cruise / v if v > 0 else 0.0
        total = t_accel + t_cruise + t_decel
        return total, t_cruise, t_accel + t_decel


def move_time_at_speed(distance, feedrate, accel, speed_factor,
                       max_vel=700.0, junction_v=5.0):
    """Calculate move time at a given speed factor."""
    scaled_feed = min(feedrate * speed_factor, max_vel)
    return move_time(distance, scaled_feed, accel, junction_v)[0]


# ── Junction velocity estimation ─────────────────────────────────

def sample_junction_velocity(cos_theta_samples, scv, avg_feedrate):
    """Estimate effective junction velocity from sampled direction changes.

    Uses the 25th percentile of junction cosines (conservative — represents
    the worst quarter of direction changes, not the absolute worst case).

    Args:
        cos_theta_samples: list of cos(θ) between consecutive extrusion moves
            cos=1 → straight, cos=0 → 90° turn, cos=-1 → reversal
        scv: square corner velocity (mm/s)
        avg_feedrate: average feedrate for the layer (mm/s)

    Returns effective junction velocity (mm/s), between scv and avg_feedrate.
    """
    if not cos_theta_samples or len(cos_theta_samples) < 5:
        return scv  # not enough data, fall back to worst case

    cos_theta_samples.sort()
    p25_idx = len(cos_theta_samples) // 4
    cos_p25 = cos_theta_samples[p25_idx]

    # Map cos_theta to junction velocity:
    # cos=1 (straight): jv ≈ feedrate (no direction change)
    # cos=0 (90° turn): jv ≈ scv (Klipper's corner velocity limit)
    # cos=-1 (reversal): jv ≈ 0 (full stop and restart)
    #
    # Use a piecewise linear mapping that matches Klipper's behavior:
    # - Above cos=0: interpolate scv → feedrate
    # - Below cos=0: interpolate 0 → scv
    if cos_p25 >= 0:
        jv = scv + (avg_feedrate - scv) * cos_p25
    else:
        jv = scv * (1.0 + cos_p25)  # scv at cos=0, 0 at cos=-1

    return max(0.5, min(jv, avg_feedrate))


# ── Alpha calculation ─────────────────────────────────────────────

def layer_alpha_from_geometry(avg_segment_mm, avg_feedrate, accel,
                              junction_v=5.0):
    """Calculate alpha for a layer from its geometry.

    α = (v² - jv²) / (a × d_avg)

    This accounts for junction velocity — at higher jv, less acceleration
    is needed, so alpha is lower (meaning speed factor helps more).

    Cap at 20.0 to allow differentiation between highly complex layers.
    """
    if avg_segment_mm > 0.01 and avg_feedrate > 0 and accel > 0:
        v_sq = avg_feedrate ** 2
        jv_sq = min(junction_v, avg_feedrate) ** 2
        alpha = max(0.001, (v_sq - jv_sq)) / (accel * avg_segment_mm)
        return min(alpha, 20.0)
    return 1.0


def optimal_speed(alpha):
    """Optimal speed factor = 1/sqrt(alpha)."""
    if alpha <= 0.001:
        return 10.0
    return min(1.0 / math.sqrt(alpha), 10.0)


# ── Calibration ──────────────────────────────────────────────────

def calibrate_profile(profile, measured_alpha, current_layer,
                      speed_factor=1.0, elapsed_time_s=0):
    """Calibrate raw profile alphas and times using live measurements.

    Two calibrations:
    1. Alpha calibration: raw gcode alphas → scaled to match measured alpha
    2. Time calibration: raw time_1x_s → scaled to match actual elapsed time

    Returns a list of dicts: [{layer, calibrated_alpha, optimal_speed_pct,
                                time_calibrated_s}, ...]
    """
    if not profile or not profile.get("layers"):
        return []

    # Weighted average of raw alphas for layers up to current
    # (weighted by time_1x to give more weight to longer layers)
    total_weight = 0.0
    weighted_sum = 0.0
    profile_elapsed = 0.0
    for l in profile["layers"]:
        if l["layer"] > current_layer:
            break
        w = l.get("time_1x_s", 1.0)
        weighted_sum += l.get("raw_alpha", l.get("alpha", 1.0)) * w
        total_weight += w
        profile_elapsed += l.get("time_1x_s", 0)

    if total_weight < 1.0 or weighted_sum < 0.01:
        return []

    avg_raw = weighted_sum / total_weight
    if avg_raw < 0.001:
        avg_raw = 0.001

    alpha_cal = measured_alpha / avg_raw

    # Time calibration
    time_cal_factor = 1.0
    if elapsed_time_s > 60 and profile_elapsed > 60:
        S = max(speed_factor, 0.1)
        predicted_elapsed = profile_elapsed * (
            (1.0 / S + measured_alpha * S) / (1.0 + measured_alpha))
        time_cal_factor = elapsed_time_s / predicted_elapsed
        time_cal_factor = max(0.3, min(time_cal_factor, 1.5))

    # Apply calibration to all layers
    calibrated = []
    for l in profile["layers"]:
        raw_a = l.get("raw_alpha", l.get("alpha", 1.0))
        cal_a = max(0.01, min(raw_a * alpha_cal, 2.0))
        opt = optimal_speed(cal_a)
        opt_pct = max(round(opt * 100), 50)

        ratio = (1.0 / opt + cal_a * opt) / (1.0 + cal_a)
        time_cal = l.get("time_1x_s", 0) * ratio * time_cal_factor

        calibrated.append({
            "layer": l["layer"],
            "raw_alpha": round(raw_a, 4),
            "calibrated_alpha": round(cal_a, 4),
            "optimal_speed_pct": opt_pct,
            "time_calibrated_s": round(time_cal, 1),
            "time_1x_s": l.get("time_1x_s", 0),
        })

    return calibrated


def calibrated_eta_remaining(calibrated_layers, current_layer,
                             progress_in_layer=0.5):
    """Calculate remaining time from calibrated layer data."""
    remaining = 0.0
    for l in calibrated_layers:
        if l["layer"] < current_layer:
            continue
        elif l["layer"] == current_layer:
            remaining += l["time_calibrated_s"] * (1.0 - progress_in_layer)
        else:
            remaining += l["time_calibrated_s"]
    return remaining


# ── Fast gcode parser ────────────────────────────────────────────

def parse_g_move(line):
    """Fast G0/G1 parameter extraction using str.split().

    ~3x faster than regex-based parse_gcode_params() on millions of lines.
    Only extracts X, Y, Z, E, F parameters (all we need for move analysis).
    """
    params = {}
    for token in line.split():
        if len(token) < 2:
            continue
        key = token[0]
        if key in 'XYZEF':
            try:
                params[key] = float(token[1:])
            except ValueError:
                pass
        elif token[0] == ';':
            break  # inline comment, stop parsing
    return params


# ── Gcode analyzer ───────────────────────────────────────────────

def analyze_gcode(filename, config, progress_callback=None):
    """Stream and analyze a gcode file from Moonraker.

    Two-phase approach:
    1. Parse: stream gcode, collect per-layer geometry stats and
       direction-change samples (no per-move time calculations)
    2. Analyze: compute effective junction velocity per layer from
       direction samples, then calculate corrected alpha and time estimates

    Returns a profile dict with per-layer complexity data.
    """
    enc = urllib.parse.quote(filename)
    url = f"{MOONRAKER}/server/files/gcodes/{enc}"

    max_vel = config["max_velocity"]
    accel = config["max_accel"]
    scv = config["scv"]

    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)

    content_length = resp.headers.get("Content-Length")
    file_size_mb = int(content_length) / (1024 * 1024) if content_length else 0

    # State tracking
    cur_x, cur_y, cur_z = 0.0, 0.0, 0.0
    cur_feedrate = 50.0  # mm/s default
    cur_layer = 0
    line_count = 0

    # Per-layer accumulators — running sums, no per-move lists
    layer_data = {}
    layer_z = {}

    JUNCTION_SAMPLE_SIZE = 1000  # max junction cosine samples per layer

    def new_layer_data():
        return {
            "segments": 0,       # extrusion move count
            "distance_mm": 0.0,  # total extrusion XY distance
            "feedrate_sum": 0.0, # sum of feedrates (for averaging)
            "accel_sum": 0.0,    # sum of per-feature accelerations
            "travel_distance": 0.0,  # total travel (non-extrusion) XY distance
            "junction_cos": [],  # reservoir-sampled cos(θ) between moves
            "prev_dx": 0.0,
            "prev_dy": 0.0,
            "prev_valid": False, # whether prev direction is set
        }

    layer_data[0] = new_layer_data()
    layer_z[0] = 0.0

    cur_feature_accel = accel  # tracks SET_VELOCITY_LIMIT ACCEL=

    last_progress_time = time.time()
    start_time = time.time()
    bytes_read = 0
    file_size = int(content_length) if content_length else 0

    # ── Phase 1: Parse ──
    for raw_line in resp:
        line_count += 1
        bytes_read += len(raw_line)

        line = raw_line.decode('utf-8', errors='replace').rstrip()

        # Progress callback every 2 seconds
        if progress_callback and time.time() - last_progress_time > 2.0:
            elapsed = time.time() - start_time
            pct = min(99, bytes_read / file_size * 100) if file_size else 0
            progress_callback(pct, line_count, cur_layer, elapsed)
            last_progress_time = time.time()

        # Fast comment/command detection (no regex)
        if line.startswith(';'):
            # Layer change: ;LAYER_CHANGE or ; LAYER_CHANGE
            stripped = line[1:].lstrip()
            if stripped.startswith('LAYER_CHANGE'):
                cur_layer += 1
                if cur_layer not in layer_data:
                    layer_data[cur_layer] = new_layer_data()
                continue
            # Z height: ;Z:0.45 or ; Z:0.45
            if stripped.startswith('Z:'):
                try:
                    layer_z[cur_layer] = float(stripped[2:].strip())
                except ValueError:
                    pass
                continue
            continue  # skip other comments

        # SET_VELOCITY_LIMIT ACCEL= (per-feature acceleration from slicer)
        if line.startswith('SET_VELOCITY_LIMIT') and 'ACCEL=' in line:
            for token in line.split():
                if token.startswith('ACCEL='):
                    try:
                        cur_feature_accel = float(token[6:])
                    except ValueError:
                        pass
            continue

        # Skip SET_PRINT_STATS_INFO and other non-move commands
        if line.startswith('SET_') or line.startswith('M') or line.startswith('T'):
            continue

        # Only process G0/G1
        if len(line) < 2:
            continue
        c0 = line[0]
        if c0 != 'G' and c0 != 'g':
            continue
        c1 = line[1] if len(line) > 1 else ''
        if c1 == '0':
            is_g1 = False
        elif c1 == '1':
            is_g1 = True
        else:
            continue
        # Must be followed by space/tab/end (not G10, G28, etc.)
        if len(line) > 2 and line[2] not in (' ', '\t', ';'):
            continue

        # Fast parameter extraction
        params = parse_g_move(line[2:].lstrip())  # skip "G0"/"G1" prefix

        # Update feedrate (F is mm/min in gcode)
        if 'F' in params:
            cur_feedrate = params['F'] / 60.0

        new_x = params.get('X', cur_x)
        new_y = params.get('Y', cur_y)
        new_z = params.get('Z', cur_z)
        cur_z = new_z

        # Calculate XY distance
        dx = new_x - cur_x
        dy = new_y - cur_y
        dist_sq = dx * dx + dy * dy
        cur_x, cur_y = new_x, new_y

        if dist_sq < 0.000001:  # < 0.001mm
            continue

        dist = math.sqrt(dist_sq)
        is_extrusion = is_g1 and 'E' in params

        if cur_layer not in layer_data:
            layer_data[cur_layer] = new_layer_data()
        ld = layer_data[cur_layer]

        if not is_extrusion:
            # Travel move — just accumulate distance
            ld["travel_distance"] += dist
            continue

        # Extrusion move — accumulate geometry stats
        ld["segments"] += 1
        ld["distance_mm"] += dist
        ld["feedrate_sum"] += min(cur_feedrate, max_vel)
        ld["accel_sum"] += min(cur_feature_accel, accel)

        # Junction angle sampling (reservoir sampling, max 1000 per layer)
        if ld["prev_valid"]:
            prev_dx, prev_dy = ld["prev_dx"], ld["prev_dy"]
            prev_d = math.sqrt(prev_dx * prev_dx + prev_dy * prev_dy)
            if prev_d > 0.001 and dist > 0.001:
                cos_t = (prev_dx * dx + prev_dy * dy) / (prev_d * dist)
                cos_t = max(-1.0, min(1.0, cos_t))
                n = len(ld["junction_cos"])
                if n < JUNCTION_SAMPLE_SIZE:
                    ld["junction_cos"].append(cos_t)
                else:
                    # Reservoir sampling: replace random element
                    j = random.randint(0, ld["segments"])
                    if j < JUNCTION_SAMPLE_SIZE:
                        ld["junction_cos"][j] = cos_t

        ld["prev_dx"] = dx
        ld["prev_dy"] = dy
        ld["prev_valid"] = True

    resp.close()
    parse_time = time.time() - start_time

    # ── Phase 2: Analyze ──
    layers = []
    for layer_num in sorted(layer_data.keys()):
        ld = layer_data[layer_num]
        if ld["segments"] == 0 and ld["travel_distance"] < 1.0:
            continue

        n_seg = ld["segments"]
        if n_seg > 0:
            avg_seg = ld["distance_mm"] / n_seg
            avg_feed = ld["feedrate_sum"] / n_seg
            avg_accel = ld["accel_sum"] / n_seg
        else:
            avg_seg = 0
            avg_feed = 50.0
            avg_accel = accel

        # Compute effective junction velocity from direction samples
        eff_jv = sample_junction_velocity(ld["junction_cos"], scv, avg_feed)

        # Corrected alpha using effective junction velocity
        raw_a = layer_alpha_from_geometry(avg_seg, avg_feed, avg_accel, eff_jv)
        opt = optimal_speed(raw_a)
        opt_pct = max(round(opt * 100), 50)  # floor at 50%

        # Time estimate: extrusion time + travel time
        if n_seg > 0:
            ext_time = n_seg * move_time(avg_seg, avg_feed, avg_accel, eff_jv)[0]
        else:
            ext_time = 0.0
        # Travel time: simple estimate with 1.3x overhead for accel/decel
        travel_time = ld["travel_distance"] / max(avg_feed, 10.0) * 1.3
        total_time = ext_time + travel_time

        # Time at optimal speed
        if raw_a > 0.001:
            ratio_optimal = (1.0 / opt + raw_a * opt) / (1.0 + raw_a)
        else:
            ratio_optimal = 1.0 / opt
        time_optimal = total_time * ratio_optimal

        layers.append({
            "layer": layer_num,
            "z": round(layer_z.get(layer_num, 0), 3),
            "segments": n_seg,
            "distance_mm": round(ld["distance_mm"], 1),
            "avg_segment_mm": round(avg_seg, 2),
            "avg_feedrate": round(avg_feed, 1),
            "junction_v": round(eff_jv, 1),
            "raw_alpha": round(raw_a, 4),
            "alpha": round(raw_a, 4),
            "optimal_speed_pct": opt_pct,
            "time_1x_s": round(total_time, 1),
            "time_optimal_s": round(time_optimal, 1),
        })

    elapsed_total = time.time() - start_time
    total_time_1x = sum(l["time_1x_s"] for l in layers)
    total_time_optimal = sum(l["time_optimal_s"] for l in layers)

    profile = {
        "filename": filename,
        "generated": datetime.now().isoformat(),
        "analysis_time_s": round(elapsed_total, 1),
        "parse_time_s": round(parse_time, 1),
        "file_size_mb": round(file_size_mb, 1),
        "lines_parsed": line_count,
        "printer_config": config,
        "total_layers": len(layers),
        "total_time_1x_s": round(total_time_1x, 1),
        "total_time_optimal_s": round(total_time_optimal, 1),
        "time_saved_pct": round((1 - total_time_optimal / total_time_1x) * 100, 1)
                          if total_time_1x > 0 else 0,
        "layers": layers,
    }

    return profile


# ── Auto-speed control ────────────────────────────────────────────

def load_auto_speed():
    """Load auto-speed configuration."""
    if os.path.exists(AUTO_SPEED_FILE):
        try:
            with open(AUTO_SPEED_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"enabled": False, "mode": "optimal", "min_speed_pct": 80,
            "max_speed_pct": 200, "skip_first_layers": 2}


def save_auto_speed(config):
    """Save auto-speed configuration."""
    with open(AUTO_SPEED_FILE, "w") as f:
        json.dump(config, f, indent=2)


def set_printer_speed(speed_pct):
    """Send M220 speed change to the printer."""
    cmd = f"M220 S{speed_pct}"
    enc = urllib.parse.quote(cmd)
    url = f"{MOONRAKER}/printer/gcode/script?script={enc}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            result = json.loads(r.read())
        return result.get("result") == "ok"
    except Exception as e:
        print(f"Error setting speed: {e}", file=sys.stderr)
        return False


def smooth_speed_target(target_pct, previous_pct, max_change=30):
    """Limit speed change between adjacent layers.

    Prevents jarring jumps (e.g. 103% → 288%) by ramping over multiple
    layers. Max change defaults to ±30% per layer transition.
    """
    if previous_pct is None:
        return target_pct
    delta = target_pct - previous_pct
    if abs(delta) > max_change:
        return previous_pct + max_change * (1 if delta > 0 else -1)
    return target_pct


def adjust_speed_for_layer(profile, current_layer, auto_config):
    """Check if speed should be adjusted for the current layer.

    Returns (should_adjust, target_speed_pct, reason) or (False, 0, "").
    """
    if not auto_config.get("enabled"):
        return False, 0, ""

    # Find current layer in profile
    layer_info = None
    for l in profile["layers"]:
        if l["layer"] == current_layer:
            layer_info = l
            break

    if not layer_info:
        return False, 0, "layer not in profile"

    # Skip first N layers
    skip = auto_config.get("skip_first_layers", 2)
    if current_layer < skip:
        return False, 0, f"skipping first {skip} layers"

    opt = layer_info["optimal_speed_pct"]

    # Apply mode
    mode = auto_config.get("mode", "optimal")
    if mode == "conservative":
        target = max(round(opt * 0.95), auto_config.get("min_speed_pct", 80))
    else:
        target = opt

    # Smooth transition from previous layer's speed
    last = auto_config.get("last_adjustment", {})
    previous_pct = last.get("speed")
    target = smooth_speed_target(target, previous_pct,
                                 auto_config.get("max_change_pct", 30))

    # Clamp to bounds
    target = max(auto_config.get("min_speed_pct", 80),
                 min(target, auto_config.get("max_speed_pct", 200)))

    # Check if already adjusted for this layer
    if last.get("layer") == current_layer:
        return False, 0, "already adjusted this layer"

    return True, target, f"layer {current_layer} alpha={layer_info['alpha']:.3f}"


# ── Profile loading (for use by other modules) ───────────────────

def load_profile(filename=None):
    """Load a cached gcode profile. Returns None if not available."""
    if not os.path.exists(PROFILE_FILE):
        return None
    try:
        with open(PROFILE_FILE) as f:
            profile = json.load(f)
        if filename and profile.get("filename") != filename:
            return None
        return profile
    except (json.JSONDecodeError, OSError):
        return None


def get_layer_info(profile, layer_num):
    """Get info for a specific layer from a profile."""
    for l in profile["layers"]:
        if l["layer"] == layer_num:
            return l
    return None


def profile_eta_remaining(profile, current_layer, progress_in_layer=0.5,
                          speed_factor=1.0):
    """Calculate remaining time using per-layer profile.

    Args:
        profile: loaded gcode profile
        current_layer: current layer number
        progress_in_layer: estimated progress within current layer (0-1)
        speed_factor: current M220 speed factor (1.0 = 100%)

    Returns remaining seconds.
    """
    remaining = 0.0
    for l in profile["layers"]:
        if l["layer"] < current_layer:
            continue

        alpha = l.get("calibrated_alpha", l.get("alpha", 1.0))
        t_1x = l.get("time_1x_s", 0)

        S = max(speed_factor, 0.1)
        ratio = (1.0 / S + alpha * S) / (1.0 + alpha)
        layer_time = t_1x * ratio

        if l["layer"] == current_layer:
            remaining += layer_time * (1.0 - progress_in_layer)
        else:
            remaining += layer_time

    return remaining


# ── CLI ───────────────────────────────────────────────────────────

def print_profile_summary(profile):
    """Print a human-readable profile summary."""
    print(f"\n{'=' * 70}")
    print(f"GCODE LAYER PROFILE")
    print(f"{'=' * 70}")
    print(f"File:           {profile['filename']}")
    print(f"Generated:      {profile['generated']}")
    print(f"Analysis time:  {profile['analysis_time_s']}s "
          f"(parse: {profile.get('parse_time_s', '?')}s)")
    print(f"File size:      {profile['file_size_mb']} MB")
    print(f"Lines parsed:   {profile['lines_parsed']:,}")
    print(f"Total layers:   {profile['total_layers']}")
    print()

    cfg = profile["printer_config"]
    print(f"Printer config: accel={cfg['max_accel']}, "
          f"max_vel={cfg['max_velocity']}, scv={cfg['scv']}")
    print()

    t1x = profile["total_time_1x_s"]
    topt = profile["total_time_optimal_s"]
    saved = profile["time_saved_pct"]
    print(f"Total time at 1x:       {t1x/3600:.1f}h")
    print(f"Total time at optimal:  {topt/3600:.1f}h")
    print(f"Time saved:             {saved}%")
    print()

    # Layer summary table
    print(f"{'Layer':>5} {'Z mm':>6} {'Segs':>7} {'Dist mm':>8} "
          f"{'AvgSeg':>7} {'JuncV':>6} {'Alpha':>6} {'OptSpd%':>7} "
          f"{'T@1x':>6} {'T@opt':>6}")
    print("-" * 78)

    for l in profile["layers"]:
        t1 = f"{l['time_1x_s']:.0f}s"
        to = f"{l['time_optimal_s']:.0f}s"
        jv = l.get('junction_v', '?')
        jv_str = f"{jv:.0f}" if isinstance(jv, (int, float)) else str(jv)
        print(f"{l['layer']:>5} {l['z']:>6.2f} {l['segments']:>7,} "
              f"{l['distance_mm']:>8.0f} {l['avg_segment_mm']:>7.2f} "
              f"{jv_str:>6} {l['alpha']:>6.3f} {l['optimal_speed_pct']:>7} "
              f"{t1:>6} {to:>6}")

    # Alpha distribution summary
    alphas = [l["alpha"] for l in profile["layers"] if l["segments"] > 0]
    if alphas:
        print(f"\nAlpha range: {min(alphas):.3f} - {max(alphas):.3f} "
              f"(mean {sum(alphas)/len(alphas):.3f})")
        simple = sum(1 for a in alphas if a < 0.2)
        moderate = sum(1 for a in alphas if 0.2 <= a < 0.5)
        complex_ = sum(1 for a in alphas if 0.5 <= a < 1.0)
        extreme = sum(1 for a in alphas if a >= 1.0)
        print(f"Simple (<0.2):    {simple}")
        print(f"Moderate (0.2-0.5): {moderate}")
        print(f"Complex (0.5-1.0):  {complex_}")
        print(f"Extreme (>1.0):     {extreme}")


def main():
    parser = argparse.ArgumentParser(description="Gcode per-layer profiler")
    parser.add_argument("--file", help="Gcode filename to analyze")
    parser.add_argument("--auto-speed", choices=["on", "off"],
                        help="Enable/disable dynamic speed adjustment")
    parser.add_argument("--mode", choices=["optimal", "conservative"],
                        help="Speed adjustment mode")
    parser.add_argument("--status", action="store_true",
                        help="Show current profile and auto-speed status")
    args = parser.parse_args()

    # Handle auto-speed toggle
    if args.auto_speed:
        cfg = load_auto_speed()
        cfg["enabled"] = (args.auto_speed == "on")
        if args.mode:
            cfg["mode"] = args.mode
        save_auto_speed(cfg)
        state = "ENABLED" if cfg["enabled"] else "DISABLED"
        print(f"Auto-speed {state} (mode: {cfg['mode']})")
        return

    # Handle status
    if args.status:
        profile = load_profile()
        if profile:
            print_profile_summary(profile)
        else:
            print("No profile cached.")

        auto = load_auto_speed()
        print(f"\nAuto-speed: {'ENABLED' if auto.get('enabled') else 'DISABLED'}"
              f" (mode: {auto.get('mode', 'optimal')})")
        if auto.get("last_adjustment"):
            la = auto["last_adjustment"]
            print(f"Last adjustment: layer {la.get('layer')} → "
                  f"{la.get('speed')}% at {la.get('timestamp', '?')}")
        return

    # Determine filename
    filename = args.file
    if not filename:
        filename, state = get_current_filename()
        if not filename:
            print("Error: No file printing and no --file specified.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Current print: {filename} (state: {state})")

    # Check if profile already cached for this file
    existing = load_profile(filename)
    if existing:
        print(f"Profile already cached ({existing['total_layers']} layers, "
              f"generated {existing['generated']})")
        print("Re-analyzing...")

    # Fetch printer kinematics from config file
    print("Fetching printer kinematics (from printer.cfg)...")
    config = fetch_printer_kinematics()
    print(f"  max_velocity={config['max_velocity']}, "
          f"max_accel={config['max_accel']}, scv={config['scv']}")

    # Progress callback
    def on_progress(pct, lines, layer, elapsed):
        print(f"\r  Analyzing: {pct:.0f}% | {lines:,} lines | "
              f"layer {layer} | {elapsed:.0f}s elapsed", end="",
              file=sys.stderr)

    # Run analysis
    print(f"\nStreaming and analyzing: {filename}")
    print(f"This may take 30-90 seconds for large files...\n")

    profile = analyze_gcode(filename, config, on_progress)
    print(file=sys.stderr)  # newline after progress

    # Save profile
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved to {PROFILE_FILE}")

    # Print summary
    print_profile_summary(profile)


if __name__ == "__main__":
    main()
