#!/usr/bin/env python3
"""Adaptive ETA weight learner for SV08 Max.

Tracks which ETA method (pace, slicer, profile) is most accurate at
different stages of a print and for different geometry complexities.
Dynamically adjusts blend weights as we complete more prints.

Architecture:
  1. During a print: log_prediction() records each method's individual
     prediction alongside progress/alpha context.
  2. When a print completes: score_completed_job() compares all logged
     predictions against the actual finish time, computing per-method
     error at each snapshot.
  3. After scoring: learn_weights() analyzes the error data across all
     completed jobs and derives optimal blend weights per
     (progress_bucket, alpha_class) cell.
  4. During future prints: get_blend_weights() returns learned weights
     (or defaults if not enough data yet).

Data is bucketed by:
  - Progress: 0-20%, 20-50%, 50-80%, 80-100%
  - Complexity: simple (α<0.2), moderate (0.2-0.5), complex (0.5-1.0),
                extreme (α≥1.0)

Weight learning uses inverse-MAE weighting: methods with lower error
get higher weight. Requires 3+ scored snapshots per bucket before
overriding defaults.
"""

import json
import math
import os
from datetime import datetime

# Persistent storage (survives reboots, accumulates across prints)
LEARN_DIR = os.path.expanduser(
    "~/Documents/Claude code/printer_learning")
os.makedirs(LEARN_DIR, exist_ok=True)

ACCURACY_FILE = os.path.join(LEARN_DIR, "completed_accuracy.json")
WEIGHTS_FILE = os.path.join(LEARN_DIR, "learned_weights.json")

# Current print predictions (ephemeral, reset per print)
PREDICT_DIR = "/tmp/printer_status"
PREDICTIONS_FILE = os.path.join(PREDICT_DIR, "predictions.jsonl")
os.makedirs(PREDICT_DIR, exist_ok=True)

# ── Bucketing ─────────────────────────────────────────────────────

PROGRESS_BUCKETS = ["0-20", "20-50", "50-80", "80-100"]
ALPHA_CLASSES = ["simple", "moderate", "complex", "extreme"]
METHODS = ["pace", "slicer", "profile"]


def progress_bucket(pct):
    """Map progress percentage to bucket name."""
    if pct < 20:
        return "0-20"
    elif pct < 50:
        return "20-50"
    elif pct < 80:
        return "50-80"
    else:
        return "80-100"


def alpha_class(alpha):
    """Map alpha to complexity class."""
    if alpha < 0.2:
        return "simple"
    elif alpha < 0.5:
        return "moderate"
    elif alpha < 1.0:
        return "complex"
    else:
        return "extreme"


# ── Default weights (current hardcoded logic, as baseline) ────────

DEFAULT_WEIGHTS = {
    "0-20": {
        "simple":   {"pace": 0.05, "slicer": 0.75, "profile": 0.20},
        "moderate": {"pace": 0.05, "slicer": 0.70, "profile": 0.25},
        "complex":  {"pace": 0.05, "slicer": 0.65, "profile": 0.30},
        "extreme":  {"pace": 0.05, "slicer": 0.65, "profile": 0.30},
    },
    "20-50": {
        "simple":   {"pace": 0.50, "slicer": 0.25, "profile": 0.25},
        "moderate": {"pace": 0.45, "slicer": 0.20, "profile": 0.35},
        "complex":  {"pace": 0.40, "slicer": 0.20, "profile": 0.40},
        "extreme":  {"pace": 0.40, "slicer": 0.20, "profile": 0.40},
    },
    "50-80": {
        "simple":   {"pace": 0.65, "slicer": 0.15, "profile": 0.20},
        "moderate": {"pace": 0.55, "slicer": 0.15, "profile": 0.30},
        "complex":  {"pace": 0.35, "slicer": 0.10, "profile": 0.55},
        "extreme":  {"pace": 0.25, "slicer": 0.10, "profile": 0.65},
    },
    "80-100": {
        "simple":   {"pace": 0.80, "slicer": 0.10, "profile": 0.10},
        "moderate": {"pace": 0.75, "slicer": 0.10, "profile": 0.15},
        "complex":  {"pace": 0.40, "slicer": 0.10, "profile": 0.50},
        "extreme":  {"pace": 0.20, "slicer": 0.10, "profile": 0.70},
    },
}


# ── Prediction logging ────────────────────────────────────────────

def job_id(filename, start_time=None):
    """Generate a stable job ID from filename + approximate start time."""
    # Use filename + date as ID (one print per file per day)
    date = datetime.now().strftime("%Y-%m-%d")
    if start_time:
        date = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d")
    return f"{filename}:{date}"


def log_prediction(filename, progress_pct, alpha, elapsed_s,
                   pace_remaining_s, slicer_remaining_s,
                   profile_remaining_s, blended_remaining_s,
                   speed_factor=1.0, current_layer=0, start_time=None):
    """Log individual method predictions for later accuracy scoring.

    Called from print_eta.calculate_eta() on each status check.
    Only logs at meaningful intervals (~2% progress change) to avoid bloat.
    """
    jid = job_id(filename, start_time)

    # Rate limiting: only log every ~2% progress
    last = _last_logged_progress(jid)
    if last is not None and abs(progress_pct - last) < 1.5:
        return

    record = {
        "ts": datetime.now().isoformat(),
        "job_id": jid,
        "filename": filename,
        "progress_pct": round(progress_pct, 1),
        "alpha": round(alpha, 4) if alpha else None,
        "elapsed_s": round(elapsed_s),
        "speed_factor": round(speed_factor, 2),
        "current_layer": current_layer,
        "predictions": {
            "pace": round(pace_remaining_s) if pace_remaining_s else None,
            "slicer": round(slicer_remaining_s) if slicer_remaining_s else None,
            "profile": round(profile_remaining_s) if profile_remaining_s else None,
            "blended": round(blended_remaining_s) if blended_remaining_s else None,
        },
    }

    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


_last_progress_cache = {}


def _last_logged_progress(jid):
    """Check last logged progress for rate limiting."""
    global _last_progress_cache
    if jid in _last_progress_cache:
        return _last_progress_cache[jid]

    # Read last line of predictions file for this job
    if not os.path.exists(PREDICTIONS_FILE):
        return None
    try:
        last_pct = None
        with open(PREDICTIONS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("job_id") == jid:
                        last_pct = r.get("progress_pct")
                except json.JSONDecodeError:
                    continue
        _last_progress_cache[jid] = last_pct
        return last_pct
    except Exception:
        return None


# ── Completion scoring ────────────────────────────────────────────

def score_completed_job(filename, actual_total_s, start_time=None):
    """Score all logged predictions for a completed print.

    Computes relative error for each method at each logged snapshot:
        error = (predicted_remaining - actual_remaining) / actual_remaining

    Positive = predicted too slow (pessimistic), negative = too fast (optimistic).

    Returns the scored job record, or None if no predictions found.
    """
    jid = job_id(filename, start_time)

    # Load predictions for this job
    predictions = []
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("job_id") == jid:
                        predictions.append(r)
                except json.JSONDecodeError:
                    continue

    if not predictions:
        return None

    # Score each snapshot
    scored_snapshots = []
    avg_alpha = 0.0
    alpha_count = 0

    for p in predictions:
        progress = p.get("progress_pct", 0)
        elapsed = p.get("elapsed_s", 0)
        a = p.get("alpha")
        preds = p.get("predictions", {})

        if progress <= 0 or elapsed <= 0:
            continue

        actual_remaining = actual_total_s - elapsed
        if actual_remaining <= 0:
            continue

        if a and a > 0:
            avg_alpha += a
            alpha_count += 1

        errors = {}
        for method in METHODS + ["blended"]:
            pred_val = preds.get(method)
            if pred_val is not None and pred_val > 0:
                errors[method] = round(
                    (pred_val - actual_remaining) / actual_remaining, 4)

        scored_snapshots.append({
            "progress_pct": round(progress, 1),
            "progress_bucket": progress_bucket(progress),
            "alpha": round(a, 4) if a else None,
            "alpha_class": alpha_class(a) if a else "moderate",
            "elapsed_s": round(elapsed),
            "actual_remaining_s": round(actual_remaining),
            "errors": errors,
        })

    if not scored_snapshots:
        return None

    avg_a = avg_alpha / alpha_count if alpha_count > 0 else 0.4

    scored_job = {
        "job_id": jid,
        "filename": filename,
        "scored_at": datetime.now().isoformat(),
        "actual_total_s": round(actual_total_s),
        "alpha_avg": round(avg_a, 4),
        "alpha_class": alpha_class(avg_a),
        "snapshot_count": len(scored_snapshots),
        "snapshots": scored_snapshots,
    }

    # Append to persistent accuracy file
    accuracy = _load_accuracy()
    # Remove any existing entry for this job (re-scoring)
    accuracy = [j for j in accuracy if j.get("job_id") != jid]
    accuracy.append(scored_job)
    _save_accuracy(accuracy)

    # Re-learn weights with the new data
    learn_weights()

    return scored_job


def _load_accuracy():
    """Load completed job accuracy data."""
    if not os.path.exists(ACCURACY_FILE):
        return []
    try:
        with open(ACCURACY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_accuracy(data):
    """Save completed job accuracy data."""
    with open(ACCURACY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Weight learning ──────────────────────────────────────────────

def learn_weights(min_samples=3):
    """Learn optimal blend weights from completed job accuracy data.

    For each (progress_bucket, alpha_class) cell:
    1. Collect all scored snapshots
    2. Compute mean absolute error (MAE) per method
    3. Derive weights as inverse-MAE (lower error → higher weight)
    4. Normalize to sum to 1.0

    Only overrides defaults when we have min_samples scored snapshots
    for a given cell.
    """
    accuracy = _load_accuracy()
    if not accuracy:
        return

    # Collect all snapshots into buckets
    # Key: (progress_bucket, alpha_class)
    # Value: list of error dicts
    buckets = {}
    for job in accuracy:
        for snap in job.get("snapshots", []):
            pb = snap.get("progress_bucket", "20-50")
            ac = snap.get("alpha_class", "moderate")
            key = (pb, ac)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(snap.get("errors", {}))

    # Learn weights per bucket
    learned = {}
    total_cells = 0
    learned_cells = 0

    for pb in PROGRESS_BUCKETS:
        learned[pb] = {}
        for ac in ALPHA_CLASSES:
            total_cells += 1
            key = (pb, ac)
            errors_list = buckets.get(key, [])

            if len(errors_list) < min_samples:
                # Not enough data — use defaults
                learned[pb][ac] = DEFAULT_WEIGHTS[pb][ac].copy()
                continue

            learned_cells += 1

            # Compute MAE per method
            mae = {}
            for method in METHODS:
                abs_errors = []
                for e in errors_list:
                    if method in e and e[method] is not None:
                        abs_errors.append(abs(e[method]))
                if abs_errors:
                    mae[method] = sum(abs_errors) / len(abs_errors)
                else:
                    mae[method] = 1.0  # high default = low weight

            # Inverse-MAE weighting
            # Add small epsilon to avoid division by zero
            eps = 0.01
            inv_mae = {m: 1.0 / (mae[m] + eps) for m in METHODS}
            total_inv = sum(inv_mae.values())

            weights = {m: round(inv_mae[m] / total_inv, 3)
                       for m in METHODS}

            # Sanity check: no method gets less than 5% weight
            for m in METHODS:
                if weights[m] < 0.05:
                    weights[m] = 0.05
            # Re-normalize
            total_w = sum(weights.values())
            weights = {m: round(weights[m] / total_w, 3) for m in METHODS}

            learned[pb][ac] = weights

    # Save learned weights
    result = {
        "version": 1,
        "updated": datetime.now().isoformat(),
        "jobs_analyzed": len(accuracy),
        "total_snapshots": sum(len(j.get("snapshots", []))
                               for j in accuracy),
        "cells_learned": learned_cells,
        "cells_total": total_cells,
        "weights": learned,
        "default_weights": DEFAULT_WEIGHTS,
    }

    with open(WEIGHTS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Weight retrieval (called by print_eta.py) ────────────────────

def get_blend_weights(progress_pct, alpha_val):
    """Get blend weights for a given progress and alpha.

    Returns dict: {"pace": w1, "slicer": w2, "profile": w3}
    where w1+w2+w3 ≈ 1.0.

    Uses learned weights if available, otherwise defaults.
    """
    pb = progress_bucket(progress_pct)
    ac = alpha_class(alpha_val) if alpha_val else "moderate"

    # Try learned weights first
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                data = json.load(f)
            weights = data.get("weights", {})
            if pb in weights and ac in weights[pb]:
                return weights[pb][ac]
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to defaults
    return DEFAULT_WEIGHTS.get(pb, {}).get(ac, {
        "pace": 0.4, "slicer": 0.3, "profile": 0.3
    })


def get_learning_stats():
    """Get summary statistics about the learning system.

    Returns dict with job count, snapshot count, cells learned, etc.
    """
    stats = {
        "jobs_scored": 0,
        "total_snapshots": 0,
        "cells_learned": 0,
        "cells_total": len(PROGRESS_BUCKETS) * len(ALPHA_CLASSES),
        "predictions_logged": 0,
    }

    if os.path.exists(ACCURACY_FILE):
        try:
            with open(ACCURACY_FILE) as f:
                accuracy = json.load(f)
            stats["jobs_scored"] = len(accuracy)
            stats["total_snapshots"] = sum(
                len(j.get("snapshots", [])) for j in accuracy)
        except Exception:
            pass

    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                data = json.load(f)
            stats["cells_learned"] = data.get("cells_learned", 0)
        except Exception:
            pass

    if os.path.exists(PREDICTIONS_FILE):
        try:
            count = 0
            with open(PREDICTIONS_FILE) as f:
                for _ in f:
                    count += 1
            stats["predictions_logged"] = count
        except Exception:
            pass

    return stats


# ── CLI ───────────────────────────────────────────────────────────

def main():
    """Show learning system status."""
    import argparse
    parser = argparse.ArgumentParser(
        description="ETA weight learning system")
    parser.add_argument("--status", action="store_true",
                        help="Show learning system status")
    parser.add_argument("--weights", action="store_true",
                        help="Show current blend weights")
    parser.add_argument("--score", metavar="FILENAME",
                        help="Score a completed print's predictions")
    parser.add_argument("--actual-time", type=float,
                        help="Actual print time in seconds (for --score)")
    args = parser.parse_args()

    if args.score:
        if not args.actual_time:
            print("Error: --actual-time required with --score",
                  file=__import__('sys').stderr)
            __import__('sys').exit(1)
        result = score_completed_job(args.score, args.actual_time)
        if result:
            print(f"Scored {result['snapshot_count']} snapshots for "
                  f"{result['filename']}")
            print(f"Alpha class: {result['alpha_class']} "
                  f"(avg α={result['alpha_avg']:.3f})")

            # Show error summary per method
            for method in METHODS + ["blended"]:
                errs = [s["errors"].get(method)
                        for s in result["snapshots"]
                        if method in s.get("errors", {})]
                if errs:
                    errs = [e for e in errs if e is not None]
                    mae = sum(abs(e) for e in errs) / len(errs)
                    bias = sum(errs) / len(errs)
                    print(f"  {method:>8}: MAE={mae:.1%}, "
                          f"bias={bias:+.1%} "
                          f"({'pessimistic' if bias > 0 else 'optimistic'})")
        else:
            print("No predictions found for this file.")
        return

    if args.weights:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE) as f:
                data = json.load(f)
            print(f"\nLearned weights (updated {data.get('updated', '?')})")
            print(f"Jobs analyzed: {data.get('jobs_analyzed', 0)}")
            print(f"Cells learned: {data.get('cells_learned', 0)}"
                  f"/{data.get('cells_total', 16)}\n")

            weights = data.get("weights", {})
            for pb in PROGRESS_BUCKETS:
                print(f"  Progress {pb}%:")
                for ac in ALPHA_CLASSES:
                    w = weights.get(pb, {}).get(ac, {})
                    src = "learned" if data.get("cells_learned", 0) > 0 else "default"
                    print(f"    {ac:>10}: pace={w.get('pace', 0):.0%} "
                          f"slicer={w.get('slicer', 0):.0%} "
                          f"profile={w.get('profile', 0):.0%}")
                print()
        else:
            print("No learned weights yet — using defaults.")
            print("Weights will be learned after first completed print.\n")
            for pb in PROGRESS_BUCKETS:
                print(f"  Progress {pb}% (default):")
                for ac in ALPHA_CLASSES:
                    w = DEFAULT_WEIGHTS[pb][ac]
                    print(f"    {ac:>10}: pace={w['pace']:.0%} "
                          f"slicer={w['slicer']:.0%} "
                          f"profile={w['profile']:.0%}")
                print()
        return

    # Default: show status
    stats = get_learning_stats()
    print(f"\n{'=' * 50}")
    print("ETA LEARNING SYSTEM STATUS")
    print(f"{'=' * 50}")
    print(f"Predictions logged (current):  {stats['predictions_logged']}")
    print(f"Jobs scored (completed):       {stats['jobs_scored']}")
    print(f"Total scored snapshots:        {stats['total_snapshots']}")
    print(f"Weight cells learned:          {stats['cells_learned']}"
          f"/{stats['cells_total']}")

    if stats['jobs_scored'] > 0:
        print(f"\nRun --weights to see current blend weights.")
    else:
        print(f"\nNo prints completed yet. Using default weights.")
        print("Weights will be learned automatically after prints finish.")


if __name__ == "__main__":
    main()
