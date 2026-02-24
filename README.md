# SV08 Print Speed Optimisation Tools

Python tools that talk to [Moonraker](https://moonraker.readthedocs.io/) to analyse, optimise, and predict print times on Klipper-based printers. No printer mods needed.

Built for the Sovol SV08 Max but should work with **any Klipper printer running Moonraker** (Voron, Ender conversions, RatRig, etc.) -- the physics is the same.

## The Problem

Slicer ETA estimates assume the printer hits target speed on every move. In reality, Klipper's trapezoid kinematics mean the printer spends most of its time accelerating and decelerating on detailed prints. A 47-hour articulated snake print with the ETA drifting by hours is what prompted this.

## The Key Insight: Alpha

Every print move follows trapezoid kinematics: **accelerate -> cruise -> decelerate**. On short segments (detail, tight corners), the printer never reaches cruise speed -- it's almost entirely accel/decel.

**Alpha** captures this as a single number:

```
alpha = v^2 / (a * d_avg)
```

where `v` = target speed, `a` = acceleration, `d_avg` = average segment length.

| Alpha | Geometry | Example |
|-------|----------|---------|
| 0.02-0.08 | Simple | Cubes, vases, large flat surfaces |
| 0.08-0.20 | Moderate | Benchies, functional parts |
| 0.20-0.40 | Complex | Detailed figurines, architectural |
| 0.40-0.60 | Very complex | Dragons, articulated, organic |
| 0.60-1.0+ | Extreme | Miniatures, fine detail |

**The key discovery**: there's a point where pushing the speed slider higher actually makes prints *slower*. The optimal speed factor is `1/sqrt(alpha)`. My articulated snake at alpha=0.55 had an optimal speed of ~135% -- running it at 160% was actually *slower* than 135% because the acceleration overhead exceeded the cruise time savings.

## The Tools

### 1. `gcode_profile.py` -- Per-Layer Gcode Profiler

Streams gcode from Moonraker, analyses every layer's geometry using trapezoid move kinematics, and calculates the optimal speed for each layer.

```bash
# Analyse the currently printing file
python3 gcode_profile.py

# Analyse a specific file
python3 gcode_profile.py --file "my_print.gcode"

# Enable auto-speed (sends M220 per layer)
python3 gcode_profile.py --auto-speed on

# Show cached profile
python3 gcode_profile.py --status
```

**Auto-speed mode** sends M220 commands to adjust the speed factor per layer -- speeding up on simple layers and backing off on complex ones. Includes smooth ramping (max 30% change per layer) to prevent jarring transitions.

**What it tracks per layer:**
- Extrusion move geometry (segment count, distance, average feedrate)
- Travel moves with separate kinematics (higher junction velocity)
- Firmware retractions (G10/G11) with proper retract speed timing
- Z-hops with Z-axis trapezoid kinematics (500mm/s^2 accel, 10mm/s max)
- Per-segment move time accumulation (not averaged -- handles varied segment sizes correctly)
- Fixed overhead (retractions + Z-hops) separated from speed-dependent move time

### 2. `gcode_analyser.py` -- Segment Statistics

Detailed breakdown of move segments by OrcaSlicer feature type (inner wall, outer wall, infill, etc.). Shows histograms of segment lengths -- useful for understanding *why* a print is slow.

```bash
python3 gcode_analyser.py "my_print.gcode"
```

### 3. `print_eta.py` -- Physics-Based ETA Calculator

Smart ETA that accounts for speed factor vs geometry complexity. Self-calibrates alpha from live print data once >3% progress.

When a gcode profile is available, uses it as the **single source of truth** -- the profile knows every layer's geometry and self-corrects by comparing predicted vs actual elapsed time (time calibration factor). Only falls back to pace/slicer blending when no profile exists.

**Calibration approach:**
- Alpha calibration: scales raw gcode alphas to match the live-measured alpha
- Time calibration: compares predicted elapsed time to actual, separately tracking move time (scales with speed) and fixed overhead (doesn't scale)
- Per-layer remaining time uses calibrated data with the current speed factor

Also logs per-layer transitions (actual vs predicted time) and progress snapshots for future optimisation.

### 4. `eta_learner.py` -- Adaptive Weight Learning

Tracks which ETA method is most accurate at different stages and geometry complexities. After each completed print, scores all predictions against actual finish time and adjusts blend weights using inverse-MAE weighting.

```bash
# Show learning status
python3 eta_learner.py --status

# Show current blend weights
python3 eta_learner.py --weights

# Score a completed print (actual time in seconds)
python3 eta_learner.py --score "my_print.gcode" --actual-time 84600
```

## Setup

### Requirements

- Python 3.8+
- A Klipper printer running Moonraker (Fluidd, Mainsail, etc.)
- No pip dependencies -- uses only Python standard library

### Configuration

Edit `config.py` and set your Moonraker URL:

```python
MOONRAKER_URL = "http://192.168.1.100:7125"
```

Or set it as an environment variable:

```bash
export MOONRAKER_URL="http://192.168.1.100:7125"
```

That's it. The tools write temporary data to `/tmp/printer_status/` and learning data to `~/.printer_learning/` (override with `PRINTER_LEARN_DIR` env var).

### Quick Test

```bash
# Check your Moonraker is reachable
curl http://YOUR_PRINTER_IP:7125/printer/info

# Run the profiler on your current print
python3 gcode_profile.py
```

## How It Works

The time for a single move at speed factor S is:

```
T(S) = d/(v*S) + (v*S)/a
        cruise    accel overhead
```

The time ratio compared to 1x speed:

```
R(S) = (1/S + alpha*S) / (1 + alpha)
```

When `alpha*S > 1/S`, increasing speed makes things slower. The crossover point (optimal speed) is `S = 1/sqrt(alpha)`.

The profiler computes alpha per layer by:
1. Streaming gcode from Moonraker (handles 600MB+ files)
2. Parsing all G0/G1 moves with per-segment trapezoid time calculation
3. Tracking G10/G11 firmware retractions and Z-hop moves
4. Sampling junction angles (direction changes between consecutive moves)
5. Computing effective junction velocity from Klipper's square corner velocity
6. Calculating alpha from average segment length, feedrate, and acceleration
7. Separating fixed overhead (retractions, Z-hops) from speed-dependent move time

**Why fixed overhead matters:** On a 47-hour articulated snake, 266K retractions and 573K Z-hops account for ~18 hours of time that doesn't speed up when you increase the speed slider. Ignoring this causes the time calibration factor to compensate incorrectly, leading to ETA drift.

**Why per-segment timing matters:** Using `N * move_time(avg_segment)` underestimates because of Jensen's inequality -- the trapezoid function is concave, so `f(average) > average(f)`. Per-segment accumulation fixes this.

## Background

This started from a Facebook post in the Sovol SV08 MAX Community. The full writeup on the physics is in that post -- these are the tools that came out of it.

## License

MIT -- do whatever you want with it.
