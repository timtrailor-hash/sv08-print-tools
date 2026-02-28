#!/usr/bin/env python3
"""Generate a modified gcode file to resume a print from a specific layer.

Streams the original gcode from Moonraker, parses layer boundaries,
captures printer state at the target layer, and generates a new file
with startup preamble + state restoration + remaining gcode.

Usage:
    python3 gcode_resume.py --file "model.gcode" --layer 52
    python3 gcode_resume.py --file "model.gcode" --layer auto  # uses last checkpoint
    python3 gcode_resume.py --file "model.gcode" --layer 52 --start  # upload + start print

Requires: Moonraker accessible on the network.
"""

import argparse
import configparser
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from printer_config import SOVOL_IP, MOONRAKER_PORT
except ImportError:
    SOVOL_IP = "[REDACTED — see printer_config.example.py]"
    MOONRAKER_PORT = 7125

MOONRAKER = f"http://{SOVOL_IP}:{MOONRAKER_PORT}"
CHECKPOINT_FILE = "/tmp/printer_status/print_checkpoint.json"


def api_get(path, timeout=10):
    url = f"{MOONRAKER}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_checkpoint_layer():
    """Read last checkpoint and return suggested resume layer (2 back for adhesion)."""
    try:
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        last_layer = cp.get("current_layer", 0)
        safe_layer = max(1, last_layer - 2)
        print(f"Checkpoint: layer {last_layer}, Z={cp.get('z_position')}mm, "
              f"{cp.get('progress_pct', 0):.1f}%")
        print(f"Resuming from layer {safe_layer} (2 layers back for adhesion)")
        return safe_layer
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Cannot read checkpoint: {e}")
        sys.exit(1)


def stream_gcode(filename):
    """Stream gcode file from Moonraker, yielding decoded lines."""
    encoded = urllib.parse.quote(filename, safe="")
    url = f"{MOONRAKER}/server/files/gcodes/{encoded}"
    print(f"Streaming gcode from Moonraker: {filename}")
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=30)
    total = int(resp.headers.get("Content-Length", 0))
    if total:
        print(f"File size: {total / 1024 / 1024:.1f} MB")
    return resp, total


def parse_and_split(resp, total_bytes, target_layer):
    """Parse gcode stream, capturing state at target layer and splitting output.

    Returns (preamble_state, remaining_lines_buffer) where preamble_state is a
    dict with temperatures, fan, accel, feedrate, E position, Z height.
    """
    cur_layer = 0
    layer_z = {}
    bytes_read = 0
    last_progress = 0

    # State tracking
    bed_temp = 60
    nozzle_temp = 220
    fan_speed = 0       # S value for M106
    last_feedrate = 0
    last_accel = None
    last_e = 0.0
    target_z = 0.0

    # Header temp scan (first 500 lines)
    header_bed = None
    header_nozzle = None
    line_num = 0

    # Capture EXCLUDE_OBJECT_DEFINE lines from header
    object_defines = []

    # Collect remaining gcode after target layer
    remaining = io.BytesIO()
    found_target = False
    target_state = None

    start = time.time()

    for raw_line in resp:
        bytes_read += len(raw_line)
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
        line_num += 1

        # Progress reporting
        if total_bytes and time.time() - start > 2:
            pct = bytes_read / total_bytes * 100
            if pct - last_progress > 5:
                print(f"  Parsing: {pct:.0f}% (layer {cur_layer})")
                last_progress = pct
                start = time.time()

        # Capture EXCLUDE_OBJECT_DEFINE from anywhere before target layer
        if not found_target and line.startswith("EXCLUDE_OBJECT_DEFINE"):
            object_defines.append(line)

        # Header temperature scan
        if line_num <= 500:
            if line.startswith("M190") or line.startswith("M140"):
                for tok in line.split():
                    if tok.startswith("S"):
                        try:
                            header_bed = float(tok[1:])
                        except ValueError:
                            pass
            elif line.startswith("M109") or line.startswith("M104"):
                for tok in line.split():
                    if tok.startswith("S"):
                        try:
                            header_nozzle = float(tok[1:])
                        except ValueError:
                            pass

        # Once we've found the target, just write remaining lines
        if found_target:
            remaining.write(raw_line)
            continue

        # Layer detection
        if line.startswith(";"):
            stripped = line[1:].lstrip()
            if stripped.startswith("LAYER_CHANGE"):
                cur_layer += 1
                if cur_layer == target_layer:
                    # Capture state at this layer boundary
                    target_state = {
                        "layer": cur_layer,
                        "z": layer_z.get(cur_layer, target_z),
                        "e": last_e,
                        "feedrate": last_feedrate,
                        "fan_speed": fan_speed,
                        "accel": last_accel,
                        "bed_temp": header_bed or bed_temp,
                        "nozzle_temp": header_nozzle or nozzle_temp,
                    }
                    found_target = True
                    # Include the layer change comment itself
                    remaining.write(raw_line)
                    print(f"\n  Target layer {target_layer} found at Z={target_state['z']:.2f}mm")
                    print(f"  E={last_e:.3f}, F={last_feedrate}, fan={fan_speed}")
                    continue
            elif stripped.startswith("Z:"):
                try:
                    z_val = float(stripped[2:].strip())
                    layer_z[cur_layer] = z_val
                    target_z = z_val
                except ValueError:
                    pass
            continue

        # Track state for pre-target layers
        if line.startswith("M106"):
            for tok in line.split():
                if tok.startswith("S"):
                    try:
                        fan_speed = int(float(tok[1:]))
                    except ValueError:
                        pass
        elif line.startswith("M107"):
            fan_speed = 0
        elif line.startswith("SET_VELOCITY_LIMIT") and "ACCEL=" in line:
            for tok in line.split():
                if tok.startswith("ACCEL="):
                    try:
                        last_accel = float(tok[6:])
                    except ValueError:
                        pass

        # Track E and F from G0/G1 moves
        if len(line) >= 2 and line[0] in ("G", "g") and line[1] in ("0", "1"):
            for tok in line.split():
                if tok.startswith("E") or tok.startswith("e"):
                    try:
                        last_e = float(tok[1:])
                    except ValueError:
                        pass
                elif tok.startswith("F") or tok.startswith("f"):
                    try:
                        last_feedrate = float(tok[1:])
                    except ValueError:
                        pass

    if not found_target:
        print(f"ERROR: Layer {target_layer} not found! File has {cur_layer} layers.")
        sys.exit(1)

    return target_state, remaining, cur_layer, object_defines


def generate_resume_gcode(state, remaining_buf, total_layers=0, object_defines=None):
    """Build the complete resume gcode file."""
    z = state["z"]
    e = state["e"]
    bed = int(state["bed_temp"])
    nozzle = int(state["nozzle_temp"])
    fan = state["fan_speed"]
    accel = state["accel"]
    feedrate = state["feedrate"]
    layer = state["layer"]

    lines = []
    lines.append(f"; RESUME FROM LAYER {layer} (Z={z:.2f}mm)")
    lines.append(f"; Generated by gcode_resume.py at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"; Original E={e:.3f} F={feedrate} fan={fan}")
    lines.append("")

    # Startup preamble — QGL yes (probes corners), NO bed mesh (would hit print)
    lines.append("; === STARTUP PREAMBLE ===")
    lines.append(f"M140 S{bed}         ; heat bed")
    lines.append(f"M104 S{nozzle}      ; heat nozzle")
    lines.append(f"M190 S{bed}         ; wait for bed")
    lines.append(f"M109 S{nozzle}      ; wait for nozzle")
    lines.append("G28                  ; home all axes")
    lines.append("QUAD_GANTRY_LEVEL    ; level gantry (probes corners only)")
    lines.append("G28 Z                ; re-home Z after QGL")
    lines.append("; NOTE: BED_MESH skipped — partial print on bed")
    lines.append("")

    # Position setup
    lines.append("; === POSITION SETUP ===")
    lines.append("G90                  ; absolute positioning")
    lines.append("")

    # Safe approach — use RELATIVE extrusion for purge
    lines.append("; === SAFE APPROACH ===")
    lines.append(f"G1 Z{z + 5:.2f} F600      ; move above print")
    lines.append(f"G1 X0 Y0 F6000            ; move to bed edge for purge")
    lines.append("M83                        ; relative extruder for purge")
    lines.append("G1 E15 F150                ; purge 15mm")
    lines.append("G1 E-0.8 F2400             ; retract")
    lines.append("")

    # Now set absolute extruder to match where the slicer expects it
    lines.append("; === STATE RESTORATION ===")
    lines.append("M82                        ; absolute extruder")
    lines.append(f"G92 E{e:.4f}              ; sync E to slicer state")
    if fan > 0:
        lines.append(f"M106 S{fan}          ; restore fan speed")
    else:
        lines.append("M107                 ; fan off")
    if accel:
        lines.append(f"SET_VELOCITY_LIMIT ACCEL={accel:.0f}  ; restore accel")
    lines.append("")

    # Tell Klipper total layer count so it can track progress
    remaining_layers = total_layers - layer if total_layers else 0
    if remaining_layers > 0:
        lines.append(f"SET_PRINT_STATS_INFO TOTAL_LAYER_COUNT={remaining_layers}")

    # Object definitions for EXCLUDE_OBJECT support (skip objects in iOS app)
    if object_defines:
        lines.append("")
        lines.append("; === OBJECT DEFINITIONS (for skip support) ===")
        for od in object_defines:
            lines.append(od)

    lines.append("")
    # Remaining gcode from slicer
    lines.append("; === BEGIN REMAINING GCODE ===")
    lines.append("")

    header = "\n".join(lines)

    # Get remaining gcode bytes
    remaining_buf.seek(0)
    remaining_data = remaining_buf.read()

    return header.encode("utf-8") + b"\n" + remaining_data


def upload_to_moonraker(filename, data):
    """Upload gcode file to Moonraker via multipart POST."""
    boundary = "----ResumeUploadBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    body += data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    url = f"{MOONRAKER}/server/files/upload"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result


def start_print(filename):
    """Start printing a file via Moonraker."""
    data = json.dumps({"filename": filename}).encode()
    url = f"{MOONRAKER}/printer/print/start"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Resume a 3D print from a specific layer")
    parser.add_argument("--file", required=True, help="Gcode filename on the printer")
    parser.add_argument("--layer", required=True, help="Layer number to resume from, or 'auto'")
    parser.add_argument("--start", action="store_true", help="Auto-start the print after upload")
    parser.add_argument("--output", help="Save locally instead of uploading (for testing)")
    args = parser.parse_args()

    # Resolve target layer
    if args.layer == "auto":
        target_layer = get_checkpoint_layer()
    else:
        target_layer = int(args.layer)
        if target_layer < 1:
            print("ERROR: Layer must be >= 1")
            sys.exit(1)

    print(f"\nResume plan: {args.file} from layer {target_layer}")
    print("=" * 60)

    # Stream and parse
    resp, total = stream_gcode(args.file)
    state, remaining, file_total_layers, object_defines = parse_and_split(resp, total, target_layer)

    print(f"\nState at layer {target_layer}:")
    print(f"  Z = {state['z']:.2f} mm")
    print(f"  E = {state['e']:.3f} mm")
    print(f"  Bed = {state['bed_temp']}\u00b0C, Nozzle = {state['nozzle_temp']}\u00b0C")
    print(f"  Fan = {state['fan_speed']}, Accel = {state['accel']}")

    # Generate resume file
    print("\nGenerating resume gcode...")
    resume_data = generate_resume_gcode(state, remaining, total_layers=file_total_layers,
                                        object_defines=object_defines)
    resume_size = len(resume_data) / 1024 / 1024
    print(f"Resume file size: {resume_size:.1f} MB")

    # Build filename
    base = os.path.splitext(args.file)[0]
    resume_name = f"RESUME_L{target_layer}_{base}.gcode"

    if args.output:
        with open(args.output, "wb") as f:
            f.write(resume_data)
        print(f"\nSaved locally: {args.output}")
    else:
        print(f"\nUploading to Moonraker: {resume_name}")
        result = upload_to_moonraker(resume_name, resume_data)
        print(f"Upload result: {result}")

        if args.start:
            print(f"\nStarting print: {resume_name}")
            result = start_print(resume_name)
            print(f"Start result: {result}")

    # Output summary as JSON for subprocess callers
    summary = {
        "ok": True,
        "resume_file": resume_name,
        "resume_layer": target_layer,
        "resume_z": state["z"],
        "skipped_layers": target_layer,
        "resume_size_mb": round(resume_size, 1),
        "bed_temp": state["bed_temp"],
        "nozzle_temp": state["nozzle_temp"],
    }
    print(f"\n__SUMMARY_JSON__:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
