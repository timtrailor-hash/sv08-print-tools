#!/usr/bin/env python3
"""
Streaming G-code analyser for SV08 Max Moonraker files.
Parses G0/G1 moves, calculates XY segment distances, and outputs
statistics broken down by OrcaSlicer feature type.
"""

import math
import re
import sys
import urllib.request
from collections import defaultdict

from config import MOONRAKER_URL as MOONRAKER

# --- helpers ---------------------------------------------------------------

def parse_gcode_params(line):
    """Extract letter-value pairs from a gcode line. Returns dict."""
    params = {}
    for m in re.finditer(r'([A-Z])([+-]?\d*\.?\d+)', line.split(';')[0]):
        params[m.group(1)] = float(m.group(2))
    return params


def bucket_label(length):
    if length < 1:
        return "0-1"
    elif length < 2:
        return "1-2"
    elif length < 5:
        return "2-5"
    elif length < 10:
        return "5-10"
    elif length < 20:
        return "10-20"
    elif length < 50:
        return "20-50"
    else:
        return "50+"


def size_class(length):
    if length < 5:
        return "short"
    elif length <= 20:
        return "medium"
    else:
        return "long"


def percentile(sorted_list, p):
    """Return the p-th percentile (0-100) from a pre-sorted list."""
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_list):
        return sorted_list[f]
    return sorted_list[f] + (k - f) * (sorted_list[c] - sorted_list[f])


def print_stats(label, lengths):
    """Print a full stats block for a list of segment lengths."""
    if not lengths:
        print(f"  (no moves)")
        return

    lengths_sorted = sorted(lengths)
    total = len(lengths_sorted)
    total_dist = sum(lengths_sorted)
    mean = total_dist / total
    median = percentile(lengths_sorted, 50)
    p10 = percentile(lengths_sorted, 10)
    p25 = percentile(lengths_sorted, 25)
    p75 = percentile(lengths_sorted, 75)
    p90 = percentile(lengths_sorted, 90)

    print(f"  Segments:  {total:>10,}")
    print(f"  Total XY:  {total_dist:>10,.1f} mm")
    print(f"  Mean:      {mean:>10.3f} mm")
    print(f"  Median:    {median:>10.3f} mm")
    print(f"  P10:       {p10:>10.3f} mm")
    print(f"  P25:       {p25:>10.3f} mm")
    print(f"  P75:       {p75:>10.3f} mm")
    print(f"  P90:       {p90:>10.3f} mm")
    print(f"  Min:       {lengths_sorted[0]:>10.3f} mm")
    print(f"  Max:       {lengths_sorted[-1]:>10.3f} mm")

    # histogram buckets
    buckets = defaultdict(int)
    for l in lengths_sorted:
        buckets[bucket_label(l)] += 1

    bucket_order = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50+"]
    max_bar = 40
    max_count = max(buckets.values()) if buckets else 1

    print(f"\n  Histogram (mm):")
    for b in bucket_order:
        count = buckets.get(b, 0)
        pct = count / total * 100
        bar = "#" * max(1, int(count / max_count * max_bar)) if count > 0 else ""
        print(f"    {b:>5} mm: {count:>9,}  ({pct:5.1f}%)  {bar}")

    # size class breakdown
    classes = defaultdict(int)
    for l in lengths_sorted:
        classes[size_class(l)] += 1
    print(f"\n  Short  (< 5 mm):   {classes['short']:>9,}  ({classes['short']/total*100:5.1f}%)")
    print(f"  Medium (5-20 mm):  {classes['medium']:>9,}  ({classes['medium']/total*100:5.1f}%)")
    print(f"  Long   (> 20 mm):  {classes['long']:>9,}  ({classes['long']/total*100:5.1f}%)")


# --- main ------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Streaming G-code segment analyser")
    parser.add_argument("filename",
                        help="Gcode filename on Moonraker (as shown in Fluidd/Mainsail)")
    args = parser.parse_args()

    FILENAME = args.filename
    url = f"{MOONRAKER}/server/files/gcodes/{urllib.request.quote(FILENAME)}"
    print(f"Streaming: {FILENAME}")
    print(f"URL: {url}\n")

    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)

    # Check file size from Content-Length header
    content_length = resp.headers.get("Content-Length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB\n")

    # State tracking
    cur_x = 0.0
    cur_y = 0.0
    cur_type = "Unknown"  # current OrcaSlicer feature type
    line_count = 0
    move_count = 0

    # Collect segment lengths per type
    all_lengths = []
    type_lengths = defaultdict(list)

    # OrcaSlicer uses "; TYPE:..." comments
    type_re = re.compile(r'^;\s*TYPE:\s*(.+)', re.IGNORECASE)

    for raw_line in resp:
        line = raw_line.decode('utf-8', errors='replace').strip()
        line_count += 1

        if line_count % 500_000 == 0:
            print(f"  ... processed {line_count:,} lines, {move_count:,} moves so far ...", file=sys.stderr)

        # Check for type comment
        tm = type_re.match(line)
        if tm:
            cur_type = tm.group(1).strip()
            continue

        # Only care about G0 / G1
        upper_start = line[:3].upper()
        if upper_start not in ("G0 ", "G1 ", "G0\t", "G1\t"):
            # Also check just "G0" or "G1" (no params)
            if line.upper().rstrip() not in ("G0", "G1"):
                continue

        params = parse_gcode_params(line.upper())

        new_x = params.get('X', cur_x)
        new_y = params.get('Y', cur_y)

        dx = new_x - cur_x
        dy = new_y - cur_y
        dist = math.sqrt(dx * dx + dy * dy)

        cur_x = new_x
        cur_y = new_y

        # Only count moves that actually move in XY
        if dist > 0.0001:
            move_count += 1
            all_lengths.append(dist)
            # Classify travel moves: G0 or G1 with no E parameter
            if line.upper().startswith("G0") or 'E' not in params:
                move_type = "Travel"
            else:
                move_type = cur_type
            type_lengths[move_type].append(dist)

    resp.close()

    # --- Output ---
    print()
    print("=" * 70)
    print(f"G-CODE ANALYSIS")
    print(f"File: {FILENAME}")
    print("=" * 70)
    print(f"Total lines parsed:     {line_count:>12,}")
    print(f"Total XY move segments: {move_count:>12,}")
    print()

    print("-" * 70)
    print("ALL MOVES (combined)")
    print("-" * 70)
    print_stats("All", all_lengths)
    print()

    # Per-type breakdown sorted by segment count descending
    print("=" * 70)
    print("BREAKDOWN BY FEATURE TYPE")
    print("=" * 70)
    for ftype, lengths in sorted(type_lengths.items(), key=lambda x: -len(x[1])):
        pct_of_total = len(lengths) / move_count * 100 if move_count else 0
        print(f"\n--- {ftype} ({len(lengths):,} moves, {pct_of_total:.1f}% of total) ---")
        print_stats(ftype, lengths)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
