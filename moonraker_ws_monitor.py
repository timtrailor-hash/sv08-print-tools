#!/usr/bin/env python3
"""Monitor Moonraker websocket stability and log drops.

Connects via websocket to Moonraker and logs every disconnect with
timestamp, duration of connection, and any error details. Also polls
the HTTP API every 5s as a secondary check.

Usage:
    python3 moonraker_ws_monitor.py          # run until Ctrl+C
    python3 moonraker_ws_monitor.py --hours 4  # run for 4 hours

Logs to: /tmp/printer_status/ws_monitor.log
"""

import argparse
import json
import os
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.request
from datetime import datetime

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from printer_config import SOVOL_IP, MOONRAKER_PORT
except ImportError:
    SOVOL_IP = "192.168.87.52"
    MOONRAKER_PORT = 7125

LOG_DIR = "/tmp/printer_status"
LOG_FILE = os.path.join(LOG_DIR, "ws_monitor.log")
os.makedirs(LOG_DIR, exist_ok=True)

_running = True


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def http_check():
    """Quick HTTP API check. Returns (ok, latency_ms, error)."""
    start = time.time()
    try:
        url = f"http://{SOVOL_IP}:{MOONRAKER_PORT}/printer/info"
        with urllib.request.urlopen(url, timeout=5) as r:
            r.read()
        return True, (time.time() - start) * 1000, None
    except Exception as e:
        return False, (time.time() - start) * 1000, str(e)


def ws_connect():
    """Connect a raw websocket to Moonraker. Returns the socket or None."""
    try:
        sock = socket.create_connection((SOVOL_IP, MOONRAKER_PORT), timeout=10)
        # Send HTTP upgrade request
        key = "dGVzdGtleQ=="  # base64 of "testkey"
        request = (
            f"GET /websocket HTTP/1.1\r\n"
            f"Host: {SOVOL_IP}:{MOONRAKER_PORT}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        sock.sendall(request.encode())
        # Read response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                return None
            response += chunk
        if b"101" not in response.split(b"\r\n")[0]:
            first_line = response.split(b"\r\n")[0]
            log(f"WS upgrade failed: {first_line}")
            sock.close()
            return None
        sock.settimeout(30)  # timeout for reads (expect pings within this)
        return sock
    except Exception as e:
        log(f"WS connect failed: {e}")
        return None


def ws_read_frame(sock):
    """Read one websocket frame. Returns (opcode, payload) or (None, None) on error."""
    try:
        header = sock.recv(2)
        if len(header) < 2:
            return None, None
        opcode = header[0] & 0x0F
        masked = header[1] & 0x80
        length = header[1] & 0x7F
        if length == 126:
            ext = sock.recv(2)
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = sock.recv(8)
            length = struct.unpack(">Q", ext)[0]
        if masked:
            mask = sock.recv(4)
        payload = b""
        while len(payload) < length:
            chunk = sock.recv(length - len(payload))
            if not chunk:
                return None, None
            payload += chunk
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload
    except socket.timeout:
        return -1, None  # timeout, not an error
    except Exception:
        return None, None


def ws_send_pong(sock, payload):
    """Send a pong frame."""
    try:
        frame = bytes([0x8A, len(payload)]) + payload
        sock.sendall(frame)
    except Exception:
        pass


def monitor_ws():
    """Main websocket monitoring loop."""
    connection_num = 0
    while _running:
        connection_num += 1
        connect_time = time.time()
        log(f"WS #{connection_num}: connecting to {SOVOL_IP}:{MOONRAKER_PORT}")

        sock = ws_connect()
        if not sock:
            log(f"WS #{connection_num}: connection failed, retrying in 5s")
            time.sleep(5)
            continue

        log(f"WS #{connection_num}: connected")
        frames_received = 0

        while _running:
            opcode, payload = ws_read_frame(sock)
            if opcode is None:
                # Connection dropped
                duration = time.time() - connect_time
                log(f"WS #{connection_num}: DROPPED after {duration:.1f}s "
                    f"({frames_received} frames received)")
                break
            elif opcode == -1:
                # Timeout — send our own ping to check
                try:
                    ping = bytes([0x89, 0])
                    sock.sendall(ping)
                except Exception:
                    duration = time.time() - connect_time
                    log(f"WS #{connection_num}: DROPPED (ping send failed) "
                        f"after {duration:.1f}s")
                    break
            elif opcode == 0x9:
                # Ping from server — send pong
                ws_send_pong(sock, payload or b"")
            elif opcode == 0x8:
                # Close frame
                duration = time.time() - connect_time
                log(f"WS #{connection_num}: SERVER CLOSED after {duration:.1f}s "
                    f"(code: {payload[:2] if payload else 'none'})")
                break
            else:
                frames_received += 1

        try:
            sock.close()
        except Exception:
            pass

        if _running:
            log(f"WS #{connection_num}: reconnecting in 2s")
            time.sleep(2)


def monitor_http():
    """Secondary HTTP polling monitor — logs drops and latency."""
    consecutive_fails = 0
    last_fail_time = None

    while _running:
        ok, latency, error = http_check()
        if not ok:
            consecutive_fails += 1
            if consecutive_fails == 1:
                last_fail_time = time.time()
            log(f"HTTP: FAIL #{consecutive_fails} ({latency:.0f}ms) - {error}")
        else:
            if consecutive_fails > 0:
                outage = time.time() - last_fail_time
                log(f"HTTP: recovered after {consecutive_fails} fails "
                    f"({outage:.1f}s outage), latency={latency:.0f}ms")
                consecutive_fails = 0
            elif latency > 1000:
                log(f"HTTP: slow response {latency:.0f}ms")

        time.sleep(5)


def main():
    global _running
    parser = argparse.ArgumentParser(description="Monitor Moonraker websocket stability")
    parser.add_argument("--hours", type=float, default=0,
                        help="Run for N hours (0 = indefinite)")
    args = parser.parse_args()

    log(f"=== Moonraker WS Monitor started ===")
    log(f"Target: {SOVOL_IP}:{MOONRAKER_PORT}")
    log(f"Log file: {LOG_FILE}")
    if args.hours:
        log(f"Will run for {args.hours} hours")

    # Start HTTP monitor in background
    http_thread = threading.Thread(target=monitor_http, daemon=True)
    http_thread.start()

    # Start WS monitor in main thread
    try:
        if args.hours:
            deadline = time.time() + args.hours * 3600
            ws_thread = threading.Thread(target=monitor_ws, daemon=True)
            ws_thread.start()
            while time.time() < deadline and _running:
                time.sleep(1)
            _running = False
        else:
            monitor_ws()
    except KeyboardInterrupt:
        _running = False
        log("Monitor stopped by user")

    log(f"=== Monitor ended ===")


if __name__ == "__main__":
    main()
