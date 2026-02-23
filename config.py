"""Shared configuration for SV08 print tools.

Set MOONRAKER_URL as an environment variable, or edit the default below.
"""

import os

MOONRAKER_URL = os.environ.get(
    "MOONRAKER_URL", "http://YOUR_PRINTER_IP:7125")
