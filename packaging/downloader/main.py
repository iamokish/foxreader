"""Main entry point for Fox Reader Downloader."""

import os
import sys

# Ensure this directory is in sys.path for local imports
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .ui import main
except ImportError:
    from ui import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
