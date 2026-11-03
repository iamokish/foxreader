import os
import sys

# Set before any fox_reader submodule imports torch: unsupported MPS operators
# can run on the CPU while the rest of the model stays on Metal. Honour an
# explicit user override (including "0" for diagnosing unsupported operators).
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.2.2"
