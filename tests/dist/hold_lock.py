"""Hold .fox-reader.lock the way launcher.c does, then let go.

Stands in for a peer launcher that took the lock and then died without its
backend ever answering -- the case the peer-wait loop used to wait on forever.

A helper, not a check: it takes the lock path and a duration on argv and holds
the file open the way the launcher does, so something else can be observed
waiting on it.

    python tests/dist/hold_lock.py <path-to-.fox-reader.lock> <seconds>
"""
import ctypes
import sys
import time
from ctypes import wintypes

if len(sys.argv) != 3:
    print(__doc__.strip().splitlines()[-1].strip())
    raise SystemExit(2)

path, seconds = sys.argv[1], float(sys.argv[2])

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
DELETE = 0x00010000
OPEN_ALWAYS = 4
FILE_ATTRIBUTE_HIDDEN = 2
FILE_FLAG_DELETE_ON_CLOSE = 0x04000000
INVALID = ctypes.c_void_p(-1).value

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
]

handle = kernel32.CreateFileW(
    path,
    GENERIC_READ | GENERIC_WRITE | DELETE,
    0,  # no sharing: the open itself is the lock
    None,
    OPEN_ALWAYS,
    FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE,
    None,
)
if handle == INVALID:
    print(f"could not take the lock: error {ctypes.get_last_error()}", flush=True)
    raise SystemExit(1)

print(f"holding {path}", flush=True)
time.sleep(seconds)
kernel32.CloseHandle(wintypes.HANDLE(handle))
print("released", flush=True)
