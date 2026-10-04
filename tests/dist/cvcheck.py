"""The OpenCV in this environment is the one the build ships, with the parts the app needs.

Two separate mistakes this catches, both silent:

  * the wrong distribution. `opencv-python` and `opencv-contrib-python` install the
    same `cv2` module, so `import cv2` proves nothing about which one is here. Only
    the contrib build carries `cv2.xphoto`, and `packaging/build.py` names
    `opencv-contrib-python` in DISTRIBUTION_METADATA -- naming a distribution that
    was never installed fails the build, and shipping the plain one instead loses
    four fill methods at runtime with no error anywhere.

  * a contrib module that imports and does nothing. OpenCV 5.0 ships an `xphoto`
    that is importable, truthy, and empty, which is why `fox_reader.clean.caps`
    asks about the exact attribute it is about to call rather than about the module.
    The same question, asked here about the whole environment.
"""
import importlib.metadata as im
import sys

WANT = "opencv-contrib-python"
#: Same `cv2`, and either would satisfy an `import cv2` while losing xphoto.
CONFLICTS = ("opencv-python", "opencv-python-headless")

#: What `fox_reader.clean.caps` actually calls. Keep in step with its `_FLAGS`.
XPHOTO_FLAGS = ("INPAINT_SHIFTMAP", "INPAINT_FSR_BEST", "INPAINT_FSR_FAST")

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print(f"python: {sys.executable}\n")

installed = {}
for name in (WANT, f"{WANT}-headless", *CONFLICTS):
    try:
        installed[name] = im.version(name)
    except Exception:
        installed[name] = None
    print(f"  dist {name:34} = {installed[name] or '(not installed)'}")

print("\n  --- who requires opencv ---")
for dist in im.distributions():
    for req in (dist.requires or []):
        if "opencv" in req.lower():
            try:
                who = dist.metadata["Name"]
            except Exception:
                who = "?"
            print(f"    {who} -> {req}")

print()
check(f"{WANT} is installed", installed[WANT] is not None,
      "the build names this one in DISTRIBUTION_METADATA")
clashing = [name for name in CONFLICTS if installed[name] is not None]
check("no non-contrib opencv alongside it", not clashing,
      f"{', '.join(clashing)} installs the same cv2 without xphoto" if clashing else "")

try:
    import cv2
except Exception as exc:
    check("cv2 imports", False, repr(exc))
    print(f"\nFAILURES: {', '.join(failures)}")
    sys.exit(1)

check("cv2 imports", True, cv2.__version__)

xphoto = getattr(cv2, "xphoto", None)
check("cv2.xphoto exists", xphoto is not None)
check("cv2.xphoto.inpaint is callable", callable(getattr(xphoto, "inpaint", None)),
      "an importable but empty xphoto is what OpenCV 5.0 ships without contrib")
for flag in XPHOTO_FLAGS:
    check(f"cv2.xphoto.{flag}", getattr(xphoto, flag, None) is not None)

for module in ("ximgproc", "aruco", "bgsegm", "face", "tracking"):
    print(f"  (info) contrib.{module:10}: {hasattr(cv2, module)}")

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
