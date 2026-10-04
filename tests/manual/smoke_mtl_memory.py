"""Exercise the MTL memory gate without torch, psutil or a downloaded model.

Nothing in this repo's scratch environment has torch, transformers
or sentencepiece installed, and none of them are needed to check the arithmetic
and the decisions around it. Stubs go into sys.modules before local_mtl is
imported, exactly as smoke_device.py does for the device probe, and each
scenario resets the module-level caches so the next one sees fresh hardware.

What is being checked: the requirement lookup, which pool each device kind is
measured against, that a shortfall on CUDA refuses instead of quietly moving to
the CPU, that an unmeasurable machine is allowed through, and that the manager
refuses before it constructs anything.
"""

from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# The refusal messages contain an arrow, which a cp1252 console cannot print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MIB = 1024 * 1024
GIB = 1024**3

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Report one assertion. The detail is only worth printing when it failed."""
    global failures

    if condition:
        print(f"ok   {label}")
        return

    failures += 1
    print(f"FAIL {label}" + (f" -- {detail}" if detail else ""))


# --- the stubs ---------------------------------------------------------------


class _Recorder:
    """Counts the calls the code under test makes into a stub."""

    def __init__(self) -> None:
        self.cuda_empty = 0
        self.mps_empty = 0
        self.mem_get_info: list[int] = []


recorder = _Recorder()


def fake_torch(
    *,
    gpus: list[tuple[str, int]] | None = None,
    free: dict[int, int] | None = None,
    mps: bool = False,
    mem_info_raises: bool = False,
) -> types.ModuleType:
    """A torch stand-in: enough surface for device.py and local_mtl.py."""
    cards = gpus or []
    free_map = free or {}

    def mem_get_info(index=0):
        recorder.mem_get_info.append(index)

        if mem_info_raises:
            raise RuntimeError("no CUDA context")

        total = cards[index][1]
        return free_map.get(index, total), total

    def cuda_empty_cache():
        recorder.cuda_empty += 1

    def mps_empty_cache():
        recorder.mps_empty += 1

    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: bool(cards),
        device_count=lambda: len(cards),
        get_device_properties=lambda index: types.SimpleNamespace(
            name=cards[index][0], total_memory=cards[index][1]
        ),
        get_device_name=lambda index: cards[index][0],
        mem_get_info=mem_get_info,
        empty_cache=cuda_empty_cache,
    )
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    module.mps = types.SimpleNamespace(empty_cache=mps_empty_cache)
    module.version = types.SimpleNamespace(cuda="12.6")

    def inference_mode(*args, **kwargs):
        def decorate(fn):
            return fn

        return decorate

    module.inference_mode = inference_mode
    return module


def fake_psutil(available: int) -> types.ModuleType:
    module = types.ModuleType("psutil")
    module.virtual_memory = lambda: types.SimpleNamespace(
        total=32 * GIB, used=32 * GIB - available, available=available
    )
    return module


def fake_model_libs() -> None:
    """sentencepiece / transformers, imported but never called."""
    sp = types.ModuleType("sentencepiece")
    sp.SentencePieceProcessor = lambda *args, **kwargs: types.SimpleNamespace()
    sys.modules["sentencepiece"] = sp

    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
    sys.modules["transformers"] = tf


fake_model_libs()
sys.modules["torch"] = fake_torch()
sys.modules["psutil"] = fake_psutil(16 * GIB)

from fox_reader import device  # noqa: E402
from fox_reader.translate import local_mtl  # noqa: E402


def install(
    torch_module: types.ModuleType | None,
    psutil_module: types.ModuleType | None,
    *,
    system: str = "Linux",
) -> None:
    """Swap the hardware out from under both modules and drop their caches."""
    if torch_module is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = torch_module

    if psutil_module is None:
        sys.modules.pop("psutil", None)
    else:
        sys.modules["psutil"] = psutil_module

    device._torch_probed = False
    device._torch_module = None
    device._hardware = None
    device.platform.system = lambda: system

    local_mtl._psutil_probed = False
    local_mtl._psutil_module = None


Q8 = "gemma-4-e4b-q8-uncensored"
Q6 = "gemma-4-e4b-q6-uncensored"
VNTL = "vntl-llama3-8b-v2"

TWO_CARDS = [("RTX 4060", 8 * GIB), ("RTX 4090", 24 * GIB)]


# --- the CPU pool -----------------------------------------------------------

print("--- cpu, with room ---")
install(fake_torch(), fake_psutil(16 * GIB))

result = local_mtl.check_memory(Q8, "cpu")
check("a check comes back", result is not None)
check("it fits", result.fits is True)
check("and it was measured", result.measured is True)
check("against system RAM", result.pool == "ram", result.pool)
check("using the cpu figure", result.required_mib == 7000, str(result.required_mib))
check("free RAM in MiB", result.available_mib == 16 * 1024, str(result.available_mib))
check("named the model", "Gemma 4 E4B" in result.message, result.message)
check("named the device", "the CPU" in result.message, result.message)

print("\n--- cpu, without room ---")
install(fake_torch(), fake_psutil(3 * GIB))

result = local_mtl.check_memory(Q8, "cpu")
check("it does not fit", result.fits is False)
check("both figures are quoted", "7000 MiB" in result.message and "3072 MiB" in result.message, result.message)
check("says RAM, not VRAM", "of RAM" in result.message and "VRAM" not in result.message, result.message)
check("and suggests something", "Close some applications" in result.message, result.message)

print("\n--- the smaller quant still fits where the larger does not ---")
install(fake_torch(), fake_psutil(int(6.5 * GIB)))

result = local_mtl.check_memory(Q6, "cpu")
check("the Q6 fits in 6.5 GiB", result.fits is True, str(result.available_mib))
check("with its own requirement", result.required_mib == 6000, str(result.required_mib))

result = local_mtl.check_memory(Q8, "cpu")
check("while the Q8 does not", result.fits is False, str(result.available_mib))


# --- the VRAM pool ----------------------------------------------------------

print("\n--- cuda, with room ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 20 * GIB}), fake_psutil(2 * GIB))
recorder.mem_get_info.clear()

result = local_mtl.check_memory(Q8, "cuda:1")
check("it fits", result.fits is True)
check("measured against VRAM", result.pool == "vram", result.pool)
check("using the cuda figure", result.required_mib == 6800, str(result.required_mib))
check("free VRAM in MiB", result.available_mib == 20 * 1024, str(result.available_mib))
check("the right card was asked", recorder.mem_get_info == [1], str(recorder.mem_get_info))
check("the card is named", "GPU 1 (RTX 4090)" in result.message, result.message)
check("short system RAM is irrelevant here", "RAM" not in result.message.replace("VRAM", ""), result.message)

print("\n--- cuda, without room ---")
install(fake_torch(gpus=TWO_CARDS, free={0: 3 * GIB}), fake_psutil(16 * GIB))

result = local_mtl.check_memory(Q8, "cuda:0")
check("it does not fit", result.fits is False)
check("the device is unchanged -- no cpu fallback", result.device == "cuda:0", result.device)
check("both figures are quoted", "6800 MiB" in result.message and "3072 MiB" in result.message, result.message)
check("says VRAM", "of VRAM" in result.message, result.message)
check("points at the device setting", "Compute Devices" in result.message, result.message)
check("and does not offer the CPU", "CPU" not in result.message, result.message)

print("\n--- another process holding the card ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 1 * GIB}), fake_psutil(16 * GIB))
result = local_mtl.check_memory(Q6, "cuda:1")
check("a 24 GB card with 1 GB free fails", result.fits is False, str(result.available_mib))
check("total memory is not what was measured", result.available_mib == 1024, str(result.available_mib))


# --- unmeasurable machines --------------------------------------------------

print("\n--- psutil missing ---")
install(fake_torch(), None)

result = local_mtl.check_memory(Q8, "cpu")
check("still answers", result is not None)
check("allowed through", result.fits is True)
check("but flagged unmeasured", result.measured is False)
check("with no figure", result.available_mib is None, str(result.available_mib))
check("and says so", "could not be measured" in result.message, result.message)

print("\n--- torch missing, cuda asked for ---")
install(None, fake_psutil(16 * GIB))

result = local_mtl.check_memory(Q8, "cuda:0")
check("allowed through", result.fits is True and result.measured is False)
check("did not silently use system RAM", result.pool == "vram", result.pool)

print("\n--- the driver refusing to answer ---")
install(fake_torch(gpus=TWO_CARDS, mem_info_raises=True), fake_psutil(16 * GIB))

result = local_mtl.check_memory(Q8, "cuda:0")
check("allowed through", result.fits is True and result.measured is False)


# --- Apple ------------------------------------------------------------------

print("\n--- mps ---")
install(fake_torch(mps=True), fake_psutil(8 * GIB), system="Darwin")

result = local_mtl.check_memory(Q8, "mps")
check("the Q8 stays on the Apple GPU", result.device == "mps", result.device)
check("measured against unified RAM", result.pool == "ram", result.pool)
check("against the gpu requirement", result.required_mib == 6800, str(result.required_mib))
check("and says where", "Apple GPU" in result.message, result.message)

result = local_mtl.check_memory(Q6, "mps")
check("so does the Q6", result.device == "mps", result.device)
check("with its own requirement", result.required_mib == 5800, str(result.required_mib))
check("effective_device leaves mps alone", local_mtl.effective_device(Q6, "mps") == "mps")
check("and leaves cuda alone", local_mtl.effective_device(Q6, "cuda:1") == "cuda:1")


# --- nothing to check -------------------------------------------------------

print("\n--- a model with no requirement ---")
install(fake_torch(), fake_psutil(1 * MIB))
check("an unknown id is not checked", local_mtl.check_memory("who-knows", "cpu") is None)


# --- the override -----------------------------------------------------------

print("\n--- the escape hatch ---")
import os  # noqa: E402

install(fake_torch(), fake_psutil(1 * GIB))
check("off by default", local_mtl.memory_check_disabled() is False)

for value, expected in [("1", True), ("true", True), ("ON", True), ("0", False), ("", False)]:
    os.environ[local_mtl.IGNORE_MEMORY_CHECK_ENV] = value
    check(f"{value!r} -> {expected}", local_mtl.memory_check_disabled() is expected)

os.environ.pop(local_mtl.IGNORE_MEMORY_CHECK_ENV, None)


# --- the manager ------------------------------------------------------------


class FakeSettings:
    def __init__(self, defaults: dict[str, str | None]) -> None:
        self.defaults = defaults

    def get_defaults(self) -> dict[str, str | None]:
        return dict(self.defaults)

    def get_default_model(self, language: str) -> str | None:
        return self.defaults.get(language.strip().lower())


built: list[str] = []


def fake_translator(model_id: str, langs: list[str]):
    class Fake:
        def __init__(self, mtl_dir):
            built.append(model_id)
            self.mtl_dir = mtl_dir

        def translate(self, text, **kwargs):
            return f"[{model_id}] {text}"

        @classmethod
        def model_id(cls):
            return model_id

        @classmethod
        def model_langs(cls):
            return langs

    return Fake


def manager(defaults: dict[str, str | None]) -> local_mtl.LocalMTLManager:
    mgr = local_mtl.LocalMTLManager(pathlib.Path("/nowhere"), settings=FakeSettings(defaults))
    mgr.available_translators = [
        fake_translator(Q6, ["japanese", "korean", "chinese"]),
        fake_translator(Q8, ["japanese", "korean", "chinese"]),
        fake_translator(VNTL, ["japanese"]),
    ]
    return mgr


print("\n--- load() refuses before it builds anything ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 2 * GIB}), fake_psutil(16 * GIB))
device.configure({"translator": "cuda:1"})

mgr = manager({"japanese": Q8, "korean": Q8, "chinese": Q8})
built.clear()

check("the manager reads the configured device", mgr.device == "cuda:1", mgr.device)

try:
    mgr.load("japanese")
    check("it raised", False, "load() returned normally")
except local_mtl.InsufficientMemoryError as exc:
    check("it raised InsufficientMemoryError", True)
    check("the message is the check's", str(exc) == exc.check.message)
    check("which is user-readable", "6800 MiB of VRAM" in str(exc), str(exc))
    check("nothing was constructed", built == [], str(built))
    check("and no model is held", mgr.model is None and mgr.current_lang is None)
except Exception as exc:  # noqa: BLE001
    check("it raised InsufficientMemoryError", False, f"{type(exc).__name__}: {exc}")

print("\n--- and translate() surfaces the same refusal ---")
try:
    mgr.translate("japanese", "テスト")
    check("it raised", False, "translate() returned normally")
except local_mtl.InsufficientMemoryError as exc:
    check("it raised on the way through load()", "VRAM" in str(exc))

print("\n--- the same card with room for the smaller quant ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 6 * GIB}), fake_psutil(16 * GIB))
device.configure({"translator": "cuda:1"})
mgr = manager({"japanese": Q6})
built.clear()
mgr.load("japanese")
check("the Q6 loads in 6 GiB of VRAM", built == [Q6], str(built))
check("and is remembered", mgr.current_lang == "japanese" and mgr.model is not None)

print("\n--- a second load of the same model does not re-check ---")
recorder.mem_get_info.clear()
mgr.load("japanese")
check("no second construction", built == [Q6], str(built))
check("and no second measurement", recorder.mem_get_info == [], str(recorder.mem_get_info))

print("\n--- the override lets a short machine through ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 1 * GIB}), fake_psutil(16 * GIB))
device.configure({"translator": "cuda:1"})
os.environ[local_mtl.IGNORE_MEMORY_CHECK_ENV] = "1"

mgr = manager({"japanese": Q8})
built.clear()
mgr.load("japanese")
check("it loaded anyway", built == [Q8], str(built))

os.environ.pop(local_mtl.IGNORE_MEMORY_CHECK_ENV, None)


# --- unload -----------------------------------------------------------------

print("\n--- unload frees the driver's blocks ---")
install(fake_torch(gpus=TWO_CARDS), fake_psutil(16 * GIB))
recorder.cuda_empty = 0
recorder.mps_empty = 0

mgr.unload()
check("the CUDA cache was emptied", recorder.cuda_empty == 1, str(recorder.cuda_empty))
check("the MPS cache was not", recorder.mps_empty == 0, str(recorder.mps_empty))
check("and the model is gone", mgr.model is None and mgr.current_lang is None)

print("\n--- unload on a machine with no torch ---")
install(None, fake_psutil(16 * GIB))
mgr = manager({"japanese": Q6})
mgr.load("japanese")
mgr.unload()
check("does not raise", mgr.model is None)


# --- the report -------------------------------------------------------------

print("\n--- memory_report ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 6 * GIB}), fake_psutil(16 * GIB))
device.configure({"translator": "cuda:1"})

report = manager({"japanese": Q8, "korean": Q6, "chinese": Q6}).memory_report()

check("enabled", report["enabled"] is True)
check("names the device", report["device"] == "cuda:1", report["device"])
check("not ok", report["ok"] is False)
check("one row per model, not per language", len(report["checks"]) == 2, str(len(report["checks"])))

rows = {row["model"]: row for row in report["checks"]}
check("the Q6 fits", rows[Q6]["fits"] is True)
check("the Q8 does not", rows[Q8]["fits"] is False)
check("the Q6's languages are collected", rows[Q6]["languages"] == ["korean", "chinese"], str(rows[Q6]["languages"]))
check("the Q8's are too", rows[Q8]["languages"] == ["japanese"], str(rows[Q8]["languages"]))
check("only the failure is an issue", [row["model"] for row in report["issues"]] == [Q8], str(report["issues"]))
check("every issue carries a message", all(row["message"] for row in report["issues"]))
check("and is JSON-ready", all(isinstance(v, (str, int, bool, list, type(None))) for row in report["checks"] for v in row.values()))

print("\n--- memory_report when everything fits ---")
install(fake_torch(gpus=TWO_CARDS, free={1: 20 * GIB}), fake_psutil(16 * GIB))
device.configure({"translator": "cuda:1"})

report = manager({"japanese": Q8}).memory_report()
check("ok", report["ok"] is True)
check("no issues", report["issues"] == [])
check("but the check is still reported", len(report["checks"]) == 1)

print("\n--- memory_report with nothing configured ---")
report = manager({}).memory_report()
check("ok", report["ok"] is True)
check("and nothing to say", report["checks"] == [])

print("\n--- memory_report with no settings manager at all ---")
mgr = local_mtl.LocalMTLManager(pathlib.Path("/nowhere"), settings=None)
report = mgr.memory_report()
check("does not raise", report["ok"] is True and report["checks"] == [])

print(f"\n{failures} FAILURE(S)" if failures else "\nall MTL memory checks passed")
sys.exit(1 if failures else 0)
