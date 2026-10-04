"""Exercise device.py's GPU paths with a stubbed torch.

The scratch environment has no torch and no CUDA card, so the interesting half
of the module — several GPUs, picking the one with the most memory, MPS, a probe
that raises — can only be reached with a stand-in. Each scenario installs a fake
`torch` in sys.modules, clears the module's caches, and asks for a resolution.
"""

from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fox_reader import device  # noqa: E402

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if not condition:
        failures += 1

    print(f"{'ok  ' if condition else 'FAIL'} {label}" + (f" -- {detail}" if detail else ""))


def fake_torch(
    *,
    gpus: list[tuple[str, int]] | None = None,
    mps: bool = False,
    cuda_version: str | None = "12.6",
    raise_on: str | None = None,
) -> types.ModuleType:
    """A torch stand-in exposing only what device.py touches."""
    gpus = gpus or []

    module = types.ModuleType("torch")

    def is_available() -> bool:
        if raise_on == "is_available":
            raise RuntimeError("no driver")

        return bool(gpus)

    def device_count() -> int:
        if raise_on == "device_count":
            raise RuntimeError("driver mismatch")

        return len(gpus)

    def get_device_properties(index: int):
        if raise_on == "properties":
            raise RuntimeError("cannot query")

        name, total = gpus[index]
        return types.SimpleNamespace(name=name, total_memory=total)

    def get_device_name(index: int) -> str:
        return gpus[index][0]

    module.cuda = types.SimpleNamespace(
        is_available=is_available,
        device_count=device_count,
        get_device_properties=get_device_properties,
        get_device_name=get_device_name,
    )
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    module.version = types.SimpleNamespace(cuda=cuda_version)

    return module


def install(module: types.ModuleType | None, system: str = "Windows") -> None:
    """Swap in a torch and reset every cache device.py keeps."""
    if module is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = module

    device._torch_probed = False
    device._torch_module = None
    device._hardware = None
    device.platform.system = lambda: system  # type: ignore[assignment]


GIB = 1024**3

print("--- two cards, the larger one second ---")
install(fake_torch(gpus=[("RTX 4060", 8 * GIB), ("RTX 4090", 24 * GIB)]))

resolution = device.resolve()
check("auto takes the most memory", resolution.resolved["bubble"] == "cuda:1", str(resolution.resolved))
check("no fallbacks", resolution.fallbacks == [], str(resolution.fallbacks))
check("paddleocr gets paddle's spelling", resolution.device.paddleocr == "gpu:1", resolution.device.paddleocr)
check("bubble gets a torch string", resolution.device.bubble == "cuda:1", resolution.device.bubble)
check("translator gets a torch string", resolution.device.translator == "cuda:1", resolution.device.translator)
check("textseg gets a torch string", resolution.device.textseg == "cuda:1", resolution.device.textseg)
check("paddleocr_vl gets a torch string", resolution.device.paddleocr_vl == "cuda:1", resolution.device.paddleocr_vl)
check("backend says cuda", resolution.device.backend == "cuda", resolution.device.backend)
check("cuda_available", resolution.device.cuda_available is True)
check("cuda_version came through", resolution.device.cuda_version == "12.6", str(resolution.device.cuda_version))
check("gpu_name names the winner", resolution.device.gpu_name == "RTX 4090", str(resolution.device.gpu_name))
check("device_count counts both", resolution.device.device_count == 2, str(resolution.device.device_count))

labels = [option.label for option in device.options()]
check("picker lists auto, cpu and both cards", len(labels) == 4, str(labels))
check("auto names the winner", labels[0] == "Auto — RTX 4090", labels[0])
check("memory is shown in GB", "24.0 GB" in labels[3], labels[3])

print("\n--- pinning each slot somewhere different ---")
resolution = device.resolve({
    "paddleocr": "cuda:0", "paddleocr_vl": "cpu", "bubble": "cpu",
    "translator": "auto", "textseg": "cuda:1",
})
check("paddleocr pinned to card 0", resolution.device.paddleocr == "gpu:0", resolution.device.paddleocr)
check("paddleocr_vl pinned to the cpu", resolution.device.paddleocr_vl == "cpu", resolution.device.paddleocr_vl)
check("bubble pinned to the cpu", resolution.device.bubble == "cpu", resolution.device.bubble)
check("translator still auto-picks", resolution.device.translator == "cuda:1", resolution.device.translator)
check("textseg pinned to the other card", resolution.device.textseg == "cuda:1", resolution.device.textseg)
check("backend still describes the machine", resolution.device.backend == "cuda", resolution.device.backend)

print("\n--- textseg diverges from paddleocr ---")
resolution = device.resolve({
    "paddleocr": "cpu", "paddleocr_vl": "cuda:1", "bubble": "auto",
    "translator": "auto", "textseg": "cuda:1",
})
check("ocr stays on the cpu", resolution.device.paddleocr == "cpu", resolution.device.paddleocr)
check("the vl engine stays on the card", resolution.device.paddleocr_vl == "cuda:1", resolution.device.paddleocr_vl)
check("cleaning stays on the card", resolution.device.textseg == "cuda:1", resolution.device.textseg)

print("\n--- a selection that predates the textseg slot ---")
resolution = device.resolve({"paddleocr": "cuda:0", "bubble": "cpu", "translator": "auto"})
check("textseg inherits paddleocr", resolution.selection["textseg"] == "cuda:0", str(resolution.selection))
check("and lands on the same card", resolution.device.textseg == "cuda:0", resolution.device.textseg)
check("paddleocr_vl inherits paddleocr too",
      resolution.selection["paddleocr_vl"] == "cuda:0", str(resolution.selection))
check("and lands on the same card", resolution.device.paddleocr_vl == "cuda:0", resolution.device.paddleocr_vl)

print("\n--- a card that is gone ---")
resolution = device.resolve({
    "paddleocr": "cuda:7", "paddleocr_vl": "auto", "bubble": "auto",
    "translator": "auto", "textseg": "auto",
})
check("the missing slot lands on the cpu", resolution.device.paddleocr == "cpu", resolution.device.paddleocr)
check("and says so once", len(resolution.fallbacks) == 1, str(resolution.fallbacks))
check("the preference is not rewritten",
      resolution.selection["paddleocr"] == "cuda:7", resolution.selection["paddleocr"])
check("the other slots are unaffected", resolution.device.bubble == "cuda:1", resolution.device.bubble)
check("an explicit textseg auto is unaffected too", resolution.device.textseg == "cuda:1", resolution.device.textseg)
check("and so is an explicit paddleocr_vl auto",
      resolution.device.paddleocr_vl == "cuda:1", resolution.device.paddleocr_vl)

print("\n--- a card that is gone, inherited by a legacy selection ---")
resolution = device.resolve({"paddleocr": "cuda:7", "bubble": "auto", "translator": "auto"})
check("the missing slot lands on the cpu", resolution.device.paddleocr == "cpu", resolution.device.paddleocr)
check("and the inherited textseg falls back too", resolution.device.textseg == "cpu", resolution.device.textseg)
check("and the inherited paddleocr_vl falls back too",
      resolution.device.paddleocr_vl == "cpu", resolution.device.paddleocr_vl)
check("with a line for each", len(resolution.fallbacks) == 3, str(resolution.fallbacks))

print("\n--- apple metal ---")
install(fake_torch(mps=True, cuda_version=None), system="Darwin")

resolution = device.resolve()
check("auto takes the apple gpu", resolution.resolved["bubble"] == "mps", str(resolution.resolved))
check("bubble uses mps", resolution.device.bubble == "mps", resolution.device.bubble)
check("translator uses mps", resolution.device.translator == "mps", resolution.device.translator)
check("textseg uses mps", resolution.device.textseg == "mps", resolution.device.textseg)
check("paddleocr_vl uses mps too, unlike classic paddleocr",
      resolution.device.paddleocr_vl == "mps", resolution.device.paddleocr_vl)
check("paddleocr cannot, so cpu", resolution.device.paddleocr == "cpu", resolution.device.paddleocr)
check("and it explains why", any("Apple GPU" in line for line in resolution.fallbacks), str(resolution.fallbacks))
check("backend says mps", resolution.device.backend == "mps", resolution.device.backend)
check("no cuda", resolution.device.cuda_available is False and resolution.device.device_count == 0)

print("\n--- mps is ignored off apple ---")
install(fake_torch(mps=True, cuda_version=None), system="Linux")
check("auto stays on the cpu", device.resolve().device.translator == "cpu")
check("and mps is not offered", all(o.id != "mps" for o in device.options()))

print("\n--- probes that raise ---")
for stage in ("is_available", "device_count", "properties"):
    install(fake_torch(gpus=[("RTX 4090", 24 * GIB)], raise_on=stage))
    resolution = device.resolve()

    if stage == "properties":
        check(f"{stage} raising still finds the card", resolution.device.bubble == "cuda:0", resolution.device.bubble)
        check(f"{stage} raising leaves a generic name", resolution.device.gpu_name == "RTX 4090", str(resolution.device.gpu_name))
    else:
        check(f"{stage} raising falls back to the cpu", resolution.device.bubble == "cpu", resolution.device.bubble)
        check(f"{stage} raising reports no cuda", resolution.device.cuda_available is False)

print("\n--- torch missing entirely ---")
install(None)
resolution = device.resolve()
check("everything on the cpu", set(resolution.resolved.values()) == {"cpu"}, str(resolution.resolved))
check("no fallback noise", resolution.fallbacks == [], str(resolution.fallbacks))
check("only auto and cpu are offered", [o.id for o in device.options()] == ["auto", "cpu"])

print("\n--- torch that explodes on import ---")


class Exploding(types.ModuleType):
    def __getattr__(self, name):
        raise RuntimeError("broken build")


install(Exploding("torch"))
check("still resolves", device.resolve().device.translator == "cpu")

print("\n--- configure mutates the singleton in place ---")
install(fake_torch(gpus=[("RTX 4090", 24 * GIB)]))
before = device.FOX_DEVICE
device.configure({"paddleocr": "auto", "bubble": "auto", "translator": "cpu", "textseg": "cpu"})

check("same object", device.FOX_DEVICE is before)
check("paddleocr updated", device.FOX_DEVICE.paddleocr == "gpu:0", device.FOX_DEVICE.paddleocr)
check("bubble updated", device.FOX_DEVICE.bubble == "cuda:0", device.FOX_DEVICE.bubble)
check("translator stayed pinned", device.FOX_DEVICE.translator == "cpu", device.FOX_DEVICE.translator)
check("textseg pinned to the cpu", device.FOX_DEVICE.textseg == "cpu", device.FOX_DEVICE.textseg)

snapshot = device.snapshot({"paddleocr": "auto", "bubble": "auto", "translator": "cpu", "textseg": "cpu"})
check("snapshot shows the request", snapshot["selection"]["translator"] == "cpu")
check("snapshot carries the new slot", "paddleocr_vl" in snapshot["slots"], str(snapshot["slots"]))
check("inherited like the resolve does",
      snapshot["selection"]["paddleocr_vl"] == "auto", str(snapshot["selection"]))
check("snapshot shows the resolution", snapshot["resolved"]["bubble"] == "cuda:0")
check("snapshot shows what is live", snapshot["active"]["bubble"] == "cuda:0", str(snapshot["active"]))
check("snapshot carries the picker", len(snapshot["options"]) == 3, str(len(snapshot["options"])))

print("\n--- the cpu's own shape ---")

# The GPU half of this module can only be reached with a stub; this half answers
# about the machine the harness is really running on, so the assertions are
# bounds rather than figures. psutil is absent from the scratch environment,
# which is exactly the fallback path worth walking.
check("there is always at least one usable cpu", device.usable_cpus() >= 1, str(device.usable_cpus()))
check(
    "the physical count is answered bare, the way the settings page asks",
    device.physical_cores() >= 1,
    str(device.physical_cores()),
)
check(
    "and never claims more cores than cpus",
    device.physical_cores() <= device.usable_cpus(),
    f"{device.physical_cores()} of {device.usable_cpus()}",
)
check("a one-cpu machine has one core", device.physical_cores(1) == 1)
check("a two-cpu machine is not halved", device.physical_cores(2) == 2)
check("and a wider one stays in range", 1 <= device.physical_cores(8) <= 8, str(device.physical_cores(8)))

check("nothing is held back on one core", device.reserved_cores(1) == 0)
check("nor on two", device.reserved_cores(2) == 0)
check("a quad-core gives up one", device.reserved_cores(4) == 1)
check("and a bigger one two", device.reserved_cores(16) == 2)
check(
    "a core is always left to work with",
    all(device.reserved_cores(n) < n for n in range(1, 65)),
)

check("this machine gets a workable thread count", device.thread_count() >= 1, str(device.thread_count()))
check(
    "within the cap",
    device.thread_count() <= device.MAX_CPU_THREADS,
    str(device.thread_count()),
)
check(
    "and fewer when the model is on a card",
    device.thread_count(on_gpu=True) <= device.MAX_GPU_THREADS,
    str(device.thread_count(on_gpu=True)),
)

check("'auto' is spelled the same as it is for a device slot", device.AUTO_THREADS == device.AUTO)
check("a count normalizes to an int, which is what yaml should hold", device.normalize_threads("6") == 6)
check(
    "an int, not a bool wearing one",
    type(device.normalize_threads(6)) is int,
    type(device.normalize_threads(6)).__name__,
)
check("and auto normalizes to the string", device.normalize_threads("auto") == device.AUTO_THREADS)

print("\n--- the field set the consumers rely on ---")
check(
    "field names are the five slots plus the machine description",
    set(device.FoxDevice.model_fields) == {
        "translator", "paddleocr", "paddleocr_vl", "bubble", "textseg",
        "backend", "cuda_available", "cuda_version", "gpu_name", "device_count",
    },
    str(sorted(device.FoxDevice.model_fields)),
)
check("detect_best_device still returns a FoxDevice", isinstance(device.detect_best_device(), device.FoxDevice))

print(f"\n{failures} FAILURE(S)" if failures else "\nall device checks passed")
sys.exit(1 if failures else 0)
