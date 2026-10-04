"""Central device configuration for all ML models.

Each consumer wants the device spelled a different way — PaddleOCR takes
``gpu:0`` while torch-based consumers take ``cuda:0`` — so the choice is
made once here and handed out per slot. The CPU's own
shape is answered here as well (see `thread_count`), for the one model that
chooses its own thread count instead of being told.

Nothing in this module is allowed to raise, apart from the two grammar parsers
`normalize_id` and `normalize_threads`, whose whole job is to reject a value.
torch is an optional extra (see the ``cpu`` / ``cu126`` / ``macos`` groups in
pyproject.toml), a CUDA build can be installed against a driver that is missing
or too old, and a probe can fail for reasons that have nothing to do with the
caller. Every one of those cases has to end at the CPU rather than at a
traceback, because a reader that runs slowly is worth more than one that will
not start.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# device ids
# ---------------------------------------------------------------------------

#: Resolve to the best device present at launch — the GPU with the most memory,
#: or the CPU when there is none. This is the default for every slot, so moving
#: the config to another machine, or adding a card to this one, needs no edit.
AUTO = "auto"

CPU = "cpu"
MPS = "mps"
CUDA = "cuda"

#: The slots a user can choose a device for. `backend` is deliberately absent:
#: it is a label derived from the hardware, not a setting. `textseg` is the
#: glyph-level text-segmentation network used by the Text Seg clean method; it
#: used to share the `paddleocr` slot and now has its own picker so OCR and
#: cleaning can sit on different devices (e.g. OCR on the CPU to free VRAM
#: while Text Seg stays on the GPU, or vice versa). `paddleocr_vl` is the
#: PaddleOCR-VL-1.6 GGUF transcriber, which runs on llama.cpp rather than
#: PaddlePaddle and therefore needs its own picker too: pinning the classic
#: pipeline to the CPU must not drag the VL engine with it, and the two must
#: be separable in VRAM.
SLOTS: tuple[str, ...] = ("paddleocr", "paddleocr_vl", "bubble", "translator", "textseg")

_BYTES_PER_GIB = 1024**3


class FoxDevice(BaseModel):
    """The resolved device for each consumer, plus what was detected.

    The five device strings are ready to pass straight to their consumer.
    ``backend``, ``cuda_available``, ``cuda_version``, ``gpu_name`` and
    ``device_count`` describe the machine rather than any one choice, which is
    what makes them useful in a bug report.
    """

    translator: str = Field(default=CPU)
    paddleocr: str = Field(default=CPU)
    #: PaddleOCR-VL transcriber, as a torch-style device string (``cuda:0`` /
    #: ``mps`` / ``cpu``). llama.cpp understands Metal, so unlike `paddleocr`
    #: an Apple GPU is honoured here rather than mapped back to the CPU.
    paddleocr_vl: str = Field(default=CPU)
    #: Bubble segmentation network, as a torch device string (``cuda:0`` /
    #: ``mps`` / ``cpu``) — the same spelling torch understands directly.
    bubble: str = Field(default=CPU)
    #: Text Seg cleaning network, as a torch device string (``cuda:0`` / ``mps``
    #: / ``cpu``) — the same spelling as `translator`, which is what torch
    #: understands directly.
    textseg: str = Field(default=CPU)

    backend: str = Field(default=CPU)
    cuda_available: bool = False
    cuda_version: str | None = None
    gpu_name: str | None = None

    device_count: int = 0


class DeviceOption(BaseModel):
    """One entry in the picker on the settings page."""

    id: str
    kind: str
    index: int | None = None
    name: str
    label: str
    total_memory: int | None = None
    available: bool = True


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------

_torch_probed = False
_torch_module: Any | None = None


def _torch() -> Any | None:
    """The torch module, or None when it cannot be used.

    Probed once. An ImportError is the ordinary case on a CPU-only install that
    skipped the extra; anything else means torch is present but unusable, which
    is worth a louder log line and the same answer.
    """
    global _torch_probed, _torch_module

    if _torch_probed:
        return _torch_module

    _torch_probed = True

    try:
        import torch
    except ImportError as exc:
        logger.info("torch is not installed, using the CPU for everything: %s", exc)
        return None
    except Exception as exc:
        logger.warning("torch could not be imported, falling back to the CPU: %s", exc)
        return None

    _torch_module = torch
    return _torch_module


def torch_module() -> Any | None:
    """The torch module, or None when it cannot be used.

    Public face of the cached probe above, for the modules that want to ask
    torch something themselves — free VRAM, an emptied cache — without each of
    them repeating the guard or paying for a second import attempt.
    """
    return _torch()


class _Gpu(BaseModel):
    """A CUDA device that answered a probe."""

    index: int
    name: str
    total_memory: int | None = None


def _cuda_gpus(torch: Any) -> list[_Gpu]:
    """Every usable CUDA device, in ordinal order.

    A machine can report a device that then fails to describe itself — a driver
    mismatch does exactly that — so a device that cannot be queried at all is
    still listed, under a generic name. Selecting it will fail later and fall
    back, which beats hiding a card that works.
    """
    try:
        if not torch.cuda.is_available():
            return []
    except Exception as exc:
        logger.warning("Could not ask torch about CUDA: %s", exc)
        return []

    try:
        count = int(torch.cuda.device_count())
    except Exception as exc:
        logger.warning("Could not count the CUDA devices: %s", exc)
        return []

    gpus: list[_Gpu] = []

    for index in range(max(count, 0)):
        name = f"CUDA device {index}"
        total: int | None = None

        try:
            properties = torch.cuda.get_device_properties(index)
            name = str(getattr(properties, "name", name) or name)
            total = int(getattr(properties, "total_memory", 0)) or None
        except Exception as exc:
            logger.debug("Could not read CUDA device %d properties: %s", index, exc)

            try:
                name = str(torch.cuda.get_device_name(index) or name)
            except Exception:
                pass

        gpus.append(_Gpu(index=index, name=name, total_memory=total))

    return gpus


def _mps_available(torch: Any) -> bool:
    """Whether Apple's Metal backend is usable.

    Guarded on the platform as well as the probe: `torch.backends.mps` exists on
    every build, and on a non-Apple one the answer is a slow no.
    """
    if platform.system() != "Darwin":
        return False

    try:
        return bool(torch.backends.mps.is_available())
    except Exception as exc:
        logger.debug("Could not ask torch about MPS: %s", exc)
        return False


def _cuda_version(torch: Any) -> str | None:
    try:
        return getattr(torch.version, "cuda", None) or None
    except Exception:
        return None


class _Hardware(BaseModel):
    """What one sweep of the machine found."""

    gpus: list[_Gpu] = Field(default_factory=list)
    mps: bool = False
    cuda_version: str | None = None

    @property
    def cuda_available(self) -> bool:
        return bool(self.gpus)

    def gpu(self, index: int) -> _Gpu | None:
        return next((gpu for gpu in self.gpus if gpu.index == index), None)

    def best(self) -> str:
        """The id `auto` resolves to: most memory wins, then the lowest ordinal.

        A card whose memory could not be read sorts as 0 and so loses to any
        card that answered — but still beats the CPU, which is the point.
        """
        if self.gpus:
            best = max(self.gpus, key=lambda gpu: (gpu.total_memory or 0, -gpu.index))
            return f"{CUDA}:{best.index}"

        if self.mps:
            return MPS

        return CPU


_hardware: _Hardware | None = None


def probe(refresh: bool = False) -> _Hardware:
    """Sweep the machine once and remember the answer.

    Called at startup, and again only when something asks for a refresh: each
    CUDA query initialises a context, which is not free.
    """
    global _hardware

    if _hardware is not None and not refresh:
        return _hardware

    torch = _torch()

    if torch is None:
        _hardware = _Hardware()
        return _hardware

    gpus = _cuda_gpus(torch)

    _hardware = _Hardware(
        gpus=gpus,
        mps=_mps_available(torch),
        cuda_version=_cuda_version(torch) if gpus else None,
    )

    return _hardware


# ---------------------------------------------------------------------------
# cpu threads
# ---------------------------------------------------------------------------

#: `mtl_threads` set to this means "work it out from the machine", which is what
#: `thread_count` does below. Anything else is a pinned number of threads.
AUTO_THREADS = AUTO

#: Hand-tune the thread count without touching the settings file. Wins over the
#: saved preference: it is for a session of batch work on an otherwise idle
#: machine, and for asking someone in a bug report to try a number.
MTL_THREADS_ENV = "FOX_READER_MTL_THREADS"

#: Past roughly this many threads llama.cpp stops getting faster on consumer
#: hardware — token generation is bound by memory bandwidth, not by cores, and
#: the extra threads mostly spin on the barrier between layers.
MAX_CPU_THREADS = 8

#: With every layer on the GPU the CPU only marshals tokens, so a handful of
#: threads is plenty and the rest are better left to the rest of the app.
MAX_GPU_THREADS = 4

_psutil_probed = False
_psutil_module: Any | None = None


def _psutil() -> Any | None:
    """The psutil module, or None when it cannot be used. Probed once."""
    global _psutil_probed, _psutil_module

    if _psutil_probed:
        return _psutil_module

    _psutil_probed = True

    try:
        import psutil
    except Exception as exc:  # noqa: BLE001 - any failure means "ask elsewhere"
        logger.info("psutil is unavailable, guessing the core layout: %s", exc)
        return None

    _psutil_module = psutil
    return _psutil_module


def _int_env(name: str) -> int | None:
    """An environment variable as an int, or None if unset or unparseable."""
    raw = os.environ.get(name, "").strip()

    if not raw:
        return None

    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: expected an integer", name, raw)
        return None


def usable_cpus() -> int:
    """Logical CPUs this process is actually allowed to run on.

    Affinity before core count: a container with a two-CPU quota and a process
    pinned with ``taskset`` both still see the whole machine through
    `os.cpu_count`, and threads that cannot be scheduled anywhere only fight
    each other for the CPUs that are left.
    """
    probes = (
        # Linux, and the only one of the three that is exact.
        lambda: len(os.sched_getaffinity(0)),  # type: ignore[attr-defined]
        # 3.13+; respects affinity on Windows too. Absent on 3.12, hence the
        # AttributeError below rather than a version check.
        lambda: os.process_cpu_count(),  # type: ignore[attr-defined]
        lambda: len(_psutil().Process().cpu_affinity()),  # type: ignore[union-attr]
        lambda: os.cpu_count(),
    )

    for probe in probes:
        try:
            value = probe()
        except Exception:  # noqa: BLE001 - each probe is absent on some platform
            continue

        if value:
            return max(1, int(value))

    return 1


def physical_cores(usable: int | None = None) -> int:
    """Cores that are not SMT siblings, as far as this machine will say.

    Worth the trouble because llama.cpp's kernels are bandwidth-bound: two
    threads on one physical core share its load/store units, so counting
    hyperthreads typically makes generation slower rather than faster. This is
    also the ceiling the settings page offers, so it has to answer on a machine
    that cannot be measured rather than give up.
    """
    if usable is None:
        usable = usable_cpus()

    psutil = _psutil()

    if psutil is not None:
        try:
            counted = psutil.cpu_count(logical=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("psutil could not count the physical cores: %s", exc)
            counted = None

        if counted:
            # psutil counts the machine; `usable` is what we are allowed to
            # touch. The narrower of the two is the honest answer.
            return max(1, min(int(counted), usable))

    # No answer available: assume SMT on anything above two logical CPUs.
    # Guessing wrong that way costs a little speed on a machine without it,
    # where guessing the other way costs the responsiveness this whole
    # heuristic exists to protect.
    return max(1, usable // 2) if usable > 2 else usable


def reserved_cores(physical: int) -> int:
    """Cores deliberately left for everything that is not the translator.

    Fox Reader is a web server, a bubble detector and an OCR pass in the same
    process as the local model. Handing llama.cpp every core makes a CPU
    translation freeze the page it was started from, which reads as a hang
    rather than as work in progress — so roughly a quarter of the machine stays
    free, and a small machine gives up less of it because it has less to give.
    """
    if physical <= 2:
        return 0

    if physical <= 4:
        return 1

    return 2


def normalize_threads(value: Any) -> str | int:
    """Parse a thread preference: `auto`, or a count of at least one.

    Grammar only, for the same reason `normalize_id` is: a settings.yaml written
    on a sixteen-core desktop should still say 12 on a laptop, with the clamp to
    what this machine can actually run happening in `thread_count` and the
    preference left as written.
    """
    if value is None:
        return AUTO_THREADS

    # Before the int branch: bool is an int, and `mtl_threads: yes` in YAML is a
    # typo rather than a request for one thread.
    if isinstance(value, bool):
        raise ValueError(f"Unknown thread count: {value!r}. Expected 'auto' or a number.")

    if isinstance(value, int):
        count = value
    else:
        text = str(value).strip().lower()

        if not text or text == AUTO_THREADS:
            return AUTO_THREADS

        if not text.isdigit():
            raise ValueError(
                f"Unknown thread count: {value!r}. Expected 'auto' or a number."
            )

        count = int(text)

    if count < 1:
        raise ValueError(f"Thread count {count} is below one. Expected 'auto' or a number.")

    return count


def thread_override() -> int | None:
    """The count MTL_THREADS_ENV forces, or None when it is unset or unusable."""
    value = _int_env(MTL_THREADS_ENV)

    if value is None:
        return None

    if value < 1:
        logger.warning("Ignoring %s=%d: need at least one thread", MTL_THREADS_ENV, value)
        return None

    return value


_mtl_threads: str | int = AUTO_THREADS


def configure_threads(value: Any = None) -> str | int:
    """Apply a saved thread preference, and answer with what was applied.

    Lenient, like the rest of startup: a value that is not a count is logged and
    becomes `auto` rather than stopping anything. Called each time a model is
    about to load, which is what lets the settings page change the number
    without a restart, so it only logs when the answer actually changed.
    """
    global _mtl_threads

    try:
        wanted = normalize_threads(value)
    except ValueError as exc:
        logger.warning("Ignoring the translator thread setting: %s -- using 'auto'", exc)
        wanted = AUTO_THREADS

    if wanted != _mtl_threads:
        logger.info("Translator threads — %s", wanted)

    _mtl_threads = wanted

    return _mtl_threads


def thread_preference() -> str | int:
    """The live thread preference: `auto`, or a pinned count.

    A function rather than a module global read directly, because unlike
    `FOX_DEVICE` this value cannot be updated in place — a caller that had done
    ``from fox_reader.device import _mtl_threads`` would keep seeing whatever was
    set when it was imported.
    """
    return _mtl_threads


def thread_count(
    *,
    on_gpu: bool = False,
    reserve: bool = True,
    preference: Any = None,
) -> int:
    """How many threads to give llama.cpp.

    In precedence order: MTL_THREADS_ENV, then the saved preference, then the
    heuristic — physical cores, minus a slice kept back for the rest of the
    application, capped where more threads stop helping.

    A pinned count is still clamped to the CPUs this process may use, since
    threads beyond that cannot run and only add contention, but it ignores
    `reserve`: an explicit number is an instruction, not a starting point.
    `reserve` is False for prompt evaluation, which is compute-bound and does
    scale with cores, and which finishes fast enough that briefly using the
    whole machine is not felt.

    `preference` defaults to the live one set by `configure_threads`; pass
    AUTO_THREADS to ask what `auto` would decide, or a number to preview one.
    """
    usable = usable_cpus()
    pinned = thread_override()
    source = MTL_THREADS_ENV

    if pinned is None:
        wanted = thread_preference() if preference is None else preference

        try:
            wanted = normalize_threads(wanted)
        except ValueError as exc:
            logger.warning("Ignoring the translator thread setting: %s", exc)
            wanted = AUTO_THREADS

        if wanted != AUTO_THREADS:
            pinned = int(wanted)
            source = "the translator thread setting"

    if pinned is not None:
        threads = min(pinned, usable)

        if threads != pinned:
            logger.warning(
                "%s=%d, but only %d CPU(s) are available to this process; using %d",
                source,
                pinned,
                usable,
                threads,
            )

        return threads

    physical = physical_cores(usable)
    cap = MAX_GPU_THREADS if on_gpu else MAX_CPU_THREADS

    return max(1, min(physical - (reserved_cores(physical) if reserve else 0), cap))


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------


def normalize_id(value: Any) -> str:
    """Parse a device id, rejecting anything that is not one.

    Grammar only — whether the device exists is a question for `resolve`, so a
    settings file written on a two-GPU machine still loads on a laptop.
    """
    text = str(value or "").strip().lower()

    if not text or text == AUTO:
        return AUTO

    if text in {CPU, MPS}:
        return text

    if text in {CUDA, "gpu"}:
        return f"{CUDA}:0"

    for prefix in (f"{CUDA}:", "gpu:"):
        if text.startswith(prefix):
            tail = text[len(prefix):]

            if tail.isdigit():
                return f"{CUDA}:{int(tail)}"

            break

    if text.isdigit():
        return f"{CUDA}:{int(text)}"

    raise ValueError(
        f"Unknown device: {value!r}. Expected 'auto', 'cpu', 'mps' or 'cuda:N'."
    )


def _cuda_index(device_id: str) -> int | None:
    if device_id.startswith(f"{CUDA}:"):
        return int(device_id.split(":", 1)[1])

    return None


def _memory_label(total: int | None) -> str:
    if not total:
        return ""

    return f" · {total / _BYTES_PER_GIB:.1f} GB"


def options(refresh: bool = False) -> list[DeviceOption]:
    """Everything the picker on the settings page should offer."""
    hardware = probe(refresh=refresh)
    best = hardware.best()

    best_name = CPU.upper()

    if (index := _cuda_index(best)) is not None:
        gpu = hardware.gpu(index)
        best_name = gpu.name if gpu else best
    elif best == MPS:
        best_name = "Apple GPU"

    found: list[DeviceOption] = [
        DeviceOption(
            id=AUTO,
            kind=AUTO,
            name="Automatic",
            label=f"Auto — {best_name}",
        ),
        DeviceOption(
            id=CPU,
            kind=CPU,
            name="CPU",
            label="CPU",
        ),
    ]

    for gpu in hardware.gpus:
        found.append(
            DeviceOption(
                id=f"{CUDA}:{gpu.index}",
                kind=CUDA,
                index=gpu.index,
                name=gpu.name,
                label=f"GPU {gpu.index} — {gpu.name}{_memory_label(gpu.total_memory)}",
                total_memory=gpu.total_memory,
            )
        )

    if hardware.mps:
        found.append(
            DeviceOption(
                id=MPS,
                kind=MPS,
                name="Apple Metal Performance Shaders",
                label="GPU — Apple Metal (MPS)",
            )
        )

    return found


# ---------------------------------------------------------------------------
# per-slot spelling
# ---------------------------------------------------------------------------


def _for_paddleocr(device_id: str) -> str:
    """PaddleOCR's own spelling: ``gpu:0``, or ``cpu``.

    Metal is not one of the backends PaddleOCR can target, so an Apple GPU
    lands on the CPU here even when the other slots use it.
    """
    if (index := _cuda_index(device_id)) is not None:
        return f"gpu:{index}"

    return CPU


def _for_bubble(device_id: str) -> str:
    """A torch device string for the bubble segmentation network.

    Same spelling as :func:`_for_translator` — torch understands ``cuda:N``,
    ``mps`` and ``cpu`` directly — kept as its own function so the two slots
    can diverge later without re-reading every call site.
    """
    if (index := _cuda_index(device_id)) is not None:
        return f"{CUDA}:{index}"

    if device_id == MPS:
        return MPS

    return CPU


def _for_translator(device_id: str) -> str:
    """A torch device string (``cuda:N`` / ``mps`` / ``cpu``)."""
    if (index := _cuda_index(device_id)) is not None:
        return f"{CUDA}:{index}"

    if device_id == MPS:
        return MPS

    return CPU


def _for_paddleocr_vl(device_id: str) -> str:
    """A torch-style device string for the PaddleOCR-VL transcriber.

    Same spelling as :func:`_for_translator` — llama.cpp takes ``cuda:N``,
    ``mps`` and ``cpu`` — kept as its own function so the two GGUF consumers
    can diverge later without re-reading every call site.
    """
    if (index := _cuda_index(device_id)) is not None:
        return f"{CUDA}:{index}"

    if device_id == MPS:
        return MPS

    return CPU


def _for_textseg(device_id: str) -> str:
    """A torch device string for the Text Seg cleaning network.

    Same spelling as :func:`_for_translator` — torch understands ``cuda:N``,
    ``mps`` and ``cpu`` directly — kept as its own function so the two slots
    can diverge later (e.g. if one backend ever drops Metal) without
    re-reading every call site.
    """
    if (index := _cuda_index(device_id)) is not None:
        return f"{CUDA}:{index}"

    if device_id == MPS:
        return MPS

    return CPU


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


class Resolution(BaseModel):
    """A resolved selection, and what had to be changed to get there."""

    device: FoxDevice
    #: Requested id per slot, `auto` included, exactly as saved.
    selection: dict[str, str] = Field(default_factory=dict)
    #: What each slot actually resolved to, after `auto` and after fallbacks.
    resolved: dict[str, str] = Field(default_factory=dict)
    #: One human-readable line per slot that did not get what it asked for.
    fallbacks: list[str] = Field(default_factory=list)


def _available_ids(hardware: _Hardware) -> set[str]:
    ids = {CPU}
    ids.update(f"{CUDA}:{gpu.index}" for gpu in hardware.gpus)

    if hardware.mps:
        ids.add(MPS)

    return ids


def resolve(
    selection: dict[str, Any] | None = None,
    *,
    refresh: bool = False,
) -> Resolution:
    """Turn a saved selection into the devices to actually use.

    A slot asking for hardware that is not here is answered with the CPU and a
    line in `fallbacks`. The saved preference is left alone on purpose: pulling
    a card out for an afternoon should not silently rewrite the config.

    A selection that predates the ``textseg`` slot (no ``textseg`` key at all)
    inherits the ``paddleocr`` choice, which is what Text Seg used to run on.
    An explicit ``textseg`` — including ``auto`` once it has been saved — is
    always honoured on its own. Without this, upgrading would silently move an
    existing install's cleaning pass onto the best GPU even when the user had
    pinned OCR to the CPU to free VRAM.

    The same holds for ``paddleocr_vl``: it used to follow the ``paddleocr``
    slot (see ``PaddleOCRVLEngine``), so a missing key inherits that choice
    rather than resetting to ``auto`` behind the user's back.
    """
    hardware = probe(refresh=refresh)
    available = _available_ids(hardware)
    best = hardware.best()

    requested: dict[str, str] = {}
    resolved: dict[str, str] = {}
    fallbacks: list[str] = []

    saved = dict(selection or {})
    if "bubble" not in saved and "yolo" in saved:
        # Pre-rename device key: honour a saved `yolo` choice once instead of
        # silently resetting a pinned bubble device back to `auto`.
        saved["bubble"] = saved.get("yolo")
    if "textseg" not in saved and "paddleocr" in saved:
        # `.get` on purpose: a caller passing {"paddleocr": None} means the
        # same as an absent value (normalize_id turns both into `auto`).
        saved["textseg"] = saved.get("paddleocr")
    if "paddleocr_vl" not in saved and "paddleocr" in saved:
        saved["paddleocr_vl"] = saved.get("paddleocr")

    for slot in SLOTS:
        raw = saved.get(slot, AUTO)

        try:
            device_id = normalize_id(raw)
        except ValueError as exc:
            logger.warning("Ignoring the %s device setting: %s", slot, exc)
            device_id = AUTO

        requested[slot] = device_id
        wanted = best if device_id == AUTO else device_id

        if wanted not in available:
            # `auto` resolving to something absent would mean the probe
            # contradicted itself, so only a pinned choice should get here.
            fallbacks.append(
                f"{slot}: {wanted} is not available on this machine, using the CPU"
            )
            wanted = CPU

        resolved[slot] = wanted

    paddle = _for_paddleocr(resolved["paddleocr"])

    if resolved["paddleocr"] == MPS:
        fallbacks.append(
            "paddleocr: PaddleOCR cannot use the Apple GPU, using the CPU"
        )

    bubble = _for_bubble(resolved["bubble"])

    best_gpu_name: str | None = None

    if (index := _cuda_index(best)) is not None:
        gpu = hardware.gpu(index)
        best_gpu_name = gpu.name if gpu else best
    elif best == MPS:
        best_gpu_name = "Apple Metal Performance Shaders"

    device = FoxDevice(
        translator=_for_translator(resolved["translator"]),
        paddleocr=paddle,
        paddleocr_vl=_for_paddleocr_vl(resolved["paddleocr_vl"]),
        bubble=bubble,
        textseg=_for_textseg(resolved["textseg"]),
        # A label for the machine, not for any one slot: CUDA is reported
        # whenever a card is present, even if every slot is pinned to the CPU.
        backend=CUDA if hardware.cuda_available else (MPS if hardware.mps else CPU),
        cuda_available=hardware.cuda_available,
        cuda_version=hardware.cuda_version,
        gpu_name=best_gpu_name,
        device_count=len(hardware.gpus),
    )

    for line in fallbacks:
        logger.warning("Device fallback — %s", line)

    return Resolution(
        device=device,
        selection=requested,
        resolved=resolved,
        fallbacks=fallbacks,
    )


def detect_best_device() -> FoxDevice:
    """The devices to use when nothing has been configured."""
    return resolve().device


#: The live configuration. Every consumer reads this, so `configure` updates it
#: in place rather than rebinding the name — a module that already did
#: `from fox_reader.device import FOX_DEVICE` keeps seeing the current values.
FOX_DEVICE: FoxDevice = detect_best_device()


def configure(
    selection: dict[str, Any] | None = None,
    *,
    refresh: bool = False,
) -> Resolution:
    """Apply a saved selection to `FOX_DEVICE`.

    Called once at startup, before any model is built. Changing a device after
    that means a restart: the OCR engines and the translator bind their device
    when they load.
    """
    resolution = resolve(selection, refresh=refresh)

    for field, value in resolution.device.model_dump().items():
        setattr(FOX_DEVICE, field, value)

    logger.info(
        "Devices — paddleocr=%s paddleocr_vl=%s bubble=%s translator=%s textseg=%s (backend=%s, %d GPU(s))",
        FOX_DEVICE.paddleocr,
        getattr(FOX_DEVICE, "paddleocr_vl", CPU),
        FOX_DEVICE.bubble,
        FOX_DEVICE.translator,
        getattr(FOX_DEVICE, "textseg", CPU),
        FOX_DEVICE.backend,
        FOX_DEVICE.device_count,
    )

    return resolution


def snapshot(
    selection: dict[str, Any] | None = None,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """What the settings page needs to draw the device section."""
    resolution = resolve(selection, refresh=refresh)

    return {
        "options": [option.model_dump() for option in options(refresh=False)],
        "slots": list(SLOTS),
        "selection": resolution.selection,
        "resolved": resolution.resolved,
        "fallbacks": resolution.fallbacks,
        # `getattr` with a default on purpose: a snapshot taken while the
        # process still holds a pre-textseg FOX_DEVICE (hot-reload, old
        # pickle) answers with the CPU rather than raising.
        "active": {slot: getattr(FOX_DEVICE, slot, CPU) for slot in SLOTS},
        "backend": FOX_DEVICE.backend,
        "cuda_available": FOX_DEVICE.cuda_available,
        "cuda_version": FOX_DEVICE.cuda_version,
        "gpu_name": FOX_DEVICE.gpu_name,
        "device_count": FOX_DEVICE.device_count,
    }
