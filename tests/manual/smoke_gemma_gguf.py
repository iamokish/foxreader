"""Exercise the GGUF translator without llama.cpp or 7.7 GiB of weights.

Everything worth checking in `gemma_e4b_q8_llamacpp` is decided before a single
tensor is read: how many threads to ask for, which device keywords to pass, which
file to hand over, and what to do with the text that comes back. So `llama_cpp`
is stubbed with a class that records its keyword arguments, and the weights are
an empty file with the right name.

What this is protecting:
  * importing the module must not need llama-cpp-python -- local_mtl imports it
    unconditionally, and most installs will not have the wheel
  * the thread count must leave something for the rest of the process
  * a numbered CUDA device must not silently spill onto card 0
  * a GPU load that fails must not fall back to the CPU -- on a GGUF that
    fallback is a 7 GiB system-RAM allocation, and it can take the process down
    before Python sees an exception
  * a CPU-only llama.cpp asked for a GPU must be refused at load, which is the
    runtime half of what `packaging/llamacpp.py`'s `verify` checks at build time
  * unload() must actually close the native context

The thread budget itself now lives in `fox_reader.device` -- the settings page
offers it as a choice, and drawing that picker must not mean importing a
translator. It is still exercised here, because this is the only model that reads
it, and the count it passes to llama.cpp is what the choice comes to.

There are two of these translators, Q8 and Q6, and they are near-identical files.
Q8 is the one exercised in full; the last section pins the twin to it, because a
copied module is one that gets fixed in one copy.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import shutil
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if condition:
        print(f"ok   {label}")
        return

    failures += 1
    print(f"FAIL {label}" + (f" -- {detail}" if detail else ""))


# --- importing it must cost nothing -----------------------------------------

print("--- the module imports without llama-cpp-python ---")

check("llama_cpp is not installed here", "llama_cpp" not in sys.modules)

from fox_reader.constants import (  # noqa: E402
    GEMMA_E4B_Q6_MODEL_DIR,
    GEMMA_E4B_Q6_MODEL_FILE,
    GEMMA_E4B_Q8_MODEL_DIR,
    GEMMA_E4B_Q8_MODEL_FILE,
    MTL_MODELS,
)
from fox_reader.translate.machine_translation import gemma_e4b_q6_llamacpp as gemma_q6  # noqa: E402
from fox_reader.translate.machine_translation import gemma_e4b_q8_llamacpp as gemma  # noqa: E402
from fox_reader import device as fox_device  # noqa: E402

Q8_ID = "gemma-4-e4b-q8-uncensored"
Q6_ID = "gemma-4-e4b-q6-uncensored"

check("the module imported anyway", gemma is not None)
check("and did not drag llama_cpp in with it", "llama_cpp" not in sys.modules)
check("model_id works with the package absent", gemma.GemmaE4BQ8Translator.model_id() == Q8_ID,
      gemma.GemmaE4BQ8Translator.model_id())
check(
    "so do the languages",
    gemma.GemmaE4BQ8Translator.model_langs() == ["japanese", "chinese", "korean"],
    str(gemma.GemmaE4BQ8Translator.model_langs()),
)

print("\n--- local_mtl registers it ---")


def _stub(name: str, **attrs) -> None:
    """A stand-in for a heavy optional dependency, if it is not really here."""
    if name in sys.modules:
        return

    module = types.ModuleType(name)

    for attr, value in attrs.items():
        setattr(module, attr, value)

    sys.modules[name] = module


# The GGUF translators import llama.cpp lazily inside the constructor, so this
# section runs even where that wheel is absent. Torch is stubbed just far
# enough to get through the device probe.
_stub(
    "torch",
    cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    version=types.SimpleNamespace(cuda=None),
    inference_mode=lambda *a, **k: (lambda fn: fn),
)

try:
    from fox_reader.translate import local_mtl  # noqa: E402

    imported = True
except Exception as exc:  # noqa: BLE001
    print(f"skip local_mtl checks -- {type(exc).__name__}: {exc}")
    local_mtl = None
    imported = False

check("local_mtl imports with the gguf translators registered", imported)

if imported:
    manager = local_mtl.LocalMTLManager(pathlib.Path("."))

    ids = [cls.model_id() for cls in manager.available_translators]
    check("the Q8 translator is registered", Q8_ID in ids, str(ids))
    check("and so is the Q6", Q6_ID in ids, str(ids))
    check("the two gguf models are registered last, matching MTL_MODELS order",
          ids[-2:] == [Q8_ID, Q6_ID], str(ids))
    check("both are offered for all three languages", all(
        model_id in [c.model_id() for c in manager._compatible_translators(lang)]
        for model_id in (Q8_ID, Q6_ID)
        for lang in ("japanese", "korean", "chinese")
    ))
    check(
        "and neither is remapped off MPS",
        all(local_mtl.effective_device(model_id, "mps") == "mps" for model_id in (Q8_ID, Q6_ID)),
        str([local_mtl.effective_device(model_id, "mps") for model_id in (Q8_ID, Q6_ID)]),
    )

    print("\n--- unloading goes through the translator's own hook ---")

    class OwnsNativeMemory:
        """Shaped like the gguf translator: a `.model`, and an unload of its own."""

        def __init__(self):
            self.model = object()
            self.tokenizer_deleted = False
            self.unloaded = False

        def unload(self):
            self.unloaded = True
            self.model = None

        @classmethod
        def model_id(cls):
            return "owns-native-memory"

    manager.model = OwnsNativeMemory()
    owner = manager.model
    manager.unload()

    check("the hook was called", owner.unloaded is True)
    check("and the manager let go", manager.model is None)

    class WithoutHook:
        """No unload of its own: the manager simply lets go of it."""

        def __init__(self):
            self.model = object()

    legacy = WithoutHook()
    manager.model = legacy
    manager.unload()

    check("a model without a hook is still released", manager.model is None)


# --- the metadata entry ------------------------------------------------------

print("\n--- the MTL_MODELS entries ---")

by_id = {model["id"]: model for model in MTL_MODELS}
entry = by_id.get(Q8_ID)
q6_entry = by_id.get(Q6_ID)

check("the Q8 entry exists", entry is not None)
check("the Q6 entry exists", q6_entry is not None)

if entry and q6_entry:
    check("they are the last two, so nothing else was displaced",
          MTL_MODELS[-2:] == [entry, q6_entry], str([m["id"] for m in MTL_MODELS]))
    check("the repo is right", entry["repo"] == "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
          entry["repo"])
    check("both quants come from the one repo", q6_entry["repo"] == entry["repo"], q6_entry["repo"])
    check("the dest matches the constant", entry["dest"] == GEMMA_E4B_Q8_MODEL_DIR, entry["dest"])
    check("only the one quant is downloaded", entry["files"] == [GEMMA_E4B_Q8_MODEL_FILE], str(entry["files"]))
    check("and the Q6 downloads only its own", q6_entry["files"] == [GEMMA_E4B_Q6_MODEL_FILE],
          str(q6_entry["files"]))
    check("the type is gguf", entry["type"] == "gguf", entry["type"])
    check("for both of them", q6_entry["type"] == "gguf", q6_entry["type"])
    check("the disk figure is the given one", entry["size_on_disk"] == "7.7 GiB", entry["size_on_disk"])
    check("the ram figure is the given one", entry["cpu_ram_mib"] == 7000, str(entry["cpu_ram_mib"]))
    check("the vram figure is the given one", entry["gpu_vram_mib"] == 6800, str(entry["gpu_vram_mib"]))
    check("the display strings were derived", entry["cpu_ram"] == "6.8 GiB" and entry["gpu_vram"] == "6.6 GiB",
          str((entry["cpu_ram"], entry["gpu_vram"])))
    check("the Q6 is the smaller of the two, which is the only reason to offer it",
          q6_entry["cpu_ram_mib"] < entry["cpu_ram_mib"]
          and q6_entry["gpu_vram_mib"] < entry["gpu_vram_mib"],
          str((q6_entry["cpu_ram_mib"], q6_entry["gpu_vram_mib"])))
    check("every id is still unique", len({m["id"] for m in MTL_MODELS}) == len(MTL_MODELS))
    check("every dest is still unique", len({m["dest"] for m in MTL_MODELS}) == len(MTL_MODELS))
    check("and the two quants do not share a download directory",
          GEMMA_E4B_Q8_MODEL_DIR != GEMMA_E4B_Q6_MODEL_DIR, GEMMA_E4B_Q8_MODEL_DIR)


# --- the thread heuristic ----------------------------------------------------

print("\n--- how many threads ---")


@contextlib.contextmanager
def fake_cpus(usable: int, physical: int | None):
    """Pretend this machine has `usable` logical and `physical` real cores.

    Patched on `fox_reader.device` rather than on the translator: that is where
    the topology is read now. `thread_count` imported into this module by value
    still sees it, because its body resolves these names from device's globals.
    """
    real_usable = fox_device.usable_cpus
    real_physical = fox_device.physical_cores

    fox_device.usable_cpus = lambda: usable
    # Callable both ways, since routes/settings.py asks for the count bare.
    fox_device.physical_cores = (
        (lambda u=None: min(physical, usable if u is None else u))
        if physical is not None
        else real_physical
    )

    try:
        yield
    finally:
        fox_device.usable_cpus = real_usable
        fox_device.physical_cores = real_physical


@contextlib.contextmanager
def no_psutil():
    """Make the physical-core probe fail, the way a trimmed install would."""
    real = fox_device._psutil
    fox_device._psutil = lambda: None

    try:
        yield
    finally:
        fox_device._psutil = real


@contextlib.contextmanager
def env(**values: str | None):
    previous = {name: os.environ.get(name) for name in values}

    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


threads = fox_device.thread_count

# Nothing has configured a preference in this process, but say so rather than
# assume it: every figure below is the heuristic's, not a pinned one.
fox_device.configure_threads(fox_device.AUTO_THREADS)

with env(**{fox_device.MTL_THREADS_ENV: None}):
    check("a real machine gets at least one thread", threads() >= 1)
    check(
        "and never more than the cap",
        threads() <= fox_device.MAX_CPU_THREADS,
        str(threads()),
    )

    with fake_cpus(1, 1):
        check("one core still yields one thread", threads() == 1)

    with fake_cpus(4, 2):
        check("a dual-core keeps nothing back", threads() == 2)

    with fake_cpus(8, 4):
        check("a quad-core gives one core back", threads() == 3)
        check("prompt eval gets it again", threads(reserve=False) == 4)

    with fake_cpus(16, 8):
        check("an eight-core keeps two back", threads() == 6)
        check(
            "and never counts hyperthreads",
            threads() < 16,
            str(threads()),
        )

    with fake_cpus(64, 32):
        check("a big machine stops at the cap", threads() == fox_device.MAX_CPU_THREADS)
        check(
            "and lower still on a GPU, where the CPU is idle anyway",
            threads(on_gpu=True) == fox_device.MAX_GPU_THREADS,
            str(threads(on_gpu=True)),
        )

    with fake_cpus(16, None), no_psutil():
        # No physical count available: the fallback halves the logical figure
        # rather than assuming every logical CPU is a core of its own.
        check("without psutil it assumes SMT", threads() == 6, str(threads()))

    with fake_cpus(2, None), no_psutil():
        check("and does not halve a two-CPU machine to nothing", threads() == 2)

print("\n--- the saved preference ---")

with env(**{fox_device.MTL_THREADS_ENV: None}), fake_cpus(16, 8):
    check("auto is the default", fox_device.thread_preference() == fox_device.AUTO_THREADS,
          str(fox_device.thread_preference()))

    check("a pinned count is taken as asked", threads(preference=3) == 3)
    check("even above what auto would allow", threads(preference=12) == 12)
    check("but not past the CPUs this process has", threads(preference=99) == 16)
    check("and a pin ignores the GPU cap too", threads(on_gpu=True, preference=10) == 10)

    check("configure_threads reports what it stored", fox_device.configure_threads(5) == 5)
    check("and the preference is what the model will read", threads() == 5)
    check("a string count is a count", fox_device.configure_threads("7") == 7)
    check("and it is stored as an int, for the yaml", threads() == 7)

    for junk in ("auto", None, "", "   ", "AUTO"):
        check(f"{junk!r} means auto", fox_device.configure_threads(junk) == fox_device.AUTO_THREADS, str(junk))
        check("so the heuristic decides again", threads() == 6, str(junk))

    # Startup is lenient: an unreadable settings.yaml value must not stop a
    # model from loading, so it falls back to the heuristic rather than to
    # whatever happened to be set before -- which nobody chose either.
    for bad in ("0", "-2", "many", "3.5", True, [4]):
        fox_device.configure_threads(4)
        check(
            f"{bad!r} falls back to auto rather than stopping the load",
            fox_device.configure_threads(bad) == fox_device.AUTO_THREADS,
            str(fox_device.thread_preference()),
        )
        check("so the heuristic decides", threads() == 6, str(bad))

    # The lenient door and the strict one: the settings page must reject garbage
    # rather than shrug at it, which is `normalize_threads` rather than
    # `configure_threads`.
    for bad in ("0", "-2", "many", "3.5", True, [4], 0):
        try:
            fox_device.normalize_threads(bad)
            check(f"normalize_threads({bad!r}) raises", False, "accepted it")
        except ValueError:
            check(f"normalize_threads({bad!r}) raises", True)

    check("and it passes a good count straight through", fox_device.normalize_threads(6) == 6)
    check("as an int, whatever it arrived as", fox_device.normalize_threads(" 6 ") == 6)

    fox_device.configure_threads(fox_device.AUTO_THREADS)

print("\n--- the override ---")

with fake_cpus(16, 8):
    with env(**{fox_device.MTL_THREADS_ENV: "12"}):
        check("an override wins", threads() == 12)
        check("and says so, so the page can", fox_device.thread_override() == 12)
        check("over a saved count as well", threads(preference=3) == 12)

    with env(**{fox_device.MTL_THREADS_ENV: "999"}):
        check("but not past what can be scheduled", threads() == 16)

    for bad in ("0", "-4", "many", "", "   "):
        with env(**{fox_device.MTL_THREADS_ENV: bad}):
            check(f"{bad!r} is ignored, not obeyed", threads() == 6, str(bad))
            check("and the page is not told there is one", fox_device.thread_override() is None, str(bad))
            # The setting still applies: a junk override must not swallow it.
            check("a saved count still counts", threads(preference=3) == 3, str(bad))


# --- device placement --------------------------------------------------------

print("\n--- device placement ---")

llama_cpp_stub = types.SimpleNamespace(LLAMA_SPLIT_MODE_NONE=0)

cpu_kwargs = gemma.llama_device_kwargs("cpu", llama_cpp_stub)
check("the cpu gets no offload", cpu_kwargs == {"n_gpu_layers": 0}, str(cpu_kwargs))

for spelling in ("", "   ", None):
    check(
        f"{spelling!r} falls back to the cpu",
        gemma.llama_device_kwargs(spelling, llama_cpp_stub)["n_gpu_layers"] == 0,
    )

mps_kwargs = gemma.llama_device_kwargs("mps", llama_cpp_stub)
check("metal gets every layer", mps_kwargs["n_gpu_layers"] == -1, str(mps_kwargs))
check("and no ordinal", "main_gpu" not in mps_kwargs, str(mps_kwargs))

# Bare `cuda` and `cuda:0` both mean "the default card", and saying so with
# main_gpu=0 and a split mode would be llama.cpp's own default spelled out --
# which is a placement argument that can be wrong on a version that renamed the
# enum. So a single-card request passes nothing but the offload.
cuda0 = gemma.llama_device_kwargs("cuda", llama_cpp_stub)
check("bare cuda offloads everything", cuda0 == {"n_gpu_layers": -1}, str(cuda0))
check("and cuda:0 is the same request", gemma.llama_device_kwargs("cuda:0", llama_cpp_stub) == cuda0,
      str(gemma.llama_device_kwargs("cuda:0", llama_cpp_stub)))

cuda2 = gemma.llama_device_kwargs("CUDA:2", llama_cpp_stub)
check("a numbered card is honoured", cuda2["main_gpu"] == 2, str(cuda2))
check("with every layer on it", cuda2["n_gpu_layers"] == -1, str(cuda2))
check(
    "and pinned there rather than split across every card",
    cuda2["split_mode"] == 0,
    str(cuda2),
)

check(
    "a version without the constant still gets a number",
    gemma.llama_device_kwargs("cuda:1", types.SimpleNamespace())["split_mode"] == 0,
)
check(
    "the old constant name is accepted too",
    gemma.llama_device_kwargs("cuda:1", types.SimpleNamespace(LLAMA_SPLIT_NONE=0))["split_mode"] == 0,
)
check(
    "and nothing is passed when llama_cpp was not handed over",
    "split_mode" not in gemma.llama_device_kwargs("cuda:1"),
)
check(
    "though the card is still chosen",
    gemma.llama_device_kwargs("cuda:1")["main_gpu"] == 1,
    str(gemma.llama_device_kwargs("cuda:1")),
)
with contextlib.redirect_stderr(io.StringIO()):
    nonsense = gemma.llama_device_kwargs("cuda:x", llama_cpp_stub)
check("a nonsense device is not read as cuda", nonsense["n_gpu_layers"] == 0, str(nonsense))


# --- the build capability gate -----------------------------------------------

print("\n--- a CPU-only build asked for a GPU ---")

# The runtime half of what packaging/llamacpp.py's `verify` checks at build time:
# a CUDA distribution that shipped a CPU-only libllama. Refusing here is the
# whole point -- offloading nothing means the GGUF lands in system RAM.
cpu_only = types.SimpleNamespace(llama_supports_gpu_offload=lambda: False)

with contextlib.redirect_stderr(io.StringIO()):
    check("a cpu request is fine on a cpu-only build",
          gemma._validate_requested_device("cpu", cpu_only) is None)

    try:
        gemma._validate_requested_device("cuda", cpu_only)
        check("a gpu request is refused", False, "accepted it")
    except RuntimeError as exc:
        check("a gpu request is refused", True)
        check("and the message says the build is the problem", "CPU-only" in str(exc), str(exc))

    check("a build that cannot answer is not second-guessed",
          gemma._validate_requested_device("cuda", types.SimpleNamespace()) is None)
    check("and one that says yes is believed",
          gemma._validate_requested_device(
              "cuda", types.SimpleNamespace(llama_supports_gpu_offload=lambda: True)) is None)
    check("a probe that raises reads as unknown rather than as no",
          gemma.supports_gpu_offload(
              types.SimpleNamespace(llama_supports_gpu_offload=lambda: 1 / 0)) is None)


# --- finding the weights -----------------------------------------------------

print("\n--- finding the weights ---")

TMP = pathlib.Path(tempfile.mkdtemp(prefix="fox-gemma-"))


def mtl_dir(name: str, *files: str) -> pathlib.Path:
    root = TMP / name / GEMMA_E4B_Q8_MODEL_DIR
    root.mkdir(parents=True, exist_ok=True)

    for relative in files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"GGUF")

    return TMP / name


declared = mtl_dir("declared", GEMMA_E4B_Q8_MODEL_FILE)
check(
    "the declared filename is preferred",
    gemma.resolve_weights(declared).name == GEMMA_E4B_Q8_MODEL_FILE,
    str(gemma.resolve_weights(declared)),
)

renamed = mtl_dir("renamed", "some-other-quant.gguf")
check(
    "a renamed quant is still found",
    gemma.resolve_weights(renamed).name == "some-other-quant.gguf",
    str(gemma.resolve_weights(renamed)),
)

nested = mtl_dir("nested", "Q8_K_P/weights.gguf")
check(
    "so is one in a subfolder, the way huggingface_hub lays it out",
    gemma.resolve_weights(nested).name == "weights.gguf",
    str(gemma.resolve_weights(nested)),
)

split = mtl_dir("split", "model-00002-of-00003.gguf", "model-00001-of-00003.gguf")
check(
    "a split weight returns the first part, which is the one llama.cpp wants",
    gemma.resolve_weights(split).name == "model-00001-of-00003.gguf",
    str(gemma.resolve_weights(split)),
)

both_present = mtl_dir("both", "aaa-other.gguf", GEMMA_E4B_Q8_MODEL_FILE)
check(
    "the declared name still wins over an alphabetically earlier file",
    gemma.resolve_weights(both_present).name == GEMMA_E4B_Q8_MODEL_FILE,
    str(gemma.resolve_weights(both_present)),
)

empty = mtl_dir("empty")

try:
    gemma.resolve_weights(empty)
    check("an empty directory raises", False)
except FileNotFoundError as exc:
    check("an empty directory raises FileNotFoundError", True)
    check("and says where it looked", GEMMA_E4B_Q8_MODEL_DIR.split("/")[-1] in str(exc), str(exc))
except Exception as exc:  # noqa: BLE001
    check("an empty directory raises FileNotFoundError", False, f"{type(exc).__name__}: {exc}")

try:
    gemma.resolve_weights(TMP / "does-not-exist")
    check("a missing directory raises", False)
except FileNotFoundError:
    check("a missing directory raises FileNotFoundError", True)
except Exception as exc:  # noqa: BLE001
    check("a missing directory raises FileNotFoundError", False, f"{type(exc).__name__}: {exc}")

# The two quants are separate downloads under separate names, so the Q6 loader
# must not find the Q8's file and load 7.7 GiB when 5.9 was asked for.
q6_root = TMP / "declared" / GEMMA_E4B_Q6_MODEL_DIR

try:
    gemma_q6.resolve_weights(declared)
    check("the Q6 loader does not pick up the Q8's weights", False, "found something")
except FileNotFoundError:
    check("the Q6 loader does not pick up the Q8's weights", True)
except Exception as exc:  # noqa: BLE001
    check("the Q6 loader does not pick up the Q8's weights", False, f"{type(exc).__name__}: {exc}")

q6_root.mkdir(parents=True, exist_ok=True)
(q6_root / GEMMA_E4B_Q6_MODEL_FILE).write_bytes(b"GGUF")
check(
    "and finds its own where it belongs",
    gemma_q6.resolve_weights(declared).name == GEMMA_E4B_Q6_MODEL_FILE,
    str(gemma_q6.resolve_weights(declared)),
)


# --- cleaning the output -----------------------------------------------------

print("\n--- cleaning the output ---")

check("plain text is untouched", gemma.clean_translation("Hello there") == "Hello there")
check("whitespace is trimmed", gemma.clean_translation("  Hello  \n") == "Hello")
check("a whole-message fence is unwrapped", gemma.clean_translation("```\nHello\n```") == "Hello")
check("even a labelled one", gemma.clean_translation("```text\nHello\n```") == "Hello")
check(
    "a multi-line body survives unwrapping",
    gemma.clean_translation("```\nOne\nTwo\n```") == "One\nTwo",
    repr(gemma.clean_translation("```\nOne\nTwo\n```")),
)
check(
    "a fence inside the text is left alone",
    gemma.clean_translation("He said ```run``` and left") == "He said ```run``` and left",
)
check(
    "quotes are kept, because dialogue has them",
    gemma.clean_translation('"Get out of here!"') == '"Get out of here!"',
)
check(
    "and a leading label is kept rather than guessed at",
    gemma.clean_translation("Translation: a sign") == "Translation: a sign",
)
check("empty stays empty", gemma.clean_translation("") == "")
check("and so does None", gemma.clean_translation(None) == "")

print("\n--- reading the response ---")

check(
    "the ordinary shape",
    gemma._message_content({"choices": [{"message": {"content": "Hi"}}]}) == "Hi",
)
check(
    "the content-parts shape",
    gemma._message_content(
        {"choices": [{"message": {"content": [{"type": "text", "text": "Hi"}]}}]}
    ) == "Hi",
)

for broken in ({}, {"choices": []}, {"choices": [{}]}, None, "nope", {"choices": [{"message": {}}]}):
    with contextlib.redirect_stderr(io.StringIO()):
        got = gemma._message_content(broken)

    check(f"{str(broken)[:28]!r} comes back empty rather than raising", got == "", repr(got))


# --- constructing it ---------------------------------------------------------

print("\n--- constructing it, against a stub ---")


class FakeLlama:
    """Records what it was built with; answers with a fixed completion."""

    instances: list[FakeLlama] = []
    fail_with: Exception | None = None

    def __init__(self, **kwargs):
        if FakeLlama.fail_with is not None:
            raise FakeLlama.fail_with

        self.kwargs = kwargs
        self.closed = False
        self.calls: list[dict] = []
        FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "```\nTranslated\n```"}}]}

    def close(self):
        self.closed = True


@contextlib.contextmanager
def stub_llama(**attrs):
    """Put a fake llama_cpp in sys.modules for the duration."""
    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama
    module.LLAMA_SPLIT_MODE_NONE = 0
    module.__version__ = "0.3.35"

    for name, value in attrs.items():
        setattr(module, name, value)

    previous = sys.modules.get("llama_cpp")
    sys.modules["llama_cpp"] = module

    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop("llama_cpp", None)
        else:
            sys.modules["llama_cpp"] = previous


with env(**{fox_device.MTL_THREADS_ENV: None, gemma.VERBOSE_ENV: None}), fake_cpus(16, 8):
    with stub_llama():
        translator = gemma.GemmaE4BQ8Translator(declared, device="cpu")
        built = translator.model.kwargs

        check("it found the weights", built["model_path"].endswith(GEMMA_E4B_Q8_MODEL_FILE),
              built["model_path"])
        check("the context window is the recorded one", built["n_ctx"] == gemma.N_CTX, str(built["n_ctx"]))
        check("it is quiet", built["verbose"] is False)
        check("no offload on the cpu", built["n_gpu_layers"] == 0, str(built["n_gpu_layers"]))
        check("generation leaves cores free", built["n_threads"] == 6, str(built["n_threads"]))
        check("and it knows it is not on a gpu", translator.on_gpu is False)

        out = translator.translate("こんにちは")
        check("it translates", out == "Translated", repr(out))

        sent = translator.model.calls[-1]
        check("the sampling settings are the given ones", (
            sent["top_k"] == 64
            and sent["temperature"] == 1.0
            and sent["top_p"] == 0.95
            and sent["max_tokens"] == gemma.MAX_NEW_TOKENS
        ), str({k: v for k, v in sent.items() if k != "messages"}))

        system, user = sent["messages"]
        check("there is a system turn", system["role"] == "system")
        check("it says not to censor", "Don't censor anything" in system["content"])
        check("it asks for English only", "Translate all user-provided text into English." in system["content"])
        check("it forbids markdown", "Do not wrap any text in markdown." in system["content"])
        check("and the text is the user turn", user == {"role": "user", "content": "こんにちは"})

        with_context = translator._build_system_prompt({"tone": "casual", "glossary": {"鬼": "oni"}})
        check("context becomes prompt lines", "Tone: casual" in with_context, with_context)
        check("and a glossary becomes a list", "- 鬼 -> oni" in with_context, with_context)
        check(
            "the closing instructions stay last",
            with_context.rstrip().endswith("Do not explain anything."),
            with_context[-60:],
        )

        check("empty text short-circuits", translator.translate("   ") == "")
        check("and did not reach the model", len(translator.model.calls) == 1)

        handle = translator.model
        translator.unload()
        check("unload closes the native context", handle.closed is True)
        check("and drops the reference", translator.model is None)
        translator.unload()
        check("unloading twice is harmless", True)

        try:
            translator.translate("text")
            check("translating after unload raises", False)
        except RuntimeError:
            check("translating after unload raises RuntimeError", True)

    print("\n--- and the verbose switch is honoured ---")

    with env(**{gemma.VERBOSE_ENV: "1"}), stub_llama():
        translator = gemma.GemmaE4BQ8Translator(declared, device="cpu")
        check("llama.cpp is told to talk", translator.model.kwargs["verbose"] is True)
        translator.unload()

    print("\n--- on a gpu ---")

    with stub_llama():
        translator = gemma.GemmaE4BQ8Translator(declared, device="cuda:1")
        built = translator.model.kwargs

        check("every layer is offloaded", built["n_gpu_layers"] == -1, str(built["n_gpu_layers"]))
        check("onto the card that was chosen", built["main_gpu"] == 1, str(built["main_gpu"]))
        check("and only that card", built["split_mode"] == 0, str(built["split_mode"]))
        check("fewer threads, since the cpu is only marshalling", built["n_threads"] == 4,
              str(built["n_threads"]))
        check("it knows it is on a gpu", translator.on_gpu is True)
        translator.unload()

    print("\n--- a gpu load that fails ---")

    FakeLlama.fail_with = RuntimeError("cudaMalloc failed: out of memory")
    before = len(FakeLlama.instances)

    with stub_llama():
        try:
            gemma.GemmaE4BQ8Translator(declared, device="cuda:0")
            check("it refuses rather than falling back", False, "constructed anyway")
        except RuntimeError as exc:
            check("it refuses rather than falling back", True)
            check("the message repeats the driver's", "out of memory" in str(exc), str(exc))
            check("and names the device that was asked for", "cuda:0" in str(exc), str(exc))
        except Exception as exc:  # noqa: BLE001
            check("it refuses rather than falling back", False, f"{type(exc).__name__}: {exc}")

        check("nothing was constructed on the cpu instead", len(FakeLlama.instances) == before,
              str(len(FakeLlama.instances)))

    print("\n--- a cpu load that fails ---")

    with stub_llama():
        try:
            gemma.GemmaE4BQ8Translator(declared, device="cpu")
            check("a cpu failure propagates", False, "constructed anyway")
        except RuntimeError as exc:
            check("a cpu failure propagates unwrapped", "cudaMalloc" in str(exc), str(exc))

    FakeLlama.fail_with = None

    print("\n--- a cpu-only build, asked for cuda ---")

    with stub_llama(llama_supports_gpu_offload=lambda: False):
        before = len(FakeLlama.instances)

        try:
            gemma.GemmaE4BQ8Translator(declared, device="cuda")
            check("the load is refused before the GGUF is allocated", False, "constructed anyway")
        except RuntimeError as exc:
            check("the load is refused before the GGUF is allocated", True)
            check("and the message says to install a GPU build", "llama-cpp-python" in str(exc), str(exc))

        check("so llama.cpp was never constructed", len(FakeLlama.instances) == before,
              str(len(FakeLlama.instances)))

    print("\n--- without the package ---")

    try:
        gemma.GemmaE4BQ8Translator(declared, device="cpu")
        check("a missing llama-cpp-python is reported", False, "constructed without it")
    except RuntimeError as exc:
        check("a missing llama-cpp-python is reported", True)
        check("and the message names the extra", "gguf" in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check("a missing llama-cpp-python is reported", False, f"{type(exc).__name__}: {exc}")

    print("\n--- with a count pinned on the settings page ---")

    # Last, because it constructs another model and the count above is asserted
    # on. This is the whole point of the setting: whatever the heuristic would
    # have said, the saved number is what llama.cpp is built with.
    fox_device.configure_threads(3)

    try:
        with stub_llama():
            translator = gemma.GemmaE4BQ8Translator(declared, device="cpu")
            built = translator.model.kwargs

            check("the saved count reaches llama.cpp", built["n_threads"] == 3, str(built["n_threads"]))
            translator.unload()
    finally:
        fox_device.configure_threads(fox_device.AUTO_THREADS)


# --- the twin ----------------------------------------------------------------

print("\n--- the Q6 module is still the same module ---")

# Two near-identical files, and the Q8 is the one exercised above. Anything fixed
# in one and not the other is a bug that only shows up on whichever quant the
# user picked, so the pair is pinned here rather than the Q6 being tested twice.
for name in (
    "llama_device_kwargs",
    "resolve_weights",
    "clean_translation",
    "supports_gpu_offload",
    "_validate_requested_device",
    "_message_content",
    "_cuda_index",
    "_split_mode_none",
    "N_CTX",
    "MAX_NEW_TOKENS",
    "VERBOSE_ENV",
):
    check(f"the Q6 module has {name} too", hasattr(gemma_q6, name))

for device in ("cpu", "cuda", "cuda:0", "cuda:2", "mps", "", None, "cuda:x"):
    with contextlib.redirect_stderr(io.StringIO()):
        same = gemma_q6.llama_device_kwargs(device, llama_cpp_stub) == \
            gemma.llama_device_kwargs(device, llama_cpp_stub)

    check(f"and places {device!r} identically", same)

check("the context window has not drifted", gemma_q6.N_CTX == gemma.N_CTX,
      f"q6={gemma_q6.N_CTX} q8={gemma.N_CTX}")
check("nor the token budget", gemma_q6.MAX_NEW_TOKENS == gemma.MAX_NEW_TOKENS,
      f"q6={gemma_q6.MAX_NEW_TOKENS} q8={gemma.MAX_NEW_TOKENS}")
check("they read the same verbose switch", gemma_q6.VERBOSE_ENV == gemma.VERBOSE_ENV)
check("the fence stripper agrees", all(
    gemma_q6.clean_translation(text) == gemma.clean_translation(text)
    for text in ("```\nHi\n```", "  Hi  ", "", None, "He said ```run```")
))
check("they are different models, though", gemma_q6.GemmaE4BQ6Translator.model_id() == Q6_ID,
      gemma_q6.GemmaE4BQ6Translator.model_id())
check("offered for the same languages",
      gemma_q6.GemmaE4BQ6Translator.model_langs() == gemma.GemmaE4BQ8Translator.model_langs())


shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{failures} FAILURE(S)" if failures else "\nthe gguf translator holds up")
sys.exit(1 if failures else 0)
