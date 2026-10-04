"""A payload to compile, not a script to run. Reports what embedded metadata answers.

Nuitka's `--include-distribution-metadata` compiles metadata records into the
binary, keyed by the name each distribution declares for itself, and reads them
back with an exact dict lookup -- nothing normalises. That is invisible from the
outside: no `.dist-info` is written into the standalone tree, so the filesystem
says nothing either way, and the only way to see what a *compiled* program is
handed is to ask a compiled program.

Hence this. Run through the interpreter it just answers whatever is installed in
the venv, which is never the question. Compile it and the answers change:

    .venv/Scripts/python -m nuitka --standalone \\
        --include-distribution-metadata=huggingface_hub \\
        --output-dir=<somewhere> tests/dist/probe_payload.py

A miss on `huggingface-hub` with a hit on `huggingface_hub` is the reported
startup crash, in two lines.
"""
import importlib.metadata as im

for name in ("huggingface_hub", "huggingface-hub", "hf_xet", "hf-xet", "transformers"):
    try:
        print(f"  {name:20} -> {im.version(name)}")
    except Exception as exc:
        print(f"  {name:20} -> {type(exc).__name__}")

print("  embedded dists:", sorted(
    (d.metadata["Name"] or "?") for d in im.distributions()
))
