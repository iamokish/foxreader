"""Static checks on packaging/launcher/launcher.c -- there is no C compiler here.

Not a substitute for compiling it. It catches the mechanical mistakes that a
single-translation-unit C file is prone to when it cannot be built: unbalanced
braces, an unmatched #if, a call to a function that was renamed on one side only,
C99 line comments where the file claims C89 declaration placement.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKSLASH = chr(92)


def strip_literals(text: str) -> str:
    """Comments, string and character literals replaced, so counting means something."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = end + 2 if end >= 0 else n
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            out.append("@LINECOMMENT@")
            i = end if end >= 0 else n
            continue
        if ch in ('"', "'"):
            quote, j = ch, i + 1
            while j < n and text[j] != quote:
                j += 2 if text[j] == BACKSLASH else 1
            out.append("@LIT@")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def main() -> int:
    path = Path(__file__).resolve().parent / "launcher" / "launcher.c"
    src = path.read_text(encoding="utf-8")
    code = strip_literals(src)
    failures: list[str] = []

    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        a, b = code.count(opener), code.count(closer)
        ok = a == b
        print(f"  {opener}{closer:<3} {a:5} / {b:<5} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"unbalanced {opener}{closer}")

    ifs = len(re.findall(r"^\s*#\s*(?:if|ifdef|ifndef)\b", src, re.M))
    endifs = len(re.findall(r"^\s*#\s*endif\b", src, re.M))
    print(f"  #if  {ifs:5} / {endifs:<5} {'ok' if ifs == endifs else 'MISMATCH'}")
    if ifs != endifs:
        failures.append("unbalanced #if/#endif")

    line_comments = code.count("@LINECOMMENT@")
    print(f"  //   {line_comments:5}         {'ok' if not line_comments else 'C99 only'}")
    if line_comments:
        failures.append(f"{line_comments} C99 line comment(s)")

    tabs = src.count("\t")
    non_ascii = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
    print(f"  tabs {tabs:5}         {'ok' if not tabs else 'present'}")
    print(f"  utf8 {len(non_ascii):5}         {'ok' if not non_ascii else 'non-ascii present'}")
    if non_ascii:
        # A stray non-ASCII byte in a source file compiled by both MSVC and gcc
        # is an encoding argument nobody wins.
        failures.append(f"non-ascii at offset {non_ascii[0][0]}")

    defined = set(re.findall(r"^static\s+[\w \*]+?\s*\**(\w+)\s*\(", src, re.M))
    macros = set(re.findall(r"^#\s*define\s+(\w+)", src, re.M))
    called = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", code))
    entry = {"main", "relay_thread", "console_handler", "signal_handler"}
    unused = sorted(defined - called - entry)
    print(f"  static functions defined: {len(defined)}")
    print(f"  macros defined:           {len(macros)}")
    print(f"  defined but never called: {', '.join(unused) if unused else 'none'}")
    if unused:
        failures.append(f"dead function(s): {', '.join(unused)}")

    if failures:
        print("\nFAILED: " + "; ".join(failures))
        return 1
    print("\nAll static checks passed. This is not a compile -- run packaging/toolchain.py")
    print("to build it for real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
