"""Directory paths, and what this machine will actually let us do with them.

Loading a folder now means committing to *two* directories -- the pages come from
one and the typeset results go to another -- so "is this path usable?" stopped
being a yes/no the caller could answer with ``os.path.isdir`` and became
something the picker has to explain to the user before they press Load.

The one non-obvious thing in here is the write test. ``os.access(path, os.W_OK)``
is documented as unreliable on Windows, and it is: it answers from the read-only
*attribute* and ignores NTFS ACLs entirely, so a directory the user cannot write
a byte to reports writable, and the failure only surfaces later as a half-saved
page. The only test that tells the truth on every platform is to create a file
and remove it again, which is what :func:`describe_dir` does.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from natsort import natsorted

logger = logging.getLogger(__name__)

#: Page extensions Fox Reader will open. Lives here rather than in the folder
#: route so that counting a directory's images and listing them for the viewer
#: can never disagree about what an image is.
IMAGE_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif",
)

#: What a destination gets called when the user does not name one.
DEFAULT_DEST_NAME = "fox_tled"


def normalize_dir(value: str) -> str:
    """The path cleanup the session has always done: unquote, forward slashes.

    Deliberately *not* absolutised -- see :func:`resolve_dir`. Pasting a quoted
    path is the common case (every file manager's "copy as path" adds them) and a
    Windows path arrives with backslashes, which this normalises so one stored
    form works on both platforms.
    """
    if not value:
        return ""
    return value.strip().strip("'").strip('"').strip().replace("\\", "/")


def resolve_dir(value: str) -> str:
    """`normalize_dir`, then expand ``~`` and make it absolute.

    Used at the edges -- the routes -- so that everything downstream compares and
    stores one spelling of a directory. The result keeps forward slashes, which
    is the form the session has always held and the form the picker shows back to
    the user; ``os.path.join`` and ``open`` accept it on Windows regardless.
    """
    cleaned = normalize_dir(value)
    if not cleaned:
        return ""
    try:
        expanded = os.path.expanduser(cleaned)
        return os.path.abspath(expanded).replace("\\", "/")
    except (OSError, ValueError):
        # A path the OS refuses to even parse (a bad drive spec, a null byte).
        # Hand back the cleaned form and let describe_dir report it as unusable.
        return cleaned


def list_images(directory: str) -> list[str]:
    """Page filenames in *directory*, in reading order. ``[]`` if unreadable.

    Dotfiles are skipped: a save in progress has a hidden temporary file beside
    the page it is writing, and that must never turn up as a page of its own.
    """
    try:
        names = [
            name for name in os.listdir(directory)
            if name.lower().endswith(IMAGE_EXTS) and not name.startswith(".")
        ]
    except OSError:
        return []
    return list(natsorted(names))


@dataclass(slots=True)
class DirReport:
    """What one directory is, and what we are permitted to do with it."""

    path: str
    exists: bool = False
    is_dir: bool = False
    is_file: bool = False

    #: Listing worked (POSIX read permission; on Windows, the ACL allows it).
    readable: bool = False
    #: A file was created and removed here. The only trustworthy write test.
    writable: bool = False
    #: Traversal into the directory. Meaningful on POSIX, always true elsewhere.
    executable: bool = False

    #: Page count, only filled in when asked for -- it costs a listdir.
    image_count: int | None = None

    parent: str = ""
    parent_exists: bool = False
    parent_writable: bool = False

    #: The first blocking problem, phrased for the picker to show verbatim.
    reason: str = ""

    def _replace_reason(self, text: str) -> None:
        if not self.reason:
            self.reason = text

    @property
    def can_create(self) -> bool:
        """Not there yet, but we could make it."""
        return not self.exists and self.parent_exists and self.parent_writable

    @property
    def ok_as_source(self) -> bool:
        return self.is_dir and self.readable

    @property
    def ok_as_dest(self) -> bool:
        return (self.is_dir and self.writable) or self.can_create

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["can_create"] = self.can_create
        data["ok_as_source"] = self.ok_as_source
        data["ok_as_dest"] = self.ok_as_dest
        return data


def _probe_writable(directory: str) -> bool:
    """Create a file in *directory* and remove it. True if both worked.

    The name is unique per call: two windows loading folders at the same moment
    must not collide on the probe and report each other's directory unwritable.
    """
    probe = os.path.join(directory, f".fox_write_probe.{uuid4().hex[:8]}")
    try:
        # "xb" rather than "wb": if something is already at this name we want to
        # hear about it instead of truncating it.
        with open(probe, "xb"):
            pass
    except OSError:
        return False

    try:
        os.unlink(probe)
    except OSError:
        # Writable but not cleanable is still writable; the stray probe file is
        # hidden and 0 bytes. Worth a log line, not a failure.
        logger.debug("Could not remove the write probe %s", probe)
    return True


def describe_dir(path: str, *, count_images: bool = False,
                 probe_write: bool = True) -> DirReport:
    """Everything the picker needs to say about one path.

    ``count_images`` costs a listing and is only wanted for the source.
    ``probe_write`` can be turned off where a write test would be pointless
    noise (reporting on a path that is a file, say).
    """
    resolved = resolve_dir(path)
    report = DirReport(path=resolved)

    if not resolved:
        report.reason = "Enter a folder path."
        return report

    report.parent = os.path.dirname(resolved.rstrip("/")) or resolved

    try:
        report.exists = os.path.exists(resolved)
        report.is_dir = os.path.isdir(resolved)
        report.is_file = os.path.isfile(resolved)
    except OSError as exc:
        report.reason = f"The system could not read this path: {exc.strerror or exc}"
        return report

    try:
        report.parent_exists = os.path.isdir(report.parent)
    except OSError:
        report.parent_exists = False

    if report.parent_exists and not report.is_dir and probe_write:
        report.parent_writable = _probe_writable(report.parent)

    if not report.exists:
        report._replace_reason(
            "This folder does not exist yet."
            if report.can_create
            else f"This folder does not exist, and {report.parent} cannot be written to."
        )
        return report

    if not report.is_dir:
        report.reason = "This is a file, not a folder."
        return report

    names: list[str] | None = None
    try:
        names = os.listdir(resolved)
        report.readable = True
    except OSError as exc:
        report.reason = f"This folder cannot be read: {exc.strerror or exc}"

    # On POSIX this is the difference between listing a directory and opening
    # anything inside it. On Windows there is no separate traverse bit, and
    # os.access reports X_OK for every directory, which is the right answer
    # there for the same reason.
    report.executable = os.access(resolved, os.X_OK)
    if report.readable and not report.executable:
        report._replace_reason("This folder cannot be opened (no execute permission).")

    if probe_write:
        report.writable = _probe_writable(resolved)
        if not report.writable:
            report._replace_reason("This folder cannot be written to.")

    if count_images:
        if names is None:
            report.image_count = 0
        else:
            report.image_count = sum(
                1 for name in names
                if name.lower().endswith(IMAGE_EXTS) and not name.startswith(".")
            )
            if report.image_count == 0 and report.readable:
                report._replace_reason("No images in this folder.")

    return report


def ensure_dir(path: str) -> DirReport:
    """Create *path* if it is missing, then describe it.

    Returns a report either way: a creation that fails is reported through the
    same shape as a directory that was never usable, so the caller has one code
    path for "cannot save here".
    """
    resolved = resolve_dir(path)
    if not resolved:
        return describe_dir(path)

    if not os.path.isdir(resolved):
        try:
            os.makedirs(resolved, exist_ok=True)
        except OSError as exc:
            report = describe_dir(resolved)
            report.reason = f"Could not create this folder: {exc.strerror or exc}"
            return report

    return describe_dir(resolved)


def _key(path: str) -> str:
    """A comparable spelling of *path*: symlinks followed, case folded on Windows."""
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return os.path.normcase(path)


def same_dir(a: str, b: str) -> bool:
    """Whether two paths name the same directory.

    ``samefile`` is the authoritative answer (it compares device and inode, so a
    junction, a bind mount or a symlink is caught) but it needs both paths to
    exist; the string comparison is the fallback for a destination that has not
    been created yet.
    """
    if not a or not b:
        return False
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return _key(a) == _key(b)


def contains(root: str, candidate: str) -> bool:
    """Whether *candidate* resolves to something inside *root*.

    The containment check behind ``/img_serve``: the filename in that URL comes
    from the browser, and joining it onto a directory unchecked is how
    ``../../../etc/passwd`` becomes a 200.
    """
    if not root or not candidate:
        return False
    root_key = _key(root)
    candidate_key = _key(candidate)
    if candidate_key == root_key:
        return True
    try:
        return os.path.commonpath([root_key, candidate_key]) == root_key
    except ValueError:
        # Different drives on Windows; commonpath refuses, and the answer is no.
        return False


def safe_join(directory: str, filename: str) -> str | None:
    """``os.path.join`` that refuses to leave *directory*. ``None`` if it would.

    Also refuses a name with any directory part at all: every caller passes a
    page filename, and a page filename never contains a separator.
    """
    if not directory or not filename:
        return None
    if filename != os.path.basename(filename) or filename in (".", ".."):
        return None
    if "\\" in filename or "/" in filename:
        return None

    joined = os.path.join(directory, filename)
    if not contains(directory, joined):
        return None
    return joined


def default_dest(source: str) -> str:
    """The destination offered for a source folder: ``<source>/fox_tled``."""
    resolved = resolve_dir(source)
    if not resolved:
        return ""
    return f"{resolved.rstrip('/')}/{DEFAULT_DEST_NAME}"
