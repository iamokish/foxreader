"""Typesetting / inpainting operations, and the preview artifact that backs them."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from fox_reader.typeset import process_typesetting
from fox_reader.utils import CACHE_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreviewArtifact:
    """The last preview rendered, and what it was rendered from."""

    token: str
    filename: str
    source: str
    path: str


class InpaintService:
    """Typesetting / inpainting operations."""

    def __init__(self, clean_service: Any | None = None) -> None:
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._clean = clean_service
        self._lock = threading.Lock()
        self._preview: PreviewArtifact | None = None

    # --------------------------------------------------------------- rendering

    def generate(self, image_path: str, dataitems: list,
                 task: Any | None = None, save_path: str | None = None) -> str:
        """Typeset ``image_path`` and write the result to ``save_path``.

        The two paths are separate so the original scan survives: rendering reads
        the source page and writes into the destination folder, and re-rendering
        the same page reads the original again rather than typesetting on top of
        an already-typeset image.

        Written to a temporary file in the destination and moved into place, for
        the same reason `save_preview` does it -- a render that fails or is killed
        halfway must not leave a truncated page behind, and on a re-save the old
        file stays intact until the new one is complete. The temporary name keeps
        the original extension last, because `typeset.save_img` picks the image
        format from it; a `.tmp` suffix would write a PNG into a `.jpg`.
        """
        target = Path(save_path or image_path)
        if not target.parent.is_dir():
            raise OSError(f"{target.parent} does not exist")

        tmp = target.with_name(f".fox_tmp_{uuid4().hex[:8]}_{target.name}")
        try:
            process_typesetting(image_path, str(tmp), dataitems,
                                clean_service=self._clean, task=task)
            os.replace(tmp, target)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove the temporary file %s", tmp)
            raise
        return str(target)

    def preview(self, image_path: str, dataitems: list, filename: str,
                task: Any | None = None) -> PreviewArtifact:
        """Render a preview into the cache and remember it.

        Returns the artifact rather than a bare path so the caller can hand the
        browser a token: saving then copies *this* file instead of typesetting the
        page a second time, which for a page with a Text Seg clean on it is the
        difference between an instant save and another 40 s.
        """
        file_ext = os.path.splitext(filename)[-1]
        preview_path = str(self._cache_dir / f"preview.{file_ext.strip('.')}")

        if image_path and os.path.exists(image_path):
            process_typesetting(image_path, preview_path, dataitems,
                                clean_service=self._clean, task=task)
        else:
            # No page open: a blank plate is still a valid answer, and it keeps
            # the preview modal from erroring on an empty folder.
            img = Image.new("RGB", (300, 300), color="white")
            img.save(preview_path)

        artifact = PreviewArtifact(token=uuid4().hex, filename=filename,
                                   source=image_path, path=preview_path)
        with self._lock:
            self._preview = artifact
        return artifact

    # ------------------------------------------------------------------ saving

    def current_preview(self) -> PreviewArtifact | None:
        with self._lock:
            return self._preview

    def save_preview(self, token: str, filename: str,
                     image_path: str) -> str:
        """Commit the previewed image to ``image_path``.

        ``image_path`` is the destination copy of the page, which is usually not
        the file the preview was rendered *from*; the caller decides where a save
        lands (see `fox_reader.routes.inpaint`).

        Verified three ways -- token, filename and the file still being there --
        because this writes over whatever is already at the target. A stale token
        (the preview was re-rendered, or the app restarted) is an error rather
        than a silent fallback to re-rendering: re-rendering with whatever entries
        the request happens to carry is not what "save this image" means.
        """
        artifact = self.current_preview()
        if artifact is None:
            raise LookupError("there is no preview to save; render one first")
        if not token or token != artifact.token:
            raise LookupError(
                "this preview is out of date; re-run the preview and save again")
        if os.path.basename(filename) != os.path.basename(artifact.filename):
            raise LookupError(
                f"the preview is of {artifact.filename!r}, not {filename!r}")
        if not os.path.isfile(artifact.path):
            raise LookupError("the previewed image is no longer in the cache")

        target = Path(image_path)
        if not target.parent.is_dir():
            raise LookupError(f"{target.parent} no longer exists")

        # Written beside the target and moved into place: a copy straight onto the
        # target leaves a half-written page if it fails halfway, and the cache
        # can be on another volume, where os.replace would not work.
        tmp = target.with_name(f".{target.name}.{uuid4().hex[:8]}.tmp")
        try:
            shutil.copyfile(artifact.path, tmp)
            os.replace(tmp, target)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove the temporary file %s", tmp)
            raise
        return str(target)
