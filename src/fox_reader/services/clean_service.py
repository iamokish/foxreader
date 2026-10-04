"""Text cleaning: turn per-entry clean settings into a repainted page.

The heavy lifting lives in :mod:`fox_reader.clean`; this is the orchestration
around it, and the orchestration is where the cost is decided:

* a detection pass runs **once per page**, not once per entry. Three entries
  cleaning with PaddleOCR share one detection; three cleaning with Text Seg share
  one segmentation over the union of their regions -- as long as they agree on the
  detector tuning, which is what :attr:`CleanJob.seg_key` groups them by;
* Text Seg runs on a *crop* -- the padded union of the regions that asked for it
  -- because the network costs 20-40 s for a full page on the CPU and a speech
  bubble is a twentieth of one;
* reconstruction is grouped by ``(fill, transport)``, so entries that agree on how
  to repaint are repainted in a single pass, and later passes see the earlier
  results as known pixels;
* everything is pure CPU work with no event loop in sight. Callers hand it to
  ``run_in_threadpool`` and pass a :class:`~fox_reader.services.progress.TaskProgress`
  so the UI can narrate it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from fox_reader.clean import (
    CLEAN_METHODS,
    DEFAULT_CLEAN_METHOD,
    DEFAULT_INPAINT_METHOD,
    DEFAULT_SPEED,
    DEFAULT_TILE,
    DEFAULT_TTA,
    INPAINT_METHODS,
    METHOD_KNOBS,
    inpaint_regions,
    normalise_speed,
    normalise_tile,
    passes_for,
    ppocr,
    refine_mask,
)

logger = logging.getLogger(__name__)

#: The Text Seg detection cache key: everything that changes the probability
#: maps. Two entries that differ only in *fill* share one segmentation; two that
#: differ in speed cannot, so each combination is detected once and no more.
SegKey = tuple[str, bool, str, bool, int]


@dataclass(slots=True)
class CleanJob:
    """One entry's request to clean its own region."""

    polygon: list[tuple[int, int]]
    method: str = DEFAULT_CLEAN_METHOD
    fill: str = DEFAULT_INPAINT_METHOD
    glow: bool = True
    transport: bool = True

    # Text Seg detector tuning. Ignored by the other two methods, but carried on
    # every job so that switching method back and forth in the panel does not
    # lose what was set -- see `METHOD_KNOBS` for which method reads which.
    speed: str = DEFAULT_SPEED
    tta: bool = DEFAULT_TTA
    tile: int = DEFAULT_TILE

    def normalised(self) -> CleanJob:
        """A copy with every field a value this build can actually honour.

        Falls back per field instead of rejecting the job: these values come out
        of saved projects, and a preset name from a newer build is not a reason to
        refuse to clean the page.
        """
        method = self.method if self.method in CLEAN_METHODS else DEFAULT_CLEAN_METHOD
        fill = self.fill if self.fill in INPAINT_METHODS else DEFAULT_INPAINT_METHOD
        return CleanJob(polygon=self.polygon, method=method, fill=fill,
                        glow=bool(self.glow), transport=bool(self.transport),
                        speed=normalise_speed(self.speed), tta=bool(self.tta),
                        tile=normalise_tile(self.tile))

    @property
    def seg_key(self) -> SegKey:
        """This job's Text Seg detection identity. Call on a normalised job."""
        return ("textseg", bool(self.glow), self.speed, bool(self.tta),
                int(self.tile))


@dataclass
class CleanPlan:
    """Detection results for one page, reusable across frames of an animation.

    An animated page is typeset frame by frame, and the lettering does not move
    between frames -- it is drawn on top of all of them. Detecting once and
    repainting per frame turns an unusable 40 s x N into 40 s plus a second a
    frame.
    """

    shape: tuple[int, int]
    masks: dict[Any, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class CleanService:
    """Text clean, sharing the OCR detector and the Text Seg model."""

    def __init__(self, ocr: Any | None = None) -> None:
        self._ocr = ocr

    # ------------------------------------------------------------ capability

    def available_methods(self) -> list[dict[str, Any]]:
        """What the UI may offer, why a method is unavailable, and its knobs.

        Deliberately cheap: no model is loaded and no import beyond a spec
        lookup, because this is answered while the settings pane is being drawn.

        ``knobs`` travels with each method so the panel can offer exactly the
        controls that do something. Keeping that list here rather than in the
        frontend means a method whose tuning changes cannot end up advertising a
        switch the backend stopped reading.
        """
        out: list[dict[str, Any]] = [
            {"id": "region", "label": "Selected Region", "available": True,
             "reason": ""},
        ]

        detector = getattr(self._ocr, "detector", None) if self._ocr else None
        out.append({
            "id": "ppocr", "label": "PaddleOCR",
            "available": detector is not None,
            "reason": "" if detector is not None else
                      "the OCR models are not loaded",
        })

        from fox_reader.clean import seg

        ok, why = seg.is_available()
        out.append({"id": "textseg", "label": "Text Seg", "available": ok,
                    "reason": why})
        for info in out:
            info["knobs"] = list(METHOD_KNOBS.get(str(info["id"]), ()))
        return out

    # ----------------------------------------------------------------- masks

    @staticmethod
    def _polygon_mask(shape: tuple[int, int],
                      polygons: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), np.uint8)
        for poly in polygons:
            pts = np.asarray(poly, np.int32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_8)
        return mask

    @staticmethod
    def _bbox(shape: tuple[int, int], mask: np.ndarray, pad: int
              ) -> tuple[int, int, int, int] | None:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        h, w = shape
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _ppocr_mask(self, bgr: np.ndarray, task: Any | None) -> np.ndarray:
        """Text blocks over the whole page.

        The whole page on purpose: DB post-processing is calibrated on page-sized
        input, and a detection cropped to one bubble loses the context that tells
        it where a line ends. It costs about two seconds either way.
        """
        if task is not None:
            task.say("Text clean: detecting text (PaddleOCR)")
        if self._ocr is None or getattr(self._ocr, "detector", None) is None:
            raise ppocr.DetectorUnavailable(
                "the OCR models are not loaded, so PaddleOCR text clean is "
                "unavailable")
        polys, scores = self._ocr.detect_text_boxes(bgr)
        cov = ppocr.coverage(bgr.shape, polys, scores)
        if task is not None:
            task.say(f"Text clean: PaddleOCR found {len(polys)} text region(s)")
        return ppocr.refine_boxes(cov)

    def _textseg_mask(self, bgr: np.ndarray, want: np.ndarray, glow: bool,
                      speed: str, tta: bool, tile: int,
                      task: Any | None) -> np.ndarray:
        """Glyph-level segmentation, run only over the regions that asked for it.

        ``speed``, ``tta`` and ``tile`` are handed straight to
        :func:`.seg.detect_at`, which resolves the preset to (flips, scales,
        polarity) exactly the way the reference CLI's ``main`` does. ``glow``
        reaches :func:`.mask.refine_mask` as ``glow_k``, which is the CLI's
        ``--no-glow``.
        """
        from fox_reader.clean import seg

        h, w = bgr.shape[:2]
        full = np.zeros((h, w), np.uint8)
        box = self._bbox((h, w), want, pad=64)
        if box is None:
            return full
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
        if task is not None:
            task.say(f"Text clean: segmenting {x1 - x0}x{y1 - y0} px "
                     f"(Text Seg, {speed}, up to {passes_for(speed, tta)} pass(es)"
                     f"{f', {tile} px tiles' if tile else ''})")

        def say(text: str) -> None:
            if task is not None:
                task.say(f"Text clean: {text}")

        cov, agree = seg.detect_at(crop, speed=speed, tta=tta, tile=tile,
                                   progress=say)
        if task is not None:
            task.say("Text clean: refining the Text Seg mask"
                     f"{'' if agree is not None else ' (one scale: no vote)'}")
        # `agree=None` is not "no filtering by accident": `detect_at` returns it
        # for the single-scale preset, and `refine_mask` reads it as "keep
        # everything found". Passing a one-scale vote map instead would call every
        # detection unsupported and empty the mask.
        refined = refine_mask(crop, cov, agree=agree,
                              glow_k=2.0 if glow else 0.0)
        full[y0:y1, x0:x1] = refined
        return full

    def plan(self, bgr: np.ndarray, jobs: Sequence[CleanJob],
             task: Any | None = None) -> CleanPlan:
        """Run every detection pass the jobs need, once each."""
        h, w = bgr.shape[:2]
        out = CleanPlan(shape=(h, w))
        jobs = [j.normalised() for j in jobs]

        if any(j.method == "ppocr" for j in jobs):
            try:
                out.masks["ppocr"] = self._ppocr_mask(bgr, task)
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                logger.warning("PaddleOCR text clean unavailable: %s", exc)
                out.notes.append(f"PaddleOCR text clean failed: {exc}")

        # One segmentation per distinct tuning, over the union of the regions
        # that asked for exactly that tuning. Sorted so the order a page is
        # narrated in does not depend on set iteration order.
        seg_jobs = [j for j in jobs if j.method == "textseg"]
        for key in sorted({j.seg_key for j in seg_jobs}):
            _, glow, speed, tta, tile = key
            want = self._polygon_mask((h, w), [j.polygon for j in seg_jobs
                                              if j.seg_key == key])
            try:
                out.masks[key] = self._textseg_mask(
                    bgr, want, glow, speed, tta, tile, task)
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                logger.warning("Text Seg clean unavailable: %s", exc)
                out.notes.append(f"Text Seg clean failed: {exc}")
        return out

    def _job_mask(self, plan: CleanPlan, job: CleanJob) -> np.ndarray:
        """What this entry removes: its own region, narrowed by its method."""
        own = self._polygon_mask(plan.shape, [job.polygon])
        if job.method == "region":
            return own
        key: Any = "ppocr" if job.method == "ppocr" else job.seg_key
        found = plan.masks.get(key)
        if found is None:
            # The detector was unavailable. Removing the whole region instead
            # would silently do something the user did not ask for, so the entry
            # is left alone and `plan.notes` explains why.
            return np.zeros(plan.shape, np.uint8)
        return cv2.bitwise_and(own, found)

    # ----------------------------------------------------------------- clean

    def clean(self, bgr: np.ndarray, jobs: Sequence[CleanJob],
              task: Any | None = None, plan: CleanPlan | None = None
              ) -> tuple[np.ndarray, CleanPlan]:
        """Repaint every job's region. Returns the page and the plan used.

        Pass the returned plan back in for the next frame of an animation to skip
        detection entirely.
        """
        jobs = [j.normalised() for j in jobs if len(j.polygon) >= 3]
        h, w = bgr.shape[:2]
        if not jobs:
            return bgr, plan or CleanPlan(shape=(h, w))

        if plan is None or plan.shape != (h, w):
            plan = self.plan(bgr, jobs, task)

        # Entries that agree on how to repaint are repainted together: one
        # clustering pass over the union, so two overlapping regions cannot leave
        # a seam where one was filled against the other's untouched text.
        groups: dict[tuple[str, bool], np.ndarray] = {}
        for job in jobs:
            m = self._job_mask(plan, job)
            if not m.any():
                continue
            key = (job.fill, job.transport)
            if key in groups:
                cv2.bitwise_or(groups[key], m, dst=groups[key])
            else:
                groups[key] = m

        if not groups:
            return bgr, plan

        out = bgr
        for i, ((fill, transport), mask) in enumerate(groups.items(), 1):
            label = f"{fill}{'' if transport else ' (no transport)'}"
            if task is not None:
                task.say(f"Text clean: repainting with {label}"
                         f"{f' [{i}/{len(groups)}]' if len(groups) > 1 else ''}")

            def report(done: int, total: int, _label: str = label) -> None:
                if task is not None and total > 1:
                    task.say(f"Text clean: {_label} {done}/{total} region(s)")

            try:
                out = inpaint_regions(out, mask, method=fill,
                                      transport=transport, progress=report)
            except Exception as exc:  # noqa: BLE001 - keep the other groups
                logger.exception("Text clean failed for %s", label)
                plan.notes.append(f"Text clean ({label}) failed: {exc}")
        return out, plan
