from math import isfinite
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FreeformRequest(BaseModel):
    filename: str
    points: list
    lang: str
    isGrayScale: bool


class CropRequest(BaseModel):
    filename: str
    x: int
    y: int
    width: int
    height: int
    lang: str
    isGrayScale: bool


class BubbleRequest(BaseModel):
    filename: str
    isGrayScale: bool


class Point(BaseModel):
    x: int
    y: int


class RectCoords(BaseModel):
    """A rectangle as the frontend stores it: origin plus size."""

    x: int
    y: int
    w: int
    h: int


class Region(BaseModel):
    # `coords` is a point list for polygons and a single {x,y,w,h} object for
    # rectangles, which is exactly what the frontend keeps in `Entry.region`.
    type: str
    coords: list[Point] | RectCoords

    def to_polygon(self) -> list[dict[str, int]]:
        """Normalise either shape into a CCW-agnostic polygon point list."""
        coords = self.coords
        if isinstance(coords, RectCoords):
            x0, y0 = int(coords.x), int(coords.y)
            x1, y1 = x0 + int(coords.w), y0 + int(coords.h)
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            return [
                {"x": x0, "y": y0},
                {"x": x1, "y": y0},
                {"x": x1, "y": y1},
                {"x": x0, "y": y1},
            ]
        return [{"x": int(p.x), "y": int(p.y)} for p in coords]


class SplitRequest(BaseModel):
    region: Region
    split_object: list[Point]
    isGrayScale: bool
    filename: str


class TranslationRequest(BaseModel):
    text: str
    source_lang: str

    # Custom endpoints send their endpoint id as `source_lang`, so the real
    # language travels alongside it. Optional: the built-in engines never
    # send it.
    lang_group: str | None = None

    # Translation context: earlier [[source, english], ...] pairs in reading
    # order, sent only for the local MTL endpoint when the Context switch is
    # on. Optional so older pages keep working; the backend ignores it for
    # models that do not render it.
    context: list[list[str]] | None = None

    # Character metadata for models that render it (VNTL). All optional so
    # older pages keep working; the backend ignores them for models without
    # character support, and degrades gracefully (tags dropped, never a 422)
    # when they are malformed.
    #
    # `character_info` is the roster snapshot: [{meta_id, name_en, name_ja,
    # gender, alias_en, alias_ja}, ...]. `context_character_links` aligns to
    # `context` by position (one meta_id or None per pair). `meta_id` names
    # the speaker of `text`.
    character_info: Any = None
    context_character_links: Any = None
    meta_id: Any = None


class CleanOptions(BaseModel):
    """Per-entry text-clean settings (see `fox_reader.clean`).

    Deliberately permissive: every field has a default, and the service
    normalises unknown values rather than rejecting them
    (`CleanJob.normalised`). These arrive from saved projects, so a project
    written by an older build -- or by a newer one naming a preset this build
    does not have -- has to keep rendering.
    """

    # region | ppocr | textseg
    method: str = "region"
    # One of fox_reader.clean.INPAINT_METHODS.
    fill: str = "hybrid-level"
    glow: bool = True
    transport: bool = True

    # Detector tuning; textseg only. `speed` names a preset in
    # fox_reader.clean.SPEEDS and decides how many forward passes to spend,
    # `tta` can only switch the preset's flip averaging off, and `tile` is one
    # of fox_reader.clean.TILE_OVERLAPS -- 0 predicts the whole crop at once.
    # The overlap is not sent: it is tabulated per tile size on the server, so
    # the two cannot arrive inconsistent with each other.
    speed: str = "fastest"
    tta: bool = True
    tile: int = 0


class DataItem(BaseModel):
    og_text: str
    text: str
    fontfile: str
    text_align: str
    points: list[tuple[int, int]]

    # Stacking order; 1 is the bottom layer. Items are drawn low to high.
    layer: int = 1

    # `None` on any of these means "auto detect from the image".
    font_size: int | None = None
    font_color: str | None = None
    stroke_width: int | None = None
    stroke_color: str | None = None

    # auto | color | transparent | clean
    bg_mode: str = "auto"
    bg_color: str | None = None
    clean: CleanOptions | None = None

    # Per-entry typesetting geometry (see `fox_reader.typeset`). The two
    # spacings are multipliers over the existing metrics, with `None` meaning
    # "leave that logic alone"; the rest are neutral at their defaults, so an
    # entry that never touched them renders exactly as it did before they
    # existed.
    #
    # `word_spacing` scales the gap between words, `line_spacing` the distance
    # between baselines, and both take part in the auto font fit. The scales
    # stretch the lettering (also part of the fit); the shifts and angles are
    # applied after fitting, to the text alone -- never to the plate behind it.
    word_spacing: float | None = None
    line_spacing: float | None = None
    font_scale_x: float = 1.0
    font_scale_y: float = 1.0
    shift_x: int = 0
    shift_y: int = 0
    angle_x: int = 0
    angle_y: int = 0
    angle_z: int = 0

    # Permissive in the same spirit as `CleanOptions`: these arrive from saved
    # projects, so a value this build cannot read has to degrade to the default
    # rather than 422 the whole page. Range clamping lives in
    # `fox_reader.typeset`, which is the single source of truth for the limits.
    @field_validator("word_spacing", "line_spacing", mode="before")
    @classmethod
    def _optional_multiplier(cls, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if isfinite(number) else None

    @field_validator("font_scale_x", "font_scale_y", mode="before")
    @classmethod
    def _scale(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 1.0
        return number if isfinite(number) else 1.0

    @field_validator("shift_x", "shift_y", "angle_x", "angle_y", "angle_z",
                     mode="before")
    @classmethod
    def _whole(cls, value: Any) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0
        return int(round(number)) if isfinite(number) else 0


class ProcessImageRequest(BaseModel):
    filename: str
    data: list[DataItem]

    # Whether an existing file in the destination may be replaced. False means
    # the backend refuses with 409 instead, and the page asks first. Defaulted so
    # that a request from an older page can never overwrite without consent.
    overwrite: bool = False


class SavePreviewRequest(BaseModel):
    filename: str
    # Guards against saving a preview that belongs to another page.
    token: str = Field(default="")
    overwrite: bool = False


class FolderRequest(BaseModel):
    path: str

    # Where typeset pages go. None means "decide for me": the destination this
    # source was last loaded with, or <source>/fox_tled. Optional so that any
    # caller written against the one-directory API keeps working.
    dest: str | None = None
    create_dest: bool = True


class FolderInspectRequest(BaseModel):
    """A permission check for the folder picker. Reads and probes; writes nothing."""

    source: str = ""
    dest: str | None = None


class LoadMLRequest(BaseModel):
    lang: str
