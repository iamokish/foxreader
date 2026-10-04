"""Tests for the Text Seg detector tuning: presets, tiles, and what reads them.

Three separate things are pinned here, and they fail in different ways:

* :mod:`fox_reader.clean.tuning` must stay importable *without* torch. It is what
  the capability endpoint and the request model name presets through, and
  importing :mod:`fox_reader.clean.seg` instead would pull in
  :mod:`fox_reader.device`, which probes the hardware at import time -- on the
  event loop, that is the health check blocked behind a CUDA probe.
* :func:`fox_reader.clean.seg.detect` must report *no agreement* (``None``) when
  it only had one scale to look at. Returning a one-scale vote map instead reads
  as agreement and silently redefines the ``single`` preset.
* the tile/overlap pairs are a table, not a formula. Each size is offered with
  exactly one overlap, and the number the UI shows has to be the number the
  network gets.

The numbers asserted here are the reference CLI's (``tt/seg.py``), spelled out
literally rather than recomputed from the tables they are meant to check.
"""
import numpy as np
import pytest

from fox_reader.clean import tuning
from fox_reader.clean.tuning import (
    DEFAULT_SPEED,
    DEFAULT_TILE,
    DEFAULT_TTA,
    SPEED_ORDER,
    SPEEDS,
    TILE_ORDER,
    TILE_OVERLAPS,
    normalise_speed,
    normalise_tile,
    overlap_for,
    passes_for,
)


class TestSpeedPresets:
    def test_matches_the_reference_cli(self):
        # (flip TTA, scales, tone-inverted pass), from tt/seg.py's SPEEDS.
        assert SPEEDS == {
            "best": (True, (1.0, 0.6, 0.4), True),
            "fast": (False, (1.0, 0.6, 0.4), True),
            "fastest": (False, (1.0, 0.6, 0.4), False),
            "single": (False, (1.0,), False),
        }

    def test_the_picker_offers_every_preset_slowest_first(self):
        assert set(SPEED_ORDER) == set(SPEEDS)
        assert [passes_for(name) for name in SPEED_ORDER] == [24, 6, 3, 1]

    def test_the_default_is_the_three_pass_preset(self):
        assert DEFAULT_SPEED == "fastest"
        assert passes_for(DEFAULT_SPEED) == 3
        assert DEFAULT_TTA is True

    def test_only_best_has_flip_tta_to_switch_off(self):
        # What the panel dims the TTA box on: the switch is real and stored, but
        # a preset that asks for no flips has nothing for it to act on.
        assert (passes_for("best", True), passes_for("best", False)) == (24, 6)
        for name in ("fast", "fastest", "single"):
            assert passes_for(name, True) == passes_for(name, False)

    def test_an_unknown_preset_falls_back_instead_of_raising(self):
        # These arrive from saved projects and from the network.
        for bad in ("BEST", "turbo", "", None, 3, True, ["fast"]):
            assert normalise_speed(bad) == DEFAULT_SPEED
        assert passes_for("turbo") == passes_for(DEFAULT_SPEED)

    def test_a_known_preset_is_left_alone(self):
        for name in SPEEDS:
            assert normalise_speed(name) == name


class TestTileSizes:
    def test_pairs_are_the_ones_the_ui_offers(self):
        assert TILE_OVERLAPS == {0: 192, 256: 48, 512: 96, 1024: 192, 2048: 192}
        assert TILE_ORDER == (0, 256, 512, 1024, 2048)
        assert DEFAULT_TILE == 0

    def test_every_offered_size_has_an_overlap(self):
        # The overlap is not sent by the client precisely because of this: the
        # pairing is the backend's, so the two cannot arrive inconsistent.
        assert set(TILE_ORDER) == set(TILE_OVERLAPS)
        for size in TILE_ORDER:
            assert overlap_for(size) == TILE_OVERLAPS[size]

    def test_the_overlap_always_fits_inside_the_tile(self):
        # A step of `tile - overlap` must be positive or the tiling loop hangs.
        for size in TILE_ORDER:
            if size:
                assert 0 < overlap_for(size) < size

    def test_an_untabulated_size_falls_back(self):
        for bad in (7, 128, 4096, -512, "512", 512.0, None, [512]):
            assert normalise_tile(bad) == DEFAULT_TILE
        assert overlap_for(9999) == TILE_OVERLAPS[DEFAULT_TILE]

    def test_a_boolean_is_not_a_tile_size(self):
        # `bool` is an `int` subclass, so True would otherwise read as tile 1 --
        # and `tile=True` is exactly what a sloppy client sends for "on".
        assert normalise_tile(True) == DEFAULT_TILE
        assert normalise_tile(False) == DEFAULT_TILE


class TestImportCost:
    def test_naming_a_preset_pulls_in_nothing(self):
        # `tuning` is the import-cheap door for the request model, the capability
        # endpoint and the job normaliser. Anything it reached would be paid for
        # on the event loop: `fox_reader.device` probes the hardware at import,
        # and `seg` imports torch. Nothing outside the stdlib, so the check is
        # simply that it names no import at all beyond `__future__`.
        import ast

        source = tuning.__file__ or ""
        assert source.endswith("tuning.py")
        with open(source, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())

        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.append(node.module or "")
        assert imported == []


class TestDetectAgreement:
    """`seg.detect`'s two-value contract, with the network stubbed out."""

    @staticmethod
    def _seg(monkeypatch, maps):
        from fox_reader.clean import seg

        seen: dict = {}

        def fake_scale_maps(bgr, **kw):
            seen.update(kw)
            return dict(maps)

        monkeypatch.setattr(seg, "scale_maps", fake_scale_maps)
        return seg, seen

    def test_one_scale_reports_no_vote_rather_than_an_empty_one(self, monkeypatch):
        one = np.full((4, 4), 0.9, np.float32)
        seg, _ = self._seg(monkeypatch, {1.0: one})

        coverage, agree = seg.detect(np.zeros((4, 4, 3), np.uint8), scales=(1.0,))

        # `None` means "no vote was taken"; `refine_mask` reads it as "keep what
        # you found". A one-scale vote map would have called every detection
        # unsupported and emptied the mask instead.
        assert agree is None
        assert np.array_equal(coverage, one)

    def test_coverage_is_the_pixelwise_maximum(self, monkeypatch):
        a = np.array([[0.1, 0.9]], np.float32)
        b = np.array([[0.8, 0.2]], np.float32)
        seg, _ = self._seg(monkeypatch, {1.0: a, 0.6: b})

        coverage, agree = seg.detect(np.zeros((1, 2, 3), np.uint8))

        assert np.allclose(coverage, [[0.8, 0.9]])
        assert agree is not None

    def test_two_scales_must_both_agree(self, monkeypatch):
        a = np.array([[0.9, 0.9, 0.1]], np.float32)
        b = np.array([[0.9, 0.1, 0.1]], np.float32)
        seg, _ = self._seg(monkeypatch, {1.0: a, 0.6: b})

        _, agree = seg.detect(np.zeros((1, 3, 3), np.uint8), votes=2)

        assert agree is not None
        assert agree.dtype == np.bool_
        assert agree.tolist() == [[True, False, False]]

    def test_two_of_three_scales_is_enough(self, monkeypatch):
        # The cross-scale vote is the text/art discriminator: a glyph is still a
        # glyph at 60 %, whereas art it mistakes for text fires at one scale.
        rows = {
            1.0: np.array([[0.9, 0.9, 0.9]], np.float32),
            0.6: np.array([[0.9, 0.9, 0.1]], np.float32),
            0.4: np.array([[0.9, 0.1, 0.1]], np.float32),
        }
        seg, _ = self._seg(monkeypatch, rows)

        _, agree = seg.detect(np.zeros((1, 3, 3), np.uint8), votes=2)

        assert agree.tolist() == [[True, True, False]]

    def test_more_votes_than_scales_is_clamped(self, monkeypatch):
        # Otherwise asking for 3 votes from 2 scales empties the mask outright.
        hot = np.full((1, 2), 0.9, np.float32)
        seg, _ = self._seg(monkeypatch, {1.0: hot, 0.6: hot})

        _, agree = seg.detect(np.zeros((1, 2, 3), np.uint8), votes=9)

        assert agree.tolist() == [[True, True]]


class TestDetectAt:
    """The preset -> kwargs wiring, which is what the CLI's `main` does."""

    @staticmethod
    def _capture(monkeypatch):
        from fox_reader.clean import seg

        seen: dict = {}

        def fake_detect(bgr, **kw):
            seen.update(kw)
            return np.zeros((1, 1), np.float32), None

        monkeypatch.setattr(seg, "detect", fake_detect)
        return seg, seen

    def test_each_preset_resolves_the_way_the_reference_cli_does(self, monkeypatch):
        seg, seen = self._capture(monkeypatch)
        img = np.zeros((8, 8, 3), np.uint8)
        expected = {
            "best": (True, (1.0, 0.6, 0.4), True),
            "fast": (False, (1.0, 0.6, 0.4), True),
            "fastest": (False, (1.0, 0.6, 0.4), False),
            "single": (False, (1.0,), False),
        }
        for name, (tta, scales, polarity) in expected.items():
            seen.clear()
            seg.detect_at(img, speed=name)
            assert (seen["tta"], seen["scales"], seen["polarity"]) == (
                tta, scales, polarity), name

    def test_the_switch_can_only_turn_flip_tta_off(self, monkeypatch):
        seg, seen = self._capture(monkeypatch)
        img = np.zeros((8, 8, 3), np.uint8)

        seg.detect_at(img, speed="best", tta=False)
        assert seen["tta"] is False
        # `fast` asks for no flips, so the switch has nothing to enable.
        seg.detect_at(img, speed="fast", tta=True)
        assert seen["tta"] is False

    def test_the_tile_brings_its_own_overlap(self, monkeypatch):
        seg, seen = self._capture(monkeypatch)
        img = np.zeros((8, 8, 3), np.uint8)

        for size in TILE_ORDER:
            seen.clear()
            seg.detect_at(img, tile=size)
            assert (seen["tile"], seen["overlap"]) == (size, TILE_OVERLAPS[size])

    def test_a_bogus_preset_or_tile_still_runs(self, monkeypatch):
        seg, seen = self._capture(monkeypatch)

        seg.detect_at(np.zeros((8, 8, 3), np.uint8), speed="turbo", tile=333)

        assert seen["scales"] == SPEEDS[DEFAULT_SPEED][1]
        assert (seen["tile"], seen["overlap"]) == (
            DEFAULT_TILE, TILE_OVERLAPS[DEFAULT_TILE])


class TestScaleGeometry:
    def test_scale_one_is_the_original_size(self):
        from fox_reader.clean import seg

        assert seg._scaled_size((300, 200), 1.0) == (200, 300)

    def test_a_scaled_size_never_collapses(self):
        from fox_reader.clean import seg

        # 32 px is the floor the network needs; 40 * 0.4 would be 16.
        assert seg._scaled_size((40, 40), 0.4) == (32, 32)
        assert seg._scaled_size((1000, 500), 0.6) == (300, 600)


class TestTileSelection:
    """Whether one prediction is tiled at all. `_predict_one` decides per scale."""

    @staticmethod
    def _stub(monkeypatch):
        from fox_reader.clean import seg

        used: list[str] = []

        def plain(model, device, rgb, tta):
            used.append("plain")
            return np.zeros(rgb.shape[:2], np.float32)

        def tiled(model, device, rgb, tta, tile, overlap, progress=None):
            used.append("tiled")
            return np.zeros(rgb.shape[:2], np.float32)

        monkeypatch.setattr(seg, "_predict_plain", plain)
        monkeypatch.setattr(seg, "_predict_tiled", tiled)
        return seg, used

    @staticmethod
    def _predict(seg, shape, s, tile):
        return seg._predict_one(None, "cpu", np.zeros((*shape, 3), np.uint8), s,
                                False, False, tile, 96, shape, None)

    def test_a_tile_larger_than_the_crop_is_not_used(self, monkeypatch):
        seg, used = self._stub(monkeypatch)

        # A speech bubble is not tiled by a 2048 px tile...
        self._predict(seg, (900, 700), 1.0, 2048)
        # ...and neither is the 0.4 scale of a crop that only just needed tiling.
        self._predict(seg, (900, 700), 0.4, 512)

        assert used == ["plain", "plain"]

    def test_a_crop_larger_than_the_tile_is_tiled(self, monkeypatch):
        seg, used = self._stub(monkeypatch)

        self._predict(seg, (900, 700), 1.0, 512)

        assert used == ["tiled"]

    def test_tiling_off_stays_off(self, monkeypatch):
        seg, used = self._stub(monkeypatch)

        self._predict(seg, (400, 300), 1.0, 0)

        assert used == ["plain"]

    def test_a_scaled_prediction_comes_back_at_the_crop_size(self, monkeypatch):
        # The mask stage indexes the maps against the crop, so a downscaled map
        # that never came back up is an immediate broadcast error.
        seg, _ = self._stub(monkeypatch)

        assert self._predict(seg, (120, 90), 0.4, 0).shape == (120, 90)


class TestModelLoading:
    """Which weight file is accepted, and which devices get channels_last."""

    @staticmethod
    def _weights(monkeypatch, tmp_path, *names):
        from fox_reader.clean import seg

        for name in names:
            (tmp_path / name).write_bytes(b"not really a checkpoint")
        monkeypatch.setattr(seg, "MODEL_PATH", tmp_path / "model.safetensors")
        monkeypatch.setattr(seg, "LEGACY_PATH", tmp_path / "model.pth")
        monkeypatch.setattr(seg, "MODEL_DIR", tmp_path)
        return seg

    def test_the_legacy_checkpoint_alone_is_enough(self, monkeypatch, tmp_path):
        # It used to refuse until the file had been converted, which reported the
        # method as broken when it was only slower to load. A fresh clone of the
        # model repository carries model.pth and nothing else.
        seg = self._weights(monkeypatch, tmp_path, "model.pth")

        ok, why = seg.is_available()

        # Past the file check; anything left is torch, which this environment
        # may or may not have installed. The architecture itself no longer
        # gates this -- it ships with the package.
        assert ok or "needs torch" in why

    def test_safetensors_alone_is_enough(self, monkeypatch, tmp_path):
        seg = self._weights(monkeypatch, tmp_path, "model.safetensors")

        ok, why = seg.is_available()

        assert ok or "needs " in why

    def test_no_weights_at_all_says_so(self, monkeypatch, tmp_path):
        seg = self._weights(monkeypatch, tmp_path)

        ok, why = seg.is_available()

        assert ok is False
        assert "missing" in why

    def test_an_unreadable_checkpoint_is_a_seg_failure(self, monkeypatch, tmp_path):
        # Half a download is the likely cause, and the panel has to be able to
        # say so -- not surface safetensors' own exception type.
        seg = self._weights(monkeypatch, tmp_path, "model.safetensors")

        with pytest.raises(seg.SegUnavailable):
            seg._load_state()

    def test_channels_last_only_where_it_was_measured(self):
        from fox_reader.clean import seg

        assert seg._channels_last("cpu") is True
        assert seg._channels_last("cuda") is True
        assert seg._channels_last("cuda:1") is True
        # Not an approximation but not a measurement either: MPS keeps the
        # layout torch gives it.
        assert seg._channels_last("mps") is False


class TestScaleMapsPlumbing:
    """`scale_maps` without a model: what reaches `_predict_one`, and the merge."""

    @staticmethod
    def _stub(monkeypatch, normal=1.0, inverted=1.0):
        """Stand in for the network: record the call, return a flat map."""
        from fox_reader.clean import seg

        calls: list[dict] = []

        def fake_predict(model, device, rgb, s, inv, tta, tile, overlap, shape,
                         progress=None):
            calls.append({"s": s, "inv": inv, "tta": tta, "tile": tile,
                          "overlap": overlap, "shape": shape})
            h, w = shape
            return np.full((h, w), inverted if inv else normal, np.float32)

        # "cuda" so that the CPU thread limiter is skipped: it would otherwise
        # call `torch.set_num_threads` for real, in the test process.
        monkeypatch.setattr(seg, "torch_device", lambda: "cuda")
        monkeypatch.setattr(seg, "load_model", lambda device=None: (object(), "cuda"))
        monkeypatch.setattr(seg, "_predict_one", fake_predict)
        return seg, calls

    @staticmethod
    def _dark_page(h=200, w=200):
        """Black, with a thin light stripe: what opens the inverted pass.

        Thin on purpose. The gate compares each pixel against a median of its
        neighbourhood, so a wide light block is its own local page tone and gates
        nothing -- a stripe narrower than the median window is light *on* dark.
        """
        page = np.zeros((h, w, 3), np.uint8)
        page[:, w // 2 - 2:w // 2 + 2] = 255
        return page

    def test_the_tabulated_overlap_reaches_the_network(self, monkeypatch):
        seg, calls = self._stub(monkeypatch)

        seg.scale_maps(np.zeros((600, 600, 3), np.uint8), scales=(1.0,),
                       polarity=False, tile=512)

        assert len(calls) == 1
        assert (calls[0]["tile"], calls[0]["overlap"]) == (512, 96)

    def test_an_overlap_that_would_hang_the_loop_is_clamped(self, monkeypatch):
        # `step = tile - overlap`; zero loops forever and negative is the same
        # mistake. Only the tests pass an overlap separately, but the clamp is
        # what makes that safe.
        seg, calls = self._stub(monkeypatch)
        img = np.zeros((600, 600, 3), np.uint8)

        seg.scale_maps(img, scales=(1.0,), polarity=False, tile=256, overlap=900)
        assert calls[-1]["overlap"] == 255
        seg.scale_maps(img, scales=(1.0,), polarity=False, tile=256, overlap=-8)
        assert calls[-1]["overlap"] == 0

    def test_one_pass_per_scale_and_polarity(self, monkeypatch):
        seg, calls = self._stub(monkeypatch)
        img = self._dark_page()

        seg.scale_maps(img, scales=(1.0, 0.6, 0.4), polarity=True)
        assert len(calls) == 6
        calls.clear()
        seg.scale_maps(img, scales=(1.0, 0.6, 0.4), polarity=False)
        assert len(calls) == 3

    def test_nothing_is_cached_between_two_identical_calls(self, monkeypatch):
        # The npy map cache is gone. It keyed on a hash of the pixels and wrote
        # only on the CPU, so this exact repeat on the CPU is the case it served.
        seg, calls = self._stub(monkeypatch)
        monkeypatch.setattr(seg, "torch_device", lambda: "cpu")
        monkeypatch.setattr(seg, "load_model",
                            lambda device=None: (object(), "cpu"))
        monkeypatch.setattr(seg, "_limit_threads", lambda: None)
        img = np.zeros((64, 64, 3), np.uint8)

        seg.scale_maps(img, scales=(1.0,), polarity=False)
        seg.scale_maps(img, scales=(1.0,), polarity=False)

        assert len(calls) == 2
        from fox_reader.utils import CACHE_DIR

        assert list(CACHE_DIR.glob("*.npy")) == []

    def test_every_map_is_keyed_by_its_scale_and_sized_to_the_crop(self, monkeypatch):
        # The mask stage indexes the maps against the crop, so a downscaled map
        # that never came back up is an immediate broadcast error.
        seg, calls = self._stub(monkeypatch)

        out = seg.scale_maps(np.zeros((120, 90, 3), np.uint8),
                             scales=(1.0, 0.4), polarity=False)

        assert sorted(out) == [0.4, 1.0]
        for prob in out.values():
            assert prob.shape == (120, 90)
        # Both scales are asked for the crop's own size back.
        assert [c["shape"] for c in calls] == [(120, 90), (120, 90)]

    def test_a_scale_list_of_zeros_falls_back_to_one_scale(self, monkeypatch):
        seg, calls = self._stub(monkeypatch)

        out = seg.scale_maps(np.zeros((64, 64, 3), np.uint8), scales=(0.0, -1.0),
                             polarity=False)

        assert list(out) == [1.0]
        assert len(calls) == 1

    def test_the_inverted_pass_is_skipped_on_a_page_with_no_dark_ground(
            self, monkeypatch):
        # Ungated, inverting turns every speech-bubble border into what looks
        # like white text on black and more than doubles the detected area. With
        # nothing dark to invert, the pass is not run at all rather than run and
        # multiplied by an all-zero gate.
        seg, calls = self._stub(monkeypatch, normal=0.0, inverted=1.0)
        page = np.full((80, 80, 3), 255, np.uint8)

        out = seg.scale_maps(page, scales=(1.0,), polarity=True)

        assert [c["inv"] for c in calls] == [False]
        assert float(out[1.0].max()) == 0.0

    def test_the_inverted_pass_is_kept_where_light_lettering_is_possible(
            self, monkeypatch):
        # The other half of the gate: a dark page with a lighter mark is exactly
        # what the normal pass is blind to, so both polarities run and the
        # inverted evidence survives inside the gate.
        seg, calls = self._stub(monkeypatch, normal=0.0, inverted=1.0)

        out = seg.scale_maps(self._dark_page(80, 80), scales=(1.0,),
                             polarity=True)

        assert [c["inv"] for c in calls] == [False, True]
        assert float(out[1.0].max()) == 1.0
        # Gated, not wholesale: the dark ground away from the stripe keeps none
        # of the inverted map.
        assert float(out[1.0].min()) == 0.0

    def test_a_greyscale_crop_is_accepted(self, monkeypatch):
        seg, _ = self._stub(monkeypatch)

        out = seg.scale_maps(np.zeros((32, 48), np.uint8), scales=(1.0,),
                             polarity=False)

        assert out[1.0].shape == (32, 48)

    def test_an_empty_crop_is_refused(self, monkeypatch):
        seg, _ = self._stub(monkeypatch)

        with pytest.raises(seg.SegUnavailable):
            seg.scale_maps(np.zeros((0, 0, 3), np.uint8))


class TestCleanJobTuning:
    @staticmethod
    def _job(**kw):
        from fox_reader.services.clean_service import CleanJob

        return CleanJob(polygon=[(0, 0), (4, 0), (4, 4)], **kw)

    def test_defaults_match_the_tuning_table(self):
        job = self._job()
        assert (job.method, job.speed, job.tta, job.tile) == (
            "region", DEFAULT_SPEED, DEFAULT_TTA, DEFAULT_TILE)

    def test_a_stale_preset_is_normalised_rather_than_refused(self):
        job = self._job(method="quantum", fill="made-up", speed="turbo",
                        tile=333, glow=1, transport=0).normalised()

        assert job.method == "region"
        assert job.fill == "hybrid-level"
        assert (job.speed, job.tile) == (DEFAULT_SPEED, DEFAULT_TILE)
        assert job.glow is True and job.transport is False

    def test_a_known_setting_survives_normalisation(self):
        job = self._job(method="textseg", fill="telea", speed="best", tta=False,
                        tile=1024).normalised()

        assert (job.method, job.fill, job.speed, job.tta, job.tile) == (
            "textseg", "telea", "best", False, 1024)

    def test_jobs_that_agree_on_tuning_share_one_segmentation(self):
        # This is what stops a three-bubble page paying 40 s three times.
        a = self._job(method="textseg", fill="telea", speed="best", tile=512)
        b = self._job(method="textseg", fill="ns", transport=False, speed="best",
                      tile=512)
        assert a.normalised().seg_key == b.normalised().seg_key

    def test_any_detector_difference_splits_the_segmentation(self):
        base = dict(method="textseg", speed="fastest", tta=True, tile=0, glow=True)
        key = self._job(**base).normalised().seg_key
        for field, value in (("speed", "best"), ("tta", False), ("tile", 512),
                             ("glow", False)):
            other = dict(base)
            other[field] = value
            assert self._job(**other).normalised().seg_key != key, field

    def test_the_key_names_the_method_it_belongs_to(self):
        # It shares a dict with the "ppocr" mask, so it cannot be a bare tuple
        # of tuning values.
        assert self._job(method="textseg").normalised().seg_key[0] == "textseg"


class TestMethodKnobs:
    def test_only_text_seg_reads_the_detector_tuning(self):
        from fox_reader.clean import METHOD_KNOBS

        assert METHOD_KNOBS["region"] == ()
        assert METHOD_KNOBS["ppocr"] == ()
        assert METHOD_KNOBS["textseg"] == ("glow", "speed", "tta", "tile")

    def test_every_method_has_an_entry(self):
        from fox_reader.clean import CLEAN_METHODS, METHOD_KNOBS

        assert set(METHOD_KNOBS) == set(CLEAN_METHODS)

    def test_the_capability_report_carries_them(self):
        from fox_reader.services.clean_service import CleanService

        methods = CleanService(None).available_methods()

        assert [m["id"] for m in methods] == ["region", "ppocr", "textseg"]
        for info in methods:
            assert "knobs" in info
        by_id = {m["id"]: m for m in methods}
        assert by_id["textseg"]["knobs"] == ["glow", "speed", "tta", "tile"]
        assert by_id["region"]["knobs"] == []
        # No detector, so PaddleOCR says why rather than offering itself.
        assert by_id["ppocr"]["available"] is False
        assert by_id["ppocr"]["reason"]


class TestRequestModel:
    def test_the_wire_defaults_match_the_tables(self):
        from fox_reader.models.requests import CleanOptions

        opts = CleanOptions()
        assert (opts.method, opts.fill) == ("region", "hybrid-level")
        assert (opts.speed, opts.tta, opts.tile) == (
            DEFAULT_SPEED, DEFAULT_TTA, DEFAULT_TILE)

    def test_an_unknown_preset_is_accepted_and_normalised_later(self):
        # Deliberately permissive: rejecting at the edge would fail a whole page
        # over a preset name a newer build wrote into the project.
        from fox_reader.models.requests import CleanOptions
        from fox_reader.services.clean_service import CleanJob

        opts = CleanOptions(speed="turbo", tile=99)
        job = CleanJob(polygon=[(0, 0), (1, 0), (1, 1)], speed=opts.speed,
                       tile=opts.tile).normalised()

        assert (job.speed, job.tile) == (DEFAULT_SPEED, DEFAULT_TILE)

    def test_no_overlap_is_carried_on_the_wire(self):
        # The pairing is the backend's, so tile and overlap cannot disagree.
        from fox_reader.models.requests import CleanOptions

        assert "overlap" not in CleanOptions.model_fields


class TestTypesetForwarding:
    """The entry -> `CleanJob` hop, which is where a new field is forgotten."""

    class Item:
        bg_mode = "clean"
        points = [(0, 0), (10, 0), (10, 10)]

        def __init__(self, clean=None):
            self.clean = clean

    class Opts:
        def __init__(self, **kw):
            for key, value in kw.items():
                setattr(self, key, value)

    def test_the_tuning_reaches_the_job(self):
        from fox_reader.typeset import _clean_jobs

        jobs = _clean_jobs([self.Item(self.Opts(
            method="textseg", fill="telea", glow=False, transport=False,
            speed="best", tta=False, tile=1024))])

        assert len(jobs) == 1
        job = jobs[0]
        assert (job.method, job.fill) == ("textseg", "telea")
        assert (job.glow, job.transport) == (False, False)
        assert (job.speed, job.tta, job.tile) == ("best", False, 1024)

    def test_an_entry_saved_before_the_tuning_existed_still_cleans(self):
        from fox_reader.typeset import _clean_jobs

        jobs = _clean_jobs([self.Item(self.Opts(method="textseg",
                                                fill="hybrid-level"))])

        assert (jobs[0].speed, jobs[0].tta, jobs[0].tile) == (
            DEFAULT_SPEED, DEFAULT_TTA, DEFAULT_TILE)

    def test_an_entry_with_no_clean_options_at_all_still_cleans(self):
        from fox_reader.typeset import _clean_jobs

        jobs = _clean_jobs([self.Item(None)])

        assert len(jobs) == 1
        assert jobs[0].method == "region"
        assert (jobs[0].speed, jobs[0].tile) == (DEFAULT_SPEED, DEFAULT_TILE)

    def test_tile_zero_is_forwarded_as_zero(self):
        # `or DEFAULT_TILE` would be right today and wrong the moment the
        # default stops being 0; the fallback is for a *missing* field.
        from fox_reader.typeset import _clean_jobs

        jobs = _clean_jobs([self.Item(self.Opts(method="textseg", tile=0))])

        assert jobs[0].tile == 0

    def test_only_clean_mode_entries_become_jobs(self):
        from fox_reader.typeset import _clean_jobs

        other = self.Item(None)
        other.bg_mode = "auto"
        assert _clean_jobs([other]) == []
