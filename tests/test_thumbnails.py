"""Preview quality, viewer scope and bounded work without loading any models."""
from __future__ import annotations

import asyncio
import os
import struct
from io import BytesIO
from threading import Event, Lock, current_thread
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image, ImageCms, ImageOps
from PIL.PngImagePlugin import PngInfo

from fox_reader.routes import folder
from fox_reader.services import thumbnails
from fox_reader.services.session_service import SessionService
from fox_reader.services.thumbnails import ThumbnailService


@pytest.fixture
def page(tmp_path):
    image = Image.new("RGB", (128, 192))
    image.putdata([((x * 17) % 256, (y * 11) % 256, ((x + y) * 9) % 256)
                   for y in range(192) for x in range(128)])
    path = tmp_path / "page.png"
    image.save(path)
    return path


@pytest.fixture
def service():
    instance = ThumbnailService()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def client(page):
    app = FastAPI()
    app.include_router(folder.router)
    app.state.session = SessionService()
    app.state.session.set_workspace(str(page.parent), str(page.parent / "saved"))
    with TestClient(app) as http:
        yield http
    if service := getattr(app.state, "thumbnails", None):
        service.close()


def preview(service, path, width=32, height=48, fit="contain", revision=""):
    return asyncio.run(service.get(str(path), width, height, fit, revision))


def decoded(data):
    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.copy()


def request(client, filename="page.png", **params):
    return client.get(f"/api/thumbnail/{filename}", params={
        "width": 32, "height": 48, "viewer": "native", **params,
    })


@pytest.mark.parametrize(("mode", "viewer", "enabled"), [
    (None, "native", True), (None, "browser", False),
    ("1", "native", True), ("1", "browser", False),
    ("2", "native", True), ("2", "browser", True),
    ("3", "native", False), ("3", "browser", False),
    ("invalid", "native", False),
])
def test_viewer_scope_never_allocates_worker_when_disabled(client, page, monkeypatch, mode, viewer, enabled):
    if mode is None:
        monkeypatch.delenv("FOX_IMAGE_RENDER_QUALITY", raising=False)
    else:
        monkeypatch.setenv("FOX_IMAGE_RENDER_QUALITY", mode)
    response = request(client, viewer=viewer)
    assert response.status_code == 200
    assert decoded(response.content).size == ((32, 48) if enabled else (128, 192))
    if not enabled:
        assert response.content == page.read_bytes()
        assert not hasattr(client.app.state, "thumbnails")
    assert client.get("/img_serve/page.png").content == page.read_bytes()


def test_home_supplies_the_mode_to_the_shared_frontend(client, monkeypatch):
    client.app.state.needs_setup = False
    client.app.state.config = SimpleNamespace(theme="dark", layout="basic")
    monkeypatch.setenv("FOX_IMAGE_RENDER_QUALITY", "2")
    monkeypatch.setattr(folder.templates, "TemplateResponse", lambda **kwargs: JSONResponse(kwargs["context"]))
    assert client.get("/").json()["image_render_quality_mode"] == 2


@pytest.mark.parametrize("fit,size", [("contain", (32, 48)), ("cover", (48, 48))])
def test_lanczos_pixels_match_reference_and_original_stays_unchanged(service, page, fit, size):
    before = page.read_bytes()
    response = preview(service, page, 48, 48, fit)
    with Image.open(page) as source:
        expected = (ImageOps.fit(source, (48, 48), Image.Resampling.LANCZOS)
                    if fit == "cover" else source.resize((32, 48), Image.Resampling.LANCZOS))
    actual = decoded(response)
    assert actual.size == size
    assert actual.tobytes() == expected.tobytes()
    assert page.read_bytes() == before


def test_keeps_orientation_transparency_and_colour_profile(service, tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image = Image.new("RGBA", (80, 40), (200, 30, 40, 128))
    image.paste((0, 0, 255, 0), (40, 0, 80, 40))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif, icc_profile=profile)
    result = decoded(preview(service, path, 20, 40))
    # The source Image object's EXIF was provided only while saving.
    with Image.open(path) as saved:
        expected = ImageOps.exif_transpose(saved).resize((20, 40), Image.Resampling.LANCZOS)
    assert result.mode == "RGBA"
    assert result.size == (20, 40)
    assert result.tobytes() == expected.tobytes()
    assert result.info["icc_profile"] == profile
    assert result.getexif().get(274, 1) == 1
    assert min(result.getchannel("A").get_flattened_data()) == 0


@pytest.mark.parametrize("mode", ["1", "P"])
def test_palette_and_one_bit_images_are_filtered_instead_of_nearest_neighbour(service, tmp_path, mode):
    image = Image.new(mode, (128, 192))
    if mode == "P":
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    image.putdata([(x + y) % 2 for y in range(192) for x in range(128)])
    path = tmp_path / f"{mode}.png"
    image.save(path)
    result = decoded(preview(service, path, 31, 47)).convert("L")
    assert any(0 < value < 255 for value in result.get_flattened_data())


@pytest.mark.parametrize("mode", ["RGB", "L", "P", "1"])
def test_colour_key_transparency_becomes_filtered_alpha(service, tmp_path, mode):
    image = Image.new(mode, (64, 96))
    if mode == "P":
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    image.paste((255, 255, 255) if mode == "RGB" else 1 if mode == "P" else 255, (32, 0, 64, 96))
    transparent = (0, 0, 0) if mode == "RGB" else 0
    path = tmp_path / "transparent.png"
    image.save(path, transparency=transparent)
    actual = decoded(preview(service, path))
    assert actual.mode in {"RGBA", "LA"}
    alpha = actual.getchannel("A")
    assert alpha.getpixel((0, 0)) == 0
    assert alpha.getpixel((31, 0)) == 255
    assert any(0 < value < 255 for value in alpha.get_flattened_data())


def test_png_colour_metadata_survives_without_an_icc_profile(service, page):
    metadata = PngInfo()
    metadata.add(b"gAMA", struct.pack(">I", 45455))
    metadata.add(b"sRGB", b"\x00")
    chromaticity = (31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
    metadata.add(b"cHRM", struct.pack(">8I", *chromaticity))
    with Image.open(page) as image:
        image.save(page, pnginfo=metadata)
    actual = decoded(preview(service, page))
    assert actual.info["gamma"] == 0.45455
    assert actual.info["srgb"] == 0
    assert actual.info["chromaticity"] == tuple(value / 100_000 for value in chromaticity)


def test_animated_corrupt_large_and_unsupported_pages_fall_back_to_original(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FOX_IMAGE_RENDER_QUALITY", "2")
    frames = [Image.new("RGB", (64, 96), colour) for colour in ("red", "blue")]
    animated = tmp_path / "animated.gif"
    frames[0].save(animated, save_all=True, append_images=frames[1:], duration=100, loop=0)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"invalid image")
    cmyk = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (64, 96)).save(cmyk)
    for path in (animated, corrupt, cmyk):
        response = request(client, path.name)
        assert response.status_code == 200
        assert response.content == path.read_bytes()
    monkeypatch.setattr(thumbnails, "MAX_SOURCE_PIXELS", 10)
    response = request(client)
    assert response.content == (tmp_path / "page.png").read_bytes()


def test_small_original_is_not_upscaled_or_reencoded(service, page):
    assert preview(service, page, 256, 256) is None


def test_saved_variant_fallback_and_later_save_choose_actual_source(client, page, monkeypatch):
    monkeypatch.setenv("FOX_IMAGE_RENDER_QUALITY", "2")
    first = request(client, variant="saved", t="folder.0")
    assert first.content == request(client, variant="original", t="folder.0").content
    destination = page.parent / "saved"
    destination.mkdir()
    Image.new("RGB", (128, 192), "red").save(destination / page.name)
    saved = request(client, variant="saved", t="folder.0")
    assert decoded(saved.content).getpixel((0, 0)) == (255, 0, 0)
    assert request(client, variant="original", t="folder.0").content == first.content


def test_modified_source_and_revision_invalidate_previews(service, page):
    first = preview(service, page, revision="0")
    stat = page.stat()
    Image.new("RGB", (128, 192), "red").save(page)
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = preview(service, page, revision="0")
    assert first != second
    assert decoded(second).getpixel((0, 0)) == (255, 0, 0)
    # A page revision also invalidates a cached entry even on filesystems whose
    # timestamps are coarse. The same metadata then represents changed bytes.
    Image.new("RGB", (128, 192), "blue").save(page)
    third = preview(service, page, revision="1")
    assert decoded(third).getpixel((0, 0)) == (0, 0, 255)


def test_folder_identity_is_part_of_cache_key(service, page, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    Image.new("RGB", (128, 192), "blue").save(other / page.name)
    assert preview(service, page, revision="0") != preview(service, other / page.name, revision="0")


@pytest.mark.parametrize("params", [
    {"width": 0}, {"width": 1025}, {"height": -1}, {"height": 1025},
    {"fit": "stretch"}, {"viewer": "unknown"}, {"t": "x" * 129},
])
def test_rejects_unbounded_or_invalid_requests_without_starting_worker(client, params):
    assert request(client, **params).status_code == 422
    assert not hasattr(client.app.state, "thumbnails")


def test_resolution_errors_match_full_image_route(client, tmp_path):
    for name in ("../secret.png", "..%2Fsecret.png", "..%5Csecret.png", "absent.png"):
        assert request(client, name).status_code in (400, 404)
    client.app.state.session.current_dir = ""
    assert request(client).status_code == 400


def test_cache_hits_and_entry_limit_avoid_or_repeat_only_needed_work(page, monkeypatch):
    service = ThumbnailService(max_entries=1)
    calls = []
    render = service._render
    monkeypatch.setattr(service, "_render", lambda key: (calls.append(key.width), render(key))[1])
    try:
        first = preview(service, page)
        assert preview(service, page) == first
        assert calls == [32]
        preview(service, page, 31)
        assert preview(service, page) == first
        assert calls == [32, 31, 32]
    finally:
        service.close()


def test_encoded_cache_byte_limit_is_enforced(page, service):
    first = preview(service, page, 32)
    second = preview(service, page, 31)
    limited = ThumbnailService(max_bytes=len(first) + len(second) - 1)
    try:
        assert preview(limited, page, 32) == first
        assert preview(limited, page, 31) == second
        assert limited._cache_bytes <= limited._max_bytes
        assert len(limited._cache) == 1
    finally:
        limited.close()


def test_coalesces_work_limits_concurrency_and_survives_one_cancelled_request(page, monkeypatch):
    service = ThumbnailService(max_pending=2)
    entered, release = Event(), Event()
    lock = Lock()
    calls, active, peak = [], 0, 0
    render = service._render

    def blocked(key):
        nonlocal active, peak
        with lock:
            calls.append((key.width, current_thread().name))
            active += 1
            peak = max(peak, active)
        entered.set()
        release.wait(3)
        try:
            return render(key)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(service, "_render", blocked)

    async def scenario():
        first = asyncio.create_task(service.get(str(page), 32, 48))
        assert await asyncio.to_thread(entered.wait, 2)
        duplicate = asyncio.create_task(service.get(str(page), 32, 48))
        queued = asyncio.create_task(service.get(str(page), 31, 48))
        await asyncio.sleep(0)
        assert await service.get(str(page), 30, 48) is None
        assert not duplicate.done()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await duplicate
        assert await queued
        assert len(calls) == 2
        assert peak == 1
        assert len({name for _, name in calls}) == 1
        assert await service.get(str(page), 32, 48)
        assert len(calls) == 2

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        service.close()


def test_closing_discards_cache_and_refuses_new_work(service, page):
    assert preview(service, page)
    service.close()
    assert not service._cache
    assert service._cache_bytes == 0
    assert preview(service, page) is None
