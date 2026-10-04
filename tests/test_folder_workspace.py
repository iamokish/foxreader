"""Source/destination folders: permissions, remembered pairings, and the routes.

The point of all of this is that the original scan survives a save, so the tests
that matter most are the ones that watch a file *not* change: `_original_kept`
below is asserted on after every write path.
"""
from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from fox_reader import fsaccess
from fox_reader.routes import folder as folder_routes
from fox_reader.routes import inpaint as inpaint_routes
from fox_reader.services.session_service import SessionService
from fox_reader.services.progress import ProgressRegistry
from fox_reader.workspaces import MAX_ENTRIES, WorkspaceStore


# --------------------------------------------------------------------- fixtures


def _page(directory, name: str, colour: str = "white") -> str:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=colour).save(directory / name)
    return name


@pytest.fixture
def pages(tmp_path):
    """A source folder with three pages in it, out of lexical order on purpose."""
    source = tmp_path / "scan"
    _page(source, "p1.png")
    _page(source, "p2.png")
    _page(source, "p10.png")
    return source


# ---------------------------------------------------------------- path handling


class TestPaths:
    def test_normalize_dir_unquotes_and_flips_slashes(self):
        assert fsaccess.normalize_dir("'C:\\Users\\me\\scan'") == "C:/Users/me/scan"
        assert fsaccess.normalize_dir('"  /home/me/scan  "') == "/home/me/scan"

    def test_normalize_dir_leaves_a_relative_path_relative(self):
        # The session has always stored what it was given; absolutising belongs to
        # `resolve_dir`, at the route edge.
        assert fsaccess.normalize_dir("scan/ch1") == "scan/ch1"

    def test_resolve_dir_is_absolute_with_forward_slashes(self, tmp_path):
        resolved = fsaccess.resolve_dir(str(tmp_path).replace("/", "\\"))
        assert "\\" not in resolved
        assert os.path.isabs(resolved)
        assert fsaccess.resolve_dir("") == ""

    def test_list_images_is_in_reading_order_and_skips_dotfiles(self, pages):
        (pages / ".fox_tmp_abcd_p1.png").write_bytes(b"not a page")
        (pages / "notes.txt").write_text("ignored")
        assert fsaccess.list_images(str(pages)) == ["p1.png", "p2.png", "p10.png"]

    def test_list_images_on_a_missing_folder_is_empty(self, tmp_path):
        assert fsaccess.list_images(str(tmp_path / "nope")) == []

    def test_same_dir_matches_spellings_of_one_folder(self, pages):
        assert fsaccess.same_dir(str(pages), str(pages) + "/") is True
        assert fsaccess.same_dir(str(pages), str(pages / "sub")) is False
        assert fsaccess.same_dir("", str(pages)) is False

    def test_same_dir_falls_back_to_strings_for_a_missing_folder(self, tmp_path):
        missing = str(tmp_path / "out")
        assert fsaccess.same_dir(missing, missing) is True

    def test_contains(self, pages):
        assert fsaccess.contains(str(pages), str(pages / "p1.png")) is True
        assert fsaccess.contains(str(pages), str(pages)) is True
        assert fsaccess.contains(str(pages), str(pages.parent / "elsewhere.png")) is False

    def test_safe_join_refuses_anything_with_a_path_in_it(self, pages):
        root = str(pages)
        assert fsaccess.safe_join(root, "p1.png") == os.path.join(root, "p1.png")
        assert fsaccess.safe_join(root, "../secrets.png") is None
        assert fsaccess.safe_join(root, "..\\secrets.png") is None
        assert fsaccess.safe_join(root, "sub/p1.png") is None
        assert fsaccess.safe_join(root, "..") is None
        assert fsaccess.safe_join(root, "") is None
        assert fsaccess.safe_join("", "p1.png") is None

    def test_default_dest_is_a_subfolder(self, pages):
        assert fsaccess.default_dest(str(pages)) == (
            f"{fsaccess.resolve_dir(str(pages))}/fox_tled"
        )
        assert fsaccess.default_dest("") == ""


# ------------------------------------------------------------------ permissions


class TestDescribeDir:
    def test_a_usable_source(self, pages):
        report = fsaccess.describe_dir(str(pages), count_images=True)
        assert report.is_dir and report.exists and not report.is_file
        assert report.readable and report.writable and report.executable
        assert report.image_count == 3
        assert report.reason == ""
        assert report.ok_as_source and report.ok_as_dest

    def test_an_empty_source_says_why_it_is_unusable(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        report = fsaccess.describe_dir(str(empty), count_images=True)
        assert report.image_count == 0
        assert report.reason == "No images in this folder."
        # Still a fine *destination*: emptiness is only a problem for a source.
        assert report.ok_as_dest is True

    def test_a_missing_folder_with_a_writable_parent_can_be_created(self, tmp_path):
        report = fsaccess.describe_dir(str(tmp_path / "fox_tled"))
        assert report.exists is False
        assert report.parent_exists and report.parent_writable
        assert report.can_create is True
        assert report.ok_as_dest is True
        assert report.ok_as_source is False
        assert report.reason == "This folder does not exist yet."

    def test_a_folder_two_levels_down_cannot_be_created(self, tmp_path):
        report = fsaccess.describe_dir(str(tmp_path / "a" / "b"))
        assert report.can_create is False
        assert report.ok_as_dest is False

    def test_a_file_is_not_a_folder(self, pages):
        report = fsaccess.describe_dir(str(pages / "p1.png"))
        assert report.is_file is True
        assert report.is_dir is False
        assert report.reason == "This is a file, not a folder."
        assert report.ok_as_source is False and report.ok_as_dest is False

    def test_an_empty_path_asks_for_one(self):
        report = fsaccess.describe_dir("")
        assert report.reason == "Enter a folder path."
        assert report.ok_as_source is False and report.ok_as_dest is False

    def test_the_write_probe_leaves_nothing_behind(self, tmp_path):
        fsaccess.describe_dir(str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_to_dict_carries_the_derived_answers(self, pages):
        data = fsaccess.describe_dir(str(pages), count_images=True).to_dict()
        assert data["ok_as_source"] is True
        assert data["ok_as_dest"] is True
        assert data["can_create"] is False
        assert data["image_count"] == 3

    def test_ensure_dir_creates_and_is_idempotent(self, tmp_path):
        target = tmp_path / "fox_tled"
        first = fsaccess.ensure_dir(str(target))
        assert first.is_dir and first.writable
        assert target.is_dir()
        second = fsaccess.ensure_dir(str(target))
        assert second.is_dir and second.writable

    def test_ensure_dir_reports_a_failure_as_a_report(self, pages):
        # A path whose parent is a file can never be made into a directory.
        report = fsaccess.ensure_dir(str(pages / "p1.png" / "out"))
        assert report.ok_as_dest is False
        assert report.reason


# ------------------------------------------------------------ remembered pairs


class TestWorkspaceStore:
    def test_round_trips_through_the_file(self, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        assert store.dest_for("/a") is None
        assert store.last_source() is None

        store.remember(str(tmp_path / "a"), str(tmp_path / "a" / "fox_tled"))
        reopened = WorkspaceStore(tmp_path / "config")
        assert reopened.dest_for(str(tmp_path / "a")) == fsaccess.resolve_dir(
            str(tmp_path / "a" / "fox_tled")
        )
        assert reopened.last_source() == fsaccess.resolve_dir(str(tmp_path / "a"))

    def test_re_remembering_moves_a_source_to_the_front(self, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        store.remember("/a", "/a/out")
        store.remember("/b", "/b/out")
        assert store.last_source() == fsaccess.resolve_dir("/b")

        store.remember("/a", "/a/other")
        assert store.last_source() == fsaccess.resolve_dir("/a")
        assert store.dest_for("/a") == fsaccess.resolve_dir("/a/other")
        # Moved, not duplicated.
        assert len(store.all()) == 2

    def test_ignores_a_pairing_with_a_blank_side(self, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        store.remember("", "/a/out")
        store.remember("/a", "")
        assert store.all() == []

    def test_forget(self, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        store.remember("/a", "/a/out")
        store.forget("/a")
        assert store.dest_for("/a") is None
        store.forget("/a")  # Twice is not an error.

    def test_caps_the_list(self, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        for index in range(MAX_ENTRIES + 5):
            store.remember(f"/src{index}", f"/dst{index}")
        assert len(store.all()) == MAX_ENTRIES
        # The oldest fell off the end, the newest is at the front.
        assert store.last_source() == fsaccess.resolve_dir(f"/src{MAX_ENTRIES + 4}")

    def test_survives_a_corrupt_file(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "workspaces.yaml").write_text("workspaces: [oh no: :\n", encoding="utf-8")

        store = WorkspaceStore(config)
        assert store.all() == []
        # And it rewrote the file, so the next start is clean.
        store.remember("/a", "/a/out")
        assert WorkspaceStore(config).dest_for("/a") == fsaccess.resolve_dir("/a/out")

    def test_skips_only_the_unusable_records(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "workspaces.yaml").write_text(
            "workspaces:\n"
            "  - source: /good\n"
            "    dest: /good/out\n"
            "  - 'not a mapping'\n"
            "  - source: ''\n"
            "    dest: /orphan\n",
            encoding="utf-8",
        )
        store = WorkspaceStore(config)
        assert [entry["source"] for entry in store.all()] == [
            fsaccess.resolve_dir("/good")
        ]

    def test_a_read_only_config_dir_does_not_raise(self, tmp_path, monkeypatch):
        store = WorkspaceStore(tmp_path / "config")

        def deny(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(type(store.path), "open", deny, raising=False)
        store.remember("/a", "/a/out")  # Logged, not raised.
        assert store.dest_for("/a") == fsaccess.resolve_dir("/a/out")


# ---------------------------------------------------------------------- routes


def _folder_app(session: SessionService, workspaces=None) -> FastAPI:
    app = FastAPI()
    app.include_router(folder_routes.router)
    app.state.session = session
    app.state.workspaces = workspaces
    return app


class TestFolderRoutes:
    def test_load_folder_defaults_to_the_fox_tled_subfolder(self, pages):
        session = SessionService()
        client = TestClient(_folder_app(session))

        response = client.post("/api/load_folder", json={"path": str(pages)})
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "success"
        assert body["files"] == ["p1.png", "p2.png", "p10.png"]
        assert len(body["dimensions"]) == 3
        assert body["dimensions"][0] == [12, 12]
        assert body["dest"].endswith("/fox_tled")
        assert body["same_dir"] is False
        assert body["saved"] == []
        assert (pages / "fox_tled").is_dir()
        assert session.save_dir == body["dest"]
        assert session.writes_in_place is False

    def test_load_folder_takes_an_explicit_destination(self, pages, tmp_path):
        out = tmp_path / "elsewhere"
        session = SessionService()
        client = TestClient(_folder_app(session))

        body = client.post(
            "/api/load_folder", json={"path": str(pages), "dest": str(out)}
        ).json()

        assert body["dest"] == fsaccess.resolve_dir(str(out))
        assert out.is_dir()
        assert (pages / "fox_tled").exists() is False

    def test_load_folder_reports_pages_already_saved(self, pages):
        out = pages / "fox_tled"
        _page(out, "p2.png", colour="black")

        body = TestClient(_folder_app(SessionService())).post(
            "/api/load_folder", json={"path": str(pages)}
        ).json()

        assert body["saved"] == ["p2.png"]

    def test_load_folder_accepts_the_source_as_its_own_destination(self, pages):
        session = SessionService()
        body = TestClient(_folder_app(session)).post(
            "/api/load_folder", json={"path": str(pages), "dest": str(pages)}
        ).json()

        assert body["same_dir"] is True
        # No Original/Saved switch to offer: the two sides would be one file.
        assert body["saved"] == []
        assert session.writes_in_place is True
        assert (pages / "fox_tled").exists() is False

    def test_load_folder_can_be_told_not_to_create_the_destination(self, pages):
        response = TestClient(_folder_app(SessionService())).post(
            "/api/load_folder",
            json={"path": str(pages), "dest": str(pages / "fox_tled"),
                  "create_dest": False},
        )
        assert response.status_code == 400
        assert response.json()["status"] == "error"
        assert (pages / "fox_tled").exists() is False

    def test_load_folder_keeps_the_old_wording_for_a_bad_path(self, tmp_path):
        response = TestClient(_folder_app(SessionService())).post(
            "/api/load_folder", json={"path": str(tmp_path / "nope")}
        )
        assert response.status_code == 400
        assert response.json() == {"status": "error", "error": "Invalid Path"}

    def test_load_folder_rejects_a_file(self, pages):
        response = TestClient(_folder_app(SessionService())).post(
            "/api/load_folder", json={"path": str(pages / "p1.png")}
        )
        assert response.status_code == 400
        assert response.json()["error"] == "Invalid Path"

    def test_load_folder_prefers_a_remembered_destination(self, pages, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        remembered = tmp_path / "remembered"
        store.remember(str(pages), str(remembered))

        body = TestClient(_folder_app(SessionService(), store)).post(
            "/api/load_folder", json={"path": str(pages)}
        ).json()

        assert body["dest"] == fsaccess.resolve_dir(str(remembered))
        assert (pages / "fox_tled").exists() is False

    def test_load_folder_remembers_what_it_was_given(self, pages, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        out = tmp_path / "chosen"

        TestClient(_folder_app(SessionService(), store)).post(
            "/api/load_folder", json={"path": str(pages), "dest": str(out)}
        )

        assert store.dest_for(str(pages)) == fsaccess.resolve_dir(str(out))

    def test_load_folder_clears_the_bubble_cache(self, pages):
        session = SessionService()
        session.set_cached_bubble("p1.png", "normal", [1])
        TestClient(_folder_app(session)).post(
            "/api/load_folder", json={"path": str(pages)}
        )
        assert session.get_cached_bubble("p1.png", "normal") is None

    def test_inspect_reports_both_sides_without_creating_anything(self, pages):
        body = TestClient(_folder_app(SessionService())).post(
            "/api/folder/inspect", json={"source": str(pages)}
        ).json()

        assert body["source"]["ok_as_source"] is True
        assert body["source"]["image_count"] == 3
        assert body["dest"]["path"].endswith("/fox_tled")
        assert body["dest"]["can_create"] is True
        assert body["dest"]["ok_as_dest"] is True
        assert body["default_dest"].endswith("/fox_tled")
        assert body["same_dir"] is False
        assert (pages / "fox_tled").exists() is False

    def test_inspect_flags_a_destination_that_is_the_source(self, pages):
        body = TestClient(_folder_app(SessionService())).post(
            "/api/folder/inspect", json={"source": str(pages), "dest": str(pages)}
        ).json()
        assert body["same_dir"] is True

    def test_inspect_offers_the_remembered_destination(self, pages, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        store.remember(str(pages), str(tmp_path / "remembered"))

        body = TestClient(_folder_app(SessionService(), store)).post(
            "/api/folder/inspect", json={"source": str(pages)}
        ).json()

        assert body["remembered_dest"] == fsaccess.resolve_dir(
            str(tmp_path / "remembered")
        )
        assert body["dest"]["path"] == body["remembered_dest"]

    def test_inspect_on_an_empty_source_asks_for_a_path(self):
        body = TestClient(_folder_app(SessionService())).post(
            "/api/folder/inspect", json={"source": ""}
        ).json()
        assert body["source"]["ok_as_source"] is False
        assert body["dest"]["path"] == ""
        assert body["same_dir"] is False

    def test_state_before_and_after_a_load(self, pages, tmp_path):
        store = WorkspaceStore(tmp_path / "config")
        session = SessionService()
        client = TestClient(_folder_app(session, store))

        before = client.get("/api/folder/state").json()
        assert before["source"] == ""
        assert before["same_dir"] is False
        assert before["default_dest_name"] == "fox_tled"

        client.post("/api/load_folder", json={"path": str(pages)})
        after = client.get("/api/folder/state").json()
        assert after["source"] == fsaccess.resolve_dir(str(pages))
        assert after["dest"].endswith("/fox_tled")
        assert after["last_source"] == after["source"]

    def test_img_serve_prefers_the_saved_copy_when_asked(self, pages):
        _page(pages / "fox_tled", "p1.png", colour="black")
        session = SessionService()
        client = TestClient(_folder_app(session))
        client.post("/api/load_folder", json={"path": str(pages)})

        original = client.get("/img_serve/p1.png")
        saved = client.get("/img_serve/p1.png", params={"variant": "saved"})

        assert original.status_code == 200
        assert saved.status_code == 200
        assert saved.content != original.content
        assert original.content == (pages / "p1.png").read_bytes()
        assert saved.content == (pages / "fox_tled" / "p1.png").read_bytes()

    def test_img_serve_falls_back_to_the_original(self, pages):
        session = SessionService()
        client = TestClient(_folder_app(session))
        client.post("/api/load_folder", json={"path": str(pages)})

        # p2 has no saved copy: a stale switch must not blank the viewer.
        response = client.get("/img_serve/p2.png", params={"variant": "saved"})
        assert response.status_code == 200
        assert response.content == (pages / "p2.png").read_bytes()

    def test_img_serve_refuses_to_leave_the_folder(self, pages, tmp_path):
        secret = tmp_path / "secret.png"
        _page(tmp_path, "secret.png", colour="red")
        assert secret.is_file()

        client = TestClient(_folder_app(SessionService()))
        client.post("/api/load_folder", json={"path": str(pages)})

        for name in ("../secret.png", "..%2Fsecret.png", "sub/p1.png"):
            response = client.get(f"/img_serve/{name}")
            assert response.status_code in (400, 404), name
            assert b"\x89PNG" not in response.content, name

    def test_img_serve_needs_a_folder(self):
        response = TestClient(_folder_app(SessionService())).get("/img_serve/p1.png")
        assert response.status_code == 400

    def test_img_serve_404s_for_a_page_that_is_not_there(self, pages):
        client = TestClient(_folder_app(SessionService()))
        client.post("/api/load_folder", json={"path": str(pages)})
        assert client.get("/img_serve/ghost.png").status_code == 404


# -------------------------------------------------------------- saving a page


class StubInpaint:
    """Stands in for the real typesetter: writes a marker file, records its args."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.saves: list[tuple[str, str]] = []

    def generate(self, image_path, dataitems, task=None, save_path=None):
        target = save_path or image_path
        self.calls.append((image_path, target))
        with open(target, "wb") as f:
            f.write(b"typeset")
        return target

    def save_preview(self, token, filename, image_path):
        if token != "good-token":
            raise LookupError("this preview is out of date")
        self.saves.append((filename, image_path))
        with open(image_path, "wb") as f:
            f.write(b"preview")
        return image_path


def _inpaint_app(session: SessionService, inpaint) -> FastAPI:
    app = FastAPI()
    app.include_router(inpaint_routes.router)
    app.state.session = session
    app.state.inpaint = inpaint
    app.state.progress = ProgressRegistry()
    return app


@pytest.fixture
def loaded(pages):
    """A session pointed at `pages` with `pages/fox_tled` as its destination."""
    session = SessionService()
    dest = pages / "fox_tled"
    dest.mkdir()
    session.set_workspace(str(pages), str(dest))
    return session


class TestSaving:
    def test_generate_writes_the_destination_and_leaves_the_original(self, pages, loaded):
        before = (pages / "p1.png").read_bytes()
        inpaint = StubInpaint()

        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/generate", json={"filename": "p1.png", "data": []}
        )
        body = response.json()

        assert response.status_code == 200
        assert body["saved"] is True
        assert body["in_place"] is False
        assert body["dest"] == loaded.save_dir
        assert (pages / "fox_tled" / "p1.png").read_bytes() == b"typeset"
        assert (pages / "p1.png").read_bytes() == before

        # Read the original, write the copy -- never the other way round, or a
        # second save would letter on top of the first.
        source, target = inpaint.calls[0]
        assert source.endswith("p1.png") and "fox_tled" not in source
        assert "fox_tled" in target.replace("\\", "/")

    def test_generate_reads_the_original_even_on_a_re_save(self, pages, loaded):
        _page(pages / "fox_tled", "p1.png", colour="black")
        inpaint = StubInpaint()

        TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/generate",
            json={"filename": "p1.png", "data": [], "overwrite": True},
        )

        source, _ = inpaint.calls[0]
        assert "fox_tled" not in source.replace("\\", "/")

    def test_generate_refuses_to_replace_without_being_told_to(self, pages, loaded):
        _page(pages / "fox_tled", "p1.png", colour="black")
        existing = (pages / "fox_tled" / "p1.png").read_bytes()
        inpaint = StubInpaint()

        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/generate", json={"filename": "p1.png", "data": []}
        )

        assert response.status_code == 409
        assert response.headers["X-Fox-Code"] == "exists"
        assert "p1.png" in response.json()["detail"]
        # Refused *before* the render: no work done, nothing touched.
        assert inpaint.calls == []
        assert (pages / "fox_tled" / "p1.png").read_bytes() == existing

    def test_generate_replaces_when_told_to(self, pages, loaded):
        _page(pages / "fox_tled", "p1.png", colour="black")
        inpaint = StubInpaint()

        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/generate",
            json={"filename": "p1.png", "data": [], "overwrite": True},
        )

        assert response.status_code == 200
        assert (pages / "fox_tled" / "p1.png").read_bytes() == b"typeset"

    def test_generate_in_place_still_needs_an_overwrite(self, pages):
        session = SessionService()
        session.set_workspace(str(pages))
        inpaint = StubInpaint()
        client = TestClient(_inpaint_app(session, inpaint))

        refused = client.post("/inpaint/generate", json={"filename": "p1.png", "data": []})
        assert refused.status_code == 409
        assert refused.headers["X-Fox-Code"] == "exists"

        allowed = client.post(
            "/inpaint/generate",
            json={"filename": "p1.png", "data": [], "overwrite": True},
        )
        assert allowed.status_code == 200
        assert allowed.json()["in_place"] is True
        assert (pages / "p1.png").read_bytes() == b"typeset"

    def test_generate_rejects_a_filename_with_a_path_in_it(self, loaded):
        inpaint = StubInpaint()
        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/generate", json={"filename": "../p1.png", "data": []}
        )
        assert response.status_code == 400
        assert inpaint.calls == []

    def test_generate_needs_a_loaded_folder(self):
        response = TestClient(_inpaint_app(SessionService(), StubInpaint())).post(
            "/inpaint/generate", json={"filename": "p1.png", "data": []}
        )
        assert response.status_code == 400

    def test_generate_without_models_is_a_503(self, loaded):
        response = TestClient(_inpaint_app(loaded, None)).post(
            "/inpaint/generate", json={"filename": "p1.png", "data": []}
        )
        assert response.status_code == 503

    def test_save_preview_lands_in_the_destination(self, pages, loaded):
        before = (pages / "p1.png").read_bytes()
        inpaint = StubInpaint()

        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/save-preview",
            json={"filename": "p1.png", "token": "good-token"},
        )
        body = response.json()

        assert response.status_code == 200
        assert body["saved"] is True
        assert (pages / "fox_tled" / "p1.png").read_bytes() == b"preview"
        assert (pages / "p1.png").read_bytes() == before

    def test_save_preview_refuses_to_replace_without_being_told_to(self, pages, loaded):
        _page(pages / "fox_tled", "p1.png", colour="black")
        inpaint = StubInpaint()

        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/save-preview",
            json={"filename": "p1.png", "token": "good-token"},
        )

        assert response.status_code == 409
        assert response.headers["X-Fox-Code"] == "exists"
        assert inpaint.saves == []

    def test_save_preview_distinguishes_a_stale_token(self, loaded):
        inpaint = StubInpaint()
        response = TestClient(_inpaint_app(loaded, inpaint)).post(
            "/inpaint/save-preview",
            json={"filename": "p1.png", "token": "who-knows"},
        )
        # Same status as the exists conflict; the header is what tells them apart,
        # because one is answerable by the user and the other is not.
        assert response.status_code == 409
        assert response.headers["X-Fox-Code"] == "stale_preview"
