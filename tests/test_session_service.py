"""Tests for SessionService."""
import pytest
from fox_reader.services.session_service import SessionService


class TestSessionService:
    def test_initial_state(self):
        svc = SessionService()
        assert svc.current_dir is None
        assert svc.has_active_tabs() is False
        assert svc.get_active_tabs() == []

    def test_current_dir_setter(self):
        svc = SessionService()
        svc.current_dir = "/home/user/manga"
        assert svc.current_dir == "/home/user/manga"

    def test_current_dir_strips_quotes(self):
        svc = SessionService()
        svc.current_dir = "'/path/to/dir'"
        assert svc.current_dir == "/path/to/dir"
        svc.current_dir = '"/another/path"'
        assert svc.current_dir == "/another/path"

    def test_current_dir_normalizes_slashes(self):
        svc = SessionService()
        svc.current_dir = "C:\\Users\\test\\manga"
        assert svc.current_dir == "C:/Users/test/manga"

    def test_current_dir_set_none(self):
        svc = SessionService()
        svc.current_dir = "/path"
        svc.current_dir = None
        assert svc.current_dir is None

    def test_dest_dir_defaults_to_none(self):
        svc = SessionService()
        assert svc.dest_dir is None
        assert svc.save_dir is None
        # No folder loaded is not "writes in place": there is nowhere to write.
        assert svc.writes_in_place is False

    def test_dest_dir_normalizes_like_current_dir(self):
        svc = SessionService()
        svc.dest_dir = "'C:\\Users\\test\\manga\\fox_tled'"
        assert svc.dest_dir == "C:/Users/test/manga/fox_tled"

    def test_save_dir_falls_back_to_source(self):
        svc = SessionService()
        svc.set_workspace("/home/user/manga")
        assert svc.save_dir == "/home/user/manga"
        assert svc.writes_in_place is True

    def test_save_dir_is_the_destination_when_set(self):
        svc = SessionService()
        svc.set_workspace("/home/user/manga", "/home/user/manga/fox_tled")
        assert svc.current_dir == "/home/user/manga"
        assert svc.save_dir == "/home/user/manga/fox_tled"
        assert svc.writes_in_place is False

    def test_destination_equal_to_source_writes_in_place(self):
        svc = SessionService()
        svc.set_workspace("/home/user/manga", "'/home/user/manga'")
        assert svc.writes_in_place is True

    def test_set_workspace_clears_a_previous_destination(self):
        svc = SessionService()
        svc.set_workspace("/a", "/a/out")
        svc.set_workspace("/b")
        assert svc.dest_dir is None
        assert svc.save_dir == "/b"

    def test_bubble_cache(self):
        svc = SessionService()
        assert svc.get_cached_bubble("page.jpg", "normal") is None

        svc.set_cached_bubble("page.jpg", "normal", [{"type": "polygon"}])
        cached = svc.get_cached_bubble("page.jpg", "normal")
        assert cached == [{"type": "polygon"}]

        # Different cache key returns None
        assert svc.get_cached_bubble("page.jpg", "gray_scale") is None

    def test_bubble_cache_multiple_keys(self):
        svc = SessionService()
        svc.set_cached_bubble("page.jpg", "normal", [1])
        svc.set_cached_bubble("page.jpg", "gray_scale", [2])
        assert svc.get_cached_bubble("page.jpg", "normal") == [1]
        assert svc.get_cached_bubble("page.jpg", "gray_scale") == [2]

    def test_clear_bubble_cache(self):
        svc = SessionService()
        svc.set_cached_bubble("page.jpg", "normal", [1])
        svc.clear_bubble_cache()
        assert svc.get_cached_bubble("page.jpg", "normal") is None

    def test_add_remove_tab(self):
        svc = SessionService()
        fake_ws = object()
        svc.add_tab(fake_ws)
        assert svc.has_active_tabs() is True
        assert svc.get_active_tabs() == [fake_ws]

        svc.remove_tab(fake_ws)
        assert svc.has_active_tabs() is False

    def test_remove_nonexistent_tab(self):
        svc = SessionService()
        fake_ws = object()
        svc.remove_tab(fake_ws)  # Should not raise
        assert svc.has_active_tabs() is False

    def test_multiple_tabs(self):
        svc = SessionService()
        ws1, ws2 = object(), object()
        svc.add_tab(ws1)
        svc.add_tab(ws2)
        assert len(svc.get_active_tabs()) == 2
        svc.remove_tab(ws1)
        assert len(svc.get_active_tabs()) == 1
        assert svc.get_active_tabs() == [ws2]
