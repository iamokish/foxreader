"""Tests for Pydantic request/response models."""
import pytest
from fox_reader.models.requests import (
    FreeformRequest, CropRequest, BubbleRequest, Point, Region,
    SplitRequest, TranslationRequest, DataItem, ProcessImageRequest,
    FolderRequest, LoadMLRequest,
)
from fox_reader.models.responses import TranslationResult


class TestFreeformRequest:
    def test_valid(self):
        req = FreeformRequest(filename="test.jpg", points=[{"x": 0, "y": 0}], lang="jap", isGrayScale=False)
        assert req.filename == "test.jpg"

    def test_missing_field(self):
        with pytest.raises(Exception):
            FreeformRequest(filename="test.jpg")


class TestCropRequest:
    def test_valid(self):
        req = CropRequest(filename="test.jpg", x=10, y=20, width=100, height=50, lang="eng", isGrayScale=True)
        assert req.x == 10
        assert req.width == 100


class TestBubbleRequest:
    def test_valid(self):
        req = BubbleRequest(filename="page001.jpg", isGrayScale=False)
        assert req.filename == "page001.jpg"
        assert req.isGrayScale is False


class TestPoint:
    def test_valid(self):
        p = Point(x=10, y=20)
        assert p.x == 10
        assert p.y == 20


class TestRegion:
    def test_valid(self):
        r = Region(type="polygon", coords=[Point(x=0, y=0), Point(x=10, y=0), Point(x=5, y=10)])
        assert r.type == "polygon"
        assert len(r.coords) == 3


class TestSplitRequest:
    def test_valid(self):
        req = SplitRequest(
            region=Region(type="polygon", coords=[Point(x=0, y=0)]),
            split_object=[Point(x=5, y=5)],
            isGrayScale=False,
            filename="test.jpg",
        )
        assert req.filename == "test.jpg"


class TestTranslationRequest:
    def test_valid(self):
        req = TranslationRequest(text="Hello", source_lang="eng")
        assert req.text == "Hello"


class TestDataItem:
    def test_valid(self):
        item = DataItem(og_text="原", text="translated", fontfile="comic.ttf", text_align="center", points=[(0, 0), (100, 0), (100, 50)])
        assert item.og_text == "原"
        assert len(item.points) == 3


class TestProcessImageRequest:
    def test_valid(self):
        req = ProcessImageRequest(
            filename="page.jpg",
            data=[DataItem(og_text="a", text="b", fontfile="f.ttf", text_align="left", points=[(0, 0)])],
        )
        assert len(req.data) == 1


class TestFolderRequest:
    def test_valid(self):
        req = FolderRequest(path="/home/user/manga")
        assert req.path == "/home/user/manga"


class TestLoadMLRequest:
    def test_valid(self):
        req = LoadMLRequest(lang="chinese")
        assert req.lang == "chinese"


class TestTranslationResult:
    def test_to_dict(self):
        result = TranslationResult(
            code=200, id=123, data="Hello", source_lang="JAP",
            target_lang="EN", method="Google", alternatives=["Hi", "Hey"],
        )
        d = result.to_dict()
        assert d["code"] == 200
        assert d["data"] == "Hello"
        assert d["alternatives"] == ["Hi", "Hey"]
        assert "message" not in d

    def test_to_dict_with_message(self):
        result = TranslationResult(
            code=400, id=1, data="", source_lang="ENG",
            target_lang="EN", method="Google", message="Error",
        )
        d = result.to_dict()
        assert d["message"] == "Error"

    def test_defaults(self):
        result = TranslationResult(
            code=200, id=1, data="test", source_lang="eng",
            target_lang="en", method="test",
        )
        assert result.alternatives == []
        assert result.message == ""
