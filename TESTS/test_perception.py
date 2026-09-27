"""
Tests for cognitive_core.perception.
"""

import pytest

from cognitive_core.perception import PerceptEvent, PerceptionLayer


class TestPerceptEvent:
    def test_dataclass_basic_fields(self):
        event = PerceptEvent(raw_input="Hello", normalized_text="hello", input_type="text")
        assert event.raw_input == "Hello"
        assert event.normalized_text == "hello"
        assert event.input_type == "text"
        assert event.source == "user"
        assert event.confidence == 1.0

    def test_to_dict_includes_all_fields(self):
        event = PerceptEvent(raw_input="test", normalized_text="test", input_type="text")
        data = event.to_dict()
        assert data["raw_input"] == "test"
        assert data["normalized_text"] == "test"
        assert data["input_type"] == "text"
        assert "timestamp" in data

    def test_to_dict_includes_image_asset_when_present(self):
        from cognitive_core.vision.interfaces import ImageAsset
        asset = ImageAsset(path="/tmp/test.png", format="PNG", size_bytes=100, width=10, height=10)
        event = PerceptEvent(raw_input="/tmp/test.png", normalized_text="[VISION]", input_type="vision", image_asset=asset)
        data = event.to_dict()
        assert "image_asset" in data
        assert data["image_asset"]["path"] == "/tmp/test.png"


class TestPerceptionLayer:
    def test_process_text_basic(self):
        layer = PerceptionLayer()
        event = layer.process_text("Come funziona il modello?")
        assert isinstance(event, PerceptEvent)
        assert event.normalized_text == "come funziona il modello?"
        assert event.input_type == "text"
        assert event.source == "user"

    def test_process_text_normalizes_whitespace(self):
        layer = PerceptionLayer()
        event = layer.process("  hello   world  ")
        assert event.normalized_text == "hello world"

    def test_detect_structural_type_text(self):
        layer = PerceptionLayer()
        event = layer.process_text("Qualsiasi testo non vuoto")
        assert event.input_type == "text"

    def test_detect_structural_type_empty(self):
        layer = PerceptionLayer()
        event = layer.process_text("   ")
        assert event.input_type == "empty"

    def test_process_none_raises_error(self):
        layer = PerceptionLayer()
        with pytest.raises(ValueError, match="None"):
            layer.process(None)

    def test_process_text_none_raises_error(self):
        layer = PerceptionLayer()
        with pytest.raises(ValueError, match="None"):
            layer.process_text(None)

    def test_process_image_file_not_found(self):
        layer = PerceptionLayer()
        with pytest.raises(FileNotFoundError):
            layer.process_image("/nonexistent/path/image.png")

    def test_process_vision_dict_no_image_path(self):
        layer = PerceptionLayer()
        with pytest.raises((FileNotFoundError, ValueError, KeyError)):
            layer.process({"type": "vision", "image_path": None})

    def test_process_structured_dict(self):
        layer = PerceptionLayer()
        data = {"facts": ["a"], "rules": [{"if": ["a"], "then": "b"}], "goal": "b"}
        event = layer.process(data)
        assert event.input_type == "structured"
        assert event.raw_input == data
        assert event.normalized_text == "b"

    def test_process_bytes_input(self):
        layer = PerceptionLayer()
        event = layer.process(b"hello world")
        assert isinstance(event, PerceptEvent)
        assert "hello" in event.normalized_text

    def test_process_string_input(self):
        layer = PerceptionLayer()
        event = layer.process("test input")
        assert event.input_type == "text"
        assert event.normalized_text == "test input"
