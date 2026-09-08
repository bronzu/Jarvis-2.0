import pytest

from interface_layer.voice_interface import VoiceInterface


class TestVoiceInterface:
    def test_speak_routes_prompt(self):
        vi = VoiceInterface()
        result = vi.speak("Chi sei?")
        assert "response" in result or "output" in result
        assert result.get("route") == "identity"

    def test_say_returns_false_without_backend(self):
        vi = VoiceInterface()
        assert vi.say("hello") is False

    def test_listen_falls_back_to_input(self, monkeypatch):
        vi = VoiceInterface()
        monkeypatch.setattr("builtins.input", lambda prompt="": "test voice")
        assert vi.listen() == "test voice"