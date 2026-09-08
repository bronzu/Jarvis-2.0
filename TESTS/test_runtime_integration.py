import os

from executive_system.capability_registry import CapabilityRegistry
from executive_system.tool_selector import ToolSelector
from interface_layer.voice_interface import VoiceInterface
from executive_system.router import ExecutiveRouter
from neural_core.providers.external_llm_provider import ExternalLLMProvider


def test_capability_registry_registers_web_search_tool():
    registry = CapabilityRegistry()
    registry.register_tool("web_search", lambda: "web")

    assert registry.state("web_search") == "available"
    assert registry.get_factory("web_search")() == "web"


def test_external_provider_reads_yaml_provider_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  default_provider: openai
  providers:
    openai:
      enabled: false
      base_url: https://api.openai.test/v1
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
      timeout_seconds: 12
    ollama:
      enabled: false
      base_url: http://localhost:11434
      model: llama3.2
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")

    provider = ExternalLLMProvider(str(config_path))

    assert provider.providers["openai"]["api_key"] == "secret-key"
    assert provider.providers["openai"]["base_url"] == "https://api.openai.test/v1"


def test_external_providers_are_optional():
    """I provider esterni sono opzionali e disabilitati per default."""
    provider = ExternalLLMProvider()
    assert provider.is_enabled("ollama") is False
    assert provider.is_enabled("openai") is False
    assert provider.default_provider == "internal"


def test_voice_interface_routes_to_executive_system_and_web_search(monkeypatch):
    """
    La VoiceInterface comunica SOLO con ExecutiveRouter, mai con un LLM esterno.
    """
    class DummyRouter:
        def generate(self, prompt):
            return {"provider": "internal", "text": "ok jarvis", "response": "ok jarvis",
                    "route": "local", "fallback": False}

    class DummyWebTool:
        def search(self, query, limit=3):
            return [{"title": "Result", "url": "https://example.com", "snippet": "demo"}]

    voice_router = DummyRouter()
    from interface_layer.voice_interface import VoiceInterface
    voice = VoiceInterface(executive_router=voice_router)
    voice.web_tool = DummyWebTool()

    llm_response = voice.handle_prompt("Riassumi il progetto")
    web_response = voice.handle_prompt("cerca su web python")

    assert llm_response["output"] == "ok jarvis"
    assert llm_response["provider"] == "internal"
    assert web_response["provider"] == "web_tool"
    assert "https://example.com" in web_response["output"]
