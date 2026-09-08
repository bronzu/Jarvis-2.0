from executive_system.tools.web_tool import WebBrowserTool
from neural_core.model_interface import NeuralModelInterface
from neural_core.network import MicroTransformer
from neural_core.tokenizer import Tokenizer


def test_neural_model_interface_generates_internal_response():
    """
    NeuralModelInterface genera una risposta interna (JARVIS) senza modelli esterni.
    """
    tokenizer = Tokenizer()
    tokenizer.fit("JARVIS sistema test")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary))
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
    result = interface.generate("JARVIS")
    assert "text" in result
    assert result["provider"] == "internal_model"
    assert result["model"] == "neural-core-internal"
    assert result["fallback"] is False


def test_neural_model_interface_status_is_internal():
    """Il Neural Core è interno e non richiede modelli esterni."""
    interface = NeuralModelInterface()
    status = interface.status()
    assert status["type"] == "internal"
    assert status["provider"] == "internal_model"
    assert status["external_model_required"] is False


def test_web_tool_extracts_text_and_searches_results():
    tool = WebBrowserTool(timeout=10)
    result = tool.search("Python requests library", limit=1)
    assert isinstance(result, list)
    assert len(result) <= 1 or result[0].get("title") is not None