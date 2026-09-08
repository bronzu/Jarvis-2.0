from unittest.mock import patch, MagicMock

import pytest

from executive_system.tools.internet_tool import InternetTool


SIMPLE_HTML = """
<html>
  <head><title>Test</title><style>.x{}</style></head>
  <body>
    <h1>Hello World</h1>
    <p>This is a test.</p>
    <script>alert('x');</script>
  </body>
</html>
"""


def test_fetch_simulated_http_request():
    tool = InternetTool()
    fake_response = MagicMock()
    fake_response.read.return_value = SIMPLE_HTML.encode("utf-8")
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)
    with patch("executive_system.tools.internet_tool.urlopen", return_value=fake_response) as mock_urlopen:
        html = tool.fetch("https://example.com")
        assert html == SIMPLE_HTML
        mock_urlopen.assert_called_once()


def test_fetch_invalid_url_raises():
    tool = InternetTool()
    with pytest.raises(ValueError):
        tool.fetch("")
    with pytest.raises(ValueError):
        tool.fetch(None)


def test_extract_text_removes_scripts_styles_and_tags():
    tool = InternetTool()
    text = tool.extract_text(SIMPLE_HTML)
    assert "Hello World" in text
    assert "This is a test" in text
    assert "<script>" not in text
    assert "<style>" not in text
    assert "<html>" not in text
    assert "alert" not in text


def test_extract_text_collapses_whitespace():
    tool = InternetTool()
    html = "<p>  spaced   out\n\ntext  </p>"
    text = tool.extract_text(html)
    assert text == "spaced out text"


def test_search_fetches_and_extracts_text():
    tool = InternetTool()
    fake_response = MagicMock()
    fake_response.read.return_value = SIMPLE_HTML.encode("utf-8")
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)
    with patch("executive_system.tools.internet_tool.urlopen", return_value=fake_response):
        text = tool.search("https://example.com")
        assert "Hello World" in text
        assert "This is a test" in text
        assert "<script>" not in text


def test_search_invalid_query_raises():
    tool = InternetTool()
    with pytest.raises(ValueError):
        tool.search("")
    with pytest.raises(ValueError):
        tool.search(None)
