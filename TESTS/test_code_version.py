"""
Tests for checkpoint_lineage.code_version parsing.
"""

import re
import subprocess
from pathlib import Path


def test_code_version_parses_from_versione_md():
    from neural_core.checkpoint_lineage import _code_version
    v = _code_version()
    assert v is not None
    assert re.match(r"^\d+\.\d+\.\d+", v), f"unexpected format: {v}"


def test_lineage_includes_code_version():
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer
    from neural_core.checkpoint_lineage import build_lineage
    text = "alpha beta gamma"
    tok = Tokenizer(max_vocab_size=10, mode="whitespace"); tok.fit(text)
    model = MicroTransformer(vocab_size=len(tok.vocabulary), d_model=8, num_heads=1, d_ff=16, num_layers=1, max_len=16, seed=7)
    lineage = build_lineage(model=model, model_class="MicroTransformer", model_config={}, tokenizer=tok, seed=7)
    assert lineage["code_version"] is not None
    assert re.match(r"^\d+\.\d+\.\d+", lineage["code_version"])


def test_versione_md_format_supported():
    """Verify VERSIONE.md is the canonical source."""
    from neural_core.checkpoint_lineage import _code_version
    candidates = [
        Path("DOCUMENTAZIONE/versionamento/VERSIONE.md"),
        Path("VERSIONE.md"),
    ]
    found = [c for c in candidates if c.exists()]
    assert found, "no VERSIONE.md candidate found"
    assert _code_version() is not None
