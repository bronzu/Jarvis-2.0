"""
Test del sistema di configurazione di JARVIS 2.0.
"""

from CORE.config.configurazione import (
    NOME_PROGETTO,
    VERSIONE,
    CARTELLA_PROGETTO,
)


def test_nome_progetto():
    assert NOME_PROGETTO == "JARVIS 2.0"


def test_versione():
    assert VERSIONE == "0.1.0"


def test_cartella_progetto_esiste():
    assert CARTELLA_PROGETTO.exists()
    assert CARTELLA_PROGETTO.is_dir()