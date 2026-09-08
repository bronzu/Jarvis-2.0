"""
Gestione degli esperimenti di JARVIS 2.0.

Fornisce una struttura di base per creare e organizzare
gli esperimenti del progetto.
"""

from pathlib import Path

from CORE.config.configurazione import CARTELLA_PROGETTO


CARTELLA_ESPERIMENTI = CARTELLA_PROGETTO / "INFRASTRUCTURE" / "esperimenti"


def inizializza_esperimenti() -> Path:
    """
    Crea la cartella degli esperimenti se non esiste
    e restituisce il relativo percorso.
    """

    CARTELLA_ESPERIMENTI.mkdir(parents=True, exist_ok=True)

    return CARTELLA_ESPERIMENTI