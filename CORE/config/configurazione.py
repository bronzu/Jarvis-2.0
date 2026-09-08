"""
Configurazione centrale di JARVIS 2.0.

Questo modulo contiene le impostazioni fondamentali del progetto.
"""

from pathlib import Path


# Cartella principale del progetto
CARTELLA_PROGETTO = Path(__file__).resolve().parents[2]

# Cartelle principali
CARTELLA_CORE = CARTELLA_PROGETTO / "CORE"
CARTELLA_INFRASTRUTTURA = CARTELLA_PROGETTO / "INFRASTRUCTURE"
CARTELLA_TEST = CARTELLA_PROGETTO / "TESTS"
CARTELLA_DOCUMENTAZIONE = CARTELLA_PROGETTO / "DOCUMENTAZIONE"


# Informazioni progetto
NOME_PROGETTO = "JARVIS 2.0"
VERSIONE = "0.1.0"
MODALITA_SVILUPPO = True