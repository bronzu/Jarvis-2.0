from pathlib import Path


def test_corpus_addestramento_esiste():
    percorso = Path(__file__).resolve().parents[2] / "corpus_addestramento.txt"
    assert percorso.exists(), "corpus_addestramento.txt non esiste."


def test_corpus_addestramento_contenuto_italiano():
    percorso = Path(__file__).resolve().parents[2] / "corpus_addestramento.txt"
    testo = percorso.read_text(encoding="utf-8")
    assert len(testo) > 200, "Il corpus deve essere sufficientemente ampio."
    parole_italiane = [
        "JARVIS",
        "sistema",
        "neurale",
        "token",
        "italiano",
        "rete",
        "apprendimento",
    ]
    for parola in parole_italiane:
        assert parola in testo, f"Parola italiana mancante: {parola}"


def test_corpus_addestramento_non_vuoto():
    percorso = Path(__file__).resolve().parents[2] / "corpus_addestramento.txt"
    testo = percorso.read_text(encoding="utf-8").strip()
    assert len(testo) > 0, "Il corpus non può essere vuoto."
