"""
Corpus reali e riproducibili per la validazione del Continual Learning.

I due task non sono sintetici: sono estratti da file che esistono gia' nel
repository, quindi il contenuto e' deterministico e il risultato riproducibile
senza dipendere da una generazione casuale.

- Task A: prosa tecnica in italiano -> `DOCUMENTAZIONE/*.md`
- Task B: codice sorgente Python -> `neural_core/*.py` e `cognitive_core/*.py`

I due domini sono separabili strutturalmente (prosa vs codice) ma condividono
lo stesso vocabolario di base (parole tecniche inglesi che compaiono in entrambi),
che e' esattamente il caso difficile: il task B deve essere appreso senza
distruggere la prosa del task A.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Documentazione: prosa italiana. Esclusi i file che contengono codice o
# tabelle massive, per non inquinare il task A con token di programmazione.
TASK_A_GLOBS = [
    "DOCUMENTAZIONE/ARCHITETTURA.md",
    "DOCUMENTAZIONE/SPECIFICA_PROGETTO.md",
    "DOCUMENTAZIONE/ARCHITETTURA_2.md",
    "DOCUMENTAZIONE/DIARIO.md",
]

# Codice sorgente Python: il task B.
TASK_B_GLOBS = [
    "neural_core/network.py",
    "neural_core/trainer.py",
    "neural_core/tokenizer.py",
    "neural_core/dataset.py",
    "neural_core/loss.py",
    "neural_core/optimizer.py",
    "neural_core/torch_model.py",
    "neural_core/backend.py",
    "neural_core/model_factory.py",
]


def _strip_markdown(text):
    """Rimuove la struttura markdown, lasciando la prosa."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)   # blocchi di codice
    text = re.sub(r"`[^`]*`", " ", text)                       # codice inline
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # intestazioni
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.M)   # liste
    text = re.sub(r"^\s{0,3}\|.*$", " ", text, flags=re.M)     # tabelle
    text = re.sub(r"^\s{0,3}>.*$", " ", text, flags=re.M)      # citazioni
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)       # link
    text = re.sub(r"[*_~]{1,3}", " ", text)                    # enfasi
    return text


def _strip_python(text):
    """Rimuove commenti e docstring, lasciando il codice."""
    text = re.sub(r'"""*.*?"""*', " ", text, flags=re.DOTALL)  # docstring
    out = []
    for line in text.splitlines():
        # rimuove i commenti ma non le '#' dentro le stringhe di codice
        in_str = False
        quote = ""
        cut = len(line)
        i = 0
        while i < len(line):
            ch = line[i]
            if not in_str and ch in "\"'":
                in_str, quote = True, ch
            elif in_str and ch == quote and (i == 0 or line[i - 1] != "\\"):
                in_str = False
            elif not in_str and ch == "#":
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def _load(globs, transform, min_words=200):
    """Legge i file indicati e restituisce testo e sorgenti usati."""
    parts, sources = [], []
    for rel in globs:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        cleaned = transform(path.read_text(encoding="utf-8", errors="ignore"))
        if len(cleaned.split()) < min_words:
            continue
        parts.append(cleaned)
        sources.append(rel)
    return "\n".join(parts), sources


def load_task_a_text():
    """Testo italiano (prosa tecnica) per il task A."""
    return _load(TASK_A_GLOBS, _strip_markdown)


def load_task_b_text():
    """Codice Python per il task B."""
    return _load(TASK_B_GLOBS, _strip_python)
