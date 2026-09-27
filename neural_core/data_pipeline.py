"""
Data Pipeline per la costruzione del dataset di addestramento di JARVIS 2.0.

Implementa tre stadi di pre-processing per dati raw:
    1. Deduplicazione semantica via MinHash/LSH
    2. Filtri euristici di qualità (entropia, ripetizione, punteggiatura)
    3. Formattazione Chain-of-Thought (CoT) sintetica

La deduplicazione riutilizza ``minhash_lsh.find_near_duplicates`` per
identificare e rimuovere documenti con similarità di Jaccard > soglia.
I filtri di qualità scartano documenti con bassa entropia o eccessiva
ripetizione di caratteri.  La formattazione CoT analizza i documenti per
estrarre blocchi logici o codice e li avvolge in un template di ragionamento
esplicito (<thinking>...</thinking> <answer>...</answer>).
"""

import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from neural_core.minhash_lsh import find_near_duplicates


# ----------------------------------------------------------------------
# 1. Deduplicazione semantica (MinHash/LSH)
# ----------------------------------------------------------------------

def semantic_deduplicate(
    documents: List[str],
    threshold: float = 0.85,
    num_perm: int = 64,
    ngram: int = 3,
    keep: str = "first",
) -> Tuple[List[str], Dict]:
    """Rimuove documenti semanticamente duplicati usando MinHash + LSH.

    Args:
        documents: lista di stringhe (documenti o paragrafi).
        threshold: soglia di Jaccard (default 0.85).  I documenti con
            similarità superiore alla soglia vengono considerati duplicati.
        num_perm: numero di permutazioni per MinHash (default 64).
        ngram: dimensione degli shingles (default 3).
        keep: quale elemento tenere in caso di duplicato
              ("first" = prima occorrenza, "last" = ultima).

    Returns:
        (documents_filtrati, report) — report contiene statistiche
        su candidate pairs, verified pairs e numero di documenti rimossi.
    """
    if not documents:
        return [], {
            "num_documents_before": 0,
            "num_candidate_pairs": 0,
            "num_verified_pairs": 0,
            "num_removed": 0,
            "num_documents_after": 0,
            "threshold": threshold,
            "num_perm": num_perm,
            "ngram": ngram,
            "method": "MinHash + LSH",
        }

    report = find_near_duplicates(
        documents, threshold=threshold, num_perm=num_perm, ngram=ngram
    )

    drop_indices: set = set()
    for i, j, _sim in report["pairs"]:
        if keep == "last":
            drop_indices.add(i)
        else:
            drop_indices.add(j)

    kept = [doc for idx, doc in enumerate(documents) if idx not in drop_indices]

    return kept, {
        "num_documents_before": len(documents),
        "num_candidate_pairs": report["num_candidate_pairs"],
        "num_verified_pairs": report["num_verified_pairs"],
        "num_removed": len(drop_indices),
        "num_documents_after": len(kept),
        "threshold": threshold,
        "num_perm": num_perm,
        "ngram": ngram,
        "method": report["method"],
    }


# ----------------------------------------------------------------------
# 2. Filtri euristici di qualità
# ----------------------------------------------------------------------

def char_repetition_ratio(text: str) -> float:
    """Rapporto di caratteri consecutivi ripetuti rispetto al totale.

    Un valore alto (> 0.3) indica testo con ripetizione eccessiva
    (es. "aaaaa!!!..." o spam).
    """
    if len(text) < 10:
        return 0.0
    runs = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            runs += 1
    return runs / len(text)


def shannon_entropy(text: str) -> float:
    """Entropia di Shannon dei caratteri (bit/simbolo).

    Un testo naturale ha entropia tipicamente > 3.0 bit.
    Valori bassi indicano testo ripetitivo o generato a mano male.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def punctuation_ratio(text: str) -> float:
    """Rapporto tra caratteri di punteggiatura e caratteri totali."""
    if not text:
        return 0.0
    punct_chars = ".,;:!?—()[]«»\"'()*..."
    punct = sum(1 for c in text if c in punct_chars)
    return punct / len(text)


def quality_filter(
    document: str,
    min_chars: int = 50,
    max_char_rep_ratio: float = 0.30,
    min_entropy: float = 2.0,
    min_punct_ratio: float = 0.02,
) -> bool:
    """Valuta se un documento supera i filtri di qualità.

    Args:
        document: testo del documento.
        min_chars: lunghezza minima in caratteri (default 50).
        max_char_rep_ratio: soglia massima di ripetizione consecutiva
            (default 0.30).
        min_entropy: entropia di Shannon minima (default 2.0 bit).
        min_punct_ratio: soglia minima di punteggiatura (default 0.02).

    Returns:
        True se il documento passa tutti i filtri, False altrimenti.
    """
    if len(document) < min_chars:
        return False
    if char_repetition_ratio(document) > max_char_rep_ratio:
        return False
    if shannon_entropy(document) < min_entropy:
        return False
    if punctuation_ratio(document) < min_punct_ratio:
        return False
    return True


def filter_documents(
    documents: List[str],
    min_chars: int = 50,
    max_char_rep_ratio: float = 0.30,
    min_entropy: float = 2.0,
    min_punct_ratio: float = 0.02,
) -> Tuple[List[str], Dict]:
    """Applica i filtri di qualità a una lista di documenti.

    Returns:
        (documents_ok, report) — report contiene il conteggio di documenti
        respinti per ciascun filtro.
    """
    kept: List[str] = []
    rejected = {
        "too_short": 0,
        "high_repetition": 0,
        "low_entropy": 0,
        "low_punctuation": 0,
    }

    for doc in documents:
        if len(doc) < min_chars:
            rejected["too_short"] += 1
            continue
        if char_repetition_ratio(doc) > max_char_rep_ratio:
            rejected["high_repetition"] += 1
            continue
        if shannon_entropy(doc) < min_entropy:
            rejected["low_entropy"] += 1
            continue
        if punctuation_ratio(doc) < min_punct_ratio:
            rejected["low_punctuation"] += 1
            continue
        kept.append(doc)

    return kept, {
        "num_documents_before": len(documents),
        "num_documents_after": len(kept),
        "rejected": rejected,
    }


# ----------------------------------------------------------------------
# 3. Formattazione Chain-of-Thought (CoT) sintetica
# ----------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
_NUMBERED_LIST_RE = re.compile(
    r"(?:^|\n)(\d+)[.)]\s+(.+?)(?=(?:\n\d+[.)] )|\Z)", re.MULTILINE | re.DOTALL
)
_LOGICAL_LABEL_RE = re.compile(
    r"^(Obiettivo|Analisi|Procedura|Verifica|Conclusione|Soluzione|Ipotesi|"
    r"Premessa|Inferenza|Ragionamento|Problema|Scenario|Dati|Azione|"
    r"Osservazione|Risultato|Enunciato|Domanda|Risposta)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER_LABEL_RE = re.compile(
    r"^(Conclusione|Soluzione|Risposta|Output)\s*:\s*(.+?)(?=\n\Z|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def _extract_logical_sections(text: str) -> Optional[str]:
    """Restituisce il testo se contiene sezioni logiche etichettate."""
    if _LOGICAL_LABEL_RE.search(text):
        return text
    return None


def _extract_code_block(text: str) -> Optional[str]:
    """Estrae il primo blocco di codice (```...```) se presente."""
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(0)
    return None


def _extract_numbered_steps(text: str) -> Optional[str]:
    """Estrae un elenco numerato (1. 2. 3.) se presente."""
    matches = _NUMBERED_LIST_RE.findall(text)
    if len(matches) >= 2:
        return "\n".join(f"{num}. {step.strip()}" for num, step in matches)
    return None


def _extract_final_answer(document: str) -> str:
    """Estrae la risposta finale da un documento, o usa l'ultimo paragrafo."""
    answer_match = _ANSWER_LABEL_RE.search(document)
    if answer_match:
        return answer_match.group(2).strip()
    paragraphs = [p.strip() for p in document.strip().split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1][:500]
    sentences = re.split(r"(?<=[.!?])\s+", document.strip())
    if sentences:
        return sentences[-1][:500]
    return document.strip()[:500]


def format_cot(document: str) -> str:
    """Avvolge un documento in un template Chain-of-Thought.

    Analizza il documento per individuare:
    - Blocchi di codice (```python ... ```)
    - Sezioni logiche etichettate (Obiettivo:, Analisi:, ...)
    - Elenchi numerati (1. 2. 3.)

    Se ne trova almeno uno, il documento viene riformattato come:

        <thinking>
        [passaggi logici dedotti]
        </thinking>
        <answer>
        [risposta finale]
        </answer>

    Se non trova strutture logiche, restituisce il documento invariato.
    """
    if not document or not document.strip():
        return document

    sections = _extract_logical_sections(document)
    code = _extract_code_block(document)
    steps = _extract_numbered_steps(document)

    if sections is None and code is None and steps is None:
        return document

    thinking_blocks: List[str] = []
    if sections:
        thinking_blocks.append(sections)
    elif code:
        thinking_blocks.append(code)
    if steps:
        thinking_blocks.append(steps)

    thinking = "\n\n".join(b.strip() for b in thinking_blocks if b.strip())
    answer = _extract_final_answer(document)

    return f"<thinking>\n{thinking}\n</thinking>\n<answer>\n{answer}\n</answer>"


# ----------------------------------------------------------------------
# Pipeline completa
# ----------------------------------------------------------------------

class DataPipeline:
    """Pipeline di pre-processing per dati raw di addestramento.

    Esegue sequenzialmente:
        1. Deduplicazione semantica (MinHash/LSH)
        2. Filtri euristici di qualità
        3. Formattazione Chain-of-Thought

    Attributes:
        dedup_threshold: soglia Jaccard per la deduplicazione.
        num_perm: numero di permutazioni MinHash.
        ngram: dimensione degli shingles.
        min_chars: lunghezza minima documento.
        max_char_rep_ratio: soglia max ripetizione caratteri.
        min_entropy: entropia Shannon minima.
        min_punct_ratio: soglia minima punteggiatura.
        cot_enabled: se True applica la formattazione CoT.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.85,
        num_perm: int = 64,
        ngram: int = 3,
        min_chars: int = 50,
        max_char_rep_ratio: float = 0.30,
        min_entropy: float = 2.0,
        min_punct_ratio: float = 0.02,
        cot_enabled: bool = True,
    ):
        self.dedup_threshold = dedup_threshold
        self.num_perm = num_perm
        self.ngram = ngram
        self.min_chars = min_chars
        self.max_char_rep_ratio = max_char_rep_ratio
        self.min_entropy = min_entropy
        self.min_punct_ratio = min_punct_ratio
        self.cot_enabled = cot_enabled

    def process(self, documents: List[str]) -> Tuple[List[str], Dict]:
        """Esegue la pipeline completa su una lista di documenti.

        Args:
            documents: lista di documenti raw (stringhe).

        Returns:
            (documents_processati, report_completo)
        """
        report: Dict = {}

        documents_dd, dedup_report = semantic_deduplicate(
            documents,
            threshold=self.dedup_threshold,
            num_perm=self.num_perm,
            ngram=self.ngram,
        )
        report["deduplication"] = dedup_report

        documents_qf, filter_report = filter_documents(
            documents_dd,
            min_chars=self.min_chars,
            max_char_rep_ratio=self.max_char_rep_ratio,
            min_entropy=self.min_entropy,
            min_punct_ratio=self.min_punct_ratio,
        )
        report["quality_filter"] = filter_report

        if self.cot_enabled:
            documents_cot = [format_cot(doc) for doc in documents_qf]
        else:
            documents_cot = documents_qf[:]

        report["cot_formatting"] = {
            "num_formatted": sum(1 for d in documents_cot if "<thinking>" in d),
            "num_passthrough": sum(1 for d in documents_cot if "<thinking>" not in d),
        }

        report["total"] = {
            "input_documents": len(documents),
            "after_dedup": len(documents_dd),
            "after_quality_filter": len(documents_qf),
            "after_cot": len(documents_cot),
        }

        return documents_cot, report


def process_raw_data(
    documents: List[str],
    dedup_threshold: float = 0.85,
    num_perm: int = 64,
    ngram: int = 3,
    min_chars: int = 50,
    max_char_rep_ratio: float = 0.30,
    min_entropy: float = 2.0,
    min_punct_ratio: float = 0.02,
    cot_enabled: bool = True,
) -> Tuple[List[str], Dict]:
    """Entry-point funzione per la pipeline di pre-processing.

    Crea una :class:`DataPipeline` con i parametri dati e processa i documenti.
    """
    pipeline = DataPipeline(
        dedup_threshold=dedup_threshold,
        num_perm=num_perm,
        ngram=ngram,
        min_chars=min_chars,
        max_char_rep_ratio=max_char_rep_ratio,
        min_entropy=min_entropy,
        min_punct_ratio=min_punct_ratio,
        cot_enabled=cot_enabled,
    )
    return pipeline.process(documents)
