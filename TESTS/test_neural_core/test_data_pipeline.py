"""
Test unitari per la data pipeline di pre-processing (Punto C).

Verifica:
  - Deduplicazione semantica via MinHash/LSH (rimozione, keep strategy)
  - Filtri euristici di qualità (char repetition, entropia, punteggiatura)
  - Formattazione Chain-of-Thought (code blocks, logical sections, numbered steps)
  - Pipeline completa end-to-end (DataPipeline.process)
"""

import numpy as np
import pytest

from neural_core.data_pipeline import (
    DataPipeline,
    semantic_deduplicate,
    quality_filter,
    filter_documents,
    format_cot,
    char_repetition_ratio,
    shannon_entropy,
    punctuation_ratio,
    process_raw_data,
)


# ----------------------------------------------------------------------
# 1. Semantic deduplication (MinHash/LSH)
# ----------------------------------------------------------------------

def test_semantic_deduplicate_identifica_duplicati():
    docs = [
        "Il gatto è seduto sul tappeto.",
        "Il gatto è seduto sul tappeto.",
        "Il cane gioca con il pallone.",
    ]
    kept, report = semantic_deduplicate(docs, threshold=0.85)
    assert len(kept) == 2
    assert report["num_removed"] == 1
    assert report["num_documents_after"] == 2


def test_semantic_deduplicate_parafrasi_vicine():
    docs = [
        "Il gatto è seduto sul tappeto. Il cane corre nel parco.",
        "Il gatto è seduto sul tappeto. Il cane corre nel parco.",
        "Il cane corre nel parco. Il gatto è seduto sul tappeto.",
    ]
    kept, report = semantic_deduplicate(docs, threshold=0.85)
    assert report["num_documents_after"] <= 2


def test_semantic_deduplicate_keep_first():
    docs = [
        "Prima versione del documento.",
        "Prima versione del documento.",
    ]
    kept, report = semantic_deduplicate(docs, threshold=0.85, keep="first")
    assert len(kept) == 1
    assert kept[0] == docs[0]


def test_semantic_deduplicate_keep_last():
    docs = [
        "Prima versione del documento. Questo è un test.",
        "Prima versione del documento. Questo è un test.",
    ]
    kept, report = semantic_deduplicate(docs, threshold=0.85, keep="last")
    assert len(kept) == 1


def test_semantic_deduplicate_lista_vuota():
    kept, report = semantic_deduplicate([])
    assert kept == []
    assert report["num_documents_after"] == 0


def test_semantic_deduplicate_documenti_unici():
    docs = [
        "Il gatto è seduto sul tappeto.",
        "La sciarpa è rossa come il fuoco.",
        "I libri sono sulla mensola.",
    ]
    kept, report = semantic_deduplicate(docs, threshold=0.85)
    assert len(kept) == 3
    assert report["num_removed"] == 0


# ----------------------------------------------------------------------
# 2. Filtri euristici di qualità
# ----------------------------------------------------------------------

def test_char_repetition_ratio_testo_normale():
    text = "Questo è un testo normale con parole diverse."
    assert char_repetition_ratio(text) < 0.1


def test_char_repetition_ratio_testo_repetitivo():
    text = "aaaaaaaaaa!!!!!!!!!!!"
    assert char_repetition_ratio(text) > 0.5


def test_char_repetition_ratio_testo_corto():
    assert char_repetition_ratio("ab") == 0.0


def test_shannon_entropy_testo_normale():
    text = "Questo è un testo normale con parole diverse."
    assert shannon_entropy(text) > 3.0


def test_shannon_entropy_testo_repetitivo():
    text = "aaaaa"
    assert shannon_entropy(text) < 1.0


def test_shannon_entropy_testo_vuoto():
    assert shannon_entropy("") == 0.0


def test_punctuation_ratio_testo_normale():
    text = "Ciao, come stai? Bene, grazie!"
    assert punctuation_ratio(text) > 0.05


def test_punctuation_ratio_testo_senza_punteggiatura():
    text = "ciao come stai bene grazie"
    assert punctuation_ratio(text) < 0.02


def test_quality_filter_passa_documento_buono():
    doc = "Il gatto è seduto sul tappeto. Il cane corre nel parco. La casa è grande."
    assert quality_filter(doc) is True


def test_quality_filter_rifiuta_documento_corto():
    doc = "corto"
    assert quality_filter(doc, min_chars=50) is False


def test_quality_filter_rifiuta_testo_repetitivo():
    doc = "a" * 200 + "!!!" * 50
    assert quality_filter(doc, min_chars=50, max_char_rep_ratio=0.30) is False


def test_quality_filter_rifiuta_testo_bassa_entropia():
    doc = "a " * 100
    assert quality_filter(doc, min_chars=50, max_char_rep_ratio=0.99, min_entropy=2.0) is False


def test_quality_filter_rifiuta_testo_senza_punteggiatura():
    doc = "questo è un testo senza alcun segno di punteggiatura " * 5
    assert quality_filter(doc, min_chars=50, max_char_rep_ratio=0.99, min_entropy=3.0, min_punct_ratio=0.02) is False


def test_filter_documents_mantiene_validi():
    docs = [
        "Il gatto è seduto sul tappeto. Il cane corre nel parco.",
        "Il cane corre nel parco. Il gatto è seduto sul tappeto.",
        "corto",
    ]
    kept, report = filter_documents(docs, threshold=0.85) if False else filter_documents(docs)
    assert len(kept) == 2
    assert report["rejected"]["too_short"] == 1


def test_filter_documents_preserva_ordine():
    docs = [
        "Primo documento valido con punteggiatura. Molto interessante.",
        "Secondo documento valido con punteggiatura. Anche questo.",
    ]
    kept, _ = filter_documents(docs)
    assert len(kept) == 2


# ----------------------------------------------------------------------
# 3. Formattazione Chain-of-Thought (CoT)
# ----------------------------------------------------------------------

def test_format_cot_identifica_sezioni_logiche():
    doc = (
        "Obiettivo: Calcolare l'area del triangolo.\n"
        "Procedura: Applico la formula base * altezza / 2.\n"
        "Conclusione: L'area è 30 cm²."
    )
    result = format_cot(doc)
    assert "<thinking>" in result
    assert "</thinking>" in result
    assert "<answer>" in result
    assert "</answer>" in result


def test_format_cot_identifica_campo_codice():
    doc = (
        "Ecco un esempio in Python:\n"
        "```python\n"
        "def area(base, altezza):\n"
        "    return base * altezza / 2\n"
        "```\n"
        "Risultato: 30"
    )
    result = format_cot(doc)
    assert "<thinking>" in result
    assert "<answer>" in result
    assert "def area" in result


def test_format_cot_identifica_elenco_numerato():
    doc = (
        "Procedimento:\n"
        "1. Misura la base.\n"
        "2. Misura l'altezza.\n"
        "3. Applica la formula.\n"
        "Risultato: area = 30"
    )
    result = format_cot(doc)
    assert "<thinking>" in result
    assert "<answer>" in result


def test_format_cot_documento_senza_struttura_rimane_come_si():
    doc = "Questo è un semplice documento senza struttura logica"
    result = format_cot(doc)
    assert result == doc


def test_format_cot_documento_vuoto():
    assert format_cot("") == ""
    assert format_cot("   ") == "   "


def test_format_cot_estrae_risposta_da_conclusione():
    doc = (
        "Obiettivo: Test\n"
        "Procedura: Qualcosa\n"
        "Conclusione: La risposta finale è 42."
    )
    result = format_cot(doc)
    assert "42" in result


def test_format_cot_estrae_codice_con_lang_specifier():
    doc = "```javascript\nconsole.log('hello');\n```\nOutput: hello"
    result = format_cot(doc)
    assert "<thinking>" in result
    assert "console.log" in result


def test_format_cot_preserve_logical_sections_and_code():
    doc = (
        "Obiettivo: Spiega il codice.\n"
        "```python\nprint('ciao')\n```\n"
        "Conclusione: Il codice stampa ciao."
    )
    result = format_cot(doc)
    assert "<thinking>" in result
    assert "Obiettivo" in result
    assert "print('ciao')" in result
    assert "ciao" in result


# ----------------------------------------------------------------------
# 4. Pipeline completa
# ----------------------------------------------------------------------

def test_data_pipeline_process_completo():
    docs = [
        "Obiettivo: Calcola l'area. Procedura: base * altezza / 2. Conclusione: 30.",
        "Obiettivo: Calcola l'area. Procedura: base * altezza / 2. Conclusione: 30.",
        "corto",
        "aaaaaaaaaaaaaaaaaaaaa",
    ]
    pipeline = DataPipeline()
    result, report = pipeline.process(docs)
    assert len(result) == 1
    assert report["total"]["input_documents"] == 4
    assert report["total"]["after_dedup"] == 3
    assert report["total"]["after_quality_filter"] == 1
    assert "<thinking>" in result[0]


def test_data_pipeline_cot_disabled():
    docs = [
        "Obiettivo: Calcola. Conclusione: Risultato = 42.",
        "Altro documento valido con punteggiatura. Interessante.",
    ]
    pipeline = DataPipeline(cot_enabled=False)
    result, report = pipeline.process(docs)
    for doc in result:
        assert "<thinking>" not in doc


def test_data_pipeline_process_empty():
    pipeline = DataPipeline()
    result, report = pipeline.process([])
    assert result == []
    assert report["total"]["input_documents"] == 0


def test_data_pipeline_parametri_personalizzati():
    docs = [
        "Il gatto è seduto sul tappeto. Il cane corre nel parco.",
        "Il gatto è seduto sul tappeto. Il cane corre nel parco.",
    ]
    pipeline = DataPipeline(dedup_threshold=0.85, min_chars=20, min_entropy=2.5)
    result, report = pipeline.process(docs)
    assert report["deduplication"]["threshold"] == 0.85


def test_process_raw_data_entry_point():
    docs = [
        "Problema: 2+2=? Analisi: si tratta di un'addizione. Risposta: 4.",
        "Il cane corre nel parco. Il gatto dorme. La casa è grande.",
    ]
    result, report = process_raw_data(docs)
    assert len(result) == 2
    assert report["total"]["input_documents"] == 2


def test_data_pipeline_report_contiene_sezioni():
    docs = [
        "Obiettivo: Test. Procedura: Qualcosa. Conclusione: Risultato.",
        "Il cane corre nel parco. Il gatto dorme. La casa è grande.",
    ]
    pipeline = DataPipeline()
    result, report = pipeline.process(docs)
    assert "deduplication" in report
    assert "quality_filter" in report
    assert "cot_formatting" in report
    assert "total" in report
    assert report["cot_formatting"]["num_formatted"] == 1
    assert report["cot_formatting"]["num_passthrough"] == 1


def test_data_pipeline_dedup_mantiene_prima_occorenza():
    doc = "Primo documento unico con struttura logica.\nObiettivo: test.\nConclusione: Risultato."
    docs = [
        doc,
        doc,
    ]
    pipeline = DataPipeline()
    result, report = pipeline.process(docs)
    assert len(result) == 1
    assert result[0].startswith("<thinking>")


def test_data_pipeline_documento_con_codice_diventa_cot():
    doc = "```python\ndef solve():\n    return 42\n```\nOutput: 42"
    result, _ = DataPipeline().process([doc])
    assert "<thinking>" in result[0]
    assert "def solve" in result[0]


def test_char_repetition_ratio_zero_on_uniform_text():
    text = "abcdefghij"
    assert char_repetition_ratio(text) == 0.1
