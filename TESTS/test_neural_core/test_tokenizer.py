import pickle

import pytest

from neural_core.tokenizer import Tokenizer


def test_tokenizer_encode_decode():
    tokenizer = Tokenizer()

    testo = "ciao JARVIS come stai"

    tokenizer.fit(testo)

    token_ids = tokenizer.encode(testo)
    testo_decodificato = tokenizer.decode(token_ids)

    assert len(token_ids) == 4
    assert testo_decodificato == testo


def test_tokenizer_unknown_word():
    tokenizer = Tokenizer()

    tokenizer.fit("ciao JARVIS")

    token_ids = tokenizer.encode("ciao mondo")

    assert token_ids[0] == tokenizer.vocabulary["ciao"]
    assert token_ids[1] == tokenizer.vocabulary["<UNK>"]


def test_tokenizer_vocabulary():
    tokenizer = Tokenizer()

    tokenizer.fit("ciao JARVIS")

    assert "<UNK>" in tokenizer.vocabulary
    assert "ciao" in tokenizer.vocabulary
    assert "JARVIS" in tokenizer.vocabulary

    assert tokenizer.vocabulary["<UNK>"] == 0


def test_tokenizer_from_corpus_file(tmp_path):
    corpus = "alpha beta gamma delta"
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(corpus, encoding="utf-8")

    tokenizer = Tokenizer.from_corpus_file(str(corpus_file))

    assert len(tokenizer.vocabulary) == 6
    assert tokenizer.encode("alpha beta") == [2, 3]


# ======================================================================
# COMPATIBILITÀ: la modalità di default resta quella storica
# ======================================================================


def test_modalita_default_e_whitespace():
    tokenizer = Tokenizer()
    assert tokenizer.mode == Tokenizer.MODE_WHITESPACE
    assert tokenizer.pre_tokenize("ciao, mondo!") == ["ciao,", "mondo!"]
    assert tokenizer.vocab_size == 2
    assert tokenizer.unk_id == 0
    assert tokenizer.end_of_text_id == 1


def test_modalita_non_supportata_rifiutata():
    with pytest.raises(ValueError):
        Tokenizer(mode="bpe-esterno")


def test_max_vocab_size_limita_il_vocabolario():
    tokenizer = Tokenizer(max_vocab_size=5)
    tokenizer.fit("a a a b b c d e f g")

    assert tokenizer.vocab_size <= 5
    assert "a" in tokenizer.vocabulary
    assert "b" in tokenizer.vocabulary


# ======================================================================
# MODALITÀ TECHNICAL: italiano moderno, tecnico, codice, matematica, scienza
# ======================================================================


def test_technical_separa_punteggiatura_italiana():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("Ciao, JARVIS! Come stai?")

    assert tokens == ["Ciao", ",", "JARVIS", "!", "Come", "stai", "?"]


def test_technical_gestisce_elisioni_e_accenti_italiani():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("Perché l'energia dell'atomo è così però")

    assert tokens == ["Perché", "l'", "energia", "dell'", "atomo", "è", "così", "però"]


def test_technical_tokenizza_codice():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("def calcola_media(valori): return somma // len(valori)")

    assert "calcola_media" in tokens
    assert "(" in tokens and ")" in tokens
    assert "//" in tokens
    assert ":" in tokens


def test_technical_riconosce_operatori_multi_carattere():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("if x != 0 and y >= 2 ** 3 -> z := x")

    for operator in ["!=", ">=", "**", "->", ":="]:
        assert operator in tokens


def test_technical_riconosce_numeri_matematici_e_scientifici():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("pi 3.14 massa 1,5 avogadro 6.022e23 esadecimale 0xFF")

    assert "3.14" in tokens
    assert "1,5" in tokens
    assert "6.022e23" in tokens
    assert "0xFF" in tokens


def test_technical_riconosce_simboli_scientifici():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("energia ∫ π ≤ √ ∑ Δ ° μ")

    for symbol in ["∫", "π", "≤", "√", "∑", "Δ", "°", "μ"]:
        assert symbol in tokens


def test_technical_riduce_il_vocabolario_rispetto_al_whitespace():
    """
    Separando la punteggiatura il tokenizer tecnico generalizza meglio: la
    stessa parola non genera più token distinti.
    """
    text = "rete, rete. rete! rete? rete; nodo, nodo. nodo! nodo? nodo"

    whitespace = Tokenizer()
    whitespace.fit(text)

    technical = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    technical.fit(text)

    assert whitespace.vocab_size > technical.vocab_size
    assert technical.vocabulary["rete"] == technical.encode("rete")[0]
    assert technical.encode("rete, rete.")[0] == technical.encode("rete")[0]


def test_technical_encode_decode_ricostruisce_italiano():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    testo = "Perché l'energia dell'atomo non è costante?"
    tokenizer.fit(testo)

    decodificato = tokenizer.decode(tokenizer.encode(testo))

    assert decodificato == testo


def test_technical_decode_non_produce_spazi_doppi():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    testo = "La velocità è 3.14 m/s, ok (sì)."
    tokenizer.fit(testo)

    decodificato = tokenizer.decode(tokenizer.encode(testo))

    assert "  " not in decodificato
    assert not decodificato.startswith(" ")
    assert not decodificato.endswith(" ")
    assert decodificato.endswith(".")


def test_technical_token_sconosciuto_diventa_unk():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokenizer.fit("sistema neurale")

    ids = tokenizer.encode("sistema quantistico")

    assert ids[0] == tokenizer.vocabulary["sistema"]
    assert ids[1] == tokenizer.unk_id


def test_lowercase_normalizza_il_testo():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL, lowercase=True)
    tokenizer.fit("Sistema NEURALE")

    assert "sistema" in tokenizer.vocabulary
    assert "Sistema" not in tokenizer.vocabulary
    assert tokenizer.encode("SISTEMA") == [tokenizer.vocabulary["sistema"]]


# ======================================================================
# TOKEN SPECIALI
# ======================================================================


def test_end_of_text_resta_atomico_in_modalita_technical():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokens = tokenizer.pre_tokenize("fine <|endoftext|> inizio")

    assert tokens == ["fine", Tokenizer.END_OF_TEXT_TOKEN, "inizio"]


def test_encode_con_end_of_text():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao mondo")

    ids = tokenizer.encode("ciao mondo", add_end_of_text=True)

    assert ids[-1] == tokenizer.end_of_text_id


def test_add_special_token_registrato_e_atomico():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    token_id = tokenizer.add_special_token("<|codice|>")

    assert tokenizer.vocabulary["<|codice|>"] == token_id
    assert tokenizer.pre_tokenize("prima <|codice|> dopo") == ["prima", "<|codice|>", "dopo"]
    assert tokenizer.decode([token_id], skip_special_tokens=True) == ""


def test_add_special_token_rifiuta_valore_vuoto():
    tokenizer = Tokenizer()
    with pytest.raises(ValueError):
        tokenizer.add_special_token("")


# ======================================================================
# SAVE / LOAD
# ======================================================================


def test_save_e_load_json_preservano_encode(tmp_path):
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL, max_vocab_size=200)
    testo = "Il sistema neurale calcola 3.14 e usa def somma(a, b): return a + b"
    tokenizer.fit(testo)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    assert path.exists()

    ricaricato = Tokenizer.load(str(path))

    assert ricaricato.mode == tokenizer.mode
    assert ricaricato.max_vocab_size == tokenizer.max_vocab_size
    assert ricaricato.vocabulary == tokenizer.vocabulary
    assert ricaricato.encode(testo) == tokenizer.encode(testo)
    assert ricaricato.decode(ricaricato.encode(testo)) == tokenizer.decode(tokenizer.encode(testo))


def test_load_file_inesistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        Tokenizer.load(str(tmp_path / "assente.json"))


def test_load_state_dict_rifiuta_stato_non_valido():
    tokenizer = Tokenizer()
    with pytest.raises(ValueError):
        tokenizer.load_state_dict({"mode": "whitespace"})
    with pytest.raises(ValueError):
        tokenizer.load_state_dict({"vocabulary": {}, "mode": "sconosciuta"})


def test_pickle_roundtrip_preserva_comportamento():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokenizer.fit("il modello calcola x ** 2")

    ricaricato = pickle.loads(pickle.dumps(tokenizer))

    assert ricaricato.mode == Tokenizer.MODE_TECHNICAL
    assert ricaricato.encode("x ** 2") == tokenizer.encode("x ** 2")


def test_pickle_legacy_senza_nuovi_attributi():
    """
    I tokenizer serializzati dalle versioni precedenti non hanno `mode`,
    `lowercase` o `special_tokens`: devono continuare a funzionare.
    """
    legacy_state = {
        "vocabulary": {"<UNK>": 0, "<|endoftext|>": 1, "ciao": 2, "jarvis": 3},
        "reverse_vocabulary": {0: "<UNK>", 1: "<|endoftext|>", 2: "ciao", 3: "jarvis"},
        "max_vocab_size": None,
    }

    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.__setstate__(legacy_state)

    assert tokenizer.mode == Tokenizer.MODE_WHITESPACE
    assert tokenizer.special_tokens == ["<UNK>", "<|endoftext|>"]
    assert tokenizer.encode("ciao jarvis") == [2, 3]
    assert tokenizer.decode([2, 3]) == "ciao jarvis"


def test_from_corpus_file_in_modalita_technical(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("energia, materia; π ≈ 3.14", encoding="utf-8")

    tokenizer = Tokenizer.from_corpus_file(
        str(corpus_file), mode=Tokenizer.MODE_TECHNICAL
    )

    assert tokenizer.mode == Tokenizer.MODE_TECHNICAL
    assert "energia" in tokenizer.vocabulary
    assert "," in tokenizer.vocabulary
    assert "3.14" in tokenizer.vocabulary
