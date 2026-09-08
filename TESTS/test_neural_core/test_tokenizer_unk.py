"""
Test del miglioramento del tokenizer per la riduzione dei token <UNK>.

Coprono i requisiti di FASE 3:
- save / load del tokenizer (JSON e pickle) preserva il comportamento e le
  metriche di copertura;
- compatibilità con i tokenizer "legacy" (solo dizionario vocabolario);
- riduzione percentuale di <UNK> all'aumentare di `max_vocab_size`.
"""

import pickle

from neural_core.tokenizer import Tokenizer


# ----------------------------------------------------------------------
# RIDUZIONE PERCENTUALE DI <UNK>
# ----------------------------------------------------------------------

def test_aumento_max_vocab_size_riduce_l_unk_su_testo_reale():
    """
    A parità di corpus, un vocabolario più ampio deve abbassare la percentuale
    di token <UNK>. Verifica il requisito "riduzione percentuale <UNK>".
    """
    corpus = (
        "Il sistema neurale interno di jarvis apprende dal corpus e genera "
        "risposte coerenti. L'architettura proprietaria combina attenzione e "
        "ragionamento per elaborare informazioni complesse in modo autonomo."
    )

    small = Tokenizer(max_vocab_size=10)
    small.fit(corpus)

    large = Tokenizer(max_vocab_size=200)
    large.fit(corpus)

    assert large.unknown_token_rate(corpus) <= small.unknown_token_rate(corpus)
    assert large.token_coverage(corpus) >= small.token_coverage(corpus)


def test_unk_rate_zero_quando_tutto_il_testo_e_nel_vocab():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao jarvis assistente persona")

    assert tokenizer.unknown_token_rate("ciao jarvis assistente persona") == 0.0
    assert tokenizer.count_unknown("ciao jarvis assistente persona") == 0


def test_unk_rate_conta_solo_i_token_sconosciuti():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao jarvis")

    # "mondo" non è nel vocabolario -> un solo token <UNK>.
    assert tokenizer.count_unknown("ciao mondo") == 1
    assert tokenizer.unknown_token_rate("ciao mondo") == 0.5


def test_unk_rate_zero_su_testo_vuoto():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao")
    assert tokenizer.unknown_token_rate("") == 0.0
    assert tokenizer.token_coverage("") == 0.0


# ----------------------------------------------------------------------
# SAVE / LOAD TOKENIZER
# ----------------------------------------------------------------------

def test_save_load_json_preserva_metriche_copertura(tmp_path):
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL, max_vocab_size=500)
    testo = (
        "Perché l'energia dell'atomo non è costante? Il sistema calcola "
        "3.14 e usa def somma(a, b): return a + b"
    )
    tokenizer.fit(testo)

    unk_originale = tokenizer.unknown_token_rate(testo)
    coverage_originale = tokenizer.token_coverage(testo)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    assert path.exists()

    ricaricato = Tokenizer.load(str(path))

    assert ricaricato.unknown_token_rate(testo) == unk_originale
    assert ricaricato.token_coverage(testo) == coverage_originale
    assert ricaricato.encode(testo) == tokenizer.encode(testo)


def test_save_load_pickle_preserva_metriche_copertura(tmp_path):
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL, max_vocab_size=400)
    testo = "Il modello calcola x ** 2 e gestisce codice def f(): return 0"
    tokenizer.fit(testo)

    unk_originale = tokenizer.unknown_token_rate(testo)

    path = tmp_path / "tokenizer.pkl"
    with open(path, "wb") as handle:
        pickle.dump(tokenizer, handle)

    with open(path, "rb") as handle:
        ricaricato = pickle.load(handle)

    assert ricaricato.unknown_token_rate(testo) == unk_originale
    assert ricaricato.encode(testo) == tokenizer.encode(testo)


# ----------------------------------------------------------------------
# COMPATIBILITÀ CON I TOKENIZER LEGACY
# ----------------------------------------------------------------------

def test_compat_tokenizer_legacy_senza_attributi_nuovi():
    """
    Un tokenizer serializzato dalle versioni precedenti (solo dizionario
    `vocabulary` / `reverse_vocabulary`) deve continuare a calcolare le
    metriche di copertura senza errori.
    """
    legacy_state = {
        "vocabulary": {"<UNK>": 0, "<|endoftext|>": 1, "ciao": 2, "jarvis": 3},
        "reverse_vocabulary": {0: "<UNK>", 1: "<|endoftext|>", 2: "ciao", 3: "jarvis"},
        "max_vocab_size": None,
    }

    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.__setstate__(legacy_state)

    assert tokenizer.unknown_token_rate("ciao jarvis") == 0.0
    assert tokenizer.count_unknown("ciao mondo") == 1
    assert tokenizer.token_coverage("ciao jarvis mondo") == 2 / 3


def test_compat_unk_rate_invariato_dopo_load_state_dict():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao jarvis sistema")

    state = tokenizer.state_dict()
    altro = Tokenizer()
    altro.load_state_dict(state)

    testo = "ciao jarvis mondo"
    assert altro.unknown_token_rate(testo) == tokenizer.unknown_token_rate(testo)
    assert altro.token_coverage(testo) == tokenizer.token_coverage(testo)
