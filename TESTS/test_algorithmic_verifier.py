"""
Test del Verificatore di Ragionamento Algoritmico (cognitive_layer/algorithmic_verifier.py).

Il dataset dei 30 casi e' fisso e incluso in questo file: non viene generato a
runtime. I test coprono i tre insiemi richiesti dalla missione:

    - 10 problemi aritmetici con soluzione nota;
    - 10 snippet di codice (5 validi, 5 con errori di sintassi deliberati);
    - 10 prompt di ragionamento aperto/non deterministico.

AVVERTENZA (regola 4 del progetto): questi test sono un TEST DI FUMO del
verificatore, non una prova generale. Verificano che la logica di punteggio e
scelta scelga il ramo corretto su un dataset fisso di 30 casi, non che il
modello sottostante sappia risolvere aritmetica o scrivere codice.
"""

import numpy as np
import pytest

from cognitive_layer.algorithmic_verifier import (
    DEFAULT_BASE_SEED,
    DISCARD_THRESHOLD,
    LOG_DIR,
    MAX_TOKENS_PER_BRANCH,
    BRANCH_TEMPERATURES,
    Branch,
    VerifierError,
    classify_domain,
    evaluate_branch,
    generate_branches,
    verify,
)


# ---------------------------------------------------------------------------
# Modeli di supporto
# ---------------------------------------------------------------------------


class ScriptedModel:
    """Modello finto che restituisce testi prefissati, uno per chiamata.

    Espone la stessa interfaccia di generazione usata dal verificatore
    (`generate(prompt, **kwargs)`), come `NeuralModelInterface`.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        index = min(len(self.calls) - 1, len(self.scripts) - 1)
        text = self.scripts[index]
        return {"text": text, "generated_tokens": len(text.split())}


def make_branch(branch_id, testo_completo, temperature=0.2, seed=1):
    """Costruisce un Branch con la CoT spezzata per riga."""
    return Branch(
        id=branch_id,
        temperature=temperature,
        seed=seed,
        testo_completo=testo_completo,
        step_intermedi=[line.strip() for line in testo_completo.splitlines() if line.strip()],
        token_count=len(testo_completo.split()),
    )


# ---------------------------------------------------------------------------
# INSIEME A — 10 problemi aritmetici con soluzione nota
# ---------------------------------------------------------------------------

# (input, espressione, risultato atteso, testo ramo corretto, testo ramo errato)
ARITHMETIC_CASES = [
    (
        "Quanto fa 2 + 3?",
        "2 + 3",
        "5",
        "2 + 3 = 5, quindi la risposta è 5",
        "2 + 3 = 5, quindi la risposta è 8",
    ),
    (
        "calcola 12 * 4",
        "12 * 4",
        "48",
        "12 * 4 = 48, quindi il risultato è 48",
        "12 * 4 = 44, quindi il risultato è 44",
    ),
    (
        "Risolvi 7 - 2 =",
        "7 - 2",
        "5",
        "7 - 2 = 5, quindi la risposta è 5",
        "7 - 2 = 9, quindi la risposta è 9",
    ),
    (
        "somma 15 + 27",
        "15 + 27",
        "42",
        "15 + 27 = 42, quindi la risposta è 42",
        "15 + 27 = 32, quindi la risposta è 32",
    ),
    (
        "prodotto 6 * 7",
        "6 * 7",
        "42",
        "6 * 7 = 42, quindi il risultato è 42",
        "6 * 7 = 36, quindi il risultato è 36",
    ),
    (
        "differenza 100 - 58",
        "100 - 58",
        "42",
        "100 - 58 = 42, quindi la risposta è 42",
        "100 - 58 = 48, quindi la risposta è 48",
    ),
    (
        "quoziente 84 / 2",
        "84 / 2",
        "42",
        "84 / 2 = 42, quindi la risposta è 42",
        "84 / 2 = 21, quindi la risposta è 21",
    ),
    (
        "esponente 3 ** 3",
        "3 ** 3",
        "27",
        "3 ** 3 = 27, quindi la risposta è 27",
        "3 ** 3 = 9, quindi la risposta è 9",
    ),
    (
        "2 + 3 * 4",
        "2 + 3 * 4",
        "14",
        "2 + 3 * 4 = 14, quindi la risposta è 14",
        "2 + 3 * 4 = 20, quindi la risposta è 20",
    ),
    (
        "(4 + 6) / 2",
        "(4 + 6) / 2",
        "5",
        "(4 + 6) / 2 = 5, quindi la risposta è 5",
        "(4 + 6) / 2 = 10, quindi la risposta è 10",
    ),
]


@pytest.mark.parametrize(
    "input_text,espressione,atteso,ramo_corretto,ramo_errato",
    ARITHMETIC_CASES,
    ids=[f"arith_{index + 1:02d}" for index in range(len(ARITHMETIC_CASES))],
)
def test_arithmetic_sceglie_il_ramo_corretto(
    input_text, espressione, atteso, ramo_corretto, ramo_errato, tmp_path
):
    """Il verificatore deve dichiarare reliable e scegliere la risposta corretta."""
    model = ScriptedModel([ramo_corretto, ramo_corretto, ramo_errato])
    result = verify(input_text, model, log_path=tmp_path / "verify.jsonl")

    assert classify_domain(input_text) == "arithmetic"
    assert result.domain == "arithmetic"
    assert result.reliable is True
    assert result.verification_failed is False
    assert result.chosen_branch_text is not None
    assert atteso in result.chosen_branch_text
    assert result.chosen_branch_score >= DISCARD_THRESHOLD


# ---------------------------------------------------------------------------
# INSIEME B — 10 snippet di codice
# ---------------------------------------------------------------------------

VALID_SNIPPETS = [
    "def somma(a, b):\n    return a + b",
    "class Punto:\n    def __init__(self, x):\n        self.x = x",
    "import math\n\ndef area(r):\n    return math.pi * r ** 2",
    "for i in range(10):\n    print(i)",
    "x = [1, 2, 3]\nprint(sum(x))",
]

INVALID_SNIPPETS = [
    "def somma(a, b)\n    return a + b",
    "class Punto\n    pass",
    "def f(:\n    return 1",
    "x = = 5",
    "if True\n    print(1)",
]


def _fenced(code):
    return f"```python\n{code}\n```"


@pytest.mark.parametrize(
    "snippet",
    VALID_SNIPPETS,
    ids=[f"code_valid_{index + 1:02d}" for index in range(len(VALID_SNIPPETS))],
)
def test_codice_valido_supera_il_controllo_sintattico(snippet):
    """Uno snippet parsabile deve ottenere score_sintassi = 1."""
    branch = make_branch("branch_0", _fenced(snippet))
    evaluation = evaluate_branch(branch, "code")
    assert evaluation.score_sintassi == 1.0
    assert evaluation.scarto is False


@pytest.mark.parametrize(
    "snippet",
    INVALID_SNIPPETS,
    ids=[f"code_invalid_{index + 1:02d}" for index in range(len(INVALID_SNIPPETS))],
)
def test_codice_non_valido_fallisce_il_controllo_sintattico(snippet):
    """Uno snippet non parsabile deve ottenere score_sintassi = 0 ed essere scartato."""
    branch = make_branch("branch_0", _fenced(snippet))
    evaluation = evaluate_branch(branch, "code")
    assert evaluation.score_sintassi == 0.0
    assert evaluation.scarto is True
    assert evaluation.score_ramo == 0.0


# ---------------------------------------------------------------------------
# INSIEME C — 10 prompt di ragionamento aperto / non deterministico
# ---------------------------------------------------------------------------

OPEN_REASONING_PROMPTS = [
    "Secondo te la musica rock e la migliore in assoluto? Spiega perche.",
    "E giusto chiudere una fabbrica che da lavoro a cento persone?",
    "Se un amico ti chiede un parere che sai che e sbagliato, cosa gli dici?",
    "Qual e la cosa piu bella che puoi immaginare?",
    "Perche la gente ha paura del buio da piccoli?",
    "Se dovessi cambiare professione domani, quale faresti e perche?",
    "E giusto che i genitori scelgano il futuro dei figli?",
    "Descrivi la sensazione di svegliarsi presto d estate.",
    "Chi e piu importante in una squadra, il capitano o il piu bravo?",
    "Ha senso lamentarsi della vita quando si sta male davvero?",
]

#: Testo di ramo pulito e coerente: nessun loop, nessuna contraddizione.
#: Sul dominio "other" resta comunque non affidabile, per costruzione.
COHERENT_BRANCH_TEXT = (
    "Il punto di vista dipende dall'esperienza personale di ciascuno.\n"
    "Esistono argomentazioni a favore e contro, entrambe legittime.\n"
    "Non esiste una risposta univoca a questa domanda."
)


@pytest.mark.parametrize(
    "prompt",
    OPEN_REASONING_PROMPTS,
    ids=[f"open_{index + 1:02d}" for index in range(len(OPEN_REASONING_PROMPTS))],
)
def test_ragionamento_aperto_e_othersempre_non_affidabile(prompt, tmp_path):
    """Il dominio e' "other" e l'output e' sempre reliable=False."""
    model = ScriptedModel([COHERENT_BRANCH_TEXT] * 3)
    result = verify(prompt, model, log_path=tmp_path / "verify.jsonl")

    assert classify_domain(prompt) == "other"
    assert result.domain == "other"
    assert result.reliable is False
    # Il punteggio del ramo resta alto: nonostante cio' il risultato non e'
    # dichiarato affidabile, perche' il dominio non e' verificabile formalmente.
    assert result.chosen_branch_score == 1.0
    assert result.chosen_branch_text == COHERENT_BRANCH_TEXT


# ---------------------------------------------------------------------------
# Contratti del modulo
# ---------------------------------------------------------------------------


def test_generate_branches_usa_temperature_fisse_e_seed_distinti():
    """Le temperature sono quelle fissate dalla missione, i seed sono distinti."""
    model = ScriptedModel(["a\nb"] * 3)
    branches = generate_branches("ciao", model)

    assert [branch.temperature for branch in branches] == list(BRANCH_TEMPERATURES)
    assert [branch.temperature for branch in branches] == [0.2, 0.6, 1.0]
    assert len({branch.seed for branch in branches}) == 3
    assert [branch.id for branch in branches] == ["branch_0", "branch_1", "branch_2"]
    assert all(branch.step_intermedi == ["a", "b"] for branch in branches)
    for call in model.calls:
        assert call["max_new_tokens"] == MAX_TOKENS_PER_BRANCH
        assert call["temperature"] in BRANCH_TEMPERATURES


def test_generate_branches_rifiuta_un_modello_senza_generate():
    with pytest.raises(VerifierError):
        generate_branches("ciao", object())


# ---------------------------------------------------------------------------
# Stato RNG globale: generate_branches non deve lasciare effetti collaterali
# ---------------------------------------------------------------------------


def _stati_rng_identici(primi, secondi):
    """Confronto fra due stati RNG di NumPy (tupla, stringa, ndarray, intero)."""
    if isinstance(primi, tuple) and isinstance(secondi, tuple):
        if len(primi) != len(secondi):
            return False
        return all(_stati_rng_identici(a, b) for a, b in zip(primi, secondi))
    if isinstance(primi, np.ndarray) or isinstance(secondi, np.ndarray):
        primo = np.asarray(primi)
        secondo = np.asarray(secondi)
        return primo.shape == secondo.shape and bool(np.array_equal(primo, secondo))
    return primi == secondi


def test_generate_branches_non_altera_lo_stato_rng_globale():
    """Lo stato RNG globale deve essere identico prima e dopo la chiamata."""
    model = ScriptedModel(["a\nb"] * 3)

    np.random.seed(4242)
    stato_prima = np.random.get_state()
    generate_branches("ciao", model)
    stato_dopo = np.random.get_state()

    assert _stati_rng_identici(stato_dopo, stato_prima)


def test_generate_branches_non_altera_lo_stato_rng_anche_su_modello_reale():
    """Stessa garanzia sul generatore vero del progetto, non solo sul finto."""
    from neural_core.model_interface import NeuralModelInterface
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro cinque sei sette otto nove dieci")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), seed=7)
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    np.random.seed(4242)
    stato_prima = np.random.get_state()
    generate_branches("uno due", interface)
    stato_dopo = np.random.get_state()

    assert _stati_rng_identici(stato_dopo, stato_prima)


def test_stato_rng_ripristinato_anche_se_il_generatore_solleva():
    """Il ripristino avviene anche quando la generazione fallisce."""
    class ModelloCheFallisce:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("guasto generico")

    np.random.seed(4242)
    stato_prima = np.random.get_state()
    with pytest.raises(RuntimeError):
        generate_branches("ciao", ModelloCheFallisce())
    stato_dopo = np.random.get_state()

    assert _stati_rng_identici(stato_dopo, stato_prima)


def test_stato_rng_ripristinato_anche_durante_le_rigenerazioni(tmp_path):
    """Il fix vale anche per i round di backtracking dello STEP 4."""
    sbagliato = "2 + 3 = 5, quindi la risposta è 8"
    model = ScriptedModel([sbagliato] * 10)

    np.random.seed(4242)
    stato_prima = np.random.get_state()
    result = verify("Quanto fa 2 + 3?", model, log_path=tmp_path / "verify.jsonl")
    stato_dopo = np.random.get_state()

    # Il percorso rigenerato è davvero stato percorso, altrimenti il test
    # non avrebbe coperto i round di backtracking.
    assert result.regeneration_rounds_used == 2
    assert len(model.calls) == 9
    assert _stati_rng_identici(stato_dopo, stato_prima)


def test_rami_restano_riproducibili_dopo_il_fix_rng():
    """Il ripristino non deve rompere la riproducibilita' dei rami."""
    testo = "uno\ndue"

    np.random.seed(99)
    primo = generate_branches("ciao", ScriptedModel([testo] * 3))
    np.random.seed(99)
    secondo = generate_branches("ciao", ScriptedModel([testo] * 3))

    assert [b.testo_completo for b in primo] == [b.testo_completo for b in secondo]
    assert [b.seed for b in primo] == [b.seed for b in secondo]


def test_loop_di_step_applica_la_penalita_fissa():
    """Due step consecutivi identici costano -0.3."""
    testo = "Passo\nPasso\nquindi la risposta è 42"
    evaluation = evaluate_branch(make_branch("branch_0", testo), "other")
    assert evaluation.loop_rilevati == 1
    assert evaluation.score_ramo == pytest.approx(0.7)


def test_contraddizione_lessicale_applica_la_penalita_fissa():
    """Marker contraddittori in step successivi costano -0.5."""
    testo = "Quindi la risposta è 42\nMa in realtà non è così"
    evaluation = evaluate_branch(make_branch("branch_0", testo), "other")
    assert evaluation.contraddizioni_rilevate >= 1
    assert evaluation.score_ramo == pytest.approx(0.5)


def test_contraddizione_su_valori_finali_diversi():
    """X != Y sullo stesso ramo, con marker 'risposta', e' una contraddizione."""
    testo = "quindi la risposta è 42\nquindi la risposta è 43"
    evaluation = evaluate_branch(make_branch("branch_0", testo), "other")
    assert evaluation.contraddizioni_rilevate >= 1
    assert evaluation.score_ramo < 1.0


def test_riconciliazione_fallita_scarta_il_ramo():
    """Il valore dichiarato diverso dal ricalcolo costa -1.0 e azzera il ramo."""
    testo = "2 + 3 = 5\nquindi la risposta è 8"
    evaluation = evaluate_branch(make_branch("branch_0", testo), "arithmetic")
    assert evaluation.riconciliazione_eseguita is True
    assert evaluation.score_ramo == 0.0
    assert evaluation.scarto is True


def test_punteggio_mai_negativo():
    """Il floor a 0.0 tiene anche in caso di penalita' multiple."""
    testo = (
        "quindi la risposta è 42\n"
        "quindi la risposta è 43\n"
        "ma in realtà non è così\n"
        "Ripeto: passo\n"
        "Ripeto: passo"
    )
    evaluation = evaluate_branch(make_branch("branch_0", testo), "other")
    assert evaluation.score_ramo == 0.0


def test_nodi_non_whitelisted_vengono_rifiutati():
    """Un nome di funzione mascherato da aritmetica fa fallire la sintassi."""
    testo = "len(abc) = 5, quindi la risposta è 5"
    evaluation = evaluate_branch(make_branch("branch_0", testo), "arithmetic")
    assert evaluation.score_sintassi == 0.0
    assert "non aritmetico camuffato" in evaluation.sintassi_dettaglio


def test_sintassi_non_applicabile_su_other():
    """Su "other" score_sintassi e' None e non penalizza."""
    evaluation = evaluate_branch(make_branch("branch_0", "testo qualsiasi"), "other")
    assert evaluation.score_sintassi is None
    assert evaluation.score_ramo == 1.0


def test_tutti_rami_scartati_espone_verification_failed(tmp_path):
    """Nessun ramo sopravvive: si restituisce il migliore, ma failed=True."""
    sbagliato = "2 + 3 = 5, quindi la risposta è 8"
    model = ScriptedModel([sbagliato] * 3)
    result = verify("Quanto fa 2 + 3?", model, log_path=tmp_path / "verify.jsonl")

    assert result.verification_failed is True
    assert result.reliable is False
    assert result.chosen_branch_id is not None
    assert result.chosen_branch_score is not None
    assert len(result.discarded_branches) == 3
    # Il ramo restituito e' il migliore tra quelli scartati.
    assert result.chosen_branch_score == max(
        item["score"] for item in result.discarded_branches
    )


def test_rigenerazione_rispetta_il_tetto_di_due_round(tmp_path):
    """Tutti i rami sbagliati: rigenerazione entro 2 round, poi decisione."""
    sbagliato = "2 + 3 = 5, quindi la risposta è 8"
    model = ScriptedModel([sbagliato] * 10)
    result = verify("Quanto fa 2 + 3?", model, log_path=tmp_path / "verify.jsonl")

    assert result.regeneration_rounds_used == 2
    assert result.verification_failed is True
    # 3 rami iniziali + 3 rigenerati per round = 3 + 6 = 9 chiamate al modello.
    assert len(model.calls) == 9


def test_rigenera_solo_i_rami_scartati(tmp_path):
    """Con un solo ramo scartato non si rigenera: sotto la soglia di 2 scarti."""
    buono = "2 + 3 = 5, quindi la risposta è 5"
    cattivo = "2 + 3 = 5, quindi la risposta è 8"
    model = ScriptedModel([buono, buono, cattivo])
    result = verify("Quanto fa 2 + 3?", model, log_path=tmp_path / "verify.jsonl")

    assert result.regeneration_rounds_used == 0
    assert len(model.calls) == 3
    assert result.reliable is True


def test_parita_ha_vince_la_temperatura_piu_bassa(tmp_path):
    """A parita' di punteggio vince il ramo piu' deterministico."""
    testo = "2 + 3 = 5, quindi la risposta è 5"
    model = ScriptedModel([testo] * 3)
    result = verify("Quanto fa 2 + 3?", model, log_path=tmp_path / "verify.jsonl")

    assert result.chosen_branch_id == "branch_0"
    assert result.chosen_branch_score == 1.0


def test_decisione_deterministica_a_parita_di_input(tmp_path):
    """Stesso input e stessi rami producono sempre la stessa decisione."""
    testo = "2 + 3 = 5, quindi la risposta è 5"
    primo = verify(
        "Quanto fa 2 + 3?",
        ScriptedModel([testo] * 3),
        log_path=tmp_path / "a.jsonl",
    )
    secondo = verify(
        "Quanto fa 2 + 3?",
        ScriptedModel([testo] * 3),
        log_path=tmp_path / "b.jsonl",
    )

    assert primo.chosen_branch_id == secondo.chosen_branch_id
    assert primo.chosen_branch_text == secondo.chosen_branch_text
    assert primo.chosen_branch_score == secondo.chosen_branch_score
    assert primo.discarded_branches == secondo.discarded_branches


def test_log_jsonlines_contiene_tutti_i_campi_e_i_punteggi_intermedi(tmp_path):
    """Un record per chiamata, con tutti i campi e i punteggi di ogni ramo."""
    import json

    buono = "2 + 3 = 5, quindi la risposta è 5"
    cattivo = "2 + 3 = 5, quindi la risposta è 8"
    log_file = tmp_path / "verify.jsonl"

    verify("Quanto fa 2 + 3?", ScriptedModel([buono, buono, cattivo]), log_path=log_file)

    righe = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(righe) == 1

    record = json.loads(righe[0])
    for campo in (
        "input_text",
        "domain",
        "reliable",
        "chosen_branch_id",
        "chosen_branch_text",
        "chosen_branch_score",
        "discarded_branches",
        "regeneration_rounds_used",
        "verification_failed",
        "timestamp",
    ):
        assert campo in record

    # I punteggi intermedi di STEP 3 devono esserci per TUTTI i rami.
    assert len(record["evaluations"]) == 3
    for evaluation in record["evaluations"]:
        for campo in (
            "branch_id",
            "temperature",
            "seed",
            "score_sintassi",
            "sintassi_dettaglio",
            "loop_rilevati",
            "contraddizioni_rilevate",
            "contraddizioni_dettaglio",
            "riconciliazione_eseguita",
            "riconciliazione_dettaglio",
            "score_ramo",
            "scarto",
            "motivo_scarto",
        ):
            assert campo in evaluation

    # Anche i seed dei rami devono essere nel log (STEP 1).
    assert len({branch["seed"] for branch in record["branches"]}) == 3


def test_log_usa_la_cartella_convenzionale_del_progetto():
    """Il log riusa la cartella audit_logs gia' presente nel progetto."""
    assert LOG_DIR.name == "audit_logs"


# ---------------------------------------------------------------------------
# Integrazione con il modello reale (NeuralModelInterface + MicroTransformer)
# ---------------------------------------------------------------------------


def test_integrazione_reale_con_microtransformer(tmp_path):
    """Verifica end to end sul generatore vero del progetto.

    Il modello non e' addestrato: il testo generato non e' un calcolo. Questo
    test verifica l'integrazione (temperature, seed riproducibili, logging),
    NON la capacita' del modello di risolvere il problema.
    """
    from neural_core.model_interface import NeuralModelInterface
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit("Risposta la risposta e quindi il risultato 2 + 3 = 5")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), seed=7)
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    primo = verify("Quanto fa 2 + 3?", interface, log_path=tmp_path / "reale.jsonl")
    secondo = verify("Quanto fa 2 + 3?", interface, log_path=tmp_path / "reale.jsonl")

    # Lo stesso seed base produce lo stesso testo: la decisione e' riproducibile.
    assert primo.chosen_branch_text == secondo.chosen_branch_text
    assert primo.chosen_branch_id == secondo.chosen_branch_id
    assert primo.domain == "arithmetic"
    assert len(primo.branches) == 3
    assert [branch["temperature"] for branch in primo.branches] == [0.2, 0.6, 1.0]


def test_generazione_rumore_usa_davvero_i_seed(tmp_path):
    """Su rumore non addestrato i 3 seed producono rami distinti."""
    from neural_core.model_interface import NeuralModelInterface
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro cinque sei sette otto nove dieci")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), seed=7)
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    branches = generate_branches("uno due", interface, base_seed=DEFAULT_BASE_SEED)

    assert len(branches) == 3
    assert len({branch.testo_completo for branch in branches}) > 1
