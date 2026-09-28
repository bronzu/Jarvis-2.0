"""
Missione 1, Fase B — Unlearning causale: test.

Il braccio causale risponde a una domanda diversa da quella di Fase A: l'informazione
di X_forget e' localizzata in qualche punto del modello, e quel punto e' piccolo?

I test sono su due livelli, come in `test_unlearning.py`:

1. **strutturali** (veloci, senza addestramento): inventario dell'architettura,
   costruzione e disgiunza della coppia clean/corrupted, comportamento dei ganci
   sulle attivazioni, ordinamento del pool di candidati;
2. **misurati** (un addestramento condiviso a livello di modulo): il patching
   localizza il segnale sulle posizioni dell'entita' sostituita, e il set minimo
   che soddisfa il solo criterio di forgetting distrugge il collaterale.

AVVERTENZA SULLA VALIDITA' DEI RISULTATI
-----------------------------------------
Corpus reale di 97 frasi, un dominio, un modello da 162.649 parametri di cui
1.536 addestrabili, un solo seme. NON e' una prova generale di localizzazione
causale. Qui il modello sovraffitta il corpus, e su un modello in
sovraffittamento la rappresentazione e' per costruzione ridondante: i risultati
di localizzazione valgono per QUESTO corpus e NON per il modello di produzione.
Le asserzioni fissano quindi proprieta' strutturali (il segno del ripristino, il
rapporto fra criteri), non numeri di un diagramma concettuale.

Alcune asserzioni fissano esiti NEGATIVI (nessuna localizzazione che rispetti
anche il collaterale). Non sono test che "vogliono fallire": impediscono a un
cambiamento futuro di dichiarare una localizzazione che non c'e'.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "experiments" / "mission1_unlearning"))

import causal_unlearning as C  # noqa: E402
import corpus_unlearning as corpus  # noqa: E402
import validate_unlearning as V  # noqa: E402

from neural_core.unlearning import SelectiveUnlearner, UnlearningConfig  # noqa: E402

#: Epoche di addestramento per i test misurati: le stesse 400 del report, cosi'
#: che i test e il report misurino lo stesso modello. Con un numero minore di
#: epoche il set minimo non veniva raggiunto e il test centrale della Fase B
#: saltava invece di verificare qualcosa: un test che salta non copre.
TEST_EPOCHS = 400


# ======================================================================
# 1. Test strutturali
# ======================================================================

@pytest.fixture(scope="module")
def pipeline():
    """Split, tokenizer e modello non addestrato: bastano per B0 e B1."""
    data = corpus.build_pipeline()
    trainer = V.build_trainer(data["tokenizer"])
    return {"splits": data["splits"], "tokenizer": data["tokenizer"],
            "trainer": trainer}


def test_b0_inventario_coerente_con_il_codice(pipeline):
    """
    L'inventario non viene da una specifica scritta: viene dagli attributi del
    modello. Il test verifica che i siti patchabili siano esattamente uno per
    blocco piu' il flusso residuo finale.
    """
    trainer = pipeline["trainer"]
    inventory = C.architecture_inventory(trainer.model, trainer, log=lambda *a, **k: None)
    config = trainer.model.config()
    assert inventory["config"] == config
    assert inventory["n_sites"] == config["num_layers"] + 1
    assert inventory["head_dim"] == config["d_model"] // config["num_heads"]
    assert head_dim_is_consistent(config)
    assert inventory["n_parameters_trainable"] == sum(
        int(np.asarray(trainer.model.get_parameters()[i]).size)
        for i in trainer._trainable_indices())


def head_dim_is_consistent(config):
    return config["d_model"] % config["num_heads"] == 0


def test_b1_coppia_pulita_corrotta_stessa_lunghezza_e_senza_unk(pipeline):
    """
    Se la sostituzione cambiasse il numero di token o introducesse token fuori
    vocabolario, la differenza misurata dal patching sarebbe di lunghezza o
    lessicale, non semantica. Il controllo e' un'asserzione, non un commento.
    """
    unlearner = SelectiveUnlearner(pipeline["trainer"],
                                   config=UnlearningConfig(),
                                   log=lambda *a, **k: None)
    clean = pipeline["splits"]["forget"]
    corrupted = C.build_corrupted(clean)
    assert corrupted != clean
    assert C.TARGET_ENTITY not in corrupted
    assert C.REPLACEMENT_ENTITY in corrupted
    info = C.validate_pair(unlearner, clean, corrupted, log=lambda *a, **k: None)
    assert info["n_tokens"] > 0
    assert info["unk_clean"] == 0
    assert info["unk_corrupted"] == 0
    # Se la coppia non cambiasse la probabilita', il patching non misurerebbe
    # niente: e' la precondizione di non-vacuita' della Fase B.
    assert info["corrupted_loglik_per_token"] != info["clean_loglik_per_token"]


def test_b1_entita_sostitutiva_e_davvero_nel_vocabolario(pipeline):
    """
    Il vocabolario di 729 token non contiene altri nomi propri di persona
    (verificato: nessuna alternativa tipo "Rossi" o "Bianchi" e' presente).
    La sostituzione doveva quindi usare un'entita' gia' nel corpus, altrimenti
    il confronto sarebbe lessicale. Il test fissa questa scelta.
    """
    vocab = pipeline["tokenizer"].vocabulary
    # La sostituzione e' di due token, quindi nel vocabolario (per token)
    # compaiono i due pezzi, non la frase intera.
    assert all(word in vocab for word in C.REPLACEMENT_ENTITY.split())
    for word in C.REPLACEMENT_ENTITY.split():
        assert word in vocab
    for assente in ("Lorenzo", "Rossi", "Bianchi", "Fabbri"):
        assert assente not in vocab, (
            "Se il vocabolario contenesse un nome proprio alternativo, la "
            "sostituzione andrebbe rivista: un nome nuovo sarebbe un test "
            "migliore di questo."
        )


def test_ganci_ripristinano_i_metodi_all_uscita(pipeline):
    """I ganci non devono lasciare il modello modificato strutturalmente."""
    model = pipeline["trainer"].model
    before = [layer.forward for layer in model.layers] + [model.norm.forward]
    with C.ResidualHooks(model) as hooks:
        assert hooks.record == {}
        assert [layer.forward for layer in model.layers] != before[:len(model.layers)]
    after = [layer.forward for layer in model.layers] + [model.norm.forward]
    assert before == after


def test_azzeramento_coordinale_salta_le_sequenze_piu_corte(pipeline):
    """
    Un intervento definito su posizioni di token non e' applicabile a un input
    piu' corto di quella posizione. Il comportamento (saltare, non sollevare) e'
    quello dichiarato nel report, e va fissato da un test perche' un
    IndexError qui renderebbe impossibile misurare il collaterale su frasi di
    lunghezza diverse.
    """
    trainer = pipeline["trainer"]
    unlearner = SelectiveUnlearner(trainer, config=UnlearningConfig(),
                                   log=lambda *a, **k: None)
    text = "uno due tre quattro cinque sei sette"
    inputs, targets = unlearner.windows_from_text(text)
    with C.ResidualHooks(trainer.model) as hooks:
        hooks.zero_coords[0] = {(0, 0), (inputs.shape[1] + 5, 0)}
        nll = unlearner._forward_nll_sum(inputs, targets)
    assert np.isfinite(nll)


def test_pool_di_candidati_e_ordinato_per_punteggio(pipeline):
    """
    Il greedy richiede candidati ordinati per punteggio causale decrescente e
    deve contenere tutte e tre le granularita'. Un pool mal ordinato produrrebbe
    un "set minimo" che non e' minimo, senza alcun errore visibile.
    """
    coord_rows = [{"site": 0, "position": 4, "coord": 2, "score": 0.02},
                  {"site": 1, "position": 4, "coord": 3, "score": 0.01}]
    site_rows = [{"site": 0, "position": 4, "score_forward": 0.08},
                 {"site": 1, "position": 5, "score_forward": -0.01}]
    pool = C.build_candidate_pool(coord_rows, site_rows, n_sites=2)
    scores = [c["score"] for c in pool]
    assert scores == sorted(scores, reverse=True)
    granularities = {c["granularity"] for c in pool}
    assert granularities == {"coordinata", "posizione", "sito"}
    assert len(pool) == 2 + 2 + 2


# ======================================================================
# 2. Test misurati sul percorso completo
# ======================================================================

@pytest.fixture(scope="module")
def trained():
    return V.build_trained(epochs=TEST_EPOCHS, log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def unlearner(trained):
    V.restore_trainable(trained["trainer"], trained["theta0"])
    return SelectiveUnlearner(trained["trainer"], config=UnlearningConfig(),
                              log=lambda *a, **k: None)


def test_b2_il_patching_localizza_l_entita_sostituita(trained, unlearner):
    """
    Il segnale causale deve concentrarsi sulle posizioni dell'entita' che e'
    stata sostituita. Se non lo facesse, la Fase B starebbe misurando rumore e
    il resto dell'esperimento non significherebbe niente.
    """
    splits = trained["splits"]
    clean = splits["forget"]
    corrupted = C.build_corrupted(clean)
    assert clean.split() != corrupted.split()
    # Indici dei token dell'entita' nella finestra tokenizzata.
    clean_inputs, clean_targets = unlearner.windows_from_text(clean)
    ids = unlearner.tokenizer.encode(clean, add_end_of_text=True)
    clean_words = clean.split()
    pos_entity = [i for i, w in enumerate(clean_words)
                  if w in C.TARGET_ENTITY.split()]
    assert pos_entity, "le posizioni dell'entita' devono essere trovabili"

    scan = C.patch_scan_site_level(unlearner, clean_inputs, clean_targets,
                                   unlearner.windows_from_text(corrupted)[0],
                                   n_sites=3, log=lambda *a, **k: None)
    rows = scan["rows"]
    assert len(rows) == 3 * len(clean_words)
    best = max(rows, key=lambda r: r["score_forward"])
    assert best["position"] in pos_entity, (
        f"il patching individua la posizione {best['position']}, "
        f"l'entita' e' alle posizioni {pos_entity}"
    )
    # Il segnale non e' distribuito su tutte le posizioni.
    by_position = {}
    for row in rows:
        by_position.setdefault(row["position"], []).append(row["score_forward"])
    mean_at_entity = float(np.mean([np.mean(by_position[p]) for p in pos_entity]))
    mean_elsewhere = float(np.mean([np.mean(v) for p, v in by_position.items()
                                    if p not in pos_entity]))
    assert mean_at_entity > mean_elsewhere
    # Le due direzioni del patching devono avere segno opposto: se la
    # dipendenza fosse un artefatto della sostituzione, i due segni coinciderebbero.
    for row in rows:
        if row["score_forward"] > 0:
            assert row["score_reverse"] < 0


def test_b6_il_criterio_di_forgetting_sole_e_soddisfacibile_solo_rompendo_il_modello(
        trained, unlearner):
    """
    ESITO MISURATO, e il punto scientifico della Fase B.

    Il criterio di successo (1) da solo e' soddisfacibile da qualunque
    intervento che distrugga il modello: azzerare il flusso residuo porta
    X_forget a una NLL quasi al massimo raggiungibile, ma porta le frasi
    correlate e il canario fuori dal 5% di ordini di grandezza.

    Il test verifica la proprieta' che conta, senza fissare quale dei due
    esiti si presenti (raggiungere la soglia, oppure saturare prima):
    l'intervento di attivazione non puo' soddisfare i due criteri insieme.
    """
    splits = trained["splits"]
    clean = splits["forget"]
    clean_inputs, clean_targets = unlearner.windows_from_text(clean)
    corrupted_inputs = unlearner.windows_from_text(C.build_corrupted(clean))[0]
    baseline = C.build_baseline_measures(unlearner, splits)
    target_nll = C._threshold_from_holdout(baseline["holdout"])
    n_sites = 3

    coord = C.patch_scan_coordinate_level(unlearner, clean_inputs, clean_targets,
                                          corrupted_inputs, n_sites,
                                          log=lambda *a, **k: None)
    site = C.patch_scan_site_level(unlearner, clean_inputs, clean_targets,
                                   corrupted_inputs, n_sites,
                                   log=lambda *a, **k: None)
    pool = C.build_candidate_pool(coord["rows"], site["rows"], n_sites)
    minimal = C.greedy_minimal_set(unlearner, clean_inputs, clean_targets, pool,
                                   target_nll, n_sites, log=lambda *a, **k: None)

    # Il ceiling dell'ablation e' la NLL di un predittore costante: e' il
    # massimo che qualunque ablation di attivazione puo' ottenere.
    assert minimal["ceiling_nll_per_token"] > minimal["baseline_nll_per_token"]
    if not minimal["reached"]:
        assert minimal["ceiling_nll_per_token"] < target_nll, (
            "il set minimo non e' stato dichiarato raggiunto ma la sua NLL "
            "migliore supera la soglia: la logica di uscita e' incoerente"
        )
        return

    V.restore_trainable(trained["trainer"], trained["theta0"])
    row = _apply_and_measure(unlearner, splits, baseline, minimal, clean_inputs)
    assert row["criteria"]["forget_nll_per_token"] >= target_nll
    assert not row["criteria"]["collateral_success"], (
        "il set minimo non vincolato ha rispettato il collaterale: il test "
        "riporta un esito diverso da quello misurato, va riletto"
    )
    assert row["criteria"]["collateral_max_change"] > C.COLLATERAL_TOLERANCE
    assert row["criteria"]["probe_max_change"] > C.COLLATERAL_TOLERANCE


def _apply_and_measure(unlearner, splits, baseline, minimal, clean_inputs):
    """Applica l'ablation del set minimo e la misura con le metriche di Fase A."""
    with C.ResidualHooks(unlearner.model) as hooks:
        zero_by_site = {}
        for comp in minimal["components"]:
            site = comp["site"]
            zero_by_site.setdefault(site, set())
            if comp["granularity"] == "coordinata":
                zero_by_site[site].add((comp["position"], comp["coord"]))
            elif comp["granularity"] == "posizione":
                for coord in range(int(unlearner.model.d_model)):
                    zero_by_site[site].add((comp["position"], coord))
            else:
                for pos in range(int(np.asarray(clean_inputs).shape[1])):
                    for coord in range(int(unlearner.model.d_model)):
                        zero_by_site[site].add((pos, coord))
        for site, coords in zero_by_site.items():
            hooks.zero_coords[site] = coords
        return C.criteria_for(unlearner, splits, baseline, "test", minimal["k"])


def test_b6c_la_ricerca_vincolata_non_inventa_localizzazioni(trained, unlearner):
    """
    Con il vincolo di collaterale il risultato deve essere "non raggiunto", o
    essere raggiunto con zero violazioni. Qualsiasi altro esito significa che
    il filtro non sta funzionando e va investigato prima di fidarsi.
    """
    splits = trained["splits"]
    clean = splits["forget"]
    clean_inputs, clean_targets = unlearner.windows_from_text(clean)
    baseline = C.build_baseline_measures(unlearner, splits)
    target_nll = C._threshold_from_holdout(baseline["holdout"])
    n_sites = 3
    corrupted_inputs = unlearner.windows_from_text(C.build_corrupted(clean))[0]

    coord = C.patch_scan_coordinate_level(unlearner, clean_inputs, clean_targets,
                                          corrupted_inputs, n_sites,
                                          log=lambda *a, **k: None)
    site = C.patch_scan_site_level(unlearner, clean_inputs, clean_targets,
                                   corrupted_inputs, n_sites,
                                   log=lambda *a, **k: None)
    pool = C.build_candidate_pool(coord["rows"], site["rows"], n_sites)
    constrained = C.greedy_minimal_set_constrained(
        unlearner, splits, baseline, clean_inputs, clean_targets, pool,
        target_nll, n_sites, log=lambda *a, **k: None)
    if constrained["reached"]:
        for entry in constrained["trace"]:
            assert entry["collateral_violations"] == 0
    else:
        broken = constrained["collateral_broken_at"]
        assert broken is not None
        assert broken["nll_per_token"] < target_nll, (
            "l'accumulo si e' fermato per collaterale rotto con una NLL gia' "
            "oltre la soglia: i due criteri sarebbero stati soddisfatti insieme"
        )


def test_riferimento_senza_intervento_e_pulito(trained, unlearner):
    """
    Ogni variazione riportata nella Fase B e' riferita a un baseline misurato
    sul modello non modificato. Se quel baseline non fosse pulito, tutte le
    variazioni sarebbero falsate. Questo test verifica che lo sia, e che il
    modello non venga lasciato modificato.
    """
    V.restore_trainable(trained["trainer"], trained["theta0"])
    splits = trained["splits"]
    baseline = C.build_baseline_measures(unlearner, splits)
    row = C.criteria_for(unlearner, splits, baseline, "riferimento", 0)
    assert row["criteria"]["collateral_max_change"] == pytest.approx(0.0, abs=1e-12)
    assert row["criteria"]["probe_max_change"] == pytest.approx(0.0, abs=1e-12)
    forget = unlearner.loglikelihood(splits["forget"])
    assert forget["nll_mean"] < 5.0, (
        "X_forget non e' memorizzato: se la NLL e' gia' alta prima di ogni "
        "intervento, la Fase B starebbe misurando una frase che il modello non "
        "sapeva"
    )
