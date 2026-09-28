"""
Missione 1 — Machine unlearning selettivo: test.

I test coprono due livelli distinti:

1. **Test strutturali** (veloci, senza addestramento): formule di proiezione,
   criterio di successo, finestre di contesto, validazione dei parametri,
   riuso della Fisher esistente, blocco dei parametri congelati.
2. **Test misurato** (un addestramento condiviso a livello di modulo): il
   percorso completo dati -> training -> Fisher -> unlearning -> criteri, con
   le asserzioni sugli esiti REALI misurati in questo file.

AVVERTENZA SULLA VALIDITA' DEI RISULTATI
-----------------------------------------
I test misurati usano un corpus reale del repository
(`neural_core/data/jarvis_dialogue.txt`, 97 frasi), un solo dominio, un solo
modello, un solo seme. NON sono una prova generale di capacita' di machine
unlearning: sono un test di fumo quantitativo. Ogni numero riportato nelle
asserzioni e' prodotto da un'esecuzione reale di questo file e i valori
misurati sono confrontati con i numeri del report di competenza
(`DOCUMENTAZIONE/MISSION1_UNLEARNING_REPORT.md`).

Alcune asserzioni fissano esiti NEGATIVI (il protocollo prescritto non
raggiunge i criteri in questa configurazione). Non e' un test che "vuole" fallire:
e' il test che impedisce a un future cambiamento di far passare sotto silenzio
una regressione o, al contrario, di dichiarare un successo non misurato.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "experiments" / "mission1_unlearning"))

# Nomi univoci (`corpus_unlearning`, `validate_unlearning`): nella stessa
# sessione pytest `TESTS/test_continual_learning_criteria.py` importa
# `corpus` e `validate` da experiments/phase4_continual_learning. Con gli
# stessi nomi, il secondo import troverebbe il modulo gia' in sys.modules e i
# due test misurerebbero la pipeline sbagliata.
import corpus_unlearning as corpus  # noqa: E402
import validate_unlearning as V  # noqa: E402

from neural_core.trainer import Trainer  # noqa: E402
from neural_core.unlearning import (  # noqa: E402
    STOP_DEGENERATE_PROJECTION, SelectiveUnlearner, UnlearningConfig,
    forgetting_success, relative_probability_change, single_vs_distribution,
)

# ----------------------------------------------------------------------
# Numeri misurati (esecuzione reale, vedi il report di competenza).
# Se l'implementazione o il protocollo cambiano, questi valori vanno
# rimesurati: sono la baseline di regressione, non un'asserzione da
# aggiornare a mano per far passare il test.
# ----------------------------------------------------------------------
MEASURED_TRAIN_LOSS_FINAL_MAX = 5.5      # loss finale osservata 5.3616
MEASURED_FORGET_NLL_BEFORE_MAX = 5.0      # NLL/token di X_forget prima 4.6834
MEASURED_HOLDOUT_NLL_MIN = 6.5            # minimo NLL/token holdout 6.8947
MEASURED_PROTOCOL_Z_MIN = 5.0             # z del criterio (1) col protocollo 7.702
MEASURED_PROTOCOL_COLLATERAL_VIOLATIONS_MIN = 10   # 12/12 oltre soglia


# ======================================================================
# 1. Test strutturali: formule e criteri, senza addestramento
# ======================================================================

def test_proiezione_prescritta_coincide_con_la_chiusura_analitica():
    """La formula prescritta e' quella dichiarata, elemento per elemento."""
    g_forget = [np.array([1.0, 2.0, -0.5]), np.array([0.25, -1.0])]
    g_retain = [np.array([0.3, -0.2, 1.5]), np.array([2.0, 0.1])]
    fisher = [np.array([0.1, 0.4, 0.9]), np.array([0.05, 0.7])]

    projected, alpha, den = SelectiveUnlearner.project_in_fisher_metric(
        g_forget, g_retain, fisher)

    inv = [1.0 / f for f in fisher]
    num = sum(float(np.sum(a * i * b)) for a, b, i in zip(g_forget, g_retain, inv))
    expected_den = sum(float(np.sum(i * b * b)) for i, b in zip(inv, g_retain))
    assert den == pytest.approx(expected_den)
    assert alpha == pytest.approx(num / expected_den)
    for got, a, i, b in zip(projected, g_forget, inv, g_retain):
        assert np.allclose(got, a - alpha * i * b)


def test_proiezione_prescritta_non_e_ortogonale_con_fisher_anisotropa():
    """
    Difetto misurabile della formula prescritta: il termine rimosso e' un
    multiplo di F^-1 g_r, quindi <g_proj, g_r> NON si annulla se F^-1 e' diverso
    dall'identita'. Il test falla se qualcuno "corregge" la formula senza
    sapere che cambia il protocollo.
    """
    g_forget = [np.array([1.0, 2.0])]
    g_retain = [np.array([0.3, -0.2])]
    fisher = [np.array([1e-3, 1e2])]        # Fisher fortemente anisotropa

    projected, _, _ = SelectiveUnlearner.project_in_fisher_metric(
        g_forget, g_retain, fisher)
    residual = float(np.sum(projected[0] * g_retain[0]))
    assert abs(residual) > 1e-6, (
        "Con questa Fisher la formula prescritta dovrebbe lasciare un residuo "
        "non nullo: se il test passa con residuo ~0 il contratto e' cambiato."
    )


def test_proiezione_metrica_ortogonale_e_davvero_ortogonale():
    """project_metric_orthogonal annulla <g_proj, F^-1 g_r> per ogni Fisher."""
    rng = np.random.RandomState(0)
    for _ in range(20):
        g_forget = [rng.randn(4), rng.randn(3)]
        g_retain = [rng.randn(4), rng.randn(3)]
        fisher = [np.abs(rng.randn(4)) * 1e-2 + 1e-6,
                  np.abs(rng.randn(3)) * 1e2 + 1e-6]
        projected, _, den = SelectiveUnlearner.project_metric_orthogonal(
            g_forget, g_retain, fisher)
        assert den > 0
        u = [(1.0 / f) * g for f, g in zip(fisher, g_retain)]
        residual = sum(float(np.sum(p * uu)) for p, uu in zip(projected, u))
        assert abs(residual) < 1e-8 * max(1.0, abs(den))


def test_proiezione_degenere_se_il_gradiente_di_retain_e_nullo():
    """Se g_retain e' nullo non esiste direzione di protezione: alpha = 0."""
    projected, alpha, den = SelectiveUnlearner.project_metric_orthogonal(
        [np.array([1.0, 2.0])], [np.zeros(2)], [np.ones(2)])
    assert den == 0.0
    assert alpha == 0.0
    assert np.allclose(projected[0], [1.0, 2.0])


def test_forgetting_success_unilaterale_e_bilaterale():
    """Il criterio unilaterale accetta anche un modello distrutto."""
    control = [-7.0, -7.5, -6.5, -7.2, -6.8, -7.4]
    dentro = forgetting_success(-7.1, control)
    assert dentro["success_one_sided"] and dentro["success_two_sided"]

    sotto = forgetting_success(-30.0, control)
    assert sotto["success_one_sided"]          # z molto negativo
    assert not sotto["success_two_sided"]      # NON e' un rientro nella distribuzione

    sopra = forgetting_success(-4.0, control)
    assert not sopra["success_one_sided"]
    assert not sopra["success_two_sided"]


def test_forgetting_success_richiede_due_controlli():
    with pytest.raises(ValueError):
        forgetting_success(-7.0, [-7.0])


def test_relative_probability_change():
    """La variazione relativa e' |p'-p|/p calcolata in log-spazio."""
    before = [-10.0, -20.0]
    after = [before[0] + math.log(1.1), before[1] - math.log(2.0)]
    change = relative_probability_change(before, after)
    assert change[0] == pytest.approx(0.1, rel=1e-9)   # p' = 1.1 p
    assert change[1] == pytest.approx(0.5, rel=1e-9)   # p' = 0.5 p -> |Δp|/p = 0.5


def test_relative_probability_change_scarta_spazi_disomogenei():
    with pytest.raises(ValueError):
        relative_probability_change([-1.0, -2.0], [-1.0])


def test_single_vs_distribution():
    """t = (x - media) / (std / sqrt(n+1)); p alto quando x e' nella coda."""
    campione = [-7.0, -7.5, -6.5, -7.2, -6.8, -7.4]
    dentro = single_vs_distribution(-7.0, campione)
    assert dentro["n"] == 6
    assert abs(dentro["p_value"]) <= 1.0
    lontano = single_vs_distribution(-20.0, campione)
    assert lontano["p_value"] < dentro["p_value"]


def test_configurazione_rifiuta_valori_invalidi():
    with pytest.raises(ValueError):
        UnlearningConfig(base_learning_rate=0.0)
    with pytest.raises(ValueError):
        UnlearningConfig(max_steps=0)
    with pytest.raises(ValueError):
        UnlearningConfig(fisher_floor=0.0)
    with pytest.raises(ValueError):
        UnlearningConfig(fisher_floor=1.0)
    with pytest.raises(ValueError):
        UnlearningConfig(projection_mode="inesistente")
    with pytest.raises(ValueError):
        UnlearningConfig(context_size=1)


def test_selettive_unlearner_esige_una_fisher():
    trainer, _ = _minimal_trainer()
    with pytest.raises(ValueError, match="Fisher"):
        SelectiveUnlearner(trainer, log=lambda *a, **k: None).forget(
            "una frase qualsiasi", ["un'altra frase"])


def test_selettive_unlearner_esige_un_trainer():
    with pytest.raises(ValueError, match="Trainer"):
        SelectiveUnlearner(None)


def test_selettive_unlearner_esige_retain_non_vuoto():
    trainer, fisher = _minimal_trainer(with_fisher=True)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, log=lambda *a, **k: None)
    with pytest.raises(ValueError):
        unlearner.forget("una frase", [])


def test_fisher_con_chiave_inesistente_viene_rifiutata():
    trainer, _ = _minimal_trainer(with_fisher=True)
    unlearner = SelectiveUnlearner(trainer, fisher={"parametro_inesistente": np.ones(3)},
                                   log=lambda *a, **k: None)
    with pytest.raises(KeyError):
        unlearner.forget("uno due tre", ["uno due tre quattro"])


# ----------------------------------------------------------------------
# Fixture leggera: modello minuscolo, nessun addestramento.
# Serve ai test strutturali che toccano il modello.
# ----------------------------------------------------------------------

def _minimal_trainer(with_fisher=False, lora_rank=0):
    import neural_core.trainer as trainer_module
    from neural_core.dataset import Dataset
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam
    from neural_core.tokenizer import Tokenizer

    text = ("uno due tre quattro cinque sei sette otto "
            "uno due tre quattro cinque sei sette otto")
    tokenizer = Tokenizer(mode="whitespace")
    tokenizer.fit(text)
    model = trainer_module.build_model("transformer", vocab_size=tokenizer.vocab_size,
                                       d_model=8, num_heads=2, d_ff=16, num_layers=1,
                                       max_len=16, lora_rank=lora_rank, seed=1)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=CrossEntropyLoss(),
                      optimizer=Adam(learning_rate=0.01), seed=1)
    dataset = Dataset(tokenizer.encode(text), context_size=8, stride=8)
    if with_fisher:
        fisher = trainer.compute_fisher_matrix(dataset, num_samples=4)
    else:
        fisher = None
    return trainer, fisher


def test_finestre_input_e_target_stessa_lunghezza():
    trainer, fisher = _minimal_trainer(with_fisher=True)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, log=lambda *a, **k: None)
    for text in ("uno due tre", "uno due tre quattro cinque sei sette otto nove dieci"):
        inputs, targets = unlearner.windows_from_text(text)
        assert inputs.shape == targets.shape
        assert inputs.shape[1] <= unlearner.model.max_len
        # il target e' l'input shiftato di uno
        assert int(targets[0][0]) == int(inputs[0][1])


def test_fisher_e_riusata_non_ricalcolata():
    """L'oggetto e' lo stesso del Trainer: nessuna stima nuova."""
    trainer, fisher = _minimal_trainer(with_fisher=True)
    trainer.fisher_matrix = fisher
    unlearner = SelectiveUnlearner(trainer, log=lambda *a, **k: None)
    assert unlearner.fisher is trainer.fisher_matrix
    assert "Trainer.fisher_matrix" in unlearner.fisher_source


def test_pavimento_relativo_sulla_fisher():
    trainer, fisher = _minimal_trainer(with_fisher=True)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher,
                                   config=UnlearningConfig(fisher_floor=0.5),
                                   log=lambda *a, **k: None)
    names, indices, _ = unlearner._trainable_slots()
    effective, raw, threshold = unlearner.fisher_arrays(names, indices)
    peak = max(float(f.max()) for f in raw)
    assert threshold == pytest.approx(0.5 * peak)
    for eff, r in zip(effective, raw):
        # il pavimento SOLO alza le voci sotto soglia: F_eff = max(F, soglia)
        assert np.allclose(eff, np.maximum(r, 0.5 * peak))


def test_fisher_tutta_zero_ritorna_uno():
    """Fisher nulla (parametri mai osservati): l'inverso e' l'identita'."""
    trainer, fisher = _minimal_trainer(with_fisher=True)
    zero = {name: np.zeros_like(np.asarray(value))
            for name, value in fisher.items()}
    unlearner = SelectiveUnlearner(trainer, fisher=zero, log=lambda *a, **k: None)
    names, indices, _ = unlearner._trainable_slots()
    effective, _, threshold = unlearner.fisher_arrays(names, indices)
    assert threshold == 0.0
    for eff in effective:
        assert np.allclose(eff, 1.0)


def test_solo_i_parametri_addestrabili_vengono_modificati():
    """Con LoRA attiva i pesi congelati devono restare identici bit per bit."""
    trainer, fisher = _minimal_trainer(with_fisher=True, lora_rank=2)
    unlearner = SelectiveUnlearner(
        trainer, fisher=fisher,
        config=UnlearningConfig(max_steps=3, halving_window=0,
                                max_restarts=0, base_learning_rate=1e-3),
        log=lambda *a, **k: None)
    before = trainer.model.get_parameters()
    snapshot = [np.array(p, copy=True) for p in before]
    indices = set(trainer._trainable_indices())
    unlearner.forget("uno due tre quattro", ["uno due tre quattro cinque"])
    after = trainer.model.get_parameters()
    for i, (old, new) in enumerate(zip(snapshot, after)):
        if i in indices:
            assert not np.allclose(old, new), "il passo non ha toccato i parametri addestrabili"
        else:
            assert np.array_equal(old, new), f"parametro congelato {i} modificato"


def test_stop_degenere_se_il_gradiente_di_retain_e_nullo():
    """
    Se X_retain_proj produce gradiente nullo la proiezione non e' definita e
    l'ascent si ferma, invece di procedere senza protezione.
    """
    trainer, fisher = _minimal_trainer(with_fisher=True)
    unlearner = SelectiveUnlearner(
        trainer, fisher=fisher,
        config=UnlearningConfig(max_steps=5, halving_window=0, max_restarts=0),
        log=lambda *a, **k: None)
    # Frasi con lo stesso token di inizio: il gradiente addestrabile e' nullo
    # perche' l'adapter LoRA e' a zero e il contributo del gradiente si annulla.
    unlearner.trainable_gradients_on_texts = lambda texts, indices: [
        np.zeros_like(np.asarray(trainer.model.get_parameters()[i]))
        for i in indices
    ]
    result = unlearner.forget("uno due tre", ["uno uno uno"])
    assert result.stop_reason == STOP_DEGENERATE_PROJECTION
    assert result.steps_executed == 0


def test_rollback_riporta_uno_stato_conforme_alla_soglia():
    """
    Con rollback attivo, lo stato finale non supera la soglia sulla loss di
    retain anche quando lo STOP e' provocato proprio dalla soglia.
    """
    trainer, fisher = _minimal_trainer(with_fisher=True)
    config = UnlearningConfig(max_steps=50, halving_window=0, max_restarts=0,
                              base_learning_rate=1e-2, retain_loss_tolerance=0.0)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, config=config,
                                   log=lambda *a, **k: None)
    result = unlearner.forget("uno due tre quattro",
                              ["uno due tre quattro cinque", "uno uno uno due"])
    if result.stop_reason == "retain_loss_increase":
        assert not result.retain_threshold_violated
        assert result.rolled_back_steps > 0
        assert result.retain_loss_after <= result.retain_loss_before + 1e-9


def test_dimezzamento_eta_e_riavvio_da_theta_0():
    """Il riavvio deve azzerare i passi e ripartire dalla loss pre-unlearning."""
    trainer, fisher = _minimal_trainer(with_fisher=True)
    config = UnlearningConfig(max_steps=50, halving_window=5, max_restarts=3,
                              base_learning_rate=1e-1, retain_loss_tolerance=0.0)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, config=config,
                                   log=lambda *a, **k: None)
    result = unlearner.forget("uno due tre quattro",
                              ["uno due tre quattro cinque", "uno uno uno due"])
    if result.restarts > 0:
        assert result.learning_rate_final == pytest.approx(
            result.learning_rate_initial / (2 ** result.restarts))
        for event in result.halving_events:
            assert event["retain_loss"] > event["threshold"]
        # ogni tentativo ricomincia dal passo 1
        per_attempt = {}
        for row in result.history:
            per_attempt.setdefault(row["attempt"], []).append(row["step_in_attempt"])
        for steps in per_attempt.values():
            assert min(steps) == 1


# ======================================================================
# 2. Test misurati sul percorso completo
# ======================================================================

@pytest.fixture(scope="module")
def trained():
    """Modello addestrato + Fisher: condiviso da tutti i test misurati."""
    return V.build_trained(epochs=V.DEFAULT_EPOCHS, log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def protocol_run(trained):
    """
    Il protocollo esatto della missione: eta 1e-04, formula prescritta, e
    SENZA il canario di sicurezza.

    Il canario e' un'aggiunta della Fase A2, non parte del protocollo
    originale: con il canario attivo l'ascent si fermerebbe al primo passo e i
    criteri non sarebbero piu' confrontabili con quelli del protocollo. Il
    confronto "con canario / senza canario" e' in `probe_run`.
    """
    V.restore_trainable(trained["trainer"], trained["theta0"])
    out = V.evaluate_unlearning(trained, log=lambda *a, **k: None, use_probe=False)
    V.restore_trainable(trained["trainer"], trained["theta0"])
    return out


@pytest.fixture(scope="module")
def probe_run(trained):
    """Il protocollo con il canario di sicurezza attivo (A2)."""
    V.restore_trainable(trained["trainer"], trained["theta0"])
    out = V.evaluate_unlearning(trained, log=lambda *a, **k: None, use_probe=True)
    V.restore_trainable(trained["trainer"], trained["theta0"])
    return out


def test_split_disgiunti_e_dimensioni_richieste():
    splits = corpus.build_splits()
    assert 10 <= len(splits["correlated"]) <= 15
    assert len(splits["unseen_holdout"]) >= 5
    assert len(splits["retain"]) > 0
    assert not set(splits["correlated"]) & set(splits["retain"])
    assert not set(splits["correlated"]) & set(splits["unseen_holdout"])
    assert not set(splits["retain"]) & set(splits["unseen_holdout"])
    assert splits["forget"] not in set(splits["unseen_holdout"])
    # Probe/canario: disgiunto sia dalla protezione sia dalla valutazione, e
    # disgiunto anche dal controllo "non protetto", altrimenti lo stesso
    # insieme sarebbe usato sia come segnale di sicurezza sia come giudizio.
    assert len(splits["probe"]) >= 5
    for other in ("retain", "correlated", "unseen_holdout", "archive_unseen",
                  "background_control"):
        assert not set(splits["probe"]) & set(splits[other]), other
    # X_forget entra nel testo di addestramento tramite build_training_text,
    # non tramite training_sentences (che sono solo le frasi del dialogo).
    assert corpus.build_training_text(splits).count(splits["forget"]) == corpus.FORGET_REPEATS


def test_x_forget_e_davvero_memorizzato(trained):
    """
    Se X_forget non fosse memorizzato, il test di forgetting sarebbe vacuo.
    Questa e' la precondizione che rende significativi i criteri.
    """
    V.restore_trainable(trained["trainer"], trained["theta0"])
    unlearner = SelectiveUnlearner(trained["trainer"], log=lambda *a, **k: None)
    splits = trained["splits"]
    forget = unlearner.loglikelihood(splits["forget"])
    control = [unlearner.loglikelihood(t)["mean_loglikelihood"]
               for t in splits["unseen_holdout"]]
    check = forgetting_success(forget["mean_loglikelihood"], control)
    assert check["z_score"] > 1.0, (
        f"X_forget non e' memorizzato (z={check['z_score']}): "
        f"NLL/token {forget['nll_mean']:.4f}. Il test sarebbe vacuo."
    )
    assert forget["nll_mean"] < MEASURED_FORGET_NLL_BEFORE_MAX
    assert forget["nll_mean"] < min(-c for c in control)
    V.restore_trainable(trained["trainer"], trained["theta0"])


def test_addestramento_riproduce_la_loss_misurata(trained):
    assert trained["train_loss_final"] < MEASURED_TRAIN_LOSS_FINAL_MAX
    assert trained["train_loss_final"] < trained["train_loss_initial"]
    assert trained["n_fisher_tensors"] > 0
    assert trained["fisher_seconds"] > 0.0


def test_protocolo_prescritto_usa_la_fisher_esistente(protocol_run):
    u = protocol_run["unlearning"]
    assert u["projection_mode"] == "prescribed"
    assert u["fisher_floor"] == 1e-3
    assert u["learning_rate_initial"] == pytest.approx(1e-4)
    assert u["steps"] <= 100
    assert u["restarts"] <= 3


def test_esito_del_protocollo_prescritto(trained, protocol_run):
    """
    ESITO MISURATO (negativo): con il protocollo esatto della missione
    (eta = lr_fine_tuning/10 = 1e-4, formula di proiezione prescritta,
    pavimento 1e-3, max 100 step) nessuno dei due criteri e' raggiunto.

    I valori sono quelli del report di competenza. Se questo test fallisce per
   che' l'esito e' cambiato, il numero va rimisurato e il report aggiornato:
    non e' un test da far passare.
    """
    crit = protocol_run["criteria"]
    u = protocol_run["unlearning"]
    assert not crit["forgetting_primary"]["success_one_sided"]
    assert not crit["forgetting_primary"]["success_two_sided"]
    assert crit["forgetting_primary"]["z_score"] > MEASURED_PROTOCOL_Z_MIN
    assert not crit["collateral"]["success"]
    assert crit["collateral"]["n_violations"] >= MEASURED_PROTOCOL_COLLATERAL_VIOLATIONS_MIN
    # La soglia sulla loss di retain viene rispettata, ma non protegge le
    # frasi correlate che sono fuori da X_retain_proj.
    assert abs(u["retain_loss_relative_change"]) <= 0.05 + 1e-9


def test_primo_passo_del_protocolto_non_e_un_ascent(trained):
    """
    DIFETTO MISURATO della formula prescritta con la Fisher diagonale della
    LoRA: il termine di proiezione alpha * F^-1 * g_r e' ~100 volte il
    gradiente di forget e ha segno opposto, quindi il passo previsto dal
    protocollo DEGRADA la loss di forget invece di aumentarla.

    Il test verifica la condizione strutturale (norma e segno del passo di
    primo ordine) sulla configurazione misurata, non il valore numerico.
    """
    V.restore_trainable(trained["trainer"], trained["theta0"])
    config = UnlearningConfig(max_steps=1, halving_window=0, max_restarts=0,
                              rollback_on_stop=False)
    unlearner = SelectiveUnlearner(trained["trainer"], config=config,
                                   log=lambda *a, **k: None)
    result = unlearner.forget(trained["splits"]["forget"], trained["splits"]["retain"])
    row = result.history[0]
    assert row["projection_ratio"] > 10.0
    assert row["first_order_forget_delta"] < 0.0
    V.restore_trainable(trained["trainer"], trained["theta0"])


def test_proiezione_corretta_protegge_meglio_la_loss_di_retain(trained):
    """
    Confronto diretto nello STESSO file di test, sullo stesso modello e sugli
    stessi dati: a parita' di eta, la proiezione ortogonale esatta rispetto a
    F^-1 g_r degrada molto meno la loss di X_retain_proj della formula
    prescritta. E' un confronto numerico nello stesso file, non un'affermazione
    di superiorita' su framework esterni.
    """
    eta = 1e-4
    esiti = {}
    for mode in ("prescribed", "metric_orthogonal"):
        V.restore_trainable(trained["trainer"], trained["theta0"])
        out = V.evaluate_unlearning(
            trained,
            unlearning_config=UnlearningConfig(projection_mode=mode),
            learning_rate=eta, log=lambda *a, **k: None, use_probe=False)
        esiti[mode] = out["unlearning"]["retain_loss_relative_change"]
    V.restore_trainable(trained["trainer"], trained["theta0"])
    assert abs(esiti["metric_orthogonal"]) < abs(esiti["prescribed"])


def test_forgetting_raggiungibile_solo_al_costo_del_collaterale(trained):
    """
    ESITO MISURATO: il forgetting si puo' ottenere aumentando eta, ma a quel
    punto il collaterale sulle frasi correlate e' di ordini di grandezza. Il
    metodo non e' "migliore" o "peggiore": e' un punto su una curva di
    compromesso, e qui si dimostra che i due criteri non sono soddisfacibili
    insieme in questa configurazione.
    """
    V.restore_trainable(trained["trainer"], trained["theta0"])
    out = V.evaluate_unlearning(
        trained,
        unlearning_config=UnlearningConfig(projection_mode="metric_orthogonal"),
        learning_rate=1e-3, log=lambda *a, **k: None, use_probe=False)
    V.restore_trainable(trained["trainer"], trained["theta0"])
    crit = out["criteria"]
    assert crit["forgetting_primary"]["success_one_sided"]
    assert not crit["collateral"]["success"]


def test_stato_addestrato_ripristinato_dopo_i_test(trained):
    """Nessun test lascia il modello modificato per chi viene dopo."""
    V.restore_trainable(trained["trainer"], trained["theta0"])
    unlearner = SelectiveUnlearner(trained["trainer"], log=lambda *a, **k: None)
    splits = trained["splits"]
    forget = unlearner.loglikelihood(splits["forget"])
    assert forget["nll_mean"] < MEASURED_FORGET_NLL_BEFORE_MAX
    control = [unlearner.loglikelihood(t)["nll_mean"]
               for t in splits["unseen_holdout"]]
    assert min(control) > MEASURED_HOLDOUT_NLL_MIN


# ======================================================================
# 3. Canario di sicurezza (A2) e diagnostica del segno del passo (A4)
# ======================================================================

def test_probe_texts_richiede_una_soglia():
    """Il probe senza `probe_tolerance` non puo' fermare nulla: deve fallire."""
    trainer, fisher = _minimal_trainer(with_fisher=True)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, log=lambda *a, **k: None)
    with pytest.raises(ValueError, match="probe_tolerance"):
        unlearner.forget("uno due tre", ["uno uno uno"], probe_texts=["uno uno"])


def test_probe_tolerance_negativo_rifiutato():
    with pytest.raises(ValueError, match="probe_tolerance"):
        UnlearningConfig(probe_tolerance=-0.01)


def test_canario_ferma_l_ascent_oltre_la_soglia(probe_run):
    """
    A2, esito misurato: con il canario attivo l'ascent si ferma al primo passo
    e il rollback riporta theta_0, quindi il modello finale e' indistinguibile
    da quello di partenza.
    """
    u = probe_run["unlearning"]
    assert u["stop_reason"] == "probe_drift"
    assert u["steps"] == 1
    assert u["rolled_back_steps"] >= 1
    assert u["retain_loss_after"] == pytest.approx(u["retain_loss_before"], abs=1e-9)
    assert u["forget_nll_after"] == pytest.approx(u["forget_nll_before"], abs=1e-9)


def test_canario_che_ferma_primo_e_un_successo_vacuo(probe_run):
    """
    A2, esito misurato e importante: il canario porta il criterio (2) a
    "RAGGIUNTO" (0/12 oltre soglia) NON perche' il metodo protegga le frasi
    correlate, ma perche' non ha fatto niente. Il rollback ha riportato il
    modello a theta_0.

    Il test impedisce che una riga "criterio (2): RAGGIUNTO" venga letta o
    riportata come successo del metodo.
    """
    crit = probe_run["criteria"]
    u = probe_run["unlearning"]
    assert crit["collateral"]["success"] is True
    assert crit["collateral"]["relative_change_max"] == pytest.approx(0.0, abs=1e-12)
    assert not crit["forgetting_primary"]["success_one_sided"]
    assert u["steps"] < 100


def test_primo_passo_senza_e_con_proiezione_corretta(trained):
    """
    A4, esito misurato: il segno di `first_order_forget_delta` dipende dalla
    formula di proiezione, non dalla natura del problema. Con la formula
    prescritta il primo passo e' NEGATIVO (degrada la loss di forget); con la
    proiezione ortogonale esatta e' POSITIVO in tutti i passi.
    """
    segni = {}
    for mode in ("prescribed", "metric_orthogonal"):
        V.restore_trainable(trained["trainer"], trained["theta0"])
        config = UnlearningConfig(projection_mode=mode, max_steps=5,
                                  halving_window=0, max_restarts=0,
                                  retain_loss_tolerance=1e9,
                                  probe_tolerance=None)
        unlearner = SelectiveUnlearner(trained["trainer"], config=config,
                                       log=lambda *a, **k: None)
        result = unlearner.forget(trained["splits"]["forget"],
                                  trained["splits"]["retain"])
        segni[mode] = [row["first_order_forget_delta"] for row in result.history]
    V.restore_trainable(trained["trainer"], trained["theta0"])

    assert all(v < 0 for v in segni["prescribed"]), segni["prescribed"]
    assert all(v > 0 for v in segni["metric_orthogonal"]), segni["metric_orthogonal"]


def test_sweep_esteso_raggiunge_il_forgetting_solo_al_costo_del_collaterale(trained):
    """
    A3, esito misurato: aumentando max_steps ed eta il forgetting VIENE
    raggiunto (criterio 1 soddisfatto anche in forma bilaterale) mentre la
    perdita sulla loss di retain resta entro il 5%.

    Il dato che conta e' il collaterale: arriva a ordini di grandezza. E' la
    dimostrazione numerica che il criterio di STOP sulla loss media di retain e'
    quasi cieco al danno per frase, perche' media su 24 frasi e su un modello
    saturo.
    """
    V.restore_trainable(trained["trainer"], trained["theta0"])
    out = V.evaluate_unlearning(
        trained,
        unlearning_config=UnlearningConfig(projection_mode="metric_orthogonal",
                                           max_steps=300, probe_tolerance=None),
        learning_rate=3e-4, log=lambda *a, **k: None, use_probe=False)
    V.restore_trainable(trained["trainer"], trained["theta0"])
    crit = out["criteria"]
    u = out["unlearning"]
    assert crit["forgetting_primary"]["success_two_sided"], crit["forgetting_primary"]
    assert abs(u["retain_loss_relative_change"]) <= 0.05 + 1e-9
    assert crit["collateral"]["relative_change_max"] > 10.0
    assert crit["collateral"]["n_violations"] == crit["collateral"]["n"]
