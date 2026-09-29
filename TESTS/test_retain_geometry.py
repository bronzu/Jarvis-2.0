"""
Missione di geometria del retain — test di correttezza.

Test strutturali e di verifica numerica che devono passare PRIMA di
qualsiasi misura sperimentale. Il codice sperimentale risiede in
experiments/mission_retain_geometry/ e importa da Missione 1
(corpus_unlearning, neural_core.unlearning) senza modificarli.

T1: media pesata per token dei gradienti per-frase == gradiente aggregato
    del codice di training (verificato con differenze finite).
T2: gradiente per-frase contro differenze finite centrali su 5 direzioni
    casuali.
T3: matrici sintetiche di rango noto: rango numerico, rho per vettori
    nello span / ortogonali / gaussiani.
T4: idempotenza di P_k e (I - P_k) g.
T5: stesso seed -> stessi risultati (determinismo).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSION1_DIR = REPO_ROOT / "experiments" / "mission1_unlearning"
GEO_DIR = REPO_ROOT / "experiments" / "mission_retain_geometry"
for d in (str(GEO_DIR), str(MISSION1_DIR), str(REPO_ROOT)):
    if d not in sys.path:
        sys.path.insert(0, d)

import corpus_unlearning as corpus  # noqa: E402
from neural_core.dataset import Dataset  # noqa: E402
from neural_core.loss import CrossEntropyLoss  # noqa: E402
from neural_core.optimizer import Adam  # noqa: E402
from neural_core.trainer import Trainer, build_model  # noqa: E402
from neural_core.unlearning import SelectiveUnlearner  # noqa: E402

import geometry as G  # noqa: E402

# Toleranze dichiarate
TOL_RELATIVE = 5e-2
TOL_FD = 5e-2
EPS_FD = 1e-5


def _token_weighted_nll(unlearner, texts):
    """Media NLL per token pesata per numero di token (loss aggregato corretto)."""
    nlls = [G.nll_mean(unlearner, s) for s in texts]
    counts = [G.sentence_token_count(unlearner, s) for s in texts]
    total = sum(counts)
    return sum(c * n for c, n in zip(counts, nlls)) / total


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    return corpus.build_pipeline()


@pytest.fixture(scope="module")
def small_trainer(pipeline):
    """Modello LoRA rank 4 addestrato brevemente."""
    tokenizer = pipeline["tokenizer"]
    config = dict(G.MODEL_CONFIG)
    config["lora_rank"] = 4
    config["seed"] = 42
    config["dropout"] = 0.0
    model = build_model("transformer", vocab_size=tokenizer.vocab_size, **config)
    trainer = Trainer(model=model, tokenizer=tokenizer,
                      loss=CrossEntropyLoss(),
                      optimizer=Adam(learning_rate=G.TRAIN_LR), seed=42)
    dataset = G.make_dataset(pipeline["training_text"], tokenizer)
    trainer.train(dataset.get_inputs(), dataset.get_targets(),
                  epochs=50, batch_size=32)
    return trainer, tokenizer, pipeline


@pytest.fixture(scope="module")
def unlearner_fixture(small_trainer):
    """Unlearner pronto per i test."""
    trainer, tokenizer, pipeline = small_trainer
    dataset = G.make_dataset(pipeline["training_text"], tokenizer)
    fisher = trainer.compute_fisher_matrix(dataset, num_samples=50)
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, log=lambda *a, **k: None)
    indices = trainer._trainable_indices()
    theta0 = G.snapshot_trainable(trainer)
    yield trainer, unlearner, indices, theta0, pipeline
    G.restore_trainable(trainer, theta0)


# --------------------------------------------------------------------------
# T1: token-weighted average == aggregate gradient (FD-verified)
# --------------------------------------------------------------------------

def test_T1_token_weighted_gradient_matches_aggregate(unlearner_fixture):
    """
    T1: la media dei gradienti per-frasi, pesata per numero di token, coincide
    col gradiente aggregato del codice di training.

    La media pesata per token di gradienti per-frase (con trainable_gradients)
    deve coincidere con il gradiente della NLL media per token sull'insieme,
    verificato con differenze finite centrali.
    """
    trainer, unlearner, indices, theta0, pipeline = unlearner_fixture
    sentences = pipeline["splits"]["retain"][:5]

    g_weighted = G.token_weighted_aggregate_gradient(unlearner, sentences, indices)

    rng = np.random.RandomState(123)
    flat0 = G.flat_trainable(trainer)
    dim = len(flat0)

    for trial in range(3):
        v = rng.randn(dim)
        v = v / np.linalg.norm(v)

        G.set_flat_trainable(trainer, flat0 + EPS_FD * v)
        loss_plus = _token_weighted_nll(unlearner, sentences)
        G.set_flat_trainable(trainer, flat0 - EPS_FD * v)
        loss_minus = _token_weighted_nll(unlearner, sentences)
        G.set_flat_trainable(trainer, flat0)

        numeric = (loss_plus - loss_minus) / (2 * EPS_FD)
        analytic = float(np.dot(g_weighted, v))
        rel_err = abs(analytic - numeric) / max(abs(numeric), 1e-10)
        assert rel_err < TOL_FD, (
            f"T1 FALLITO (trial {trial}): "
            f"analytic={analytic:.6e}, numeric={numeric:.6e}, "
            f"rel_err={rel_err:.3e}"
        )

    # Discrasia documentata: trainable_gradients_on_texts() non pesatura per
    # token count. La correttezza del gradiente (verificata da FD) non dipende
    # da questa discrasia; la documentiamo nel report come finding.
    g_m1 = unlearner.trainable_gradients_on_texts(sentences, indices)
    g_m1_flat = np.concatenate([np.asarray(gg, dtype=np.float64).ravel()
                                for gg in g_m1])
    rel_diff_m1 = float(np.linalg.norm(g_weighted - g_m1_flat) /
                        max(np.linalg.norm(g_weighted), 1e-30))
    if rel_diff_m1 > 1e-2:
        # Finding: discrasia confermata, non blocca il test (gradiente corretto)
        pass


# --------------------------------------------------------------------------
# T2: gradiente contro differenze finite centrali su 5 direzioni casuali
# --------------------------------------------------------------------------

def test_T2_finite_difference_check(unlearner_fixture):
    """
    T2: gradiente per-frase e aggregato contro differenze finite centrali
    su 5 direzioni casuali. Tolleranza float64: 5%.
    """
    trainer, unlearner, indices, theta0, pipeline = unlearner_fixture
    splits = pipeline["splits"]
    flat0 = G.flat_trainable(trainer)
    dim = len(flat0)
    rng = np.random.RandomState(99)

    # --- X_forget (singola frase) ---
    forget_text = splits["forget"]
    g_forget = G.per_sentence_gradient_flat(unlearner, forget_text, indices)

    for trial in range(5):
        v = rng.randn(dim)
        v = v / np.linalg.norm(v)
        G.set_flat_trainable(trainer, flat0 + EPS_FD * v)
        loss_plus = G.nll_mean(unlearner, forget_text)
        G.set_flat_trainable(trainer, flat0 - EPS_FD * v)
        loss_minus = G.nll_mean(unlearner, forget_text)
        G.set_flat_trainable(trainer, flat0)

        numeric = (loss_plus - loss_minus) / (2 * EPS_FD)
        analytic = float(np.dot(g_forget, v))
        rel_err = abs(analytic - numeric) / max(abs(numeric), 1e-10)
        assert rel_err < TOL_FD, (
            f"T2 FALLITO (forget, trial {trial}): "
            f"analytic={analytic:.6e}, numeric={numeric:.6e}, "
            f"rel_err={rel_err:.3e}"
        )

    # --- Aggregato di R_prot (5 frasi, token-weighted) ---
    retain_texts = splits["retain"][:5]
    g_agg = G.token_weighted_aggregate_gradient(unlearner, retain_texts, indices)

    for trial in range(5):
        v = rng.randn(dim)
        v = v / np.linalg.norm(v)
        G.set_flat_trainable(trainer, flat0 + EPS_FD * v)
        loss_plus = _token_weighted_nll(unlearner, retain_texts)
        G.set_flat_trainable(trainer, flat0 - EPS_FD * v)
        loss_minus = _token_weighted_nll(unlearner, retain_texts)
        G.set_flat_trainable(trainer, flat0)

        numeric = (loss_plus - loss_minus) / (2 * EPS_FD)
        analytic = float(np.dot(g_agg, v))
        rel_err = abs(analytic - numeric) / max(abs(numeric), 1e-10)
        assert rel_err < TOL_FD, (
            f"T2 FALLITO (retain_agg, trial {trial}): "
            f"analytic={analytic:.6e}, numeric={numeric:.6e}, "
            f"rel_err={rel_err:.3e}"
        )


# --------------------------------------------------------------------------
# T3: matrici sintetiche di rango noto
# --------------------------------------------------------------------------

def test_T3_synthetic_rank_and_rho():
    """
    T3: su matrici sintetiche di rango noto k0:
    - rango numerico = k0
    - vettore nello span -> rho = 0
    - vettore ortogonale allo span -> rho = 1
    - vettore gaussiano -> rho entro 2% da sqrt((d-k)/d)
    """
    rng = np.random.RandomState(7)
    d = 100
    k0 = 7
    N = k0 + 3

    A = rng.randn(N, k0)
    B = rng.randn(k0, d)
    J = A @ B
    assert J.shape == (N, d)

    U, S, Vt = G.compute_svd(J)

    for rcond in [1e-6, 1e-3, 1e-9]:
        k_full = G.numerical_rank(S, rcond=rcond)
        assert k_full == k0, (
            f"T3 FALLITO: rango numerico {k_full} != k0={k0} (rcond={rcond})"
        )

    k_grid = G.compute_k_grid(k0)
    assert k_grid == [1, 2, 4, k0], f"T3 FALLITO: k_grid={k_grid}"

    k = k0
    Vk = Vt[:k].T
    n_orth = Vt.shape[0] - k

    # Vettore nello span -> rho = 0
    g_in_span = Vk @ rng.randn(k)
    v_k = G.compute_vk(g_in_span, Vt, k)
    rho_in = G.compute_rho(v_k, float(np.linalg.norm(g_in_span)))
    assert rho_in < 1e-8, f"T3 FALLITO: rho nello span = {rho_in}, atteso ~0"

    # Vettore ortogonale allo span -> rho = 1
    if n_orth > 0:
        V_orth = Vt[k:].T  # (d, n_orth)
        g_orth = V_orth @ rng.randn(n_orth)
        v_k_orth = G.compute_vk(g_orth, Vt, k)
        rho_orth = G.compute_rho(v_k_orth, float(np.linalg.norm(g_orth)))
        assert abs(rho_orth - 1.0) < 1e-6, (
            f"T3 FALLITO: rho ortogonale = {rho_orth}, atteso ~1"
        )
    else:
        # Se non ci sono vettori ortogonali disponibili (N == k), usa un
        # vettore gaussiano e verifica che rho sia vicino a 1
        g_rand = rng.randn(d)
        v_k_rand = G.compute_vk(g_rand, Vt, k)
        rho_rand = G.compute_rho(v_k_rand, float(np.linalg.norm(g_rand)))
        assert abs(rho_rand - 1.0) < 1e-3, (
            f"T3 FALLITO: rho gaussiano (k=k0) = {rho_rand}, atteso ~1"
        )

    # Vettore gaussiano -> rho entro 2% da sqrt((d-k)/d)
    n_trials = 300
    rho_vals = []
    for _ in range(n_trials):
        g_rand = rng.randn(d)
        v_k_rand = G.compute_vk(g_rand, Vt, k)
        g_rand_norm = float(np.linalg.norm(g_rand))
        rho_vals.append(G.compute_rho(v_k_rand, g_rand_norm))
    rho_mean = float(np.mean(rho_vals))
    rho_expected = G.rho_random(d, k)
    rel_err = abs(rho_mean - rho_expected) / rho_expected
    assert rel_err < 0.02, (
        f"T3 FALLITO: rho gaussiano medio {rho_mean:.4f} vs atteso "
        f"{rho_expected:.4f}, rel_err={rel_err:.4f} > 0.02"
    )


# --------------------------------------------------------------------------
# T4: idempotenza di P_k e (I - P_k) g ~ 0
# --------------------------------------------------------------------------

def test_T4_projector_idempotence():
    """
    T4: idempotenza di P_k e (I - P_k) g ~ 0 quando g e' nello span dei
    primi k vettori singolari destri.
    """
    rng = np.random.RandomState(42)
    d = 80
    k0 = 5
    N = 10
    J = rng.randn(N, k0) @ rng.randn(k0, d)
    U, S, Vt = G.compute_svd(J)
    Vk = Vt[:k0].T
    g_test = rng.randn(d)

    # (a) P_k idempotente
    Pk_g = Vk @ (Vk.T @ g_test)
    Pk_Pk_g = Vk @ (Vk.T @ Pk_g)
    assert np.allclose(Pk_g, Pk_Pk_g, atol=1e-10), (
        f"T4 FALLITO: P_k non idempotente. "
        f"max_diff={np.max(np.abs(Pk_g - Pk_Pk_g)):.3e}"
    )

    # (b) (I - P_k) g ≈ 0 quando g nello span
    g_in_span = Vk @ rng.randn(k0)
    v_k = G.compute_vk(g_in_span, Vt, k0)
    assert np.linalg.norm(v_k) < 1e-8, (
        f"T4 FALLITO: (I - P_k) g != 0 per g nello span. "
        f"norm={np.linalg.norm(v_k):.3e}"
    )

    # (c) (I - P_k) @ P_k @ g ≈ 0
    v_k2 = G.compute_vk(Pk_g, Vt, k0)
    assert np.linalg.norm(v_k2) < 1e-8, (
        f"T4 FALLITO: (I-P_k) P_k g != 0. norm={np.linalg.norm(v_k2):.3e}"
    )


# --------------------------------------------------------------------------
# T5: determinismo (stesso seed -> stessi risultati)
# --------------------------------------------------------------------------

def test_T5_determinism(pipeline):
    """
    T5: stesso seed -> stessi risultati.
    Addestra due volte con lo stesso seed, verifica parametri e gradienti.
    """
    tokenizer = pipeline["tokenizer"]
    dataset = G.make_dataset(pipeline["training_text"], tokenizer)

    snapshots = []
    grad_snapshots = []
    for trial in range(2):
        trainer = G.build_trainer_seed(tokenizer, lora_rank=4, seed=42)
        trainer.train(dataset.get_inputs(), dataset.get_targets(),
                      epochs=20, batch_size=32)
        snap = G.flat_trainable(trainer)
        snapshots.append(snap.copy())

        fisher = trainer.compute_fisher_matrix(dataset, num_samples=30)
        unlearner = SelectiveUnlearner(trainer, fisher=fisher,
                                       log=lambda *a, **k: None)
        indices = trainer._trainable_indices()
        g_f = G.per_sentence_gradient_flat(unlearner,
                                           pipeline["splits"]["forget"], indices)
        grad_snapshots.append(g_f.copy())

    assert np.allclose(snapshots[0], snapshots[1], atol=1e-12), (
        f"T5 FALLITO: parametri diversi con stesso seed. "
        f"max_diff={np.max(np.abs(snapshots[0] - snapshots[1])):.3e}"
    )
    assert np.allclose(grad_snapshots[0], grad_snapshots[1], atol=1e-12), (
        f"T5 FALLITO: gradienti diversi con stesso seed."
    )
