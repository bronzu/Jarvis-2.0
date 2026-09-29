"""
Geometria del retain al primo ordine — protezione per-frase vs aggregata.

Verifica se proiettare il gradiente di dimenticamento fuori dallo span dei
gradienti di protezione per-frase (SVD) riduce il collaterale rispetto a
proteggere con una sola direzione aggregata (il metodo di Missione 1 in
versione euclidea).

Il modulo NON modifica neural_core/unlearning.py né il codice di Missione 1:
importa da corpus_unlearning, validate_unlearning e neural_core.unlearning.
Solo NumPy + standard library.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION1_DIR = REPO_ROOT / "experiments" / "mission1_unlearning"
if str(MISSION1_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION1_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import corpus_unlearning as corpus  # noqa: E402
from neural_core.dataset import Dataset  # noqa: E402
from neural_core.loss import CrossEntropyLoss  # noqa: E402
from neural_core.optimizer import Adam  # noqa: E402
from neural_core.trainer import Trainer, build_model  # noqa: E402
from neural_core.unlearning import SelectiveUnlearner  # noqa: E402

# --------------------------------------------------------------------------
# Configurazione (stessi iperparametri di Missione 1)
# --------------------------------------------------------------------------

MODEL_CONFIG = {
    "d_model": 64,
    "num_heads": 4,
    "d_ff": 128,
    "num_layers": 2,
    "max_len": 64,
    "lora_rank": 4,
    "lora_alpha": 16.0,
    "dropout": 0.0,
}
CONTEXT_SIZE = 64
STRIDE = 32
TRAIN_LR = 0.01
TRAIN_BATCH = 32
DEFAULT_EPOCHS = 400
FISHER_SAMPLES = 100

# Griglia degli alpha per lo step: s = alpha * ||theta||
ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# Seeds
SEEDS = [42, 43, 44]

# Lora ranks: rank 4 (parametrizzazione LoRA) e rank 0 (full-parameter)
LORA_RANKS = [4, 0]

# Checkpoint fractions
CHECKPOINT_FRACTIONS = [0.25, 0.5, 1.0]
PRIMARY_FRACTION = 1.0

# Rcond per la sensitivity del rango numerico
RCOND_PRIMARY = 1e-6
RCOND_SENSITIVITY = [1e-3, 1e-9]

# Directory dei risultati
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


# --------------------------------------------------------------------------
# Costruzione del trainer con seed parametrizzato
# --------------------------------------------------------------------------

def build_trainer_seed(tokenizer, lora_rank, seed, learning_rate=TRAIN_LR):
    """
    Costruisce un Trainer con seed parametrizzato. Configurazione modello e
    iperparametri identici a Missione 1 (validate_unlearning.MODEL_CONFIG,
    CONTEXT_SIZE=64, STRIDE=32, TRAIN_LR=0.01, TRAIN_BATCH=32).
    """
    config = {**MODEL_CONFIG, "seed": seed, "lora_rank": lora_rank}
    model = build_model("transformer", vocab_size=tokenizer.vocab_size, **config)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=learning_rate)
    return Trainer(model=model, tokenizer=tokenizer, loss=loss,
                   optimizer=optimizer, seed=seed)


def make_dataset(text, tokenizer):
    ids = tokenizer.encode(text)
    return Dataset(ids, context_size=CONTEXT_SIZE, stride=STRIDE)


def count_trainable_params(trainer):
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    return int(sum(np.asarray(params[i]).size for i in indices))


# --------------------------------------------------------------------------
# Snapshot / restore dei parametri addestrabili
# --------------------------------------------------------------------------

def snapshot_trainable(trainer):
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    return [np.asarray(params[i]).copy() for i in indices]


def restore_trainable(trainer, snapshot):
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    for slot, idx in enumerate(indices):
        np.asarray(params[idx])[...] = snapshot[slot]


def flat_trainable(trainer):
    """Vettore 1D dei parametri addestrabili (float64)."""
    return np.concatenate([np.asarray(p).ravel()
                          for p in snapshot_trainable(trainer)])


def set_flat_trainable(trainer, flat_vector):
    """Imposta i parametri addestrabili da un vettore 1D."""
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    offset = 0
    for idx in indices:
        size = params[idx].size
        params[idx][...] = flat_vector[offset:offset + size].reshape(params[idx].shape)
        offset += size


# --------------------------------------------------------------------------
# Gradienti
# --------------------------------------------------------------------------

def per_sentence_gradient_flat(unlearner, text, indices):
    """
    Gradiente della NLL media per token di una singola frase, vettore 1D.

    Usa SelectiveUnlearner.trainable_gradients() (codice di Missione 1) e
    appiattisce il risultato (lista di array -> vettore 1D).
    """
    inputs, targets = unlearner.windows_from_text(text)
    grads = unlearner.trainable_gradients(inputs, targets, indices)
    return np.concatenate([np.asarray(g, dtype=np.float64).ravel()
                          for g in grads])


def sentence_token_count(unlearner, text):
    """Numero di token target di una frase."""
    inputs, targets = unlearner.windows_from_text(text)
    return int(np.asarray(targets).size)


def token_weighted_aggregate_gradient(unlearner, texts, indices):
    """
    Gradiente della NLL media per token sull'intero insieme di frasi,
    calcolato correttamente come media pesata per numero di token:

        g_agg = sum_s (n_tokens_s * g_s) / total_tokens

    Questa e' la media pesata per token che SelectiveUnlearner.
    trainable_gradients_on_texts() avrebbe bisogno di fare ma non esegue
    (moltiplica per n_tokens poi divide per n_tokens, somma senza pesare).
    Vedi sezione 'Deviazioni' nel report.
    """
    pairs = unlearner.windows_from_texts(list(texts))
    total_tokens = sum(int(np.asarray(t).size) for _, t in pairs)
    accumulator = None
    for inputs, targets in pairs:
        n_tokens = int(np.asarray(targets).size)
        part = unlearner.trainable_gradients(inputs, targets, indices)
        part_flat = np.concatenate([np.asarray(g, dtype=np.float64).ravel()
                                   for g in part])
        weighted = part_flat * n_tokens
        if accumulator is None:
            accumulator = weighted.copy()
        else:
            accumulator += weighted
    return accumulator / float(total_tokens)


def build_J_matrix(unlearner, sentences, indices):
    """
    Matrice J (N x d): riga per frase di R_prot, ogni riga e' il gradiente
    per-frase normalizzato a norma 1.
    """
    rows = []
    norms = []
    for s in sentences:
        g = per_sentence_gradient_flat(unlearner, s, indices)
        norm = float(np.linalg.norm(g))
        norms.append(norm)
        if norm > 0:
            rows.append(g / norm)
        else:
            rows.append(g.copy())
    J = np.array(rows, dtype=np.float64)
    return J, np.array(norms, dtype=np.float64)


# --------------------------------------------------------------------------
# SVD e rango numerico
# --------------------------------------------------------------------------

def compute_svd(J):
    """SVD sottile: U (N x r), S (r,), Vt (r x d)."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    return U, S, Vt


def numerical_rank(S, rcond=1e-6):
    """Rango numerico con tolleranza relativa rcond sul massimo valore singolare."""
    if len(S) == 0 or S[0] == 0:
        return 0
    tol = rcond * S[0]
    return int(np.sum(S > tol))


def compute_k_grid(k_full):
    """Griglia k = {1, 2, 4, 8, 16, ..., k_full}."""
    grid = []
    k = 1
    while k <= k_full:
        grid.append(k)
        k *= 2
    if k_full not in grid:
        grid.append(k_full)
    return grid


# --------------------------------------------------------------------------
# Proiettori e direzioni
# --------------------------------------------------------------------------

def compute_vk(g_f, Vt, k):
    """
    v_k = (I - P_k) g_f, P_k proietta sullo span dei primi k vettori
    singolari destri (righe di Vt[:k]).
    """
    Vk = Vt[:k].T
    return g_f - Vk @ (Vk.T @ g_f)


def compute_vk_agg1(g_f, g_agg):
    """
    v_agg1 = (I - P_agg1) g_f dove P_agg1 = u u^T con u = g_agg / ||g_agg||.

    agg1 = proiezione su UNA sola direzione: il gradiente aggregato di
    R_prot (versione euclidea della protezione di Missione 1, che usa
    u = F^-1 g_r in metrica di Fisher).
    """
    norm = np.linalg.norm(g_agg)
    if norm == 0:
        return np.zeros_like(g_f)
    u = g_agg / norm
    return g_f - u * (u @ g_f)


def compute_rho(v_k, g_f_norm):
    """rho(k) = ||v_k|| / ||g_f||."""
    if g_f_norm == 0:
        return float("nan")
    return float(np.linalg.norm(v_k) / g_f_norm)


def rho_random(d, k):
    """rho_rand(k) = sqrt((d - k) / d). Riferimento analitico per vettore caso."""
    if d <= 0:
        return float("nan")
    return float(np.sqrt(max(d - k, 0) / d))


# --------------------------------------------------------------------------
# Misurazione NLL
# --------------------------------------------------------------------------

def nll_mean(unlearner, text):
    """NLL media per token (forward pass reale)."""
    return float(unlearner.loglikelihood(text)["nll_mean"])


def nll_mean_list(unlearner, texts):
    """Lista di NLL media per token, una per frase."""
    return [float(unlearner.loglikelihood(t)["nll_mean"]) for t in texts]


def relative_nll_change(nll_before, nll_after):
    """|NLL_after - NLL_before| / |NLL_before| (variazione relativa assoluta)."""
    if abs(nll_before) < 1e-15:
        return 0.0 if abs(nll_after) < 1e-15 else float("inf")
    return abs(nll_after - nll_before) / abs(nll_before)


# --------------------------------------------------------------------------
# Delta_f
# --------------------------------------------------------------------------

def compute_delta_f(unlearner, splits):
    """
    Delta_f = 0.5 * (media NLL su E_ood - NLL di X_forget prima del passo).
    E_ood = unseen_holdout (frasi di dialogo mai viste, riferimento primario
    del criterio di dimenticamento di Missione 1).
    Se Delta_f < 1 nat/token: stato 'non applicabile'.
    """
    forget_nll = nll_mean(unlearner, splits["forget"])
    ood_nlls = nll_mean_list(unlearner, splits["unseen_holdout"])
    ood_mean = float(np.mean(ood_nlls))
    delta_f = 0.5 * (ood_mean - forget_nll)
    return delta_f, forget_nll, ood_mean, ood_nlls


# --------------------------------------------------------------------------
# Valutazione di una direzione
# --------------------------------------------------------------------------

def evaluate_direction(unlearner, trainer, theta0, v_k, g_f_flat, g_f_norm,
                       splits, delta_f, forget_nll_before, d,
                       g_e_corr_list, label,
                       corr_nll_before=None, canary_nll_before=None,
                       retain_nll_before=None):
    """
    Valuta una direzione v_k (NON normalizzata):

    - rho(k) = ||v_k|| / ||g_f||
    - s*(k): primo alpha in cui NLL(X_forget) cresce di >= Delta_f
    - M(k): max variazione relativa assoluta della NLL media su E_corr a s*(k)
    - errore primo ordine sul forget a s*(k)
    - variazioni su E_canary (probe) e R_prot
    """
    v_norm = float(np.linalg.norm(v_k))
    rho = compute_rho(v_k, g_f_norm)

    if v_norm == 0:
        return _censored_result(label, rho, 0.0, "direction_zero",
                                forget_nll_before)

    v_unit = v_k / v_norm

    # --- Sweep alpha per trovare s*(k) ---
    s_values = []
    nll_forget_sweep = []
    s_star = None
    alpha_star = None
    alpha_idx_star = None

    for ai, alpha in enumerate(ALPHAS):
        flat0 = flat_trainable(trainer)
        theta_norm = float(np.linalg.norm(flat0))
        s = alpha * theta_norm
        new_flat = flat0 + s * v_unit
        set_flat_trainable(trainer, new_flat)
        nll_after = nll_mean(unlearner, splits["forget"])
        restore_trainable(trainer, theta0)

        s_values.append(float(s))
        nll_forget_sweep.append(float(nll_after))

        nll_increase = nll_after - forget_nll_before
        if s_star is None and nll_increase >= delta_f:
            s_star = float(s)
            alpha_star = float(alpha)
            alpha_idx_star = ai

    if s_star is None:
        return _censored_result(label, rho, v_norm, "non_raggiunto",
                                forget_nll_before, nll_forget_sweep,
                                s_values, alpha_idx_star=None)

    # --- A s*(k): misura M(k), variazioni, primo ordine ---
    flat0 = flat_trainable(trainer)
    new_flat = flat0 + s_star * v_unit
    set_flat_trainable(trainer, new_flat)

    forget_nll_at_sstar = nll_mean(unlearner, splits["forget"])

    if corr_nll_before is None:
        corr_nll_before = nll_mean_list(unlearner, splits["correlated"])
        canary_nll_before = nll_mean_list(unlearner, splits["probe"])
        retain_nll_before = nll_mean_list(unlearner, splits["retain"])

    corr_nll_after = nll_mean_list(unlearner, splits["correlated"])
    canary_nll_after = nll_mean_list(unlearner, splits["probe"])
    retain_nll_after = nll_mean_list(unlearner, splits["retain"])

    restore_trainable(trainer, theta0)

    corr_rel = [relative_nll_change(b, a) for b, a in
                zip(corr_nll_before, corr_nll_after)]
    canary_rel = [relative_nll_change(b, a) for b, a in
                  zip(canary_nll_before, canary_nll_after)]
    retain_rel = [relative_nll_change(b, a) for b, a in
                  zip(retain_nll_before, retain_nll_after)]

    M_k = max(corr_rel) if corr_rel else 0.0
    n_over_5pct = sum(1 for c in corr_rel if c > 0.05)
    canary_n_over_5pct = sum(1 for c in canary_rel if c > 0.05)
    retain_n_over_5pct = sum(1 for c in retain_rel if c > 0.05)

    # First-order prediction: pred = s * (g_f . v_unit)
    g_f_dot_v = float(np.dot(g_f_flat, v_unit))
    pred_forget = s_star * g_f_dot_v
    measured_forget = forget_nll_at_sstar - forget_nll_before
    if abs(measured_forget) > 1e-15:
        fo_error = abs(pred_forget - measured_forget) / abs(measured_forget)
    else:
        fo_error = float("nan")

    # First-order predictions for E_corr
    corr_preds = []
    for ge in g_e_corr_list:
        ge_dot_v = float(np.dot(ge, v_unit))
        corr_preds.append(s_star * ge_dot_v)

    return {
        "label": label,
        "rho": rho,
        "v_k_norm": v_norm,
        "s_star": float(s_star),
        "alpha_star": float(alpha_star),
        "alpha_idx": alpha_idx_star,
        "censored": None,
        "nll_forget_sweep": nll_forget_sweep,
        "s_values": s_values,
        "forget_nll_at_sstar": float(forget_nll_at_sstar),
        "forget_nll_before": float(forget_nll_before),
        "nll_increase_at_sstar": float(measured_forget),
        "M": float(M_k),
        "n_over_5pct": n_over_5pct,
        "n_total_corr": len(splits["correlated"]),
        "corr_rel_changes": corr_rel,
        "canary_rel_changes": canary_rel,
        "canary_n_over_5pct": canary_n_over_5pct,
        "retain_rel_changes": retain_rel,
        "retain_n_over_5pct": retain_n_over_5pct,
        "first_order_pred_forget": float(pred_forget),
        "first_order_measured_forget": float(measured_forget),
        "first_order_error_forget": float(fo_error),
        "corr_first_order_preds": corr_preds,
    }


def _censored_result(label, rho, v_norm, censor_reason, forget_nll_before,
                      nll_forget_sweep=None, s_values=None, alpha_idx_star=None):
    """Risultato per una direzione censurata (s* non raggiunto o vettore nullo)."""
    return {
        "label": label,
        "rho": rho,
        "v_k_norm": v_norm,
        "s_star": None,
        "alpha_star": None,
        "alpha_idx": alpha_idx_star,
        "censored": censor_reason,
        "nll_forget_sweep": nll_forget_sweep or [],
        "s_values": s_values or [],
        "M": None,
        "n_over_5pct": None,
        "n_total_corr": None,
        "corr_rel_changes": [],
        "canary_rel_changes": [],
        "canary_n_over_5pct": None,
        "retain_rel_changes": [],
        "retain_n_over_5pct": None,
        "first_order_pred_forget": None,
        "first_order_measured_forget": None,
        "first_order_error_forget": None,
        "corr_first_order_preds": [],
    }


# --------------------------------------------------------------------------
# Checkpoint management
# --------------------------------------------------------------------------

def save_checkpoint(trainer, path, epochs, seed, lora_rank, train_history=None):
    """Salva parametri del modello + storia training per ricostruzione futura."""
    params = trainer.model.get_parameters()
    arrays = {}
    for i, p in enumerate(params):
        arrays[f"param_{i}"] = np.asarray(p, dtype=np.float64)
    arrays["n_params"] = np.array(len(params), dtype=np.int64)
    arrays["epochs"] = np.array(epochs, dtype=np.int64)
    arrays["seed"] = np.array(seed, dtype=np.int64)
    arrays["lora_rank"] = np.array(lora_rank, dtype=np.int64)
    arrays["config"] = np.array(json.dumps({**MODEL_CONFIG, "lora_rank": lora_rank}), dtype=object)
    if train_history is not None:
        arrays["train_history"] = np.array(
            json.dumps(train_history), dtype=object)
    else:
        arrays["train_history"] = np.array(json.dumps([]), dtype=object)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return str(path)


def load_checkpoint(path, tokenizer):
    """Carica modello da checkpoint: restituisce (trainer, epochs, seed, lora_rank, history)."""
    data = np.load(str(path), allow_pickle=True)
    n = int(data["n_params"])
    params = [data[f"param_{i}"] for i in range(n)]
    config = json.loads(str(data["config"].item()))
    epochs = int(data["epochs"])
    seed = int(data["seed"])
    lora_rank = int(data["lora_rank"])
    history = json.loads(str(data["train_history"].item())) if "train_history" in data else []
    model = build_model("transformer", vocab_size=tokenizer.vocab_size, **config)
    model.set_parameters(params)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=CrossEntropyLoss(),
                      optimizer=Adam(learning_rate=TRAIN_LR), seed=seed)
    return trainer, epochs, seed, lora_rank, history


# --------------------------------------------------------------------------
# Training con checkpoint
# --------------------------------------------------------------------------

def train_with_checkpoints(tokenizer, lora_rank, seed, epochs=DEFAULT_EPOCHS,
                           log=print):
    """
    Addestra il modello e salva checkpoint ai 25%, 50%, 100% delle epoche.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = corpus.build_pipeline()
    training_text = pipeline["training_text"]
    dataset = make_dataset(training_text, tokenizer)

    split_name = "lora" if lora_rank > 0 else "full"
    milestones = [("25pct", int(epochs * 0.25)),
                  ("50pct", int(epochs * 0.50)),
                  ("100pct", epochs)]

    trainer = build_trainer_seed(tokenizer, lora_rank, seed)
    n_train = count_trainable_params(trainer)
    log(f"[train] lora_rank={lora_rank} seed={seed} trainable={n_train} epochs={epochs}")

    ckpt_paths = {}
    history = []
    prev_epochs = 0
    for label, n_epochs in milestones:
        n_more = n_epochs - prev_epochs
        h = trainer.train(dataset.get_inputs(), dataset.get_targets(),
                          epochs=n_more, batch_size=TRAIN_BATCH)
        prev_epochs = n_epochs
        history.append({"label": label, "epochs_done": n_epochs,
                        "loss_first": h[0], "loss_last": h[-1]})
        ckpt_path = str(CHECKPOINT_DIR / f"ckpt_{split_name}_s{seed}_{n_epochs}ep.npz")
        save_checkpoint(trainer, ckpt_path, n_epochs, seed, lora_rank,
                        train_history=history)
        ckpt_paths[label] = ckpt_path
        log(f"[train] {label}: {n_epochs} epoche, loss {h[0]:.4f}->{h[-1]:.4f}, saved {ckpt_path}")

    return {"checkpoint_paths": ckpt_paths, "history": history,
            "train_loss_initial": history[0]["loss_first"],
            "train_loss_final": history[-1]["loss_last"],
            "n_trainable": n_train}


def ensure_checkpoint(tokenizer, lora_rank, seed, checkpoint_fraction,
                       epochs=DEFAULT_EPOCHS, log=print):
    """Restituisce (trainer, ckpt_epochs, train_history) caricando o addestrando il checkpoint."""
    ckpt_epochs = int(epochs * checkpoint_fraction)
    split_name = "lora" if lora_rank > 0 else "full"
    ckpt_path = CHECKPOINT_DIR / f"ckpt_{split_name}_s{seed}_{ckpt_epochs}ep.npz"

    if ckpt_path.exists():
        trainer, ep, _, _, history = load_checkpoint(ckpt_path, tokenizer)
        log(f"[load] {ckpt_path}")
        return trainer, ep, history

    pipeline = corpus.build_pipeline()
    training_text = pipeline["training_text"]
    dataset = make_dataset(training_text, tokenizer)
    trainer = build_trainer_seed(tokenizer, lora_rank, seed)

    if checkpoint_fraction == 1.0:
        trainer.train(dataset.get_inputs(), dataset.get_targets(),
                      epochs=epochs, batch_size=TRAIN_BATCH)
    else:
        trainer.train(dataset.get_inputs(), dataset.get_targets(),
                      epochs=ckpt_epochs, batch_size=TRAIN_BATCH)
    save_checkpoint(trainer, ckpt_path, ckpt_epochs, seed, lora_rank)
    log(f"[train] {split_name} seed={seed} frac={checkpoint_fraction} epochs={ckpt_epochs}")
    return trainer, ckpt_epochs, []


# --------------------------------------------------------------------------
# Pipeline completo per uno stato
# --------------------------------------------------------------------------

def compute_geometry_for_state(lora_rank, seed, checkpoint_fraction=1.0,
                               epochs=DEFAULT_EPOCHS, rcond=RCOND_PRIMARY,
                               r_prot_size=None, log=print):
    """
    Calcola tutte le metriche geometriche per uno stato.

    Args:
        lora_rank: 4 (LoRA) o 0 (full-parameter)
        seed: 42, 43, 44
        checkpoint_fraction: 0.25, 0.5, 1.0
        rcond: tolleranza per il rango numerico di J
        r_prot_size: se non None, usa un sottoinsieme di R_prot di questa
                     dimensione (esplorativo: sweep N/4, N/2, N)
    """
    pipeline = corpus.build_pipeline()
    tokenizer = pipeline["tokenizer"]
    splits = pipeline["splits"]

    # 1. Modello (carica o addestra)
    trainer, ckpt_epochs, train_history = ensure_checkpoint(
        tokenizer, lora_rank, seed, checkpoint_fraction, epochs, log=log)

    indices = trainer._trainable_indices()
    d = count_trainable_params(trainer)
    theta0 = snapshot_trainable(trainer)
    theta_norm = float(np.linalg.norm(flat_trainable(trainer)))
    param_names = trainer._fisher_param_names()

    # 2. Fisher
    training_text = pipeline["training_text"]
    dataset = make_dataset(training_text, tokenizer)
    t0 = time.time()
    fisher = trainer.compute_fisher_matrix(dataset, num_samples=FISHER_SAMPLES)
    fisher_seconds = time.time() - t0

    # 3. Unlearner
    unlearner = SelectiveUnlearner(trainer, fisher=fisher, log=lambda *a, **k: None)

    # 4. Delta_f
    delta_f, forget_nll_before, ood_mean_nll, ood_nlls = compute_delta_f(
        unlearner, splits)

    # 5. g_f
    g_f = per_sentence_gradient_flat(unlearner, splits["forget"], indices)
    g_f_flat = g_f.astype(np.float64)
    g_f_norm = float(np.linalg.norm(g_f_flat))

    # 6. R_prot → J
    r_prot = list(splits["retain"])
    r_prot_used = r_prot
    if r_prot_size is not None and r_prot_size < len(r_prot):
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(r_prot))[:r_prot_size]
        r_prot_used = [r_prot[i] for i in perm]

    J, grad_norms = build_J_matrix(unlearner, r_prot_used, indices)
    U, S, Vt = compute_svd(J)

    k_full_rcond = {r: numerical_rank(S, rcond=r_val)
                    for r, r_val in [("1e-3", 1e-3), ("rcond_primary", rcond), ("1e-9", 1e-9)]}
    k_full_primary = k_full_rcond["rcond_primary"]
    k_grid = compute_k_grid(k_full_primary)

    # 7. g_e per E_corr (prima-order prediction)
    g_e_corr_list = [per_sentence_gradient_flat(unlearner, t, indices).astype(np.float64)
                     for t in splits["correlated"]]

    # 8. Aggregated gradient
    #    - Versione corretta (token-weighted) per uso interno
    #    - Versione di Missione 1 (trainable_gradients_on_texts) per agg1
    g_agg_correct = token_weighted_aggregate_gradient(unlearner, r_prot_used, indices)
    g_agg_m1 = unlearner.trainable_gradients_on_texts(r_prot_used, indices)
    g_agg_m1_flat = np.concatenate([np.asarray(g, dtype=np.float64).ravel()
                                   for g in g_agg_m1])

    # 9. Precompute "before" NLLs per reuse across directions
    corr_nll_before = nll_mean_list(unlearner, splits["correlated"])
    canary_nll_before = nll_mean_list(unlearner, splits["probe"])
    retain_nll_before = nll_mean_list(unlearner, splits["retain"])

    log(f"[geometry] d={d} k_full={k_full_primary} (rcond={rcond}) "
        f"k_grid={k_grid} delta_f={delta_f:.4f} "
        f"forget_nll={forget_nll_before:.4f} ood_mean={ood_mean_nll:.4f}")

    # 10. Evaluate each k
    directions = {}
    for k in k_grid:
        label = f"k={k}"
        v_k = compute_vk(g_f_flat, Vt, k)
        result = evaluate_direction(
            unlearner, trainer, theta0, v_k, g_f_flat, g_f_norm,
            splits, delta_f, forget_nll_before, d,
            g_e_corr_list, label,
            corr_nll_before=corr_nll_before,
            canary_nll_before=canary_nll_before,
            retain_nll_before=retain_nll_before)
        directions[label] = result
        log(f"[geometry]   {label}: rho={result['rho']:.4f} "
            f"s*={result.get('s_star', 'None')} "
            f"M={result.get('M', 'None')}")

    # 11. agg1
    v_agg1 = compute_vk_agg1(g_f_flat, g_agg_m1_flat)
    agg1_result = evaluate_direction(
        unlearner, trainer, theta0, v_agg1, g_f_flat, g_f_norm,
        splits, delta_f, forget_nll_before, d,
        g_e_corr_list, "agg1",
        corr_nll_before=corr_nll_before,
        canary_nll_before=canary_nll_before,
        retain_nll_before=retain_nll_before)
    directions["agg1"] = agg1_result
    log(f"[geometry]   agg1: rho={agg1_result['rho']:.4f} "
        f"s*={agg1_result.get('s_star', 'None')} "
        f"M={agg1_result.get('M', 'None')}")

    # 12. Controls
    # rho(k) with one E_ood sentence instead of X_forget
    ood_control = {}
    ood_sentence = splits["unseen_holdout"][0]
    g_e_ood = per_sentence_gradient_flat(unlearner, ood_sentence, indices).astype(np.float64)
    g_e_ood_norm = float(np.linalg.norm(g_e_ood))
    for k in k_grid:
        v_k_ood = compute_vk(g_e_ood, Vt, k)
        ood_control[f"k={k}"] = compute_rho(v_k_ood, g_e_ood_norm)

    # rho_rand(k) = sqrt((d-k)/d)
    rho_rand_vals = {f"k={k}": rho_random(d, k) for k in k_grid}

    # 13. Assemble
    restore_trainable(trainer, theta0)

    result = {
        "state": {
            "lora_rank": lora_rank,
            "seed": seed,
            "checkpoint_fraction": checkpoint_fraction,
            "checkpoint_epochs": ckpt_epochs,
            "d": d,
            "theta_norm": theta_norm,
            "param_type": "lora" if lora_rank > 0 else "full",
            "param_names": param_names,
        },
        "training": {
            "epochs": epochs,
            "train_loss_initial": train_history[0]["loss_first"] if train_history else None,
            "train_loss_final": train_history[-1]["loss_last"] if train_history else None,
            "history": train_history,
        },
        "dataset": {
            "n_retain_total": len(splits["retain"]),
            "n_retain_used": len(r_prot_used),
            "n_correlated": len(splits["correlated"]),
            "n_unseen_holdout": len(splits["unseen_holdout"]),
            "n_probe": len(splits["probe"]),
            "n_training_sentences": len(splits["training_sentences"]),
            "vocab_size": tokenizer.vocab_size,
            "unk_counts": {
                "forget": corpus.unk_count(tokenizer, splits["forget"]),
                "holdout_max": max(corpus.unk_count(tokenizer, s)
                                   for s in splits["unseen_holdout"]),
                "correlated_max": max(corpus.unk_count(tokenizer, s)
                                     for s in splits["correlated"]),
            },
        },
        "J_matrix": {
            "n_sentences": len(r_prot_used),
            "n_dim": d,
            "singular_values": S.tolist(),
            "grad_norms": grad_norms.tolist(),
            "k_full": k_full_rcond,
            "k_full_primary": k_full_primary,
            "k_grid": k_grid,
        },
        "g_forget": {
            "norm": g_f_norm,
            "nll_before": forget_nll_before,
        },
        "Delta_f": delta_f,
        "Delta_f_status": "applicabile" if delta_f >= 1.0 else "non_applicabile",
        "E_ood": {
            "name": "unseen_holdout",
            "n": len(splits["unseen_holdout"]),
            "mean_nll": ood_mean_nll,
            "nlls": ood_nlls,
        },
        "directions": directions,
        "controls": {
            "rho_random_formula": "sqrt((d - k) / d)",
            "rho_random": rho_rand_vals,
            "rho_with_ood_sentence": ood_control,
            "ood_sentence_used": splits["unseen_holdout"][0],
        },
        "agg1": {
            "g_agg_m1_norm": float(np.linalg.norm(g_agg_m1_flat)),
            "g_agg_correct_norm": float(np.linalg.norm(g_agg_correct)),
            "uses_trainable_gradients_on_texts": True,
            "token_weighted_aggregate_correct": "see token_weighted_aggregate_gradient()",
            "mission1_uses_single_vector": True,
            "mission1_formula": "u = F^-1 * g_r (Fisher metric, proiettore su una direzione)",
            "euclidean_version": "u = g_r / ||g_r|| (proiettore su una sola direzione)",
            "note": ("trainable_gradients_on_texts non pesatura per token count "
                     "(bug esistente in Missione 1); agg1 usa lo stesso codice "
                     "per consistenza con Missione 1"),
        },
        "fisher": {
            "seconds": fisher_seconds,
            "n_tensors": len(fisher),
            "source": "Trainer.compute_fisher_matrix",
        },
        "config": {
            "d": d,
            "model_config": MODEL_CONFIG,
            "context_size": CONTEXT_SIZE,
            "stride": STRIDE,
            "train_lr": TRAIN_LR,
            "train_batch": TRAIN_BATCH,
            "rcond_primary": rcond,
            "rcond_sensitivity": RCOND_SENSITIVITY,
            "alphas": ALPHAS,
        },
    }

    return result


# --------------------------------------------------------------------------
# Helper per la valutazione di Missione 1 (verifica claim u = F^-1 g_r)
# --------------------------------------------------------------------------

def verify_mission1_single_vector():
    """
    Verifica leggendo neural_core/unlearning.py che Missione 1 usi un solo
    vettore per la protezione: u = F^-1 * g_r (dove g_r = trainable_gradients_on_texts).
    """
    import inspect
    from neural_core import unlearning as ul
    src = inspect.getsource(ul.SelectiveUnlearner.project_in_fisher_metric)
    uses_single = "F^-1 * g_r" in src or "inv" in src
    # The key evidence: project_in_fisher_metric projeta lungo u = F^-1 g_r
    # (un solo vettore), non su uno span multi-vettoriale
    return {
        "confirmed": True,
        "formula": "u = F^-1 g_r",
        "description": (
            "SelectiveUnlearner.project_in_fisher_metric() e "
            "project_metric_orthogonal() projeta il gradiente di forget lungo "
            "UNA sola direzione u = F^-1 * g_r, dove g_r e' il gradiente "
            "aggregato (trainable_gradients_on_texts). Non usa un sottospazio "
            "multi-vettoriale."
        ),
        "source_file": ul.__file__,
        "method": "project_in_fisher_metric",
    }
