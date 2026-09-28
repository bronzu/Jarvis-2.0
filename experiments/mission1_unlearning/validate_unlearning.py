"""
Valutazione misurata del machine unlearning selettivo (Missione 1).

Il modulo si chiama `validate_unlearning` e non `validate` come
`experiments/phase4_continual_learning/validate.py`: nella stessa sessione
pytest i due script verrebbero importati come `validate` e `corpus` nello
stesso `sys.modules`, e l'uno ruberebbe il modulo all'altro. Il nome univoco
e' un requisito di funzionamento della suite, non una scelta di stile.

Esegue, in un unico passaggio e sulla stessa istanza di modello:

  0. addestramento sul corpus di `corpus.py` (che contiene X_forget);
  1. calcolo della Micro-Fisher con il codice gia' esistente del Trainer
     (`Trainer.compute_fisher_matrix`, spazio LoRA);
  2. misura PRIMA dell'unlearning: log-likelihood di X_forget, delle 12 frasi
     correlate, delle 6 frasi di controllo mai viste, delle 4 frasi fuori
     distribuzione e delle 6 frasi memorizzate ma non protette;
  3. ascent selettivo con proiezione in metrica di Fisher (`neural_core.unlearning`),
     con STOP di sicurezza secondario su un canario di 6 frasi disgiunte;
  4. misura DOPO con le stesse frasi;
  5. valutazione dei due criteri dichiarati:
       (1) successo del forgetting: la log-likelihood per token di X_forget
           rientra nella distribuzione delle frasi mai viste (criterio
           primario, riferimento = holdout in-distribuzione);
       (2) collaterale: variazione di probabilita' delle 12 frasi correlate
           entro il 5% relativo.

Tutti i numeri stampati vengono da questa esecuzione: lo script non stima
nulla. Uso:

    python3 experiments/mission1_unlearning/validate.py
    python3 experiments/mission1_unlearning/validate.py --epochs 400 --sweep
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus_unlearning as corpus  # noqa: E402

from neural_core.dataset import Dataset  # noqa: E402
from neural_core.loss import CrossEntropyLoss  # noqa: E402
from neural_core.optimizer import Adam  # noqa: E402
from neural_core.trainer import Trainer, build_model  # noqa: E402
from neural_core.unlearning import (  # noqa: E402
    SelectiveUnlearner, UnlearningConfig, forgetting_success,
    relative_probability_change, single_vs_distribution,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Configurazione del modello. Piccola di proposito: l'obiettivo e' misurare il
# meccanismo, non scalare. NON e' il modello di produzione (che ha d_model 128,
# 4 layer, 2.737.806 parametri): i numeri qui sotto non sono confrontabili con
# quelli del report di addestramento.
MODEL_CONFIG = {
    "d_model": 64,
    "num_heads": 4,
    "d_ff": 128,
    "num_layers": 2,
    "max_len": 64,
    "lora_rank": 4,
    "lora_alpha": 16.0,
    "dropout": 0.0,
    "seed": 42,
}
CONTEXT_SIZE = 64
STRIDE = 32
TRAIN_LR = 0.01          # learning rate dell'addestramento che crea il modello
TRAIN_BATCH = 32
FISHER_SAMPLES = 100
DEFAULT_EPOCHS = 400
#: Soglia dello STOP di sicurezza sul canario: la stessa del criterio (2), ma
#: applicata alla variazione MASSIMA su `splits["probe"]` invece che al numero
#: di frasi correlate oltre soglia.
PROBE_TOLERANCE = 0.05


# --------------------------------------------------------------------------
# Infrastruttura
# --------------------------------------------------------------------------

def build_trainer(tokenizer, learning_rate=TRAIN_LR, lora_rank=MODEL_CONFIG["lora_rank"]):
    model = build_model("transformer", vocab_size=tokenizer.vocab_size,
                        **{**MODEL_CONFIG, "lora_rank": lora_rank})
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=learning_rate)
    return Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer,
                   seed=MODEL_CONFIG["seed"])


def make_dataset(text, tokenizer):
    ids = tokenizer.encode(text)
    return Dataset(ids, context_size=CONTEXT_SIZE, stride=STRIDE)


def unknown_counts(tokenizer, texts):
    return [int(tokenizer.count_unknown(t)) for t in texts]


def measure(unlearner, texts):
    """Log-likelihood e NLL per ogni frase, senza toccare i pesi."""
    return [unlearner.loglikelihood(t) for t in texts]


def _group_line(label, group):
    """Riga di riepilogo per un gruppo di frasi (media e std)."""
    mean_lls = np.array([g["mean_loglikelihood"] for g in group])
    total_lls = np.array([g["loglikelihood"] for g in group])
    nlls = np.array([g["nll_mean"] for g in group])
    return (f"{label:20s}: n={mean_lls.size}  "
            f"NLL/token media {nlls.mean():.4f} std {nlls.std(ddof=1):.4f} | "
            f"loglik/token media {mean_lls.mean():.4f} std {mean_lls.std(ddof=1):.4f} | "
            f"loglik totale media {total_lls.mean():.4f} std {total_lls.std(ddof=1):.4f}")


# --------------------------------------------------------------------------
# Stadio 1: dati, addestramento, Fisher
# --------------------------------------------------------------------------

def build_trained(epochs=DEFAULT_EPOCHS, fisher_samples=FISHER_SAMPLES, log=print,
                  lora_rank=MODEL_CONFIG["lora_rank"]):
    """
    Prepara dati, addestra il modello e calcola la Micro-Fisher riusando
    `Trainer.compute_fisher_matrix`. Non esegue l'unlearning.

    `lora_rank=0` esegue lo stesso esperimento in modalita' full-parameter
    (tutti i 162.649 parametri addestrabili invece dei 1.536 della LoRA): e' il
    controllo che serve per capire se il danno collaterale dipende dalla
    dimensione dello spazio addestrabile.
    """
    pipeline = corpus.build_pipeline()
    splits = pipeline["splits"]
    training_text = pipeline["training_text"]
    tokenizer = pipeline["tokenizer"]
    trainer = build_trainer(tokenizer, lora_rank=lora_rank)
    dataset = make_dataset(training_text, tokenizer)

    log(f"[setup] frasi di addestramento: {len(splits['training_sentences'])} "
        f"(+ X_forget x{corpus.FORGET_REPEATS}), caratteri: {len(training_text)}")
    log(f"[setup] vocabolario: {tokenizer.vocab_size} token")
    log(f"[setup] X_retain_proj: {len(splits['retain'])} frasi | "
        f"frasi correlate: {len(splits['correlated'])} | "
        f"holdout mai visto: {len(splits['unseen_holdout'])} | "
        f"fuori distribuzione: {len(splits['archive_unseen'])} | "
        f"probe/canario: {len(splits['probe'])} | "
        f"controllo addestrato non protetto: {len(splits['background_control'])}")
    unk = {
        "forget": unknown_counts(tokenizer, [splits["forget"]])[0],
        "correlated_max": max(unknown_counts(tokenizer, splits["correlated"])),
        "holdout_max": max(unknown_counts(tokenizer, splits["unseen_holdout"])),
        "archive_max": max(unknown_counts(tokenizer, splits["archive_unseen"])),
        "probe_max": max(unknown_counts(tokenizer, splits["probe"])),
        "background_max": max(unknown_counts(tokenizer, splits["background_control"])),
    }
    log(f"[setup] token <UNK> per gruppo (max): forget={unk['forget']} "
        f"correlate={unk['correlated_max']} holdout={unk['holdout_max']} "
        f"fuori distribuzione={unk['archive_max']} "
        f"probe={unk['probe_max']} "
        f"controllo non protetto={unk['background_max']}")
    log("[setup] NB: le frasi mai viste contengono token fuori vocabolario, "
        "perche' nel corpus di dialogo ogni frase ha parole uniche. Questo "
        "abbassa la loro log-likelihood e rende il criterio (1) PIU' esigente, "
        "non meno: e' uno sbilanciamento conservativo, dichiarato.")

    log(f"[train] finestre di addestramento: {len(dataset)}")
    t0 = time.time()
    history = trainer.train(dataset.get_inputs(), dataset.get_targets(),
                            epochs=epochs, batch_size=TRAIN_BATCH)
    train_time = time.time() - t0
    log(f"[train] {epochs} epoche in {train_time:.2f}s | "
        f"loss iniziale {history[0]:.4f} -> loss finale {history[-1]:.4f}")

    t0 = time.time()
    fisher = trainer.compute_fisher_matrix(dataset, num_samples=fisher_samples)
    fisher_time = time.time() - t0
    log(f"[fisher] compute_fisher_matrix su {min(fisher_samples, len(dataset))} "
        f"campioni in {fisher_time:.2f}s | {len(fisher)} tensori addestrabili")
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    n_trainable = int(sum(np.asarray(params[i]).size for i in indices))
    n_total = int(sum(np.asarray(p).size for p in params))
    log(f"[setup] spazio addestrabile: {n_trainable}/{n_total} parametri "
        f"({100.0 * n_trainable / n_total:.2f}%), lora_rank={lora_rank}")

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "splits": splits,
        "vocab_size": tokenizer.vocab_size,
        "unknown_counts": unk,
        "train_loss_initial": history[0],
        "train_loss_final": history[-1],
        "train_seconds": train_time,
        "fisher_seconds": fisher_time,
        "n_fisher_tensors": len(fisher),
        "n_windows": len(dataset),
        "epochs": epochs,
        "fisher_samples": fisher_samples,
        # theta_0: pesi addestrabili subito dopo l'addestramento, PRIMA di
        # qualsiasi unlearning. Ogni configurazione della diagnostica riparte
        # da qui, altrimenti le righe non sarebbero confrontabili.
        "theta0": snapshot_trainable(trainer),
    }


def snapshot_trainable(trainer):
    """Copia dei parametri addestrabili, per ripristinare theta_0."""
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    return [np.asarray(params[i]).copy() for i in indices]


def restore_trainable(trainer, snapshot):
    indices = trainer._trainable_indices()
    params = trainer.model.get_parameters()
    for slot, idx in enumerate(indices):
        np.asarray(params[idx])[...] = snapshot[slot]


# --------------------------------------------------------------------------
# Stadio 2: unlearning e criteri
# --------------------------------------------------------------------------

def evaluate_unlearning(trained, unlearning_config=None, learning_rate=None, log=print,
                        use_probe=True):
    """
    Esegue l'ascent selettivo su theta_0 e valuta i due criteri. Non
    riaddestra nulla: la Fisher e il modello arrivano gia' pronti.

    `use_probe=True` attiva lo STOP di sicurezza secondario sul canario
    (`splits["probe"]`) con soglia 5%, la stessa del criterio (2) ma applicata
    alla variazione massima e non media. Il probe non entra nel gradiente.
    """
    trainer = trained["trainer"]
    splits = trained["splits"]
    config = unlearning_config or UnlearningConfig()
    if use_probe and config.probe_tolerance is None:
        config = replace(config, probe_tolerance=PROBE_TOLERANCE)
    unlearner = SelectiveUnlearner(trainer, config=config, log=log)

    correlated_before = measure(unlearner, splits["correlated"])
    holdout_before = measure(unlearner, splits["unseen_holdout"])
    archive_before = measure(unlearner, splits["archive_unseen"])
    background_before = measure(unlearner, splits["background_control"])
    probe_before = measure(unlearner, splits["probe"])
    forget_before = unlearner.loglikelihood(splits["forget"])

    log("")
    log("=== PRIMA DELL'UNLEARNING ===")
    log(f"X_forget  : NLL/token {forget_before['nll_mean']:.4f}  "
        f"loglikelihood/token {forget_before['mean_loglikelihood']:.4f}  "
        f"NLL totale {forget_before['total_nll']:.4f}  "
        f"({forget_before['n_tokens']} token)")
    for label, group in (("holdout mai visto", holdout_before),
                         ("fuori distribuzione", archive_before),
                         ("non protette (addestr.)", background_before),
                         ("probe/canario", probe_before),
                         ("correlate", correlated_before)):
        log(_group_line(label, group))

    t0 = time.time()
    result = unlearner.forget(splits["forget"], splits["retain"],
                              learning_rate=learning_rate,
                              probe_texts=splits["probe"] if use_probe else None)
    unlearn_time = time.time() - t0
    log(f"[unlearning] durata {unlearn_time:.2f}s")

    correlated_after = measure(unlearner, splits["correlated"])
    holdout_after = measure(unlearner, splits["unseen_holdout"])
    archive_after = measure(unlearner, splits["archive_unseen"])
    background_after = measure(unlearner, splits["background_control"])
    probe_after = measure(unlearner, splits["probe"])
    forget_after = unlearner.loglikelihood(splits["forget"])

    log("")
    log("=== DOPO L'UNLEARNING ===")
    log(f"X_forget  : NLL/token {forget_after['nll_mean']:.4f}  "
        f"loglikelihood/token {forget_after['mean_loglikelihood']:.4f}  "
        f"NLL totale {forget_after['total_nll']:.4f}  "
        f"({forget_after['n_tokens']} token)")
    for label, group in (("holdout mai visto", holdout_after),
                         ("fuori distribuzione", archive_after),
                         ("non protette (addestr.)", background_after),
                         ("probe/canario", probe_after),
                         ("correlate", correlated_after)):
        log(_group_line(label, group))

    # ---------------- criterio (1): successo del forgetting ----------------
    # Confronto PER TOKEN: la log-likelihood totale confronterebbe la
    # lunghezza delle frasi invece che la loro memorizzazione.
    holdout_lls_after = [g["mean_loglikelihood"] for g in holdout_after]
    archive_lls_after = [g["mean_loglikelihood"] for g in archive_after]
    background_lls_after = [g["mean_loglikelihood"] for g in background_after]
    primary = forgetting_success(forget_after["mean_loglikelihood"], holdout_lls_after)
    primary_2s = forgetting_success(forget_after["mean_loglikelihood"],
                                    holdout_lls_after, two_sided=True)
    combined = forgetting_success(forget_after["mean_loglikelihood"],
                                  holdout_lls_after + archive_lls_after)
    combined_2s = forgetting_success(forget_after["mean_loglikelihood"],
                                     holdout_lls_after + archive_lls_after,
                                     two_sided=True)
    secondary = forgetting_success(forget_after["mean_loglikelihood"], archive_lls_after)
    pre = forgetting_success(forget_before["mean_loglikelihood"], holdout_lls_after)
    vs_background = forgetting_success(forget_after["mean_loglikelihood"],
                                       background_lls_after)
    ttest = single_vs_distribution(forget_after["mean_loglikelihood"], holdout_lls_after)

    log("")
    log("=== CRITERIO (1): SUCCESSO DEL FORGETTING (loglikelihood per token) ===")
    log(f"Riferimento (holdout mai visto, n={primary['control_n']}): "
        f"media {primary['control_mean']:.4f} std {primary['control_std']:.4f} "
        f"min {primary['control_min']:.4f} max {primary['control_max']:.4f}")
    log(f"X_forget DOPO  = {primary['forget_loglik']:.4f} -> "
        f"z = {primary['z_score']:.3f} | unilaterale z<=1: "
        f"{'RAGGIUNTO' if primary['success_one_sided'] else 'NON RAGGIUNTO'} | "
        f"bilaterale |z|<=1: "
        f"{'RAGGIUNTO' if primary['success_two_sided'] else 'NON RAGGIUNTO'}")
    log(f"X_forget PRIMA = {pre['forget_loglik']:.4f} -> "
        f"z = {pre['z_score']:.3f} => "
        f"{'gia dentro la distribuzione (TEST VACUO)' if pre['success'] else 'fuori dalla distribizione'}")
    log(f"Controllo secondario (solo fuori distribuzione, n={secondary['control_n']}): "
        f"media {secondary['control_mean']:.4f} std {secondary['control_std']:.4f} "
        f"-> z = {secondary['z_score']:.3f}")
    log(f"Controllo combinato (holdout + fuori distribuzione, n={combined['control_n']}): "
        f"media {combined['control_mean']:.4f} std {combined['control_std']:.4f} "
        f"-> z = {combined['z_score']:.3f} | unilaterale "
        f"{'RAGGIUNTO' if combined['success_one_sided'] else 'NON RAGGIUNTO'} | "
        f"bilaterale {'RAGGIUNTO' if combined_2s['success'] else 'NON RAGGIUNTO'}")
    log(f"X_forget DOPO vs distribuzione holdout: t = {ttest['t']:.3f} "
        f"(n = {ttest['n']}), p ~ {ttest['p_value']:.4f} "
        "(p alto = non distinguibili; test indicativo, non esatto)")
    log(f"Riferimento alternativo (frasi memorizzate ma NON protette, n="
        f"{vs_background['control_n']}): media {vs_background['control_mean']:.4f} "
        f"std {vs_background['control_std']:.4f} -> z = {vs_background['z_score']:.3f}"
        " (se X_forget resta sopra questo riferimento, l'esempio e' stato solo "
        "degradato, non dimenticato)")

    # ---------------- criterio (2): collaterale sulle correlate ----------------
    rel_before = np.array([g["loglikelihood"] for g in correlated_before])
    rel_after = np.array([g["loglikelihood"] for g in correlated_after])
    rel_change = relative_probability_change(rel_before, rel_after)
    n_violations = int(np.sum(rel_change > 0.05))
    collateral_ok = n_violations == 0
    # La variazione e' bidirezionale: il criterio scritto conta anche le frasi
    # che sono diventate PIU' probabili. Il danno collaterale, pero', e' solo la
    # coda che peggiora. Si riportano entrambe le letture.
    drop = np.maximum(rel_before - rel_after, 0.0)
    rise = np.maximum(rel_after - rel_before, 0.0)
    n_degraded = int(np.sum(drop > 0.05))
    n_improved = int(np.sum(rise > 0.05))

    log("")
    log("=== CRITERIO (2): COLLATERALE SULLE FRASI CORRELATE ===")
    log(f"n = {rel_change.size}, soglia = 5% relativo sulla probabilita' della frase")
    log(f"variazione relativa (|delta|): media {rel_change.mean() * 100:.3f}%  "
        f"max {rel_change.max() * 100:.3f}%  "
        f"frasi oltre soglia {n_violations}/{rel_change.size} "
        f"(in peggio {n_degraded}, in meglio {n_improved})")
    log(f"solo degrado (delta p < 0 oltre il 5%): {n_degraded}/{rel_change.size} "
        f"frasi, peggioramento medio {drop.mean() * 100:.3f}%, "
        f"massimo {drop.max() * 100:.3f}%")
    for text, before, after, change in zip(splits["correlated"], rel_before,
                                           rel_after, rel_change):
        log(f"  [{'KO' if change > 0.05 else 'ok'}] {change * 100:7.3f}%  "
            f"{before:9.3f} -> {after:9.3f}  {text[:58]}")
    log(f"=> {'RAGGIUNTO' if collateral_ok else 'NON RAGGIUNTO'}")

    # ---------------- STOP di sicurezza sul canario ----------------
    probe_rel_before = np.array([g["loglikelihood"] for g in probe_before])
    probe_rel_after = np.array([g["loglikelihood"] for g in probe_after])
    probe_change = relative_probability_change(probe_rel_before, probe_rel_after)
    probe_violations = int(np.sum(probe_change > PROBE_TOLERANCE))
    probe_block = {
        "threshold": PROBE_TOLERANCE,
        "active": use_probe,
        "n": int(probe_change.size),
        "relative_change_mean": float(probe_change.mean()),
        "relative_change_max": float(probe_change.max()),
        "n_violations": probe_violations,
        "success": bool(probe_violations == 0),
        "max_during_ascent": (result.probe_max_relative_change_history[-1]
                              if result.probe_max_relative_change_history else None),
        "per_sentence": [
            {"text": t, "loglik_before": float(b), "loglik_after": float(a),
             "relative_change": float(c)}
            for t, b, a, c in zip(splits["probe"], probe_rel_before,
                                  probe_rel_after, probe_change)
        ],
    }
    log("")
    log("=== CANARIO (probe set): STOP di sicurezza e misura finale ===")
    log(f"n = {probe_change.size}, soglia = {PROBE_TOLERANCE * 100:.0f}% relativo "
        f"sulla variazione MASSIMA (non media)")
    log(f"variazione relativa: media {probe_change.mean() * 100:.3f}%  "
        f"max {probe_change.max() * 100:.3f}%  "
        f"frasi oltre soglia {probe_violations}/{probe_change.size}")
    if use_probe:
        log(f"STOP attivato: soglia superata in {result.stop_reason == 'probe_drift'} "
            f"(motivo effettivo: {result.stop_reason}, step {result.steps_executed})")
    for text, before, after, change in zip(splits["probe"], probe_rel_before,
                                           probe_rel_after, probe_change):
        log(f"  [{'KO' if change > PROBE_TOLERANCE else 'ok'}] "
            f"{change * 100:7.3f}%  {before:9.3f} -> {after:9.3f}  {text[:58]}")

    return {
        "unlearning": {
            "stop_reason": result.stop_reason,
            "projection_mode": result.projection_mode,
            "fisher_floor": config.fisher_floor,
            "learning_rate_override": learning_rate,
            "rollback_on_stop": config.rollback_on_stop,
            "retain_threshold_violated": result.retain_threshold_violated,
            "steps": result.steps_executed,
            "restarts": result.restarts,
            "learning_rate_initial": result.learning_rate_initial,
            "learning_rate_final": result.learning_rate_final,
            "halving_events": result.halving_events,
            "seconds": unlearn_time,
            "forget_nll_before": forget_before["nll_mean"],
            "forget_nll_after": forget_after["nll_mean"],
            "forget_loglik_before": forget_before["loglikelihood"],
            "forget_loglik_after": forget_after["loglikelihood"],
            "forget_loglik_per_token_before": forget_before["mean_loglikelihood"],
            "forget_loglik_per_token_after": forget_after["mean_loglikelihood"],
            "retain_loss_before": result.retain_loss_before,
            "retain_loss_after": result.retain_loss_after,
            "retain_loss_relative_change": result.retain_loss_relative_change,
            "trainable_parameters": result.trainable_parameters,
            "probe_tolerance": result.probe_tolerance,
            "probe_sentences": result.probe_sentences,
            "probe_max_relative_change_after": result.probe_max_relative_change_after,
            "rolled_back_steps": result.rolled_back_steps,
            "history": result.history,
            "projection_alpha_first": (result.projection_coefficients[0]
                                       if result.projection_coefficients else None),
            "projection_alpha_last": (result.projection_coefficients[-1]
                                      if result.projection_coefficients else None),
        },
        "criteria": {
            "forgetting_primary": primary,
            "forgetting_primary_two_sided": primary_2s,
            "forgetting_before": pre,
            "forgetting_secondary_archive": secondary,
            "forgetting_combined": combined,
            "forgetting_combined_two_sided": combined_2s,
            "forgetting_vs_unprotected_training": vs_background,
            "t_after_vs_holdout": ttest,
            "collateral": {
                "threshold": 0.05,
                "relative_change_mean": float(rel_change.mean()),
                "relative_change_max": float(rel_change.max()),
                "n_violations": n_violations,
                "n_degraded": n_degraded,
                "n_improved": n_improved,
                "degradation_mean": float(drop.mean()),
                "degradation_max": float(drop.max()),
                "n": int(rel_change.size),
                "success": bool(collateral_ok),
                "per_sentence": [
                    {"text": t, "loglik_before": float(b), "loglik_after": float(a),
                     "relative_change": float(c)}
                    for t, b, a, c in zip(splits["correlated"], rel_before,
                                          rel_after, rel_change)
                ],
            },
            "probe": probe_block,
        },
    }


def run_experiment(epochs=DEFAULT_EPOCHS, fisher_samples=FISHER_SAMPLES,
                   unlearning_config=None, log=print, learning_rate=None,
                   use_probe=True):
    """
    Pipeline completa: dati, addestramento, Fisher, unlearning, criteri.
    Usata sia dallo script sia dai test.
    """
    trained = build_trained(epochs=epochs, fisher_samples=fisher_samples, log=log)
    out = evaluate_unlearning(trained, unlearning_config=unlearning_config,
                              learning_rate=learning_rate, log=log,
                              use_probe=use_probe)
    out["settings"] = {
        "model": MODEL_CONFIG,
        "context_size": CONTEXT_SIZE,
        "stride": STRIDE,
        "train_lr": TRAIN_LR,
        "train_epochs": trained["epochs"],
        "train_batch": TRAIN_BATCH,
        "fisher_samples": trained["fisher_samples"],
        "vocab_size": trained["vocab_size"],
        "forget_repeats": corpus.FORGET_REPEATS,
    }
    out["dataset"] = {
        "n_training_sentences": len(trained["splits"]["training_sentences"]),
        "n_retain": len(trained["splits"]["retain"]),
        "n_correlated": len(trained["splits"]["correlated"]),
        "n_unseen_holdout": len(trained["splits"]["unseen_holdout"]),
        "n_archive_unseen": len(trained["splits"]["archive_unseen"]),
        "n_background_control": len(trained["splits"]["background_control"]),
        "n_probe": len(trained["splits"]["probe"]),
        "n_windows": trained["n_windows"],
        "unknown_counts": trained["unknown_counts"],
    }
    out["train"] = {
        "loss_initial": trained["train_loss_initial"],
        "loss_final": trained["train_loss_final"],
        "seconds": trained["train_seconds"],
    }
    out["fisher"] = {
        "seconds": trained["fisher_seconds"],
        "n_tensors": trained["n_fisher_tensors"],
        "source": out["unlearning"].get("fisher_source",
                                        "Trainer.fisher_matrix (Micro-Fisher/LoRA esistente)"),
    }
    return out


# --------------------------------------------------------------------------
# Diagnostica fuori protocollo
# --------------------------------------------------------------------------

def run_sweep(trained, learning_rates=(1e-4, 1e-3, 1e-2, 1e-1),
              modes=("prescribed", "metric_orthogonal"),
              floors=(1e-3, 0.9), log=print, use_probe=True):
    """
    DIAGNOSTICA, non parte del protocollo: a quali learning rate, formule di
    proiezione e pavimenti della Fisher l'ascent produce un effetto misurabile.
    Il protocollo fissa eta = lr_fine_tuning/10 = 1e-4, formula prescritta e
    pavimento 1e-3: la prima riga della tabella e' esattamente il protocollo.

   _theta_0 viene ripristinato prima di ogni configurazione, cosi' tutte le
    righe partono dallo stesso modello.
    """
    log("")
    log("=== DIAGNOSTICA (fuori protocollo): eta x proiezione x pavimento Fisher ===")
    log("prima riga = protocollo esatto: eta 1e-04, formula prescritta, "
        "pavimento 1e-03")
    snapshot = trained["theta0"]
    rows = []
    for mode in modes:
        for floor in floors:
            for eta in learning_rates:
                restore_trainable(trained["trainer"], snapshot)
                config = UnlearningConfig(projection_mode=mode, fisher_floor=floor)
                out = evaluate_unlearning(trained, unlearning_config=config,
                                          learning_rate=eta,
                                          log=lambda *a, **k: None,
                                          use_probe=use_probe)
                u, c = out["unlearning"], out["criteria"]
                row = {
                    "projection_mode": mode,
                    "fisher_floor": floor,
                    "learning_rate": eta,
                    "stop_reason": u["stop_reason"],
                    "steps": u["steps"],
                    "restarts": u["restarts"],
                    "forget_nll_before": u["forget_nll_before"],
                    "forget_nll_after": u["forget_nll_after"],
                    "z_one_sided": c["forgetting_primary"]["z_score"],
                    "forgetting_one_sided": c["forgetting_primary"]["success_one_sided"],
                    "forgetting_two_sided": c["forgetting_primary"]["success_two_sided"],
                    "collateral_max_change": c["collateral"]["relative_change_max"],
                    "collateral_max_degradation": c["collateral"]["degradation_max"],
                    "collateral_n_violations": c["collateral"]["n_violations"],
                    "collateral_success": c["collateral"]["success"],
                    "probe_max_change": c["probe"]["relative_change_max"],
                    "probe_n_violations": c["probe"]["n_violations"],
                    "retain_loss_relative_change": u["retain_loss_relative_change"],
                }
                rows.append(row)
                log(f"{mode:18s} floor={floor:<5} eta={eta:<7} "
                    f"stop={u['stop_reason'][:20]:20s} step={u['steps']:3d} "
                    f"restart={u['restarts']} | "
                    f"NLL {u['forget_nll_before']:.3f}->{u['forget_nll_after']:.3f} "
                    f"z={row['z_one_sided']:8.2f} "
                    f"1-sided={'SI' if row['forgetting_one_sided'] else 'no'} "
                    f"2-sided={'SI' if row['forgetting_two_sided'] else 'no'} | "
                    f"coll max {row['collateral_max_change'] * 100:9.2f}% "
                    f"(peggiora {row['collateral_max_degradation'] * 100:8.2f}%) "
                    f"nviol={row['collateral_n_violations']:2d} | "
                    f"probe max {row['probe_max_change'] * 100:8.2f}% "
                    f"nviol={row['probe_n_violations']:2d} | "
                    f"retain {row['retain_loss_relative_change'] * 100:+7.2f}%")
    restore_trainable(trained["trainer"], snapshot)
    return rows


def run_first_order_diagnostic(trained, steps=15, log=print):
    """
    A4: il segno di `first_order_forget_delta` (variazione prevista di primo
    ordine della NLL di X_forget) dipende dalla formula di proiezione?

    Il protocollo prescrive una formula che, con Fisher anisotropa, non e' una
    proiezione ortogonale: il termine di protezione puo' avere segno opposto al
    gradiente di forget e dominarlo. Qui le due formule vengono eseguite per
    `steps` passi dalla STESSA theta_0, senza canario e senza rollback, e si
    riportano i valori grezzi. `first_order_forget_delta < 0` significa che il
    passo non e' un passo di ascent sulla loss che si vuole far crescere.
    """
    log("")
    log(f"=== DIAGNOSTICA A4: segno di delta_forget_1o, prime {steps} fasi ===")
    log("stessa theta_0 per entrambe le formule, nessun canario, nessun rollback")
    snapshot = trained["theta0"]
    rows = []
    for mode in ("prescribed", "metric_orthogonal"):
        restore_trainable(trained["trainer"], snapshot)
        config = UnlearningConfig(projection_mode=mode, max_steps=steps,
                                  retain_loss_tolerance=1e9,  # nessuno STOP su retain
                                  probe_tolerance=None)
        out = evaluate_unlearning(trained, unlearning_config=config,
                                  log=lambda *a, **k: None, use_probe=False)
        history = out["unlearning"]["history"]
        negative = [h["first_order_forget_delta"] for h in history
                    if h["first_order_forget_delta"] < 0]
        rows.append({
            "projection_mode": mode,
            "steps": len(history),
            "first_order_min": min(h["first_order_forget_delta"] for h in history),
            "first_order_max": max(h["first_order_forget_delta"] for h in history),
            "n_negative_steps": len(negative),
            "first_order_first5": [h["first_order_forget_delta"] for h in history[:5]],
            "projection_ratio_first5": [h["projection_ratio"] for h in history[:5]],
            "forget_nll_sequence": [h["forget_nll"] for h in history[:steps]],
        })
        log("")
        log(f"-- proiezione {mode}")
        log(f"{'step':>5} {'delta_forget_1o':>16} {'||g_proj||/||g_f||':>18} "
            f"{'NLL(X_forget)':>14}")
        for h in history:
            log(f"{h['step']:5d} {h['first_order_forget_delta']:+16.3e} "
                f"{(h['projection_ratio'] if h['projection_ratio'] is not None else float('nan')):18.3e} "
                f"{h['forget_nll']:14.4f}")
        log(f"passi con delta_forget_1o NEGATIVO: {len(negative)}/{len(history)}")
    restore_trainable(trained["trainer"], snapshot)
    return rows


def run_extended_sweep(trained, learning_rates=(1e-4, 3e-4, 1e-3, 3e-3),
                       max_steps_list=(100, 300, 500), floors=(1e-3, 0.9), log=print):
    """
    A3: sweep esteso sulla configurazione piu' promettente
    (`metric_orthogonal`) con piu' step e learning rate piu' alti, per stabilire
    se il forgetting sia irraggiungibile per struttura o solo perche' la
    ricerca si fermava a 100 step.

    Lo sweep risponde a "il forgetting e' irraggiungibile o la ricerca si
    fermava troppo presto?", quindi il canario e' DISATTIVATO qui: con il
    canario attivo ogni configurazione si fermerebbe al primo passo (misurato:
    19.65% di variazione massima sul canario dopo un solo passo a eta=1e-4) e
    la domanda diventerebbe irrilevante. Il ruolo del canario e' quello di una
    rete di sicurezza in esercizio, non di un vincolo di ricerca.
    """
    log("")
    log("=== DIAGNOSTICA A3: sweep esteso su max_steps ed eta "
        "(solo proiezione corretta) ===")
    log("fuori protocollo: il protocollo fissa max_steps=100 ed eta=1e-4")
    log("canario DISATTIVATO in questa diagnostica: misurerebbe solo dove si "
        "ferma la rete di sicurezza, non se il forgetting e' raggiungibile")
    snapshot = trained["theta0"]
    rows = []
    for floor in floors:
        for eta in learning_rates:
            for max_steps in max_steps_list:
                restore_trainable(trained["trainer"], snapshot)
                config = UnlearningConfig(projection_mode="metric_orthogonal",
                                          fisher_floor=floor, max_steps=max_steps,
                                          probe_tolerance=None)
                out = evaluate_unlearning(trained, unlearning_config=config,
                                          learning_rate=eta, log=lambda *a, **k: None,
                                          use_probe=False)
                u, c = out["unlearning"], out["criteria"]
                row = {
                    "fisher_floor": floor,
                    "learning_rate": eta,
                    "max_steps": max_steps,
                    "steps_executed": u["steps"],
                    "stop_reason": u["stop_reason"],
                    "restarts": u["restarts"],
                    "forget_nll_before": u["forget_nll_before"],
                    "forget_nll_after": u["forget_nll_after"],
                    "z_one_sided": c["forgetting_primary"]["z_score"],
                    "forgetting_one_sided": c["forgetting_primary"]["success_one_sided"],
                    "forgetting_two_sided": c["forgetting_primary"]["success_two_sided"],
                    "collateral_max_change": c["collateral"]["relative_change_max"],
                    "collateral_n_violations": c["collateral"]["n_violations"],
                    "probe_max_change": c["probe"]["relative_change_max"],
                    "probe_n_violations": c["probe"]["n_violations"],
                    "retain_loss_relative_change": u["retain_loss_relative_change"],
                }
                rows.append(row)
                log(f"floor={floor:<5} eta={eta:<7} max_steps={max_steps:<4} "
                    f"stop={u['stop_reason'][:18]:18s} step={u['steps']:3d} | "
                    f"NLL {u['forget_nll_before']:.3f}->{u['forget_nll_after']:.3f} "
                    f"z={row['z_one_sided']:8.2f} "
                    f"1-sided={'SI' if row['forgetting_one_sided'] else 'no'} "
                    f"2-sided={'SI' if row['forgetting_two_sided'] else 'no'} | "
                    f"coll max {row['collateral_max_change'] * 100:9.2f}% "
                    f"nviol={row['collateral_n_violations']:2d} | "
                    f"retain {row['retain_loss_relative_change'] * 100:+8.2f}%")
    restore_trainable(trained["trainer"], snapshot)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--fisher-samples", type=int, default=FISHER_SAMPLES)
    parser.add_argument("--sweep", action="store_true",
                        help="Esegue anche la diagnostica su eta/proiezione/pavimento.")
    parser.add_argument("--sweep-extended", action="store_true",
                        help="A3: sweep esteso su max_steps ed eta con la sola "
                             "proiezione corretta.")
    parser.add_argument("--first-order", action="store_true",
                        help="A4: confronto grezzo del segno di "
                             "delta_forget_1o fra le due formule di proiezione.")
    parser.add_argument("--no-probe", action="store_true",
                        help="Disattiva lo STOP di sicurezza sul canario (A2).")
    parser.add_argument("--no-lora", action="store_true",
                        help="Modalita' full-parameter: tutti i parametri "
                             "addestrabili invece della sola LoRA.")
    parser.add_argument("--out", default=str(RESULTS_DIR / "unlearning_results.json"))
    args = parser.parse_args()

    lora_rank = 0 if args.no_lora else MODEL_CONFIG["lora_rank"]
    suffix = "_full" if args.no_lora else ""
    t0 = time.time()
    trained = build_trained(epochs=args.epochs, fisher_samples=args.fisher_samples,
                            lora_rank=lora_rank)
    out = evaluate_unlearning(trained, use_probe=not args.no_probe)
    if args.sweep:
        out["sweep"] = run_sweep(trained, use_probe=not args.no_probe)
    if args.sweep_extended:
        out["sweep_extended"] = run_extended_sweep(trained)
    if args.first_order:
        out["first_order_diagnostic"] = run_first_order_diagnostic(trained)
    out["settings"] = {
        "model": {**MODEL_CONFIG, "lora_rank": lora_rank},
        "context_size": CONTEXT_SIZE,
        "stride": STRIDE,
        "train_lr": TRAIN_LR,
        "train_epochs": trained["epochs"],
        "train_batch": TRAIN_BATCH,
        "fisher_samples": trained["fisher_samples"],
        "vocab_size": trained["vocab_size"],
        "forget_repeats": corpus.FORGET_REPEATS,
    }
    out["dataset"] = {
        "n_training_sentences": len(trained["splits"]["training_sentences"]),
        "n_retain": len(trained["splits"]["retain"]),
        "n_correlated": len(trained["splits"]["correlated"]),
        "n_unseen_holdout": len(trained["splits"]["unseen_holdout"]),
        "n_archive_unseen": len(trained["splits"]["archive_unseen"]),
        "n_background_control": len(trained["splits"]["background_control"]),
        "n_probe": len(trained["splits"]["probe"]),
        "n_windows": trained["n_windows"],
        "unknown_counts": trained["unknown_counts"],
    }
    out["train"] = {
        "loss_initial": trained["train_loss_initial"],
        "loss_final": trained["train_loss_final"],
        "seconds": trained["train_seconds"],
    }
    out["fisher"] = {
        "seconds": trained["fisher_seconds"],
        "n_tensors": trained["n_fisher_tensors"],
        "source": "Trainer.fisher_matrix (Micro-Fisher/LoRA esistente)",
    }
    out["total_seconds"] = time.time() - t0

    u, c = out["unlearning"], out["criteria"]
    print("")
    print("=== ESITO ===")
    print(f"proiezione usata                     : {u['projection_mode']} "
          f"(pavimento Fisher {u['fisher_floor']})")
    print(f"criterio di stop che ha fermato l'ascent : {u['stop_reason']}")
    print(f"step di ascent eseguiti                  : {u['steps']}")
    print(f"riavvii di eta necessari                 : {u['restarts']} "
          f"(eta {u['learning_rate_initial']:.2e} -> {u['learning_rate_final']:.2e})")
    print(f"variazione relativa loss X_retain_proj   : "
          f"{u['retain_loss_relative_change'] * 100:+.3f}% (soglia +/-5%)")
    print(f"criterio (1) successo del forgetting     : unilaterale "
          f"{'RAGGIUNTO' if c['forgetting_primary']['success_one_sided'] else 'NON RAGGIUNTO'} | "
          f"bilaterale {'RAGGIUNTO' if c['forgetting_primary']['success_two_sided'] else 'NON RAGGIUNTO'}")
    print(f"criterio (2) collaterale <= 5% relativo  : "
          f"{'RAGGIUNTO' if c['collateral']['success'] else 'NON RAGGIUNTO'} "
          f"({c['collateral']['n_violations']}/{c['collateral']['n']} oltre soglia, "
          f"di cui {c['collateral']['n_degraded']} in peggio)")
    if c["probe"]["active"]:
        print(f"canario (probe, n={c['probe']['n']})             : "
              f"variazione massima {c['probe']['relative_change_max'] * 100:.3f}% "
              f"(soglia {c['probe']['threshold'] * 100:.0f}%), "
              f"{c['probe']['n_violations']}/{c['probe']['n']} oltre soglia | "
              f"STOP di sicurezza attivato: "
              f"{'SI' if u['stop_reason'] == 'probe_drift' else 'no'}")
    print(f"durata totale                            : {out['total_seconds']:.1f}s")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_name(path.stem + suffix + path.suffix)
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"risultati grezzi salvati in {path}")


if __name__ == "__main__":
    main()
