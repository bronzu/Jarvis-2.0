"""
Validazione misurata del Continual Learning (Micro-EWC con LoRA).

Esegue e stampa i quattro criteri dichiarati nel protocollo:

  (a) riduzione della RAM occupata da `fisher_matrix` + `optimal_weights`
      rispetto alla versione full-parameter;
  (b) latenza di `compute_fisher_matrix` (riportata per completezza);
  (c) ritenzione sul task precedente, misurata su loss e accuratezza reali,
      con EWC attivo e con EWC disattivato;
  (d) apprendimento del task nuovo: LoRA contro fine-tuning completo, a parita'
      di dati e di epoche.

Tutti i numeri stampati vengono da questa esecuzione: lo script non stima
nulla. Raw output: vedi `results/validation_output.txt`.

Uso:
    python3 experiments/phase4_continual_learning/validate.py            # tutto
    python3 experiments/phase4_continual_learning/validate.py --criterion a
"""

import argparse
import json
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402

from neural_core.network import MicroTransformer  # noqa: E402
from neural_core.tokenizer import Tokenizer  # noqa: E402
from neural_core.dataset import Dataset  # noqa: E402
from neural_core.loss import CrossEntropyLoss  # noqa: E402
from neural_core.optimizer import Adam  # noqa: E402
from neural_core.trainer import Trainer  # noqa: E402

# --------------------------------------------------------------------------
# Configurazione. Tutti i valori sono espliciti e riproducibili.
# --------------------------------------------------------------------------

# Configurazione di produzione del modello (17_MODEL_TRAINING_PROVENANCE.md
# sezione 2.1). Usata per il criterio (a): il confronto e' richiesto rispetto
# ai 2,7M di parametri del modello reale, non di un modello ridotto.
PRODUCTION_CONFIG = dict(vocab_size=8590, d_model=128, num_heads=4, d_ff=256,
                         num_layers=4, max_len=256)

# Configurazione dei criteri (c) e (d): abbastanza grande da essere un
# esperimento, abbastanza piccola da girare su CPU in tempi utili.
LEARNING_CONFIG = dict(d_model=128, num_heads=4, d_ff=256, num_layers=2, max_len=32)

LORA_RANK = 16
LORA_ALPHA = 16.0
EWC_LAMBDA = 100.0
# 40 epoch: la curva di apprendimento sul task A satura intorno all'epoch 20
# (accuratezza di validazione ~0,12, loss ~7,83 contro 8,81 casuale). Con un
# numero inferiore di epoch il task A non e' appreso e la ritenzione del
# criterio (c) non sarebbe misurabile.
EPOCHS = 40
LEARNING_RATE = 0.02
BATCH_SIZE = 32
CONTEXT_SIZE = 16
MAX_VOCAB = 8000
FISHER_SAMPLES = 50
VAL_FRACTION = 0.25
REPLAY_RATIO = 0.2
SEED = 42

RANKS_MEASURED = (4, 16)


# --------------------------------------------------------------------------
# Utilita' di misura
# --------------------------------------------------------------------------

def rss_peak_mb():
    """RSS di picco del processo in MB (macOS: ru_maxrss e' in byte)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def dict_bytes(d):
    """Byte effettivamente occupati dagli array numpy di un dizionario."""
    return int(sum(np.asarray(v).nbytes for v in d.values()))


def make_dataset(text, tokenizer, context_size=CONTEXT_SIZE):
    ds = Dataset(tokenizer.encode(text), context_size=context_size, stride=context_size)
    return np.asarray(ds.get_inputs()), np.asarray(ds.get_targets())


def split_train_val(inputs, targets, val_fraction=VAL_FRACTION, seed=SEED):
    """Split deterministico: la ritenzione e' misurata su dati mai visti."""
    n = len(inputs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (inputs[train_idx], targets[train_idx],
            inputs[val_idx], targets[val_idx])


def build_tokenizer():
    text_a, sources_a = corpus.load_task_a_text()
    text_b, sources_b = corpus.load_task_b_text()
    tok = Tokenizer(max_vocab_size=MAX_VOCAB, mode="whitespace")
    tok.fit(text_a + "\n" + text_b)
    return tok, text_a, text_b, sources_a, sources_b


def make_trainer(model, tokenizer, lr=LEARNING_RATE, ewc_lambda=EWC_LAMBDA):
    return Trainer(model=model, tokenizer=tokenizer, loss=CrossEntropyLoss(),
                   optimizer=Adam(learning_rate=lr), seed=SEED,
                   ewc_lambda=ewc_lambda)


def accuracy(trainer, inputs, targets, batch_size=BATCH_SIZE):
    """Accuratezza token-level su next-token prediction."""
    correct = total = 0
    for start in range(0, len(inputs), batch_size):
        b_in = inputs[start:start + batch_size]
        b_tgt = np.asarray(targets[start:start + batch_size])
        preds = np.argmax(trainer.model.forward(b_in, training=False), axis=-1)
        if preds.shape != b_tgt.shape:
            b_tgt = b_tgt.reshape(preds.shape)
        correct += int(np.sum(preds == b_tgt))
        total += int(preds.size)
    return correct / total if total else 0.0


def build_lora_model(vocab_size, seed=SEED, lora_rank=LORA_RANK):
    return MicroTransformer(seed=seed, lora_rank=lora_rank, lora_alpha=LORA_ALPHA,
                            vocab_size=vocab_size, **LEARNING_CONFIG)


def build_full_model(vocab_size, seed=SEED):
    return MicroTransformer(seed=seed, lora_rank=0, vocab_size=vocab_size,
                            **LEARNING_CONFIG)


def train_on(trainer, inputs, targets, epochs=EPOCHS, replay_buffer=None):
    return trainer.train(inputs, targets, epochs=epochs,
                         batch_size=BATCH_SIZE, replay_buffer=replay_buffer)


def anchor_fisher(trainer, inputs, targets, samples=FISHER_SAMPLES):
    trainer.compute_fisher_matrix((inputs, targets), num_samples=samples)


# --------------------------------------------------------------------------
# Criterio (a) — riduzione della RAM
# --------------------------------------------------------------------------

def criterion_a():
    print("=" * 78)
    print("CRITERIO (a) — RIDUZIONE DELLA RAM: fisher_matrix + optimal_weights")
    print("=" * 78)
    vocab = PRODUCTION_CONFIG["vocab_size"]
    rng = np.random.default_rng(SEED)
    fisher_ds = (rng.integers(0, vocab, size=(FISHER_SAMPLES, CONTEXT_SIZE)),
                 rng.integers(0, vocab, size=(FISHER_SAMPLES, CONTEXT_SIZE)))

    rows = []

    # --- versione full-parameter (pre-refactoring): tutti i 2,7M parametri ---
    model = MicroTransformer(seed=SEED, lora_rank=0, **PRODUCTION_CONFIG)
    trainer = make_trainer(model, Tokenizer(max_vocab_size=vocab, mode="whitespace"))
    n_params = model.parameter_count()
    tracemalloc.start()
    t0 = time.perf_counter()
    trainer.compute_fisher_matrix(fisher_ds, num_samples=FISHER_SAMPLES)
    latency = (time.perf_counter() - t0) * 1000
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    full_bytes = dict_bytes(trainer.fisher_matrix) + dict_bytes(trainer.optimal_weights)
    full_entries = sum(int(np.asarray(v).size) for v in trainer.fisher_matrix.values())
    rows.append(dict(variant="full-parameter", lora_rank=None, params=n_params,
                     trainable=n_params, entries=full_entries,
                     bytes=full_bytes, tracemalloc_peak=int(tracemalloc_peak),
                     latency_ms=latency))
    del trainer, model

    # --- versione Micro-EWC/LoRA ---
    for rank in RANKS_MEASURED:
        model = MicroTransformer(seed=SEED, lora_rank=rank, lora_alpha=LORA_ALPHA,
                                 **PRODUCTION_CONFIG)
        trainer = make_trainer(model, Tokenizer(max_vocab_size=vocab, mode="whitespace"))
        n_trainable = model.trainable_parameter_count()
        tracemalloc.start()
        t0 = time.perf_counter()
        trainer.compute_fisher_matrix(fisher_ds, num_samples=FISHER_SAMPLES)
        latency = (time.perf_counter() - t0) * 1000
        _, tracemalloc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        lora_bytes = dict_bytes(trainer.fisher_matrix) + dict_bytes(trainer.optimal_weights)
        lora_entries = sum(int(np.asarray(v).size) for v in trainer.fisher_matrix.values())
        rows.append(dict(variant=f"Micro-EWC/LoRA rank {rank}", lora_rank=rank,
                         params=model.parameter_count(), trainable=n_trainable,
                         entries=lora_entries, bytes=lora_bytes,
                         tracemalloc_peak=int(tracemalloc_peak), latency_ms=latency))
        del trainer, model

    base = rows[0]
    print(f"\nConfigurazione di produzione: vocab {vocab}, d_model "
          f"{PRODUCTION_CONFIG['d_model']}, d_ff {PRODUCTION_CONFIG['d_ff']}, "
          f"{PRODUCTION_CONFIG['num_layers']} layer")
    print(f"Campioni Fisher: {FISHER_SAMPLES}\n")
    header = (f"{'variante':<24}{'params':>12}{'addestrabili':>15}"
              f"{'entries':>12}{'RAM (MB)':>11}{'tracemalloc (MB)':>18}{'latenza (ms)':>14}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['variant']:<24}{r['params']:>12,}{r['trainable']:>15,}"
              f"{r['entries']:>12,}{r['bytes']/1048576:>11.4f}"
              f"{r['tracemalloc_peak']/1048576:>18.4f}{r['latency_ms']:>14.2f}")

    print()
    for r in rows[1:]:
        red = 100.0 * (1.0 - r["bytes"] / base["bytes"])
        print(f"Riduzione strutture persistenti, {base['variant']} -> "
              f"{r['variant']}: {base['bytes']:,} B -> {r['bytes']:,} B = {red:.4f}%")
    rows[0]["reduction_pct"] = None
    for r in rows[1:]:
        r["reduction_pct"] = 100.0 * (1.0 - r["bytes"] / base["bytes"])

    # --- RSS di processo: una variante per subprocesso pulito -----------------
    print("\nRSS reale di processo (`ru_maxrss`, una variante per processo):")
    print(f"{'variante':<24}{'baseline (MB)':>15}{'picco (MB)':>13}"
          f"{'delta (MB)':>13}{'riduzione':>12}")
    print("-" * 77)
    rss_rows = []
    probe = str(Path(__file__).resolve().parent / "rss_probe.py")
    for r in rows:
        rank_arg = "0" if r["lora_rank"] is None else str(r["lora_rank"])
        proc = subprocess.run([sys.executable, probe, rank_arg],
                              capture_output=True, text=True, check=True)
        _, b0, peak, delta, persistent, entries = proc.stdout.split()
        b0, peak, delta = float(b0), float(peak), float(delta)
        r["rss_baseline_mb"] = b0
        r["rss_peak_mb"] = peak
        r["rss_delta_mb"] = delta
        rss_rows.append(r)
        print(f"{r['variant']:<24}{b0:>15.2f}{peak:>13.2f}{delta:>13.2f}", end="")
        if r["rss_delta_mb"] and base["rss_delta_mb"]:
            print(f"{100.0*(1-r['rss_delta_mb']/base['rss_delta_mb']):>11.2f}%")
        else:
            print(f"{'-':>12}")
    print(f"\nRSS di picco del processo principale a fine esecuzione: "
          f"{rss_peak_mb():.2f} MB")
    return rows


# --------------------------------------------------------------------------
# Criteri (c) e (d) — stesso esperimento A -> B
# --------------------------------------------------------------------------

def _run_scenario(lora_rank, ewc_lambda, use_replay, tok, data, epochs=EPOCHS):
    """Addestra su A, poi su B. Restituisce tutte le metriche misurate."""
    a_in_tr, a_tg_tr, a_in_va, a_tg_va, b_in_tr, b_tg_tr, b_in_va, b_tg_va = data
    vocab_size = len(tok.vocabulary)
    model = build_lora_model(vocab_size) if lora_rank else build_full_model(vocab_size)
    trainer = make_trainer(model, tok, ewc_lambda=ewc_lambda)

    # --- Task A ---
    train_on(trainer, a_in_tr, a_tg_tr, epochs=epochs)
    anchor_fisher(trainer, a_in_tr, a_tg_tr)
    a_loss_before = trainer.evaluate(a_in_va, a_tg_va, batch_size=BATCH_SIZE)
    a_acc_before = accuracy(trainer, a_in_va, a_tg_va)

    # --- nuovo task ---
    if lora_rank:
        # `add_task_adapter` invalida da se' la cache degli indici addestrabili
        # e `Trainer._trainable_indices()` delega al modello a ogni chiamata.
        model.add_task_adapter(warm_start=False)   # cold start
    replay = None
    if use_replay:
        # `set_replay_buffer` vuole una lista di tuple per singolo campione.
        # Il rapporto 80/20 e' il default di `Trainer._sample_replay`.
        replay = trainer.set_replay_buffer(
            [(a_in_tr[i], a_tg_tr[i]) for i in range(len(a_in_tr))]
        )
        replay_buffer = replay if replay else None
    else:
        replay_buffer = None
    train_on(trainer, b_in_tr, b_tg_tr, epochs=epochs, replay_buffer=replay_buffer)

    a_loss_after = trainer.evaluate(a_in_va, a_tg_va, batch_size=BATCH_SIZE)
    a_acc_after = accuracy(trainer, a_in_va, a_tg_va)
    b_loss = trainer.evaluate(b_in_va, b_tg_va, batch_size=BATCH_SIZE)
    b_acc = accuracy(trainer, b_in_va, b_tg_va)

    return dict(
        lora_rank=lora_rank, ewc_lambda=ewc_lambda, replay=use_replay,
        replay_ratio=REPLAY_RATIO if use_replay else 0.0,
        trainable=model.trainable_parameter_count(),
        total_params=model.parameter_count(),
        a_loss_before=a_loss_before, a_loss_after=a_loss_after,
        a_acc_before=a_acc_before, a_acc_after=a_acc_after,
        b_loss=b_loss, b_acc=b_acc,
    )


def _pct(before, after):
    return 100.0 * (after - before) / before if before else float("nan")


# Chiavi degli scenari. Gli scenari sono calcolati una volta sola e riusati:
# il run "LoRA + EWC attivo" e' contemporaneamente il confronto di (c) e il
# lato LoRA di (d). Ricalcolarlo produrrebbe due numeri diversi solo per il
# rumore di schedulazione, che renderebbe il gap di (d) non confrontabile.
SC_LORA_EWCON = "lora_ewc_on"
SC_LORA_EWCOFF = "lora_ewc_off"
SC_FULLFT = "full_finetune"


def criteria_c_d():
    tok, text_a, text_b, sources_a, sources_b = build_tokenizer()
    a_all_in, a_all_tg = make_dataset(text_a, tok)
    b_all_in, b_all_tg = make_dataset(text_b, tok)
    a_in_tr, a_tg_tr, a_in_va, a_tg_va = split_train_val(a_all_in, a_all_tg)
    b_in_tr, b_tg_tr, b_in_va, b_tg_va = split_train_val(b_all_in, b_all_tg)
    data = (a_in_tr, a_tg_tr, a_in_va, a_tg_va, b_in_tr, b_tg_tr, b_in_va, b_tg_va)

    print("=" * 78)
    print("CRITERI (c) e (d) — RITENZIONE E APPRENDIMENTO, STESSO ESPERIMENTO A->B")
    print("=" * 78)
    print("Task A (prosa tecnica italiana):")
    for s in sources_a:
        print(f"  - {s}")
    print(f"  parole: {len(text_a.split()):,}  campioni totali: {len(a_all_in):,}"
          f"  (train {len(a_in_tr):,} / val {len(a_in_va):,})")
    print("Task B (codice sorgente Python):")
    for s in sources_b:
        print(f"  - {s}")
    print(f"  parole: {len(text_b.split()):,}  campioni totali: {len(b_all_in):,}"
          f"  (train {len(b_in_tr):,} / val {len(b_in_va):,})")
    print(f"\nVocabolario: {len(tok.vocabulary):,} token "
          f"(loss casuale = ln(vocab) = {np.log(len(tok.vocabulary)):.4f})")
    print(f"Modello: d_model {LEARNING_CONFIG['d_model']}, d_ff {LEARNING_CONFIG['d_ff']}, "
          f"{LEARNING_CONFIG['num_layers']} layer, num_heads {LEARNING_CONFIG['num_heads']}")
    print(f"Addestramento: {EPOCHS} epoch per task, batch {BATCH_SIZE}, "
          f"Adam lr {LEARNING_RATE}, seed {SEED}")
    print(f"LoRA: rank {LORA_RANK}, alpha {LORA_ALPHA}  |  ewc_lambda: {EWC_LAMBDA:g}")
    print(f"Replay episodico: {int(REPLAY_RATIO*100)}% di dati storici del task A")
    print(f"Valutazione su split di validazione mai visto in addestramento "
          f"({int(VAL_FRACTION*100)}% dei campioni): la ritenzione non e' misurata "
          f"su dati memorizzati.")

    print("\n" + "-" * 78)
    print("CRITERIO (c) — RITENZIONE SUL TASK A (EWC ATTIVO vs DISATTIVATO)")
    print("-" * 78)

    cache = {}

    def run(key, lora_rank, ewc_lambda):
        if key not in cache:
            print(f"\n[scenario: {key}] ...")
            cache[key] = _run_scenario(lora_rank, ewc_lambda, True, tok, data)
        return cache[key]

    lora_on = run(SC_LORA_EWCON, LORA_RANK, EWC_LAMBDA)
    lora_off = run(SC_LORA_EWCOFF, LORA_RANK, 0.0)
    full_ft = run(SC_FULLFT, 0, 0.0)

    for key, r in ((SC_LORA_EWCON, lora_on), (SC_LORA_EWCOFF, lora_off)):
        state = "EWC ATTIVO" if r["ewc_lambda"] else "EWC DISATTIVATO"
        print(f"\n  Micro-EWC/LoRA rank {LORA_RANK} — [{state}] "
              f"(ewc_lambda {r['ewc_lambda']:g})")
        print(f"    addestrabili: {r['trainable']:,} su {r['total_params']:,} "
              f"({100.0*r['trainable']/r['total_params']:.3f}%)")
        print(f"    Task A loss     : {r['a_loss_before']:.6f} -> {r['a_loss_after']:.6f}"
              f"   (degrado {_pct(r['a_loss_before'], r['a_loss_after']):+.2f}%)")
        print(f"    Task A accuracy : {r['a_acc_before']:.6f} -> {r['a_acc_after']:.6f}"
              f"   (degrado {_pct(r['a_acc_before'], r['a_acc_after']):+.2f}%)")
        print(f"    Task B accuracy : {r['b_acc']:.6f}   loss {r['b_loss']:.6f}")

    d_loss = (_pct(lora_on["a_loss_before"], lora_on["a_loss_after"])
              - _pct(lora_off["a_loss_before"], lora_off["a_loss_after"]))
    d_acc = (_pct(lora_on["a_acc_before"], lora_on["a_acc_after"])
             - _pct(lora_off["a_acc_before"], lora_off["a_acc_after"]))
    print(f"\n  -> EWC on/off, differenza di degrado sulla LOSS del task A: {d_loss:+.2f} punti percentuali")
    print(f"  -> EWC on/off, differenza di degrado sull'ACCURATEZZA del task A: {d_acc:+.2f} punti percentuali")

    print("\n" + "-" * 78)
    print("CRITERIO (d) — IL TASK NUOVO VIENE IMPARATO?")
    print("-" * 78)
    print(f"Confronto a parita' di dati (stesso Task B, stessi {EPOCHS} epoch, "
          f"stesso batch {BATCH_SIZE}, stesso seed).")
    acc_gap_pp = 100.0 * (lora_on["b_acc"] - full_ft["b_acc"])
    loss_gap = lora_on["b_loss"] - full_ft["b_loss"]
    print(f"\n  LoRA (rank {LORA_RANK}, {lora_on['trainable']:,} addestrabili, "
          f"{100.0*lora_on['trainable']/lora_on['total_params']:.3f}% dei parametri):")
    print(f"    Task B accuracy {lora_on['b_acc']:.6f}   loss {lora_on['b_loss']:.6f}")
    print(f"  Fine-tuning completo ({full_ft['trainable']:,} addestrabili, 100% dei parametri):")
    print(f"    Task B accuracy {full_ft['b_acc']:.6f}   loss {full_ft['b_loss']:.6f}")
    print(f"\n  GAP accuratezza (LoRA - full-FT): {acc_gap_pp:+.4f} punti percentuali")
    print(f"  GAP loss assoluta  (LoRA - full-FT): {loss_gap:+.6f}"
          f"   (ratio {lora_on['b_loss']/full_ft['b_loss']:.2f}x)")
    print(f"\n  Ritenzione del task A nel fine-tuning completo (contesto):")
    print(f"    Task A accuracy {full_ft['a_acc_before']:.6f} -> {full_ft['a_acc_after']:.6f}"
          f"   (degrado {_pct(full_ft['a_acc_before'], full_ft['a_acc_after']):+.2f}%)")
    print(f"    Task A loss     {full_ft['a_loss_before']:.6f} -> {full_ft['a_loss_after']:.6f}"
          f"   (degrado {_pct(full_ft['a_loss_before'], full_ft['a_loss_after']):+.2f}%)")

    return {SC_LORA_EWCON: lora_on, SC_LORA_EWCOFF: lora_off, SC_FULLFT: full_ft,
            "acc_gap_pp": acc_gap_pp, "loss_gap": loss_gap,
            "ewc_effect_loss_pp": d_loss, "ewc_effect_acc_pp": d_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="all", choices=["all", "a", "cd", "diag"])
    ap.add_argument("--json", default=None, help="scrivi i risultati in JSON")
    args = ap.parse_args()

    out = {}
    if args.criterion in ("all", "a"):
        out["a"] = criterion_a()
    if args.criterion in ("all", "cd"):
        out["cd"] = criteria_c_d()
    if args.criterion == "diag":
        out["diag"] = diagnostic()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nRisultati JSON scritti in {args.json}")



def diagnostic():
    """
    Diagnostica: il gap del criterio (d) e' un limite di capacita' o di
    ottimizzazione (underfitting)?

    Stesso protocollo A->B di (c)/(d), ma con una griglia di learning rate per
    il ramo LoRA. Se il gap si chiude aumentando il learning rate, il limite
    e' di ottimizzazione; se persiste, e' di capacita'. Non e' un criterio del
    protocollo: serve a decidere dove investire.
    """
    tok, text_a, text_b, _, _ = build_tokenizer()
    a_in, a_tg = make_dataset(text_a, tok)
    b_in, b_tg = make_dataset(text_b, tok)
    a_tr, a_tg_tr, a_va, a_tg_va = split_train_val(a_in, a_tg)
    b_tr, b_tg_tr, b_va, b_tg_va = split_train_val(b_in, b_tg)
    data = (a_tr, a_tg_tr, a_va, a_tg_va, b_tr, b_tg_tr, b_va, b_tg_va)

    print("=" * 78)
    print("DIAGNOSTICA — il gap di (d) e' capacita' o ottimizzazione?")
    print("=" * 78)
    print(f"{EPOCHS} epoch per task, replay 80/20, seed {SEED}\n")

    def run(label, lora_rank, lr, ewc_lambda):
        model = (build_lora_model(len(tok.vocabulary)) if lora_rank
                 else build_full_model(len(tok.vocabulary)))
        trainer = Trainer(model=model, tokenizer=tok, loss=CrossEntropyLoss(),
                          optimizer=Adam(learning_rate=lr), seed=SEED,
                          ewc_lambda=ewc_lambda)
        a_tr_d, a_tg_d, a_va_d, a_tg_va_d = data[:4]
        b_tr_d, b_tg_d, b_va_d, b_tg_va_d = data[4:]
        trainer.train(a_tr_d, a_tg_d, epochs=EPOCHS, batch_size=BATCH_SIZE)
        trainer.compute_fisher_matrix((a_tr_d, a_tg_d), num_samples=FISHER_SAMPLES)
        a_l0 = trainer.evaluate(a_va_d, a_tg_va_d, batch_size=BATCH_SIZE)
        a_a0 = accuracy(trainer, a_va_d, a_tg_va_d)
        if lora_rank:
            model.add_task_adapter(warm_start=False)
        trainer.set_replay_buffer([(a_tr_d[i], a_tg_d[i]) for i in range(len(a_tr_d))])
        trainer.train(b_tr_d, b_tg_d, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      replay_buffer=trainer.replay_buffer)
        a_l1 = trainer.evaluate(a_va_d, a_tg_va_d, batch_size=BATCH_SIZE)
        a_a1 = accuracy(trainer, a_va_d, a_tg_va_d)
        b_l = trainer.evaluate(b_va_d, b_tg_va_d, batch_size=BATCH_SIZE)
        b_a = accuracy(trainer, b_va_d, b_tg_va_d)
        print(f"\n{label}")
        print(f"  addestrabili: {model.trainable_parameter_count():,}")
        print(f"  Task A: loss {a_l0:.4f} -> {a_l1:.4f}  ({_pct(a_l0,a_l1):+.2f}%)   "
              f"acc {a_a0:.4f} -> {a_a1:.4f}  ({_pct(a_a0,a_a1):+.2f}%)")
        print(f"  Task B: acc {b_a:.4f}   loss {b_l:.4f}")
        return dict(label=label, a_loss_before=a_l0, a_loss_after=a_l1,
                    a_acc_before=a_a0, a_acc_after=a_a1, b_loss=b_l, b_acc=b_a)

    out = [run(f"LoRA rank {LORA_RANK}, lr {lr:g}, ewc_lambda {EWC_LAMBDA:g}",
               LORA_RANK, lr, EWC_LAMBDA) for lr in (LEARNING_RATE, 0.1)]
    out.append(run(f"Full fine-tuning, lr {LEARNING_RATE:g}", 0, LEARNING_RATE, 0.0))

    print("\n" + "-" * 78)
    print("Confronto sul task B")
    print("-" * 78)
    for r in out:
        print(f"  {r['label']:<46} acc {r['b_acc']:.4f}  loss {r['b_loss']:.4f}")
    best_lora = max((r for r in out if "LoRA" in r["label"]), key=lambda r: r["b_acc"])
    full = [r for r in out if "Full" in r["label"]][0]
    print(f"\n  miglior LoRA della griglia: acc {best_lora['b_acc']:.4f}")
    print(f"  full fine-tuning          : acc {full['b_acc']:.4f}")
    print(f"  gap residuo: {100.0*(best_lora['b_acc']-full['b_acc']):+.4f} punti percentuali")
    return out

if __name__ == "__main__":
    main()
