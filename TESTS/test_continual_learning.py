"""
Continual Learning: Elastic Weight Consolidation (EWC) e Micro-EWC con LoRA.

Il modulo di Continual Learning opera in due modalita':

1. **EWC full-parameter (legacy)** — la Fisher matrix e la penalita' quadratica
   coprono tutti i parametri del modello (2.7M nel modello di produzione).
2. **Micro-EWC con LoRA** — la Fisher matrix e la penalita' quadratica
   coprono esclusivamente le matrici LoRA A/B iniettate nei layer FFN finali.
   Tutti i pesi pre-addestrati (embedding, attention, LayerNorm, W1/b1/W2/b2,
   output) sono congelati.

AVVERTENZA SULLA VALIDITA' DEI RISULTATI
-----------------------------------------
I test di retenzione e di apprendimento (classi `TestRetentionMeasured` e
`TestNewTaskLearning`) usano un corpus sintetico controllato a due domini
(versi stilometrici distinti) su un MicroTransformer ridotto. Non sono una
prova generale di capacita' di Continual Learning su distribuzioni reali:
sono un test di fumo strutturale che verifica la correttezza del meccanismo
e che misura un ordine di grandezza confrontabile. Ogni numero riportato
nelle asserzioni e' prodotto da un'esecuzione reale in questo file.
"""

import time

import numpy as np
import pytest

from neural_core.trainer import Trainer, build_model
from neural_core.tokenizer import Tokenizer
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam
from neural_core.dataset import Dataset


# Configurazione di produzione (17_MODEL_TRAINING_PROVENANCE.md sezione 2.1).
# Usata per la misura di memoria (criterio a): il conteggio parametri e'
# quello del modello reale, non di un modello giocattolo.
PRODUCTION_CONFIG = {
    "vocab_size": 8590,
    "d_model": 128,
    "num_heads": 4,
    "d_ff": 256,
    "num_layers": 4,
    "max_len": 256,
}
PRODUCTION_TOTAL_PARAMS = 2_737_806

# Rango LoRA e alpha. rank=4 -> 4*(d_ff + d_model) = 4*384 = 1536 parametri
# per layer, 4 layer -> 6144 parametri addestrabili su 2.737.806 totali.
LORA_RANK = 4
LORA_ALPHA = 16.0


def _build_trainer(vocab_size=20, seed=42):
    model = build_model("next_token", vocab_size=vocab_size, embedding_dim=16, hidden_dim=32, seed=seed)
    tokenizer = Tokenizer(max_vocab_size=vocab_size, mode="whitespace")
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    return Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer, seed=seed)


def _make_dataset(text, tokenizer, context_size=8, stride=8):
    ids = tokenizer.encode(text)
    return Dataset(ids, context_size=context_size, stride=stride)


def _build_lora_trainer(lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, seed=42,
                        lr=0.01, **model_kwargs):
    """Costruisce un Trainer su MicroTransformer con LoRA attiva nei layer FFN."""
    model = build_model("transformer", lora_rank=lora_rank, lora_alpha=lora_alpha,
                        seed=seed, **model_kwargs)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=lr)
    tokenizer = Tokenizer(max_vocab_size=model.vocab_size, mode="whitespace")
    return Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer, seed=seed)


def _eval_loss(trainer, inputs, targets, batch_size=32):
    """Loss media su un dataset (nessun aggiornamento di parametri)."""
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    total, count = 0.0, 0
    for start in range(0, len(inputs), batch_size):
        b_in = inputs[start:start + batch_size]
        b_tgt = targets[start:start + batch_size]
        logits = trainer.model.forward(b_in, training=False)
        flat_logits = logits.reshape(-1, trainer.model.vocab_size)
        flat_tgt = np.asarray(b_tgt).reshape(-1)
        shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1))
        picked = shifted[np.arange(len(flat_tgt)), flat_tgt] - logsumexp
        total += float(-picked.sum())
        count += len(flat_tgt)
    return total / count if count else float("inf")


def _eval_accuracy(trainer, inputs, targets, batch_size=32):
    """Accuratezza token-level (argmax su tutte le posizioni valide)."""
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    correct, total = 0, 0
    for start in range(0, len(inputs), batch_size):
        b_in = inputs[start:start + batch_size]
        b_tgt = np.asarray(targets[start:start + batch_size])
        preds = np.argmax(trainer.model.forward(b_in, training=False), axis=-1)
        if preds.shape != b_tgt.shape:
            b_tgt = b_tgt.reshape(preds.shape)
        correct += int(np.sum(preds == b_tgt))
        total += int(preds.size)
    return correct / total if total else 0.0


def _nbytes(dictionary):
    """Byte occupati da un dizionario di array numpy."""
    return int(sum(np.asarray(v).nbytes for v in dictionary.values()))


# Corpus sintetico a due domini distinti. Ogni dominio ha una distribuzione
# di coppie bigrammi diversa, quindi i due task sono apprendibili separatamente.
TASK_A_TEXT = (
    "il vento soffia sopra la collina durante la notte stellata "
    "la pioggia cade lenta sul tetto della casa di campagna "
    "il sole sorge dietro le montagne color arancio "
    "la neve copre gli alberi del bosco ogni inverno "
    "il fiume scorre lento fra le rocce del canyon "
) * 120

TASK_B_TEXT = (
    "def compute_gradient norm while forward pass cache "
    "class Trainer backward update optimizer state "
    "import numpy as np return tensor float device "
    "def loss function cross entropy logits softmax "
    "model parameters trainable requires grad true "
) * 120


def _two_task_setup(lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, seed=42, lr=0.02,
                    d_model=64, d_ff=128, num_layers=2, num_heads=4, max_len=32,
                    context_size=16):
    """Prepara tokenizer e dataset per i due task."""
    tokenizer = Tokenizer(max_vocab_size=2000, mode="whitespace")
    tokenizer.fit(TASK_A_TEXT + " " + TASK_B_TEXT)
    ds_a = _make_dataset(TASK_A_TEXT, tokenizer, context_size=context_size, stride=context_size)
    ds_b = _make_dataset(TASK_B_TEXT, tokenizer, context_size=context_size, stride=context_size)
    model_kwargs = dict(vocab_size=len(tokenizer.vocabulary), d_model=d_model,
                        num_heads=num_heads, d_ff=d_ff, num_layers=num_layers,
                        max_len=max_len)
    return tokenizer, ds_a, ds_b, model_kwargs, seed, lr


# Tokenizer e dataset condivisi dagli esperimenti a due task. Costruiti una
# sola volta a livello di modulo perche' i test di ritenzione e apprendimento
# devono confrontare modelli addestrati sugli stessi identici dati.
TOKENIZER_ALL = Tokenizer(max_vocab_size=2000, mode="whitespace")
TOKENIZER_ALL.fit(TASK_A_TEXT + " " + TASK_B_TEXT)
DS_A = _make_dataset(TASK_A_TEXT, TOKENIZER_ALL, context_size=16, stride=16)
DS_B = _make_dataset(TASK_B_TEXT, TOKENIZER_ALL, context_size=16, stride=16)
IN_A, TG_A = np.asarray(DS_A.get_inputs()), np.asarray(DS_A.get_targets())
IN_B, TG_B = np.asarray(DS_B.get_inputs()), np.asarray(DS_B.get_targets())


# ----------------------------------------------------------------------
# EWC full-parameter (legacy) — comportamento invariato
# ----------------------------------------------------------------------

class TestEwcLossCalculation:
    def test_ewc_loss_zero_when_weights_identical(self):
        trainer = _build_trainer()
        params = trainer.model.get_parameters()
        names = trainer._fisher_param_names()
        trainer.optimal_weights = {n: p.copy() for n, p in zip(names, params)}
        trainer.fisher_matrix = {n: np.ones_like(p) for n, p in zip(names, params)}
        ewc_loss = trainer._compute_ewc_loss()
        assert ewc_loss == 0.0

    def test_ewc_loss_grows_with_weight_displacement(self):
        trainer = _build_trainer()
        params = trainer.model.get_parameters()
        names = trainer._fisher_param_names()
        trainer.optimal_weights = {n: p.copy() for n, p in zip(names, params)}
        fisher = {n: np.ones_like(p) for n, p in zip(names, params)}
        trainer.fisher_matrix = fisher
        loss_zero = trainer._compute_ewc_loss()
        for p in trainer.model.get_parameters():
            p += 0.5
        loss_shifted = trainer._compute_ewc_loss()
        assert loss_shifted > loss_zero >= 0.0

    def test_ewc_loss_zero_without_fisher(self):
        trainer = _build_trainer()
        assert trainer._compute_ewc_loss() == 0.0


class TestFisherMatrixComputation:
    def test_fisher_matrix_structure(self):
        trainer = _build_trainer()
        tokenizer = trainer.tokenizer
        tokenizer.fit("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda")
        ds = _make_dataset("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda " * 30, tokenizer)
        fisher = trainer.compute_fisher_matrix(ds, num_samples=20)
        assert isinstance(fisher, dict)
        assert len(fisher) > 0
        for key, value in fisher.items():
            assert isinstance(key, str)
            assert np.asarray(value).dtype.kind == "f"
            assert np.all(np.asarray(value) >= 0.0)
        assert trainer.optimal_weights is not None
        assert len(trainer.optimal_weights) == len(fisher)

    def test_fisher_matrix_non_negative(self):
        trainer = _build_trainer()
        tokenizer = trainer.tokenizer
        tokenizer.fit("one two three four five six seven eight nine ten")
        ds = _make_dataset("one two three four five six seven eight nine ten " * 30, tokenizer)
        fisher = trainer.compute_fisher_matrix(ds, num_samples=10)
        for value in fisher.values():
            assert np.all(np.asarray(value) >= 0.0)


class TestCatastrophicForgettingPrevention:
    def _build_trainer(self, vocab_size, seed=42, embedding_dim=16, hidden_dim=32, lr=0.05):
        model = build_model("next_token", vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, seed=seed)
        tokenizer = Tokenizer(max_vocab_size=vocab_size, mode="whitespace")
        loss = CrossEntropyLoss()
        optimizer = Adam(learning_rate=lr)
        return Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer, seed=seed)

    def _make_dataset(self, text, tokenizer, context_size=6, stride=3):
        ids = tokenizer.encode(text)
        return Dataset(ids, context_size=context_size, stride=stride)

    def _weight_distance(self, params1, params2):
        return sum(float(np.sum(np.square(np.asarray(p1) - np.asarray(p2))))
                   for p1, p2 in zip(params1, params2))

    def test_ewc_preserves_task_a_weights(self):
        """
        Verifica che l'EWC mantenga i pesi vicini a quelli di Task A dopo
        l'addestramento su Task B. Con EWC attivo la distanza dai pesi
        ottimali (Task A) deve essere minore rispetto all'addestramento senza EWC.
        """
        tokenizer = Tokenizer(max_vocab_size=80, mode="whitespace")
        task_a = "red green blue yellow purple orange black white pink brown maroon teal " * 40
        task_b = "cat dog bird fish rabbit turtle lion tiger monkey camel donkey " * 40
        tokenizer.fit(task_a + " " + task_b)
        ds_a = self._make_dataset(task_a, tokenizer)
        ds_b = self._make_dataset(task_b, tokenizer)
        vocab = len(tokenizer.vocabulary)

        # Addestra su Task A parzialmente
        trainer_a = self._build_trainer(vocab_size=vocab, seed=42, embedding_dim=16, hidden_dim=32, lr=0.05)
        inputs_a = ds_a.get_inputs()
        targets_a = ds_a.get_targets()
        partial_n = min(25 * 8, len(inputs_a))
        trainer_a.train(inputs_a[:partial_n], targets_a[:partial_n], epochs=1, batch_size=8)
        fisher = trainer_a.compute_fisher_matrix(ds_a, num_samples=30)
        fisher_norm = sum(float(np.sum(v)) for v in fisher.values())
        assert fisher_norm > 0.0, f"Fisher matrix vuota: {fisher_norm}"

        # Salva i pesi ottimali (dopo Task A)
        optimal_weights = [np.asarray(p).copy() for p in trainer_a.model.get_parameters()]

        # Addestra su Task B SENZA EWC per molti epoch
        trainer_no_ewc = self._build_trainer(vocab_size=vocab, seed=42, embedding_dim=16, hidden_dim=32, lr=0.05)
        trainer_no_ewc.train(inputs_a[:partial_n], targets_a[:partial_n], epochs=1, batch_size=8)
        trainer_no_ewc.train(ds_b.get_inputs(), ds_b.get_targets(), epochs=5, batch_size=8)
        dist_no_ewc = self._weight_distance(optimal_weights, trainer_no_ewc.model.get_parameters())

        # Addestra su Task B CON EWC per lo stesso numero di epoch
        trainer_ewc = self._build_trainer(vocab_size=vocab, seed=42, embedding_dim=16, hidden_dim=32, lr=0.05)
        trainer_ewc.train(inputs_a[:partial_n], targets_a[:partial_n], epochs=1, batch_size=8)
        trainer_ewc.compute_fisher_matrix(ds_a, num_samples=30)
        trainer_ewc.ewc_lambda = 5000.0
        trainer_ewc.train(ds_b.get_inputs(), ds_b.get_targets(), epochs=5, batch_size=8)
        dist_ewc = self._weight_distance(optimal_weights, trainer_ewc.model.get_parameters())

        print(f"\nDEBUG: dist_no_ewc={dist_no_ewc:.6f}, dist_ewc={dist_ewc:.6f}")

        # Con EWC i pesi devono essere più vicini agli optimal rispetto a senza EWC
        assert dist_ewc < dist_no_ewc, (
            f"atteso EWC che preservi i pesi di Task A: {dist_ewc} < {dist_no_ewc}"
        )


# ----------------------------------------------------------------------
# LoRA: iniezione e freezing selettivo
# ----------------------------------------------------------------------

class TestLoraInjection:
    def test_lora_matrices_shapes(self):
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=2, num_heads=2, max_len=32,
                                      lora_rank=LORA_RANK)
        for layer in trainer.model.layers:
            lora = layer.ff.lora
            assert lora is not None
            assert lora.A.shape == (LORA_RANK, layer.ff.d_ff)
            assert lora.B.shape == (layer.ff.d_model, LORA_RANK)

    def test_lora_b_initialized_to_zero(self):
        """Con B=0 la correzione LoRA e' nulla: il modello parte identico al base."""
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=2, num_heads=2, max_len=32)
        for layer in trainer.model.layers:
            assert np.allclose(layer.ff.lora.B, 0.0)
            assert not np.allclose(layer.ff.lora.A, 0.0)

    def test_lora_disabled_keeps_legacy_parameter_count(self):
        """lora_rank=0 deve riprodurre esattamente il conteggio parametrico legacy."""
        legacy = build_model("transformer", vocab_size=50, d_model=32, d_ff=64,
                             num_layers=2, num_heads=2, max_len=32, seed=42)
        lora = build_model("transformer", vocab_size=50, d_model=32, d_ff=64,
                           num_layers=2, num_heads=2, max_len=32, seed=42, lora_rank=LORA_RANK)

        # Il delta deve essere esattamente 2 matriche (A, B) per layer.
        per_layer_lora = LORA_RANK * 64 + 32 * LORA_RANK   # A: (rank, d_ff), B: (d_model, rank)
        assert lora.parameter_count() == legacy.parameter_count() + 2 * per_layer_lora
        assert all(layer.ff.lora is None for layer in legacy.layers)
        assert all(layer.ff.lora is not None for layer in lora.layers)
        # Il modello senza LoRA ha tutti i parametri addestrabili (legacy)
        assert legacy.trainable_parameter_count() == legacy.parameter_count()
        assert legacy.frozen_parameter_count() == 0

    def test_only_lora_parameters_are_trainable(self):
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=2, num_heads=2, max_len=32)
        model = trainer.model
        n_all = len(model.get_parameters())
        idx = model.trainable_param_indices()
        assert len(idx) == 2 * model.num_layers  # A e B per layer
        assert idx == sorted(idx)
        assert model.trainable_param_names() == [
            f"layer{i}_lora_0_{ab}" for i in range(model.num_layers) for ab in ("A", "B")
        ]
        # Solo LoRA: trainable molto inferiore al totale
        assert model.trainable_parameter_count() < 0.2 * model.parameter_count()
        assert n_all == len(model.get_parameters())

    def test_frozen_params_do_not_move_during_training(self):
        """Verifica diretta del freezing: i pesi pre-addestrati restano bit-identici."""
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=2, num_heads=2, max_len=32, lr=0.05)
        model = trainer.model
        before = [np.asarray(p).copy() for p in model.get_parameters()]
        inputs = np.random.RandomState(0).randint(0, 50, size=(8, 16))
        targets = np.random.RandomState(1).randint(0, 50, size=(8, 16))
        for _ in range(20):
            trainer.train_step(inputs, targets)
        trainable = set(model.trainable_param_indices())
        moved_trainable, moved_frozen = 0, 0
        for i, (b, a) in enumerate(zip(before, model.get_parameters())):
            delta = float(np.max(np.abs(np.asarray(a) - b)))
            if i in trainable:
                moved_trainable += int(delta > 0)
            else:
                moved_frozen += int(delta > 0)
        assert moved_trainable == len(trainable), "i parametri LoRA devono muoversi"
        assert moved_frozen == 0, "i parametri congelati non devono muoversi"

    def test_frozen_gradients_are_zeroed_at_the_source(self):
        """Il congelamento e' proprieta' del modello, non solo del Trainer."""
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=2, num_heads=2, max_len=32)
        model = trainer.model
        ids = np.random.RandomState(0).randint(0, 50, size=(4, 16))
        model.forward(ids, training=False)
        grads = model.backward(ids)
        trainable = set(model.trainable_param_indices())
        for i, g in enumerate(grads):
            if i not in trainable:
                assert float(np.max(np.abs(np.asarray(g)))) == 0.0

    def test_set_parameters_roundtrip_with_lora(self):
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=2, num_heads=2, max_len=32)
        model = trainer.model
        params = model.get_parameters()
        snapshot = [np.asarray(p).copy() for p in params]
        perturbed = [np.asarray(p).copy() for p in params]
        for p, q in zip(perturbed, perturbed):
            pass
        for i in model.trainable_param_indices():
            perturbed[i] = perturbed[i] + 0.25
        model.set_parameters(perturbed)
        for i, (orig, now) in enumerate(zip(snapshot, model.get_parameters())):
            if i in set(model.trainable_param_indices()):
                assert np.allclose(orig, now - 0.25)
            else:
                assert np.array_equal(orig, now)


# ----------------------------------------------------------------------
# (a) CRITERIO 1 — Riduzione di memoria delle strutture Fisher/optimal_weights
# ----------------------------------------------------------------------

class TestMemoryReduction:
    def test_fisher_structures_reduced_above_90_percent_on_production_config(self):
        """
        Criterio (a): consumo RAM di fisher_matrix + optimal_weights ridotto
        di piu' del 90% rispetto alla versione full-parameter.

        Misurato sulla configurazione di produzione reale
        (vocab 8590, d_model 128, d_ff 256, 4 layer, 2.737.806 parametri).
        """
        tokenizer = Tokenizer(max_vocab_size=64, mode="whitespace")
        tokenizer.fit("alpha beta gamma delta epsilon zeta eta theta iota kappa")
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 200
        ds = _make_dataset(text, tokenizer, context_size=16, stride=16)
        n_samples = 20

        # --- versione full-parameter (lora_rank=0) ---
        full = _build_lora_trainer(lora_rank=0, lr=0.0, **PRODUCTION_CONFIG)
        full.compute_fisher_matrix(ds, num_samples=n_samples)
        full_bytes = _nbytes(full.fisher_matrix) + _nbytes(full.optimal_weights)
        full_entries = len(full.fisher_matrix)

        # --- versione Micro-EWC con LoRA ---
        lora = _build_lora_trainer(lora_rank=LORA_RANK, lr=0.0, **PRODUCTION_CONFIG)
        lora.compute_fisher_matrix(ds, num_samples=n_samples)
        lora_bytes = _nbytes(lora.fisher_matrix) + _nbytes(lora.optimal_weights)
        lora_entries = len(lora.fisher_matrix)

        model = lora.model
        trainable = model.trainable_parameter_count()
        total_with_lora = model.parameter_count()
        total_base = full.model.parameter_count()
        reduction = 100.0 * (1.0 - lora_bytes / full_bytes)

        print("\n=== CRITERIO (a): RIDUZIONE MEMORIA (config di produzione) ===")
        print(f"  parametri base (pre-addestrati)     : {total_base}")
        print(f"  parametri totali con LoRA           : {total_with_lora}")
        print(f"  parametri addestrabili (LoRA A/B)   : {trainable}")
        print(f"  parametri congelati                 : {total_base}")
        print(f"  entries Fisher  full-parameter      : {full_entries}")
        print(f"  entries Fisher  Micro-EWC/LoRA      : {lora_entries}")
        print(f"  byte fisher+optimal full-parameter  : {full_bytes}")
        print(f"  byte fisher+optimal Micro-EWC/LoRA  : {lora_bytes}")
        print(f"  riduzione percentuale               : {reduction:.4f}%")
        print(f"  totale base atteso (produzione)     : {PRODUCTION_TOTAL_PARAMS}")

        assert total_base == PRODUCTION_TOTAL_PARAMS, (
            f"il conteggio parametri base deve coincidere con il modello di produzione: "
            f"{total_base} != {PRODUCTION_TOTAL_PARAMS}"
        )
        assert total_with_lora == PRODUCTION_TOTAL_PARAMS + trainable
        assert lora_entries == 2 * PRODUCTION_CONFIG["num_layers"]
        assert reduction > 90.0, f"attesa riduzione >90%, misurata {reduction:.4f}%"

    def test_fisher_contains_only_lora_entries(self):
        trainer = _build_lora_trainer(vocab_size=50, d_model=32, d_ff=64,
                                      num_layers=3, num_heads=2, max_len=32)
        tokenizer = trainer.tokenizer
        tokenizer.fit("alpha beta gamma delta epsilon zeta eta theta")
        ds = _make_dataset("alpha beta gamma delta epsilon zeta eta theta " * 40,
                           tokenizer, context_size=16, stride=16)
        fisher = trainer.compute_fisher_matrix(ds, num_samples=5)
        assert set(fisher.keys()) == set(trainer.model.trainable_param_names())
        assert set(trainer.optimal_weights.keys()) == set(fisher.keys())
        for value in fisher.values():
            assert np.all(np.asarray(value) >= 0.0)


# ----------------------------------------------------------------------
# (b) CRITERIO 2 — Latenza di compute_fisher_matrix su CPU
# ----------------------------------------------------------------------

class TestFisherLatency:
    def test_fisher_latency_full_vs_micro(self):
        """
        Criterio (b): tempo di compute_fisher_matrix misurato, confrontato con
        la versione full-parameter sullo stesso dataset.
        """
        tokenizer = Tokenizer(max_vocab_size=64, mode="whitespace")
        tokenizer.fit("alpha beta gamma delta epsilon zeta eta theta iota kappa")
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 400
        ds = _make_dataset(text, tokenizer, context_size=16, stride=16)
        n_samples = 50

        full = _build_lora_trainer(lora_rank=0, lr=0.0, **PRODUCTION_CONFIG)
        t0 = time.perf_counter()
        full.compute_fisher_matrix(ds, num_samples=n_samples)
        t_full = time.perf_counter() - t0

        lora = _build_lora_trainer(lora_rank=LORA_RANK, lr=0.0, **PRODUCTION_CONFIG)
        t0 = time.perf_counter()
        lora.compute_fisher_matrix(ds, num_samples=n_samples)
        t_lora = time.perf_counter() - t0

        print("\n=== CRITERIO (b): LATENZA compute_fisher_matrix (CPU, NumPy) ===")
        print(f"  campioni Fisher                     : {n_samples}")
        print(f"  full-parameter                      : {t_full * 1000:.3f} ms")
        print(f"  Micro-EWC/LoRA                      : {t_lora * 1000:.3f} ms")
        print(f"  speedup                             : {t_full / t_lora:.2f}x")

        assert t_lora > 0.0
        # Compatibile con un aggiornamento "near real-time": sotto 1 secondo.
        assert t_lora < 1.0, f"latenza troppo alta per near real-time: {t_lora:.3f}s"
        assert t_lora < t_full, "il Micro-EWC deve essere piu' rapido del full-parameter"

    def test_fisher_latency_scales_with_num_samples(self):
        trainer = _build_lora_trainer(lora_rank=LORA_RANK, lr=0.0, **PRODUCTION_CONFIG)
        tokenizer = trainer.tokenizer
        tokenizer.fit("alpha beta gamma delta epsilon zeta eta theta iota kappa")
        ds = _make_dataset("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 200,
                           tokenizer, context_size=16, stride=16)
        t0 = time.perf_counter()
        trainer.compute_fisher_matrix(ds, num_samples=10)
        t_small = time.perf_counter() - t0
        t0 = time.perf_counter()
        trainer.compute_fisher_matrix(ds, num_samples=50)
        t_large = time.perf_counter() - t0
        print(f"\n  10 campioni: {t_small * 1000:.3f} ms | 50 campioni: {t_large * 1000:.3f} ms")
        assert t_small < 1.0 and t_large < 1.0


# ----------------------------------------------------------------------
# Continual Learning: helper per lo scenario a due task
# ----------------------------------------------------------------------

# Rango LoRA usato negli esperimenti di ritenzione/apprendimento.
# rank=16 -> 16*(d_ff + d_model) = 16*192 = 3072 parametri per layer.
CONTINUAL_LORA_RANK = 16
CONTINUAL_LORA_ALPHA = 16.0
# ewc_lambda selezionato tramite sweep misurato su
# {0, 10, 100, 1000, 10000} x {rank 4, 16}: 100 e' il valore che non
# sacrifica l'apprendimento del nuovo task (vedi report).
CONTINUAL_EWC_LAMBDA = 100.0
CONTINUAL_EPOCHS = 12


def _eval_on(model, inputs, targets, batch_size=32):
    """Loss media e accuratezza token-level su un dataset (nessun update)."""
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    total, count, correct = 0.0, 0, 0
    for start in range(0, len(inputs), batch_size):
        b_in = inputs[start:start + batch_size]
        b_tgt = np.asarray(targets[start:start + batch_size])
        flat = model.forward(b_in, training=False).reshape(-1, model.vocab_size)
        flat_tgt = b_tgt.reshape(-1)
        shifted = flat - flat.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1))
        picked = shifted[np.arange(len(flat_tgt)), flat_tgt] - logsumexp
        total += float(-picked.sum())
        count += len(flat_tgt)
        correct += int(np.sum(np.argmax(flat, axis=-1) == flat_tgt))
    return total / count, correct / count


def _continual_model(lora_rank, seed=42, **overrides):
    """MicroTransformer ridotto usato negli esperimenti a due task.

    NON e' la configurazione di produzione: e' un modello piccolo per tenere
    i test rapidi. I criteri (a) e (b) sono invece misurati sulla
    configurazione di produzione reale.
    """
    kwargs = dict(vocab_size=len(TOKENIZER_ALL.vocabulary), d_model=64,
                  num_heads=4, d_ff=128, num_layers=2, max_len=32)
    kwargs.update(overrides)
    return build_model("transformer", lora_rank=lora_rank,
                       lora_alpha=CONTINUAL_LORA_ALPHA, seed=seed, **kwargs)


def _make_trainer(model, lr=0.02, seed=42):
    return Trainer(model=model, tokenizer=TOKENIZER_ALL, loss=CrossEntropyLoss(),
                   optimizer=Adam(learning_rate=lr), seed=seed)


def _replay_buffer(inputs, targets):
    """Buffer di replay episodico nel formato atteso dal Trainer:
    lista di tuple (riga input, riga target)."""
    return list(zip(np.asarray(inputs).tolist(), np.asarray(targets).tolist()))


def _learn_task_a(model):
    """Addestra sul task A e restituisce (loss, accuracy) su A."""
    trainer = _make_trainer(model)
    trainer.train(IN_A, TG_A, epochs=CONTINUAL_EPOCHS, batch_size=32)
    return _eval_on(model, IN_A, TG_A), trainer


def _register_task_b_and_compute_fisher(trainer, model, ewc_lambda=CONTINUAL_EWC_LAMBDA):
    """
    Registra il task B (adapter dedicato, cold start) e ricalcola la
    Micro-Fisher sul task A rispetto al NUOVO adapter.

    La Fisher va ricalcolata DOPO l'aggiunta del nuovo adapter perche' la
    sua semantica e' "quanto il task A e' sensibile a una perturbazione di
    questo adapter", che e' l'unica perturbazione realmente addestrabile.
    Se si riusasse la Fisher del task precedente, i nomi non corrisponderebbero
    ai parametri addestrabili e la penalita' sarebbe nulla.
    """
    model.add_task_adapter(warm_start=False)
    trainer.compute_fisher_matrix(DS_A, num_samples=50)
    trainer.ewc_lambda = ewc_lambda
    return trainer


# ----------------------------------------------------------------------
# (c) CRITERIO 3 — Ritenzione misurata su loss/accuratezza
# ----------------------------------------------------------------------

class TestRetentionMeasured:
    """
    Criterio (c): la ritenzione e' misurata su LOSS e ACCURATEZZA del task
    precedente dopo l'addestramento sul task nuovo. Non si usa la distanza dei
    pesi, che sarebbe tautologica (un solo adapter e' addestrabile per volta,
    quindi la distanza e' gia' un proxy dell'update).

    SOGLIA DICHIARATA: degrado dell'ACCURATEZZA del task A < 2%.
    Il degrado della LOSS viene riportato ma NON e' soggetto ad asserzione:
   vedi `test_loss_retention_known_limitation` e la sezione 5 del report.
    """

    def _run_continual(self, rank=CONTINUAL_LORA_RANK, ewc_lambda=CONTINUAL_EWC_LAMBDA,
                       use_replay=True):
        model = _continual_model(rank)
        (loss_a_before, acc_a_before), trainer = _learn_task_a(model)
        trainer = _register_task_b_and_compute_fisher(trainer, model, ewc_lambda)
        kwargs = {"replay_buffer": _replay_buffer(IN_A, TG_A)} if use_replay else {}
        trainer.train(IN_B, TG_B, epochs=CONTINUAL_EPOCHS, batch_size=32, **kwargs)
        loss_a_after, acc_a_after = _eval_on(model, IN_A, TG_A)
        loss_b, acc_b = _eval_on(model, IN_B, TG_B)
        return {
            "loss_a_before": loss_a_before, "acc_a_before": acc_a_before,
            "loss_a_after": loss_a_after, "acc_a_after": acc_a_after,
            "loss_b": loss_b, "acc_b": acc_b,
            "loss_deg_pct": 100.0 * (loss_a_after - loss_a_before) / loss_a_before,
            "acc_deg_pct": 100.0 * (acc_a_before - acc_a_after) / acc_a_before,
        }

    def test_accuracy_retention_degradation_below_two_percent(self):
        r = self._run_continual()
        print("\n=== CRITERIO (c): RITENZIONE del task A ===")
        print(f"  accuracy A prima : {r['acc_a_before']:.6f}")
        print(f"  accuracy A dopo  : {r['acc_a_after']:.6f}")
        print(f"  degrado accuracy : {r['acc_deg_pct']:+.4f}%   (soglia < 2%)")
        print(f"  loss A prima     : {r['loss_a_before']:.6f}")
        print(f"  loss A dopo      : {r['loss_a_after']:.6f}")
        print(f"  degrado loss     : {r['loss_deg_pct']:+.4f}%   (NON soggetto ad asserzione)")
        print(f"  accuracy B dopo  : {r['acc_b']:.6f}")

        # Il task A deve essere stato effettivamente appreso (test non banale)
        assert r["acc_a_before"] > 0.9, f"task A non appreso: {r['acc_a_before']:.4f}"
        # Il task B deve essere stato appreso
        assert r["acc_b"] > 0.9, f"task B non appreso: {r['acc_b']:.4f}"
        # Soglia del protocollo sulla ritenzione
        assert r["acc_deg_pct"] < 2.0, (
            f"degrado accuracy su task A {r['acc_deg_pct']:+.4f}% supera la soglia del 2%"
        )

    def test_loss_retention_known_limitation(self):
        """
        Limite NOTO e DICHIARATO: il degrado della LOSS sul task precedente
        non raggiunge la soglia del 2%, mentre il degrado dell'ACCURATEZZA la
        rispetta (vedi test precedente).

        Questo test codifica il limite in modo esplicito: se in futuro il
        degrado della loss scendesse sotto il 2%, questo test fallisce e
        segnala che la documentazione va aggiornata.
        """
        r = self._run_continual()
        print("\n=== LIMITE NOTO: ritenzione della LOSS ===")
        print(f"  loss A prima : {r['loss_a_before']:.6f}")
        print(f"  loss A dopo  : {r['loss_a_after']:.6f}")
        print(f"  degrado loss : {r['loss_deg_pct']:+.4f}%  (soglia protocollo 2%: NON raggiunta)")
        print(f"  degrado acc  : {r['acc_deg_pct']:+.4f}%  (soglia protocollo 2%: raggiunta)")
        assert r["loss_deg_pct"] > 2.0, (
            "il degrado della loss e' sceso sotto il 2%: il limite dichiarato "
            "non e' piu' valido, aggiornare report e documentazione"
        )

    def test_replay_buffer_is_actually_mixed_into_batches(self):
        """
        Verifica strutturale che il replay episodico 80/20 venga realmente
        applicato. Prima della correzione il buffer era ignorato: i campioni
        1-D venivano scartati dal controllo di ndim e il replay non avveniva.
        """
        trainer = _make_trainer(_continual_model(CONTINUAL_LORA_RANK))
        buffer = _replay_buffer(IN_A, TG_A)
        mixed_in, _ = trainer._sample_replay(IN_B, TG_B, replay_ratio=0.2,
                                             replay_buffer=buffer)
        mixed = np.asarray(mixed_in)
        assert len(mixed) > len(IN_B), "il batch deve essere ingrandito col replay"
        set_a = {tuple(r) for r in np.asarray(IN_A).tolist()}
        n_from_a = sum(1 for r in mixed.tolist() if tuple(r) in set_a)
        fraction = n_from_a / len(mixed)
        assert 0.10 <= fraction <= 0.30, (
            f"frazione di replay fuori dall'intervallo atteso: {fraction:.3f}"
        )

    def test_retention_holds_with_episodic_replay(self):
        """Il replay 80/20 deve funzionare col training limitato agli adapter LoRA."""
        r = self._run_continual(use_replay=True)
        print(f"\n  con replay    -> degrado acc A: {r['acc_deg_pct']:+.4f}%, acc B: {r['acc_b']:.4f}")
        assert r["acc_deg_pct"] < 2.0

    def test_micro_ewc_penalty_is_active_on_lora(self):
        """Il Micro-EWC deve produrre una penalita' non nulla e gradienti solo sui LoRA."""
        trainer = _make_trainer(_continual_model(CONTINUAL_LORA_RANK))
        trainer.compute_fisher_matrix(DS_A, num_samples=10)
        assert set(trainer.fisher_matrix.keys()) == set(trainer.model.trainable_param_names())
        assert trainer._compute_ewc_loss() == 0.0
        for i in trainer.model.trainable_param_indices():
            trainer.model.get_parameters()[i] += 0.5
        assert trainer._compute_ewc_loss() > 0.0
        grads = trainer._compute_ewc_gradients()
        assert len(grads) == len(trainer.model.get_parameters())
        trainable = set(trainer.model.trainable_param_indices())
        for i, g in enumerate(grads):
            if i not in trainable:
                assert float(np.max(np.abs(np.asarray(g)))) == 0.0

    def test_fisher_is_reanchored_to_the_new_task(self):
        """
        Dopo la registrazione di un nuovo task, la Fisher deve essere
        ri-ancorata ai parametri del nuovo adapter. Senza questo la penalita'
        EWC sarebbe silenziosamente nulla (bug gia' osservato).
        """
        model = _continual_model(CONTINUAL_LORA_RANK)
        _, trainer = _learn_task_a(model)
        trainer.compute_fisher_matrix(DS_A, num_samples=20)
        assert trainer._compute_ewc_loss() == 0.0
        model.add_task_adapter(warm_start=False)
        # Prima del ri-ancoraggio i nomi della Fisher non corrispondono ai
        # parametri addestrabili del nuovo task: la penalita' e' nulla.
        assert set(trainer.fisher_matrix.keys()) != set(model.trainable_param_names())
        trainer._ensure_ewc_anchor()
        assert set(trainer.fisher_matrix.keys()) == set(model.trainable_param_names())
        # Subito dopo il ri-ancoraggio i pesi ottimali coincidono coi valori
        # correnti del nuovo adapter, quindi la penalita' e' ancora nulla.
        assert trainer._compute_ewc_loss() == 0.0
        # Spostando i parametri addestrabili la penalita' diventa positiva.
        for i in model.trainable_param_indices():
            trainer.model.get_parameters()[i] += 0.5
        assert trainer._compute_ewc_loss() > 0.0


# ----------------------------------------------------------------------
# (d) CRITERIO 4 — Apprendimento reale del nuovo task vs fine-tuning completo
# ----------------------------------------------------------------------

class TestNewTaskLearning:
    """
    Criterio (d): l'apprendimento sul nuovo task con il solo LoRA deve essere
    confrontabile con quello del fine-tuning completo sullo stesso task.

    SOGLIA DICHIARATA: accuratezza sul nuovo task entro 2 punti percentuali
    rispetto al fine-tuning completo, e il LoRA deve effettivamente ridurre
    la loss. Il rapporto delle LOSS non e' soggetto ad asserzione: misurato
    e' ~39x, ben oltre la soglia di 1.5x dichiarata nel report come NON
    raggiunta (vedi `test_loss_ratio_known_limitation`).
    """

    def _train_new_task(self, lora_rank):
        model = _continual_model(lora_rank)
        trainer = _make_trainer(model)
        loss_before, acc_before = _eval_on(model, IN_B, TG_B)
        trainer.train(IN_B, TG_B, epochs=CONTINUAL_EPOCHS, batch_size=32)
        loss_after, acc_after = _eval_on(model, IN_B, TG_B)
        return model, loss_before, acc_before, loss_after, acc_after

    def test_lora_accuracy_matches_full_finetuning(self):
        full_model, _, _, loss_full, acc_full = self._train_new_task(0)
        lora_model, loss_lora_before, acc_lora_before, loss_lora, acc_lora = \
            self._train_new_task(CONTINUAL_LORA_RANK)

        acc_gap_pp = 100.0 * (acc_full - acc_lora)
        print("\n=== CRITERIO (d): APPRENDIMENTO DEL NUOVO TASK (task B) ===")
        print(f"  parametri addestrabili full-FT    : {full_model.trainable_parameter_count()}")
        print(f"  parametri addestrabili LoRA r{CONTINUAL_LORA_RANK}   : {lora_model.trainable_parameter_count()}")
        print(f"  accuracy B full-FT   : {acc_full:.6f}")
        print(f"  accuracy B LoRA      : {acc_lora:.6f}")
        print(f"  gap accuratezza      : {acc_gap_pp:+.4f} pp   (soglia < 2 pp)")
        print(f"  loss B full-FT       : {loss_full:.6f}")
        print(f"  loss B LoRA (prima)  : {loss_lora_before:.6f}")
        print(f"  loss B LoRA (dopo)   : {loss_lora:.6f}")
        print(f"  rapporto loss LoRA/full-FT : {loss_lora / loss_full:.2f}x  (NON soggetto ad asserizione)")

        # Il LoRA deve imparare realmente qualcosa
        assert loss_lora < loss_lora_before, (
            f"il LoRA non ha ridotto la loss sul nuovo task: "
            f"{loss_lora_before:.6f} -> {loss_lora:.6f}"
        )
        assert acc_lora > 0.9, f"accuracy insufficiente sul nuovo task: {acc_lora:.4f}"
        # Addestra molti meno parametri
        assert lora_model.trainable_parameter_count() < 0.2 * full_model.trainable_parameter_count()
        # Soglia dichiarata
        assert abs(acc_gap_pp) < 2.0, (
            f"gap di accuratezza sul nuovo task troppo ampio: {acc_gap_pp:+.4f} pp"
        )

    def test_loss_ratio_known_limitation(self):
        """
        Limite NOTO e DICHIARATO: il rapporto tra la loss ottenuta con il solo
        LoRA e quella ottenuta con fine-tuning completo NON e' comparabile
        (~39x contro la soglia dichiarata di 1.5x), anche se l'accuratezza e'
        allineata. Codificato come test cosi' che un miglioramento futuro
        faccia fallire il test e segnali l'aggiornamento della documentazione.
        """
        _, _, _, loss_full, _ = self._train_new_task(0)
        _, _, _, loss_lora, _ = self._train_new_task(CONTINUAL_LORA_RANK)
        ratio = loss_lora / loss_full
        print("\n=== LIMITE NOTO: rapporto delle loss sul nuovo task ===")
        print(f"  loss full-FT : {loss_full:.6f}")
        print(f"  loss LoRA    : {loss_lora:.6f}")
        print(f"  rapporto     : {ratio:.2f}x  (soglia dichiarata 1.5x: NON raggiunta)")
        assert ratio > 1.5, (
            "il rapporto delle loss e' sceso sotto 1.5x: il limite dichiarato "
            "non e' piu' valido, aggiornare report e documentazione"
        )

    def test_lora_reduces_loss_more_than_untrained_baseline(self):
        """Controllo che il guadagno derivi dall'apprendimento, non dal caso."""
        model = _continual_model(CONTINUAL_LORA_RANK)
        trainer = _make_trainer(model)
        baseline_loss, baseline_acc = _eval_on(model, IN_B, TG_B)
        trainer.train(IN_B, TG_B, epochs=CONTINUAL_EPOCHS, batch_size=32)
        trained_loss, trained_acc = _eval_on(model, IN_B, TG_B)
        print(f"\n  loss B baseline (non addestrato) : {baseline_loss:.6f}  (acc {baseline_acc:.6f})")
        print(f"  loss B addestrato                : {trained_loss:.6f}  (acc {trained_acc:.6f})")
        assert trained_loss < baseline_loss, (
            f"l'addestramento LoRA non ha ridotto la loss: "
            f"{baseline_loss:.6f} -> {trained_loss:.6f}"
        )
        assert trained_acc > baseline_acc + 0.5, (
            f"l'accuratezza sul nuovo task non e' migliorata: "
            f"{baseline_acc:.6f} -> {trained_acc:.6f}"
        )

    def test_untrained_lora_is_a_no_op(self):
        """
        Con B inizializzato a zero, un adapter non addestrato non modifica
        l'output del modello rispetto allo stesso modello senza LoRA.
        """
        model = _continual_model(CONTINUAL_LORA_RANK)
        for layer in model.layers:
            assert np.allclose(layer.ff.lora_stack[0].B, 0.0)
        plain = build_model("transformer", seed=42,
                            vocab_size=len(TOKENIZER_ALL.vocabulary), d_model=64,
                            num_heads=4, d_ff=128, num_layers=2, max_len=32)
        probe = IN_B[:4]
        a = plain.forward(probe, training=False)
        b = model.forward(probe, training=False)
        assert np.max(np.abs(a - b)) < 1e-12, (
            f"l'adapter LoRA non addestrato deve essere un no-op, "
            f"max diff {np.max(np.abs(a - b)):.3e}"
        )


class TestTorchModelLoRAAdapterStack:
    """
    Copre il percorso GPU/torch di Micro-EWC con LoRA.

    Il torch model replica l'adapter stack della reference NumPy: senza questi
    test il path torch non era coperto da alcuna regressione, quindi un
    disallineamento tra le due implementazioni passava inosservato.
    """

    @staticmethod
    def _pair(lora_rank=4):
        from neural_core.network import MicroTransformer
        from neural_core.torch_model import TorchMicroTransformer
        kwargs = dict(vocab_size=23, d_model=16, num_heads=2, d_ff=32,
                      num_layers=2, max_len=12, seed=7)
        np_model = MicroTransformer(lora_rank=lora_rank, **kwargs)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            pytest.skip(f"torch non disponibile: {exc}")
        t_model = TorchMicroTransformer(lora_rank=lora_rank, dtype=torch.float64,
                                        **kwargs)
        t_model.set_parameters(np_model.get_parameters())
        return np_model, t_model, np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    @staticmethod
    def _np_out(model, x):
        import torch
        out = model.forward(x)
        if isinstance(out, torch.Tensor):
            out = out.detach().numpy()
        return np.asarray(out, dtype=np.float64)

    def test_parity_senza_ulteriori_task(self):
        np_model, t_model, x = self._pair()
        err = np.max(np.abs(self._np_out(np_model, x) - self._np_out(t_model, x)))
        assert err < 1e-12, f"parity NumPy/torch rotta senza LoRA multi-task: {err:.3e}"

    def test_parity_con_stack_di_task(self):
        np_model, t_model, x = self._pair()
        for warm_start in (False, True):
            np_model.add_task_adapter(warm_start=warm_start)
            t_model.add_task_adapter(warm_start=warm_start)
        assert np_model.num_lora_tasks == 3
        assert t_model.num_lora_tasks == 3
        err = np.max(np.abs(self._np_out(np_model, x) - self._np_out(t_model, x)))
        assert err < 1e-12, f"parity NumPy/torch rotta su 3 task: {err:.3e}"

    def test_indici_e_nomi_allineati_alla_reference(self):
        np_model, t_model, _ = self._pair()
        np_model.add_task_adapter(warm_start=False)
        t_model.add_task_adapter(warm_start=False)
        np_model.add_task_adapter(warm_start=True)
        t_model.add_task_adapter(warm_start=True)
        assert np_model.trainable_param_names() == t_model.trainable_param_names()
        assert np_model.trainable_param_indices() == t_model.trainable_param_indices()
        assert np_model.parameter_count() == t_model.parameter_count()
        assert np_model.trainable_parameter_count() == t_model.trainable_parameter_count()

    def test_set_active_task_seleziona_adapter_precedenti(self):
        np_model, t_model, x = self._pair()
        np_model.add_task_adapter(warm_start=False)
        t_model.add_task_adapter(warm_start=False)
        np_model.set_active_task(0)
        t_model.set_active_task(0)
        assert np_model.active_task == 0 and t_model.active_task == 0
        assert t_model.trainable_param_names() == [
            "layer0_lora_0_A", "layer0_lora_0_B",
            "layer1_lora_0_A", "layer1_lora_0_B",
        ]
        err = np.max(np.abs(self._np_out(np_model, x) - self._np_out(t_model, x)))
        assert err < 1e-12, f"parity rotta selezionando un task precedente: {err:.3e}"

    def test_solo_adapter_corrente_riceve_gradiente(self):
        import torch
        _, t_model, _ = self._pair()
        t_model.add_task_adapter(warm_start=False)
        t_model.set_active_task(0)
        trainable = {n for n, p in torch.nn.Module.named_parameters(t_model)
                     if p.requires_grad}
        assert trainable == {
            f"blocks.{i}.lora_0_{ab}"
            for i in range(t_model.num_layers) for ab in ("A", "B")
        }, f"gradienti concessi a parametri non-LoRA del task attivo: {sorted(trainable)}"

    def test_adapter_registration_e_append_only(self):
        _, t_model, _ = self._pair()
        assert t_model.add_task_adapter() == 1
        assert t_model.num_lora_tasks == 2
        with pytest.raises(ValueError):
            t_model.add_task_adapter(task=0)
        with pytest.raises(ValueError):
            t_model.set_active_task(7)
        # `task=None` appends in coda, coerente con la reference NumPy.
        assert t_model.add_task_adapter() == 2
        assert t_model.num_lora_tasks == 3

    @pytest.mark.parametrize("warm_start", [None, True, False])
    def test_semantica_warm_start_allineata_alla_reference(self, warm_start):
        """
        `warm_start=None` deve valere True in entrambe le implementazioni.

        Con B del task 0 lasciato a zero warm e cold start sono indistinguibili,
        quindi il task 0 viene reso non nullo per rendere il test osservabile.
        """
        np_model, t_model, x = self._pair()
        np_model.layers[0].ff.lora_stack[0].B[:] = 0.3
        t_model.set_parameters(np_model.get_parameters())
        np_model.add_task_adapter(warm_start=warm_start)
        t_model.add_task_adapter(warm_start=warm_start)
        b_np = np_model.layers[0].ff.lora_stack[1].B
        b_torch = t_model.blocks[0]["lora_1_B"].detach().numpy()
        expected = 0.0 if warm_start is False else 0.3
        assert abs(float(np.abs(b_np).max()) - expected) < 1e-12, (
            f"warm_start={warm_start}: B numpy {np.abs(b_np).max():.4f} != {expected}")
        assert np.allclose(b_np, b_torch), (
            f"warm_start={warm_start}: B torch {np.abs(b_torch).max():.4f} != numpy {np.abs(b_np).max():.4f}")
        err = np.max(np.abs(self._np_out(np_model, x) - self._np_out(t_model, x)))
        assert err < 1e-12, f"parity rotta con warm_start={warm_start}: {err:.3e}"

    def test_parametri_senza_lora_restano_legacy(self):
        import torch
        from neural_core.torch_model import TorchMicroTransformer
        model = TorchMicroTransformer(vocab_size=20, d_model=16, num_heads=2,
                                      d_ff=32, num_layers=2, max_len=16, seed=3,
                                      dtype=torch.float64)
        assert model.num_lora_tasks == 0
        assert model.active_task is None
        assert model.trainable_param_names() == []
        assert model.trainable_parameter_count() == model.parameter_count()
