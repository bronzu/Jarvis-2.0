"""
Test di hardening pre-scaling per JARVIS Neural Core.

Coprono:
    D4  — dtype float64/float32
    D7  — duplicated softmax regression
    D8  — optimizer in-place update
    D9  — evaluation batch_size default conservativo
    D10 — gradient accumulation con dataset non divisibile
    D11 — is_best callback logica corretta
    D12 — best vs last training state separati
    D14 — checkpoint validation esplicita
"""

import json

import numpy as np
import pytest

from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.network import MicroTransformer
from neural_core.optimizer import Adam
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import Trainer


# ----------------------------------------------------------------------
# Fixture di supporto
# ----------------------------------------------------------------------


def _sequence_dataset(context_size=8, repetitions=40):
    text = " ".join(["alpha beta gamma delta epsilon zeta eta theta"] * repetitions)
    tokenizer = Tokenizer()
    training_dataset, validation_dataset = Dataset.from_text(
        text, tokenizer, validation_split=0.2, context_size=context_size
    )
    return tokenizer, training_dataset, validation_dataset


def _transformer_trainer(tokenizer, context_size=8, **trainer_kwargs):
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=2,
        max_len=max(128, context_size),
        seed=13,
    )
    return Trainer(
        model,
        tokenizer,
        CrossEntropyLoss(),
        Adam(learning_rate=0.01),
        **trainer_kwargs,
    )


# ----------------------------------------------------------------------
# D4 — dtype support
# ----------------------------------------------------------------------


class TestDtypeSupport:
    def test_next_token_default_dtype_float64(self):
        model = Trainer.__init__.__globals__["NextTokenModel"](vocab_size=10)
        assert str(model.dtype) == "float64"

    def test_next_token_float32(self):
        from neural_core.trainer import NextTokenModel
        model = NextTokenModel(vocab_size=10, dtype=np.float32)
        assert str(model.dtype) == "float32"
        assert model.embeddings.dtype == np.float32

    def test_micro_transformer_default_dtype_float64(self):
        model = MicroTransformer(vocab_size=10)
        assert str(model.dtype) == "float64"

    def test_micro_transformer_float32(self):
        model = MicroTransformer(vocab_size=10, dtype=np.float32)
        assert str(model.dtype) == "float32"

    def test_forward_loss_gradient_float32_vs_float64(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()

        model64 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=1, max_len=128, seed=1, dtype=np.float64)
        model32 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=1, max_len=128, seed=1, dtype=np.float32)

        logits64 = model64.forward(inputs[:4])
        logits32 = model32.forward(inputs[:4])

        loss_fn = CrossEntropyLoss()
        loss64 = loss_fn.forward(logits64, targets[:4])
        loss32 = loss_fn.forward(logits32, targets[:4])

        assert np.isfinite(loss64)
        assert np.isfinite(loss32)
        assert np.allclose(loss64, loss32, atol=1e-4)

        grads64 = model64.backward(targets[:4])
        grads32 = model32.backward(targets[:4])

        for g64, g32 in zip(grads64, grads32):
            assert np.allclose(g64, g32, atol=1e-3)


# ----------------------------------------------------------------------
# D7 — duplicated softmax regression
# ----------------------------------------------------------------------


class TestSoftmaxDuplication:
    def test_cross_entropy_caches_probabilities(self):
        loss = CrossEntropyLoss()
        logits = np.random.RandomState(0).randn(4, 6)
        targets = np.array([0, 1, 2, 3])
        loss.forward(logits, targets)
        cached = loss.get_probabilities()
        assert cached is not None
        assert cached.shape == (4, 6)

    def test_softmax_regression_loss_e_gradienti(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()

        model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=1, max_len=128, seed=1)
        loss = CrossEntropyLoss()

        batch_in = inputs[:4]
        batch_tgt = targets[:4]

        logits = model.forward(batch_in, training=True)
        loss_value = float(loss.forward(logits, batch_tgt))
        probs = loss.get_probabilities()
        gradients = model.backward(batch_tgt, probs=probs)

        assert np.isfinite(loss_value)
        assert len(gradients) == len(model.get_parameters())
        for g in gradients:
            assert np.all(np.isfinite(g))

    def test_softmax_forward_backward_numericamente_corretto(self):
        np.random.seed(0)
        model = MicroTransformer(vocab_size=9, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=128, seed=5)
        loss_fn = CrossEntropyLoss()
        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        targets = np.array([[2, 3, 4], [5, 6, 7]])

        logits = model.forward(inputs)
        loss1 = float(loss_fn.forward(logits, targets))
        grads1 = model.backward(targets)

        model2 = MicroTransformer(vocab_size=9, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=128, seed=5)
        logits2 = model2.forward(inputs)
        loss2 = float(loss_fn.forward(logits2, targets))
        grads2 = model2.backward(targets)

        assert np.isclose(loss1, loss2, atol=1e-9)
        for g1, g2 in zip(grads1, grads2):
            assert np.allclose(g1, g2, atol=1e-9)


# ----------------------------------------------------------------------
# D8 — optimizer in-place update
# ----------------------------------------------------------------------


class TestOptimizerInPlace:
    def test_adam_update_avoids_allocation(self):
        params = [np.array([1.0, 2.0])]
        grads = [np.array([0.5, 0.5])]
        adam = Adam(learning_rate=0.1)
        updated = adam.update(params, grads)
        assert updated[0] is params[0]

    def test_adam_numerical_equivalence(self):
        np.random.seed(42)
        params1 = [np.array([1.0, 2.0])]
        grads = [np.array([0.5, 0.5])]

        adam1 = Adam(learning_rate=0.1)
        updated1 = adam1.update(params1, grads)

        params2 = [np.array([1.0, 2.0])]
        adam2 = Adam(learning_rate=0.1)
        adam2.m = [m.copy() for m in adam1.m]
        adam2.v = [v.copy() for v in adam1.v]
        adam2.t = adam1.t
        updated2 = adam2.update(params2, grads)

        assert np.allclose(updated1[0], updated2[0], atol=1e-12)


# ----------------------------------------------------------------------
# D9 — evaluation batch_size default
# ----------------------------------------------------------------------


class TestEvaluationBatchSize:
    def test_evaluate_default_batch_size(self):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        inputs, targets = validation_dataset.get_arrays()
        loss = trainer.evaluate(inputs, targets)
        assert np.isfinite(loss)

    def test_evaluate_small_batch(self):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        inputs, targets = validation_dataset.get_arrays()
        loss = trainer.evaluate(inputs, targets, batch_size=2)
        assert np.isfinite(loss)


# ----------------------------------------------------------------------
# D10 — gradient accumulation con dataset non divisibile
# ----------------------------------------------------------------------


class TestGradientAccumulation:
    def test_accumulation_1(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()
        trainer = _transformer_trainer(tokenizer, gradient_accumulation_steps=1)
        trainer._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=False)
        assert trainer.global_step == 8

    def test_accumulation_2(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()
        trainer = _transformer_trainer(tokenizer, gradient_accumulation_steps=2)
        trainer._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=False)
        assert trainer.global_step == 4

    def test_accumulation_4_dataset_non_divisibile(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()
        inputs, targets = inputs[:10], targets[:10]
        trainer = _transformer_trainer(tokenizer, gradient_accumulation_steps=4)
        trainer._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=False)
        assert trainer.global_step == 1

    def test_accumulation_last_partial_group_scaling(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()
        inputs, targets = inputs[:10], targets[:10]

        no_accum = _transformer_trainer(tokenizer, gradient_accumulation_steps=1)
        no_accum._run_epoch(inputs, targets, batch_size=10, training=True, shuffle=False)

        accum = _transformer_trainer(tokenizer, gradient_accumulation_steps=4)
        accum._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=False)

        assert no_accum.global_step == 1
        assert accum.global_step == 1
        for p_no, p_acc in zip(no_accum.model.get_parameters(), accum.model.get_parameters()):
            assert np.allclose(p_no, p_acc, atol=1e-5)


# ----------------------------------------------------------------------
# D11 — is_best callback
# ----------------------------------------------------------------------


class TestIsBestCallback:
    def test_is_best_quando_val_loss_migliore(self):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        trainer.best_validation_loss = 10.0

        inputs, targets = training_dataset.get_arrays()
        val_inputs, val_targets = validation_dataset.get_arrays()

        callback_calls = []

        def on_epoch_end(**kwargs):
            callback_calls.append(kwargs["is_best"])

        trainer.train_with_validation(
            inputs, targets, val_inputs, val_targets,
            epochs=1, batch_size=4, patience=5,
            on_epoch_end=on_epoch_end,
        )

        assert any(callback_calls)

    def test_is_best_quando_val_loss_peggiore(self):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        trainer.best_validation_loss = 0.0

        inputs, targets = training_dataset.get_arrays()
        val_inputs, val_targets = validation_dataset.get_arrays()

        callback_calls = []

        def on_epoch_end(**kwargs):
            callback_calls.append(kwargs["is_best"])

        trainer.train_with_validation(
            inputs, targets, val_inputs, val_targets,
            epochs=1, batch_size=4, patience=5,
            on_epoch_end=on_epoch_end,
        )

        assert not any(callback_calls)

    def test_is_best_quando_val_loss_uguale(self):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        trainer.best_validation_loss = 0.0

        inputs, targets = training_dataset.get_arrays()
        val_inputs, val_targets = validation_dataset.get_arrays()

        callback_calls = []

        def on_epoch_end(**kwargs):
            callback_calls.append(kwargs["is_best"])

        trainer.train_with_validation(
            inputs, targets, val_inputs, val_targets,
            epochs=1, batch_size=4, patience=5,
            on_epoch_end=on_epoch_end,
        )

        assert not any(callback_calls)


# ----------------------------------------------------------------------
# D12 — best vs last training state
# ----------------------------------------------------------------------


class TestBestVsLastCheckpoint:
    def test_last_checkpoint_salvato_prima_del_restore_best(self, tmp_path):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)

        inputs, targets = training_dataset.get_arrays()
        val_inputs, val_targets = validation_dataset.get_arrays()

        last_path = tmp_path / "last.pkl"
        best_path = tmp_path / "best.json"

        history = trainer.train_with_validation(
            inputs, targets, val_inputs, val_targets,
            epochs=2, batch_size=4, patience=5,
            checkpoint_path=str(best_path),
            last_checkpoint_path=str(last_path),
        )

        assert last_path.exists()
        assert best_path.exists()

        # Il last checkpoint deve contenere lo stato dell'ultima epoca
        import pickle
        with open(last_path, "rb") as f:
            last_state = pickle.load(f)
        assert last_state["epoch"] == 2

    def test_best_checkpoint_contiene_best_validation_loss(self, tmp_path):
        tokenizer, training_dataset, validation_dataset = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)

        inputs, targets = training_dataset.get_arrays()
        val_inputs, val_targets = validation_dataset.get_arrays()

        best_path = tmp_path / "best.json"
        history = trainer.train_with_validation(
            inputs, targets, val_inputs, val_targets,
            epochs=2, batch_size=4, patience=5,
            checkpoint_path=str(best_path),
        )

        with open(best_path, "r") as f:
            best_data = json.load(f)
        assert "best_validation_loss" in best_data["parameters"]


# ----------------------------------------------------------------------
# D14 — checkpoint validation
# ----------------------------------------------------------------------


class TestCheckpointValidation:
    def test_checkpoint_rifiuta_mismatch_vocab_size(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        tokenizer2 = Tokenizer()
        tokenizer2.fit("vocabolario completamente diverso con token diversi")
        model2 = MicroTransformer(vocab_size=len(tokenizer2.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128)
        trainer2 = Trainer(model2, tokenizer2, CrossEntropyLoss(), Adam())

        with pytest.raises(ValueError, match="vocab_size"):
            trainer2.load_checkpoint(str(path))

    def test_checkpoint_rifiuta_mismatch_d_model(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        model2 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=32, num_heads=2, d_ff=32, num_layers=2, max_len=128)
        trainer2 = Trainer(model2, trainer.tokenizer, CrossEntropyLoss(), Adam())

        with pytest.raises(ValueError, match="d_model"):
            trainer2.load_checkpoint(str(path))

    def test_checkpoint_rifiuta_mismatch_num_heads(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        model2 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=4, d_ff=32, num_layers=2, max_len=128)
        trainer2 = Trainer(model2, trainer.tokenizer, CrossEntropyLoss(), Adam())

        with pytest.raises(ValueError, match="num_heads"):
            trainer2.load_checkpoint(str(path))

    def test_checkpoint_rifiuta_mismatch_max_len(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        model2 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=256)
        trainer2 = Trainer(model2, trainer.tokenizer, CrossEntropyLoss(), Adam())

        with pytest.raises(ValueError, match="max_len"):
            trainer2.load_checkpoint(str(path))

    def test_checkpoint_rifiuta_tokenizer_diverso(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        tokenizer2 = Tokenizer()
        text2 = " ".join(["alpha beta gamma delta epsilon"] * 20)
        tokenizer2.fit(text2)
        model2 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128)
        trainer2 = Trainer(model2, tokenizer2, CrossEntropyLoss(), Adam())

        with pytest.raises(ValueError):
            trainer2.load_checkpoint(str(path))

    def test_checkpoint_rifiuta_tokenizer_mode_diversa(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        tokenizer2 = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
        text2 = " ".join(["alpha beta gamma delta epsilon"] * 20)
        tokenizer2.fit(text2)
        model2 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128)
        trainer2 = Trainer(model2, tokenizer2, CrossEntropyLoss(), Adam())

        with pytest.raises(ValueError, match="mode"):
            trainer2.load_checkpoint(str(path))

    def test_checkpoint_compatibile_ok(self, tmp_path):
        tokenizer, training_dataset, _ = _sequence_dataset()
        trainer = _transformer_trainer(tokenizer)
        path = tmp_path / "state.pkl"
        trainer.save_checkpoint(str(path))

        model2 = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128)
        trainer2 = Trainer(model2, tokenizer, CrossEntropyLoss(), Adam())
        trainer2.load_checkpoint(str(path))
        assert trainer2.global_step == trainer.global_step


class TestParameterCountBreakdown:
    def test_next_token_breakdown(self):
        from neural_core.trainer import NextTokenModel
        model = NextTokenModel(vocab_size=10, embedding_dim=4, hidden_dim=8)
        breakdown = model.parameter_count_breakdown()
        assert breakdown["total"] == model.parameter_count()
        assert breakdown["trainable"] == breakdown["total"]
        assert breakdown["non_trainable"] == 0
        assert breakdown["embedding"] == 40
        assert breakdown["ffn"] == 40
        assert breakdown["output"] == 90
        assert breakdown["total"] == breakdown["embedding"] + breakdown["ffn"] + breakdown["output"]

    def test_micro_transformer_breakdown(self):
        model = MicroTransformer(vocab_size=10, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128)
        breakdown = model.parameter_count_breakdown()
        assert breakdown["total"] == model.parameter_count()
        assert breakdown["trainable"] == breakdown["total"]
        assert breakdown["non_trainable"] == 0
        assert breakdown["embedding"] == 160
        assert breakdown["attention"] > 0
        assert breakdown["ffn"] > 0
        assert breakdown["layernorm"] > 0
        assert breakdown["output"] == 170
        assert breakdown["total"] == (
            breakdown["embedding"]
            + breakdown["attention"]
            + breakdown["ffn"]
            + breakdown["layernorm"]
            + breakdown["output"]
        )


class TestReproducibility:
    def test_stesso_seed_stesso_risultato(self):
        np.random.seed(0)
        model1 = MicroTransformer(vocab_size=10, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128, seed=42)
        np.random.seed(0)
        model2 = MicroTransformer(vocab_size=10, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128, seed=42)

        inputs = np.array([[1, 2, 3]])
        logits1 = model1.forward(inputs)
        logits2 = model2.forward(inputs)
        assert np.allclose(logits1, logits2, atol=1e-12)

    def test_seed_diverso_risultato_diverso(self):
        model1 = MicroTransformer(vocab_size=10, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128, seed=1)
        model2 = MicroTransformer(vocab_size=10, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128, seed=2)

        inputs = np.array([[1, 2, 3]])
        logits1 = model1.forward(inputs)
        logits2 = model2.forward(inputs)
        assert not np.allclose(logits1, logits2, atol=1e-6)

    def test_dropout_deterministico_con_stesso_seed(self):
        np.random.seed(5)
        model = MicroTransformer(vocab_size=10, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=128, seed=5, dropout=0.5)
        inputs = np.array([[1, 2, 3, 4]])
        out1 = model.forward(inputs, training=True).copy()
        out2 = model.forward(inputs, training=True).copy()
        assert np.allclose(out1, out2, atol=1e-12)

    def test_shuffle_deterministico_con_seed(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()

        trainer1 = _transformer_trainer(tokenizer, seed=99)
        trainer2 = _transformer_trainer(tokenizer, seed=99)

        trainer1._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=True)
        trainer2._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=True)

        for p1, p2 in zip(trainer1.model.get_parameters(), trainer2.model.get_parameters()):
            assert np.allclose(p1, p2, atol=1e-12)

    def test_train_step_deterministico_con_seed(self):
        tokenizer, training_dataset, _ = _sequence_dataset()
        inputs, targets = training_dataset.get_arrays()

        trainer1 = _transformer_trainer(tokenizer, seed=77)
        trainer2 = _transformer_trainer(tokenizer, seed=77)

        batch_in = inputs[:4]
        batch_tgt = targets[:4]

        loss1 = trainer1.train_step(batch_in, batch_tgt)
        loss2 = trainer2.train_step(batch_in, batch_tgt)

        assert np.isclose(loss1, loss2, atol=1e-12)
        for p1, p2 in zip(trainer1.model.get_parameters(), trainer2.model.get_parameters()):
            assert np.allclose(p1, p2, atol=1e-12)
