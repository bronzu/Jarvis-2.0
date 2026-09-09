import pickle

import numpy as np
import pytest

from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.network import MicroTransformer
from neural_core.optimizer import Adam, LearningRateScheduler
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import (
    NextTokenModel,
    Trainer,
    build_model,
    load_model_from_weights,
)


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


def test_trainer_trains_on_small_dataset(tmp_path):
    text = "alpha beta gamma delta epsilon alpha beta gamma delta epsilon"
    tokenizer = Tokenizer()
    training_dataset, _ = Dataset.from_text(text, tokenizer, validation_split=0.2)

    model = NextTokenModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=8)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model, tokenizer, loss, optimizer)

    inputs = [[t] for t in training_dataset.get_inputs()]
    targets = training_dataset.get_targets()

    history = trainer.train(inputs, targets, epochs=2)

    assert len(history) == 2
    assert all(isinstance(l, float) for l in history)
    assert all(l > 0 for l in history)


def test_trainer_saves_and_loads_weights(tmp_path):
    text = "alpha beta gamma delta epsilon zeta eta theta"
    tokenizer = Tokenizer()
    training_dataset, _ = Dataset.from_text(text, tokenizer, validation_split=0.25)

    model = NextTokenModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=8)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model, tokenizer, loss, optimizer)

    inputs = [[t] for t in training_dataset.get_inputs()]
    targets = training_dataset.get_targets()
    trainer.train(inputs, targets, epochs=1)

    weights_path = tmp_path / "jarvis_neural_weights.pth"
    trainer.save_weights(str(weights_path))
    assert weights_path.exists()

    new_model = NextTokenModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=8)
    new_trainer = Trainer(new_model, tokenizer, loss, optimizer)
    new_trainer.load_weights(str(weights_path))

    np.testing.assert_array_equal(new_model.embeddings, model.embeddings)
    np.testing.assert_array_equal(new_model.W2, model.W2)
    np.testing.assert_array_equal(new_model.b2, model.b2)


def test_trainer_from_corpus(tmp_path):
    corpus = "alpha beta gamma delta epsilon alpha beta gamma delta"
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(corpus, encoding="utf-8")

    trainer, training_dataset, validation_dataset = Trainer.from_corpus(
        str(corpus_file), validation_split=0.25, embedding_dim=8
    )

    assert isinstance(trainer, Trainer)
    assert len(training_dataset) > 0
    assert len(validation_dataset) > 0

    inputs = [[t] for t in training_dataset.get_inputs()]
    targets = training_dataset.get_targets()
    history = trainer.train(inputs, targets, epochs=1)
    assert len(history) == 1


# ======================================================================
# TRAINING STEP MINIMO SUL TRANSFORMER
# ======================================================================


def test_train_step_minimo_riduce_la_loss_del_transformer():
    np.random.seed(5)
    tokenizer, training_dataset, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    inputs, targets = training_dataset.get_arrays()
    batch_inputs = inputs[:4]
    batch_targets = targets[:4]

    loss_before = trainer.loss.forward(
        trainer.model.forward(batch_inputs), batch_targets
    )
    step_loss = trainer.train_step(batch_inputs, batch_targets)
    loss_after = trainer.loss.forward(
        trainer.model.forward(batch_inputs), batch_targets
    )

    assert isinstance(step_loss, float)
    assert np.isclose(step_loss, loss_before)
    assert loss_after < loss_before
    assert trainer.global_step == 1


def test_training_su_sequenze_riduce_la_loss_per_epoca():
    np.random.seed(6)
    tokenizer, training_dataset, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    inputs, targets = training_dataset.get_arrays()
    history = trainer.train(inputs, targets, epochs=3, batch_size=4)

    assert len(history) == 3
    assert all(isinstance(value, float) and value > 0 for value in history)
    assert history[-1] < history[0]
    assert trainer.epoch == 3


def test_mini_batch_produce_piu_update_per_epoca():
    np.random.seed(7)
    tokenizer, training_dataset, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    inputs, targets = training_dataset.get_arrays()
    trainer.train(inputs, targets, epochs=1, batch_size=4)

    expected_batches = int(np.ceil(len(inputs) / 4))
    assert trainer.global_step == expected_batches


def test_gradient_accumulation_riduce_il_numero_di_update():
    np.random.seed(8)
    tokenizer, training_dataset, _ = _sequence_dataset()
    inputs, targets = training_dataset.get_arrays()

    senza = _transformer_trainer(tokenizer)
    senza.train(inputs, targets, epochs=1, batch_size=4)

    con = _transformer_trainer(tokenizer, gradient_accumulation_steps=4)
    con.train(inputs, targets, epochs=1, batch_size=4)

    assert con.global_step < senza.global_step
    assert con.global_step == int(np.ceil(senza.global_step / 4))


def test_gradient_accumulation_equivale_al_batch_grande():
    """
    Accumulare 2 micro-batch da 4 deve produrre lo stesso update di un singolo
    batch da 8 (a meno dell'errore numerico).
    """
    tokenizer, training_dataset, _ = _sequence_dataset()
    inputs, targets = training_dataset.get_arrays()
    inputs, targets = inputs[:8], targets[:8]

    grande = _transformer_trainer(tokenizer)
    grande._run_epoch(inputs, targets, batch_size=8, training=True, shuffle=False)

    accumulato = _transformer_trainer(tokenizer, gradient_accumulation_steps=2)
    accumulato._run_epoch(inputs, targets, batch_size=4, training=True, shuffle=False)

    assert grande.global_step == accumulato.global_step == 1
    for atteso, ottenuto in zip(grande.model.get_parameters(), accumulato.model.get_parameters()):
        assert np.allclose(atteso, ottenuto, atol=1e-6)


def test_gradient_accumulation_non_valido():
    tokenizer, _, _ = _sequence_dataset()
    with pytest.raises(ValueError):
        _transformer_trainer(tokenizer, gradient_accumulation_steps=0)


def test_gradient_clipping_limita_la_norma():
    np.random.seed(9)
    tokenizer, training_dataset, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer, max_grad_norm=0.01)

    gradients = [np.ones((2, 2)) * 10.0, np.ones(2) * 10.0]
    clipped, total = trainer._clip_gradients(gradients)
    clipped_norm = float(np.sqrt(sum(np.sum(np.square(g)) for g in clipped)))

    assert total > 0.01
    assert clipped_norm <= 0.01 + 1e-9


def test_max_grad_norm_non_valido():
    tokenizer, _, _ = _sequence_dataset()
    with pytest.raises(ValueError):
        _transformer_trainer(tokenizer, max_grad_norm=0)


# ======================================================================
# SCHEDULER E WARMUP
# ======================================================================


def test_warmup_aumenta_il_learning_rate_durante_i_primi_step():
    np.random.seed(10)
    tokenizer, training_dataset, _ = _sequence_dataset()
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=4, total_steps=20, schedule="cosine"
    )
    trainer = _transformer_trainer(tokenizer, scheduler=scheduler)

    inputs, targets = training_dataset.get_arrays()
    learning_rates = []

    for index in range(4):
        trainer.train_step(inputs[index:index + 2], targets[index:index + 2])
        learning_rates.append(trainer.optimizer.learning_rate)

    assert learning_rates == sorted(learning_rates)
    assert learning_rates[0] < learning_rates[-1]
    assert np.isclose(learning_rates[-1], 0.01)


def test_scheduler_decade_dopo_il_warmup():
    np.random.seed(11)
    tokenizer, training_dataset, _ = _sequence_dataset()
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=1, total_steps=6, schedule="cosine"
    )
    trainer = _transformer_trainer(tokenizer, scheduler=scheduler)

    inputs, targets = training_dataset.get_arrays()
    learning_rates = []
    for index in range(5):
        trainer.train_step(inputs[index:index + 2], targets[index:index + 2])
        learning_rates.append(trainer.optimizer.learning_rate)

    assert learning_rates[1] > learning_rates[-1]
    assert scheduler.last_step == trainer.global_step - 1


# ======================================================================
# CHECKPOINT E RESUME
# ======================================================================


def test_checkpoint_completo_salva_e_ripristina_lo_stato(tmp_path):
    np.random.seed(12)
    tokenizer, training_dataset, validation_dataset = _sequence_dataset()
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=2, total_steps=30, schedule="cosine"
    )
    trainer = _transformer_trainer(
        tokenizer, scheduler=scheduler, gradient_accumulation_steps=2, max_grad_norm=1.0
    )

    inputs, targets = training_dataset.get_arrays()
    validation_inputs, validation_targets = validation_dataset.get_arrays()
    trainer.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=2, batch_size=4, patience=5,
    )

    checkpoint_path = tmp_path / "training_state.pkl"
    trainer.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists()

    ripristinato = _transformer_trainer(tokenizer)
    ripristinato.load_checkpoint(str(checkpoint_path))

    assert ripristinato.global_step == trainer.global_step
    assert ripristinato.epoch == trainer.epoch
    assert ripristinato.best_validation_loss == trainer.best_validation_loss
    assert ripristinato.gradient_accumulation_steps == 2
    assert ripristinato.max_grad_norm == 1.0
    assert ripristinato.optimizer.t == trainer.optimizer.t
    assert ripristinato.scheduler is not None
    assert ripristinato.scheduler.warmup_steps == 2
    assert len(ripristinato.history) == len(trainer.history)

    for atteso, ottenuto in zip(trainer.model.get_parameters(), ripristinato.model.get_parameters()):
        assert np.allclose(atteso, ottenuto)

    assert np.isclose(
        trainer.evaluate(validation_inputs, validation_targets),
        ripristinato.evaluate(validation_inputs, validation_targets),
    )


def test_resume_training_continua_dallo_stato_salvato(tmp_path):
    np.random.seed(13)
    tokenizer, training_dataset, validation_dataset = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    inputs, targets = training_dataset.get_arrays()
    validation_inputs, validation_targets = validation_dataset.get_arrays()
    trainer.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=1, batch_size=4, patience=5,
    )

    checkpoint_path = tmp_path / "resume.pkl"
    trainer.save_checkpoint(str(checkpoint_path))
    step_prima = trainer.global_step

    ripreso = _transformer_trainer(tokenizer)
    history = ripreso.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=1, batch_size=4, patience=5,
        resume_path=str(checkpoint_path),
    )

    assert ripreso.global_step > step_prima
    assert ripreso.epoch == 2
    assert history[0]["epoch"] == 2
    assert ripreso.resume(str(checkpoint_path))["global_step"] == step_prima


def test_load_checkpoint_rifiuta_architettura_diversa(tmp_path):
    tokenizer, training_dataset, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)
    checkpoint_path = tmp_path / "transformer.pkl"
    trainer.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists()

    altro_modello = NextTokenModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=8)
    altro = Trainer(altro_modello, tokenizer, CrossEntropyLoss(), Adam())

    with pytest.raises(ValueError):
        altro.load_checkpoint(str(checkpoint_path))


def test_load_checkpoint_file_inesistente(tmp_path):
    tokenizer, _, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(str(tmp_path / "assente.pkl"))


def test_load_checkpoint_formato_non_valido(tmp_path):
    tokenizer, _, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)
    path = tmp_path / "sbagliato.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"foo": "bar"}, handle)

    with pytest.raises(ValueError):
        trainer.load_checkpoint(str(path))


def test_checkpoint_puo_ripristinare_il_tokenizer(tmp_path):
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokenizer.fit("il modello calcola 3.14 e x ** 2")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, seed=1)
    trainer = Trainer(model, tokenizer, CrossEntropyLoss(), Adam())

    path = tmp_path / "state.pkl"
    trainer.save_checkpoint(str(path))

    vuoto = Tokenizer()
    altro = Trainer(
        MicroTransformer(**model.config()), vuoto, CrossEntropyLoss(), Adam()
    )
    altro.load_checkpoint(str(path), load_tokenizer=True)

    assert altro.tokenizer.mode == Tokenizer.MODE_TECHNICAL
    assert altro.tokenizer.vocabulary == tokenizer.vocabulary


# ======================================================================
# BEST MODEL
# ======================================================================


def test_best_model_viene_salvato_e_ricaricato(tmp_path):
    np.random.seed(14)
    tokenizer, training_dataset, validation_dataset = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    inputs, targets = training_dataset.get_arrays()
    validation_inputs, validation_targets = validation_dataset.get_arrays()

    best_path = tmp_path / "best_model.json"
    history = trainer.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=3, batch_size=4, patience=5, checkpoint_path=str(best_path),
    )

    assert best_path.exists()
    assert trainer.best_validation_loss == min(
        entry["validation_loss"] for entry in history
    )

    parametri_best = [np.asarray(p).copy() for p in trainer.model.get_parameters()]

    altro = _transformer_trainer(tokenizer)
    checkpoint = altro.load_best_checkpoint(str(best_path))

    assert checkpoint["model_class"] == "MicroTransformer"
    assert altro.best_validation_loss == trainer.best_validation_loss
    for atteso, ottenuto in zip(parametri_best, altro.model.get_parameters()):
        assert np.allclose(atteso, ottenuto)


def test_best_model_e_quello_con_validation_loss_minore():
    np.random.seed(15)
    tokenizer, training_dataset, validation_dataset = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    inputs, targets = training_dataset.get_arrays()
    validation_inputs, validation_targets = validation_dataset.get_arrays()

    history = trainer.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=3, batch_size=4, patience=5,
    )

    loss_finale = trainer.evaluate(validation_inputs, validation_targets)
    migliore = min(entry["validation_loss"] for entry in history)

    assert np.isclose(loss_finale, migliore, atol=1e-9)


def test_early_stopping_interrompe_il_training():
    np.random.seed(16)
    tokenizer, training_dataset, validation_dataset = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)
    trainer.best_validation_loss = 0.0  # nessun miglioramento sarà possibile

    inputs, targets = training_dataset.get_arrays()
    validation_inputs, validation_targets = validation_dataset.get_arrays()

    history = trainer.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=10, batch_size=8, patience=1,
    )

    assert len(history) == 1


# ======================================================================
# PESI, CONFIGURAZIONE E COSTRUZIONE DEI MODELLI
# ======================================================================


def test_save_e_load_weights_del_transformer(tmp_path):
    tokenizer, _, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    path = tmp_path / "model.pkl"
    trainer.save_weights(str(path))

    ricostruito = load_model_from_weights(str(path))

    assert isinstance(ricostruito, MicroTransformer)
    assert ricostruito.config() == trainer.model.config()

    inputs = np.array([[1, 2, 3]])
    assert np.allclose(ricostruito.forward(inputs), trainer.model.forward(inputs))


def test_load_model_from_weights_formato_storico(tmp_path):
    """Il formato storico di NextTokenModel resta caricabile."""
    path = tmp_path / "legacy.pkl"
    legacy = {
        "embeddings": np.ones((7, 4), dtype=np.float32),
        "W1": np.ones((4, 5), dtype=np.float32),
        "b1": np.zeros(5, dtype=np.float32),
        "W2": np.ones((5, 7), dtype=np.float32),
        "b2": np.zeros(7, dtype=np.float32),
        "vocab_size": 7,
        "embedding_dim": 4,
        "hidden_dim": 5,
        "dropout": 0.1,
    }
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle)

    model = load_model_from_weights(str(path))

    assert isinstance(model, NextTokenModel)
    assert model.vocab_size == 7
    assert model.embedding_dim == 4
    np.testing.assert_array_equal(model.embeddings, legacy["embeddings"])
    assert model.forward([0, 1]).shape == (2, 7)


def test_load_model_from_weights_rifiuta_payload_invalido(tmp_path):
    path = tmp_path / "rotto.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"qualcosa": 1, "vocab_size": 5}, handle)

    with pytest.raises(ValueError):
        load_model_from_weights(str(path))


def test_build_model_costruisce_le_architetture_proprietarie():
    next_token = build_model("next_token", vocab_size=10, embedding_dim=8)
    transformer = build_model("transformer", vocab_size=10, d_model=16, num_heads=2, d_ff=32)

    assert isinstance(next_token, NextTokenModel)
    assert isinstance(transformer, MicroTransformer)

    with pytest.raises(ValueError):
        build_model("gpt", vocab_size=10)


def test_next_token_model_supporta_sequenze():
    """
    Il modello a contesto singolo accetta gli stessi batch di sequenze del
    Transformer, così il Trainer resta unico.
    """
    np.random.seed(17)
    model = NextTokenModel(vocab_size=12, embedding_dim=8, hidden_dim=16, dropout=0.0)
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    targets = np.array([[2, 3, 4], [5, 6, 7]])

    logits = model.forward(inputs)
    assert logits.shape == (2, 3, 12)

    gradients = model.backward(targets)
    assert len(gradients) == len(model.get_parameters())
    for parameter, gradient in zip(model.get_parameters(), gradients):
        assert np.asarray(parameter).shape == np.asarray(gradient).shape


def test_next_token_model_config_e_context_illimitato():
    model = NextTokenModel(vocab_size=9, embedding_dim=4, hidden_dim=8)

    assert model.config()["vocab_size"] == 9
    assert model.max_len is None
    assert model.context_size is None
    assert model.parameter_count() > 0


def test_trainer_from_corpus_costruisce_il_transformer(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        " ".join(["uno due tre quattro cinque sei"] * 30), encoding="utf-8"
    )

    trainer, training_dataset, validation_dataset = Trainer.from_corpus(
        str(corpus_file),
        validation_split=0.2,
        architecture="transformer",
        context_size=8,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        learning_rate=0.01,
        warmup_steps=2,
        total_steps=10,
        schedule="cosine",
    )

    assert isinstance(trainer.model, MicroTransformer)
    assert trainer.model.max_len >= 8
    assert trainer.scheduler is not None
    assert training_dataset.context_size == 8
    assert validation_dataset.context_size == 8

    inputs, targets = training_dataset.get_arrays()
    history = trainer.train(inputs, targets, epochs=1, batch_size=4)
    assert len(history) == 1
    assert history[0] > 0


def test_trainer_from_dataset_allinea_la_context_window():
    tokenizer, training_dataset, _ = _sequence_dataset(context_size=16)

    trainer = Trainer.from_dataset(
        training_dataset,
        tokenizer,
        architecture="transformer",
        model_config={"d_model": 16, "num_heads": 2, "d_ff": 32, "num_layers": 1},
    )

    assert trainer.model.max_len == 16
    assert trainer.model.vocab_size == len(tokenizer.vocabulary)


def test_prepare_batch_rifiuta_forme_incoerenti():
    tokenizer, _, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    with pytest.raises(ValueError):
        trainer._prepare_batch(np.zeros((2, 4), dtype=int), np.zeros((2, 5), dtype=int))


def test_evaluate_non_modifica_i_parametri():
    np.random.seed(18)
    tokenizer, _, validation_dataset = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    prima = [np.asarray(p).copy() for p in trainer.model.get_parameters()]
    inputs, targets = validation_dataset.get_arrays()
    loss = trainer.evaluate(inputs, targets, batch_size=4)

    assert loss > 0
    assert trainer.global_step == 0
    for atteso, ottenuto in zip(prima, trainer.model.get_parameters()):
        assert np.allclose(atteso, ottenuto)


def test_evaluate_rifiuta_dataset_vuoto():
    tokenizer, _, _ = _sequence_dataset()
    trainer = _transformer_trainer(tokenizer)

    with pytest.raises(ValueError):
        trainer.evaluate([], [])

