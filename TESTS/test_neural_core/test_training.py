import numpy as np

from neural_core.model import Neuron
from neural_core.network import MicroTransformer
from neural_core.optimizer import LearningRateScheduler
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import NextTokenModel
from train import build_training_setup, train_and_save


def test_neuron_train_returns_history():
    neuron = Neuron(input_size=2)

    inputs = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]

    targets = [
        1.0,
        2.0,
        3.0,
    ]

    history = neuron.train(
        inputs,
        targets,
        epochs=3
    )

    assert isinstance(history, list)
    assert len(history) == 3

    assert all(
        isinstance(loss, float)
        for loss in history
    )


def test_neuron_train_with_validation():
    neuron = Neuron(input_size=2)

    training_inputs = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]

    training_targets = [
        1.0,
        2.0,
        3.0,
    ]

    validation_inputs = [
        [4.0, 5.0],
        [5.0, 6.0],
    ]

    validation_targets = [
        4.0,
        5.0,
    ]

    history = neuron.train_with_validation(
        training_inputs,
        training_targets,
        validation_inputs,
        validation_targets,
        epochs=3
    )

    assert isinstance(history, list)
    assert len(history) == 3

    for epoch in history:
        assert "training_loss" in epoch
        assert "validation_loss" in epoch


def test_neuron_train_with_validation_saves_best_checkpoint(tmp_path):
    neuron = Neuron(input_size=2)

    training_inputs = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]

    training_targets = [
        1.0,
        2.0,
        3.0,
    ]

    validation_inputs = [
        [4.0, 5.0],
        [5.0, 6.0],
    ]

    validation_targets = [
        4.0,
        5.0,
    ]

    checkpoint_path = tmp_path / "best_model.json"

    history = neuron.train_with_validation(
        training_inputs,
        training_targets,
        validation_inputs,
        validation_targets,
        epochs=3,
        checkpoint_path=str(checkpoint_path)
    )

    assert len(history) == 3
    assert checkpoint_path.exists()


def test_neuron_load_checkpoint_restores_parameters(tmp_path):
    checkpoint_path = tmp_path / "model.json"

    neuron = Neuron(input_size=2)

    neuron.weights = [0.5, 0.8]
    neuron.bias = 0.3

    neuron.save_checkpoint(str(checkpoint_path))

    restored_neuron = Neuron(input_size=2)

    restored_neuron.load_checkpoint(
        str(checkpoint_path)
    )

    assert restored_neuron.weights == [0.5, 0.8]
    assert restored_neuron.bias == 0.3


# ======================================================================
# ENTRY POINT DI TRAINING (train.py)
# ======================================================================


def _corpus_files(tmp_path, ripetizioni=60):
    frase = "il sistema neurale interno di jarvis apprende dal corpus"
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "validation.txt"
    test_file = tmp_path / "test.txt"

    train_file.write_text(" ".join([frase] * ripetizioni), encoding="utf-8")
    val_file.write_text(" ".join([frase] * (ripetizioni // 4)), encoding="utf-8")
    test_file.write_text(" ".join([frase] * (ripetizioni // 4)), encoding="utf-8")

    return train_file, val_file, test_file


def test_build_training_setup_prepara_il_transformer_senza_addestrare(tmp_path):
    train_file, val_file, test_file = _corpus_files(tmp_path)

    trainer, training_dataset, validation_dataset, test_dataset = build_training_setup(
        train_path=train_file,
        val_path=val_file,
        test_path=test_file,
        architecture="transformer",
        context_size=128,
        max_vocab_size=200,
        tokenizer_mode=Tokenizer.MODE_TECHNICAL,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        learning_rate=0.01,
        warmup_steps=2,
        total_steps=10,
        schedule="cosine",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        seed=71,
    )

    assert isinstance(trainer.model, MicroTransformer)
    assert trainer.model.max_len == 128
    assert trainer.tokenizer.mode == Tokenizer.MODE_TECHNICAL
    assert isinstance(trainer.scheduler, LearningRateScheduler)
    assert trainer.gradient_accumulation_steps == 2
    assert trainer.max_grad_norm == 1.0

    assert training_dataset.context_size == 128
    assert validation_dataset.context_size == 128
    assert test_dataset is not None

    # Nessun addestramento è stato eseguito.
    assert trainer.global_step == 0
    assert trainer.epoch == 0
    assert trainer.history == []


def test_build_training_setup_prepara_il_modello_a_contesto_singolo(tmp_path):
    train_file, val_file, _ = _corpus_files(tmp_path)

    trainer, training_dataset, _, test_dataset = build_training_setup(
        train_path=train_file,
        val_path=val_file,
        architecture="next_token",
        max_vocab_size=100,
        embedding_dim=8,
        hidden_dim=16,
    )

    assert isinstance(trainer.model, NextTokenModel)
    assert trainer.model.embedding_dim == 8
    assert trainer.scheduler is None
    assert test_dataset is None
    assert training_dataset.context_size is None


def test_train_and_save_produce_tutti_gli_artefatti(tmp_path):
    """
    Training brevissimo (1 epoca su un corpus minuscolo) per verificare che
    l'entry point salvi tokenizer, pesi, best model e stato di resume.
    """
    np.random.seed(81)
    train_file, val_file, test_file = _corpus_files(tmp_path)
    output_dir = tmp_path / "artefatti"

    history, best_val_loss, _metrics = train_and_save(
        train_path=train_file,
        val_path=val_file,
        output_dir=output_dir,
        test_path=test_file,
        epochs=1,
        batch_size=4,
        patience=1,
        max_vocab_size=100,
        architecture="transformer",
        context_size=128,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        learning_rate=0.01,
        warmup_steps=1,
        schedule="cosine",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        seed=81,
        verbose=False,
    )

    assert len(history) == 1
    assert best_val_loss > 0
    assert history[0]["epoch"] == 1

    checkpoints_dir = output_dir / "checkpoints"
    for nome in ("tokenizer.pkl", "tokenizer.json", "model.pkl",
                 "best_model.json", "last_training_state.pkl"):
        assert (checkpoints_dir / nome).exists(), f"artefatto mancante: {nome}"


def test_train_and_save_supporta_il_resume(tmp_path):
    np.random.seed(82)
    train_file, val_file, _ = _corpus_files(tmp_path)
    output_dir = tmp_path / "run1"

    train_and_save(
        train_path=train_file,
        val_path=val_file,
        output_dir=output_dir,
        epochs=1,
        batch_size=4,
        patience=2,
        max_vocab_size=100,
        architecture="transformer",
        context_size=128,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        learning_rate=0.01,
        seed=82,
        verbose=False,
    )

    state_path = output_dir / "checkpoints" / "last_training_state.pkl"
    assert state_path.exists()

    output_dir_2 = tmp_path / "run2"
    history = train_and_save(
        train_path=train_file,
        val_path=val_file,
        output_dir=output_dir_2,
        epochs=1,
        batch_size=4,
        patience=2,
        max_vocab_size=100,
        architecture="transformer",
        context_size=128,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        learning_rate=0.01,
        seed=82,
        resume_path=str(state_path),
        verbose=False,
    )[0]

    assert history[0]["epoch"] == 2


def test_train_and_save_integra_dataset_dialoghi_e_metriche(tmp_path):
    """
    Verifica l'integrazione del dataset conversazionale (FASE 4): il training
    diventa "misto" (testo generale + dialoghi JARVIS) e vengono esposte le
    metriche aggiuntive (unknown token rate e dialogue loss).
    """
    np.random.seed(83)
    train_file, val_file, test_file = _corpus_files(tmp_path)

    dialogue_file = tmp_path / "dialogue.txt"
    templates = [
        "Utente: ciao\nAssistente: Ciao, sono JARVIS. Come posso assisterti?\n\n",
        "Utente: chi sei?\nAssistente: Sono JARVIS, un assistente personale.\n\n",
        "Utente: cosa sai fare?\nAssistente: Posso analizzare informazioni e gestire strumenti.\n\n",
        "Utente: buongiorno\nAssistente: Buongiorno. Sono JARVIS e ti aiuto con le attività.\n\n",
        "Utente: grazie\nAssistente: Di nulla. Sono qui per assisterti quando serve.\n\n",
        "Utente: come stai?\nAssistente: Sono operativo e pronto ad aiutarti ora.\n\n",
        "Utente: aiutami\nAssistente: Certamente. Dimmi cosa devo fare per te.\n\n",
        "Utente: chi ti ha creato?\nAssistente: Sono stato sviluppato internamente come Neural Core proprietario.\n\n",
    ]
    dialogue_file.write_text("".join(templates * 5), encoding="utf-8")

    output_dir = tmp_path / "run_dlg"

    history, best_val_loss, metrics = train_and_save(
        train_path=train_file,
        val_path=val_file,
        output_dir=output_dir,
        test_path=test_file,
        epochs=1,
        batch_size=4,
        patience=2,
        max_vocab_size=100,
        architecture="transformer",
        context_size=128,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        learning_rate=0.01,
        seed=83,
        dialogue_path=str(dialogue_file),
        verbose=False,
    )

    # Il dataset misto contiene più campioni di quello solo-generale.
    main_inputs, _ = _main_only_inputs(train_file)

    assert metrics["dialogue_loss"] is not None
    assert metrics["dialogue_loss"] > 0
    assert 0.0 <= metrics["unknown_token_rate_dialogue"] <= 1.0
    assert metrics["mixed_training_samples"] > len(main_inputs)


def _main_only_inputs(train_file):
    from neural_core.dataset import Dataset
    from neural_core.tokenizer import Tokenizer

    tok = Tokenizer(max_vocab_size=100)
    train_text = train_file.read_text(encoding="utf-8")
    tok.fit(train_text)
    ds = Dataset(tok.encode(train_text), context_size=128, stride=128)
    return ds.get_arrays()


