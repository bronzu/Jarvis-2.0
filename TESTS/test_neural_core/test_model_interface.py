import numpy as np
import pytest

from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.model_interface import NeuralModelInterface
from neural_core.model import Decoder
from neural_core.network import MicroTransformer
from neural_core.optimizer import Adam
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import NextTokenModel, Trainer, load_model_from_weights


def test_generazione_senza_fallback_statico():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao JARVIS test sistema")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary))
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
    result = interface.generate("ciao")
    assert "text" in result
    assert result["provider"] == "internal_model"
    assert result["model"] == "neural-core-internal"
    assert result["fallback"] is False


def test_ciclo_autoregressivo_funzionante():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro cinque sei sette otto nove dieci")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary))
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
    result = interface.generate("uno", max_length=5)
    text = result["text"]
    tokens = text.split()
    assert len(tokens) >= 1
    assert len(tokens) <= 6


def test_stop_token_funzionante():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre stop cinque")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary))
    stop_id = tokenizer.vocabulary["stop"]
    original_sample = Decoder.sample

    def fake_sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
        if len(fake_sample.calls) == 0:
            fake_sample.calls.append(1)
            return stop_id
        return original_sample(logits, temperature, top_k, top_p, rng)

    fake_sample.calls = []
    Decoder.sample = staticmethod(fake_sample)
    try:
        interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
        result = interface.generate("uno", max_length=100, stop_token=stop_id)
        text = result["text"]
        tokens = text.split()
        assert tokens[-1] == "stop"
    finally:
        Decoder.sample = staticmethod(original_sample)


# ======================================================================
# CONTEXT WINDOW IN INFERENCE
# ======================================================================


def test_context_window_esposta_dal_transformer():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), max_len=128)
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    assert interface.context_window() == 128
    assert interface.status()["context_window"] == 128
    assert interface.status()["model_architecture"] == "MicroTransformer"


def test_context_window_none_senza_modello():
    interface = NeuralModelInterface()
    assert interface.context_window() is None


def test_context_window_none_per_modello_a_contesto_singolo():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre")
    model = NextTokenModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=8)
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    assert interface.context_window() is None


def test_crop_to_context_mantiene_gli_ultimi_token():
    interface = NeuralModelInterface()

    assert interface._crop_to_context([1, 2, 3, 4, 5], 3) == [3, 4, 5]
    assert interface._crop_to_context([1, 2], 5) == [1, 2]
    assert interface._crop_to_context([1, 2], None) == [1, 2]


def test_generazione_oltre_la_context_window_non_va_in_errore():
    """
    Il ciclo autoregressivo deve troncare la sequenza alla context window del
    modello, altrimenti il positional encoding andrebbe fuori intervallo.
    """
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro cinque sei sette otto nove dieci")
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=128, seed=3,
    )
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    result = interface.generate("uno due", max_new_tokens=6, max_context=4)

    assert result["context_window"] == 4
    assert result["generated_tokens"] == 6
    assert len(result["text"].split()) == 8


def test_prompt_piu_lungo_della_context_window_segnalato():
    tokenizer = Tokenizer()
    tokenizer.fit("alpha beta gamma delta epsilon")
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=128, seed=4,
    )
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    result = interface.generate(
        "alpha beta gamma delta epsilon", max_new_tokens=1, max_context=2
    )

    assert result["prompt_truncated"] is True


def test_generazione_deterministica_con_temperatura_zero():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro cinque")
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=128, seed=5,
    )
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    primo = interface.generate("uno", max_new_tokens=5, temperature=0)
    secondo = interface.generate("uno", max_new_tokens=5, temperature=0)

    assert primo["text"] == secondo["text"]


def test_generate_richiede_modello_e_tokenizer():
    interface = NeuralModelInterface()

    with pytest.raises(ValueError):
        interface.generate(None)

    with pytest.raises(RuntimeError):
        interface.generate("ciao")

    tokenizer = Tokenizer()
    tokenizer.fit("ciao")
    solo_modello = NeuralModelInterface(
        model=MicroTransformer(vocab_size=len(tokenizer.vocabulary))
    )
    with pytest.raises(RuntimeError):
        solo_modello.generate("ciao")


def test_status_riporta_tokenizer_e_parametri():
    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokenizer.fit("il modello calcola 3.14")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, seed=6)
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    status = interface.status()

    assert status["tokenizer_mode"] == Tokenizer.MODE_TECHNICAL
    assert status["vocab_size"] == len(tokenizer.vocabulary)
    assert status["parameter_count"] == model.parameter_count()
    assert status["external_model_required"] is False
    assert interface.is_internal() is True


# ======================================================================
# INTEGRAZIONE: addestramento -> salvataggio -> caricamento -> inference
# ======================================================================


def test_ciclo_completo_training_salvataggio_caricamento_inference(tmp_path):
    """
    Test di integrazione reale: il Transformer viene addestrato per pochi step,
    salvato, ricaricato e usato per generare tramite `NeuralModelInterface`.
    """
    np.random.seed(31)
    text = " ".join(["alpha beta gamma delta epsilon zeta"] * 40)
    tokenizer = Tokenizer()
    training_dataset, validation_dataset = Dataset.from_text(
        text, tokenizer, validation_split=0.2, context_size=8
    )

    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=2, max_len=128, seed=31,
    )
    trainer = Trainer(model, tokenizer, CrossEntropyLoss(), Adam(learning_rate=0.01))

    inputs, targets = training_dataset.get_arrays()
    validation_inputs, validation_targets = validation_dataset.get_arrays()
    history = trainer.train_with_validation(
        inputs, targets, validation_inputs, validation_targets,
        epochs=2, batch_size=4, patience=3,
    )
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]

    weights_path = tmp_path / "model.pkl"
    tokenizer_path = tmp_path / "tokenizer.json"
    trainer.save_weights(str(weights_path))
    tokenizer.save(str(tokenizer_path))

    modello_caricato = load_model_from_weights(str(weights_path))
    tokenizer_caricato = Tokenizer.load(str(tokenizer_path))
    interface = NeuralModelInterface(model=modello_caricato, tokenizer=tokenizer_caricato)

    assert interface.status()["model_available"] is True
    assert interface.context_window() == 128

    result = interface.generate("alpha beta", max_new_tokens=4, temperature=0)

    assert result["provider"] == "internal_model"
    assert result["fallback"] is False
    assert result["generated_tokens"] == 4
    assert len(result["text"].split()) == 6

    # I pesi ricaricati producono esattamente gli stessi logits.
    prompt_ids = np.array([tokenizer_caricato.encode("alpha beta")])
    assert np.allclose(modello_caricato.forward(prompt_ids), model.forward(prompt_ids))


def test_train_ed_evaluate_delegano_al_trainer():
    np.random.seed(32)
    text = " ".join(["uno due tre quattro"] * 30)
    tokenizer = Tokenizer()
    training_dataset, _ = Dataset.from_text(
        text, tokenizer, validation_split=0.2, context_size=4
    )

    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=128, seed=32,
    )
    trainer = Trainer(model, tokenizer, CrossEntropyLoss(), Adam(learning_rate=0.01))
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer, trainer=trainer)

    inputs, targets = training_dataset.get_arrays()
    history = interface.train(inputs, targets, epochs=1, batch_size=4)

    assert len(history) == 1
    assert history[0] > 0
    assert interface.status()["trainer_available"] is True


def test_train_senza_trainer_ne_modello_addestrabile():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary))
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)

    with pytest.raises(RuntimeError):
        interface.train([[1]], [1])

    with pytest.raises(RuntimeError):
        interface.evaluate([[1]], [1])


def test_save_e_load_checkpoint_tramite_interfaccia(tmp_path):
    from neural_core.checkpoint import Checkpoint

    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre")
    model = MicroTransformer(vocab_size=len(tokenizer.vocabulary), d_model=16, seed=8)
    interface = NeuralModelInterface(
        model=model, tokenizer=tokenizer, checkpoint=Checkpoint()
    )

    path = tmp_path / "checkpoint.json"
    interface.save(str(path))
    assert path.exists()

    loaded = interface.load(str(path))
    assert loaded["parameters"]["model_type"] == "MicroTransformer"
    assert loaded["parameters"]["tokenizer_vocab_size"] == len(tokenizer.vocabulary)

    with pytest.raises(FileNotFoundError):
        interface.load(str(tmp_path / "assente.json"))

