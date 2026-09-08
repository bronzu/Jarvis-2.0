import numpy as np
import pytest

from executive_system.router import ExecutiveRouter
from neural_core.loss import CrossEntropyLoss
from neural_core.model_interface import NeuralModelInterface
from neural_core.network import MicroTransformer
from neural_core.optimizer import Adam
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import NextTokenModel, Trainer


def test_router_routes_local_and_internal_systems():
    router = ExecutiveRouter()

    local_route = router.route("Spiega in breve il tempo meteorologico")
    internal_route = router.route("Ciao, come stai?")

    assert local_route["route"] == "local"
    assert local_route["provider"] == "internal"
    assert internal_route["route"] == "local"
    assert internal_route["provider"] == "internal"


def test_router_routes_to_internal_systems_never_external():
    """Il router instrada SEMPRE a sistemi interni, mai a modelli esterni."""
    router = ExecutiveRouter()

    complex_route = router.route("Analizza un piano strategico multi-step con full context e risorse esterne")
    identity_route = router.route("Chi ti ha creato?")
    memory_route = router.route("Ricorda questo momento importante")
    knowledge_route = router.route("Cosa sappiamo della conoscenza")

    assert complex_route["route"] in {"reasoning", "local", "memory", "knowledge"}
    assert complex_route["provider"] == "internal"
    assert identity_route["route"] == "identity"
    assert identity_route["provider"].lower() == "jarvis"
    assert memory_route["route"] == "memory"
    assert knowledge_route["route"] == "knowledge"


def test_router_requires_permission_for_generated_code_execution():
    router = ExecutiveRouter()

    with pytest.raises(PermissionError):
        router.execute_generated_code("print('hello')")

    assert router.execute_generated_code("print('hello')", approved=True) == "print('hello')"


def test_router_never_uses_external_model():
    """Il router non usa mai provider esterni (ollama/openai)."""
    router = ExecutiveRouter()
    result = router.generate("Analizza una strategia complessa multi-step per il sistema")
    assert result["provider"] == "internal"
    assert result["route"] in {"reasoning", "local", "identity", "memory", "knowledge"}
    assert "response" in result


# ======================================================================
# INTEGRAZIONE CON LE ARCHITETTURE DEL NEURAL CORE
# ======================================================================


def test_router_carica_pesi_transformer_da_disco(tmp_path):
    """
    `_load_model` deve riconoscere i checkpoint del Transformer proprietario e
    ricostruirlo con gli stessi pesi.
    """
    router = ExecutiveRouter()

    tokenizer = Tokenizer()
    tokenizer.fit("il sistema neurale interno di jarvis")
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=128, seed=61,
    )
    trainer = Trainer(model, tokenizer, CrossEntropyLoss(), Adam())

    weights_path = tmp_path / "model.pkl"
    trainer.save_weights(str(weights_path))

    router.tokenizer = tokenizer
    caricato = router._load_model(weights_path)

    assert isinstance(caricato, MicroTransformer)
    assert caricato.config() == model.config()

    token_ids = tokenizer.encode("il sistema")
    assert np.allclose(caricato.forward(token_ids), model.forward(token_ids))


def test_router_ripiega_su_modello_nuovo_se_i_pesi_sono_corrotti(tmp_path):
    router = ExecutiveRouter()

    corrupted = tmp_path / "model.pkl"
    corrupted.write_bytes(b"non-un-pickle-valido")

    model = router._load_model(corrupted)

    assert isinstance(model, NextTokenModel)
    assert model.vocab_size == len(router.tokenizer.vocabulary)


def test_router_carica_tokenizer_json_se_manca_il_pickle(tmp_path):
    router = ExecutiveRouter()

    tokenizer = Tokenizer(mode=Tokenizer.MODE_TECHNICAL)
    tokenizer.fit("il sistema neurale calcola 3.14")
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    caricato = router._load_tokenizer(tmp_path / "tokenizer.pkl")

    assert caricato.mode == Tokenizer.MODE_TECHNICAL
    assert caricato.vocabulary == tokenizer.vocabulary


def test_router_genera_con_il_transformer():
    router = ExecutiveRouter()

    tokenizer = Tokenizer()
    tokenizer.fit("ciao jarvis come stai oggi tutto bene grazie")
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=2, max_len=128, seed=62,
    )
    router.neural_interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
    router.coordinator.neural_interface = router.neural_interface

    result = router.generate("ciao jarvis come stai")

    assert result["provider"] == "internal"
    assert result["model"] == "neural-core"
    assert result["fallback"] is False
    assert len(result["response"]) > 0
    assert router.neural_interface.context_window() == 128
