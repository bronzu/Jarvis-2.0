import numpy as np

from cognitive_core.coordinator import CognitiveCoordinator
from cognitive_core.memory.unified_memory import UnifiedMemory
from neural_core.model_interface import NeuralModelInterface
from neural_core.network import MicroTransformer
from neural_core.tokenizer import Tokenizer


def _transformer_interface():
    tokenizer = Tokenizer()
    tokenizer.fit(
        "jarvis analizza il sistema neurale interno e produce una risposta "
        "coerente con la memoria e gli strumenti disponibili"
    )
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=2, max_len=128, seed=51,
    )
    return NeuralModelInterface(model=model, tokenizer=tokenizer)


def test_coordinator_preserves_reasoning_result_as_episode():
    problem = {"facts": ["a"], "rules": [{"if": ["a"], "then": "b"}], "goal": "b"}
    coordinator = CognitiveCoordinator()
    
    # Processa il problema (senza Neural Core, userà Reasoner)
    result = coordinator.process(problem)
    
    # Verifica che il risultato sia dal reasoner
    assert result is not None
    assert result["proved"]
    
    # Verifica che episodic_memory registri l'operazione
    recalled = coordinator.episodic_memory.recall()
    assert len(recalled) > 0
    
    # Il primo elemento della memoria deve contenere il resultado
    first_episode = recalled[0]
    assert "outcome" in first_episode
    outcome = first_episode["outcome"]
    
    # Verifica che il risultato sia stato salvato
    assert "result" in outcome or "conclusion" in outcome or outcome.get("proved")


# ======================================================================
# INTEGRAZIONE CON IL TRANSFORMER PROPRIETARIO
# ======================================================================


def test_coordinator_usa_il_transformer_tramite_neural_interface():
    """
    Il Cognitive Core deve funzionare con il nuovo Transformer senza modifiche:
    comunica solo tramite `NeuralModelInterface`.
    """
    interface = _transformer_interface()
    coordinator = CognitiveCoordinator(neural_interface=interface)

    result = coordinator.process("jarvis produce una risposta")

    assert isinstance(result, dict)
    assert result["source"] == "neural_core"
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    assert result.get("fallback") is not True


def test_coordinator_con_transformer_e_memoria_unificata(tmp_path):
    interface = _transformer_interface()
    memory = UnifiedMemory(disk_path=str(tmp_path / "memoria.json"))
    coordinator = CognitiveCoordinator(neural_interface=interface, memory=memory)

    coordinator.process("il sistema neurale interno produce una risposta")
    result = coordinator.process("il sistema neurale interno")

    assert isinstance(result, dict)
    assert len(result["response"]) > 0
    assert len(memory.search_semantic("sistema neurale")) > 0


def test_neural_priority_disattivabile_con_transformer():
    interface = _transformer_interface()
    coordinator = CognitiveCoordinator(neural_interface=interface)
    coordinator.set_neural_priority(False)

    result = coordinator.process("jarvis produce una risposta")

    assert coordinator.is_neural_priority_enabled() is False
    assert result["source"] != "neural_core"


def test_prompt_arricchito_oltre_la_context_window_non_rompe_il_coordinator():
    """
    Il contesto aggiunto dal Cognitive Core può superare la context window: il
    Neural Core deve troncare invece di andare in errore.
    """
    tokenizer = Tokenizer()
    tokenizer.fit(" ".join(f"parola{i}" for i in range(40)))
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=128, seed=52,
    )
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
    coordinator = CognitiveCoordinator(neural_interface=interface)

    prompt = " ".join(f"parola{i % 40}" for i in range(300))
    result = coordinator.process(prompt)

    assert isinstance(result, dict)
    assert result["source"] == "neural_core"
    assert len(result["response"]) > 0

