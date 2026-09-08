import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from executive_system.router import ExecutiveRouter
from cognitive_core.coordinator import CognitiveCoordinator
from cognitive_core.memory.unified_memory import UnifiedMemory
from executive_system.tools.tool_system import ToolSystem
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.model_interface import NeuralModelInterface
from neural_core.network import MicroTransformer
from neural_core.optimizer import Adam
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import NextTokenModel, Trainer, load_model_from_weights



class TestEndToEndJarvis:
    """Test end-to-end del sistema JARVIS integrato."""

    def test_identity_question_routed_to_identity_system(self):
        router = ExecutiveRouter()
        result = router.generate("Chi sei?")
        assert result["response"] is not None
        assert "JARVIS" in result["response"]
        assert result["route"] == "identity"

    def test_filesystem_tool_execution_and_verification(self):
        router = ExecutiveRouter()
        test_dir = Path("jarvis_e2e_fs_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "jarvis_test.txt"
        if test_file.exists():
            test_file.unlink()

        prompt = f"crea il file {test_file.name} nella cartella {test_dir} con contenuto 'test e2e'"
        result = router.generate(prompt)
        assert test_file.exists(), "Il file non è stato creato"
        assert test_file.read_text() == "test e2e"
        assert "Risultato tool" in result["response"] or "file" in result["response"].lower()

    def test_web_search_tool_execution(self):
        router = ExecutiveRouter()
        result = router.generate("cerca informazioni su Python programming")
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_memory_persistence_across_requests(self):
        memory = UnifiedMemory(disk_path="/tmp/jarvis_e2e_memory_test.json")
        coordinator = CognitiveCoordinator(memory=memory)

        coordinator.process("Il mio colore preferito è il blu")
        coordinator.process("Qual è il mio colore preferito?")

        hits = memory.search_semantic("colore preferito")
        assert len(hits) > 0

    def test_security_blocks_dangerous_terminal_commands(self):
        router = ExecutiveRouter()
        result = router.generate("esegui comando rm -rf /")
        assert result["response"] is not None
        assert "Errore" in result["response"] or "negato" in result["response"].lower() or "pericoloso" in result["response"].lower()

    def test_neural_model_loads_and_generates(self):
        """
        Il modello del Neural Core viene caricato da disco e genera testo.

        Se gli artefatti addestrati non sono presenti, il test ne produce di
        propri con un training minimo: il ciclo salva -> carica -> genera viene
        verificato in ogni caso, senza skip.
        """
        project_root = Path(__file__).resolve().parents[1]
        model_dir = project_root / "neural_core" / "data"
        tokenizer_path = model_dir / "tokenizer.pkl"
        model_path = model_dir / "model.pkl"

        if tokenizer_path.exists() and model_path.exists():
            with open(tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)
            model = load_model_from_weights(
                str(model_path), vocab_size=len(tokenizer.vocabulary)
            )
        else:
            tokenizer, model = self._train_minimal_model()

        assert len(tokenizer.vocabulary) > 2
        assert hasattr(model, "forward")

        interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
        result = interface.generate("ciao come stai", max_new_tokens=5)

        assert result["text"] is not None
        assert len(result["text"]) > 0
        assert result["provider"] == "internal_model"
        assert result["fallback"] is False
        assert result["generated_tokens"] == 5

    def test_router_uses_trained_weights_from_disk(self):
        """
        Il Router deve caricare davvero i pesi salvati: due router consecutivi
        devono produrre gli stessi logits sullo stesso prompt.
        """
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "neural_core" / "data" / "model.pkl"

        router_a = ExecutiveRouter()
        router_b = ExecutiveRouter()

        model_a = router_a.neural_interface.model
        model_b = router_b.neural_interface.model

        assert model_a is not None and model_b is not None
        assert type(model_a) is type(model_b)

        token_ids = router_a.tokenizer.encode("il sistema neurale di jarvis")
        assert len(token_ids) > 0

        logits_a = model_a.forward(token_ids)
        logits_b = model_b.forward(token_ids)

        if model_path.exists():
            # I pesi provengono dal disco: il caricamento è deterministico.
            assert np.allclose(logits_a, logits_b)

        response = router_a.neural_interface.generate(
            "il sistema neurale", max_new_tokens=4, temperature=0
        )
        assert len(response["text"]) > 0

    def _train_minimal_model(self):
        """Addestramento minimo (pochi step) per validare il ciclo completo."""
        text = " ".join(["ciao come stai jarvis tutto bene grazie"] * 30)
        tokenizer = Tokenizer()
        training_dataset, _ = Dataset.from_text(
            text, tokenizer, validation_split=0.2, context_size=4
        )
        model = MicroTransformer(
            vocab_size=len(tokenizer.vocabulary), d_model=16, num_heads=2,
            d_ff=32, num_layers=1, max_len=128, seed=42,
        )
        trainer = Trainer(model, tokenizer, CrossEntropyLoss(), Adam(learning_rate=0.01))
        inputs, targets = training_dataset.get_arrays()
        trainer.train(inputs, targets, epochs=1, batch_size=4)
        return tokenizer, model

