"""
ARCHITECTURAL INDEPENDENCE TESTS
=================================

Verifica che JARVIS NON dipenda da modelli esterni (Ollama/OpenAI) come cervello.

TEST OBBLIGATORI:
1. Executive System non importa: ollama, openai, external provider
2. Cognitive Core comunica solo con Neural Core Interface
3. JARVIS si inizializza senza connessione internet
4. L'assenza di Ollama non impedisce il caricamento dell'architettura
5. Gli external models sono opzionali e non obbligatori
6. L'identità di JARVIS non proviene da external model
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_python_source(relative_path):
    """Legge il sorgente Python di un modulo dal progetto."""
    file_path = PROJECT_ROOT / relative_path
    return file_path.read_text(encoding="utf-8")


def _module_imports(relative_path, forbidden_names):
    """Verifica che un modulo non importi direttamente i nomi proibiti."""
    source = _read_python_source(relative_path)
    tree = ast.parse(source)

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                parts = node.module.split(".")
                imported_names.add(parts[0])
                imported_names.add(parts[-1])
            for alias in node.names:
                imported_names.add(alias.name)

    violations = imported_names.intersection(forbidden_names)
    return violations


# ==============================================================================
# TEST 1: Executive System non importa ollama/openai/external provider
# ==============================================================================

class TestExecutiveSystemNoExternalModels:
    """Executive System non deve conoscere modelli esterni."""

    def test_router_does_not_import_ollama(self):
        """ExecutiveRouter non importa 'ollama'."""
        violations = _module_imports("executive_system/router.py", {"ollama"})
        assert not violations, f"ExecutiveRouter importa 'ollama': {violations}"

    def test_router_does_not_import_openai(self):
        """ExecutiveRouter non importa 'openai'."""
        violations = _module_imports("executive_system/router.py", {"openai"})
        assert not violations, f"ExecutiveRouter importa 'openai': {violations}"

    def test_router_does_not_import_llm_client(self):
        """ExecutiveRouter non importa llm_client."""
        violations = _module_imports("executive_system/router.py", {"llm_client", "LLMClient"})
        assert not violations, f"ExecutiveRouter importa LLMClient: {violations}"

    def test_router_has_no_ollama_config_attributes(self):
        """ExecutiveRouter non ha attributi tipo ollama_host/ollama_port/ollama_model."""
        from executive_system.router import ExecutiveRouter
        router = ExecutiveRouter()
        forbidden_attrs = [
            "ollama_host",
            "ollama_port",
            "ollama_model",
            "ollama_base_url",
            "fallback_model",
        ]
        for attr in forbidden_attrs:
            assert not hasattr(router, attr), f"ExecutiveRouter ha attributo vietato: {attr}"

    def test_router_only_uses_internal_provider(self):
        """Il router restituisce SEMPRE provider=internal (mai ollama/openai)."""
        from executive_system.router import ExecutiveRouter
        router = ExecutiveRouter()
        result = router.generate("Analizza una strategia complessa multi-step per il piano")
        assert result["provider"] == "internal"
        assert result["route"] in {"reasoning", "local", "identity", "memory", "knowledge"}


# ==============================================================================
# TEST 2: Cognitive Core comunica solo con Neural Core Interface
# ==============================================================================

class TestCognitiveCoreUsesNeuralInterfaceOnly:
    def test_cognitive_core_uses_neural_model_interface(self):
        """Cognitive Core usa NeuralModelInterface, non LLM esterni."""
        from neural_core.model_interface import NeuralModelInterface

        # Verifica che l'interfaccia esista e abbia i metodi obbligatori
        required_methods = ["generate", "train", "evaluate", "save", "load"]
        for method in required_methods:
            assert hasattr(NeuralModelInterface, method), f"NeuralModelInterface manca: {method}"

    def test_cognitive_core_imports_no_external_llm(self):
        """Nessun modulo cognitive_core importa llm_client/ollama/openai."""
        cognitive_files = [
            "cognitive_core/coordinator.py",
            "cognitive_core/reasoning/reasoner.py",
            "cognitive_core/memory/memory_manager.py",
            "cognitive_core/context_manager.py",
        ]
        for file in cognitive_files:
            violations = _module_imports(file, {"ollama", "openai", "llm_client", "LLMClient"})
            assert not violations, f"{file} importa moduli vietati: {violations}"


# ==============================================================================
# TEST 3: JARVIS si inizializza senza connessione internet
# ==============================================================================

class TestJarvisInitializesWithoutInternet:
    def test_neural_core_interface_initializes_offline(self):
        """NeuralModelInterface si inizializza senza rete."""
        from neural_core.model_interface import NeuralModelInterface
        interface = NeuralModelInterface()
        assert interface is not None
        assert interface.status()["external_model_required"] is False

    def test_executive_router_initializes_offline(self):
        """ExecutiveRouter si inizializza senza rete."""
        from executive_system.router import ExecutiveRouter
        router = ExecutiveRouter()
        assert router is not None
        assert router.neural_interface is not None

    def test_voice_interface_initializes_offline(self):
        """VoiceInterface si inizializza senza rete."""
        from interface_layer.voice_interface import VoiceInterface
        voice = VoiceInterface()
        assert voice is not None


# ==============================================================================
# TEST 4: L'assenza di Ollama non impedisce il caricamento dell'architettura
# ==============================================================================

class TestNoOllamaRequired:
    def test_architecture_loads_without_ollama(self):
        """Importare e usare i moduli principali senza che Ollama sia presente."""
        from neural_core.model_interface import NeuralModelInterface
        from executive_system.router import ExecutiveRouter
        from interface_layer.voice_interface import VoiceInterface
        from cognitive_core.coordinator import CognitiveCoordinator

        interface = NeuralModelInterface()
        router = ExecutiveRouter(neural_interface=interface)
        voice = VoiceInterface(executive_router=router)
        coordinator = CognitiveCoordinator()

        assert interface.status()["type"] == "internal"
        assert coordinator.process("test request") is not None

    def test_router_uses_internal_model_without_external_provider(self):
        """Il router risponde usando solo il Neural Core interno."""
        from executive_system.router import ExecutiveRouter
        from neural_core.network import MicroTransformer
        from neural_core.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit("ciao come stai oggi")
        model = MicroTransformer(vocab_size=len(tokenizer.vocabulary))
        router = ExecutiveRouter()
        router.neural_interface = type(router.neural_interface)(
            model=model, tokenizer=tokenizer
        )
        result = router.generate("Ciao, come stai?")
        assert "response" in result
        assert result["provider"] == "internal"

    def test_generate_identity_question_without_ollama(self):
        """Le domande di identità sono SEMPRE risolte senza Ollama."""
        from executive_system.router import ExecutiveRouter
        router = ExecutiveRouter()
        result = router.generate("Chi ti ha creato?")
        assert result["route"] == "identity"
        assert result["provider"].lower() == "jarvis"
        assert "Ollama" not in result["response"]
        assert "Riccardo Bronzuoli" in result["response"]


# ==============================================================================
# TEST 5: External models opzionali
# ==============================================================================

class TestExternalModelsAreOptional:
    def test_external_models_disabled_by_default(self):
        """Il config.yaml non contiene provider esterni (ollama/openai)."""
        config_path = PROJECT_ROOT / "config.yaml"
        content = config_path.read_text(encoding="utf-8")

        assert "external_models:" not in content
        assert "ollama:" not in content
        assert "openai:" not in content
        assert "default_provider: internal" in content

    def test_llm_client_defaults_to_internal(self):
        """LLMClient (deprecato) defaulta a internal, non a ollama."""
        from neural_core.llm_client import LLMClient
        client = LLMClient()
        assert client.default_provider == "internal"
        assert client.is_enabled("ollama") is False  # Ollama opzionale e disabilitato
        assert client.is_enabled("openai") is False


# ==============================================================================
# TEST 6: L'identità non è mai external model
# ==============================================================================

class TestIdentityIsNotExternalModel:
    def test_identity_creator_from_config(self):
        """Il creatore di JARVIS proviene dalla configurazione interna."""
        from CORE.system.identity import get_canonical_identity
        identity = get_canonical_identity()
        assert identity.creator == "Riccardo Bronzuoli"
        assert "Ollama" not in identity.creator

    def test_identity_system_type_not_ollama(self):
        """system_type non deve mai essere ollama."""
        from CORE.system.identity import get_canonical_identity
        identity = get_canonical_identity()
        assert "ollama" not in identity.system_type.lower()

    def test_no_module_says_ollama_is_jarvis(self):
        """Nessun modulo sorgente dichiara che Ollama è JARVIS."""
        forbidden_phrases = [
            "Ollama è JARVIS",
            "Ollama è il cervello",
            "Ollama ha creato JARVIS",
            "JARVIS is Ollama",
            "created by Ollama",
        ]
        source_dirs = ["CORE", "executive_system", "cognitive_core", "neural_core", "interface_layer"]
        for directory in source_dirs:
            base_dir = PROJECT_ROOT / directory
            if not base_dir.exists():
                continue
            for file in base_dir.rglob("*.py"):
                if "test" in file.name:
                    continue
                content = file.read_text(encoding="utf-8", errors="ignore")
                for phrase in forbidden_phrases:
                    assert phrase not in content, f"{file} contiene frase vietata: {phrase}"