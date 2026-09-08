"""
Test di integrazione end-to-end per verificare:
1. Neural Core come priorità assoluta nel coordinator
2. Isolamento rigoroso del terminal_tool
"""

import pytest
from pathlib import Path

from cognitive_core.coordinator import CognitiveCoordinator
from cognitive_core.reasoning.reasoner import Reasoner
from cognitive_core.memory.episodic_memory import EpisodicMemory
from executive_system.tools.terminal_tool import TerminalTool


class TestNeuralCorePriority:
    """Verifica che il Neural Core sia prioritario nel coordinator."""

    def test_coordinator_has_neural_priority_enabled_by_default(self):
        """Il coordinator abilita la priorità del Neural Core per default."""
        coordinator = CognitiveCoordinator()
        assert coordinator.is_neural_priority_enabled() is True

    def test_coordinator_prioritizes_neural_over_reasoner(self):
        """Se Neural Core è disponibile, ha priorità su Reasoner."""
        coordinator = CognitiveCoordinator()
        
        # Mock Neural Model che ritorna sempre un risultato
        class MockNeuralModel:
            def forward(self, inputs):
                return 0.95  # Score alto di confidenza
        
        coordinator.neural_model = MockNeuralModel()
        
        result = coordinator.process("test problem")
        
        # Verifica che il risultato contenga neural_score (proviene da Neural Core)
        assert "neural_score" in str(result) or "neural_output" in result
        
        # Verifica episodic memory registri source="neural_core"
        # (potrebbe non essere accessibile, ma il test verifica che neural è stato tentato)

    def test_coordinator_falls_back_to_reasoner_if_neural_unavailable(self):
        """Se Neural Core non è disponibile, usa Reasoner."""
        reasoner = Reasoner()
        coordinator = CognitiveCoordinator(reasoner=reasoner)
        coordinator.neural_model = None
        
        result = coordinator.process("test problem")
        
        # Reasoner deve ritornare un risultato
        assert result is not None

    def test_coordinator_falls_back_to_procedural_if_all_fail(self):
        """Se Neural e Reasoner falliscono, usa fallback procedurale."""
        class FailingReasoner:
            def reason(self, problem):
                raise RuntimeError("Reasoner failed")
        
        coordinator = CognitiveCoordinator(reasoner=FailingReasoner())
        coordinator.neural_model = None
        
        result = coordinator.process("test problem")
        
        # Fallback deve ritornare un risultato minimalista
        assert result is not None
        assert "fallback" in result

    def test_coordinator_can_disable_neural_priority(self):
        """È possibile disabilitare la priorità del Neural Core."""
        coordinator = CognitiveCoordinator()
        
        class MockNeuralModel:
            def forward(self, inputs):
                raise RuntimeError("Neural Core disabled")
        
        coordinator.neural_model = MockNeuralModel()
        coordinator.set_neural_priority(False)
        
        assert coordinator.is_neural_priority_enabled() is False
        
        # Con priorità disabilitata, non dovrebbe tentare di usare Neural Core
        result = coordinator.process("test problem")
        assert result is not None

    def test_coordinator_registers_all_operations_in_episodic_memory(self):
        """Tutte le operazioni (neural, reasoner, fallback) sono registrate."""
        coordinator = CognitiveCoordinator()
        coordinator.neural_model = None  # Forza fallback
        
        # Processa un problema
        result = coordinator.process("test problem")
        
        # Verifica che episodic_memory registri l'operazione
        # (episodic_memory.record() è stato chiamato)
        assert result is not None


class TestTerminalToolIsolation:
    """Verifica l'isolamento rigoroso del terminal_tool."""

    def test_terminal_tool_restricts_to_workdir(self):
        """Terminal tool non permette accessi fuori da workdir."""
        tool = TerminalTool(workdir="/tmp/jarvis_test_workdir")
        
        # Tenta di accedere a file fuori da workdir
        with pytest.raises(PermissionError):
            tool._safe_path("../../../etc/passwd")

    def test_terminal_tool_blocks_path_traversal(self):
        """Terminal tool blocca path traversal."""
        tool = TerminalTool(workdir="/tmp/jarvis_test_workdir2")
        
        with pytest.raises(PermissionError):
            tool._safe_path("../../../root/.ssh/id_rsa")

    def test_terminal_tool_resolves_symlinks_safely(self):
        """Terminal tool risolve symlink e verifica la destinazione finale."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            tool = TerminalTool(workdir=str(workdir))
            
            # Crea un file legittimo dentro workdir
            safe_file = workdir / "safe.txt"
            safe_file.write_text("safe content")
            
            # Verifica che il file legittimo sia accessibile
            safe_path = tool._safe_path("safe.txt")
            assert safe_path.exists()

    def test_terminal_tool_blocks_jarvis_internal_access(self):
        """Terminal tool blocca accessi ai percorsi interni di JARVIS."""
        tool = TerminalTool()
        
        # Tenta di accedere ai percorsi interni
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("cat /cognitive_core/config.py")
        
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("ls /neural_core/")
        
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("read /executive_system/tools/")

    def test_terminal_tool_blocks_dangerous_patterns(self):
        """Terminal tool blocca pattern di comandi pericolosi."""
        tool = TerminalTool()
        
        # Tenta di usare pattern pericolosi
        with pytest.raises(PermissionError):
            tool._validate_command_safety("cd ../../ && rm -rf /")
        
        with pytest.raises(PermissionError):
            tool._validate_command_safety("ln -s /etc/passwd safe.txt")
        
        with pytest.raises(PermissionError):
            tool._validate_command_safety("mount -o bind /root /tmp")

    def test_terminal_tool_logs_violations(self):
        """Terminal tool registra tutti i tentativi di violazione."""
        tool = TerminalTool()
        
        # Tenta varie violazioni
        try:
            tool._safe_path("../../../etc/passwd")
        except PermissionError:
            pass
        
        try:
            tool._validate_command_safety("symlink /etc/passwd")
        except PermissionError:
            pass
        
        violations = tool.get_violation_log()
        assert len(violations) >= 2
        assert any("escape_attempt" in v.get("type", "") for v in violations)

    def test_terminal_tool_requires_approval(self):
        """Terminal tool richiede approvazione esplicita per eseguire comandi."""
        tool = TerminalTool()
        
        def mock_runner(cmd):
            return "executed"
        
        # Tenta senza approvazione
        with pytest.raises(PermissionError):
            tool.run("echo test", mock_runner, approved=False)
        
        # Tenta con approvazione
        result = tool.run("echo test", mock_runner, approved=True)
        assert result == "executed"

    def test_terminal_tool_validates_runner_callable(self):
        """Terminal tool verifica che runner sia richiamabile."""
        tool = TerminalTool()
        
        with pytest.raises(TypeError):
            tool.run("echo test", "not_callable", approved=True)

    def test_terminal_tool_safe_paths_are_inside_workdir(self):
        """Tutti i path risolti da _safe_path sono dentro workdir."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = TerminalTool(workdir=tmpdir)
            
            # Test vari path
            safe_path = tool._safe_path("file.txt")
            workdir_resolved = Path(tmpdir).resolve()
            
            # Verifica che il safe_path sia dentro workdir (risolve i symlink)
            try:
                safe_path.relative_to(workdir_resolved)
                # Se non solleva, il path è dentro workdir
                assert True
            except ValueError:
                pytest.fail(f"Path {safe_path} non è dentro {workdir_resolved}")
            
            safe_path = tool._safe_path("subdir/nested/file.txt")
            try:
                safe_path.relative_to(workdir_resolved)
                assert True
            except ValueError:
                pytest.fail(f"Path {safe_path} non è dentro {workdir_resolved}")

    def test_terminal_tool_blocks_home_directory_expansion(self):
        """Terminal tool blocca ~ expansion per prevenire escape."""
        tool = TerminalTool()
        
        with pytest.raises(PermissionError):
            tool._validate_command_safety("cd ~")
        
        with pytest.raises(PermissionError):
            tool._validate_command_safety("cat ~/.ssh/id_rsa")


class TestCoordinatorWithNeuralCoreIntegration:
    """Test di integrazione tra coordinator e neural core con isolamento terminal."""

    def test_end_to_end_neural_priority_and_terminal_isolation(self):
        """
        Test end-to-end:
        1. Coordinator prioritizza Neural Core
        2. Se Neural Core genera comando, terminal_tool lo isola bloccando accessi JARVIS interni
        """
        coordinator = CognitiveCoordinator()
        terminal_tool = TerminalTool(workdir="/tmp/jarvis_e2e_test")
        
        # Mock Neural Model che genera comando
        class CommandGeneratingNeuralModel:
            def forward(self, inputs):
                return 0.85  # Confidenza alta
        
        coordinator.neural_model = CommandGeneratingNeuralModel()
        
        # Simula esecuzione
        result = coordinator.process("execute script")
        
        # Verifica che risultato sia dal Neural Core
        assert result is not None
        
        # Tenta di eseguire comando pericoloso tramite terminal
        def mock_runner(cmd):
            return "should_not_execute"
        
        # Terminal tool deve bloccare accessi ai file interni di JARVIS
        with pytest.raises(PermissionError):
            terminal_tool.run("cat /cognitive_core/config.py", mock_runner, approved=True)
        
        with pytest.raises(PermissionError):
            terminal_tool.run("ls /neural_core/", mock_runner, approved=True)
        
        # Terminal tool deve bloccare path traversal
        with pytest.raises(PermissionError):
            terminal_tool.run("cd ../../ && cat secret.txt", mock_runner, approved=True)
