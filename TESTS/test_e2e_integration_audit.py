"""
JARVIS 2.4 - LEAD SYSTEMS ENGINEER - END-TO-END INTEGRATION AUDIT

Questo script simula il flusso completo di una richiesta utente attraverso 
l'intera architettura e identifica punti critici, race conditions e colli di bottiglia.

FLUSSO SIMULATO:
1. Percezione: Richiesta utente complessa
2. Inferenza Neural Core: Processamento prioritario
3. Controllo Sicurezza: Permission Guard validation
4. Tool Generation: Sandbox code generation (eventuale)
5. Salvataggio Memoria: MemoryManager + ContextManager
6. Valutazione Benchmark: Metriche di performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import threading
import json
from datetime import datetime

# Import core modules
from cognitive_core.coordinator import CognitiveCoordinator
from cognitive_core.context_manager import ContextManager
from cognitive_core.memory.memory_manager import MemoryManager
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.sandbox.sandbox_env import SandboxEnvironment
from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
from neural_core.model import Neuron, TextGenerator


class E2EIntegrationAudit:
    """Orchestratore di audit end-to-end per JARVIS 2.4."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.audit_log = []
        self.issues_found = []
        self.coordinator = CognitiveCoordinator()
        self.context_manager = ContextManager(max_turns=10)
        self.memory_manager = MemoryManager()
        self.permission_guard = PermissionGuard()
        self.sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_e2e_audit")
        self.benchmark = BenchmarkComparator()
        
        # Setup neural model
        try:
            self.neural_model = Neuron(input_size=64, learning_rate=0.01)
            self.coordinator.neural_model = self.neural_model
        except Exception as e:
            self._log_issue("WARNING", f"Neural model init failed: {e}")
    
    def _log(self, level, message):
        """Log con timestamp."""
        timestamp = datetime.now().isoformat()
        entry = {"timestamp": timestamp, "level": level, "message": message}
        self.audit_log.append(entry)
        print(f"[{timestamp}] [{level}] {message}")
    
    def _log_issue(self, severity, description, location=None, recommendation=None):
        """Registra un issue trovato durante l'audit."""
        issue = {
            "severity": severity,
            "description": description,
            "location": location,
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        self.issues_found.append(issue)
        print(f"\n⚠️  ISSUE [{severity}]: {description}")
        if location:
            print(f"   Location: {location}")
        if recommendation:
            print(f"   Recommendation: {recommendation}\n")
    
    def simulate_user_request(self, user_input):
        """FASE 1: Simula percezione di una richiesta utente complessa."""
        self._log("INFO", f"FASE 1: Percezione richiesta utente: '{user_input}'")
        
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Aggiungi al context manager
        context_turn = self.context_manager.add_turn(user_input, {"request_id": request_id})
        self._log("INFO", f"  ✓ Context registered (turn {context_turn['turn']})")
        
        return request_id, user_input
    
    def neural_core_inference(self, request_id, user_input):
        """FASE 2: Inference nel Neural Core con priorità assoluta."""
        self._log("INFO", f"FASE 2: Inference Neural Core (priorità assoluta)")
        
        # Verifica che Neural Core abbia priorità
        if not self.coordinator.is_neural_priority_enabled():
            self._log_issue("CRITICAL", 
                "Neural Core priority is disabled!",
                location="CognitiveCoordinator.is_neural_priority_enabled()",
                recommendation="Ensure neural priority is always enabled by default")
        
        # Processa la richiesta
        try:
            start = time.time()
            result = self.coordinator.process(user_input)
            elapsed = time.time() - start
            
            self._log("INFO", f"  ✓ Neural inference completed in {elapsed:.4f}s")
            self._log("DEBUG", f"    Result: {json.dumps(result, default=str)[:100]}...")
            
            return result, elapsed
        except Exception as e:
            self._log_issue("CRITICAL",
                f"Neural inference failed: {e}",
                location="CognitiveCoordinator.process()",
                recommendation="Implement better error handling and recovery")
            return None, 0.0
    
    def security_validation(self, request_id, inference_result):
        """FASE 3: Validazione sicurezza - Permission Guard."""
        self._log("INFO", f"FASE 3: Validazione sicurezza (Permission Guard)")
        
        # Simula un comando generato dall'inferenza
        command = "list_directory /tmp/jarvis_workspace"
        
        try:
            # Valida il comando
            self.permission_guard.validate_command(command, approved=True)
            self._log("INFO", f"  ✓ Command approved: '{command}'")
            return True
        except PermissionError as e:
            self._log("INFO", f"  ✓ Dangerous command blocked: {e}")
            return False
        except Exception as e:
            self._log_issue("HIGH",
                f"Permission guard error: {e}",
                location="PermissionGuard.validate_command()",
                recommendation="Add defensive programming for edge cases")
            return False
    
    def tool_generation_in_sandbox(self, request_id):
        """FASE 4: Generazione eventuale di tool nella sandbox."""
        self._log("INFO", f"FASE 4: Tool generation in sandbox (eventuale)")
        
        tool_code = """
def calculate_metrics(data):
    '''Generated tool for metric calculation.'''
    return {
        'sum': sum(data),
        'avg': sum(data) / len(data),
        'count': len(data)
    }
"""
        
        try:
            # Scrivi il codice nella sandbox
            tool_path = self.sandbox.write_file(
                f"tool_{request_id}.py",
                tool_code,
                approved=True
            )
            self._log("INFO", f"  ✓ Tool generated in sandbox: {tool_path}")
            
            # Verifica che il file sia davvero nella sandbox
            content = self.sandbox.read_file(f"tool_{request_id}.py")
            assert "calculate_metrics" in content
            self._log("INFO", f"  ✓ Tool verification passed")
            
            return True
        except Exception as e:
            self._log_issue("HIGH",
                f"Tool generation failed: {e}",
                location="SandboxEnvironment.write_file() or read_file()",
                recommendation="Add transaction semantics for tool generation")
            return False
    
    def memory_persistence(self, request_id, inference_result, tool_generated):
        """FASE 5: Salvataggio in memoria con coerenza."""
        self._log("INFO", f"FASE 5: Persistenza in memoria (MemoryManager + ContextManager)")
        
        # Prepara payload completo
        payload = {
            "request_id": request_id,
            "inference_result": inference_result,
            "tool_generated": tool_generated,
            "timestamp": datetime.now().isoformat(),
            "source": "neural_core" if inference_result else "fallback"
        }
        
        try:
            # Salva in memory manager
            mem_result = self.memory_manager.store(request_id, payload)
            self._log("INFO", f"  ✓ Stored in MemoryManager (id={request_id})")
            
            # Verifica coerenza tra MemoryManager e ContextManager
            context = self.context_manager.get_context()
            if context["turn_count"] == 0:
                self._log_issue("HIGH",
                    "ContextManager turn_count is 0 after add_turn",
                    location="ContextManager.add_turn() or get_context()",
                    recommendation="Verify turn counting logic")
            
            # RACE CONDITION CHECK: Thread-safety tra memory manager e context
            # Se due thread scrivono simultaneamente, cosa succede?
            self._check_memory_race_conditions(request_id)
            
            # Check: ContextManager ha max_turns, cosa succede ai dati?
            if context["turn_count"] > self.context_manager.max_turns:
                self._log_issue("MEDIUM",
                    f"Context history might exceed max_turns ({self.context_manager.max_turns})",
                    location="ContextManager.add_turn() truncation logic",
                    recommendation="Verify history truncation doesn't lose critical data")
            
            return True
        except Exception as e:
            self._log_issue("CRITICAL",
                f"Memory persistence failed: {e}",
                location="MemoryManager.store()",
                recommendation="Implement transaction logging for data integrity")
            return False
    
    def _check_memory_race_conditions(self, request_id):
        """Simula accesso concorrente a memoria e context."""
        self._log("DEBUG", "  Checking memory race conditions...")
        
        results = []
        
        def concurrent_store(thread_id, value):
            try:
                self.memory_manager.store(f"{request_id}_thread_{thread_id}", {"value": value})
                results.append((thread_id, "success"))
            except Exception as e:
                results.append((thread_id, f"error: {e}"))
        
        threads = [
            threading.Thread(target=concurrent_store, args=(i, i*100))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        successful = sum(1 for _, status in results if status == "success")
        self._log("DEBUG", f"  ✓ Concurrent stores: {successful}/5 successful")
        
        if successful < 5:
            self._log_issue("HIGH",
                f"Only {successful}/5 concurrent stores succeeded",
                location="MemoryManager.store() with threading",
                recommendation="Verify RLock covers all critical sections")
    
    def benchmark_evaluation(self, inference_result, tool_generated):
        """FASE 6: Valutazione benchmark e metriche."""
        self._log("INFO", f"FASE 6: Benchmark evaluation")
        
        # Metriche simulate per versione attuale
        current_metrics = {
            "logical_accuracy": 0.88,
            "execution_time": 1.25,
            "memory_usage": 450,
            "json_adherence": 0.92,
            "tool_success": 0.85
        }
        
        # Metriche simulate per versione candidata
        candidate_metrics = {
            "logical_accuracy": 0.93,  # Meglio
            "execution_time": 0.95,    # Meglio
            "memory_usage": 380,       # Meglio
            "json_adherence": 0.96,    # Meglio
            "tool_success": 0.90       # Meglio
        }
        
        try:
            report = self.benchmark.compare_versions(
                "v1_current",
                "v2_candidate",
                current_metrics,
                candidate_metrics
            )
            
            winner = report.get("winner")
            self._log("INFO", f"  ✓ Benchmark winner: {winner}")
            
            # Analizza i delta
            for metric, delta_info in report.get("summary", {}).items():
                delta = delta_info.get("delta", 0)
                improvement = "↑" if delta > 0 else "↓"
                self._log("DEBUG", f"    {metric}: {improvement} {abs(delta):.3f}")
            
            # ISSUE: Benchmark non misura il tempo reale
            if "execution_time" in candidate_metrics:
                self._log_issue("MEDIUM",
                    "Benchmark compares provided execution_time values, not real wall-clock time",
                    location="BenchmarkComparator.compare_versions()",
                    recommendation="Add actual timing instrumentation with @timer decorator")
            
            return report
        except Exception as e:
            self._log_issue("HIGH",
                f"Benchmark evaluation failed: {e}",
                location="BenchmarkComparator.compare_versions()",
                recommendation="Add validation for metric types and ranges")
            return None
    
    def run_full_audit(self):
        """Esegui il ciclo completo di audit."""
        self.start_time = time.time()
        self._log("INFO", "=" * 80)
        self._log("INFO", "JARVIS 2.4 END-TO-END INTEGRATION AUDIT STARTED")
        self._log("INFO", "=" * 80)
        
        # Simula richiesta utente complessa
        user_input = "Analizza il sentiment di questi dati, genera un tool di previsione se necessario, e valuta le prestazioni"
        request_id, _ = self.simulate_user_request(user_input)
        
        # Neural Core inference
        inference_result, inference_time = self.neural_core_inference(request_id, user_input)
        
        # Security validation
        security_ok = self.security_validation(request_id, inference_result)
        
        # Tool generation (eventuale)
        tool_generated = self.tool_generation_in_sandbox(request_id) if security_ok else False
        
        # Memory persistence
        memory_ok = self.memory_persistence(request_id, inference_result, tool_generated)
        
        # Benchmark evaluation
        benchmark_report = self.benchmark_evaluation(inference_result, tool_generated)
        
        self.end_time = time.time()
        self._log("INFO", "=" * 80)
        self._log("INFO", "JARVIS 2.4 END-TO-END INTEGRATION AUDIT COMPLETED")
        self._log("INFO", "=" * 80)
        
        return self.generate_report()
    
    def generate_report(self):
        """Genera report finale dell'audit."""
        total_time = self.end_time - self.start_time
        
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_time,
            "total_log_entries": len(self.audit_log),
            "issues_found": len(self.issues_found),
            "issues_by_severity": self._count_issues_by_severity(),
            "critical_issues": [i for i in self.issues_found if i["severity"] == "CRITICAL"],
            "all_issues": self.issues_found
        }
        
        return report
    
    def _count_issues_by_severity(self):
        """Conta gli issue per severità."""
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in self.issues_found:
            severity = issue.get("severity", "LOW")
            if severity in counts:
                counts[severity] += 1
        return counts
    
    def print_final_report(self):
        """Stampa il report finale formattato."""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("AUDIT FINAL REPORT")
        print("=" * 80)
        print(f"\nDuration: {report['total_duration_seconds']:.2f}s")
        print(f"Log Entries: {report['total_log_entries']}")
        print(f"\nIssues Found: {report['issues_found']}")
        print(f"  - CRITICAL: {report['issues_by_severity']['CRITICAL']}")
        print(f"  - HIGH: {report['issues_by_severity']['HIGH']}")
        print(f"  - MEDIUM: {report['issues_by_severity']['MEDIUM']}")
        
        if report["critical_issues"]:
            print("\n🔴 CRITICAL ISSUES:")
            for issue in report["critical_issues"]:
                print(f"\n  • {issue['description']}")
                if issue.get('location'):
                    print(f"    Location: {issue['location']}")
                if issue.get('recommendation'):
                    print(f"    → Fix: {issue['recommendation']}")
        
        print("\n" + "=" * 80)
        
        return report


def main():
    """Entry point dell'audit."""
    audit = E2EIntegrationAudit()
    report = audit.run_full_audit()
    audit.print_final_report()
    return report


def test_e2e_integration_audit():
    """Test pytest per l'audit end-to-end."""
    report = main()
    assert report is not None
    assert "issues_found" in report
    assert "critical_issues" in report
    assert "all_issues" in report
    assert report["issues_found"] >= 0

    critical_issues = report.get("critical_issues", [])
    assert critical_issues == [], f"Critical issues found: {critical_issues}"


if __name__ == "__main__":
    main()
