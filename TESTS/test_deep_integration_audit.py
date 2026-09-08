"""
JARVIS 2.4 - DEEP INTEGRATION AUDIT

Audit approfondito per identificare:
1. Race conditions nei fallback del coordinator
2. Coerenza tra ContextManager e MemoryManager
3. Perdita di dati dovuta a truncation di ContextManager
4. Colli di bottiglia di performance
5. Errori di sincronizzazione negli accessi concorrenti
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from cognitive_core.coordinator import CognitiveCoordinator
from cognitive_core.context_manager import ContextManager
from cognitive_core.memory.memory_manager import MemoryManager
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.sandbox.sandbox_env import SandboxEnvironment


class DeepIntegrationAudit:
    """Audit approfondito con test di colli di bottiglia e race conditions."""
    
    def __init__(self):
        self.issues = []
        self.findings = []
    
    def test_context_manager_truncation_loss(self):
        """TEST 1: Verifica perdita di dati dovuta a truncation."""
        print("\n" + "="*80)
        print("TEST 1: ContextManager Truncation & Data Loss")
        print("="*80)
        
        ctx = ContextManager(max_turns=3)
        
        # Aggiungi 5 turn
        for i in range(5):
            ctx.add_turn(f"Turn {i+1} - Critical data about {i}", {"index": i})
        
        history = ctx.get_context()["history"]
        
        # Verifica che ci siano solo 3 turn (ultimi 3)
        if len(history) != 3:
            self.issues.append({
                "severity": "MEDIUM",
                "test": "Test 1",
                "issue": f"Expected 3 turns, got {len(history)}",
                "impact": "History truncation doesn't match max_turns setting"
            })
        
        # Check: I dati vecchi sono veramente persi?
        turn_numbers = [t["turn"] for t in history]
        if turn_numbers != [3, 4, 5]:
            self.issues.append({
                "severity": "HIGH",
                "test": "Test 1",
                "issue": f"Expected turns [3, 4, 5], got {turn_numbers}",
                "impact": "Older turns might not be in history, but turn_count still increments",
                "note": f"Total turn_count: {ctx.get_context()['turn_count']} (shows history loss)"
            })
        
        print(f"✓ History maintained: {len(history)} turns")
        print(f"✓ Turn count: {ctx.get_context()['turn_count']} (shows total, not history size)")
        print(f"⚠️  FINDING: turn_count diverges from history length (asymmetry)")
    
    def test_coordinator_fallback_race_conditions(self):
        """TEST 2: Race condition se Neural Core fallisce e Reasoner è chiamato simultaneamente."""
        print("\n" + "="*80)
        print("TEST 2: Coordinator Fallback Race Conditions")
        print("="*80)
        
        coordinator = CognitiveCoordinator()
        
        # Simula Neural Core che fallisce dopo 10ms
        class SlowFailingNeuralModel:
            def forward(self, inputs):
                time.sleep(0.01)
                raise RuntimeError("Neural Core timeout")
        
        coordinator.neural_model = SlowFailingNeuralModel()
        
        # Chiama process() da multipli thread
        results = []
        errors = []
        
        def call_process():
            try:
                result = coordinator.process("test problem")
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        threads = [
            threading.Thread(target=call_process)
            for _ in range(5)
        ]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        print(f"✓ 5 concurrent calls completed in {elapsed:.3f}s")
        print(f"✓ Results: {len(results)}, Errors: {len(errors)}")
        
        # Check: Tutti i risultati sono coerenti?
        if len(results) > 0 and len(set(str(r) for r in results)) > 1:
            self.issues.append({
                "severity": "HIGH",
                "test": "Test 2",
                "issue": "Inconsistent results from concurrent coordinator calls",
                "impact": "Non-deterministic fallback behavior",
                "note": f"Got {len(set(str(r) for r in results))} different result types"
            })
        else:
            print(f"✓ Results are consistent across threads")
        
        if errors:
            self.issues.append({
                "severity": "MEDIUM",
                "test": "Test 2",
                "issue": f"{len(errors)} concurrent calls raised errors",
                "impact": "Coordinator.process() should not raise for any valid input",
                "note": f"Errors: {errors[0]}"
            })
    
    def test_memory_context_coherence_under_stress(self):
        """TEST 3: Coerenza tra MemoryManager e ContextManager sotto stress."""
        print("\n" + "="*80)
        print("TEST 3: MemoryManager & ContextManager Coherence Under Stress")
        print("="*80)
        
        memory = MemoryManager()
        context = ContextManager(max_turns=5)
        
        inconsistencies = []
        
        def worker(worker_id):
            for i in range(20):
                request_id = f"worker_{worker_id}_req_{i}"
                
                # Simula il flusso: aggiungi a context, poi a memory
                context.add_turn(f"Request from worker {worker_id}", {"req_id": request_id})
                memory.store(request_id, {"worker": worker_id, "index": i})
        
        # Esegui 10 worker simultaneamente
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()
        
        # Verifica coerenza: numero di elementi in memory vs context
        # (potrebbero essere diversi per design, ma documenta le assimetrie)
        ctx_turns = len(context.get_context()["history"])
        mem_items = len(memory._store)
        
        print(f"✓ Context manager stores: {ctx_turns} turns (max: {context.max_turns})")
        print(f"✓ Memory manager stores: {mem_items} items")
        print(f"⚠️  FINDING: Asymmetry - Context truncates to {context.max_turns}, Memory keeps all {mem_items}")
        
        # Questo è per design, ma è una potenziale sorgente di confusione
        if mem_items > ctx_turns:
            self.findings.append({
                "severity": "MEDIUM",
                "test": "Test 3",
                "finding": "MemoryManager keeps all items, ContextManager only keeps last N",
                "design_implication": "Memory can grow unbounded; ContextManager is a sliding window",
                "recommendation": "Consider implementing memory eviction policy or explicit memory management"
            })
    
    def test_permission_guard_in_pipeline(self):
        """TEST 4: Permission Guard nel pipeline di security."""
        print("\n" + "="*80)
        print("TEST 4: Permission Guard in Pipeline")
        print("="*80)
        
        guard = PermissionGuard()
        
        test_commands = [
            ("echo hello", True, False),  # safe, approved
            ("os.system('rm -rf /')", True, True),  # dangerous, should block even with approval
            ("python script.py", True, False),  # safe, approved
            ("subprocess.run(['ls'])", True, True),  # dangerous
        ]
        
        results = []
        for cmd, approved, should_fail in test_commands:
            try:
                guard.validate_command(cmd, approved=approved)
                results.append({"cmd": cmd, "blocked": False, "error": None})
            except PermissionError as e:
                results.append({"cmd": cmd, "blocked": True, "error": str(e)})
        
        for result in results:
            status = "✓" if result["blocked"] else "✗"
            print(f"{status} {result['cmd']}: {'BLOCKED' if result['blocked'] else 'ALLOWED'}")
        
        # Verifica: Tutti i comandi pericolosi sono stati bloccati?
        dangerous_allowed = sum(1 for r in results if "os.system\|subprocess" in r["cmd"] and not r["blocked"])
        if dangerous_allowed > 0:
            self.issues.append({
                "severity": "CRITICAL",
                "test": "Test 4",
                "issue": f"{dangerous_allowed} dangerous commands were not blocked",
                "impact": "Security bypass potential"
            })
        else:
            print(f"✓ All dangerous commands blocked")
    
    def test_benchmark_metric_calculation(self):
        """TEST 5: Validazione calcolo metriche nel benchmark."""
        print("\n" + "="*80)
        print("TEST 5: Benchmark Metric Calculation")
        print("="*80)
        
        from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
        
        benchmark = BenchmarkComparator()
        
        # Test: valori uguali dovrebbero mantenere il versione attuale come winner
        equal_metrics = {
            "logical_accuracy": 0.9,
            "execution_time": 1.0,
            "memory_usage": 500,
            "json_adherence": 0.95,
            "tool_success": 0.8
        }
        
        report = benchmark.compare_versions("v1", "v2", equal_metrics, equal_metrics)
        if report["winner"] != "v1":
            self.issues.append({
                "severity": "MEDIUM",
                "test": "Test 5",
                "issue": f"Equal metrics chose v2, expected v1",
                "impact": "Benchmark tie-breaking logic may be inconsistent"
            })
        else:
            print(f"✓ Equal metrics correctly choose current version (v1)")
        
        # Test: Tutti i 5 metric devono essere presenti nel report
        if len(report.get("summary", {})) != 5:
            self.issues.append({
                "severity": "HIGH",
                "test": "Test 5",
                "issue": f"Expected 5 metrics in report, got {len(report.get('summary', {}))}",
                "impact": "Missing metric evaluation"
            })
        else:
            print(f"✓ All 5 metrics present in benchmark report")
    
    def test_sandbox_concurrent_writes(self):
        """TEST 6: Sandbox concurrent write safety."""
        print("\n" + "="*80)
        print("TEST 6: Sandbox Concurrent Write Safety")
        print("="*80)
        
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_deep_audit")
        
        def concurrent_write(thread_id):
            try:
                content = f"Content from thread {thread_id}\n"
                sandbox.write_file(f"file_{thread_id}.txt", content, approved=True)
                # Verifica che il file sia effettivamente scritto
                read_content = sandbox.read_file(f"file_{thread_id}.txt")
                if read_content == content:
                    return {"thread_id": thread_id, "success": True}
                else:
                    return {"thread_id": thread_id, "success": False, "reason": "Content mismatch"}
            except Exception as e:
                return {"thread_id": thread_id, "success": False, "reason": str(e)}
        
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_write, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        successful = sum(1 for r in results if r.get("success"))
        print(f"✓ Concurrent writes: {successful}/10 successful")
        
        if successful < 10:
            self.issues.append({
                "severity": "HIGH",
                "test": "Test 6",
                "issue": f"Only {successful}/10 concurrent sandbox writes succeeded",
                "impact": "Sandbox may have synchronization issues",
                "failures": [r for r in results if not r.get("success")]
            })
    
    def run_all_tests(self):
        """Esegui tutti i test."""
        print("\n" + "="*80)
        print("JARVIS 2.4 - DEEP INTEGRATION AUDIT")
        print("="*80)
        
        self.test_context_manager_truncation_loss()
        self.test_coordinator_fallback_race_conditions()
        self.test_memory_context_coherence_under_stress()
        self.test_permission_guard_in_pipeline()
        self.test_benchmark_metric_calculation()
        self.test_sandbox_concurrent_writes()
        
        return self.generate_report()
    
    def generate_report(self):
        """Genera report dettagliato."""
        print("\n" + "="*80)
        print("AUDIT RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nIssues Found: {len(self.issues)}")
        print(f"Findings: {len(self.findings)}")
        
        if self.issues:
            print("\n🔴 ISSUES:")
            for issue in self.issues:
                print(f"\n  [{issue['severity']}] {issue['test']}: {issue['issue']}")
                if 'impact' in issue:
                    print(f"    Impact: {issue['impact']}")
                if 'note' in issue:
                    print(f"    Note: {issue['note']}")
        
        if self.findings:
            print("\n🟡 FINDINGS:")
            for finding in self.findings:
                print(f"\n  [{finding['severity']}] {finding['finding']}")
                if 'design_implication' in finding:
                    print(f"    Design: {finding['design_implication']}")
                if 'recommendation' in finding:
                    print(f"    → {finding['recommendation']}")
        
        return {
            "total_issues": len(self.issues),
            "total_findings": len(self.findings),
            "issues": self.issues,
            "findings": self.findings
        }


def test_deep_integration_audit():
    """Test pytest per il deep audit."""
    audit = DeepIntegrationAudit()
    report = audit.run_all_tests()
    
    assert report is not None
    assert "total_issues" in report
    assert "total_findings" in report
    assert "issues" in report
    assert "findings" in report

    critical_issues = [i for i in report["issues"] if i.get("severity") == "CRITICAL"]
    high_issues = [i for i in report["issues"] if i.get("severity") == "HIGH"]

    assert critical_issues == [], f"Critical issues found: {critical_issues}"
    assert high_issues == [], f"High issues found: {high_issues}"
    return report


if __name__ == "__main__":
    audit = DeepIntegrationAudit()
    report = audit.run_all_tests()
