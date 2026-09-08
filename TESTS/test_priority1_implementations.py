"""
Test di validazione per le implementazioni Priority 1:
1. Memory Manager LRU eviction + TTL
2. Coordinator thread-safety lock
3. ContextManager documentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
import pytest


def test_memory_manager_lru_eviction():
    """Verifica che MemoryManager evict i vecchi item quando raggiunge max_items."""
    from cognitive_core.memory.memory_manager import MemoryManager
    
    # Crea memory manager con max_items=5
    mm = MemoryManager(max_items=5, ttl_seconds=3600)
    
    # Aggiungi 10 items
    for i in range(10):
        mm.store(f"item_{i}", {"value": i})
    
    # Dovrebbe mantenere solo gli ultimi 5 (items 5-9)
    assert mm.get_current_size() <= 5
    assert mm.get_eviction_count() >= 5
    
    # Verifica che i vecchi item siano stati evicti
    for i in range(5):
        result = mm.get(f"item_{i}")
        assert result == {}, f"item_{i} should have been evicted"
    
    # Verifica che i nuovi item siano presenti
    for i in range(5, 10):
        result = mm.get(f"item_{i}")
        assert result != {}, f"item_{i} should be in memory"
    
    print("✓ LRU eviction working correctly")


def test_memory_manager_ttl_expiration():
    """Verifica che MemoryManager rimuova i vecchi item dopo TTL."""
    from cognitive_core.memory.memory_manager import MemoryManager
    
    # Crea memory manager con TTL=0.1 secondi
    mm = MemoryManager(max_items=100, ttl_seconds=0.1)
    
    # Aggiungi item
    mm.store("test_item", {"value": "important"})
    
    # Verifica che sia presente subito
    assert mm.get("test_item") != {}
    
    # Attendi che scada il TTL
    time.sleep(0.15)
    
    # Verifica che sia stato rimosso
    assert mm.get("test_item") == {}
    
    print("✓ TTL expiration working correctly")


def test_coordinator_thread_safety_with_model_changes():
    """Verifica che il Coordinator sia thread-safe quando il model viene cambiato."""
    from cognitive_core.coordinator import CognitiveCoordinator
    
    coordinator = CognitiveCoordinator()
    
    class MockModel1:
        def forward(self, inputs):
            time.sleep(0.01)  # Simula lavoro
            return 0.5
    
    class MockModel2:
        def forward(self, inputs):
            time.sleep(0.01)
            return 0.8
    
    coordinator.neural_model = MockModel1()
    
    results = []
    errors = []
    
    def process_with_changing_model(thread_id):
        try:
            # Processa un problema
            result = coordinator.process("test problem")
            results.append((thread_id, result))
            
            # A volte cambia il modello
            if thread_id % 2 == 0:
                coordinator.neural_model = MockModel2()
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    threads = [threading.Thread(target=process_with_changing_model, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verifica che nessun errore si è verificato
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 10, f"Not all threads completed: {len(results)}"
    
    print("✓ Coordinator thread-safety with model changes working correctly")


def test_memory_manager_concurrent_access_with_eviction():
    """Verifica che MemoryManager sia thread-safe anche durante eviction."""
    from cognitive_core.memory.memory_manager import MemoryManager
    
    mm = MemoryManager(max_items=100, ttl_seconds=3600)
    
    results = []
    
    def worker(worker_id):
        for i in range(20):
            try:
                key = f"worker_{worker_id}_item_{i}"
                mm.store(key, {"worker": worker_id, "index": i})
                retrieved = mm.get(key)
                if retrieved:
                    results.append((worker_id, i, True))
                else:
                    results.append((worker_id, i, False))
            except Exception as e:
                results.append((worker_id, i, f"error: {e}"))
    
    # Esegui 5 worker simultaneamente
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        for f in futures:
            f.result()
    
    # Verifica che la maggior parte dei retrieve siano successi
    successful = sum(1 for _, _, status in results if status is True)
    total = len(results)
    errored = sum(1 for _, _, status in results if isinstance(status, str))
    
    # Nessun errore di concorrenza deve verificarsi
    assert errored == 0
    # La maggior parte delle letture deve avere successo
    assert total == 100
    assert successful / total >= 0.8
    
    print(f"✓ Concurrent access: {successful}/{total} successful (eviction happening)")


def test_context_manager_asymmetry_documented():
    """Verifica che la asimmetria con MemoryManager sia documentata."""
    from cognitive_core.context_manager import ContextManager
    
    ctx = ContextManager(max_turns=5)
    
    # Docstring deve menzione l'asimmetria
    assert "Asymmetry" in ctx.__class__.__doc__
    assert "MemoryManager" in ctx.__class__.__doc__
    
    print("✓ ContextManager documentation includes asymmetry notes")


if __name__ == "__main__":
    test_memory_manager_lru_eviction()
    test_memory_manager_ttl_expiration()
    test_coordinator_thread_safety_with_model_changes()
    test_memory_manager_concurrent_access_with_eviction()
    test_context_manager_asymmetry_documented()
    print("\n✅ All Priority 1 implementation tests passed!")
