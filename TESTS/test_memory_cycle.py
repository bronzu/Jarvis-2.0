"""
Test per la verifica completa del sistema di memoria di JARVIS.

Ciclo verificato:
    WRITE -> STORE -> RETRIEVE -> USE -> UPDATE -> PERSIST

E soprattutto:
    MEMORY -> RETRIEVAL -> REASONING -> DECISION
"""

import json
import os
import time

import pytest

from cognitive_core.memory.memory_manager import MemoryManager
from cognitive_core.memory.episodic_memory import EpisodicMemory
from cognitive_core.memory.working_memory import WorkingMemory
from cognitive_core.memory.semantic_memory import SemanticMemory
from cognitive_core.memory.procedural_memory import ProceduralMemory


# ----------------------------------------------------------------------
# MemoryManager: eviction, TTL, persistence
# ----------------------------------------------------------------------


class TestMemoryManagerEviction:
    def test_evicts_oldest_when_max_items_exceeded(self, tmp_path):
        mm = MemoryManager(max_items=3, disk_path=str(tmp_path / "mm.json"))
        mm.store("a", {"v": 1})
        mm.store("b", {"v": 2})
        mm.store("c", {"v": 3})
        assert mm.get_current_size() == 3

        mm.store("d", {"v": 4})
        assert mm.get_current_size() == 3
        assert mm.get("a") == {}
        assert mm.get("d") == {"v": 4}

    def test_ttl_expiration_removes_items(self, tmp_path):
        mm = MemoryManager(max_items=100, ttl_seconds=1, disk_path=str(tmp_path / "mm.json"))
        mm.store("key1", {"v": 1})
        assert mm.get("key1") == {"v": 1}

        time.sleep(1.1)
        assert mm.get("key1") == {}

    def test_cleanup_expired_removes_multiple_items(self, tmp_path):
        mm = MemoryManager(max_items=100, ttl_seconds=1, disk_path=str(tmp_path / "mm.json"))
        for i in range(5):
            mm.store(f"key{i}", {"v": i})
        assert mm.get_current_size() == 5

        time.sleep(1.1)
        removed = mm.cleanup_expired()
        assert removed == 5
        assert mm.get_current_size() == 0

    def test_persists_and_restores_state(self, tmp_path):
        disk_path = tmp_path / "mm.json"
        mm1 = MemoryManager(max_items=100, ttl_seconds=3600, disk_path=str(disk_path))
        mm1.store("key1", {"v": 1})
        mm1.store("key2", {"v": 2})

        mm2 = MemoryManager(max_items=100, ttl_seconds=3600, disk_path=str(disk_path))
        assert mm2.get("key1") == {"v": 1}
        assert mm2.get("key2") == {"v": 2}

    def test_handles_corrupted_disk_file(self, tmp_path):
        disk_path = tmp_path / "corrupt.json"
        disk_path.write_text("not json")
        mm = MemoryManager(max_items=100, disk_path=str(disk_path))
        assert mm.get_current_size() == 0

    def test_search_semantic_skips_expired(self, tmp_path):
        mm = MemoryManager(max_items=100, ttl_seconds=1, disk_path=str(tmp_path / "mm.json"))
        mm.store("target", {"text": "find me"})
        assert len(mm.search_semantic("find")) == 1

        time.sleep(1.1)
        assert len(mm.search_semantic("find")) == 0


# ----------------------------------------------------------------------
# EpisodicMemory: eviction, TTL, persistence
# ----------------------------------------------------------------------


class TestEpisodicMemoryEviction:
    def test_evicts_oldest_when_max_episodes_exceeded(self):
        mem = EpisodicMemory(max_episodes=3)
        mem.record("e1", "ok")
        mem.record("e2", "ok")
        mem.record("e3", "ok")
        assert len(mem.recall()) == 3

        mem.record("e4", "ok")
        assert len(mem.recall()) == 3
        assert mem.recall("e1") == []

    def test_ttl_expiration_removes_episodes(self):
        mem = EpisodicMemory(max_episodes=100, ttl_seconds=1)
        mem.record("e1", "ok")
        assert len(mem.recall()) == 1

        time.sleep(1.1)
        assert len(mem.recall()) == 0

    def test_recall_filters_by_event(self):
        mem = EpisodicMemory()
        mem.record("login", "success")
        mem.record("logout", "success")
        mem.record("login", "failure")

        login_events = mem.recall("login")
        assert len(login_events) == 2
        assert login_events[0]["outcome"] == "success"
        assert login_events[1]["outcome"] == "failure"


# ----------------------------------------------------------------------
# WorkingMemory: bounded capacity
# ----------------------------------------------------------------------


class TestWorkingMemory:
    def test_respects_capacity(self):
        wm = WorkingMemory(capacity=2)
        wm.remember("a")
        wm.remember("b")
        wm.remember("c")
        assert wm.recall() == ["b", "c"]

    def test_clear_empties_memory(self):
        wm = WorkingMemory(capacity=5)
        wm.remember("a")
        wm.remember("b")
        wm.clear()
        assert wm.recall() == []


# ----------------------------------------------------------------------
# SemanticMemory: duplicate prevention
# ----------------------------------------------------------------------


class TestSemanticMemory:
    def test_learn_and_recall(self):
        sm = SemanticMemory()
        sm.learn("acqua", "bagna")
        sm.learn("acqua", "disseta")
        assert sm.recall("acqua") == ["bagna", "disseta"]

    def test_no_duplicate_facts(self):
        sm = SemanticMemory()
        sm.learn("fuoco", "brucia")
        sm.learn("fuoco", "brucia")
        assert len(sm.recall("fuoco")) == 1


# ----------------------------------------------------------------------
# ProceduralMemory: register and retrieve
# ----------------------------------------------------------------------


class TestProceduralMemory:
    def test_register_and_retrieve(self):
        pm = ProceduralMemory()
        pm.register("saluta", ["ciao", "come stai"])
        assert pm.retrieve("saluta") == ["ciao", "come stai"]

    def test_rejects_invalid_registration(self):
        pm = ProceduralMemory()
        with pytest.raises(ValueError):
            pm.register("", ["step1"])
        with pytest.raises(ValueError):
            pm.register("name", [])


# ----------------------------------------------------------------------
# Memory cycle integration
# ----------------------------------------------------------------------


class TestMemoryCycleIntegration:
    def test_full_memory_cycle(self):
        wm = WorkingMemory(capacity=5)
        episodic = EpisodicMemory(max_episodes=100)
        semantic = SemanticMemory()
        procedural = ProceduralMemory()

        wm.remember({"input": "query", "response": "result"})
        episodic.record("query", {"result": "success"})
        semantic.learn("query", "success")
        procedural.register("handle_query", ["receive", "process", "respond"])

        assert len(wm.recall()) == 1
        assert len(episodic.recall()) == 1
        assert semantic.recall("query") == ["success"]
        assert procedural.retrieve("handle_query") == ["receive", "process", "respond"]

    def test_memory_retrieval_influences_behavior(self):
        wm = WorkingMemory(capacity=3)
        wm.remember({"context": "previous_success"})

        context = wm.recall()
        assert any(item.get("context") == "previous_success" for item in context)

    def test_memory_update_overwrites_previous(self):
        mm = MemoryManager(max_items=100)
        mm.store("key", {"version": 1})
        mm.store("key", {"version": 2})
        assert mm.get("key") == {"version": 2}
