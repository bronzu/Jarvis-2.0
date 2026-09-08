import json
import os

from cognitive_core.memory.memory_manager import MemoryManager


def test_memory_manager_creation_starts_empty(tmp_path):
    disk_path = tmp_path / "memory_db.json"
    mm = MemoryManager(disk_path=str(disk_path))
    assert mm.get_current_size() == 0
    assert mm.get_eviction_count() == 0

    mm.store("key1", {"value": "data1"})
    assert mm.get("key1") == {"value": "data1"}

    assert disk_path.exists()
    with open(disk_path, "r") as f:
        state = json.load(f)
    assert state["store"] == {"key1": {"value": "data1"}}
    assert "key1" in state["timestamps"]
    assert state["eviction_count"] == 0


def test_memory_manager_load_state_after_restart(tmp_path):
    disk_path = tmp_path / "memory_db.json"
    mm1 = MemoryManager(disk_path=str(disk_path))
    mm1.store("key1", {"value": "data1"})
    mm1.store("key2", {"value": "data2"})

    mm2 = MemoryManager(disk_path=str(disk_path))

    assert mm2.get_current_size() == 2
    assert mm2.get("key1") == {"value": "data1"}
    assert mm2.get("key2") == {"value": "data2"}
    assert mm2.get_eviction_count() == 0


def test_memory_manager_persistence_preserves_lru_order(tmp_path):
    disk_path = tmp_path / "memory_db.json"
    mm = MemoryManager(max_items=3, disk_path=str(disk_path))
    mm.store("a", {"v": 1})
    mm.store("b", {"v": 2})
    mm.store("c", {"v": 3})

    mm2 = MemoryManager(max_items=3, disk_path=str(disk_path))
    mm2.store("d", {"v": 4})
    assert mm2.get_current_size() == 3
    assert mm2.get("a") == {}
    assert mm2.get("d") == {"v": 4}


def test_memory_manager_handles_corrupted_disk_file(tmp_path):
    disk_path = tmp_path / "memory_db.json"
    disk_path.write_text("not json")

    mm = MemoryManager(disk_path=str(disk_path))
    assert mm.get_current_size() == 0
    assert mm.get_eviction_count() == 0
