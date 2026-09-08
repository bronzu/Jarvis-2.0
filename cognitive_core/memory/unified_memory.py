import threading
from collections import OrderedDict
import time
import json
import os
import numpy as np

from cognitive_core.memory.working_memory import WorkingMemory
from cognitive_core.memory.episodic_memory import EpisodicMemory
from cognitive_core.memory.semantic_memory import SemanticMemory
from cognitive_core.memory.procedural_memory import ProceduralMemory
from cognitive_core.knowledge.vector_memory import VectorMemory
from cognitive_core.knowledge.knowledge_graph import KnowledgeGraph


class UnifiedMemory:
    def __init__(self, max_items=10000, ttl_seconds=3600, disk_path=None, working_capacity=10):
        self._lock = threading.RLock()
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.vector = VectorMemory()
        self.graph = KnowledgeGraph()
        self._store = OrderedDict()
        self._timestamps = {}
        self._max_items = max_items
        self._ttl_seconds = ttl_seconds
        self._disk_path = disk_path or os.path.join(os.path.dirname(__file__), "unified_memory_db.json")
        self.load_state_from_disk()

    def remember_working(self, item):
        with self._lock:
            self.working.remember(item)

    def recall_working(self):
        with self._lock:
            return self.working.recall()

    def record_episodic(self, event, outcome):
        with self._lock:
            return self.episodic.record(event, outcome)

    def recall_episodic(self, event=None):
        with self._lock:
            return self.episodic.recall(event)

    def learn_semantic(self, concept, fact):
        with self._lock:
            self.semantic.learn(concept, fact)

    def recall_semantic(self, concept):
        with self._lock:
            return self.semantic.recall(concept)

    def register_procedural(self, name, steps):
        with self._lock:
            self.procedural.register(name, steps)

    def retrieve_procedural(self, name):
        with self._lock:
            return self.procedural.retrieve(name)

    def add_vector(self, item_id, vector):
        with self._lock:
            self.vector.add(item_id, vector)

    def search_vector(self, query_vector, top_k=5):
        with self._lock:
            return self.vector.search(query_vector, top_k=top_k)

    def assert_graph_fact(self, fact):
        with self._lock:
            self.graph.assert_fact(fact)

    def relate(self, source, relation, target):
        with self._lock:
            self.graph.relate(source, relation, target)

    def related(self, source):
        with self._lock:
            return self.graph.related(source)

    def store(self, item_id, payload):
        with self._lock:
            self._evict_if_needed()
            self._store[item_id] = dict(payload)
            self._timestamps[item_id] = time.time()
            result = dict(self._store[item_id])
            self.save_state_to_disk()
            return result

    def get(self, item_id):
        with self._lock:
            if item_id in self._timestamps:
                age = time.time() - self._timestamps[item_id]
                if age > self._ttl_seconds:
                    self._store.pop(item_id, None)
                    self._timestamps.pop(item_id, None)
                    return {}
            return dict(self._store.get(item_id, {}))

    def search_semantic(self, query):
        with self._lock:
            current_time = time.time()
            matches = []
            for item_id, payload in self._store.items():
                if current_time - self._timestamps.get(item_id, 0) > self._ttl_seconds:
                    continue
                text = " ".join(str(value).lower() for value in payload.values())
                if query.lower() in text:
                    matches.append({"id": item_id, "score": 1.0, "payload": dict(payload)})
            for episode in self.episodic.recall():
                text = " ".join(str(value).lower() for value in episode.values())
                if query.lower() in text:
                    matches.append({"id": episode.get("event", ""), "score": 0.9, "payload": dict(episode)})
            for concept, facts in self.semantic._knowledge.items():
                if query.lower() in concept.lower():
                    matches.append({"id": concept, "score": 1.0, "payload": {"concept": concept, "facts": facts}})
                for fact in facts:
                    if query.lower() in fact.lower():
                        matches.append({"id": concept, "score": 0.8, "payload": {"concept": concept, "fact": fact}})
            return matches

    def _evict_if_needed(self):
        while len(self._store) >= self._max_items:
            oldest_id = next(iter(self._store))
            self._store.pop(oldest_id)
            self._timestamps.pop(oldest_id, None)

    def save_state_to_disk(self):
        with self._lock:
            state = {
                "store": dict(self._store),
                "timestamps": self._timestamps,
            }
            tmp_path = self._disk_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, self._disk_path)

    def load_state_from_disk(self):
        with self._lock:
            if not os.path.exists(self._disk_path):
                return
            try:
                with open(self._disk_path, "r") as f:
                    state = json.load(f)
                self._store = OrderedDict(state.get("store", {}))
                self._timestamps = {k: v for k, v in state.get("timestamps", {}).items()}
            except (json.JSONDecodeError, IOError):
                pass
