import json
import os
import threading
from collections import OrderedDict
import time


class MemoryManager:
    """
    Gestore di memoria con accessi thread-safe e ricerca semantica semplice.
    
    **DESIGN NOTES**:
    - Uses RLock for thread-safe access to all operations
    - Implements LRU eviction when max_items is exceeded
    - Implements TTL-based cleanup for expired items
    - Similar to ContextManager but with unbounded retention option
    - WARNING: Without eviction policy, memory usage grows linearly with requests
    
    **Memory Policy**:
    - max_items: Maximum items to keep (default: 10000)
    - ttl_seconds: Time-to-live for each item (default: 3600 = 1 hour)
    - When max_items exceeded: Removes oldest items (LRU)
    - When items expire: Removed on next access or cleanup
    """

    def __init__(self, max_items=10000, ttl_seconds=3600, disk_path=None, sync_to_disk=True):
        self._lock = threading.RLock()
        self._store = OrderedDict()  # Maintains insertion order for LRU
        self._timestamps = {}  # Track creation time for TTL
        self._max_items = max_items
        self._ttl_seconds = ttl_seconds
        self._eviction_count = 0  # Track evictions for monitoring
        self._disk_path = disk_path
        self._sync_to_disk = sync_to_disk
        if self._disk_path:
            self.load_state_from_disk()

    def store(self, item_id, payload):
        """Store item with TTL and eviction if needed."""
        with self._lock:
            # Check if we need to evict
            self._evict_if_needed()
            
            # Store the item
            self._store[item_id] = dict(payload)
            self._timestamps[item_id] = time.time()
            
            result = dict(self._store[item_id])
            if self._sync_to_disk:
                self.save_state_to_disk()
            return result

    def sync(self):
        """Explicitly persist current state to disk."""
        if self._disk_path:
            self.save_state_to_disk()

    def get(self, item_id):
        """Retrieve item, checking if it has expired."""
        with self._lock:
            # Check TTL
            if item_id in self._timestamps:
                age = time.time() - self._timestamps[item_id]
                if age > self._ttl_seconds:
                    # Item has expired, remove it
                    self._store.pop(item_id, None)
                    self._timestamps.pop(item_id, None)
                    return {}
            
            return dict(self._store.get(item_id, {}))

    def search_semantic(self, query):
        """Search with TTL checking."""
        with self._lock:
            current_time = time.time()
            matches = []
            
            for item_id, payload in self._store.items():
                # Check TTL
                if current_time - self._timestamps.get(item_id, 0) > self._ttl_seconds:
                    continue  # Skip expired items
                
                text = " ".join(str(value).lower() for value in payload.values())
                if query.lower() in text:
                    matches.append({
                        "id": item_id,
                        "score": 1.0,
                        "payload": dict(payload),
                    })
            return matches

    def _evict_if_needed(self):
        """Remove oldest items if max_items exceeded (LRU eviction)."""
        while len(self._store) >= self._max_items:
            # Remove oldest item (first item in OrderedDict)
            oldest_id = next(iter(self._store))
            self._store.pop(oldest_id)
            self._timestamps.pop(oldest_id, None)
            self._eviction_count += 1
    
    def get_eviction_count(self):
        """Return number of evictions performed (for monitoring)."""
        with self._lock:
            return self._eviction_count
    
    def get_current_size(self):
        """Return current number of items in store (for monitoring)."""
        with self._lock:
            return len(self._store)

    def cleanup_expired(self):
        """Remove expired items proactively (TTL-based cleanup)."""
        with self._lock:
            current_time = time.time()
            expired = [
                item_id for item_id, ts in self._timestamps.items()
                if current_time - ts > self._ttl_seconds
            ]
            for item_id in expired:
                self._store.pop(item_id, None)
                self._timestamps.pop(item_id, None)
            return len(expired)

    def save_state_to_disk(self):
        """Save memory state to disk in a thread-safe manner."""
        if not self._disk_path:
            return
        with self._lock:
            state = {
                "store": dict(self._store),
                "timestamps": self._timestamps,
                "eviction_count": self._eviction_count,
            }
            tmp_path = self._disk_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, self._disk_path)

    def load_state_from_disk(self):
        """Load memory state from disk in a thread-safe manner."""
        if not self._disk_path:
            return
        with self._lock:
            if not os.path.exists(self._disk_path):
                return
            try:
                with open(self._disk_path, "r") as f:
                    state = json.load(f)
                self._store = OrderedDict(state.get("store", {}))
                self._timestamps = {k: v for k, v in state.get("timestamps", {}).items()}
                self._eviction_count = state.get("eviction_count", 0)
            except (json.JSONDecodeError, IOError):
                pass
