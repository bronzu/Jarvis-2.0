import json
import time
from collections import OrderedDict


class EpisodicMemory:
    """
    Memoria episodica: registra eventi e outcomes con TTL e bound.
    
    **Policy**:
    - max_episodes: limite massimo di episodi (default: 10000, LRU eviction)
    - ttl_seconds: time-to-live per episodio (default: 3600s = 1 hour)
    - Gli episodi scaduti sono rimossi al prossimo accesso
    - Ogni episodio ha un ID univoco, anche per eventi duplicati
    """

    def __init__(self, max_episodes=10000, ttl_seconds=3600):
        if max_episodes < 1:
            raise ValueError("max_episodes deve essere almeno 1")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds deve essere >= 0")
        self._episodes = OrderedDict()
        self._timestamps = {}
        self._event_index = {}
        self._max_episodes = max_episodes
        self._ttl_seconds = ttl_seconds
        self._counter = 0

    def _next_id(self):
        self._counter += 1
        return f"ep_{self._counter}"

    def _event_key(self, event):
        if isinstance(event, str):
            return event
        try:
            return json.dumps(event, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(event)

    def _evict_if_needed(self):
        while len(self._episodes) >= self._max_episodes:
            oldest = next(iter(self._episodes))
            event = self._episodes[oldest]["event"]
            event_key = self._event_key(event)
            self._event_index[event_key].remove(oldest)
            if not self._event_index[event_key]:
                del self._event_index[event_key]
            del self._episodes[oldest]
            self._timestamps.pop(oldest, None)

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            key for key, ts in self._timestamps.items()
            if now - ts > self._ttl_seconds
        ]
        for key in expired:
            event = self._episodes[key]["event"]
            event_key = self._event_key(event)
            self._event_index[event_key].remove(key)
            if not self._event_index[event_key]:
                del self._event_index[event_key]
            self._episodes.pop(key, None)
            self._timestamps.pop(key, None)

    def record(self, event, outcome):
        if event is None:
            raise ValueError("L'evento non può essere None")
        
        self._cleanup_expired()
        self._evict_if_needed()
        
        key = self._next_id()
        episode = {"event": event, "outcome": outcome}
        self._episodes[key] = episode
        self._timestamps[key] = time.time()
        event_key = self._event_key(event)
        self._event_index.setdefault(event_key, []).append(key)
        return dict(episode)

    def recall(self, event=None):
        self._cleanup_expired()
        if event is None:
            return [dict(episode) for episode in self._episodes.values()]
        event_key = self._event_key(event)
        keys = self._event_index.get(event_key, [])
        return [dict(self._episodes[key]) for key in keys if key in self._episodes]

    def get_stats(self):
        return {
            "total_episodes": len(self._episodes),
            "max_episodes": self._max_episodes,
            "ttl_seconds": self._ttl_seconds,
        }
