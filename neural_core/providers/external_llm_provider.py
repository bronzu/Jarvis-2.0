"""
External LLM Provider Adapter
==============================

Questo modulo è UN ADAPTER di comunicazione verso modelli esterni opzionali
(Ollama, OpenAI, ecc.).

RESPONSABILITÀ:
- solo comunicazione HTTP con provider esterni;
- invio di prompt e ricezione di testo;
- gestione configurazione provider.

NON CONTAINS:
- logica cognitiva
- decision making
- memoria
- ragionamento
- identità JARVIS

Gli external models sono SOLO strumenti opzionali di sviluppo, benchmark
e debugging. NON fanno parte del flusso decisionale principale di JARVIS.
"""

import os
from pathlib import Path

import requests


class ExternalLLMProvider:
    """Adapter di comunicazione verso modelli esterni opzionali (Ollama, OpenAI)."""

    def __init__(self, config_path=None, system_prompt=None):
        self.config_path = config_path or Path(__file__).resolve().parents[2] / "config.yaml"
        self.config = self._load_config(self.config_path)
        self.default_provider = self.config.get("llm", {}).get("default_provider", "internal")
        self.providers = self.config.get("llm", {}).get("providers", {})
        self.system_prompt = system_prompt or ""

    def _resolve_env(self, value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    def _coerce_scalar(self, value):
        if value is None:
            return None
        candidate = value.strip()
        if candidate.lower() in {"true", "false"}:
            return candidate.lower() == "true"
        if candidate.lower() in {"null", "none"}:
            return None
        try:
            if "." in candidate:
                return float(candidate)
            return int(candidate)
        except ValueError:
            return self._resolve_env(candidate)

    def _parse_simple_yaml(self, text):
        root = {}
        stack = [(-1, root)]

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue

            key, _, value = stripped.partition(":")
            indent = len(line) - len(line.lstrip(" "))
            key = key.strip()
            value = value.strip()

            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]
            if value == "":
                child = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = self._coerce_scalar(value)

        return root

    def _load_config(self, path):
        config = {
            "llm": {
                "default_provider": "internal",
                "providers": {
                    "ollama": {
                        "enabled": False,
                        "base_url": "http://localhost:11434",
                        "host": "localhost",
                        "port": 11434,
                        "model": "qwen2.5-coder",
                        "api_key": None,
                    },
                    "openai": {
                        "enabled": False,
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-4o-mini",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                },
            }
        }

        if not Path(path).exists():
            return config

        try:
            import yaml
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except Exception:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = self._parse_simple_yaml(handle.read())

        if isinstance(loaded, dict):
            llm_cfg = loaded.get("llm", {})
            if isinstance(llm_cfg, dict):
                config["llm"].update({
                    key: self._resolve_env(value)
                    for key, value in llm_cfg.items()
                    if key != "providers"
                })
                providers = llm_cfg.get("providers", {})
                if isinstance(providers, dict):
                    for provider_name, provider_cfg in providers.items():
                        if isinstance(provider_cfg, dict):
                            config["llm"]["providers"].setdefault(provider_name, {})
                            config["llm"]["providers"][provider_name].update({
                                key: self._resolve_env(value)
                                for key, value in provider_cfg.items()
                            })

        return config

    def _provider_config(self, provider=None):
        provider_name = provider or self.default_provider
        return self.providers.get(provider_name, self.providers.get("ollama", {}))

    def is_enabled(self, provider=None):
        """Verifica se un provider esterno è abilitato."""
        provider_name = provider or self.default_provider
        provider_cfg = self._provider_config(provider_name)
        return bool(provider_cfg.get("enabled", False))

    def call_ollama(self, prompt, model=None):
        provider = self._provider_config("ollama")
        if not provider.get("enabled", False):
            raise RuntimeError("Ollama non abilitato (opzionale, solo sviluppo/benchmark/debugging)")

        import requests
        base_url = provider.get("base_url", "http://localhost:11434")
        payload = {
            "model": model or provider.get("model", "qwen2.5-coder"),
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=provider.get("timeout_seconds", 30),
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def call_openai(self, prompt, model=None):
        provider = self._provider_config("openai")
        api_key = provider.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not provider.get("enabled", False) or not api_key:
            raise RuntimeError("OpenAI non configurato (opzionale, solo sviluppo/benchmark/debugging)")

        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or provider.get("model", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
        }
        response = requests.post(
            f"{provider.get('base_url', 'https://api.openai.com/v1')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=provider.get("timeout_seconds", 15),
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt, provider=None, model=None):
        """
        Genera testo tramite un modello esterno.

        Questo è SOLO un adapter di comunicazione. Non contiene logica
        decisionale. I risultati NON sono mai utilizzati come cervello
        principale di JARVIS.
        """
        if prompt is None:
            raise ValueError("prompt non può essere None")

        provider_name = provider or self.default_provider
        if not self.is_enabled(provider_name):
            raise RuntimeError(
                f"External provider '{provider_name}' non configurato o disabilitato. "
                "Gli external models sono opzionali (sviluppo/benchmark/debugging)."
            )

        if provider_name == "openai":
            text = self.call_openai(prompt, model=model)
            return {"provider": "openai", "text": text, "fallback": False, "external": True}

        text = self.call_ollama(prompt, model=model)
        return {"provider": "ollama", "text": text, "fallback": False, "external": True}

    # ------------------------------------------------------------------
    # Compatibilità con vecchia API LLMClient
    # ------------------------------------------------------------------

    def generate_text(self, prompt, provider=None, model=None):
        """
        Wrapper compatibile con il vecchio LLMClient.
        Ritorna solo testo o None se il provider non è disponibile.
        """
        try:
            result = self.generate(prompt, provider=provider, model=model)
            return result.get("text") or result.get("response") or ""
        except Exception:
            return None

    @property
    def providers_config(self):
        return self.providers


# Alias di compatibilità per il vecchio nome LLMClient
# NOTA: LLMClient è deprecata. Utilizzare ExternalLLMProvider
# o meglio NeuralModelInterface.
LLMClient = ExternalLLMProvider