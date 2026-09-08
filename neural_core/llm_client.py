"""
LLM Client (DEPRECATO - Solo compatibilità)
=============================================

ATTENZIONE: L'architettura di JARVIS NON utilizza modelli esterni
come cervello. Questo modulo è DEPRECATO e mantiene SOLO compatibilità
con codice esistente.

Utilizzare invece:
- neural_core.model_interface.NeuralModelInterface  (interfaccia unica)
- neural_core.providers.external_llm_provider.ExternalLLMProvider (adapter esterno)

Gli external models (Ollama/OpenAI) sono SOLO strumenti opzionali di
sviluppo/benchmark/debugging. NON fanno parte del flusso decisionale
principale di JARVIS.
"""

import warnings

from neural_core.providers.external_llm_provider import ExternalLLMProvider

warnings.warn(
    "neural_core.llm_client.LLMClient is deprecated. "
    "Use neural_core.model_interface.NeuralModelInterface or "
    "neural_core.providers.external_llm_provider.ExternalLLMProvider instead.",
    DeprecationWarning,
    stacklevel=2,
)


class LLMClient(ExternalLLMProvider):
    """
    DEPRECATO: Adapter verso modelli esterni opzionali.

    Mantiene compatibilità con il vecchio codice ma NON è più il
    motore principale del sistema. JARVIS usa NeuralModelInterface.
    """

    def __init__(self, config_path=None, system_prompt=None):
        super().__init__(config_path=config_path, system_prompt=system_prompt)
        # Default provider interno: i provider esterni sono opzionali
        if self.default_provider in ("ollama", "openai"):
            self.default_provider = "internal"