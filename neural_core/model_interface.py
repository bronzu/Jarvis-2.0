"""
Neural Model Interface
=======================

INTERFACCIA UNICA tra i sistemi JARVIS e il Neural Core.

Tutti i sistemi JARVIS devono comunicare con questa interfaccia.
Nessun altro modulo deve importare direttamente:
- ollama
- openai
- llm_client
- provider esterni

ARCHITETTURA:
    INPUT UTENTE
    ↓
    INTERFACE LAYER
    ↓
    EXECUTIVE SYSTEM
    ↓
    COGNITIVE SYSTEM
    ↓
    NEURAL CORE (NeuralModelInterface)
    ↓
    DECISIONE
    ↓
    AZIONE / RISPOSTA

I modelli esterni (Ollama/OpenAI) sono SOLO strumenti opzionali di
sviluppo/benchmark/debugging. NON fanno parte del flusso decisionale
principale.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class NeuralModelInterface:
    """
    Interfaccia standard del Neural Core di JARVIS.

    Tutti i sistemi JARVIS (Executive, Cognitive, Interface Layer)
    comunicano con il modello tramite questa interfaccia.

    Metodi obbligatori:
    - generate()
    - train()
    - evaluate()
    - save()
    - load()
    """

    def __init__(self, model=None, tokenizer=None, dataset=None, embeddings=None,
                 loss=None, optimizer=None, checkpoint=None, trainer=None,
                 evaluator=None, config_path=None):
        """
        Inizializza l'interfaccia con i componenti del Neural Core Proprio.

        Args:
            model: Modello neurale interno (es. Neuron, reti personalizzate)
            tokenizer: Tokenizer interno
            dataset: Dataset interno
            embeddings: Embeddings interni
            loss: Funzione di loss interna
            optimizer: Ottimizzatore interno
            checkpoint: Gestore checkpoint interno
            trainer: Trainer interno
            evaluator: Evaluator interno
            config_path: Percorso configurazione (opzionale)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.embeddings = embeddings
        self.loss = loss
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.trainer = trainer
        self.evaluator = evaluator
        self.config_path = config_path or Path(__file__).resolve().parents[1] / "config.yaml"
        self._load_configuration()

    def _load_configuration(self):
        """Carica la configurazione del Neural Core (se disponibile)."""
        self.config = {
            "neural_core": {
                "type": "internal",
                "provider": "internal_model",
                "enabled": True,
            }
        }
        try:
            path = Path(self.config_path)
            if path.exists():
                import yaml
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict) and isinstance(loaded.get("neural_core"), dict):
                    self.config["neural_core"].update(loaded["neural_core"])
        except Exception:
            # Configurazione opzionale: il Neural Core funziona comunque
            pass

    # ------------------------------------------------------------------
    # CONTEXT WINDOW
    # ------------------------------------------------------------------

    def context_window(self) -> Optional[int]:
        """
        Context window del modello corrente.

        Returns:
            Numero massimo di token processabili, `None` se illimitato.
        """
        if self.model is None:
            return None

        for attribute in ("max_len", "context_size", "max_context"):
            value = getattr(self.model, attribute, None)
            if isinstance(value, int) and value > 0:
                return value

        return None

    def _crop_to_context(self, token_ids: List[int], limit: Optional[int]) -> List[int]:
        """Mantiene solo gli ultimi `limit` token della sequenza."""
        if limit is None or limit <= 0 or len(token_ids) <= limit:
            return token_ids
        return token_ids[-limit:]

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Genera una risposta usando SOLO il Neural Core interno.

        Args:
            prompt: Testo in input
            **kwargs: Parametri aggiuntivi (max_new_tokens, stop_token,
                temperature, top_k, top_p, max_context)

        Returns:
            Dict con la risposta del Neural Core interno
        """
        if prompt is None:
            raise ValueError("prompt non può essere None")

        if self.model is None:
            raise RuntimeError("Modello neurale non disponibile per la generazione.")

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non disponibile per la generazione.")

        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_length", 50))
        stop_token = kwargs.get("stop_token", None)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", None)
        top_p = kwargs.get("top_p", None)

        max_context = kwargs.get("max_context", None)
        if max_context is None:
            max_context = self.context_window()

        if isinstance(stop_token, str):
            stop_token = self.tokenizer.vocabulary.get(stop_token)

        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [self.tokenizer.vocabulary.get(getattr(self.tokenizer, "UNK_TOKEN", "<UNK>"), 0)]

        generated_ids = list(token_ids)
        truncated = max_context is not None and len(generated_ids) > max_context

        for _ in range(max_new_tokens):
            context_ids = self._crop_to_context(generated_ids, max_context)
            logits = self.model.forward(context_ids)
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                last_logits = logits[-1, :]
            else:
                raise RuntimeError("Formato logits non supportato.")

            if temperature <= 0:
                next_token_id = int(np.argmax(last_logits))
            else:
                from neural_core.model import Decoder
                decoder = Decoder()
                next_token_id = decoder.sample(
                    last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            if stop_token is not None and next_token_id == stop_token:
                generated_ids.append(next_token_id)
                break

            generated_ids.append(next_token_id)

        text = self.tokenizer.decode(generated_ids)
        return {
            "text": text,
            "model": "neural-core-internal",
            "provider": "internal_model",
            "fallback": False,
            "context_window": max_context,
            "prompt_truncated": bool(truncated),
            "generated_tokens": len(generated_ids) - len(token_ids),
        }

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    def train(self, inputs: List[Any], targets: List[Any], epochs: int = 1, **kwargs) -> Any:
        """
        Addestra il Neural Core interno.

        Args:
            inputs: Dati di training
            targets: Target di training
            epochs: Numero di epoche
            **kwargs: Parametri extra

        Returns:
            Storico di training o risultato
        """
        if self.trainer is not None:
            return self.trainer.train(inputs, targets, epochs=epochs, **kwargs)

        if self.model is not None and hasattr(self.model, "train"):
            return self.model.train(inputs, targets, epochs=epochs, **kwargs)

        raise RuntimeError("Nessun trainer o modello trainabile disponibile nel Neural Core.")

    # ------------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------------

    def evaluate(self, inputs: List[Any], targets: List[Any], **kwargs) -> Any:
        """
        Valuta le prestazioni del Neural Core interno.

        Args:
            inputs: Input di valutazione
            targets: Target di valutazione
            **kwargs: Parametri extra

        Returns:
            Metriche di valutazione
        """
        if self.evaluator is not None:
            return self.evaluator.evaluate(inputs, targets, **kwargs)

        if self.model is not None and hasattr(self.model, "evaluate"):
            return self.model.evaluate(inputs, targets, **kwargs)

        raise RuntimeError("Nessun evaluator o modello disponibile nel Core.")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(self, path: str, **kwargs) -> None:
        """
        Salva il Neural Core interno su disco (checkpoint).

        Args:
            path: Percorso del file checkpoint
            **kwargs: Parametri extra
        """
        if self.checkpoint is not None:
            checkpoint_data = self._build_checkpoint_data()
            self.checkpoint.save(checkpoint_data, path)
            return

        if self.model is not None and hasattr(self.model, "save_checkpoint"):
            self.model.save_checkpoint(path)
            return

        raise RuntimeError("Nessun checkpoint manager disponibile nel Core.")

    def _build_checkpoint_data(self) -> Dict[str, Any]:
        """Costruisce i dati del checkpoint interno."""
        return {
            "model_name": "jarvis-neural-core",
            "parameters": {
                "model_type": type(self.model).__name__ if self.model else "none",
                "tokenizer_vocab_size": len(self.tokenizer.vocabulary) if self.tokenizer else 0,
            }
        }

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load(self, path: str, **kwargs) -> Any:
        """
        Carica il Neural Core interno da disco (checkpoint).

        Args:
            path: Percorso del file checkpoint
            **kwargs: Parametri extra

        Returns:
            Dati del checkpoint caricato
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")

        if self.checkpoint is not None:
            return self.checkpoint.load(path)

        if self.model is not None and hasattr(self.model, "load_checkpoint"):
            return self.model.load_checkpoint(path)

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Restituisce lo stato del Neural Core interno."""
        return {
            "type": self.config.get("neural_core", {}).get("type", "internal"),
            "provider": self.config.get("neural_core", {}).get("provider", "internal_model"),
            "enabled": self.config.get("neural_core", {}).get("enabled", True),
            "model_available": self.model is not None,
            "model_architecture": type(self.model).__name__ if self.model is not None else None,
            "context_window": self.context_window(),
            "parameter_count": (
                self.model.parameter_count()
                if self.model is not None and hasattr(self.model, "parameter_count")
                else None
            ),
            "tokenizer_available": self.tokenizer is not None,
            "tokenizer_mode": getattr(self.tokenizer, "mode", None) if self.tokenizer else None,
            "vocab_size": len(self.tokenizer.vocabulary) if self.tokenizer is not None else 0,
            "dataset_available": self.dataset is not None,
            "embeddings_available": self.embeddings is not None,
            "loss_available": self.loss is not None,
            "optimizer_available": self.optimizer is not None,
            "checkpoint_available": self.checkpoint is not None,
            "trainer_available": self.trainer is not None,
            "evaluator_available": self.evaluator is not None,
            "external_model_required": False,
        }

    def is_internal(self) -> bool:
        """Verifica che il Neural Core sia interno (non esterno)."""
        return not self.config.get("neural_core", {}).get("external", False)