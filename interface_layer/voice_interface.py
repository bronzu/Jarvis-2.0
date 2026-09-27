"""
Voice Interface - Interface Layer
 ===================================

 La voce fa SOLO:
 - input audio/testo;
 - invio della richiesta al JARVIS Executive System.

 La voce riceve:
 - la risposta finale di JARVIS.

 La voce NON sa:
 - quale modello viene usato;
 - dove viene generata l'intelligenza;
 - quale provider esiste;
 - se esiste un modello esterno.

 ARCHITETTURA:
     INPUT UTENTE
     ↓
     INTERFACE LAYER (VoiceInterface)
     ↓
     EXECUTIVE SYSTEM (ExecutiveRouter)
     ↓
     COGNITIVE SYSTEM
     ↓
     NEURAL CORE
     ↓
     DECISIONE
     ↓
     AZIONE / RISPOSTA
 """

import threading
from typing import Optional

from executive_system.capability_registry import CapabilityRegistry
from executive_system.tool_selector import ToolSelector
from executive_system.tools.web_tool import WebBrowserTool
from executive_system.router import ExecutiveRouter
from cognitive_core.perception import PerceptionLayer


class VoiceInterface:
    """Wrapper end-to-end per dialogo vocale e web search.

    La voce comunica SOLO con l'Executive System di JARVIS.
    Non conosce modelli esterni, LLMClient, Ollama o OpenAI.
    """

    def __init__(self, executive_router=None, registry=None, perception_layer=None):
        self.executive_router = executive_router or ExecutiveRouter()
        self.registry = registry or CapabilityRegistry()
        self.registry.register("jarvis_generate")
        self.registry.register_tool("web_search", WebBrowserTool)
        self.tool_selector = ToolSelector(self.registry)
        self.web_tool = self.tool_selector.select("web_search") or WebBrowserTool()
        self._tts_backend = self._load_tts_backend()
        self._stt_backend = self._load_stt_backend()
        self.perception_layer = perception_layer or PerceptionLayer()

    def speak(self, prompt):
        if prompt is None or str(prompt).strip() == "":
            raise ValueError("prompt non può essere vuoto")

        text = str(prompt).strip()
        if any(keyword in text.lower() for keyword in ["cerca", "search", "web", "naviga", "google", "duckduckgo"]):
            result = self.web_tool.search(text, limit=3)
            return {
                "output": "\n".join(
                    f"{item.get('title', 'Risultato')} - {item.get('url', '')}"
                    for item in result
                ) or "Nessun risultato trovato.",
                "route": "web_search",
                "provider": "web_tool",
                "fallback": False,
            }

        percept_event = self.perception_layer.process_text(text)
        result = self.executive_router.generate(percept_event.normalized_text)
        return {
            "output": result.get("response") or result.get("text") or "",
            "route": result.get("route", "internal"),
            "provider": result.get("provider", "internal"),
            "fallback": result.get("fallback", False),
            "metadata": result,
        }

    def handle_prompt(self, prompt):
        return self.speak(prompt)

    def say(self, text: str, blocking: bool = True):
        """Sintesi vocale (TTS). Usa backend disponibile o fallback testuale."""
        if self._tts_backend is not None:
            try:
                if blocking:
                    self._tts_backend.speak(text)
                else:
                    threading.Thread(target=self._tts_backend.speak, args=(text,), daemon=True).start()
                return True
            except Exception:
                pass
        print(f"[TTS] {text}")
        return False

    def listen(self, prompt: Optional[str] = None) -> str:
        """Riconoscimento vocale (STT). Usa backend disponibile o fallback testuale."""
        if self._stt_backend is not None:
            try:
                return self._stt_backend.listen(prompt)
            except Exception:
                pass
        if prompt:
            return input(f"[STT] {prompt} ")
        return input("[STT] Parlare: ")

    def _load_tts_backend(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            return engine
        except Exception:
            return None

    def _load_stt_backend(self):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            return recognizer
        except Exception:
            return None