"""
Perception Layer - Cognitive Core
===================================

Il Perception Layer è l'organo che riceve l'input grezzo (testo, immagine, dati strutturati)
e lo normalizza in un PerceptEvent, pronto per l'Understanding.

Responsabilità:
  - Normalizzazione del testo (strip, whitespace collapse, lower per matching)
  - Estrazione di metadati dall'input grezzo (fonte, timestamp, tipo, charset)
  - Classificazione strutturale del tipo di input (text/vision/structured/empty)
  - Caricamento di immagini tramite ImageAsset/load_image per input visionivi
  - Costruzione di un PerceptEvent che l'Understanding può consumare

La classificazione dell'*intent* (command/question/information) è responsabilità
esclusiva di Understanding, NON del Perception Layer.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from cognitive_core.vision.interfaces import ImageAsset, load_image


@dataclass
class PerceptEvent:
    """Evento di percezione normalizzato prodotto dal PerceptionLayer.

    Rappresenta l'input grezzo dopo la normalizzazione e l'etichettatura,
    pronto per essere passato all'Understanding.
    """

    raw_input: str
    normalized_text: str
    input_type: str = "text"
    source: str = "user"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_asset: Optional[ImageAsset] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "raw_input": self.raw_input,
            "normalized_text": self.normalized_text,
            "input_type": self.input_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }
        if self.image_asset is not None:
            data["image_asset"] = {
                "path": self.image_asset.path,
                "format": self.image_asset.format,
                "size_bytes": self.image_asset.size_bytes,
                "width": self.image_asset.width,
                "height": self.image_asset.height,
            }
        return data


class PerceptionLayer:
    """Organo di percezione: normalizza l'input grezzo in PerceptEvent.

    Classifica il tipo strutturale dell'input (text/vision/structured/empty)
    ma NON l'intent — quella è responsabilità di Understanding.
    """

    def __init__(self, source: str = "user"):
        self._source = source

    def process(self, raw_input: Any, source: Optional[str] = None) -> PerceptEvent:
        """Router principale: indirizza l'input al processore corretto in base al tipo strutturale."""
        if raw_input is None:
            raise ValueError("L'input non può essere None")

        if isinstance(raw_input, dict):
            input_type = raw_input.get("type")
            if input_type == "vision":
                return self._process_vision_dict(raw_input, source)
            if input_type == "text":
                return self._process_text_input(raw_input.get("text", ""), source)
            if "text" in raw_input:
                return self._process_text_input(raw_input["text"], source)
            return self._process_structured_dict(raw_input, source)

        if isinstance(raw_input, ImageAsset):
            return self._process_image_asset(raw_input, source)

        if isinstance(raw_input, (str, bytes)):
            text = raw_input if isinstance(raw_input, str) else raw_input.decode("utf-8", errors="replace")
            return self._process_text_input(text, source)

        return self._process_text_input(str(raw_input), source)

    def process_text(self, text: str, source: Optional[str] = None) -> PerceptEvent:
        """Normalizza un input testuale in un PerceptEvent."""
        if text is None:
            raise ValueError("Il testo non può essere None")
        return self._process_text_input(text, source)

    def process_image(self, image_path: str, source: Optional[str] = None) -> PerceptEvent:
        """Carica un'immagine e produce un PerceptEvent di tipo vision."""
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        image_asset = load_image(image_path)
        return self._build_vision_event(image_asset, source)

    def _process_text_input(self, text: str, source: Optional[str] = None) -> PerceptEvent:
        stripped = text.strip()
        normalized = self._normalize_text(stripped)
        detected_type = self._detect_structural_type(stripped)
        return PerceptEvent(
            raw_input=stripped,
            normalized_text=normalized,
            input_type=detected_type,
            source=source or self._source,
            metadata={"length": len(stripped), "char_set": "ascii" if all(ord(c) < 128 for c in stripped) else "unicode"},
            confidence=1.0,
        )

    def _process_structured_dict(self, data: Dict[str, Any], source: Optional[str] = None) -> PerceptEvent:
        text_repr = str(data.get("goal", data.get("question", data)))
        return PerceptEvent(
            raw_input=data,
            normalized_text=text_repr,
            input_type="structured",
            source=source or self._source,
            metadata={"keys": list(data.keys()), "structured": True},
            confidence=1.0,
        )

    def _process_vision_dict(self, problem: Dict[str, Any], source: Optional[str] = None) -> PerceptEvent:
        image_path = problem.get("image_path")
        if not image_path:
            raise FileNotFoundError("Nessun image_path fornito nel problem dict vision")
        image_asset = load_image(image_path)
        event = self._build_vision_event(image_asset, source)
        event.metadata["original_problem"] = {k: v for k, v in problem.items() if k != "image_path"}
        return event

    def _process_image_asset(self, image_asset: ImageAsset, source: Optional[str] = None) -> PerceptEvent:
        return self._build_vision_event(image_asset, source)

    def _build_vision_event(self, image_asset: ImageAsset, source: Optional[str] = None) -> PerceptEvent:
        return PerceptEvent(
            raw_input=image_asset.path,
            normalized_text=f"[VISION] {image_asset.path}",
            input_type="vision",
            source=source or self._source,
            metadata={"image_path": image_asset.path, "format": image_asset.format, "size_bytes": image_asset.size_bytes},
            image_asset=image_asset,
            confidence=1.0,
        )

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized.lower() if normalized else normalized

    def _detect_structural_type(self, text: str) -> str:
        """Classifica il tipo strutturale dell'input: text o empty.

        L'intent classification (command/question/information) è
        responsabilità esclusiva di Understanding.
        """
        if not text:
            return "empty"
        return "text"
