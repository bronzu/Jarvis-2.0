from pathlib import Path
from typing import Optional

from cognitive_core.vision.interfaces import ImageAsset, MockVisionBackend, VisionBackend, load_image
from cognitive_core.vision.ocr import get_ocr_backend
from cognitive_core.vision.screenshot import capture_screen


class VisionSystem:
    def __init__(self, backend: Optional[VisionBackend] = None, config: Optional[dict] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.screenshot_enabled = bool(self.config.get("screenshot_enabled", False))
        self.ocr_enabled = bool(self.config.get("ocr_enabled", False))
        self.backend = backend or MockVisionBackend()
        self.ocr_backend = get_ocr_backend() if self.ocr_enabled else None

    def load_image(self, path: str) -> ImageAsset:
        return load_image(path)

    def analyze_image(self, path: str) -> dict:
        if not self.enabled:
            return {
                "success": False,
                "available": False,
                "description": "Vision system disabilitato.",
                "objects": [],
                "text": [],
                "metadata": {},
            }
        try:
            image = load_image(path)
        except FileNotFoundError:
            return {
                "success": False,
                "available": False,
                "description": "File immagine non trovato.",
                "objects": [],
                "text": [],
                "metadata": {"error": "FileNotFoundError"},
            }
        except ValueError:
            return {
                "success": False,
                "available": False,
                "description": "Formato immagine non supportato.",
                "objects": [],
                "text": [],
                "metadata": {"error": "UnsupportedFormat"},
            }
        except Exception as e:
            return {
                "success": False,
                "available": False,
                "description": f"Errore caricamento immagine: {e}",
                "objects": [],
                "text": [],
                "metadata": {"error": str(e)},
            }
        try:
            result = self.backend.analyze(image)
        except Exception as e:
            return {
                "success": False,
                "available": False,
                "description": f"Errore analisi immagine: {e}",
                "objects": [],
                "text": [],
                "metadata": {"error": str(e)},
            }
        result["image_metadata"] = {
            "path": image.path,
            "format": image.format,
            "size_bytes": image.size_bytes,
            "width": image.width,
            "height": image.height,
        }
        return result

    def capture_screen(self) -> Optional[ImageAsset]:
        if not self.screenshot_enabled:
            raise PermissionError("Acquisizione screenshot disabilitata.")
        return capture_screen()

    def analyze_screenshot(self) -> dict:
        if not self.screenshot_enabled:
            return {
                "success": False,
                "available": False,
                "description": "Screenshot disabilitato.",
                "objects": [],
                "text": [],
                "metadata": {},
            }
        try:
            image = capture_screen()
        except Exception as e:
            return {
                "success": False,
                "available": False,
                "description": f"Errore acquisizione screenshot: {e}",
                "objects": [],
                "text": [],
                "metadata": {"error": str(e)},
            }
        if image is None:
            return {
                "success": False,
                "available": False,
                "description": "Screenshot non acquisito.",
                "objects": [],
                "text": [],
                "metadata": {},
            }
        return self.analyze_image(image.path)

    def extract_text(self, path: str) -> dict:
        if not self.ocr_enabled or self.ocr_backend is None:
            return {
                "success": False,
                "available": False,
                "text": "",
                "metadata": {"error": "OCR non disponibile"},
            }
        try:
            image = load_image(path)
        except Exception as e:
            return {
                "success": False,
                "available": False,
                "text": "",
                "metadata": {"error": str(e)},
            }
        result = self.ocr_backend.extract_text(image)
        return {
            "success": result.available,
            "available": result.available,
            "text": result.text,
            "metadata": {"confidence": result.confidence},
        }
