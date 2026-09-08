import abc
import importlib
import io
from dataclasses import dataclass, field
from typing import Optional

from cognitive_core.vision.interfaces import ImageAsset, VisionBackend


@dataclass
class OCRResult:
    text: str
    confidence: Optional[float] = None
    available: bool = False


class OCRBackend(abc.ABC):
    @abc.abstractmethod
    def extract_text(self, image: ImageAsset) -> OCRResult:
        pass


class MockOCRBackend(OCRBackend):
    def extract_text(self, image: ImageAsset) -> OCRResult:
        return OCRResult(text="", available=False)


def get_ocr_backend() -> OCRBackend:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        class TesseractBackend(OCRBackend):
            def extract_text(self, image: ImageAsset) -> OCRResult:
                try:
                    if image.data_bytes is None:
                        data = Path(image.path).read_bytes()
                    else:
                        data = image.data_bytes
                    pil_image = Image.open(io.BytesIO(data))
                    text = pytesseract.image_to_string(pil_image)
                    return OCRResult(text=text, available=True)
                except Exception as e:
                    return OCRResult(text="", available=False)

        return TesseractBackend()
    except ImportError:
        pass
    return MockOCRBackend()
