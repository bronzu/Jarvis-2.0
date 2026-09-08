import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageAsset:
    path: str
    format: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    data_bytes: Optional[bytes] = None


class VisionBackend(abc.ABC):
    @abc.abstractmethod
    def analyze(self, image: ImageAsset) -> dict:
        pass


class MockVisionBackend(VisionBackend):
    def analyze(self, image: ImageAsset) -> dict:
        return {
            "success": True,
            "available": False,
            "description": "Backend vision non disponibile.",
            "objects": [],
            "text": [],
            "metadata": {"image_path": image.path, "format": image.format},
        }


def _detect_format(path: Path) -> Optional[str]:
    try:
        header = path.read_bytes()[:12]
    except Exception:
        return None
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if header[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "WEBP"
    return None


def _detect_dimensions(path: Path):
    try:
        with open(path, "rb") as f:
            header = f.read(32)
        fmt = _detect_format(Path(path))
        if fmt == "PNG" and len(header) >= 24:
            import struct
            w = struct.unpack(">I", header[16:20])[0]
            h = struct.unpack(">I", header[20:24])[0]
            return w, h
        if fmt == "JPEG":
            return None, None
        if fmt == "WEBP" and len(header) >= 30:
            import struct
            w = struct.unpack("<I", header[26:30])[0]
            h = struct.unpack("<I", header[30:34])[0]
            return w, h
    except Exception:
        pass
    return None, None


def load_image(path: str) -> ImageAsset:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Immagine non trovata: {path}")
    fmt = _detect_format(p)
    if fmt is None:
        raise ValueError(f"Formato immagine non supportato: {path}")
    w, h = _detect_dimensions(p)
    size_bytes = p.stat().st_size
    return ImageAsset(path=str(p), format=fmt, size_bytes=size_bytes, width=w, height=h)
