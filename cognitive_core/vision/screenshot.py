import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from cognitive_core.vision.interfaces import ImageAsset, load_image


def capture_screen(output_path: Optional[str] = None) -> Optional[ImageAsset]:
    if sys.platform != "darwin":
        raise NotImplementedError("Screenshot supportato solo su macOS in questa configurazione")
    out = Path(output_path) if output_path else Path(tempfile.gettempdir()) / f"jarvis_screenshot_{os.getpid()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["screencapture", "-x", str(out)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Acquisizione screenshot fallita: {e}") from e
    return load_image(str(out))
