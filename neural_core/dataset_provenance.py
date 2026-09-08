"""
Dataset provenance and template leakage tracking for JARVIS 2.0.

Provides:
    - DatasetProvenance: dataclass for provenance metadata
    - build_provenance_manifest: scans dataset files and produces provenance metadata
    - detect_template_leakage: estimates whether dataset sentences derive from known templates
    - write_provenance_manifest: persists provenance to JSON
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DatasetProvenance:
    name: str
    path: str
    size_bytes: int
    md5: str
    examples: int
    source_tags: List[str] = field(default_factory=list)
    template_families: List[str] = field(default_factory=list)
    generation_seed: Optional[int] = None
    notes: str = ""


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_template_leakage(
    dataset_lines: List[str],
    known_templates: List[str],
    threshold: float = 0.92,
    sample_size: int = 500,
) -> Dict[str, Any]:
    if len(dataset_lines) > sample_size:
        import random
        rng = random.Random(42)
        dataset_lines = rng.sample(dataset_lines, sample_size)

    dataset_tokens = [_tokenize(t) for t in dataset_lines]
    template_tokens = [_tokenize(t) for t in known_templates]

    matches = []
    for i, dt in enumerate(dataset_tokens):
        best_score = 0.0
        best_template = None
        for j, tt in enumerate(template_tokens):
            score = _jaccard(dt, tt)
            if score > best_score:
                best_score = score
                best_template = known_templates[j]
        if best_score >= threshold and best_template is not None:
            matches.append({
                "dataset_index": i,
                "template": best_template,
                "jaccard": round(best_score, 4),
            })

    return {
        "threshold": threshold,
        "sample_size": len(dataset_lines),
        "match_count": len(matches),
        "matches": matches[:50],
    }


def build_provenance_manifest(
    data_dir: Path = Path("neural_core/data"),
    templates: Optional[List[str]] = None,
    template_family: str = "unknown",
    seed: Optional[int] = None,
    notes: str = "",
) -> Dict[str, Any]:
    if templates is None:
        templates = []

    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix in {".txt", ".json"}])
    manifest = {
        "data_dir": str(data_dir),
        "template_family": template_family,
        "seed": seed,
        "files": [],
        "summary": {},
    }

    all_lines: List[str] = []
    for path in files:
        if path.suffix != ".txt":
            continue
        lines = _read_lines(path)
        prov = DatasetProvenance(
            name=path.name,
            path=str(path),
            size_bytes=path.stat().st_size,
            md5=_md5(path),
            examples=len(lines),
            template_families=[template_family] if template_family != "unknown" else [],
            generation_seed=seed,
            notes=notes,
        )
        manifest["files"].append(asdict(prov))
        all_lines.extend(lines)

    leakage = detect_template_leakage(all_lines, templates) if templates else {"match_count": 0, "matches": []}
    manifest["summary"] = {
        "total_files": len(manifest["files"]),
        "total_examples": sum(f["examples"] for f in manifest["files"]),
        "template_leakage": leakage,
    }
    return manifest


def write_provenance_manifest(
    manifest: Dict[str, Any],
    output_path: Path = Path("neural_core/data/provenance_manifest.json"),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_provenance_manifest(
    path: Path = Path("neural_core/data/provenance_manifest.json"),
) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
