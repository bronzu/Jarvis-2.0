#!/usr/bin/env python3
"""CLI interattiva production-ready per Jarvis 2.0."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


def locate_project_root(explicit_root: str | None = None) -> Path:
    """Trova la root del progetto cercando la directory `cognitive_core`."""
    if explicit_root:
        candidate = Path(explicit_root).expanduser().resolve()
        if (candidate / "cognitive_core" / "coordinator.py").exists():
            return candidate
        if candidate.exists() and (candidate / "cognitive_core").exists():
            return candidate
        raise FileNotFoundError(
            f"La root fornita '{candidate}' non contiene 'cognitive_core/coordinator.py'."
        )

    candidate_dirs: list[Path] = []
    script_dir = Path(__file__).resolve().parent
    candidate_dirs.extend([script_dir, *script_dir.parents])

    try:
        cwd = Path.cwd().resolve()
    except OSError:
        cwd = script_dir
    candidate_dirs.extend([cwd, *cwd.parents])

    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            candidate_dirs.append(Path(entry).expanduser().resolve())

    seen: set[Path] = set()
    for base in candidate_dirs:
        if not base.exists():
            continue
        for attempt in [base, *base.parents]:
            attempted = attempt.resolve()
            if attempted in seen:
                continue
            seen.add(attempted)
            marker = attempted / "cognitive_core" / "coordinator.py"
            if marker.exists():
                return attempted

    for maybe in [script_dir, cwd]:
        for attempt in [maybe, *maybe.parents]:
            if (attempt / "cognitive_core").exists():
                return attempt

    raise FileNotFoundError("Impossibile individuare la root del progetto: directory 'cognitive_core' non trovata.")


def ensure_project_root_on_path(explicit_root: str | None = None) -> Path:
    """Aggiunge la root del progetto a `sys.path` in modo robusto."""
    project_root = locate_project_root(explicit_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def build_coordinator(explicit_root: str | None = None) -> Any:
    """Istanzia il coordinatore cognitivo standard."""
    project_root = ensure_project_root_on_path(explicit_root)
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from cognitive_core.coordinator import CognitiveCoordinator
        from cognitive_core.reasoning.reasoner import Reasoner
        from cognitive_core.memory.episodic_memory import EpisodicMemory
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Impossibile importare i moduli necessari: {exc}. "
            "Controlla il PYTHONPATH o usa --project-root."
        ) from exc

    try:
        return CognitiveCoordinator(
            reasoner=Reasoner(),
            episodic_memory=EpisodicMemory(),
            neural_model=None,
        )
    except Exception as exc:
        raise RuntimeError(f"Errore durante l'inizializzazione del coordinatore: {exc}") from exc


def safe_json_dump(value: Any) -> str:
    """Converte valori Python in stringa leggibile."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def format_response(value: Any) -> str:
    """Formatta la risposta del coordinatore per CLI human-readable."""
    if value is None:
        return "Nessuna risposta disponibile."
    if isinstance(value, dict):
        if "default_response" in value:
            return str(value["default_response"])
        if "raw_output" in value:
            return str(value["raw_output"])
        if "text" in value:
            return str(value["text"])
    if isinstance(value, (dict, list, tuple)):
        return safe_json_dump(value)
    return str(value)


def run_self_test(coordinator: Any) -> None:
    """Esegue un self-test minimale prima dell'avvio della chat."""
    test_cases: Iterable[str] = ("status", "ping", "hello")

    for case in test_cases:
        try:
            response = coordinator.process(case)
        except Exception as exc:
            raise RuntimeError(f"Self-test fallito per '{case}': {exc}") from exc

        if response is None:
            raise RuntimeError(f"Self-test fallito: la risposta per '{case}' è None.")

    print("[Self-test] OK: il coordinatore risponde correttamente ai comandi di controllo.")


def run_repl(coordinator: Any) -> int:
    """Loop interattivo della CLI che usa SOLO il Neural Core interno di JARVIS.

    NOTA ARCHITETTURALE:
    JARVIS non usa modelli esterni come cervello. Ogni richiesta passa
    attraverso il Cognitive Coordinator e il Neural Core interno.
    """
    try:
        from executive_system.router import ExecutiveRouter
        router = ExecutiveRouter(coordinator=coordinator, config_path=None)
    except Exception:
        router = None

    print("\nJarvis 2.0 CLI pronta.")
    print("Digitare 'exit' o 'quit' per uscire.\n")

    while True:
        try:
            user_input = input("Tu > ").strip()
        except KeyboardInterrupt:
            print("\nJarvis > Interruzione da tastiera. Addio.")
            return 0
        except EOFError:
            print("\nJarvis > Fine input. Addio.")
            return 0

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit"}:
            print("Jarvis > Arrivederci.")
            return 0

        response_text = None

        # JARVIS elabora SEMPRE internamente (Neural Core + Cognitive Core)
        try:
            if router is not None:
                result = router.generate(user_input)
                response_text = result.get("response") or result.get("text")
            else:
                response = coordinator.process(user_input)
                response_text = format_response(response)
        except Exception as exc:
            response_text = f"Errore di elaborazione: {exc}"

        print(f"Jarvis > {response_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI testuale interattiva per Jarvis 2.0")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Esegue il self-test e termina senza aprire il REPL.",
    )
    parser.add_argument(
        "--no-self-test",
        action="store_true",
        help="Salta il controllo iniziale e apre direttamente il REPL.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Percorso esplicito della root del progetto.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        coordinator = build_coordinator(args.project_root)
    except Exception as exc:
        print(f"Errore critico: {exc}", file=sys.stderr)
        return 1

    if args.test:
        try:
            run_self_test(coordinator)
            print("[Self-test] OK - avvio verificato. Nessuna eccezione rilevata.")
            return 0
        except Exception as exc:
            print(f"[Self-test] FALLITO: {exc}", file=sys.stderr)
            return 1

    if not args.no_self_test:
        try:
            run_self_test(coordinator)
        except Exception as exc:
            print(f"[Avvio] Self-test fallito: {exc}", file=sys.stderr)
            return 1

    return run_repl(coordinator)


if __name__ == "__main__":
    raise SystemExit(main())
