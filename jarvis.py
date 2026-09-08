"""
JARVIS 2.0 - Assistente Personale
===================================

Avvia l'assistente JARVIS in modalità interattiva.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from executive_system.router import ExecutiveRouter
from cognitive_core.runtime import AgentRuntime


def main():
    print("=" * 50)
    print("JARVIS 2.0 - Assistente Personale")
    print("=" * 50)
    print("Scrivi 'exit' o 'quit' per uscire.\n")

    agent_runtime = AgentRuntime()
    router = ExecutiveRouter(agent_runtime=agent_runtime)

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrivederci.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "esci"}:
            print("JARVIS: Arrivederci.")
            break

        try:
            result = router.generate(user_input)
            response = result.get("response", str(result))
            print(f"JARVIS: {response}\n")
        except Exception as e:
            print(f"JARVIS: Errore - {e}\n")


if __name__ == "__main__":
    main()
