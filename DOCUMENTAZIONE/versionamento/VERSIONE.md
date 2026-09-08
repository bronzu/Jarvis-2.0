# JARVIS 2.0 — VERSIONAMENTO

## Versione corrente

**0.2.0**

## Stato

Prima implementazione integrata — Fasi 0–13 completate

## Cronologia

### 0.2.0 — Prima implementazione integrata

- Completate e testate le Fasi 3–13 della roadmap.
- Aggiunti Reasoning, Memory, Knowledge, World Model e Cognitive Coordinator.
- Aggiunti Executive System, Tools controllati e Discovery.
- Aggiunti Tool Factory, Self Analysis, Self Improvement e Autonomy Manager.
- Introdotto il Permission Guard: il sistema non può concedersi autorizzazioni e ogni azione sensibile richiede approvazione esplicita dell'utente.
- Suite integrata verificata: 69/69 test superati.
- Self-improvement potenziato: ciclo controllato completo con osservazione, valutazione, identificazione problemi (10 tipi), generazione ipotesi, simulazione in sandbox, benchmark multi-metrica, comparazione con baseline, approvazione/rifiuto, deploy, monitoraggio e rollback.
- Aggiunti SystemBenchmark per valutazione funzionale (instruction following, reasoning, memory, tool selection, safety, latency, stability).

### 0.1.0 — Fondazione del progetto

- Creazione del progetto JARVIS 2.0.
- Definizione dell'architettura iniziale.
- Creazione della struttura principale.
- Implementazione della configurazione centrale.
- Implementazione del sistema di logging.
- Creazione del sistema di testing.
- Creazione della sandbox.
- Creazione del sistema di gestione degli esperimenti.
- Preparazione della documentazione per il versionamento.

## Regola di versionamento

Le modifiche importanti all'architettura dovranno essere registrate
con una nuova versione.

Le modifiche minori potranno essere registrate all'interno
della versione corrente quando non modificano i contratti
fondamentali del sistema.
