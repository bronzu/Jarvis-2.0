# JARVIS 2.0 — ARCHITETTURA UFFICIALE DEL SISTEMA

**Versione documento:** 2.0
**Data:** 9 agosto 2026
**Stato:** ARCHITETTURA UFFICIALE — BASE DI PROGETTAZIONE
**Progetto:** JARVIS 2.0

---

# 1. Scopo del documento

Questo documento definisce l'architettura ufficiale di JARVIS 2.0.

La sua funzione è trasformare la visione generale contenuta in `SPECIFICA_PROGETTO.md` in una struttura organizzata di responsabilità, componenti, dipendenze, interfacce, sicurezza, versionamento e modalità di evoluzione.

Questo documento costituisce il riferimento architetturale principale del progetto.

La struttura concreta dei file Python, delle cartelle e delle tecnologie potrà cambiare durante lo sviluppo.

L'architettura delle responsabilità fondamentali dovrà invece rimanere coerente con questo documento.

Il principio fondamentale è:

> **Non organizzare il progetto in base ai file. Organizzare il progetto in base alle responsabilità.**

---

# 2. Stato dell'architettura

Questa versione rappresenta la:

> **ARCHITETTURA UFFICIALE BASE DI JARVIS 2.0**

La presente architettura non dovrà essere modificata arbitrariamente.

Eventuali modifiche architetturali future dovranno:

1. essere motivate;
2. essere discusse prima dell'implementazione;
3. essere documentate;
4. indicare cosa cambia;
5. indicare perché cambia;
6. indicare quali componenti vengono coinvolti;
7. essere sottoposte a verifica;
8. ricevere una nuova versione ufficiale.

La versione precedente non dovrà essere cancellata.

In questo modo l'architettura mantiene una storia evolutiva verificabile.

---

# 3. Principio fondamentale: JARVIS non è un'applicazione legata a un dispositivo

JARVIS 2.0 non deve essere progettato come un programma destinato esclusivamente al computer sul quale viene sviluppato.

Il computer utilizzato durante lo sviluppo rappresenta solamente il primo ambiente di costruzione e test.

L'architettura deve essere progettata fin dall'inizio considerando la possibilità futura di utilizzare JARVIS attraverso differenti dispositivi e ambienti.

Esempi:

```text
PC
│
├── sviluppo
├── esecuzione
└── interazione

SMARTPHONE
│
├── applicazione
└── interfaccia JARVIS

AUTOMOBILE
│
├── sistema di bordo
└── interfaccia JARVIS

SMART GLASSES
│
├── visualizzazione
├── audio
└── interazione JARVIS

ALTRI DISPOSITIVI
│
└── future interfacce
```

Il principio è:

> **JARVIS deve essere pensato come un sistema distribuito e portabile, non come un singolo programma legato a un computer.**

---

# 4. Separazione tra Core e dispositivi

Il cuore di JARVIS non dovrà dipendere direttamente dall'hardware specifico di un singolo dispositivo.

Schema concettuale:

```text
                    JARVIS CORE
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
         PC          SMARTPHONE      AUTOMOBILE
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                   SMART GLASSES
```

Il Core contiene le capacità fondamentali.

I dispositivi forniscono invece:

* sensori;
* output;
* interfacce;
* capacità hardware;
* connessione;
* contesto ambientale.

La logica cognitiva principale non deve essere duplicata arbitrariamente per ogni dispositivo.

---

# 5. Architettura generale

JARVIS non deve essere considerato un singolo programma monolitico.

Il sistema sarà composto da più layer specializzati:

```text
JARVIS
│
├── NEURAL CORE
│
├── COGNITIVE CORE
│
├── EXECUTIVE SYSTEM
│
├── EXPANSION SYSTEM
│
├── INFRASTRUCTURE
│
└── INTERFACE
```

Questi layer rappresentano responsabilità architetturali.

Non rappresentano necessariamente cartelle o file Python definitivi.

---

# 6. Separazione tra JARVIS e Neural Core

Il Neural Core non rappresenta JARVIS nella sua totalità.

Il Neural Core costituisce il componente neurale del sistema.

JARVIS comprende:

```text
JARVIS
│
├── Neural Core
├── Cognitive Core
├── Executive System
├── Expansion System
├── Infrastructure
└── Interface
```

Il Neural Core fornisce capacità di elaborazione e apprendimento neurale.

Gli altri sistemi utilizzano queste capacità insieme a memoria, conoscenza, ragionamento, pianificazione, strumenti, valutazione, sicurezza e altri componenti.

---

# 7. Neural Core

## Responsabilità

Gestire il modello neurale proprietario e il relativo processo di apprendimento.

Comprende progressivamente:

* Tokenizer
* Dataset
* Embeddings
* Neural Network
* Forward Pass
* Loss
* Backpropagation
* Optimizer
* Trainer
* Checkpoint
* Model Evaluator

Pipeline:

```text
DATASET
↓
TOKENIZER
↓
EMBEDDINGS
↓
NEURAL NETWORK
↓
FORWARD PASS
↓
LOSS
↓
BACKPROPAGATION
↓
OPTIMIZER
↓
PARAMETER UPDATE
↓
CHECKPOINT
↓
EVALUATION
```

Il modello dovrà essere progettato, implementato, addestrato e valutato all'interno del progetto.

Non dovrà essere sostituito da un modello neurale pre-addestrato esterno come sostituzione permanente del Neural Core proprietario.

---

# 8. Cognitive Core

## Responsabilità

Coordinare i processi cognitivi necessari per comprendere, ragionare, ricordare, apprendere e costruire rappresentazioni del mondo.

Comprende progressivamente:

* Understanding
* Reasoning
* Planning
* Memory
* Knowledge
* World Model
* Evaluation
* Learning
* Imagination
* Discovery

Il Cognitive Core non dovrà diventare un singolo modulo monolitico.

Ogni capacità dovrà mantenere responsabilità e interfacce definite.

---

# 9. Cognitive Architecture

Il Cognitive Core sarà progressivamente composto da organi specializzati.

Schema:

```text
PERCEPTION
    ↓
UNDERSTANDING
    ↓
MEMORY / KNOWLEDGE
    ↓
REASONING
    ↓
WORLD MODEL
    ↓
PLANNING
    ↓
ACTION
    ↓
OBSERVATION
    ↓
EVALUATION
    ↓
LEARNING
```

I componenti potranno comunicare tra loro solamente attraverso interfacce definite.

Un componente non deve assumere responsabilità appartenenti a un altro componente.

---

# 10. Reasoning Architecture

Il sistema di ragionamento dovrà progressivamente supportare:

* ragionamento deduttivo;
* ragionamento induttivo;
* ragionamento abduttivo;
* ragionamento causale;
* ragionamento analogico;
* ragionamento probabilistico;
* ragionamento matematico;
* ragionamento simbolico;
* ragionamento controfattuale.

Pipeline:

```text
PROBLEMA
↓
COMPRENSIONE
↓
SCOMPOSIZIONE
↓
IPOTESI
↓
RAGIONAMENTO
↓
CONCLUSIONE
↓
VERIFICA
```

La presenza di un modulo di reasoning non sarà considerata sufficiente per dichiarare JARVIS capace di ragionare.

La capacità dovrà essere dimostrata tramite test e benchmark.

---

# 11. Memory Architecture

La memoria sarà organizzata in più livelli:

```text
WORKING MEMORY
↓
contesto corrente

EPISODIC MEMORY
↓
esperienze

SEMANTIC MEMORY
↓
conoscenze e concetti

PROCEDURAL MEMORY
↓
abilità e procedure

SELF MEMORY
↓
identità e storia di JARVIS
```

La memoria dovrà progressivamente poter:

* conservare informazioni;
* recuperare informazioni;
* collegare informazioni;
* aggiornare informazioni;
* generalizzare;
* gestire informazioni non più utili.

La memoria non deve essere considerata semplicemente un archivio.

---

# 12. Knowledge Architecture

Knowledge e World Model devono rimanere separati.

## Knowledge

Rappresenta ciò che il sistema considera conoscenza.

```text
X è vero.
```

## World Model

Rappresenta relazioni, dinamiche e possibili conseguenze.

```text
Se X è vero
e faccio Y
allora potrebbe verificarsi Z.
```

Il World Model sarà utilizzato progressivamente per:

* previsione;
* pianificazione;
* simulazione;
* causalità;
* problem solving;
* discovery.

---

# 13. Learning Architecture

Il termine Learning rappresenta più livelli differenti.

## Knowledge Learning

```text
Ho scoperto X.
```

## Capability Learning

```text
Ho imparato a fare Y.
```

## Experience Learning

```text
Ho provato X
↓
ha funzionato / non ha funzionato
↓
ho imparato qualcosa
```

## Model Learning

```text
Il modello neurale può essere migliorato.
```

## Architecture Learning

```text
Il modo in cui sto affrontando questo problema può essere migliorato.
```

Questi livelli non devono essere confusi.

L'apprendimento di una nuova informazione non implica necessariamente una modifica dei pesi del modello neurale.

---

# 14. Executive System

## Responsabilità

Trasformare obiettivi e decisioni cognitive in azioni verificabili.

Comprende:

* Goal Manager
* Capability Registry
* Tool Selection
* Action
* Observation
* Feedback

Pipeline:

```text
GOAL
↓
CAPABILITY ANALYSIS
↓
PLAN
↓
TOOL SELECTION
↓
ACTION
↓
OBSERVATION
↓
EVALUATION
```

L'Executive System non deve concedersi autonomamente privilegi che richiedono autorizzazione.

---

# 15. Capability Registry

Il Capability Registry rappresenta lo stato delle capacità disponibili a JARVIS.

Una capacità potrà essere classificata come:

```text
AVAILABLE
LIMITED
UNAVAILABLE
UNAUTHORIZED
MISSING
```

Dovrà distinguere almeno tra:

* capacità disponibile;
* capacità disponibile con limitazioni;
* capacità che richiede autorizzazione;
* strumento mancante;
* conoscenza mancante;
* procedura mancante;
* capacità del modello insufficiente;
* capacità cognitiva mancante.

Schema:

```text
GOAL
↓
CAPABILITY ANALYSIS
↓
CAN I DO THIS?
│
├── YES
│   ↓
│   PLAN
│   ↓
│   EXECUTE
│
└── NO
    ↓
    WHAT IS MISSING?
    ↓
    CAPABILITY GAP
```

Il Capability Registry rappresenta il ponte architetturale verso la Tool Factory.

---

# 16. Tool Architecture

JARVIS dovrà poter utilizzare strumenti per raggiungere i propri obiettivi.

Esempi:

* filesystem;
* terminale;
* Python;
* browser;
* computer;
* calcolatore;
* strumenti di sviluppo;
* strumenti di simulazione.

La selezione dello strumento dovrà essere collegata all'obiettivo e alle capacità necessarie.

Non dovrà essere concepita semplicemente come una lista statica di comandi.

---

# 17. Tool Factory

La Tool Factory dovrà trasformare un Capability Gap in una nuova capacità verificata.

Pipeline:

```text
CAPABILITY GAP
↓
PROBLEM ANALYSIS
↓
TOOL DESIGN
↓
CODE GENERATION
↓
SANDBOX
↓
TEST
↓
BENCHMARK
↓
EVALUATION
↓
USER APPROVAL
↓
INSTALLATION
↓
NEW CAPABILITY
```

La Tool Factory non deve essere considerata semplicemente un generatore di codice.

Il suo obiettivo è produrre una capacità funzionante, testata e valutata.

---

# 18. Expansion System

## Responsabilità

Permettere a JARVIS di acquisire progressivamente:

* nuove conoscenze;
* nuove capacità;
* nuovi strumenti;
* nuove procedure;
* nuove possibilità operative.

Comprende:

* Discovery;
* Experimentation;
* Tool Factory;
* Self Analysis;
* Self Improvement.

Il sistema deve permettere l'espansione delle capacità senza permettere a JARVIS di espandere autonomamente i propri privilegi.

---

# 19. Discovery Architecture

Discovery rappresenta il sistema attraverso il quale JARVIS può affrontare problemi per i quali non possiede già una soluzione conosciuta.

Pipeline:

```text
KNOWN KNOWLEDGE
↓
PROBLEM
↓
UNKNOWN
↓
HYPOTHESIS GENERATION
↓
PREDICTION
↓
EXPERIMENT / SIMULATION
↓
OBSERVATION
↓
COMPARISON
↓
HYPOTHESIS UPDATE
↓
NEW KNOWLEDGE
```

Quando manca uno strumento:

```text
DISCOVERY
↓
UNKNOWN
↓
HYPOTHESIS
↓
EXPERIMENT
↓
TOOL MISSING
↓
TOOL FACTORY
```

---

# 20. Self Analysis

JARVIS dovrà progressivamente poter analizzare le proprie prestazioni.

Pipeline:

```text
PERFORMANCE
↓
WEAKNESS DETECTION
↓
ANALYSIS
↓
IMPROVEMENT PROPOSAL
```

Una debolezza non deve produrre direttamente una modifica della versione attiva.

Deve produrre una proposta di miglioramento.

---

# 21. Self Improvement

Il miglioramento autonomo deve utilizzare versioni candidate.

Pipeline:

```text
CURRENT VERSION
↓
WEAKNESS DETECTION
↓
IMPROVEMENT PROPOSAL
↓
CANDIDATE VERSION
↓
SANDBOX
↓
TESTS
↓
BENCHMARK
↓
EVALUATION
```

Risultato:

```text
WORSE
↓
DELETE

SAME
↓
DELETE

BETTER
↓
USER APPROVAL
↓
NEW VERSION
```

La versione attiva non deve essere modificata direttamente durante l'esperimento.

---

# 22. Autonomia e autorizzazioni

L'architettura distingue tre livelli di autonomia.

## Autonomia cognitiva

JARVIS può decidere:

```text
Per risolvere questo problema devo fare X.
```

Questa autonomia è prevista.

## Autonomia costruttiva

JARVIS può decidere:

```text
Non possiedo X.
Posso costruire X.
```

Questa autonomia è prevista.

## Autonomia autorizzativa

JARVIS decide:

```text
Mi concedo da solo il permesso di fare X.
```

Questa autonomia non è prevista.

Regola fondamentale:

```text
AUTONOMIA COGNITIVA
        ↓
       SI

AUTONOMIA COSTRUTTIVA
        ↓
       SI

AUTONOMIA AUTORIZZATIVA
        ↓
       NO
```

JARVIS deve poter diventare più capace senza diventare automaticamente più autorizzato.

---

# 23. Infrastructure

## Responsabilità

Fornire l'ambiente controllato nel quale JARVIS può operare, sperimentare, essere testato e versionato.

Comprende:

* Sandbox;
* Security;
* Permissions;
* Versioning;
* Logging;
* Configuration;
* Data;
* Model Storage;
* Testing.

L'Infrastructure deve proteggere il sistema attivo dagli esperimenti e dalle modifiche candidate.

---

# 24. Sandbox Architecture

La Sandbox rappresenta l'ambiente isolato nel quale JARVIS potrà:

* creare codice;
* modificare codice;
* costruire strumenti;
* effettuare esperimenti;
* testare versioni candidate;
* eseguire simulazioni;
* verificare miglioramenti.

Regola fondamentale:

> **Gli esperimenti non devono modificare direttamente il sistema attivo.**

---

# 25. Security Architecture

Il sistema di sicurezza dovrà controllare progressivamente:

* quali file possono essere modificati;
* quali programmi possono essere eseguiti;
* quali strumenti possono essere utilizzati;
* quali dati possono essere letti;
* quali azioni richiedono autorizzazione;
* quali capacità possono essere installate.

La sicurezza non deve essere aggirabile semplicemente perché JARVIS ha costruito un nuovo strumento.

---

# 26. Permission Architecture

Creazione di uno strumento e concessione dei privilegi sono responsabilità differenti.

Schema:

```text
JARVIS
↓
CREA STRUMENTO
↓
TEST
↓
VALUTAZIONE
↓
RICHIESTA PERMESSO
↓
USER
↓
APPROVA / RIFIUTA
```

JARVIS può essere autonomo nella costruzione.

L'utente rimane l'autorità finale per i privilegi sensibili.

---

# 27. Versioning Architecture

Il versionamento è parte integrante dell'architettura.

Ogni versione dovrà essere:

* identificabile;
* riproducibile;
* testabile;
* confrontabile;
* reversibile;
* registrata.

Ogni modifica significativa dovrebbe poter essere descritta tramite:

```text
FROM:
JARVIS vX

TO:
JARVIS vY

WHY:
weakness detected

CHANGE:
...

RESULT:
...

BENCHMARK:
...

APPROVED:
YES / NO
```

Il sistema dovrà mantenere la storia evolutiva di JARVIS.

---

# 28. Evaluation Architecture

Evaluation deve esistere fin dalle prime fasi.

Ogni versione significativa dovrà poter essere sottoposta a:

```text
VERSIONE
↓
TEST
↓
BENCHMARK
↓
RISULTATO
↓
CONFRONTO
```

La valutazione dovrà permettere di stabilire se una nuova versione è:

* migliore;
* equivalente;
* peggiore.

La valutazione deve essere basata su risultati misurabili e non sull'impressione del sistema.

---

# 29. Testing Architecture

Ogni organo dovrà possedere test specifici.

Dovranno inoltre esistere progressivamente:

* unit test;
* integration test;
* regression test;
* benchmark;
* safety test;
* capability test;
* model evaluation;
* system test.

Schema:

```text
ORGANO
↓
UNIT TEST
↓
INTEGRATION TEST
↓
BENCHMARK
↓
SYSTEM TEST
```

Un organo non sarà considerato completato solamente perché il relativo codice esiste.

Dovrà dimostrare di funzionare.

---

# 30. Logging Architecture

Il sistema dovrà registrare gli eventi significativi.

Il logging dovrà permettere di ricostruire almeno:

* cosa è successo;
* quando è successo;
* quale componente è stato coinvolto;
* quale azione è stata eseguita;
* quale risultato è stato prodotto;
* quale versione del sistema era attiva;
* eventuali errori;
* eventuali autorizzazioni.

---

# 31. Configuration Architecture

La configurazione dovrà essere separata dalla logica dei singoli organi.

Le configurazioni non dovranno essere disperse arbitrariamente nei moduli.

Il sistema di configurazione dovrà permettere di gestire in modo controllato i parametri necessari all'esecuzione e alla sperimentazione.

---

# 32. Data Architecture

I dati utilizzati da JARVIS dovranno essere separati logicamente dal codice.

La gestione dei dati comprenderà progressivamente:

* dati di training;
* dati di valutazione;
* dati di memoria;
* dati di esperienza;
* dati di configurazione;
* dati di logging;
* dati relativi alle versioni;
* checkpoint del modello.

La struttura fisica definitiva sarà stabilita durante l'implementazione.

---

# 33. Model Storage

Il Neural Core dovrà possedere un sistema controllato per la conservazione dei modelli e dei checkpoint.

I checkpoint dovranno permettere almeno:

* salvataggio;
* identificazione;
* recupero;
* ripresa del training;
* confronto tra versioni.

Il modello non dovrà essere considerato separato dal sistema di versionamento e valutazione.

---

# 34. Interface Architecture

L'Interface rappresenta il livello di comunicazione tra JARVIS, l'utente e l'ambiente.

Comprenderà progressivamente:

```text
VOICE
TEXT
WAKE WORD
GUI
COMPUTER INTERACTION
DEVICE INTERACTION
```

L'Interface non deve contenere la logica cognitiva principale.

Il flusso generale sarà:

```text
USER / ENVIRONMENT
        ↓
INTERFACE
        ↓
PERCEPTION / UNDERSTANDING
        ↓
COGNITIVE CORE
        ↓
EXECUTIVE SYSTEM
        ↓
ACTION
        ↓
OBSERVATION
        ↓
INTERFACE
        ↓
USER / ENVIRONMENT
```

---

# 35. Multi-Device Architecture

JARVIS dovrà essere progettato per poter essere utilizzato attraverso differenti interfacce e dispositivi.

La piattaforma specifica non deve determinare l'architettura cognitiva.

Schema concettuale:

```text
                    ┌──────────────────┐
                    │    JARVIS CORE   │
                    │                  │
                    │ Neural           │
                    │ Cognitive        │
                    │ Executive        │
                    │ Expansion        │
                    └────────┬─────────┘
                             │
                    INFRASTRUCTURE / API
                             │
        ┌────────────┬───────┼────────┬────────────┐
        │            │       │        │            │
        ▼            ▼       ▼        ▼            ▼
       PC       SMARTPHONE  AUTO    GLASSES     FUTURO
```

Ogni dispositivo potrà possedere un proprio adapter/interface.

Esempio concettuale:

```text
JARVIS CORE
     │
     ▼
DEVICE INTERFACE
     │
 ┌───┼────┬────────┐
 ▼   ▼    ▼        ▼
PC  PHONE AUTO   GLASSES
```

L'implementazione definitiva potrà essere:

* centralizzata;
* distribuita;
* ibrida;

in base alle esigenze di latenza, sicurezza, hardware, connettività, privacy e capacità computazionale.

Questa decisione verrà presa durante le fasi appropriate della progettazione.

Il principio architetturale rimane:

> **Il dispositivo è un mezzo attraverso il quale JARVIS opera, non ciò che definisce JARVIS.**

---

# 36. Contratti tra organi

Ogni organo dovrà possedere una responsabilità chiaramente definita.

Prima della sua implementazione dovrà essere definita una scheda tecnica contenente almeno:

```text
ORGANO:
...

RESPONSABILITÀ:
...

INPUT:
...

OUTPUT:
...

DIPENDENZE:
...

INTERFACCE:
...

PUÒ MODIFICARE:
...

NON PUÒ MODIFICARE:
...

TEST:
...

CRITERIO DI SUCCESSO:
...

SICUREZZA:
...

VERSIONAMENTO:
...
```

Questa scheda rappresenterà il contratto dell'organo.

Un organo non dovrà assumere responsabilità non appartenenti al proprio contratto.

---

# 37. Dipendenze

Le dipendenze tra organi dovranno essere esplicite.

Un componente non dovrebbe dipendere direttamente da implementazioni interne non necessarie di un altro componente.

Principio:

```text
ORGANO A
    ↓
INTERFACCIA
    ↓
ORGANO B
```

e non:

```text
ORGANO A
    ↓
ACCESSO DIRETTO
    ↓
INTERNI DI ORGANO B
```

Questo permetterà di sostituire o migliorare un componente senza dover riscrivere l'intero sistema.

---

# 38. Principio di sostituibilità

Ogni organo dovrebbe poter essere:

* testato separatamente;
* sostituito;
* migliorato;
* versionato;
* confrontato con una versione precedente.

L'architettura non deve dipendere inutilmente dalla particolare implementazione interna di un organo.

---

# 39. Ordine di costruzione

La costruzione seguirà la roadmap stabilita nella specifica del progetto.

```text
FASE 0
FONDAMENTA
↓
FASE 1
PROPRIETARY NEURAL NETWORK
↓
FASE 2
MODEL LEARNING
↓
FASE 3
REASONING
↓
FASE 4
MEMORY + KNOWLEDGE
↓
FASE 5
WORLD MODEL
↓
FASE 6
COGNITIVE CORE
↓
FASE 7
EXECUTIVE SYSTEM
↓
FASE 8
TOOLS
↓
FASE 9
DISCOVERY
↓
FASE 10
TOOL FACTORY
↓
FASE 11
SELF ANALYSIS
↓
FASE 12
SELF IMPROVEMENT
↓
FASE 13
AUTONOMY
```

L'ordine non dovrà essere modificato arbitrariamente.

Eventuali problemi architetturali dovranno essere identificati e discussi prima di cambiare la roadmap.

---

# 40. Regola "un organo alla volta"

La regola operativa del progetto è:

> **Costruire un organo di JARVIS alla volta.**

Costruire un organo non significa semplicemente creare un file.

Il ciclo completo sarà:

```text
PROGETTAZIONE
↓
IMPLEMENTAZIONE
↓
TEST ISOLATO
↓
INTEGRAZIONE
↓
TEST CON IL SISTEMA
↓
VERIFICA
```

Se qualcosa non funziona:

```text
PROBLEMA
↓
ANALISI
↓
MODIFICA
↓
NUOVO TEST
```

Un giorno di sviluppo deve quindi aggiungere un blocco funzionale verificabile, anche se l'organo non è ancora perfetto.

---

# 41. Architettura evolutiva

L'architettura non sarà considerata immutabile nei dettagli implementativi.

Potranno essere modificati:

* nomi dei file;
* struttura delle cartelle;
* implementazioni;
* tecnologie;
* suddivisione interna dei moduli;
* modalità di distribuzione.

Non dovranno invece essere modificati arbitrariamente:

* obiettivo generale;
* vincoli fondamentali;
* separazione delle responsabilità;
* principio di verifica;
* controllo delle autorizzazioni;
* necessità di versionamento;
* necessità di testing;
* principio di costruzione progressiva;
* indipendenza del Core dal dispositivo;
* possibilità futura di utilizzo multi-device.

Ogni modifica architetturale significativa dovrà essere motivata e registrata.

---

# 42. Principio di controllo dell'architettura

Ogni nuovo organo dovrà rispondere alle seguenti domande:

```text
Chi è responsabile?
Cosa riceve?
Cosa produce?
Da cosa dipende?
Cosa può modificare?
Cosa non può modificare?
Come viene testato?
Come viene valutato?
Come viene versionato?
```

Se queste domande non hanno una risposta chiara, l'organo non è ancora sufficientemente progettato per essere implementato.

---

# 43. Architettura completa

La visione architetturale completa è:

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │    INTERFACE    │
                  │                 │
                  │ Voice           │
                  │ Text            │
                  │ GUI             │
                  │ Wake Word       │
                  │ Device Adapters │
                  └────────┬────────┘
                           │
                           ▼
                    PERCEPTION
                           │
                           ▼
                    UNDERSTANDING
                           │
                           ▼
                  ┌─────────────────┐
                  │   NEURAL CORE   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ COGNITIVE CORE  │
                  │                 │
                  │ Memory          │
                  │ Reasoning       │
                  │ Knowledge       │
                  │ World Model     │
                  │ Planning        │
                  │ Learning        │
                  │ Imagination     │
                  └────────┬────────┘
                           │
                           ▼
                  CAPABILITY CHECK
                     │           │
                   CAN          CANNOT
                     │           │
                     ▼           ▼
                   PLAN     CAPABILITY GAP
                     │           │
                     │           ▼
                     │      TOOL FACTORY
                     │           │
                     │           ▼
                     │     SANDBOX / TEST
                     │           │
                     └─────┬─────┘
                           │
                           ▼
                  EXECUTIVE SYSTEM
                           │
                           ▼
                         ACTION
                           │
                           ▼
                      OBSERVATION
                           │
                           ▼
                       EVALUATION
                           │
                           ▼
                        LEARNING
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        DISCOVERY                  SELF ANALYSIS
             │                           │
             ▼                           ▼
       NEW KNOWLEDGE              SELF IMPROVEMENT
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    CANDIDATE VERSION
                           │
                           ▼
                     SANDBOX / TEST
                           │
                           ▼
                       BENCHMARK
                           │
                           ▼
                    USER APPROVAL
                           │
                           ▼
                     NEW JARVIS
```

L'intero sistema è supportato dall'Infrastructure:

```text
┌─────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE                     │
│                                                     │
│ Security                                             │
│ Permissions                                          │
│ Sandbox                                              │
│ Versioning                                           │
│ Logging                                              │
│ Configuration                                        │
│ Data                                                 │
│ Model Storage                                        │
│ Testing                                              │
└─────────────────────────────────────────────────────┘
```

---

# 44. Visione multi-device futura

La visione finale non è:

```text
PC
↓
JARVIS
```

ma:

```text
                         JARVIS
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
            PC        SMARTPHONE      AUTOMOBILE
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                     SMART GLASSES
                           │
                           ▼
                    FUTURE DEVICES
```

JARVIS dovrà quindi essere progettato affinché l'interfaccia possa cambiare senza dover ricostruire il suo nucleo cognitivo.

Il PC è il primo ambiente di sviluppo.

Non è il limite architetturale del progetto.

---

# 45. Principio di portabilità

La portabilità non significa necessariamente che ogni componente debba essere eseguito localmente su ogni dispositivo.

Il progetto potrà utilizzare, in futuro, differenti modalità:

```text
LOCAL
REMOTE
DISTRIBUTED
HYBRID
```

La scelta dipenderà da:

* potenza computazionale;
* latenza;
* connessione;
* autonomia energetica;
* privacy;
* sicurezza;
* disponibilità hardware;
* dimensione dei modelli;
* esigenze dell'utente.

La decisione tecnica definitiva verrà presa quando il progetto raggiungerà le fasi nelle quali sarà necessaria.

Il requisito architetturale fondamentale rimane la separazione tra:

```text
JARVIS CORE
        │
        ▼
DEVICE / PLATFORM
```

---

# 46. Principio di crescita

JARVIS 2.0 non verrà costruito tentando di realizzare immediatamente il sistema finale.

Il progetto seguirà:

```text
SMALL
↓
UNDERSTOOD
↓
TESTED
↓
VERIFIED
↓
IMPROVED
↓
INTEGRATED
↓
EXPANDED
```

Ogni nuovo livello dovrà poggiare su quello precedente.

---

# 47. Principio conclusivo

JARVIS 2.0 sarà costruito secondo i seguenti principi fondamentali:

1. **Costruzione progressiva.**
2. **Un organo alla volta.**
3. **Ogni organo deve possedere un contratto.**
4. **Ogni componente deve essere testabile.**
5. **Le responsabilità devono essere separate.**
6. **Gli esperimenti devono avvenire in ambienti controllati.**
7. **La versione attiva non deve essere modificata direttamente durante gli esperimenti.**
8. **JARVIS può aumentare le proprie capacità senza aumentare autonomamente i propri privilegi.**
9. **L'utente mantiene l'autorità finale sulle autorizzazioni sensibili.**
10. **Ogni modifica significativa deve essere verificabile e versionata.**
11. **Il Core non deve essere legato a un singolo dispositivo.**
12. **PC, smartphone, automobile, occhiali e futuri dispositivi devono essere considerati interfacce/piattaforme attraverso cui JARVIS può operare.**
13. **La struttura dei file deve derivare dall'architettura e non viceversa.**
14. **La funzionalità deve essere dimostrata tramite test e risultati misurabili.**
15. **La storia evolutiva del progetto deve essere conservata.**

L'obiettivo non è costruire immediatamente il JARVIS finale.

L'obiettivo è costruire progressivamente sistemi sempre più:

* comprensibili;
* verificabili;
* intelligenti;
* capaci;
* autonomi;

mantenendo sempre il controllo dell'utente sulle modifiche e sulle autorizzazioni sensibili.

---

# 48. Stato ufficiale

**Progetto:** JARVIS 2.0

**Documento:** Architettura Ufficiale del Sistema

**Versione:** 2.0

**Data:** 9 agosto 2026

**Stato:** APPROVATA COME ARCHITETTURA UFFICIALE BASE

**Neural Core:** definito architetturalmente

**Cognitive Core:** definito architetturalmente

**Executive System:** definito architetturalmente

**Expansion System:** definito architetturalmente

**Infrastructure:** definita architetturalmente

**Interface:** definita architetturalmente

**Multi-device architecture:** definita come requisito architetturale

**Implementazione:** non ancora iniziata

**Specifiche tecniche dei singoli organi:** da definire progressivamente

**Codice JARVIS 2.0:** non ancora iniziato

**Vecchio JARVIS:** conservato separatamente

---

# 49. Approvazione

Questa versione costituisce il riferimento ufficiale di progettazione di JARVIS 2.0.

Non dovrà essere sostituita silenziosamente.

Qualsiasi futura modifica architetturale dovrà essere registrata come nuova revisione, mantenendo disponibile la versione precedente.

La regola è:

> **L'architettura può evolvere, ma non può essere dimenticata.**

---

**JARVIS 2.0 — ARCHITETTURA UFFICIALE v2.0**

**Stato: APPROVATA COME BASE DI PROGETTAZIONE — 9 agosto 2026**

