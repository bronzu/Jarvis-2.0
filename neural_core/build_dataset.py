"""
Pipeline di costruzione del dataset per il Neural Core di JARVIS 2.0.

Step:
    1. Pulizia della letteratura classica (file Desktop/Jarvis costruzione/Allenamento jarvis).
    2. Generazione materiale sintetico per categorie B-H.
    3. Unione in un unico corpus UTF-8.
    4. Suddivisione in train / validation / test.
    5. Verifica dimensione finale.
"""

import random
import re
from pathlib import Path
from typing import List

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "neural_core" / "data"
DESKTOP_LITERATURE_DIR = Path.home() / "Desktop" / "Jarvis costruzione" / "Allenamento jarvis"

LITERATURE_CLEAN = DATA_DIR / "cleaned_literature.txt"
SYNTHETIC_OUTPUT = DATA_DIR / "synthetic_corpus.txt"
FULL_CORPUS = DATA_DIR / "full_corpus.txt"
TRAIN_FILE = DATA_DIR / "train.txt"
VAL_FILE = DATA_DIR / "validation.txt"
TEST_FILE = DATA_DIR / "test.txt"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pulizia testo
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_STRICT_NORM_TABLE = str.maketrans({
    ".": "", ",": "", ";": "", ":": "", "!": "", "?": "",
    "—": "", "–": "", "-": "",
    "«": "", "»": "", "‹": "", "›": "",
    "“": "", "”": "", "‘": "", "’": "",
    "(": "", ")": "", "[": "", "]": "", "{": "", "}": "",
    '"': "", "'": "", "*": "", "/": "", "\\": "",
    "<": "", ">": "",
})


def _strict_normalize(sentence: str) -> str:
    """Strict normalization for sentence-level deduplication.

    Removes ALL punctuation, lowercases, collapses whitespace.
    Two sentences are considered the same if their strict-normalized forms
    are equal, e.g. '— Un momento !' == 'Un momento !' == 'Un momento !.'.
    """
    text = sentence.lower()
    text = text.translate(_STRICT_NORM_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pre_split_normalize(text: str) -> str:
    """Normalize the raw corpus BEFORE sentence splitting so that fragments
    with extra spaces around apostrophes or punctuation variants collapse
    consistently. Applied to the full corpus text in main() / regenerate.
    """
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])\s*", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def remove_ocr_artifacts(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        letters = sum(c.isalpha() for c in stripped)
        symbols = sum(not c.isalnum() and not c.isspace() for c in stripped)
        if letters == 0 and symbols > 0:
            continue
        if letters < 3 and symbols > letters:
            continue
        if re.search(r"^(\.{3,}|\-{3,}|\_{3,})$", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def clean_text(text: str) -> str:
    text = remove_ocr_artifacts(text)
    text = normalize_whitespace(text)
    text = text.replace("«", " << ")
    text = text.replace("»", " >> ")
    text = re.sub(r"([.,;:!?\"()\[\]{}])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Letteratura classica
# ---------------------------------------------------------------------------

def clean_literature() -> str:
    if not DESKTOP_LITERATURE_DIR.exists():
        raise FileNotFoundError(
            f"Cartella letteratura non trovata: {DESKTOP_LITERATURE_DIR}"
        )
    files = sorted(DESKTOP_LITERATURE_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError("Nessun file .txt nella cartella di letteratura.")

    merged = []
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        merged.append(clean_text(raw))

    result = "\n\n".join(merged)
    LITERATURE_CLEAN.write_text(result, encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Generatori sintetici
# ---------------------------------------------------------------------------

def _choice(options: List[str]) -> str:
    return random.choice(options)


def _choices(options: List[str], k: int) -> List[str]:
    return random.choices(options, k=k)


# ------------------------------
# B - Italiano moderno
# ------------------------------

ITALIAN_MODERN_TEMPLATES = [
    "La cucina italiana è famosa in tutto il mondo per la sua varietà e il suo sapore autentico.",
    "Ogni regione d'Italia ha piatti tradizionali che raccontano la storia e il territorio.",
    "Un buon piatto di pasta richiede pochi ingredienti ma molta attenzione alla qualità.",
    "Il caffè espresso è una tradizione italiana che unisce le persone ogni giorno.",
    "Viaggiare in Italia offre la possibilità di scoprire arte, natura e storia nello stesso viaggio.",
    "Le città d'arte come Roma, Firenze e Venezia attraggono milioni di visitatori ogni anno.",
    "Un itinerario in Toscana può includere colline, vigneti e borghi medievali.",
    "La cultura italiana si esprime anche attraverso il design, la moda e il cinema.",
    "Il festival di Venezia è uno degli eventi cinematografici più prestigiosi al mondo.",
    "La musica italiana spazia dalla lirica al pop, dal jazz al rock underground.",
    "Lo sport italiano appassiona milioni di tifosi, soprattutto nel calcio e nella motoGP.",
    "Una partita allo stadio è un'esperienza che unisce emozione, tifo e comunità.",
    "La moda italiana è sinonimo di eleganza, creatività e artigianato di qualità.",
    "Milano è una delle capitali mondiali della moda con settimane dedicate alle collezioni.",
    "Rispettare l'ambiente significa adottare abitudini quotidiane più sostenibili.",
    "Il riciclo dei materiali e la riduzione degli sprechi sono gesti concreti per il pianeta.",
    "L'energia rinnovabile come il solare e l'eolico rappresentano il futuro della produzione energetica.",
    "Gli smartphone sono diventati strumenti essenziali per comunicare, lavorare e informarsi.",
    "Le applicazioni di messaggistica permettono di rimanere in contatto con amici e colleghi in tempo reale.",
    "Il lavoro da remoto ha cambiato le abitudini di milioni di persone in tutto il mondo.",
    "Una riunione efficace richiede un ordine del giorno chiaro e la partecipazione attiva di tutti.",
    "L'educazione continua è fondamentale per adattarsi ai cambiamenti del mercato del lavoro.",
    "Corsi online e tutorial permettono di apprendere nuove competenze in modo flessibile.",
    "La lettura è una delle abitudini più utili per ampliare il lessico e la capacità di ragionamento.",
    "Un libro ben scritto può cambiare la prospettiva con cui vediamo il mondo.",
    "La fotografia digitale permette di catturare momenti e condividerli con semplicità.",
    "Scattare foto durante un viaggio aiuta a conservare ricordi che durano nel tempo.",
    "Il giardinaggio è un'attività che combina pazienza, conoscenza delle piante e contatto con la natura.",
    "Prendersi cura delle piante in casa migliora l'umore e la qualità dell'aria.",
    "Il caffè italiano è una cerimonia sociale che si ripete più volte al giorno.",
    "Una tazzina di caffè al bar è un momento di pausa e conversazione.",
    "La salute fisica e mentale sono legate: muoversi regolarmente migliora il benessere generale.",
    "Una passeggiata all'aperto riduce lo stress e migliora la concentrazione.",
    "Le relazioni interpersonali si costruiscono con ascolto, rispetto e comunicazione chiara.",
    "Un buon consiglio è spesso più utile di una soluzione preconfezionata.",
]


def generate_italian_modern(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = _choices(ITALIAN_MODERN_TEMPLATES, k=length)
        parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ------------------------------
# C - Scienza
# ------------------------------

SCIENCE_TEMPLATES = [
    "La {materia} studia il comportamento di {fenomeno} attraverso il {metodo}.",
    "Nel campo della {materia}, il concetto di {concetto} è fondamentale per comprendere {applicazione}.",
    "Secondo i principi della {materia}, {fenomeno} si verifica quando {condizione}.",
    "Un esperimento di {materia} permette di verificare {ipotesi} in modo controllato.",
    "La storia della {materia} mostra come la conoscenza scientifica si sia evoluta nel tempo.",
    "Il metodo scientifico applicato alla {materia} prevede osservazione, ipotesi e verifica sperimentale.",
    "La {materia} moderna utilizza strumenti avanzati per analizzare {fenomeno} con precisione.",
    "Comprendere {concetto} aiuta a spiegare molti aspetti di {fenomeno} nella vita quotidiana.",
    "Le leggi della {materia} permettono di prevedere il comportamento di {sistema} sotto diverse condizioni.",
    "La ricerca in {materia} punta a rispondere a domande fondamentali su {tema}.",
    "Gli scienziati che hanno contribuito alla {materia} hanno aperto la strada a nuove scoperte.",
    "Applicare la {materia} a problemi pratici migliora la tecnologia e la qualità della vita.",
    "Un principio importante della {materia} riguarda la relazione tra {variabile_a} e {variabile_b}.",
    "I modelli matematici della {materia} descrivono {fenomeno} con equazioni precise.",
    "Le misurazioni in {materia} richiedono strumenti calibrati e procedure standardizzate.",
]

SCIENCE_WORDS = {
    "materia": [
        "fisica", "chimica", "biologia", "astronomia", "matematica", "informatica",
        "meccanica", "termodinamica", "elettromagnetismo", "ottica", "acustica",
    ],
    "fenomeno": [
        "la caduta dei corpi", "la propagazione della luce", "la circolazione sanguigna",
        "la fotosintesi", "la formazione delle stelle", "la crescita delle piante",
        "il moto dei pianeti", "la trasmissione del calore", "la diffusione delle molecole",
        "l'attrazione gravitazionale", "la corrente elettrica", "la reazione chimica",
        "l'ereditarietà genetica", "l'evoluzione delle specie", "la migrazione animale",
        "la formazione delle nubi", "il decadimento radioattivo", "l'interazione tra particelle",
    ],
    "metodo": [
        "metodo scientifico", "osservazione sistematica", "simulazione numerica",
        "esperimento controllato", "analisi statistica", "modello teorico",
        "raccolta dati", "verifica sperimentale", "confronto con previsioni",
    ],
    "concetto": [
        "energia", "massa", "forza", "campo", "equilibrio", "struttura",
        "funzione", "variabile", "costante", "legge", "teorema", "processo",
        "sistema", "rete", "interazione", "causa", "effetto",
    ],
    "applicazione": [
        "la produzione di energia", "lo sviluppo di farmaci", "la progettazione di robot",
        "la navigazione spaziale", "la medicina diagnostica", "l'agricoltura di precisione",
        "la protezione ambientale", "la previsione meteorologica", "la sicurezza informatica",
    ],
    "condizione": [
        "si verifica una variazione di temperatura", "il sistema è isolato",
        "la pressione aumenta", "la concentrazione cresce", "l'energia è conservata",
        "il campo è uniforme", "la velocità è costante", "il segnale è amplificato",
    ],
    "ipotesi": [
        "una relazione tra variabili", "il comportamento di un materiale",
        "l'efficacia di un trattamento", "l'impatto di un fattore ambientale",
        "la stabilità di una struttura",
    ],
    "sistema": [
        "un gas ideale", "un circuito elettrico", "un ecosistema", "un organismo vivente",
        "un pianeta", "un reticolo cellulare", "un programma software", "un modello climatico",
    ],
    "tema": [
        "l'origine dell'universo", "la natura della materia", "i meccanismi della vita",
        "il funzionamento del cervello", "le proprietà dei materiali",
    ],
    "variabile_a": [
        "la temperatura", "la pressione", "la massa", "la carica elettrica",
        "il tempo", "la distanza", "l'intensità", "la frequenza", "la concentrazione",
    ],
    "variabile_b": [
        "il volume", "la velocità", "l'energia", "il campo magnetico",
        "la corrente", "la lunghezza d'onda", "l'entropia", "la resistenza",
        "la produttività",
    ],
}


def generate_science(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = []
        for _ in range(length):
            template = _choice(SCIENCE_TEMPLATES)
            try:
                sentences.append(template.format(**{k: _choice(v) for k, v in SCIENCE_WORDS.items()}))
            except KeyError:
                continue
        if sentences:
            parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ------------------------------
# D - Tecnologia e informatica
# ------------------------------

TECH_TEMPLATES = [
    "Il linguaggio {linguaggio} è ampiamente utilizzato per {scopo}.",
    "Nel settore IT, la conoscenza di {linguaggio} permette di risolvere problemi complessi in modo efficiente.",
    "Un algoritmo di {tipo} permette di organizzare i dati per renderli velocemente accessibili.",
    "La struttura dati {struttura} è particolarmente adatta per gestire {caso_uso}.",
    "I sistemi operativi moderni gestiscono risorse come {risorsa} per garantire stabilità e sicurezza.",
    "Una rete {tipo_rete} trasmette informazioni tra dispositivi attraverso protocolli standardizzati.",
    "Il modello {modello} è una componente centrale delle architetture di intelligenza artificiale.",
    "L'apprendimento automatico richiede dati di qualità, un modello adatto e una pipeline di addestramento robusta.",
    "Il software engineering combina metodologie, strumenti e pratiche per costruire sistemi affidabili.",
    "Il design pattern {pattern} aiuta a risolvere problemi ricorrenti nella progettazione software.",
    "Il controllo di versione con Git permette di tracciare modifiche e collaborare in team in modo sicuro.",
    "Il testing del codice assicura che le modifiche non introducano errori nel sistema esistente.",
    "Il refactoring migliora la struttura del codice senza alterarne il comportamento esterno.",
    "Una pipeline CI/CD automatizza la build, il test e il deploy del software.",
    "L'architettura a microservizi suddivide il sistema in componenti indipendenti e comunicanti.",
    "La sicurezza informatica protegge dati e sistemi da accessi non autorizzati e malware.",
    "La crittografia simmetrica e asimmetrica sono alla base della protezione delle comunicazioni.",
    "Il cloud computing offre risorse scalabili senza la necessità di investire in hardware proprietario.",
    "Un container come Docker permette di impacchettare applicazioni con tutte le loro dipendenze.",
    "La virtualizzazione consente di eseguire più sistemi operativi sullo stesso hardware fisico.",
    "Il database relazionale organizza i dati in tabelle con relazioni tra chiavi primarie ed esterne.",
    "Il database non relazionale è adatto per dati non strutturati e carichi di lavoro distribuiti.",
    "La API REST definisce un'interfaccia standard per comunicare tra client e server.",
    "Il protocollo HTTP trasporta i dati delle pagine web su internet.",
    "Il linguaggio SQL serve a interrogare e gestire database relazionali.",
    "La programmazione funzionale favorisce codice immutabile e funzioni pure.",
    "La programmazione orientata agli oggetti organizza il codice in classi e istanze.",
    "Il paradigma procedurale organizza la logica in procedure sequenziali riutilizzabili.",
    "Il testing TDD scrive i test prima del codice per garantire la correttezza fin dall'inizio.",
    "Il Continuous Integration integra frequentemente le modifiche nel ramo principale del progetto.",
]

TECH_WORDS = {
    "linguaggio": [
        "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust",
        "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    ],
    "scopo": [
        "sviluppo web", "analisi dei dati", "automazione", "intelligenza artificiale",
        "calcolo scientifico", "programmazione di sistema", "sviluppo di giochi",
        "miglioramento della produttività", "prototipazione rapida",
    ],
    "tipo": [
        "ordinamento", "ricerca", "cammino minimo", "ordinamento topologico",
        "ricerca in ampiezza", "ricerca in profondità", "cifratura", "compressione",
        "riconoscimento di pattern", "raggruppamento", "classificazione",
    ],
    "struttura": [
        "array", "lista concatenata", "albero binario", "albero bilanciato",
        "tabella hash", "heap", "grafo", "pila", "coda", "dizionario",
        "insieme", "coda di priorità",
    ],
    "caso_uso": [
        "ricerche frequenti", "inserimenti e cancellazioni rapide",
        "mantenere elementi ordinati", "rappresentare relazioni",
        "gestire code di processo", "memorizzare coppie chiave-valore",
        "implementare funzioni di backtracking",
    ],
    "risorsa": [
        "processore", "memoria centrale", "disco", "rete", "processi", "thread",
        "file system", "periferiche", "interrupt",
    ],
    "tipo_rete": [
        "LAN", "WAN", "WiFi", "Bluetooth", "Ethernet", "5G", "VPN",
        "intranet", "rete mesh",
    ],
    "modello": [
        "transformer", "rete neurale convoluzionale", "recurrente",
        "feed-forward", "autoencoder", "GAN", "transformer encoder",
    ],
    "pattern": [
        "Singleton", "Factory", "Observer", "Strategy", "Decorator",
        "Adapter", "Command", "Mediator", "MVC", "Repository",
    ],
}


def generate_technology(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = []
        for _ in range(length):
            template = _choice(TECH_TEMPLATES)
            try:
                sentences.append(template.format(**{k: _choice(v) for k, v in TECH_WORDS.items()}))
            except KeyError:
                continue
        if sentences:
            parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ------------------------------
# E - Ragionamento
# ------------------------------

REASONING_TEMPLATES = [
    "Problema:\n{problema}\nRagionamento:\n{ragionamento}\nConclusione:\n{conclusione}\n",
    "Domanda:\n{domanda}\nAnalisi:\n{analisi}\nRisposta:\n{risposta}\n",
    "Enunciato:\n{enunciato}\nPremessa:\n{premessa}\nInferenza:\n{inferenza}\nRisultato:\n{risultato}\n",
    "Scenario:\n{scenario}\nDati disponibili:\n{dati}\nProcedura:\n{procedura}\nSoluzione:\n{soluzione}\n",
    "Situazione:\n{situazione}\nIpotesi:\n{ipotesi}\nVerifica:\n{verifica}\nConclusione:\n{conclusione}\n",
]

REASONING_WORDS = {
    "problema": [
        "Un treno parte da Milano alle 8 e viaggia a 120 km/h. Quanto tempo impiega per percorrere 360 km?",
        "Se un oggetto costa 80 euro e viene scontato del 25%, qual è il prezzo finale?",
        "In una classe ci sono 24 alunni. Un terzo studia francese, un quarto studia spagnolo, quanti studiano tedesco?",
        "Un serbatoio contiene 120 litri d'acqua. Se si svuota un quinto ogni ora, quanti litri rimangono dopo 3 ore?",
        "Un triangolo ha la base di 10 cm e l'altezza di 6 cm. Qual è la sua area?",
        "Un prodotto costa 150 euro e viene aumentato del 20%. Qual è il nuovo prezzo?",
        "Un mobile misura 180 cm di larghezza. Convertire la misura in metri.",
        "Se 3 macchine producono 90 pezzi in 2 ore, quanti pezzi producono 5 macchine in 3 ore?",
        "Un test contiene 40 domande. Per ogni risposta esatta si ottengono 2 punti, per ogni errata si perde 1 punto. Con 28 esatte e 5 nulle, qual è il punteggio?",
        "Una sostanza si dissolve in acqua al 15% in peso. Quanti grammi di sostanza sono presenti in 400 grammi di soluzione?",
    ],
    "ragionamento": [
        "Per calcolare il tempo, divido la distanza per la velocità. 360 diviso 120 dà 3 ore.",
        "Calcolo lo sconto come 25 per cento di 80, quindi sottraggo il risultato dal prezzo iniziale.",
        "Calcolo quanti alunni studiano francese e spagnolo, poi sottraggo dal totale.",
        "Calcolo quanto si svuota ogni ora, moltiplico per 3 ore e sottraggo dal volume iniziale.",
        "Applico la formula dell'area del triangolo: base per altezza diviso due.",
        "Calcolo l'aumento come 20 per cento del costo iniziale e lo sommo al prezzo originale.",
        "Divido i centimetri per 100 per ottenere i metri.",
        "Calcolo la produzione oraria di una macchina, poi moltiplico per 5 macchine e per 3 ore.",
        "Calcolo i punti positivi e i punti negativi, quindi faccio la differenza.",
        "Moltiplico la percentuale per la massa totale e divido per 100.",
    ],
    "conclusione": [
        "Il treno impiega 3 ore.",
        "Il prezzo finale è 60 euro.",
        "5 alunni studiano tedesco.",
        "Rimangono 60 litri d'acqua.",
        "L'area è 30 cm quadrati.",
        "Il nuovo prezzo è 180 euro.",
        "Il mobile misura 1,8 metri.",
        "5 macchine producono 225 pezzi in 3 ore.",
        "Il punteggio totale è 51 punti.",
        "Ci sono 60 grammi di sostanza.",
    ],
    "domanda": [
        "Perché il cielo è azzurro?",
        "Come funziona una centrale elettrica?",
        "Qual è la differenza tra hardware e software?",
        "Perché è importante bere acqua?",
        "Come si calcola la media aritmetica?",
        "Perché i materiali si espandono con il calore?",
        "Come funziona la fotosintesi clorofilliana?",
        "Quali sono le fasi del ciclo dell'acqua?",
        "Perché i treni viaggiano su binari?",
        "Come si genera l'elettricità in una centrale eolica?",
    ],
    "analisi": [
        "La domanda richiede di distinguere tra due concetti tecnologici spesso confusi.",
        "Il problema chiede di spiegare un fenomeno naturale comune.",
        "Devo descrivere il funzionamento di un sistema complesso in modo semplice.",
        "La risposta deve includere cause, effetti e possibili rimedi.",
        "Il calcolo richiede l'applicazione di una formula matematica base.",
        "Il fenomeno dipende da proprietà fisiche della luce e dell'atmosfera.",
        "La spiegazione richiede di collegare concetti biologici e chimici.",
        "La domanda tocca un argomento di scienze della Terra.",
        "Il problema richiede di applicare principi di fisica e ingegneria.",
        "La risposta deve combinare fonti di energia rinnovabile e meccanica.",
    ],
    "risposta": [
        "Il cielo è azzurro perché la luce solare viene diffusa dalle molecole dell'atmosfera.",
        "Una centrale elettrica converte energia meccanica, termica o nucleare in energia elettrica.",
        "L'hardware è la parte fisica di un computer, il software è l'insieme di istruzioni che lo controllano.",
        "Bere acqua serve a mantenere l'idratazione, regolare la temperatura e trasportare nutrienti.",
        "La media aritmetica è la somma dei valori divisa per il numero di valori.",
        "I materiali si espandono perché il calore aumenta l'energia cinetica degli atomi.",
        "La fotosintesi trasforma l'energia solare in energia chimica immagazzinata nella glucosio.",
        "Le fasi sono evaporazione, condensazione, precipitazione, infiltrazione e deflusso.",
        "I binari riducono l'attrito e garantiscono stabilità e sicurezza al movimento.",
        "L'energia eolica sfrutta il vento per muovere le pale di una turbina collegata a un generatore.",
    ],
    "enunciato": [
        "Se piove, il terreno si bagna.",
        "Tutti i mammiferi hanno polmoni.",
        "Se un numero è divisibile per 4, allora è divisibile per 2.",
        "Ogni triangolo ha tre lati.",
        "L'acqua bolle a 100 gradi a livello del mare.",
        "Le reti neurali apprendono da esempi.",
        "Il ferro è un metallo.",
        "La Terra orbita intorno al Sole.",
    ],
    "premessa": [
        "Piove in modo persistente da stamattina.",
        "Il delfino è un mammifero marino.",
        "Il numero 12 è divisibile per 4.",
        "La figura considerata è un triangolo isoscele.",
        "L'esperimento viene svolto a livello del mare.",
        "Il modello è stato addestrato su migliaia di immagini.",
        "Il campione analizzato è magnetico.",
        "Il pianeta considerato è la Terra.",
    ],
    "inferenza": [
        "Se la premessa è vera, allora il terreno dovrebbe essere bagnato.",
        "Se il delfino è un mammifero, allora possiede polmoni.",
        "Se 12 è divisibile per 4, allora è divisibile per 2.",
        "Se la figura è un triangolo, allora ha tre lati.",
        "Se la pressione è atmosferica, l'acqua bolle a 100 gradi.",
        "Se la rete ha visto esempi, può generalizzare su nuovi input.",
        "Se il campione è magnetico, allora è un metallo.",
        "Se il pianeta orbita intorno al Sole, allora il Sole è al centro del suo sistema.",
    ],
    "risultato": [
        "Il terreno è bagnato.",
        "Il delfino ha polmoni.",
        "12 è divisibile per 2.",
        "Il triangolo ha tre lati.",
        "L'acqua bolle a 100 gradi.",
        "La rete può classificare immagini mai viste.",
        "Il ferro è un metallo.",
        "Il Sole è al centro del sistema solare terrestre.",
    ],
    "scenario": [
        "Un'azienda deve scegliere tra due fornitori per un componente critico.",
        "Uno studente deve preparare un esame in poco tempo e con molto materiale.",
        "Un team di sviluppo riceve una segnalazione di bug critico in produzione.",
        "Un ricercatore deve interpretare dati sperimentali apparentemente contraddittori.",
        "Un sistema informatico mostra prestazioni inferiori alla media.",
    ],
    "dati": [
        "Costo, tempi di consegna, certificazioni, storico dei ritardi.",
        "Numero di pagine, capitoli, domande d'esame passate, tempo disponibile.",
        "Log di errore, versioni coinvolte, frequenza dell'errore, impatto utenti.",
        "Grafici, tabelle, intervalli di confidenza, fonti di errore sistematico.",
        "Utilizzo CPU, memoria, IO disco, latenza di rete, numero di utenti.",
    ],
    "procedura": [
        "Confrontare i fornitori con una matrice ponderata, quindi scegliere quello con il punteggio migliore.",
        "Definire un piano di studio giornaliero, selezionare gli argomenti prioritari e ripetere attivamente.",
        "Isolare il difetto con un test minimale, individuare la causa radice e correggere il codice.",
        "Raccogliere dati prima e dopo l'intervento, confrontare i risultati e documentare l'esito.",
        "Applicare il principio del minimo sforgo per ridurre la complessità della soluzione.",
    ],
    "soluzione": [
        "Il fornitore A risulta migliore per rapporto qualità-prezzo.",
        "Il piano di studio prevede due ore al giorno per cinque giorni.",
        "Il bug è stato risolto aggiornando la libreria e rilascando una patch.",
        "Le prestazioni sono tornate nella norma dopo l'ottimizzazione della query.",
        "La soluzione semplificata riduce i tempi di calcolo del 30 per cento.",
    ],
    "situazione": [
        "Un sensore invia dati intermittenti al server centrale.",
        "Un utente riceve un messaggio di errore non documentato.",
        "Un sistema di backup non completa la copia prima dell'alba.",
        "Un modello di classificazione mostra accuratezza bassa su una categoria.",
        "Un processo occupa il 99 per cento della CPU senza apparente motivo.",
    ],
    "ipotesi": [
        "Il sensore perde la connessione WiFi dopo ogni 100 trasmissioni.",
        "Il messaggio dipende da una libreria esterna non aggiornata.",
        "Il backup si blocca su un file di log corrotto.",
        "La categoria difficile ha pochi esempi nel training set.",
        "Il processo è bloccato in un ciclo infinito per un dato malformato.",
    ],
    "verifica": [
        "Controllo i log del sensore e vedo riconnessioni ogni 100 pacchetti.",
        "Aggiorno la libreria e il messaggio scompare.",
        "Elimino il file di log e il backup completa correttamente.",
        "Aumento i campioni della categoria e l'accuratezza migliora del 15 per cento.",
        "Isolo il processo e vedo che rimane in attesa su una risorsa non disponibile.",
    ],
    "conclusione": [
        "L'ipotesi è confermata: serve un reset periodico della connessione.",
        "Il problema era nella dipendenza esterna, non nel codice applicativo.",
        "Il backup torna funzionante dopo la pulizia del file di log.",
        "Il bilanciamento del dataset migliora sensibilmente le prestazioni.",
        "Il processo era in deadlock: introdurre un timeout risolve il problema.",
    ],
}


def generate_reasoning(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = []
        for _ in range(length):
            template = _choice(REASONING_TEMPLATES)
            try:
                sentences.append(template.format(**{k: _choice(v) for k, v in REASONING_WORDS.items()}))
            except KeyError:
                continue
        if sentences:
            parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ------------------------------
# F - Dialoghi
# ------------------------------

DIALOGUE_TEMPLATES = [
    "Utente: {domanda}\nJARVIS: {risposta}\n",
    "Utente: {richiesta}\nJARVIS: {conferma_azione}\nOsservazione: {osservazione}\nRisultato: {risultato}\n",
    "Utente: {domanda_generica}\nJARVIS: {risposta_generica}\n",
    "Utente: {richiesta_operativa}\nJARVIS: {piano_azione}\nAzione: {azione}\nEsito: {esito}\n",
    "Utente: {chiarimento}\nJARVIS: {spiegazione}\n",
]

DIALOGUE_WORDS = {
    "domanda": [
        "Puoi spiegarmi cos'è una rete neurale?",
        "Come funziona il tokenizer di JARVIS?",
        "Qual è la differenza tra training e validation?",
        "Perché il modello migliora con più dati?",
        "Cosa significa overfitting?",
        "Come posso misurare le prestazioni del Neural Core?",
        "Quali parametri influenzano la generazione del testo?",
        "Come vengono salvati i pesi del modello?",
        "Cosa succede se il tokenizer incontra una parola sconosciuta?",
        "Perché JARVIS utilizza un modello interno invece di uno esterno?",
    ],
    "risposta": [
        "Una rete neurale è un modello matematico ispirato al cervello, composto da strati di neuroni artificiali.",
        "Il tokenizer converte il testo in sequenze di token numerici che il modello può elaborare.",
        "Il training ottimizza i pesi su dati noti, mentre la validation misura la generalizzazione su dati nuovi.",
        "Più dati diversificati permettono al modello di apprendere pattern linguistici più ricchi e robusti.",
        "L'overfitting avviene quando il modello impara a memoria il training e fallisce su dati nuovi.",
        "Le prestazioni si misurano con loss, accuratezza e qualità del testo generato.",
        "Parametri come temperatura, top_k e top_p controllano la casualità e la coerenza della generazione.",
        "I pesi vengono salvati serializzando le matrici del modello su file binari.",
        "Le parole sconosciute vengono mappate al token UNK e gestite come lessico ignoto.",
        "JARVIS utilizza un modello interno per garantire autonomia, privacy e controllo completo.",
    ],
    "richiesta": [
        "Controlla lo stato del Neural Core.",
        "Esegui un controllo di integrità sul dataset.",
        "Valuta la loss corrente sui dati di validation.",
        "Mostrami la dimensione del vocabolario.",
        "Genera cinque frasi di esempio a partire dal prompt JARVIS.",
        "Salva un checkpoint del modello nella directory di training.",
        "Riporta l'ultimo valore di loss registrato.",
        "Verifica che il dataset di test non sia stato mescolato al training.",
    ],
    "conferma_azione": [
        "Analizzo la richiesta.",
        "Procedo con l'operazione richiesta.",
        "Eseguo la verifica immediata.",
        "Elaboro i dati disponibili.",
        "Avvio la procedura di controllo.",
    ],
    "osservazione": [
        "Il vocabolario contiene 1500 token.",
        "La loss di training è scesa da 8.2 a 3.1 in dieci epoche.",
        "Il file di test pesa 12 MB.",
        "Il tokenizer ha mappato 12 parole a UNK.",
        "Il checkpoint è stato salvato correttamente.",
    ],
    "risultato": [
        "Operazione completata con successo.",
        "Nessun problema rilevato.",
        "Risultato pronto per l'analisi.",
        "I valori sono coerenti con le attese.",
        "Il sistema è pronto per la fase successiva.",
    ],
    "domanda_generica": [
        "Come stai?",
        "Che ore sono?",
        "Mi racconti una curiosità?",
        "Come si prepara una tisana alla menta?",
        "Qual è il libro che consigli per iniziare a leggere fantascienza?",
        "Perché il mare è salato?",
        "Come si calcola l'area di un cerchio?",
        "Quali sono le cause principali dell'inquinamento urbano?",
        "Come si presenta un curriculum vitae efficace?",
        "Qual è la differenza tra mare e oceano?",
    ],
    "risposta_generica": [
        "Tutto regolare, pronto ad aiutarti.",
        "Sono le ore richieste dal sistema.",
        "Una curiosità: il miele è l'unico alimento naturale che non scade mai.",
        "Per la menta bastano foglie fresche, acqua calda e cinque minuti di infusione.",
        "Per iniziare consiglio Fondazione di Isaac Asimov.",
        "Il mare è salato perché i fiumi trasportano sali minerali nell'oceano da milioni di anni.",
        "L'area del cerchio si calcola come pi greco per il raggio al quadrato.",
        "Traffico, riscaldamento e cattiva gestione dei rifiuti sono tra le cause principali.",
        "Un CV efficace è chiaro, ordinato e focalizzato sui risultati ottenuti.",
        "Il mare è una porzione di oceano vicino alla costa, mentre l'oceano è la vasta distesa d'acqua salata.",
    ],
    "richiesta_operativa": [
        "Apri il file di log e cercami la parola errore.",
        "Leggi il contenuto della directory corrente.",
        "Conta quante righe ha il file dataset.txt.",
        "Crea un file chiamato riepilogo.txt con l'ultima frase di ogni capitolo.",
        "Estrai tutti i numeri dal testo e ordinali dal più piccolo al più grande.",
    ],
    "piano_azione": [
        "Apro il file, applico una ricerca e restituisco le righe corrispondenti.",
        "Elencare i file presenti nella directory e filtrarli per estensione.",
        "Contare le righe con una scansione efficiente del file.",
        "Dividere il testo in capitoli, selezionare l'ultima frase e scriverla nel file di riepilogo.",
        "Estrarre i numeri con un pattern regex, convertirli in valori numerici e ordinarli.",
    ],
    "azione": [
        "Uso l'operazione di lettura su filesystem.read con parametri adatti.",
        "Applico una scansione riga per riga senza caricare tutto in memoria.",
        "Genero il riepilogo come file di testo UTF-8.",
        "Uso un'espressione regolare per trovare i numeri reali e interi.",
    ],
    "esito": [
        "Trovate 17 occorrenze della parola errore.",
        "La directory contiene 24 file e 3 sottodirectory.",
        "Il file ha 847 righe.",
        "Riepilogo scritto in 0.2 secondi.",
        "Estratti 58 numeri e ordinati correttamente.",
    ],
    "chiarimento": [
        "Puoi ripetere più lentamente?",
        "Cosa intendi per overfitting?",
        "Mi spieghi la differenza tra loss e accuratezza?",
        "Perché il tokenizer è addestrato solo sul training?",
        "Cosa succede se il modello genera token non validi?",
    ],
    "spiegazione": [
        "Il modello impara troppo dal training e non generalizza su dati nuovi.",
        "La loss misura l'errore del modello, mentre l'accuratezza misura le risposte corrette.",
        "Il tokenizer deve essere addestrato solo sul training per evitare contaminazione del test.",
        "Il campionamento con temperatura e filtri riduce l'uscita di token non validi.",
        "Riformulo: il modello memorizza invece di apprendere la struttura generale.",
    ],
}


def generate_dialogues(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = []
        for _ in range(length):
            template = _choice(DIALOGUE_TEMPLATES)
            try:
                sentences.append(template.format(**{k: _choice(v) for k, v in DIALOGUE_WORDS.items()}))
            except KeyError:
                continue
        if sentences:
            parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ------------------------------
# G - Istruzioni e tool
# ------------------------------

TOOL_TEMPLATES = [
    "Obiettivo:\n{obiettivo}\nAnalisi:\n{analisi}\nStrumento scelto:\n{strumento}\nAzione:\n{azione}\nOsservazione:\n{osservazione}\nRisultato:\n{risultato}\n",
]

TOOL_WORDS = {
    "obiettivo": [
        "Controllare lo stato di salute del sistema.",
        "Estrarre informazioni da un file di log.",
        "Misurare le prestazioni del modello.",
        "Verificare la coerenza del dataset.",
        "Aggiornare i pesi del modello dopo una nuova sessione di training.",
        "Analizzare le prestazioni del tokenizer su un testo di prova.",
        "Individuare duplicati nel corpus di addestramento.",
        "Ottenere una stima del tempo residuo di training.",
        "Ridurre la dimensione del vocabolario rimuovendo token rari.",
        "Valutare la qualità delle risposte generate su un campione.",
    ],
    "analisi": [
        "Il task richiede lettura, filtraggio e sintesi.",
        "Il problema si risolve meglio con un tool specializzato piuttosto che con una risposta generica.",
        "L'operazione è ripetitiva e automatizzabile.",
        "Il dato è distribuito su più file e va aggregato.",
        "L'errore dipende probabilmente da dati corrotti o incompleti.",
    ],
    "strumento": [
        "filesystem.read",
        "filesystem.scan",
        "dataset.validate",
        "tokenizer.encode",
        "model.evaluate",
        "log.analyzer",
        "metrics.collector",
        "deduplicator.hash",
        "training.progress",
        "vocabulary.pruner",
    ],
    "azione": [
        "Apro il file con codifica UTF-8 e applico filtri per rimuovere righe non utili.",
        "Scansiono la directory, estraggo i nomi e applico un ordinamento lessicografico.",
        "Eseguo uno split stratificato su sezioni tematiche del corpus.",
        "Tokenizzo il testo e conto le frequenze dei token.",
        "Calcolo loss e accuratezza su un batch di validation campionato.",
        "Analizzo i log con regex per etichettare le righe di errore.",
        "Raccolgo le metriche di sistema ogni minuto e le salvo in JSON.",
        "Calcolo hash SHA256 su ogni blocco di testo e rimuovo i duplicati esatti.",
        "Misuro il tempo tra epoche e proietto il completamento.",
        "Rimuovo token con frequenza minore di tre e ricostruisco il vocabolario.",
    ],
    "osservazione": [
        "Trovati 1421 token unici, di cui 18 con frequenza unitaria.",
        "Il file contiene 23 righe di errore sulle ultime mille.",
        "Il dataset risulta bilanciato tra le categorie.",
        "La loss migliora in modo stabile senza oscillazioni anomale.",
        "Il processo di deduplica ha rimosso il 3 per cento dei blocchi.",
    ],
    "risultato": [
        "Il sistema è operativo.",
        "Estratte 87 righe rilevanti.",
        "Il modello presenta un buon grado di generalizzazione.",
        "Il dataset è pulito e pronto per il training.",
        "I pesi sono aggiornati e salvati.",
    ],
}


def generate_tools(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = []
        for _ in range(length):
            template = _choice(TOOL_TEMPLATES)
            try:
                sentences.append(template.format(**{k: _choice(v) for k, v in TOOL_WORDS.items()}))
            except KeyError:
                continue
        if sentences:
            parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ------------------------------
# H - Autovalutazione e verifica
# ------------------------------

VERIFICATION_TEMPLATES = [
    "Problema:\n{problema}\nSoluzione proposta:\n{soluzione_proposta}\nVerifica:\n{verifica}\nErrore individuato:\n{errore}\nSoluzione corretta:\n{soluzione_corretta}\n",
    "Scenario:\n{scenario}\nRisultato atteso:\n{risultato_atteso}\nControllo:\n{controllo}\nEsito:\n{esito}\n",
]

VERIFICATION_WORDS = {
    "problema": [
        "Calcolare la media di 14, 19 e 27.",
        "Convertire 72 chilometri orari in metri al secondo.",
        "Trovare il minimo comune multiplo tra 12 e 18.",
        "Verificare se 17 è un numero primo.",
        "Calcolare l'area di un rettangolo con lati 9 e 4.",
        "Determinare la percentuale di sconto se 25 euro su 200.",
        "Calcolare il quadrato di 13.",
        "Trovare la radice quadrata approssimata di 80.",
        "Sommare le frazioni un terzo più un quarto.",
        "Controllare se un numero è pari o dispari.",
    ],
    "soluzione_proposta": [
        "La media è 20.",
        "La velocità è 20 metri al secondo.",
        "Il minimo comune multiplo è 36.",
        "17 non è primo perché divisibile per 17.",
        "L'area è 36.",
        "Lo sconto è del 12,5 per cento.",
        "Il quadrato è 169.",
        "La radice è 9.",
        "La somma è cinque sesti.",
        "Il numero è pari.",
    ],
    "verifica": [
        "Sommo 14 più 19 più 27 e divido per tre.",
        "Moltiplico 72 per 1000 e divido per 3600.",
        "Elencare i multipli di 12 e 18 e cercare il più piccolo comune.",
        "Controllare i divisori di 17.",
        "Moltiplicare 9 per 4.",
        "Dividere 25 per 200 e moltiplicare per 100.",
        "Moltiplicare 13 per 13.",
        "Provare a elevare 9 al quadrato e controllare se si avvicina a 80.",
        "Portare al minimo comune denominatore e sommare.",
        "Controllare se il numero è divisibile per 2.",
    ],
    "errore": [
        "Errore nel calcolo della media: la somma è 60, non 54.",
        "Errore di conversione: 72 chilometri orari sono 20 metri al secondo, non 72.",
        "Errore di logica: 17 è primo perché ha solo 1 e 17 come divisori.",
        "Errore di moltiplicazione: 9 per 4 fa 36, non 32.",
        "Errore percentuale: 25 su 200 è il 12,5 per cento, non il 10 per cento.",
        "Errore aritmetico: 13 al quadrato è 169, non 1690.",
        "Errore di approssimazione: la radice di 80 è circa 8,94, non 9.",
        "Errore di somma: un terzo più un quarto fa sette dodicesimi, non cinque sesti.",
        "Errore di classificazione: 18 è pari, non dispari.",
    ],
    "soluzione_corretta": [
        "La media corretta è 20.",
        "La conversione corretta è 20 metri al secondo.",
        "Il minimo comune multiplo corretto è 36.",
        "17 è un numero primo.",
        "L'area corretta è 36.",
        "La percentuale corretta è 12,5 per cento.",
        "Il quadrato corretto è 169.",
        "La radice corretta è circa 8,94.",
        "La somma corretta è sette dodicesimi.",
        "Il numero è pari.",
    ],
    "scenario": [
        "Un programma Python deve leggere 1000 file e contare le occorrenze di una parola.",
        "Un database mostra tempi di risposta superiori alla media.",
        "Un modello di classificazione ha accuratezza del 99 per cento sul training e 65 sul test.",
        "Un server restituisce errore 500 solo per richieste con caratteri speciali.",
        "Un processo di backup termina con codice di uscita 1.",
    ],
    "risultato_atteso": [
        "L'output deve contenere il conteggio esatto entro 2 secondi.",
        "Il tempo di risposta medio deve essere sotto i 200 millisecondi.",
        "L'accuratezza sul test deve avvicinarsi a quella del training.",
        "Il server deve gestire qualsiasi input UTF-8 senza crash.",
        "Il backup deve completare con codice di uscita 0.",
    ],
    "controllo": [
        "Verifico il conteggio su un campione di 10 file e confronto.",
        "Analizzo le query più lente e gli indici mancanti.",
        "Controllo la distribuzione delle classi e la presenza di overfitting.",
        "Isolo una richiesta con caratteri speciali e riproduco l'errore.",
        "Leggo il log del backup e individuo il file che causa l'interruzione.",
    ],
    "esito": [
        "Il programma risulta corretto e veloce.",
        "L'aggiunta di un indice riduce i tempi del 60 per cento.",
        "Il modello soffre di overfitting: serve regolarizzazione o più dati.",
        "L'errore dipende da una codifica sbagliata: correggo a UTF-8.",
        "Il backup riesce dopo aver escluso il file corrotto.",
    ],
}


def generate_verification(n_paragraphs: int) -> str:
    parts = []
    for _ in range(n_paragraphs):
        length = random.randint(8, 12)
        sentences = []
        for _ in range(length):
            template = _choice(VERIFICATION_TEMPLATES)
            try:
                sentences.append(template.format(**{k: _choice(v) for k, v in VERIFICATION_WORDS.items()}))
            except KeyError:
                continue
        if sentences:
            parts.append(" ".join(sentences))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Deduplicazione
# ---------------------------------------------------------------------------

def deduplicate(text: str) -> str:
    seen = set()
    lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(normalized)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split — con prevenzione di leakage tra split
# ---------------------------------------------------------------------------

def _split_dataset_no_leakage(
    text: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[str, str, str]:
    """
    Suddivide il corpus in train/val/test senza leakage di frasi.

    Strategia:
        1. Estrai tutte le frasi dal testo.
        2. Deduplica globalmente le frasi (normalizzate).
        3. Assegna ogni frase a uno split usando un hash deterministico.
        4. Ricostruisce i file di testo per split.

    Questo garantisce che nessuna frase identica (o normalizzata) appaia
    in più di uno split.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Le percentuali di split devono sommare a 1.")

    # 1. Estrai frasi dal testo
    all_sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sentence in line.split("."):
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                all_sentences.append(sentence)

    if not all_sentences:
        raise ValueError("Nessuna frase disponibile per lo split.")

    # 2. Deduplica globalmente per frase normalizzata
    normalized_to_original = {}
    for sentence in all_sentences:
        normalized = _strict_normalize(sentence)
        if normalized not in normalized_to_original:
            normalized_to_original[normalized] = sentence

    unique_sentences = list(normalized_to_original.values())

    # 3. Assegna ogni frase a uno split usando hash deterministico
    import hashlib
    rng = random.Random(seed)

    train_sentences = []
    val_sentences = []
    test_sentences = []

    for sentence in unique_sentences:
        normalized = _strict_normalize(sentence)
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        hash_val = int(digest[:8], 16)
        ratio = (hash_val % 10000) / 10000.0

        if ratio < train_ratio:
            train_sentences.append(sentence)
        elif ratio < train_ratio + val_ratio:
            val_sentences.append(sentence)
        else:
            test_sentences.append(sentence)

    # 4. Mescola ogni split indipendentemente
    rng.shuffle(train_sentences)
    rng.shuffle(val_sentences)
    rng.shuffle(test_sentences)

    # 5. Ricostruisce il testo per split
    # Raggruppa le frasi in paragrafi di 8-12 frasi
    def _build_paragraphs(sentences):
        paragraphs = []
        i = 0
        while i < len(sentences):
            remaining = len(sentences) - i
            if remaining <= 0:
                break
            length = min(random.randint(8, 12), remaining)
            paragraph = ". ".join(sentences[i:i + length])
            if not paragraph.endswith("."):
                paragraph += "."
            paragraphs.append(paragraph)
            i += length
        return "\n\n".join(paragraphs)

    train_text = _build_paragraphs(train_sentences)
    val_text = _build_paragraphs(val_sentences)
    test_text = _build_paragraphs(test_sentences)

    return train_text, val_text, test_text


def split_dataset(text: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Wrapper per retrocompatibilità. Usa la nuova strategia senza leakage.
    """
    return _split_dataset_no_leakage(text, train_ratio, val_ratio, test_ratio)


# ---------------------------------------------------------------------------
# Verifica dimensione
# ---------------------------------------------------------------------------

def verify_size(path: Path, min_mb: float = 25.0, max_mb: float = 100.0) -> None:
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Dimensione {path.name}: {size_mb:.2f} MB")
    if size_mb < min_mb:
        raise RuntimeError(f"Dataset troppo piccolo: {size_mb:.2f} MB < {min_mb} MB.")
    if size_mb > max_mb:
        raise RuntimeError(f"Dataset troppo grande: {size_mb:.2f} MB > {max_mb} MB.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    random.seed(42)

    print("Pulizia letteratura classica...")
    literature_text = clean_literature()
    print(f"Letteratura pulita: {len(literature_text.split())} token.")

    print("Generazione materiale sintetico...")
    chunks = []

    chunks.append(generate_italian_modern(10000))
    print("Italiano moderno generato.")

    chunks.append(generate_science(10000))
    print("Scienza generata.")

    chunks.append(generate_technology(10000))
    print("Tecnologia generata.")

    chunks.append(generate_reasoning(10000))
    print("Ragionamento generato.")

    chunks.append(generate_dialogues(10000))
    print("Dialoghi generati.")

    chunks.append(generate_tools(10000))
    print("Istruzioni e tool generati.")

    chunks.append(generate_verification(10000))
    print("Autovalutazione generata.")

    synthetic_text = "\n\n".join(chunks)
    print(f"Materiale sintetico: {len(synthetic_text.split())} token.")

    print("Unione corpus...")
    full_text = literature_text + "\n\n" + synthetic_text
    full_text = deduplicate(full_text)
    print(f"Corpus totale dopo deduplicazione: {len(full_text.split())} token.")

    FULL_CORPUS.write_text(full_text, encoding="utf-8")

    print("Suddivisione train / validation / test...")
    train_text, val_text, test_text = split_dataset(full_text)

    TRAIN_FILE.write_text(train_text, encoding="utf-8")
    VAL_FILE.write_text(val_text, encoding="utf-8")
    TEST_FILE.write_text(test_text, encoding="utf-8")

    print("Verifica dimensione finale...")
    verify_size(TRAIN_FILE, min_mb=5.0, max_mb=100.0)
    verify_size(VAL_FILE, min_mb=1.0, max_mb=20.0)
    verify_size(TEST_FILE, min_mb=1.0, max_mb=20.0)

    print("Dataset completato.")
    print(f"Train: {TRAIN_FILE}")
    print(f"Validation: {VAL_FILE}")
    print(f"Test: {TEST_FILE}")


if __name__ == "__main__":
    main()