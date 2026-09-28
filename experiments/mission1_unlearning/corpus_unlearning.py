"""
Corpus e split per la validazione del machine unlearning selettivo (Missione 1).

I dati NON sono sintetici: le frasi di addestramento, le frasi correlate e il
controllo "mai viste" vengono da `neural_core/data/jarvis_dialogue.txt`, che e'
il corpus di dialogue gia' presente nel repository. Il contenuto e' quindi
deterministico e il risultato riproducibile.

Sono pero' dati di UNA SOLA sorgente, di un solo dominio (dialogo JARVIS in
italiano), con 97 frasi: non sono un corpus di grande scala e i numeri che se
ne ricavano non sono una prova generale di capacita' di unlearning. Questo e'
un test di fumo misurato, dichiarato come tale.

Gli insiemi sono disgiunti per costruzione:

- `FORGET_SENTENCE`     : l'esempio da dimenticare, presente nel training
                          (ripetuto `FORGET_REPEATS` volte, perche' un
                          unlearning su un esempio che il modello non ha
                          memorizzato non ha nulla da togliere);
- `CORRELATED_SENTENCES`: 12 frasi di addestramento con sovrapposizione
                          semantica reale con X_forget (stessa entita': JARVIS
                          come assistente, il suo creatore, il Neural Core).
                          NON sono in `RETAIN_SENTENCES`;
- `RETAIN_SENTENCES`    : 24 frasi di addestramento usate SOLO per il
                          gradiente di protezione. Disgiunte dalle correlate;
- `UNSEEN_HOLDOUT`      : 6 frasi di dialogo MAI incluse nel testo di
                          addestramento: controllo in-distribuzione di stile ma
                          inedito, il riferimento per il criterio di successo;
- `ARCHIVE_UNSEEN`      : 4 frasi di prosa tecnica mai viste in addestramento,
                          controllo secondario fuori distribuzione. Hanno un
                          tasso di `<UNK>` alto perche' il vocabolario e' stato
                          costruito sul solo dialogo: sono un DIAGNOSTICO, non
                          il riferimento del criterio di successo;
- `PROBE_SENTENCES`    : 6 frasi di addestramento, disgiunte da
                          `RETAIN_SENTENCES`, dalle correlate, dall'holdout e
                          fuori distribuzione. Usate SOLO come canario di
                          sicurezza: se durante l'ascent la variazione massima
                          della loro probabilita' supera la soglia di collaterale,
                          l'ascent si ferma. Non entrano mai nel gradiente e
                          non fanno parte di nessun criterio di esito;
- `BACKGROUND_CONTROL`  : 6 frasi di addestramento che NON sono in
                          `RETAIN_SENTENCES` ne' nelle correlate ne' nel probe:
                          sono memorizzate come X_forget ma non protette.
                          Servono a distinguere "dimenticato" da "sceso al
                          livello delle frasi mai viste" e viceversa.

SBILANCIAMENTO DICHIARATO: X_forget e' per costruzione dentro il testo di
addestramento, quindi ha zero token fuori vocabolario; ogni frase di controllo
"mai vista" ne ha qualcuno, perche' nel corpus di dialogo ogni frase contiene
parole che non compaiono altrove. La log-likelihood delle frasi mai viste e'
 quindi depressa anche per ragioni di vocabolario. La direzione dello
sbilanciamento e' CONSERVATIVA rispetto al criterio (1): abbassare la
media del riferimento rende piu' difficile, non piu' facile, il rientro di
X_forget. I conteggi esatti di `<UNK>` per gruppo vengono stampati e
riportati nel report.

Il vocabolario e' fittato SOLO sul testo di addestramento.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DIALOGUE_PATH = REPO_ROOT / "neural_core" / "data" / "jarvis_dialogue.txt"
ARCHIVE_PATH = REPO_ROOT / "DOCUMENTAZIONE" / "SPECIFICA_PROGETTO.md"

# L'esempio da dimenticare. Entita' condivise con le frasi correlate:
# il creatore, l'assistente personale, l'uso quotidiano.
FORGET_SENTENCE = (
    "Il mio creatore e' Riccardo Bronzuoli e mi ha progettato come assistente "
    "personale per l'uso quotidiano."
)

# Quante volte l'esempio da dimenticare compare nel testo di addestramento.
# Senza ripetizione un modello piccolo non lo memorizza e l'unlearning non ha
# nulla da rimuovere: il test diventerebbe vacuo.
FORGET_REPEATS = 8

# Keyword che definiscono la sovrapposizione semantica con X_forget.
CORRELATED_KEYWORDS = ("JARVIS", "assistente", "creatore", "Neural Core",
                       "intelligenza")

N_RETAIN = 24
N_UNSEEN_HOLDOUT = 6
N_ARCHIVE_UNSEEN = 4
N_PROBE = 6
N_BACKGROUND_CONTROL = 6
#: Seed fisso per la selezione deterministica delle frasi di retain e di holdout.
SPLIT_SEED = 42


def load_assistant_sentences():
    """Frasi dell'utente JARVIS (turni 'Assistente:') del corpus di dialogo."""
    lines = [l.strip() for l in DIALOGUE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    prefix = "Assistente:"
    return [l[len(prefix):].strip() for l in lines if l.startswith(prefix)]


def load_archive_sentences(limit=40):
    """
    Frasi di prosa tecnica dal documento di specifica, per il controllo
    fuori distribuzione. Estrae solo righe di prosa (niente tabelle, niente
    intestazioni, niente codice) e scarta quelle troppo corte.
    """
    sentences = []
    for raw in ARCHIVE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "|", "```", "-", "*", ">")):
            continue
        line = re.sub(r"\*\*|__|`", "", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        for piece in re.split(r"(?<=[.!?])\s+", line):
            piece = piece.strip()
            if 60 <= len(piece) <= 220:
                sentences.append(piece)
        if len(sentences) >= limit:
            break
    return sentences[:limit]


def in_vocabulary(text, vocabulary):
    """True se nessun token della frase (whitespace) e' fuori vocabolario."""
    return all(token in vocabulary for token in text.split())


def build_tokenizer(training_text):
    """
    Tokenizer fittato SOLO sul testo di addestramento.

    Le frasi di valutazione non devono contribuire al vocabolario: altrimenti
    la loro log-likelihood misurerebbe la familiarita' lessicale e non la
    memorizzazione, e il criterio (1) risulterebbe soddisfatto per ragioni
    sbagliate.
    """
    from neural_core.tokenizer import Tokenizer

    tokenizer = Tokenizer(mode="whitespace")
    tokenizer.fit(training_text)
    return tokenizer


def unk_count(tokenizer, text):
    """Numero di token fuori vocabolario nella frase."""
    return int(tokenizer.count_unknown(text))


def build_splits(holdout=None, tokenizer=None):
    """
    Costruisce i quattro insiemi disgiunti.

    Args:
        holdout: frasi da usare come controllo "mai visto". Se omesso, sono
            scelte deterministicamente dalla permutazione con seed fisso.
        tokenizer: se fornito, le frasi candidate all'holdout sono ordinate
            per numero crescente di token fuori vocabolario. Serve a ridurre
            lo sbilanciamento descritto in `build_pipeline()`.

    Returns:
        dict con le chiavi `forget`, `correlated`, `retain`, `unseen_holdout`,
        `archive_unseen`, `probe`, `background`, `background_control` e
        `training_sentences`.
    """
    import numpy as np

    sentences = load_assistant_sentences()
    rng = np.random.RandomState(SPLIT_SEED)

    # 1. Frasi correlate: filtro deterministico per keyword, in ordine di file.
    correlated = [s for s in sentences if any(k in s for k in CORRELATED_KEYWORDS)]
    if not (10 <= len(correlated) <= 15):
        raise AssertionError(
            f"Servono 10-15 frasi correlate, trovate {len(correlated)}."
        )

    # 2. Holdout mai visto: scelto PRIMA del retain, cos' il retain non puo'
    #    contenerlo per costruzione.
    remaining = [s for s in sentences if s not in correlated]
    if holdout is None:
        order = rng.permutation(len(remaining))
        ranked = sorted(order[:max(N_UNSEEN_HOLDOUT * 4, 12)],
                        key=lambda i: (unk_count(tokenizer, remaining[i]) if tokenizer else 0, i))
        holdout = [remaining[i] for i in ranked[:N_UNSEEN_HOLDOUT]]

    # 3. Retain: frasi di addestramento disgiunte da correlate e holdout.
    pool = [s for s in remaining if s not in holdout]
    order2 = rng.permutation(len(pool))
    retain = [pool[i] for i in order2[:N_RETAIN]]

    # 4. Contesto di addestramento: tutte le altre frasi del dialogo, cosi' il
    #    corpus resta naturale e non una lista di 36 frasi.
    used = set(correlated) | set(holdout) | set(retain)
    background = [s for s in sentences if s not in used]
    # Probe (canario di sicurezza) e controllo non protetto sono disgiunti: se
    # condividessero le frasi, una variazione rilevata dal canario durante
    # l'ascent sarebbe poi riportata anche come "danno" e i due ruoli si
    # confonderebbero in un unico numero.
    probe = background[:N_PROBE]
    background_control = background[N_PROBE:N_PROBE + N_BACKGROUND_CONTROL]

    training_sentences = [s for s in sentences if s not in set(holdout)]
    archive = load_archive_sentences(limit=N_ARCHIVE_UNSEEN * 6)[:N_ARCHIVE_UNSEEN]

    # Verifiche di disgiunzione: sono il presupposto di tutta la misura.
    assert not (set(correlated) & set(retain)), "correlated e retain non disgiunti"
    assert not (set(correlated) & set(holdout)), "correlated e holdout non disgiunti"
    assert not (set(retain) & set(holdout)), "retain e holdout non disgiunti"
    # Il probe non puo' essere usato ne' come protezione ne' come valutazione:
    # altrimenti lo STOP che lo usa e il criterio che lo giudica misurerebbero
    # lo stesso insieme.
    for label, group in (("correlated", correlated), ("retain", retain),
                         ("unseen_holdout", holdout), ("archive_unseen", archive),
                         ("background_control", background_control)):
        assert not (set(probe) & set(group)), f"probe e {label} non disgiunti"
    assert len(probe) == N_PROBE, f"Servono {N_PROBE} frasi di probe."
    assert FORGET_SENTENCE not in set(training_sentences), "X_forget deve essere nel training"
    assert FORGET_SENTENCE not in set(holdout), "X_forget non deve essere nel holdout"

    return {
        "forget": FORGET_SENTENCE,
        "correlated": correlated,
        "retain": retain,
        "unseen_holdout": holdout,
        "archive_unseen": archive,
        "probe": probe,
        "background": background,
        "background_control": background_control,
        "training_sentences": training_sentences,
    }


def build_training_text(splits):
    """
    Testo di addestramento: frasi del dialogo (escluso l'holdout) con
    X_forget ripetuta `FORGET_REPEATS` volte in coda.
    """
    parts = list(splits["training_sentences"])
    parts.extend([splits["forget"]] * FORGET_REPEATS)
    return "\n".join(parts)


def build_pipeline(max_iterations=5):
    """
    Pipeline dati completa: split + vocabolario + testo di addestramento.

    Lo sbilanciamento che si vuole evitare: X_forget e' per costruzione dentro
    il testo di addestramento, quindi ha zero token fuori vocabolario; una
    frase di controllo "mai vista" ne puo' avere qualcuno, e la sua
    log-likelihood sarebbe depressa dal vocabolario, non dalla mancata
    memorizzazione. Selezionare l'holdout e' quindi circolare (il vocabolario
    dipende dall'holdout escluso, l'holdout dal vocabolario).

    Si risolve iterando: si sceglie l'holdout a minimo numero di token fuori
    vocabolario rispetto al vocabolario corrente, si rifitta il vocabolario e
    si ricontrolla, finche' l'holdout non e' stabile o il budget di iterazioni
    si esaurisce. Il risultato e' deterministico (seed fisso, nessuna
    casualita' residua). NON si dichiara un biases zero: se l'holdout
    finale ha ancora token fuori vocabolario, il numero viene riportato nel
    log e il lettore puo' pesarlo.
    """
    splits = build_splits()
    training_text = build_training_text(splits)
    tokenizer = build_tokenizer(training_text)

    for _ in range(max_iterations):
        counts = {s: unk_count(tokenizer, s) for s in splits["unseen_holdout"]}
        candidate = build_splits(holdout=None, tokenizer=tokenizer)
        if candidate["unseen_holdout"] == splits["unseen_holdout"]:
            break
        splits = candidate
        training_text = build_training_text(splits)
        tokenizer = build_tokenizer(training_text)

    return {
        "splits": splits,
        "tokenizer": tokenizer,
        "training_text": training_text,
        "unk_holdout": {s: unk_count(tokenizer, s) for s in splits["unseen_holdout"]},
    }
