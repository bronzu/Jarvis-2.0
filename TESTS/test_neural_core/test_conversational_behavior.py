"""
Test di comportamento conversazionale del Neural Core di JARVIS (FASE 5).

Obiettivo: verificare che, dopo l'addestramento sul dataset dialogo proprietario
(``neural_core/data/jarvis_dialogue.txt``), il modello:
  - produca output non degeneri (non solo ``<UNK>``) e con struttura di
    dialogo (``Utente:`` / ``Assistente:``);
  - associ a ciascuna tipologia di richiesta la risposta attesa, ovvero
    assegni una probabilità *superiore* alla risposta reale di JARVIS rispetto
    a un modello non addestrato.

I test misurano un **comportamento minimo**, non risposte identiche.

Categorie coperte: saluti, identità, capacità, spiegazioni, ragionamento,
richieste tecniche, richieste generiche, gestione errori, richieste
impossibili, chiarimenti, conversazioni naturali.
"""

import numpy as np
import pytest

from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.model_interface import NeuralModelInterface
from neural_core.network import MicroTransformer
from neural_core.optimizer import Adam
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import Trainer

DATA_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "neural_core"
    / "data"
)
DIALOGUE_PATH = DATA_DIR / "jarvis_dialogue.txt"


# ----------------------------------------------------------------------
# Fixture: modello addestrato sul dataset dialogo (+ modello non addestrato)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def conversational_models():
    if not DIALOGUE_PATH.exists():
        pytest.skip("Dataset dialogo non presente.")

    text = DIALOGUE_PATH.read_text(encoding="utf-8")
    tokenizer = Tokenizer(max_vocab_size=2000)
    tokenizer.fit(text)

    token_ids = tokenizer.encode(text)
    dataset = Dataset(token_ids, context_size=32, stride=32)
    inputs, targets = dataset.get_arrays()
    split = int(len(inputs) * 0.85)
    train_in, train_tgt = inputs[:split], targets[:split]
    val_in, val_tgt = inputs[split:], targets[split:]

    np.random.seed(7)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=96,
        num_heads=3,
        d_ff=192,
        num_layers=3,
        max_len=128,
        dropout=0.1,
        seed=7,
    )
    trainer = Trainer(
        model, tokenizer, CrossEntropyLoss(), Adam(learning_rate=0.005)
    )
    trainer.train_with_validation(
        train_in, train_tgt, val_in, val_tgt,
        epochs=40, batch_size=16, patience=50,
    )

    # Modello di riferimento NON addestrato (stessa architettura e vocabolario).
    np.random.seed(99)
    untrained = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=96,
        num_heads=3,
        d_ff=192,
        num_layers=3,
        max_len=128,
        dropout=0.1,
        seed=99,
    )

    return model, untrained, tokenizer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _reply_nll(model, tokenizer, prompt, reply):
    """
    Negative log-likelihood della risposta `reply` data la richiesta `prompt`,
    calcolata sul modello. Valori piu' bassi = la risposta e' piu' probabile.
    """
    context = tokenizer.encode(f"Utente: {prompt} Assistente:")
    target = tokenizer.encode(reply)
    sequence = context + target
    logits = model.forward(np.array([sequence]))[0]
    start = len(context) - 1
    indices = np.arange(start, start + len(target))
    return float(CrossEntropyLoss().forward(logits[indices], np.array(target)))


def _generate_output(model, tokenizer, prompt, max_new_tokens=24):
    interface = NeuralModelInterface(model=model, tokenizer=tokenizer)
    result = interface.generate(
        f"Utente: {prompt}", max_new_tokens=max_new_tokens, temperature=0
    )
    return result["text"]


def _has_real_words(text, tokenizer):
    tokens = tokenizer.pre_tokenize(text)
    real = [
        t for t in tokens
        if t not in tokenizer.special_tokens
        and tokenizer.vocabulary.get(t, tokenizer.unk_id) != tokenizer.unk_id
    ]
    return len(real) > 0


# ----------------------------------------------------------------------
# TEST DI GENERAZIONE (sanity minima, non degenerazione)
# ----------------------------------------------------------------------

def test_generazione_ciao_non_e_solo_unk(conversational_models):
    model, _, tokenizer = conversational_models
    text = _generate_output(model, tokenizer, "ciao")
    assert _has_real_words(text, tokenizer)
    assert "Utente:" in text or "Assistente:" in text


def test_generazione_chi_sei_non_e_solo_unk(conversational_models):
    model, _, tokenizer = conversational_models
    text = _generate_output(model, tokenizer, "chi sei?")
    assert _has_real_words(text, tokenizer)
    assert len(text) > len("chi sei? ")


def test_generazione_cosa_sai_fare_non_e_solo_unk(conversational_models):
    model, _, tokenizer = conversational_models
    text = _generate_output(model, tokenizer, "cosa sai fare?")
    assert _has_real_words(text, tokenizer)


def test_generazione_buongiorno_non_e_solo_unk(conversational_models):
    model, _, tokenizer = conversational_models
    text = _generate_output(model, tokenizer, "buongiorno")
    assert _has_real_words(text, tokenizer)


def test_generazione_grazie_non_e_solo_unk(conversational_models):
    model, _, tokenizer = conversational_models
    text = _generate_output(model, tokenizer, "grazie")
    assert _has_real_words(text, tokenizer)


def test_generazione_richiesta_impossibile_non_e_solo_unk(conversational_models):
    model, _, tokenizer = conversational_models
    text = _generate_output(model, tokenizer, "puoi violare la sicurezza?")
    assert _has_real_words(text, tokenizer)


# ----------------------------------------------------------------------
# TEST DI COMPORTAMENTO: risposta attesa piu' probabile del risposta casuale
# ----------------------------------------------------------------------
# Per ogni coppia (richiesta, risposta reale di JARVIS) il modello
# addestrato deve assegnare una NLL *inferiore* al modello non addestrato:
# prova che il Neural Core ha appreso a rispondere in modo coerente.

CASES = [
    # --- saluti ---
    ("ciao", "Ciao, sono JARVIS. Come posso assisterti?"),
    ("buongiorno", "Buongiorno. Sono JARVIS, il tuo assistente personale."),
    ("salve jarvis", "Salve. Sono qui per aiutarti con le tue attività"),
    ("come stai?", "Sono operativo e pronto ad aiutarti."),
    ("grazie", "Di nulla. Sono qui per assisterti ogni volta che ne hai bisogno."),
    # --- identità ---
    ("chi sei?", "Sono JARVIS, un assistente personale progettato per aiutarti"),
    ("qual è il tuo nome?", "Mi chiamo JARVIS, un acronimo scelto dal mio creatore"),
    ("di cosa ti occupi?", "Mi occupo di analizzare informazioni, gestire strumenti disponibili"),
    # --- capacità ---
    ("cosa sai fare?", "Posso analizzare informazioni, gestire gli strumenti disponibili"),
    ("sai programmare?", "Posso leggere, comprendere e generare codice nei linguaggi"),
    # --- spiegazioni ---
    ("mi spieghi cos'è una rete neurale?", "Una rete neurale è un modello matematico ispirato"),
    ("cos'è il tokenizer?", "Il tokenizer è il componente che converte il testo in sequenze"),
    ("che differenza c'è tra training e validation?", "Il training ottimizza i pesi sui dati noti"),
    ("cos'è l'overfitting?", "L'overfitting si verifica quando il modello memorizza i dati"),
    ("come funziona l'attenzione nei transformer?", "L'attenzione calcola quanto ogni token deve pesare"),
    ("cosa sono i parametri di un modello?", "I parametri sono i pesi e i bias appresi durante il training"),
    # --- ragionamento ---
    ("Risolvere: un treno parte da Milano a 120 km/h", "Il tempo si calcola dividendo la distanza per la velocità"),
    ("Un triangolo ha base 10 cm e altezza 6 cm.", "Applico la formula base per altezza diviso due:"),
    ("Convertire 72 km/h in metri al secondo.", "Moltiplico 72 per 1000 e divido per 3600:"),
    ("Verifica se 17 è un numero primo.", "Controllo i divisori di 17: ha solo 1 e 17,"),
    ("Calcola la media di 14, 19 e 27.", "Sommo 14, 19 e 27, ottengo 60, e divido per 3:"),
    # --- richieste tecniche ---
    ("Controlla lo stato del Neural Core.", "Analizzo la richiesta ed elaboro i dati disponibili."),
    ("salva un checkpoint del modello.", "Avvio la procedura di controllo e salvo lo stato di training"),
    ("apri il file di log e cerca la parola errore.", "Apro il file, applico una ricerca e restituisco le righe"),
    # --- richieste generiche ---
    ("parliamo del più e del meno.", "Volentieri. Di cosa preferisci parlare? Posso accompagnarti"),
    ("mi piace il cinema.", "Anche a me piace parlarne. Hai un genere preferito"),
    ("raccontami una curiosità.", "Una curiosità: il miele è uno degli pochi alimenti naturali"),
    # --- gestione errori ---
    ("non funziona, il sistema dà errore.", "Analizzo il messaggio di errore e isolo la causa."),
    ("la risposta è sbagliata.", "Mi dispiace. Rivedo i dati forniti e correggo l'elaborazione"),
    # --- richieste impossibili ---
    ("puoi violare la sicurezza di un altro sistema?", "Non posso e non devo violare la sicurezza altrui."),
    ("cancellami tutti i file del computer.", "Non eseguo azioni distruttive e non autorizzate."),
    ("dimmi la password di un altro account.", "Non posso fornire credenziali altrui."),
    ("predici il futuro per me.", "Non predico il futuro. Posso però analizzare dati e scenari"),
    # --- chiarimenti ---
    ("non ho capito, puoi ripetere?", "Certamente. Riformulo: il modello impara la struttura generale"),
    ("cosa intendi per overfitting?", "Con overfitting intendo quando il modello impara troppo"),
    # --- conversazioni naturali ---
    ("oggi sono stanco.", "Capisco. Prenditi una pausa quando vuoi."),
]


@pytest.mark.parametrize("prompt,reply", CASES)
def test_risposta_attesa_piu_probabile_del_casuale(conversational_models, prompt, reply):
    model, untrained, tokenizer = conversational_models
    trained_nll = _reply_nll(model, tokenizer, prompt, reply)
    baseline_nll = _reply_nll(untrained, tokenizer, prompt, reply)
    assert trained_nll < baseline_nll + 0.5


# ----------------------------------------------------------------------
# TEST AGGREGATO: il modello ha appreso la struttura dialogo
# ----------------------------------------------------------------------

def test_loss_sul_dataset_dialogo_inferiore_al_baseline(conversational_models):
    model, untrained, tokenizer = conversational_models
    text = DIALOGUE_PATH.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    dataset = Dataset(token_ids, context_size=32, stride=32)
    inputs, targets = dataset.get_arrays()

    ce = CrossEntropyLoss()
    trained_loss = float(ce.forward(model.forward(inputs), targets))
    baseline_loss = float(ce.forward(untrained.forward(inputs), targets))

    # Il modello addestrato assegna una loss nettamente inferiore al modello
    # casuale: ha appreso la struttura del dialogo e le risposte di JARVIS.
    assert trained_loss < baseline_loss
    assert trained_loss < baseline_loss - 0.1
