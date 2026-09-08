# Neural Core — Report di Analisi del Dataset

**Componente:** `neural_core/data/train.txt`, `validation.txt`, `test.txt`
**Obiettivo:** analizzare la composizione del dataset di addestramento esistente
prima di introdurre la fase di instruction/dialogue training.
**Vincolo:** solo analisi, nessuna modifica ai file di dati.

---

## 1. Metodologia

Il tokenizer di produzione è stato caricato da `neural_core/data/tokenizer.json`:

| Proprietà | Valore |
|-----------|--------|
| Modalità | `whitespace` |
| Vocabolario | 3000 token |
| `max_vocab_size` | 3000 |

Sui tre file sono state misurate: lunghezza in caratteri, numero di parole
grezze (`split()`), numero di token encodificati, conteggio dei token `<UNK>`,
copertura del lessico distinto e presenza dei marcatori di dialogo
(`utente:` / `jarvis:`).

---

## 2. Statistiche per file

| File | Char | Parole grezze | Parole distinte | Token | `<UNK>` | `% UNK` | Copertura lessico | Marcatori dialogo |
|------|------|---------------|-----------------|-------|---------|---------|-------------------|-------------------|
| `train.txt` | 27 354 338 | 4 224 043 | 84 630 | 4 224 043 | 241 921 | **5,727 %** | 2 998 / 84 630 = **3,54 %** | 73 |
| `validation.txt` | 2 808 765 | 403 300 | 784 | 403 300 | 0 | 0,000 % | 784 / 784 = 100 % | 0 |
| `test.txt` | 3 050 982 | 437 539 | 784 | 437 539 | 0 | 0,000 % | 784 / 784 = 100 % | 0 |

---

## 3. Verifiche richieste

### 3.1 Percentuale di testo narrativo
Il corpus di training è composto quasi interamente da **testo narrativo /
esplicativo** (frasi del tipo *"Il design pattern Mediator aiuta a risolvere
problemi…"*, *"Il database relazionale organizza i dati…"*). Non è presente
alcuna struttura di turno Utente/Assistente: i marcatori `utente:` / `jarvis:`
compaiono solo **73 volte su 4,2M di token** (lo 0,0017 %), e in contesto
narrativo, non conversazionale.

### 3.2 Presenza di dialoghi
**Praticamente assente.** Validation e test contengono 0 dialoghi e solo 784
parole distinte ciascuno (frasi altamente ripetute), il che spiega il loro
`% UNK` nullo e la loro copertura del 100 %: il testo è ricavato da un pool
molto ristretto di frasi.

### 3.3 Distribuzione dei token
Il vocabolario è limitato a **3000 token** (modalità `whitespace`). Su
84 630 parole distinte presenti nel training, solo **2 998** (3,54 %) rientrano
nel vocabolario; il resto viene mappato a `<UNK>`.

### 3.4 Frequenza `<UNK>`
Sul training il tasso di `<UNK>` è del **5,727 %** (241 921 token su 4 224 043).
Validation e test hanno `<UNK>` nullo perché costruiti sulle stesse 784 frasi
gia presenti nel vocabolario.

### 3.5 Copertura del vocabolario
Estremamente bassa sul lessico complessivo (3,54 %). Il modello "conosce" solo
le 3000 parole più frequenti del corpus narrativo e non ha alcun lemma
conversazionale dedicato (saluti, identità, capacità, gestione errori…).

---

## 4. Impatto di `max_vocab_size` sul tasso `<UNK>`

Misurazione su `train.txt` con tokenizer addestrato solo su quel corpus:

| `max_vocab_size` | Vocabolario effettivo | `% UNK` |
|------------------|-----------------------|---------|
| 3000 | 3000 | **5,727 %** |
| 5000 | 5000 | 4,471 % |
| 8000 | 8000 | **3,504 %** |
| 10000 | 10000 | 3,086 % |
| 15000 | 15000 | 2,393 % |

Il passaggio da 3000 a 8000 riduce il `<UNK>` del **~39 % relativo**, confermando
che aumentare la copertura del vocabolario è una leva valida e misurabile.

---

## 5. Conclusioni

1. Il dataset attuale insegna quasi esclusivamente **continuazione di testo
   narrativo**, non comportamento da assistente.
2. I **dialoghi sono assenti** (73 marcatori su 4,2M di token, in contesto
   non conversazionale).
3. Il **tasso `<UNK>` del 5,7 %** e la **copertura del 3,5 %** indicano che il
   vocabolario è troppo ristretto per coprire il lessico del corpus.
4. Occorre introdurre un **dataset conversazionale proprietario** bilanciato
   (50 % assistente, 30 % conoscenza, 20 % ragionamento) e aumentare la
   copertura del tokenizer (vocabolario 8000, fit prioritario sui dialoghi)
   per ridurre `<UNK>` e insegnare la risposta da assistente.

Questo giustifica le Fasi 2–6 (creazione dataset dialogo, miglioramento
tokenizer, pipeline mista, test di comportamento, training di verifica).
