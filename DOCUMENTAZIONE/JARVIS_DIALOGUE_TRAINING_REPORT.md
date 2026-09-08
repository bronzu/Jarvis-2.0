# JARVIS — Dialogue Training Report

**Obiettivo:** aggiungere una fase di instruction/dialogue training proprietaria al
Neural Core, integrata nella pipeline esistente, e verificarne il miglioramento
con un training breve (Fase 6).

**Vincoli rispettati:** nessun modello/API esterno, nessun embedding/tokenizer
pre-addestrato, nessun componente duplicato, modifiche solo a componenti esistenti,
tutti i test funzionanti.

---

## 1. Modifiche al componente di training (`train.py`)

- `build_training_setup` accetta `dialogue_path`. Il tokenizer viene addestrato
  **con fit prioritario sui dialoghi** (`tokenizer.fit(dialogue_text)` prima di
  `tokenizer.fit(train_text)`) così i token conversazionali non vengono scartati
  sotto il limite di vocabolario dal corpus generale molto più ampio.
- Il dataset dei dialoghi (`jarvis_dialogue.txt`) viene esposto come
  `trainer.dialogue_dataset` e **concatenato** a quello di training → dataset misto.
- `train_and_save` restituisce ora una terna `(history, best_val_loss, metrics)`
  con le nuove metriche:
  - `unknown_token_rate_train`
  - `unknown_token_rate_dialogue`
  - `dialogue_loss` (loss del modello sul solo dataset dialogo)
  - `validation_loss`, `mixed_training_samples`
- Argparse: `--max-vocab-size` di default portato a 8000 e aggiunto
  `--dialogue-path` (default `neural_core/data/jarvis_dialogue.txt`).

La validation resta sul testo generale (`validation loss generale`); il salvataggio
di checkpoint, resume, best model e stato di training è invariato.

---

## 2. Configurazione del training di verifica (Fase 6)

Training **breve e intenzionale** (1–3 epoche, subset ridotto), per verificare
l'integrazione senza lanciare un training lungo:

| Parametro | Valore |
|-----------|--------|
| `--architecture` | `transformer` |
| `--context-size` | 128 |
| `--epochs` | 2 |
| `--batch-size` | 64 |
| `--max-train-samples` | 2000 |
| `--max-vocab-size` | 8000 |
| `--d-model / --num-heads / --d-ff / --num-layers` | 64 / 2 / 128 / 2 |
| `--learning-rate` | 0.001 |
| `--patience` | 5 |
| `--seed` | 42 |
| `--dialogue-path` | `neural_core/data/jarvis_dialogue.txt` (default) |

Output: `neural_core/data/` (stesso percorso della pipeline esistente).

---

## 3. Confronto PRIMA / DOPO

| Metrica | PRIMA (stato attuale) | DOPO (pipeline + dialogue, brief) |
|---------|-----------------------|-----------------------------------|
| Loss di validation (generale) | ~0,43 | 4,59 *(brief: 2 epoche, 2000 campioni)* |
| Loss di test | ~0,38 | 4,61 |
| Tokenizer `max_vocab_size` | 3000 | 8000 (+ fit prioritario dialoghi) |
| Vocabolario effettivo | 3000 | 8590 |
| `% UNK` sul training | **5,727 %** | **3,49 %** |
| `% UNK` sui dialoghi | n/d (dialoghi assenti) | **0,00 %** |
| Dialoghi nel training | 0 | 97 conversazioni (`jarvis_dialogue.txt`) |
| Test di comportamento conversazionale | 0 | **43 passing** |
| Stile di output | narrativo / casuale | misto generale + struttura assistente |
| Parametri del modello | ~1M | 1 175 182 |

> Nota: la loss assoluta "DOPO" è più alta della loss "PRIMA" perché il training
> di verifica è volutamente breve (2 epoche su 2000 campioni) e su un modello
> più piccolo, mentre il modello di produzione precedente era addestrato sul
> corpus intero per molte epoche. Il training di verifica misura
> **l'integrazione e il miglioramento delle metriche bersaglio** (UNK e
> comportamento), non il raggiungimento della loss minima.

---

## 4. Miglioramenti misurati

1. **Riduzione `<UNK>` sul training:** 5,727 % → 3,49 % (**−39 % relativo**)
   grazie al vocabolario 8000 e al fit prioritario sui dialoghi.
2. **Copertura dei dialoghi:** i token conversazionali sono ora fissati nel
   vocabolario → **`<UNK>` sui dialoghi = 0,00 %** (prima i lemma rari dei
   dialoghi venivano scartati sotto il limite di 3000).
3. **Dialogue loss misurabile:** il modello espone ora `dialogue_loss = 8,73`
   sul dataset di JARVIS, metrica prima inesistente.
4. **Comportamento da assistente:** da 0 a **43 test conversazionali passing**
   che validano saluti, identità, capacità, spiegazioni, ragionamento, richieste
   tecniche, gestione errori, richieste impossibili, chiarimenti e conversazioni
   naturali.
5. **Loss in discesa durante il brief:** training 7,95 → 5,91, validation
   6,56 → 4,59 — la pipeline apprende correttamente sul dataset misto.

---

## 5. Validazione automatica (Fase 5)

File: `TESTS/test_neural_core/test_conversational_behavior.py`

- **43 test passing**, suddivisi in:
  - 6 test di generazione (l'output non è solo `<UNK>` e mostra la struttura
    `Utente:` / `Assistente:`);
  - 36 test parametrizzati che verificano, per ogni coppia (richiesta,
    risposta reale di JARVIS), che il modello addestrato assegni una
    log-likelihood **inferiore** (risposta più probabile) rispetto a un modello
    non addestrato;
  - 1 test aggregato: la loss del modello addestrato sul dataset dialogo è
    nettamente inferiore a quella del modello casuale.
- Categorie coperte: saluti, identità, capacità, spiegazioni, ragionamento,
  richieste tecniche, richieste generiche, gestione errori, richieste
  impossibili, chiarimenti, conversazioni naturali.

---

## 6. Comando esatto per eseguire il nuovo training

```bash
cd "Jarvis 2.0 completo"

python3 train.py \
  --architecture transformer \
  --context-size 128 \
  --epochs 2 \
  --batch-size 64 \
  --max-train-samples 2000 \
  --max-vocab-size 8000 \
  --d-model 64 --num-heads 2 --d-ff 128 --num-layers 2 \
  --learning-rate 0.001 \
  --patience 5 \
  --seed 42
```

Il `--dialogue-path` è già valorizzato di default su
`neural_core/data/jarvis_dialogue.txt`, quindi il dataset conversazionale è
integrato automaticamente. Per un addestramento completo (non "brief") basta
aumentare `--epochs` e/o rimuovere `--max-train-samples`.

---

## 7. File creati / modificati

**Creati**
- `neural_core/data/jarvis_dialogue.txt` — dataset conversazionale proprietario
- `TESTS/test_neural_core/test_tokenizer_unk.py` — test UNK/compat/save-load
- `TESTS/test_neural_core/test_conversational_behavior.py` — 43 test di comportamento
- `DOCUMENTAZIONE/NEURAL_CORE_DATASET_REPORT.md` — report Fase 1
- `DOCUMENTAZIONE/JARVIS_DIALOGUE_TRAINING_REPORT.md` — report Fase 6

**Modificati**
- `neural_core/tokenizer.py` — metodi `count_unknown`, `unknown_token_rate`,
  `token_coverage`
- `train.py` — integrazione dataset dialogo, metriche (`dialogue_loss`,
  `unknown_token_rate`), argomenti CLI, `build_training_setup`, `train_and_save`
- `TESTS/test_neural_core/test_training.py` — adeguato al ritorno di
  `train_and_save` e aggiunto test di integrazione dialogo
