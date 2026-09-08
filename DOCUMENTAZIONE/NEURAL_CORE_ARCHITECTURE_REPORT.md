# NEURAL CORE — REPORT DI ANALISI PRE-MODIFICA

Progetto: **JARVIS 2.0 completo**
Fase: **Evoluzione architetturale Neural Core (solo implementazione codice)**
Vincoli rispettati: nessun progetto nuovo, nessuna architettura parallela, nessun
componente AI esterno, nessun modello/tokenizer pre-addestrato, nessun training lungo.

---

## 1. STRUTTURA ATTUALE ANALIZZATA

### 1.1 File del Neural Core

| File | Contenuto | Ruolo attuale |
|---|---|---|
| `neural_core/model.py` | `gelu`, `gelu_derivative`, `softmax`, `SelfAttention`, `Decoder`, `TextGenerator`, `Neuron` | Primitive matematiche + attenzione single-head (solo forward) + sampling + neurone legacy con checkpoint |
| `neural_core/network.py` | `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `PositionalEncoding`, `MicroTransformer` | Transformer proprietario **solo inference** (nessun backward) |
| `neural_core/trainer.py` | `NextTokenModel`, `Trainer` | Modello bigram (1 token → 1 token) + loop di training |
| `neural_core/tokenizer.py` | `Tokenizer` | Tokenizer word-level su `str.split()` |
| `neural_core/dataset.py` | `Dataset` | Coppie `(token[i], token[i+1])` + split train/val/test |
| `neural_core/embeddings.py` | `Embeddings` | Tabella embedding pura Python, **inizializzata a zero** |
| `neural_core/loss.py` | `MSELoss`, `CrossEntropyLoss` | Loss 1D/2D |
| `neural_core/optimizer.py` | `SGD`, `Adam` | Ottimizzatori proprietari |
| `neural_core/checkpoint.py` | `Checkpoint` | Serializzazione JSON |
| `neural_core/evaluator.py` | `ModelEvaluator` | MAE |
| `neural_core/model_interface.py` | `NeuralModelInterface` | Interfaccia unica verso i sistemi JARVIS |
| `neural_core/llm_client.py` | `LLMClient` | **Deprecato**, default `internal` |
| `neural_core/providers/` | `ExternalLLMProvider` | Opzionale, disabilitato di default |
| `neural_core/build_dataset.py` | Costruzione corpus | Non toccato (divieto di creare nuovi dati) |
| `train.py` | `train_and_save()` | Entry point training |

### 1.2 Dati presenti in `neural_core/data/`

| File | Dimensione | Note |
|---|---|---|
| `full_corpus.txt` | 33 MB | Corpus completo |
| `train.txt` | 27 MB | Split di training |
| `validation.txt` | 2.8 MB | Split di validazione |
| `test.txt` | 3.0 MB | Split di test |
| `cleaned_literature.txt` | 5.7 MB | Sorgente pulita |
| `tokenizer.pkl` | 58 KB | Tokenizer serializzato (pickle) |
| `model.pkl` | 2.3 MB | Pesi `NextTokenModel` (`embeddings`, `W1`, `b1`, `W2`, `b2`) |
| `best_model.json` | 8.4 MB | Best checkpoint JSON `NextTokenModel` |

### 1.3 Pipeline attuale

```
train.py
  └─ Tokenizer(max_vocab_size)          # word-level, split()
     └─ Dataset.from_files()            # coppie bigram
        └─ NextTokenModel               # embedding → GELU → logits
           └─ Trainer.train_with_validation()
              ├─ Adam
              ├─ CrossEntropyLoss
              ├─ early stopping (patience)
              └─ best_model.json / model.pkl / tokenizer.pkl

jarvis.py
  └─ ExecutiveRouter
     ├─ _load_tokenizer(tokenizer.pkl)
     ├─ _load_model(model.pkl) → NextTokenModel
     └─ NeuralModelInterface(model, tokenizer)
        └─ CognitiveCoordinator → generate()
```

### 1.4 Stato test (baseline, prima delle modifiche)

```
293 passed, 1 skipped in 338.48s
```

Lo skip è in `TESTS/test_e2e_jarvis.py::test_neural_model_loads_and_generates`.

---

## 2. VERIFICHE RICHIESTE

### 2.1 Modello attuale

Sono presenti **due** modelli, non allineati:

1. **`NextTokenModel`** (`trainer.py`) — *addestrabile*, ma è di fatto un modello
   **bigram**: riceve un solo token e predice il successivo.
   `forward(token_ids) -> (N, vocab)`; nessun contesto, nessuna attenzione.
   È il modello effettivamente caricato dal `Router` in produzione.
2. **`MicroTransformer`** (`network.py`) — architettura Transformer corretta
   (embedding, positional encoding, multi-head attention, FFN, LayerNorm,
   residui, causal mask) ma **solo forward**: nessun `backward()`,
   nessun `get_parameters()`/`set_parameters()`. **Non addestrabile.**

### 2.2 Checkpoint

- `Trainer.save_weights()` scrive `embeddings, W1, b1, W2, b2, vocab_size, embedding_dim, hidden_dim, dropout`.
- `Trainer._save_best_checkpoint()` scrive lo stesso set in JSON.
- **Nessun checkpoint contiene lo stato dell'ottimizzatore, l'epoca o lo step** →
  *resume training impossibile*.
- `Checkpoint` (JSON) è usato solo dal `Neuron` legacy.

### 2.3 Compatibilità inference

`NeuralModelInterface.generate()` gestisce già logits 2D e 3D, quindi è
compatibile con entrambi i modelli. **Però**:

- non applica alcun **troncamento al context window**: il loop autoregressivo
  passa al modello la sequenza completa e cresce senza limiti. Con
  `MicroTransformer` questo produce `IndexError` sul positional encoding oltre
  `max_len`;
- non usa `<|endoftext|>` come stop token di default.

### 2.4 Pipeline training

Punti mancanti rispetto agli obiettivi:

- nessun **gradient accumulation**;
- nessuno **scheduler** di learning rate, nessun **warmup**;
- nessun **resume** (manca lo stato di Adam `m`, `v`, `t`);
- il best model è tenuto solo in memoria + JSON parziale;
- `Dataset` produce solo coppie a 1 token → impossibile addestrare su sequenze.

---

## 3. DIFETTI CRITICI INDIVIDUATI

| # | Difetto | File | Impatto |
|---|---|---|---|
| D1 | `MicroTransformer` non ha `backward()` | `network.py` | Il Transformer non può essere addestrato |
| D2 | `Embeddings` inizializzata **a zero** | `embeddings.py` | Con `MicroTransformer` tutti i token producono lo stesso vettore → logits identici → generazione degenere |
| D3 | `MicroTransformer` accetta solo `batch_size == 1` | `network.py` | Nessun mini-batch → training non praticabile |
| D4 | `_causal_mask` costruita con doppio loop Python | `network.py` | O(n²) in Python: con 512 token = 262 144 iterazioni per forward |
| D5 | Contesto = 1 token | `dataset.py`, `trainer.py` | Nessun apprendimento di dipendenze a lungo raggio |
| D6 | `CrossEntropyLoss` non gestisce logits 3D `(B,T,V)` | `loss.py` | Loss inutilizzabile con sequenze |
| D7 | Tokenizer su `str.split()` puro | `tokenizer.py` | `"funzione(x)"`, `"3.14"`, `"dell'acqua"`, `"a=b"` diventano token unici e irripetibili → vocabolario esploso e generalizzazione nulla su codice/matematica |
| D8 | Tokenizer senza `save`/`load` propri | `tokenizer.py` | Persistenza affidata al pickle esterno del `Trainer` |
| D9 | `Router._load_model()` legge `params["output_weights"]` / `params["output_bias"]` — chiavi **inesistenti** | `executive_system/router.py` | `except` silenzioso → **il modello addestrato viene scartato ad ogni avvio** e sostituito da uno random |
| D10 | `NeuralModelInterface` non tronca al context window | `model_interface.py` | Crash / degrado in generazione lunga |
| D11 | Nessun clipping del gradiente | `trainer.py` | Instabilità su Transformer |
| D12 | `test_neural_model_loads_and_generates` usa un path errato e chiavi errate | `TESTS/test_e2e_jarvis.py` | Test permanentemente skippato (falso verde) |

---

## 4. PIANO DI EVOLUZIONE (riuso, nessuna duplicazione)

**Regola applicata**: ogni file esistente viene *modificato ed esteso*.
Nessun `*_v2`, `*_new`, nessuna cartella duplicata, nessun nuovo modulo AI.

| Fase | File esistenti modificati | Intervento |
|---|---|---|
| 2 | `neural_core/network.py` | `MicroTransformer` diventa Transformer autoregressivo **addestrabile**: backward completo di `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, embedding, proiezione di output; batch N; init scalato; mask vettorizzata e cache-ata |
| 2 | `neural_core/model.py` | `SelfAttention` con causal masking e backward; `softmax` con `axis` |
| 2 | `neural_core/embeddings.py` | storage NumPy, init casuale opzionale (default zero = retrocompatibile), `forward_matrix`, `backward` |
| 2 | `neural_core/loss.py` | `CrossEntropyLoss` su `(B,T,V)` con `ignore_index` |
| 3 | `neural_core/network.py` | `max_len` = context window configurabile (128 / 256 / 512), default 256 |
| 3 | `neural_core/dataset.py` | `context_size` + `stride`: finestre `(T,) → (T,)` shiftate; API legacy invariata |
| 3 | `neural_core/model_interface.py` | troncamento a `max_len`, stop token `<|endoftext|>`, `max_context` |
| 4 | `neural_core/tokenizer.py` | modalità `technical` (italiano moderno/tecnico, codice, matematica, scienza), `save`/`load` JSON, `__setstate__` per compatibilità pickle |
| 5 | `neural_core/optimizer.py` | `Adam` con `state_dict`/`load_state_dict`, `weight_decay`, `set_learning_rate`; nuovo `LearningRateScheduler` (warmup + cosine/linear/constant) |
| 5 | `neural_core/trainer.py` | mini-batch, gradient accumulation, scheduler+warmup, grad clipping, `save_checkpoint`/`load_checkpoint` con stato completo, resume, best model |
| 6 | `executive_system/router.py` | `_load_model` corretto e compatibile con `NextTokenModel` **e** `MicroTransformer` |
| 6 | `train.py` | selezione architettura (`transformer` / `next_token`), context window, warmup, accumulation, resume — **senza avviare training lungo** |
| 7 | `TESTS/test_neural_core/*`, `TESTS/test_e2e_jarvis.py` | test reali su forward/backward/mask/dimensioni, tokenizer, trainer, checkpoint, resume, integrazione |

### Compatibilità garantita

- `Dataset(tokens)` senza `context_size` → comportamento identico a prima.
- `Tokenizer()` senza `mode` → comportamento identico a prima (vocabolario
  `{<UNK>:0, <|endoftext|>:1}`, `split()`, `decode` con spazi).
- `Embeddings(vocab, dim)` senza `init_scale` → matrice di zeri come prima.
- `NextTokenModel` e `Neuron` **restano** e continuano a funzionare.
- `NeuralModelInterface`, `CognitiveCoordinator`, `ExecutiveRouter`,
  `Memory`, `Tool System`, `Executive System` non cambiano contratto.

### Protocollo modello unificato (nessuna interfaccia nuova)

Entrambi i modelli espongono lo stesso protocollo, già usato dal `Trainer`:

```
forward(token_ids, training=False) -> logits
backward(targets)                  -> list[np.ndarray]
get_parameters()                   -> list[np.ndarray]
set_parameters(list[np.ndarray])   -> None
```

---

## 5. COSA NON VIENE FATTO IN QUESTA FASE

- Nessun training completo o lungo.
- Nessuna generazione/modifica di dataset o corpus.
- Nessun modello, embedding o tokenizer pre-addestrato importato.
- Nessuna dipendenza da GPT / Llama / Mistral / Gemma / HuggingFace / API AI.
