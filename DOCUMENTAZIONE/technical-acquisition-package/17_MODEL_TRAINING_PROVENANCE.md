# Model / Training Provenance

**Date:** 2026-09-08  
**Classification:** Due Diligence Only  
**Version:** 1.0

---

## 1. Executive Summary

This document provides a complete provenance record for the JARVIS 2.0 MicroTransformer model, from dataset through preprocessing, tokenization, training, checkpointing, to final model artifact. The training is **experimental / proof-of-architecture training** on a small corpus (33.4 MB / ~5M tokens), not production-scale language model training.

---

## 2. Model Architecture

| Aspect | Value | Source |
|---|---|---|
| **Architecture Name** | MicroTransformer | `neural_core/network.py`, `model_config.py` |
| **Architecture Version** | `jarvis2-microtransformer-v1` | `checkpoint_lineage.py:33` |
| **Model Class** | `MicroTransformer` | `model.pkl` metadata |
| **Backend** | Pure NumPy (CPU reference) | `neural_core/network.py` |
| **Optional GPU Backend** | PyTorch (MPS/CUDA) | `neural_core/torch_model.py` |

### 2.1 Hyperparameters (Production Model)

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 8,590 | From fitted tokenizer (includes special tokens) |
| `d_model` | 128 | Model dimension |
| `num_heads` | 4 | Attention heads |
| `d_ff` | 256 | Feed-forward dimension |
| `num_layers` | 4 | Transformer layers |
| `max_len` | 256 | Context window |
| `dropout` | 0.1 | Dropout rate |
| `dtype` | float64 | NumPy backend precision |
| **Total Parameters** | **2,737,806** | Verified from `model.pkl` (69 arrays) |

### 2.2 Architecture Invariants (from `model_config.py`)

- Decoder-only, causal self-attention, autoregressive next-token
- tanh-based GELU activation
- Manual LayerNorm with `eps=1e-6`
- Embedding kept SEPARATE from output projection (NO weight tying)
- Same positional encoding, dropout, Adam and CrossEntropyLoss semantics

---

## 3. Tokenizer

| Aspect | Value | Source |
|---|---|---|
| **Tokenizer Class** | `neural_core.tokenizer.Tokenizer` | `tokenizer.py` |
| **Vocabulary Size** | 8,590 tokens | `tokenizer.pkl` |
| **Max Vocab Size (config)** | 8,000 | Training config |
| **Pre-tokenization Mode** | `whitespace` | `tokenizer.py` (default) |
| **Alternative Mode** | `technical` | For Italian/code/math |
| **Special Tokens** | `<UNK>`, `<|endoftext|>` | `tokenizer.py` |
| **Serialization** | `tokenizer.pkl` (pickle) + `tokenizer.json` (JSON) | `train.py:348-349` |

### 3.1 Tokenizer Training

- Fitted on **combined text**: `train.txt` + `jarvis_dialogue.txt`
- Dialogue text fitted **first** (priority) to preserve conversational tokens
- Then general training text fitted
- This reduces `<UNK>` rate in dialogue responses

---

## 4. Dataset Pipeline

### 4.1 Data Flow

```
full_corpus.txt (33.4 MB, 30,360 lines)
    ↓ build_dataset.py (splitting, deduplication)
train.txt (4.95 MB) / validation.txt (617 KB) / test.txt (614 KB)
    ↓ tokenizer.fit() on train.txt + jarvis_dialogue.txt
tokenizer.pkl / tokenizer.json (8,590 vocab)
    ↓ tokenizer.encode()
train_ids / val_ids / test_ids (token sequences)
    ↓ Dataset(context_size=256, stride=128)
TrainingDataset / ValidationDataset / TestDataset
    ↓ Trainer.train_with_validation()
model.pkl (2.7M params, 21.9 MB)
```

### 4.2 Dataset Statistics (from `provenance_manifest.json`)

| Split | Size | Records | SHA256 |
|---|---|---|---|
| `train.txt` | 4.95 MB | 3,573 | `48fcf61be05468483950788fc0afe183` |
| `validation.txt` | 617 KB | 458 | `a6465db84bd420664c39102e25f4c3d6` |
| `test.txt` | 614 KB | 439 | `605dac564a70521b375a90e6e0fac7f2` |

**Deduplication (verified):**
- Exact duplicates: 0
- Normalized duplicates: 0
- Sentence duplicates: 0
- Near duplicates (MinHash/LSH, threshold 0.85): 0

---

## 5. Training Configuration

### 5.1 Training Script

**Entry Point:** `train.py`  
**Core Training Logic:** `neural_core/trainer.py` (`Trainer` class)

### 5.2 Default Training Config (from `train.py` defaults)

| Parameter | Value |
|---|---|
| `architecture` | `transformer` |
| `context_size` | 256 |
| `stride` | 128 (context_size // 2) |
| `max_vocab_size` | 8,000 |
| `tokenizer_mode` | `whitespace` |
| `d_model` | 128 |
| `num_heads` | 4 |
| `d_ff` | 256 |
| `num_layers` | 4 |
| `learning_rate` | 0.001 |
| `warmup_steps` | 0 |
| `schedule` | `constant` |
| `gradient_accumulation_steps` | 1 |
| `max_grad_norm` | None |
| `epochs` | 8 |
| `batch_size` | 4096 |
| `patience` | 4 |
| `seed` | 42 (fixed) |

### 5.3 Optimizer & Loss

| Component | Implementation | Source |
|---|---|---|
| **Optimizer** | Adam (custom NumPy) | `neural_core/optimizer.py` |
| **Learning Rate Scheduler** | Constant / Linear / Cosine | `neural_core/optimizer.py` |
| **Loss Function** | CrossEntropyLoss (custom) | `neural_core/loss.py` |
| **Gradient Clipping** | `max_grad_norm` (optional) | `trainer.py` |
| **Gradient Accumulation** | Supported | `trainer.py` |

---

## 6. Training Execution (Actual Run)

### 6.1 Training State (from `training_state.pkl`)

| Metric | Value |
|---|---|
| **Format** | `jarvis-neural-core-checkpoint` v1 |
| **Epochs Completed** | 10 |
| **Global Steps** | 2,580 |
| **Best Validation Loss** | 0.4271 |
| **Gradient Accumulation** | 1 |
| **Max Grad Norm** | None |

### 6.2 Training History (10 epochs)

| Epoch | Train Loss | Val Loss | Best |
|---|---|---|---|
| 1 | ~2.8 | ~1.2 | ✓ |
| 2 | ~1.5 | ~0.8 | ✓ |
| ... | ... | ... | |
| 10 | ~0.35 | ~0.43 | ✓ |

*Full history in `training_state.pkl` → `history` list (10 entries)*

---

## 7. Model Artifacts

### 7.1 Primary Artifacts

| File | Path | Format | Size | Content |
|---|---|---|---|---|
| **Production Model** | `neural_core/data/model.pkl` | pickle | 21.9 MB | Trained weights (69 arrays), config, model_class |
| **Training State** | `neural_core/data/training_state.pkl` | pickle | 59.5 MB | Full checkpoint for resume (optimizer, scheduler, history) |
| **Tokenizer (pickle)** | `neural_core/data/tokenizer.pkl` | pickle | 176 KB | Fitted Tokenizer object |
| **Tokenizer (JSON)** | `neural_core/data/tokenizer.json` | JSON | 176 KB | Portable vocabulary export |

### 7.2 Secondary Artifacts

| File | Path | Format | Size | Content |
|---|---|---|---|---|
| Alternative Config | `neural_core/data/best_model.json` | JSON | 53.4 MB | Alternative config (num_layers=2) |
| Baseline Model | `neural_core/data/baseline_model.pkl` | pickle | 2.7 MB | Baseline (730K params) |
| Baseline Manifest | `neural_core/data/baseline_manifest.json` | JSON | 927 B | SHA256, param count, config |

---

## 8. Checkpoint Lineage

The `checkpoint_lineage.py` module embeds complete provenance in every checkpoint:

```json
{
  "git_commit": null,
  "code_version": "0.2.0",
  "architecture_version": "jarvis2-microtransformer-v1",
  "model_class": "MicroTransformer",
  "model_config": { "vocab_size": 8590, "d_model": 128, ... },
  "tokenizer_identity": { "vocab_size": 8590, "mode": "whitespace", "class": "Tokenizer" },
  "dataset_identity": {
    "train.txt": { "size_bytes": 4952699, "sha256": "48fcf61b..." },
    "validation.txt": { "size_bytes": 639516, "sha256": "a6465db8..." },
    "test.txt": { "size_bytes": 616334, "sha256": "605dac56..." }
  },
  "parameter_count": 2737806,
  "seed": 42,
  "training_config": { "epochs": 8, "batch_size": 4096, ... },
  "python_version": "3.11.x",
  "numpy_version": "2.4.2",
  "platform": "macOS-...",
  "timestamp": "2026-XX-XXTXX:XX:XXZ"
}
```

**Key Properties:**
- `git_commit`: **null** (repository was not a git repo at training time)
- `code_version`: Parsed from `DOCUMENTAZIONE/versionamento/VERSIONE.md` → "0.2.0"
- All fields are **real values** from runtime; missing = `null` (never fabricated)

---

## 9. Reproducibility

| Aspect | Status | Evidence |
|---|---|---|
| **Fixed Seed** | ✅ | `seed=42` in config and checkpoint |
| **Config Saving** | ✅ | `training_config.json` in checkpoint dir |
| **Tokenizer Serialization** | ✅ | `tokenizer.pkl` + `tokenizer.json` |
| **Dataset SHA256** | ✅ | `provenance_manifest.json` per split |
| **Code Version** | ✅ | `0.2.0` from `VERSIONE.md` |
| **Environment Capture** | ✅ | Python, NumPy, platform in checkpoint |
| **Run A == Run B** | ✅ (tested) | `test_reproducibility.py` passes |

**Limitation:** No Git commit hash available (repository was not a git repo during training). The `code_version` (0.2.0) serves as the version identifier.

---

## 10. Evaluation / Benchmarks

### 10.1 Inference Benchmarks (from `FINAL_BENCHMARK.json`)

| Metric | Value | Platform |
|---|---|---|
| Startup time | 8.6 ms | Apple M1 Pro, macOS 14.5 |
| Model loading | 8.8 ms | Apple M1 Pro, macOS 14.5 |
| Inference throughput | 56,692 tok/s | Apple M1 Pro, MPS |
| Inference latency | 1.76 ms / 100 tokens | Apple M1 Pro, MPS |
| Model size | 21.9 MB | `model.pkl` file size |

### 10.2 GPU/MPS Benchmarks (from `audit_logs/gpu_bench.json`)

| Metric | Value |
|---|---|
| Inference latency (median) | 2.20 ms (with `torch.mps.synchronize()`) |
| Inference tokens/sec (median) | 58,252 |
| Batch=4 throughput | 59,228 tok/s |
| Batch=8 throughput | 103,070 tok/s |

**Note:** CPU (NumPy) and GPU (PyTorch) benchmarks are **not directly comparable** — different backends.

---

## 11. Verification Artifacts

| Test | File | Status |
|---|---|---|
| Gradient Check | `TESTS/test_neural_core/test_gradient_check.py` | ✅ PASS |
| GPU Parity | `TESTS/test_neural_core/test_gpu_parity.py` | ✅ PASS |
| Checkpoint Corruption Detection | `TESTS/test_neural_core/test_checkpoint_corruption.py` (7 tests) | ✅ PASS |
| Model Persistence | `TESTS/test_neural_core/test_model_persistence.py` | ✅ PASS |
| Reproducibility | `TESTS/test_reproducibility.py` | ✅ PASS |
| Dataset Leakage | `TESTS/test_neural_core/test_dataset_leakage.py` | ✅ PASS |

---

## 12. Critical Gaps for Acquisition

| Gap | Severity | Impact |
|---|---|---|
| **No Git commit hash in checkpoint** | HIGH | Cannot cryptographically link model to source code version |
| **Dataset provenance incomplete** | CRITICAL | `full_corpus.txt` sources unknown — legal risk |
| **Training scale experimental** | MEDIUM | 33 MB corpus, 2.7M params — not production LLM |
| **No production benchmark at scale** | MEDIUM | No concurrent QPS, sustained load, adversarial robustness data |

---

## 13. Continual Learning (Real-time Online Training)

### 13.1 Overview

Il modulo supporta l'apprendimento continuo in tempo reale (Continual Learning)
attraverso due tecniche complementari per prevenire il **Catastrophic Forgetting**:

1. **Elastic Weight Consolidation (EWC)** — penalizza la deviazione dei pesi più
   importanti per i task precedenti tramite una matrice di Fisher.
2. **Episodic Replay** — mescola una frazione (20%) di dati storici al batch corrente.

### 13.2 Equazioni Matematiche

**Fisher Matrix** (stimata su un sottoinsieme di `num_samples` campioni):

$$F_\theta = \frac{1}{N} \sum_{n=1}^{N} \left( \nabla_\theta \log p(x_n | \theta) \right)^2$$

dove $\nabla_\theta \log p$ è il gradiente del log-likelihood per il campione $n$.

**Penalità EWC** (aggiunta alla loss di base):

$$L_{\text{tot}} = L_{\text{base}} + \frac{\lambda}{2} \sum_{\theta} F_\theta \, (\theta - \theta_A)^2$$

dove $\theta_A$ sono i pesi ottimali salvati al checkpoint precedente e $\lambda$
(`ewc_lambda`, default 0.5) controlla la forza della penalità.

**Gradiente della penalità EWC** (somma ai gradienti del modello):

$$\frac{\partial L_{\text{EWC}}}{\partial \theta} = \lambda \cdot F_\theta \cdot (\theta - \theta_A)$$

### 13.3 Implementazione

La classe `Trainer` (`neural_core/trainer.py`) è stata estesa con:

| Campo / Metodo | Descrizione |
|---|---|
| `self.fisher_matrix` | Dizionario `param_name -> np.ndarray` con la matrice di Fisher. |
| `self.optimal_weights` | Dizionario `param_name -> np.ndarray` con i pesi ottimali $W_A$. |
| `self.ewc_lambda` | Float (default 0.5): forza della penalità EWC. |
| `compute_fisher_matrix(dataset, num_samples=100)` | Calcola la Fisher matrix e salva i pesi attuali in `optimal_weights`. |
| `_compute_ewc_loss()` | Calcola la penalità EWC corrente. |
| `_compute_ewc_gradients()` | Gradiente della penalità EWC. |
| `set_replay_buffer(buffer)` | Imposta il buffer di replay episodico. |
| `_sample_replay(...)` | Miscela l'80% di dati nuovi con il 20% dal buffer. |

### 13.4 Integrazione nel ciclo di auto-miglioramento

In `expansion_system/self_improvement/experimental_trainer.py`, il metodo
`run_training` calcola automaticamente la Fisher matrix sul dataset di training
 prima di avviare l'addestramento incrementale sul candidato, quando è disponibile
un checkpoint baseline.

In `expansion_system/self_improvement/improvement_cycle.py`, la fase
`_phase_checkpoint_validation` utilizza il metodo `_check_retention` per verificare
che la loss sui dati di baseline non subisca un degrado superiore al 2%,
usando `BenchmarkComparator` per misurare la regressione.

### 13.5 Tabella dei risultati dei test

| Test | File | Stato | Metrica chiave |
|---|---|---|---|
| EWC loss calculation | `TESTS/test_continual_learning.py` | ✅ PASS | Penalità = 0 con pesi identici; cresce quadraticamente |
| Fisher matrix computation | `TESTS/test_continual_learning.py` | ✅ PASS | Valori non negativi per ogni parametro |
| Catastrophic forgetting prevention | `TESTS/test_continual_learning.py` | ✅ PASS | Distanza dei pesi con EWC < senza EWC |
| LoRA injection & freezing (6 test) | `TESTS/test_continual_learning.py` | ✅ PASS | Gradient check analitico vs numerico; pesi congelati bit-identici |
| Memoria / latenza / ritenzione / apprendimento | `TESTS/test_continual_learning.py` | ⚠️ 4 criteri, 2 parziali | Vedi sezione 13.7.4: (a) e (b) raggiunti, (c) e (d) soddisfatti su accuratezza ma **non** su loss |

La riga "Catastrophic forgetting prevention" verifica la **distanza dei
pesi**, che è una misura tautologica e non dimostra ritenzione. La verifica
reale di ritenzione su loss/accuratezza è in `TestRetentionMeasured`
(sezione 13.7.4).

### 13.6 Impatto sul ciclo di auto-miglioramento

L'integrazione del Continual Learning permette al ciclo di auto-miglioramento di:

- Addestrare candidati su nuovi dati senza dimenticare i task precedenti.
- Verificare automaticamente che il degrado sui task di baseline sia entro il 2%.
- Mantenere una tracciabilità completa della Fisher matrix e dei pesi ottimali nei checkpoint.

### 13.7 Micro-EWC con LoRA (rank-adaptive Continual Learning)

Questa sezione documenta il rifattorizzamento del Continual Learning da
EWC full-parameter a **Micro-EWC con LoRA**.

I numeri di 13.7.4 provengono da due esecuzioni reali e distinte, entrambe
riproducibili:

- **criteri (a) e (b)** — `python3 experiments/phase4_continual_learning/validate.py --criterion a`,
  sulla configurazione di produzione (2.737.806 parametri);
- **criteri (c) e (d)** — `python3 experiments/phase4_continual_learning/validate.py --criterion cd`,
  su corpus reali del repository, con output grezzo in
  `experiments/phase4_continual_learning/results/validation_cd_raw.txt`.

I test in `TESTS/test_continual_learning.py` coprono la correttezza strutturale
del meccanismo su corpus sintetico; **non** sono la fonte dei criteri (c) e (d),
per il motivo esposto in 13.7.8.

#### 13.7.1 Decisione di design: LoRA per-task (opzione b)

Il protocollo imponeva di dichiarare una scelta fra due architetture:

- **(a) blocco LoRA condiviso**, riaddestrato a ogni task;
- **(b) blocco LoRA dedicato per task**.

**Scelta implementata: (b), un adapter LoRA dedicato per ciascun task.**

Motivazione: con un blocco condiviso, i parametri che codificano il task A
vengono riscritti dal task B nello stesso identico blocco. La Fisher misurata
su A non distingue piu' "cosa ho imparato in A" da "cosa sto imparando in B", e
il forgetting si comprime dentro un solo insieme di parametri. Con l'adapter
dedicato, l'adapter del task A resta letteralmente congelato dopo
l'addestramento di A: la perturbazione che l'addestramento di B puo' introdurre
e' limitata a un insieme di parametri separato e noto.

**Costo dichiarato:** la memoria cresce linearmente col numero di task
(2 matriche per layer per task). Con la configurazione di produzione e
`lora_rank=4` sono 6144 parametri addestrabili per task su 2.737.806 totali
(+0,22% per task, ~112 task prima di raddoppiare il modello). Il vincolo
dominante resta la memoria, non i parametri addestrabili.

#### 13.7.2 Architettura

`LoRALinear` (`neural_core/network.py`) implementa la correzione

$$\Delta = B \cdot A, \qquad \text{output} = W x + b + \frac{\alpha}{r}\, \Delta x$$

con $A \in \mathbb{R}^{r \times d_{ff}}$ inizializzata Kaiming uniform e
$B \in \mathbb{R}^{d_{model} \times r}$ inizializzata a **zero** (convenzione
LoRA standard: a inizializzazione la correzione e' nulla e il modello si
comporta esattamente come il modello base).

L'iniezione avviene **esclusivamente sui layer FFN finali**, come richiesto
dal protocollo: la correzione e' aggiunta all'output di `W2` di ogni
`FeedForward`, cioe' `output = h1 @ W2 + b2 + scaling * B @ A @ h1`.

Sono congelati: embedding, `W_q/W_k/W_v/W_o`, tutti i bias, `W1/b1/W2/b2`,
LayerNorm, `output_layer`, `output_bias`, e gli adapter dei task precedenti.
Il congelamento e' applicato **alla fonte** in `MicroTransformer.backward`:
i gradienti dei parametri congelati sono azzerati prima di uscire dal
modello, quindi un chiamante esterno che usi `backward()` direttamente li
riceve nulli. L'ottimizzatore riceve inoltre **solo** il sottoinsieme
addestrabile.

Il forward somma le correzioni di tutti gli adapter registrati:

$$\text{output} = \text{FFN}_{base}(x) + \sum_{k} \frac{\alpha}{r}\, B_k A_k \, h_1$$

Il modello deployed resta quindi un oggetto unico e non richiede di sapere a
quale task appartiene la query.

Lo stesso stack e' disponibile su `TorchMicroTransformer`
(`neural_core/torch_model.py`): i blocchi sono `ParameterDict` con chiavi
`lora_{k}_A` / `lora_{k}_B`, il forward somma tutti gli adapter come sopra, e
`requires_grad=True` e' concesso **solo** all'adapter attivo, cosi' gli adapter
dei task precedenti sono congelati anche nel percorso GPU. `set_parameters()`,
`get_parameters()`, `parameter_count()`, `trainable_parameter_count()` e
`trainable_param_indices()` restituiscono risultati coerenti con la reference
NumPy. La parita' e' verificata in `float64` su 1 e 3 task con errore massimo
**1,11e-15** (`TESTS/test_continual_learning.py::TestTorchModelLoRAAdapterStack`).
La semantica di `warm_start` e' allineata nei tre casi (`None` → warm start,
`True`, `False`) ed e' coperta da test; i due dettagli che lo rendono non
banale sono che `None` significa warm start (non cold) e che il warm start e'
inosservabile finche' `B` del task precedente e' zero, quindi il test imposta
`B != 0` prima di confrontare.

#### 13.7.3 Micro-EWC

La Fisher e la penalita' quadratica sono calcolate **esclusivamente** sulle
matrici A/B dell'adapter attivo, mai sui 2,7M di parametri totali. Le formule
sono quelle della sezione 13.2, ristrette allo spazio $\Theta_k$ dell'adapter
del task $k$.

**Ri-ancoraggio.** Dopo la registrazione di un nuovo task, la Fisher va
ricalcolata rispetto al nuovo adapter: la sua semantica e' "quanto il task A
e' sensibile a una perturbazione di *questo* adapter", che e' l'unica
perturbazione realmente addestrabile. `Trainer._ensure_ewc_anchor()` esegue
il ri-ancoraggio automaticamente quando il task attivo cambia. Senza di esso la
penalita' sarebbe **silenziosamente nulla** (i nomi della Fisher non
corrisponderebbero ai parametri addestrabili): questo difetto e' stato
osservato sperimentalmente durante lo sviluppo, con `ewc_lambda` da 0 a 1e5
produceva risultati identici bit per bit.

**Warm start vs cold start.** Sono supportati entrambi
(`add_task_adapter(warm_start=...)`). Il warm start (nuovo adapter inizializzato
con i pesi del precedente) replica nello spazio LoRA l'assunzione
$\theta_{k+1} = \theta_k$ dell'EWC full-parameter. Misure sperimentali sullo
scenario a due task mostrano pero' che il warm start e' nettamente peggiore:
l'adapter del task B deve prima *annullare* la codifica del task A prima di
poter imparare B. Il **cold start** (`warm_start=False`, $B=0$) preserva
esattamente il comportamento del task A all'istante iniziale e lascia all'adapter
nuovo uno spazio libero. La configurazione documentata e' cold start.

#### 13.7.4 Risultati misurati — quattro criteri

I numeri di questa tabella provengono dall'esecuzione del 27 settembre 2026 su
**corpus reali del repository**, non da frasi sintetiche:

- **Task A** — prosa tecnica italiana, 16.953 parole, da 4 file di
  `DOCUMENTAZIONE/` (`ARCHITETTURA.md`, `SPECIFICA_PROGETTO.md`,
  `ARCHITETTURA_2.md`, `DIARIO.md`), 1.059 campioni (795 train / 264 val);
- **Task B** — codice sorgente Python, 11.917 parole, da 8 file di
  `neural_core/`, 744 campioni (558 train / 186 val).

Vocabolario condiviso: 6.736 token (loss casuale = 8,8152). I due domini
condividono il vocabolario tecnico inglese, che e' il caso difficile: il task B
va appreso senza distruggere la prosa del task A. Il 25% dei campioni di ogni
task e' tenuto fuori dall'addestramento e usato per la valutazione, quindi la
ritenzione e' misurata su dati **mai visti**.

Riproduzione: `python3 experiments/phase4_continual_learning/validate.py`.
Output grezzo: `experiments/phase4_continual_learning/results/validation_cd_raw.txt`.

Configurazione: LoRA rank 16 / alpha 16, `ewc_lambda` 100, 40 epoch per task,
batch 32, Adam lr 0,02, seed 42, replay episodico 80/20,
`d_model` 128 / `d_ff` 256 / 2 layer / 4 head.

| Criterio | Soglia dichiarata | Misurato | Esito |
|---|---|---|---|
| (a) Riduzione memoria — strutture persistenti | > 90% | **99,7756%** (rank 4) / **99,1023%** (rank 16) | ✅ raggiunto |
| (a) Riduzione memoria — RSS di processo reale | > 90% (se inteso come RAM del processo) | **15,67%** (rank 4) / **15,16%** (rank 16), intervallo osservato 9,57–22,02% / 7,41–21,46% | ❌ **NON raggiunto** |
| (b) Latenza `compute_fisher_matrix` su CPU | near real-time | **289,5 ms** (50 campioni), 1,16x–1,31x | ✅ raggiunto (modesto) |
| (c) Ritenzione task A — **accuratezza**, EWC attivo | < 2% | **−61,18%** (0,112216 → 0,043561) | ❌ **NON raggiunto** |
| (c) Ritenzione task A — **loss**, EWC attivo | < 2% | **+4,90%** (7,886261 → 8,272436) | ❌ **NON raggiunto** |
| (c) Ritenzione task A — accuratezza, EWC **disattivato** | (confronto) | −45,99% (0,112216 → 0,060606) | riferimento |
| (c) Ritenzione task A — loss, EWC **disattivato** | (confronto) | +7,31% (7,886261 → 8,462465) | riferimento |
| (d) Apprendimento task B — **accuratezza** LoRA vs full-FT | entro 2 pp | **−17,7083 pp** (0,088038 vs 0,265121) | ❌ **NON raggiunto** |
| (d) Apprendimento task B — **loss** LoRA vs full-FT | entro 1,5x | 0,80x (8,405028 vs 10,503369) | ⚠️ valore entro soglia ma **fuorviante**, vedi il commento sotto |

**Esito: (c) e (d) non sono raggiunti.** Il criterio (a) e' raggiunto sulle
strutture persistenti (99,78%) ma **non** sul consumo reale di processo (~15%);
il criterio (b) e' raggiunto in modo marginale (1,16x–1,31x). Il criterio (d) ha
un numero di loss entro soglia, ma e' fuorviante e non va letto come un
raggiungimento: la metrica giusta, l'accuratezza, e' di 17,71 pp peggio. Lo
stesso codice, misurato sul corpus sintetico della sessione precedente,
riportava `+0,0000%` di degrado sull'accuratezza e `+0,0000 pp` di gap: quei
numeri erano un artefatto del corpus, non una proprieta' del meccanismo. Il
confronto fra le due misure e' riportato in 13.7.8.

**Dettaglio (a) — riduzione memoria**, configurazione di produzione reale
(`vocab 8590, d_model 128, d_ff 256, 4 layer, 2.737.806 parametri`),
50 campioni Fisher:

| | full-parameter | LoRA rank 4 | LoRA rank 16 |
|---|---|---|---|
| parametri totali | 2.737.806 | 2.743.950 | 2.762.382 |
| parametri addestrabili | 2.737.806 | 6.144 | 24.576 |
| entries Fisher | 2.737.806 | 6.144 | 24.576 |
| `fisher_matrix` + `optimal_weights` | 43.804.896 B (**41,7756 MB**) | 98.304 B (**0,0938 MB**) | 393.216 B (**0,3750 MB**) |
| **riduzione strutture persistenti** | — | **99,7756%** | **99,1023%** |
| picco `tracemalloc` durante il calcolo | 86,5338 MB | 67,9005 MB | 68,3282 MB |
| RSS di processo, delta mediano (7 esecuzioni) | 112,78 MB | 95,11 MB | 95,69 MB |
| **riduzione RSS di processo (mediana)** | — | **15,67%** | **15,16%** |
| **riduzione RSS di processo (intervallo osservato)** | — | 9,57% – 22,02% | 7,41% – 21,46% |

La distinzione fra le ultime righe e' il punto piu' importante di questo
criterio e va letta con attenzione:

- Il **99,78%** e' la riduzione delle **strutture persistenti**
  (`fisher_matrix` + `optimal_weights`), che sono l'unica memoria che il
  meccanismo Micro-EWC introduce e che viene riallocata a ogni task. Questo
  numero e' esatto e riproducibile: dipende da quanti parametri sono
  addestrabili, non da come sono allocati.
- Il **15,67%** e' la riduzione del **RSS di processo**, isolando una variante
  per processo pulito. E' il numero che un utente osserverebbe, ed e' molto
  piu' basso perche' il picco e' dominato dal working set di
  forward/backward, che le due varianti pagano allo stesso modo: il modello
  resta di 2,7M parametri in entrambi i casi, LoRA **aggiunge** parametri
  (2.737.806 → 2.743.950) e non ne toglie.
- Questa seconda misura e' **rumorosa** e va citata con l'intervallo, non con
  un singolo valore: su 7 esecuzioni la riduzione osservata va dal 9,57% al
  22,02% (rank 4). La dispersione dipende dal momento in cui il campionatore
  del sistema incontra il picco di allocazione e dal RSS di base del processo,
  che essendo una fotografia non e' identico fra esecuzioni. Il valore
  aggregato e' in `results/rss_aggregation.txt`.
- Il picco `tracemalloc` (86,5 → 67,9 MB) non distingue le varianti in modo
  utile: misura l'allocazione **transitiva** durante `compute_fisher_matrix`,
  dominata dalle attivazioni intermedie, non da cio' che resta memorizzato. Se
  la riduzione della RAM fosse 99,78%, qui si vedrebbe; non si vede, e il
  motivo e' quello descritto sopra.

**Conclusione onesta sul criterio (a):** il requisito ">90% di riduzione di
memoria" e' soddisfatto **solo** se si intende la memoria delle strutture
Fisher/optimal, che e' l'unica aggiunta dal meccanismo. Se si intende la RAM
del processo, la riduzione e' del ~15% e il requisito **non** e' soddisfatto.
Il protocollo non specificava quale delle due intendesse; il numero che va
citato dipende dalla lettura, e vanno citati entrambi.

**Dettaglio (b) — latenza**, 50 campioni, CPU, NumPy, **stessa configurazione di
produzione del dettaglio (a)** (vocab 8590, `d_model` 128, `d_ff` 256, 4 layer),
minimo di 3 ripetizioni dopo un warm-up:

| | full-parameter | Micro-EWC/LoRA |
|---|---|---|
| latency rank 4 | 378,80 ms | **289,53 ms** (**1,31x**) |
| latency rank 16 | 378,80 ms | **288,90 ms** (**1,31x**) |

Il speedup e' modesto e sostanzialmente indipendente dal rango perche'
`compute_fisher_matrix` esegue comunque un forward e un backward completi per
campione: la riduzione riguarda l'accumulo e la memoria delle strutture, non il
numero di passaggi di propagazione. Su 10 campioni la latenza e' 55,2 ms.

> Le latenze sono sensibili al carico della macchina: runs precedenti sulla stessa
> configurazione hanno dato 387,5 ms → 334,8 ms (rank 4) e 494,9 ms → 399,2 ms
> (rank 16). Il rapporto resta nell'intervallo **1,16x–1,31x**; i numeri nella
> tabella sono quelli dell'ultima esecuzione reale riportata sopra.

**Dettaglio (c) — ritenzione, EWC attivo vs disattivato.** Misurata su **loss e
accuratezza reali** del task A, non sulla distanza dei pesi (tautologica: un solo
adapter e' addestrabile per volta, quindi la distanza e' gia' un proxy
dell'update). Valutata sullo split di validazione mai visto in addestramento.

| | Task A loss | degrado | Task A accuracy | degrado | Task B accuracy |
|---|---|---|---|---|---|
| EWC **attivo** (`lambda` 100) | 7,886261 → 8,272436 | **+4,90%** | 0,112216 → 0,043561 | **−61,18%** | 0,088038 |
| EWC **disattivato** (`lambda` 0) | 7,886261 → 8,462465 | +7,31% | 0,112216 → 0,060606 | −45,99% | 0,152218 |

L'EWC **aiuta**: risparmia 2,41 punti percentuali di degrado sulla loss e 15,19
punti percentuali sull'accuratezza. Ma il degrado resta ben oltre la soglia del
2% su entrambe le metriche, quindi **il criterio (c) non e' raggiunto** neppure
nella configurazione migliore. Il dato e' brutale: l'accuratezza del task A
scende a 0,043561, contro 0,112216 prima del task B e contro 0,019264 di
accuratezza casuale riferita al vocabolario — il task A non e' dimenticato
totalmente, ma degradato a circa un terzo.

Da notare che EWC attivo e disattivato partono dallo **stesso** checkpoint
sul task A (loss 7,886261 in entrambi i casi): la differenza e' attribuibile
alla penalita' e non a una divergenza dell'addestramento sul primo task.

**Dettaglio (d) — apprendimento del task nuovo.** Stesso Task B, stesse 40
epoch, stesso batch 32, stesso seed, replay identico.

| | parametri addestrabili | Task B accuracy | Task B loss |
|---|---|---|---|
| LoRA rank 16 | 12.288 (0,608%) | 0,088038 | 8,405028 |
| Fine-tuning completo | 1.996.368 (100%) | 0,265121 | 10,503369 |
| **gap** | | **−17,7083 pp** | −2,098341 (0,80x) |

**Il gap in accuratezza e' di −17,71 punti percentuali: il criterio (d) non e'
raggiunto.** Il LoRA con 0,608% dei parametri addestrabili e' tre volte peggio
in accuratezza del fine-tuning completo. Questo e' esattamente il caso da
dichiarare apertamente: un sistema che non dimentica nulla perche' non impara
nulla di nuovo non risolve il problema, e qui il caso e' esattamente opposto —
il LoRA **dimentica molto** (criterio (c) fallito) **e impara poco** (criterio
(d) fallito). Il meccanismo non e' una soluzione del Continual Learning: e' un
compromesso che si e' rivelato peggiore del fine-tuning completo su entrambi i
fronti.

La loss va nella direzione opposta (8,405 contro 10,503, quindi 0,80x, entro la
soglia di 1,5x) e va letta con cautela: la cross-entropy e' dominata dai campioni
in cui il modello e' sbagliato in modo netto, mentre l'accuratezza conta solo
l'argmax. Il fine-tuning completo ha loss peggiore e accuratezza migliore
perche' e' **piu' sicuro di se**: quando sbaglia, sbaglia con confidenza. Riportare
"0,80x, criterio raggiunto" senza questa precisazione sarebbe fuorviante, e
suggerirebbe che il LoRA "allinei" il fine-tuning completo: non e' cosi'.

Per contesto, nel fine-tuning completo la ritenzione del task A e' migliore in
loss (8,780731 → 8,645600, −1,54%, cioe' migliorata) e in accuracy
(0,199100 → 0,181581, −8,80%), ma su una base di partenza piu' alta
(0,199100 contro 0,112216 del LoRA, perche' il fine-tuning completo impara
meglio anche il primo task). I due rami non sono confrontabili in valore
assoluto: sono confrontabili nel rapporto fra cio' che ciascuno perde e cio' che
guadagna.

**Diagnostica: il gap di (d) e' di capacita', non di ottimizzazione.** Un
learning rate 5 volte piu' alto e' la leva usuale quando un adapter LoRA
sottoaddestra, quindi il sospetto naturale era l'underfitting a epoche fissate.
Misurato sullo stesso protocollo A→B (`--criterion diag`, output in
`experiments/phase4_continual_learning/results/diagnostic_raw.txt`):

| | Task A accuracy | degrado | Task B accuracy | Task B loss |
|---|---|---|---|---|
| LoRA rank 16, lr 0,02 | 0,1122 → 0,0440 | −60,76% | 0,0887 | 8,4020 |
| LoRA rank 16, lr 0,10 | 0,1117 → 0,0464 | −58,47% | 0,0901 | 8,4292 |
| Fine-tuning completo, lr 0,02 | 0,1991 → 0,1816 | −8,80% | 0,2651 | 10,5034 |

Il learning rate 5x sposta l'accuratezza sul task B di **+0,14 punti
percentuali** (0,0887 → 0,0901) e la ritenzione di 2,29 punti percentuali:
nessuna delle due e' sufficiente a colmare il gap, che resta a
**−17,5067 pp**. Il limite non e' che il LoRA sia mal ottimizzato alle 40 epoch
fissate, e' che iniettare la LoRA **solo sulla proiezione finale della FFN**
(il vincolo del protocollo) non basta a sostituire la capacita' di
un fine-tuning completo. La ritenzione migliora anch'essa col learning rate
piu' alto, ma di poco e senza invertire la tendenza.

Il resto del limite e' il perimetro del protocollo: la LoRA e' iniettata solo
sui layer FFN finali (criterio 6 nei limiti), nessuna selezione per task di
`ewc_lambda` (criterio 5), nessun terzo task ne valutazione su distribuzioni
non viste (criterio 4). Se il meccanismo va fatto funzionare, la direzione da
prendere e' **estendere la superficie di LoRA** (W1, proiezioni di attenzione,
eventualmente embedding) e non tarare gli iperparametri di addestramento.

#### 13.7.5 Limiti noti e dichiarati

1. **Il criterio (c) non e' raggiunto.** Il degrado della ritenzione sul task A
   e' di +4,90% in loss e −61,18% in accuratezza con EWC attivo, contro una
   soglia dichiarata del 2%. L'EWC riduce il forgetting rispetto a EWC
   disattivato (−2,41 pp di loss, −15,19 pp di accuratezza) ma non lo annulla.
   Non e' un problema di misura: la valutazione e' su dati mai visti, la
   distanza dei pesi non e' usata come proxy e il confronto EWC on/off parte
   dallo stesso checkpoint.
2. **Il criterio (d) non e' raggiunto.** Il LoRA e' **17,71 punti percentuali
   peggio** del fine-tuning completo in accuratezza sul task nuovo, contro una
   soglia dichiarata di 2 pp. Unendo (c) e (d): il meccanismo dimentica molto
   *e* impara poco. Non e' una soluzione del Continual Learning ed e'
   dichiarato come tale. La diagnostica sul learning rate (5x) esclude
   l'underfitting come spiegazione: il gap residuo e' di **−17,5067 pp** ed e'
   attribuito al perimetro dell'iniezione LoRA, non agli iperparametri.
3. **La riduzione di memoria del 99,78% riguarda le strutture persistenti, non
   la RAM del processo.** Sul RSS di processo reale la riduzione e' del 15,67%
   (rank 4) / 15,16% (rank 16), con un intervallo osservato del 9,57-22,02% e
   7,41-21,46% su 7 esecuzioni: la misura e' rumorosa e va citata con l'intervallo.
   Il picco e' dominato dal working set di
   forward/backward, che LoRA non riduce: il modello resta di 2,7M parametri e
   LoRA ne aggiunge. Riportare il 99,78% come "riduzione di RAM" senza questa
   distinzione sarebbe semplicemente falso.
4. **Questi numeri provengono da due domini reali ma da un solo paio A/B e da
   un modello piccolo** (`d_model` 128, 2 layer, 6.736 token di vocabolario,
   795/558 campioni di addestramento). Sono una misura reale e riproducibile, ma
   non un benchmark Continual Learning: mancano terzo task, task multilingue,
   riordino dei task e valutazione su distribuzioni non viste in addestramento.
   L'accuratezza assoluta resta bassa (~0,09–0,27 su 6.736 classi): il confronto
   e' valido come differenza relativa fra rami allenati allo stesso modo, non
   come misura di capacita' assoluta del sistema.
5. **L'hyperparameter `ewc_lambda` e' fortemente sensibile al task.** Lo sweep
   misurato su {0, 10, 100, 1000, 10000} x {rank 4, 16} mostra che valori
   alti preservano il task A ma azzerano l'apprendimento del task B
   (accuracy B 0,0000 a lambda 1e5 senza replay). Non esiste un valore
   robustamente valido: serve una procedura di selezione per task. Il valore
   100 usato qui e' quello dichiarato in 13.7.3, non un valore ottimizzato: la
   griglia non e' stata ricalcolata sul corpus reale.
6. **La LoRA e' iniettata solo sui layer FFN finali**, come imposto dal
   protocollo, ed e' la causa identificata del fallimento del criterio (d): la
   diagnostica sul learning rule mostra che estendere la superficie di LoRA
   (W1, proiezioni di attenzione, eventualmente embedding) e' la direzione da
   prendere, mentre tarare `lr` ed `ewc_lambda` non e' sufficiente. Non e' stato
   valutato se l'iniezione estesa riduca anche l'interferenza misurata da (c).
7. **La riduzione di latenza e' limitata (1,16x–1,31x)** perche' il
   forward/backward per campione resta integrale. Un guadagno maggiore
   richiederebbe di campionare la Fisher senza propagazione completa, non
   implementato.
8. `TorchMicroTransformer` replica l'adapter stack per-task della reference
   NumPy: `add_task_adapter()` / `set_active_task()` / `num_lora_tasks`,
   forward cumulativo su tutti gli adapter e `requires_grad=True` **solo** sulle
   matrici A/B del task attivo (gli adapter dei task precedenti restano
   congelati anche in torch). La parita' NumPy/torch e' verificata su 1 e 3 task
   con errore massimo **1,11e-15** in `float64`
   (`TestTorchModelLoRAAdapterStack`). Resta pero' il limite che `Trainer` e'
   basato su NumPy e non pilota direttamente il modello torch: il percorso
   torch puo' eseguire forward e backward, ma il ciclo Micro-EWC completo
   (Fisher, penalita', `_apply_update`) non e' stato portato su torch.

#### 13.7.6 Bug preesistenti corretti in questa occasione

Durante la validazione sono emersi due difetti **preesistenti** nel replay
episodico, che rendevano il meccanismo silenziosamente inerte. Entrambi sono
corretti in `neural_core/trainer.py`:

1. Il parametro `replay_buffer` passato a `train()` / `train_step()` non
   veniva mai usato: `_sample_replay` leggeva esclusivamente
   `self.replay_buffer`, che resta `None` se non si chiama
   `set_replay_buffer()`. Il replay non avveniva.
2. Anche con il buffer impostato, i campioni 1-D venivano scartati dal
   controllo di dimensionalita contro batch 2-D, quindi **tutti** i campioni
   di replay venivano ignorati. Verificato: batch corrente 18 righe, mix
   prodotto 18 righe, 0 provenienti dal buffer. Dopo la correzione: 22 righe,
   4 dal buffer, frazione storica 18,2%.

La conseguenza misurabile e' che **tutte le misure di ritenzione precedenti
alla correzione del replay erano falsate**: riportavano degradi del 300-400%
anche con il 20% di dati storici in ogni batch.

#### 13.7.7 Bug corretti nel porting a torch (parita' NumPy/torch)

Nel chiudere la parita' fra `MicroTransformer` (NumPy) e
`TorchMicroTransformer` sono emersi due difetti, entrambi corretti:

1. **`warm_start=None` significava cose diverse nei due backend.**
   `FeedForward.add_adapter()` in `network.py` applica
   `if warm_start is None: warm_start = True`, cioe' il default e' il **warm
   start**. La versione torch, appena scritta, testava la verita' dell'argomento
   (`block if warm_start else None`), quindi `None` cadeva nel ramo falsy e
   produceva un **cold start**: due modelli con lo stesso codice di chiamata
   partivano da adapter diversi. Corretto in `torch_model.py` normalizzando
   `None -> True` prima di scegliere la sorgente della clonazione. Il difetto
   era invisibile finche' `B` del task precedente era a zero, perche' clonarne
   uno zero e' indistinguibile dall'inizializzarne uno zero: il test di
   regressione imposta `B != 0` prima di confrontare i due backend, altrimenti
   avrebbe continuato a passare senza verificare nulla.
2. **Riga morta in `network.py:828`.** In `FeedForward.add_adapter()` era
   presente `use_warm = (not self.lora_stack[-1].A.any() and False) if False
   else None`: un'espressione la cui condizione e' `False`, il cui risultato
   non e' mai letto e che valuta `self.lora_stack[-1].A.any()` solo per
   scartarne il valore. Rimosso. Non alterava il comportamento, ma rendeva
   ambigua la logica di scelta del warm start proprio nel punto in cui quel
   difetto era stato introdotto.

Entrambi i casi sono coperti da test di regressione in
`TESTS/test_continual_learning.py::TestTorchModelLoRAAdapterStack`
(parita' su 1 e 3 task, errore massimo 1,11e-15 in `float64`; semantica di
`warm_start` verificata per `None`, `True` e `False`).

#### 13.7.8 Perche' i numeri sono cambiati rispetto alla sessione precedente

La sessione precedente misurava (c) e (d) su un corpus sintetico di 5 frasi
ripetute 120 volte, con vocabolario ridotto. Riportava `+0,0000%` di degrado
sull'accuratezza del task A e `+0,0000 pp` di gap sul task nuovo, cioe' due
criteri "raggiunti". La misura odierna, su corpus reali, dà `−61,18%` e
`−17,7083 pp`: due criteri falliti.

La differenza non e' un peggioramento del codice — fra le due sessioni il
codice non e' cambiato nel meccanismo EWC, solo nella parita' torch — ma
**una differenza di difficolta' del compito**:

| | corpus sintetico (sessione precedente) | corpus reale (questa sessione) |
|---|---|---|
| Task A | 5 frasi italiane, distribuzione di bigrammi separata | 16.953 parole di prosa tecnica, 6.736 token di vocabolario |
| Task B | 5 frasi di codice, distribuzione separata | 11.917 parole di Python reale |
| campioni di addestramento | ~37 per task | 795 (A) / 558 (B) |
| accuratezza A dopo il task A | 0,989099 (satura) | 0,112216 |
| valutazione | sugli stessi dati di addestramento | su 25% di dati mai visti |
| (c) accuratezza task A | +0,0000% ✅ | −61,18% ❌ |
| (d) gap accuratezza | +0,0000 pp ✅ | −17,7083 pp ❌ |

Sul corpus sintetico i due domini erano separabili da una distribuzione di
bigrammi disgiunta e il modello li saturava, quindi l'accuratezza non aveva
dove degradare: il "perfect retention" misurato era una proprieta' del corpus,
non del meccanismo anti-forgetting. Sul corpus reale i domini condividono il
vocabolario tecnico e il modello non satura, quindi il forgetting emerge dove
esiste davvero.

La lezione operativa e' che **un criterio di ritenzione misurato su dati che il
modello ha gia' saturato non ha potere diagnostico**. Le misure su corpus
sintetico restano utili come test di fumo strutturale del meccanismo (e i test
in `TESTS/test_continual_learning.py` continuano a usarle cosi'), ma non
possono essere citate come evidenza del criterio (c) o (d). I numeri da usare
per la validazione sono quelli della tabella in 13.7.4, che e' eseguibile con un
comando e riproducibile.

---

## 14. Reconstruction Pipeline

To reproduce the model from scratch:

```bash
# 1. Prepare data (already done, files in neural_core/data/)
# 2. Train from scratch
python train.py --architecture transformer --context-size 256 --epochs 8 --seed 42

# 3. Verify reproducibility
python -m pytest TESTS/test_reproducibility.py -v

# 4. Load and test
python -c "
from neural_core.model_interface import NeuralModelInterface
interface = NeuralModelInterface()
print(interface.status())
print(interface.generate('Ciao', max_tokens=50))
"
```

---

*This document is part of the Technical Acquisition Package.*