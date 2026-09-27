# Neural Core — JARVIS 2.0

Modulo di addestramento e inferenza del Neural Core di JARVIS 2.0.

Supporta due architetture proprietarie:

- `NextTokenModel`: modello a contesto singolo, leggero.
- `MicroTransformer`: Transformer autoregressivo con context window configurabile.

Entrambi rispettano lo stesso protocollo:

```
forward(token_ids, training=False) -> logits
backward(targets)                  -> list[np.ndarray]
get_parameters()                   -> list[np.ndarray]
set_parameters(list[np.ndarray])   -> None
config()                           -> dict
```

---

## Continual Learning & Elastic Weight Consolidation (EWC)

### Panorama

Il modulo supporta l'apprendimento continuo in tempo reale (Continual Learning)
attraverso due tecniche complementari:

1. **Elastic Weight Consolidation (EWC)** — previene il *Catastrophic Forcing*
   penalizzando la deviazione dei pesi più importanti per i task precedenti.
2. **Episodic Replay** — mescola una frazione (20%) di dati storici al batch
   corrente, riprendendo l'addestramento sui vecchi esempi.

### Modifiche al `Trainer`

La classe `Trainer` (`neural_core/trainer.py`) è stata estesa con i seguenti campi
e metodi:

| Campo / Metodo | Descrizione |
|---|---|
| `self.fisher_matrix` | Dizionario `param_name -> np.ndarray` con la matrice di Fisher stimata. |
| `self.optimal_weights` | Dizionario `param_name -> np.ndarray` con i pesi ottimali $W_A$ salvati al checkpoint precedente. |
| `self.ewc_lambda` | Float (default `0.5`): forza della penalità EWC. |
| `compute_fisher_matrix(dataset, num_samples=100)` | Calcola la Fisher matrix accumulando il quadrato dei gradienti del log-likelihood su un sottoinsieme del dataset. Salva i pesi attuali in `self.optimal_weights`. |
| `_compute_ewc_loss()` | Calcola $\sum F \odot (W - W_A)^2$ solo quando `fisher_matrix` e `optimal_weights` sono valorizzati. |
| `_compute_ewc_gradients()` | Gradiente della penalità EWC: $\lambda \cdot F \cdot (W - W_A)$. |
| `set_replay_buffer(buffer)` | Imposta il buffer di replay episodico (lista di tuple `(inputs, targets)`). |
| `_sample_replay(...)` | Miscela l'80% di dati nuovi con il 20% di campioni dal buffer. |

### Integrazione nel ciclo di training

Nel metodo `_run_epoch` (e di conseguenza `train` e `train_with_validation`):

1. Per ogni batch, se `self.fisher_matrix` e `self.optimal_weights` sono valorizzati,
   la loss totale diventa:
   $$L_{\text{tot}} = L_{\text{base}} + \frac{\lambda}{2} \sum_{\theta} F_\theta \, (\theta - \theta_A)^2$$
2. Il gradiente della penalità EWC viene sommato ai gradienti del modello prima
   dell'update dell'ottimizzatore.
3. Se un `replay_buffer` è stato impostato, i batch vengono miscelati con un
   rapporto del 20% di dati dal buffer.

### Checkpoint

`state_dict()` e `load_checkpoint()` sono stati estesi per salvare e ripristinare
anche `fisher_matrix`, `optimal_weights`, `ewc_lambda` e `replay_buffer`, permettendo
il resume di sessioni di continual learning.

### Esempio d'uso

```python
from neural_core.trainer import Trainer

trainer = Trainer(model, tokenizer, loss, optimizer)

# Addestra su Task A
trainer.train(inputs_a, targets_a, epochs=5)

# Calcola la Fisher matrix e salva i pesi ottimali
trainer.compute_fisher_matrix(dataset_a, num_samples=100)

# Imposta il replay buffer con campioni di Task A
trainer.set_replay_buffer([(inputs_a[i], targets_a[i]) for i in range(100)])

# Addestra su Task B con EWC e replay attivi
trainer.ewc_lambda = 0.5
trainer.train(inputs_b, targets_b, epochs=5)
```

### Test

I test dedicati sono in `TESTS/test_continual_learning.py`:

- `test_ewc_loss_calculation` — verifica che la penalità EWC sia zero con pesi
  identici e cresca quadraticamente con la deviazione.
- `test_fisher_matrix_computation` — verifica che la Fisher matrix contenga
  valori non negativi per ogni parametro del modello.
- `test_catastrophic_forgetting_prevention` — verifica che l'EWC preservi i pesi
  di Task A dopo l'addestramento su Task B.

---

## Struttura dei file

| File | Descrizione |
|---|---|
| `trainer.py` | Trainer con EWC, replay, checkpoint, scheduler. |
| `model.py` | `NextTokenModel` (embedding -> GELU -> logits). |
| `network.py` | `MicroTransformer` decoder-only autoregressivo. |
| `dataset.py` | Dataset con modalità token-singolo e contesto-window. |
| `tokenizer.py` | Tokenizer interno con vocabolario configurabile. |
| `loss.py` | `CrossEntropyLoss` proprietaria. |
| `optimizer.py` | `Adam` con decoupled weight decay e resume. |
