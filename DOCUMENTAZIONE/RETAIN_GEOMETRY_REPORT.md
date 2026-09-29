# Report — Protezione per-frase vs protezione aggregata: geometria al primo ordine

Esperimento pre-registrato in `experiments/mission_retain_geometry/PREREGISTRATION.md`.
Definizioni, ipotesi e soglie non modificate.

## 1. Configurazione

| Parametro | Valore |
|---|---|
| Epochs | 400 |
| LR | 0.01 |
| Batch size | 32 |
| Context size | 64 |
| Stride | 32 |
| d_model | 64 |
| num_heads | 4 |
| d_ff | 128 |
| num_layers | 2 |
| LoRA rank | 4 / 0 (full) |
| LoRA alpha | 16.0 |
| Seeds | 42, 43, 44 |
| rcond (primario) | 1e-6 |
| rcond (sensibilità) | 1e-3, 1e-9 |
| ALPHAS | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 |
| FISHER_SAMPLES | 100 |
| Checkpoint fractions | 0.25, 0.5, 1.0 |

## 2. Stati primari (checkpoint 100%, R_prot completo)

| # | Tipo | Seed | d | k_full | Delta_f | rho(k_full) | M(k_full) | M(agg1) | s*(k) | Errore FO | Stato |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lora | 42 | 1536 | 24 | 1.2867 (applicabile) | 0.7475 | 0.1000 | 0.1413 | 0.6465 | 0.4731 | H1=FALSO H2=VERO H3=FALSO |
| 2 | lora | 43 | 1536 | 24 | 1.0920 (applicabile) | 0.7721 | 0.0935 | 0.0999 | 0.6574 | 0.4895 | H1=FALSO H2=VERO H3=FALSO |
| 3 | lora | 44 | 1536 | 24 | 1.2006 (applicabile) | 0.8676 | 0.0702 | 0.0935 | 0.6674 | 0.2403 | H1=FALSO H2=VERO H3=VERO |
| 4 | full | 42 | 161113 | 24 | 4.8142 (applicabile) | 0.9873 | 6.1507 | 6.7438 | 7.6814 | 2.9205 | H1=FALSO H2=VERO H3=FALSO |
| 5 | full | 43 | 161113 | 24 | 4.1696 (applicabile) | 0.9932 | 2.0451 | 2.2652 | 7.6894 | 6.7432 | H1=FALSO H2=VERO H3=FALSO |
| 6 | full | 44 | 161113 | 24 | 4.8611 (applicabile) | 0.9820 | 4.0111 | 4.0750 | 7.6772 | 5.4564 | H1=FALSO H2=VERO H3=FALSO |

## 3. Esito delle ipotesi (6 stati primari)

| Ipotesi | Soglia | Passa | Dato censurato | Non applicabile | Esito |
|---|---|---|---|---|---|
| H1: M(k_full) <= M(agg1)/3 | >=5/6 | 0/6 | 0 | 0 | FALSO |
| H2: rho(k_full) >= 0.5 | >=5/6 | 6/6 | 0 | 0 | VERO |
| H3: errore FO <= 25% a s*(k_full) | >=5/6 | 1/6 | 0 | 0 | FALSO |

## 4. Conteggi dettagliati (6 stati primari)

| # | Tipo | Seed | M(k) frasi oltre 5% | M(agg1) frasi oltre 5% | Canary oltre 5% | Retain oltre 5% |
|---|---|---|---|---|---|---|
| 1 | lora | 42 | 5/12 | 5/12 | 2/6 | 6/24 |
| 2 | lora | 43 | 2/12 | 9/12 | 2/6 | 2/24 |
| 3 | lora | 44 | 3/12 | 5/12 | 5/6 | 1/24 |
| 4 | full | 42 | 12/12 | 12/12 | 6/6 | 24/24 |
| 5 | full | 43 | 12/12 | 12/12 | 5/6 | 24/24 |
| 6 | full | 44 | 12/12 | 12/12 | 6/6 | 22/24 |

## 5. Stati censurati / non applicabili

Nessuno stato censurato o non applicabile.

## 6. Controlli descrittivi

### 6.1 rho_rand(k) = sqrt((d - k) / d)

| # | Tipo | Seed | d | k=1 | k=2 | k=4 | k=8 | k=16 | k=24 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lora | 42 | 1536 | 0.9997 | 0.9993 | 0.9987 | 0.9974 | 0.9948 | 0.9922 |
| 2 | lora | 43 | 1536 | 0.9997 | 0.9993 | 0.9987 | 0.9974 | 0.9948 | 0.9922 |
| 3 | lora | 44 | 1536 | 0.9997 | 0.9993 | 0.9987 | 0.9974 | 0.9948 | 0.9922 |
| 4 | full | 42 | 161113 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 |
| 5 | full | 43 | 161113 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 |
| 6 | full | 44 | 161113 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 |

### 6.2 rho(k) con E_ood al posto di X_forget (controllo)

| # | Tipo | Seed | k_full | rho(k, E_ood) | rho(k, X_forget) |
|---|---|---|---|---|---|
| 1 | lora | 42 | 24 | 0.7912 | 0.7475 |
| 2 | lora | 43 | 24 | 0.8468 | 0.7721 |
| 3 | lora | 44 | 24 | 0.8053 | 0.8676 |
| 4 | full | 42 | 24 | 0.9970 | 0.9873 |
| 5 | full | 43 | 24 | 0.9943 | 0.9932 |
| 6 | full | 44 | 24 | 0.9924 | 0.9820 |

### 6.3 Sensibilità rcond (k_full)

| # | Tipo | Seed | rcond=1e-3 | rcond=1e-6 (primario) | rcond=1e-9 |
|---|---|---|---|---|---|
| 1 | lora | 42 | 24 | 24 | 24 |
| 2 | lora | 43 | 24 | 24 | 24 |
| 3 | lora | 44 | 24 | 24 | 24 |
| 4 | full | 42 | 24 | 24 | 24 |
| 5 | full | 43 | 24 | 24 | 24 |
| 6 | full | 44 | 24 | 24 | 24 |

### 6.4 Verifica Missione 1: u = F^-1 * g_r (singola direzione)

```
confirmed: True
formula: u = F^-1 g_r
description: SelectiveUnlearner.project_in_fisher_metric() e project_metric_orthogonal() projeta il gradiente di forget lungo UNA sola direzione u = F^-1 * g_r, dove g_r e' il gradiente aggregato (trainable_gradients_on_texts). Non usa un sottospazio multi-vettoriale.
source_file: /Users/riccardobronzuoli/Documents/Jarvis 2.0 completo/.kilo/worktrees/protezione-per-frase-vs-protezione-aggregata-geom/neural_core/unlearning.py
method: project_in_fisher_metric
```

## 7. Training metadata

| # | Tipo | Seed | Loss iniziale | Loss finale | Epochs |
|---|---|---|---|---|---|
| 1 | lora | 42 | 7.0411 | 5.3616 | 400 |
| 2 | lora | 43 | 7.1151 | 5.3923 | 400 |
| 3 | lora | 44 | 6.9940 | 5.3848 | 400 |
| 4 | full | 42 | 6.7089 | 0.0020 | 400 |
| 5 | full | 43 | 6.8297 | 0.0015 | 400 |
| 6 | full | 44 | 6.6743 | 0.0017 | 400 |

## 8. Deviazioni

Modifiche a `geometry.py` successive al primo run sperimentale (prima del re-run):

1. **save_checkpoint** (linea ~494): ora salva `{**MODEL_CONFIG, 'lora_rank': lora_rank}` invece di `MODEL_CONFIG`.
   **Impatto**: Sì. I checkpoint full-parameter ora ricostruiscono correttamente il modello senza LoRA adapter.
   Prima della correzione, `load_checkpoint` falliva con `ValueError: MicroTransformer richiede 41 parametri`.
2. **load_checkpoint** (linea ~515): aggiunto ritorno della `train_history`. Ora restituisce 5 valori.
   **Impatto**: Sì — `ensure_checkpoint` e `compute_geometry_for_state` aggiornati per unpack a 5 elementi.
3. **ensure_checkpoint** (linea ~591): salva checkpoint a tutte le frazioni (25%, 50%, 100%), non solo 100%.
   **Impatto**: Sì — i checkpoint a 25%/50% ora vengono persistiti e riutilizzati.
4. **compute_geometry_for_state** (linea ~743): `train_loss_initial`, `train_loss_final`, `history` ora popolati da checkpoint.
   **Impatto**: Sì — i metadati di training nei JSON sono ora popolati invece di essere `None`.

**Re-run post-correzione**: identico per tutti i valori geometrici (rho, M, s*, errore FO, Delta_f).
Solo i campi di tempo (`fisher.seconds`, `compute_seconds`, `train_seconds`) differiscono (non deterministici).

## 9. Limiti

- I risultati sono a livello di un singolo modello micro-trasformers (d_model=64, 2 layer) su un corpus di prova.
- La geometria è valutata al primo ordine: la linearizzazione può non valere per passi grandi (s >> 1).
- k_full = 24 in tutti gli stati primari: J è pienamente rangiato, quindi non c'è margine per testare la
  transizione tra sottospazi di diversa dimensionalità.
- E_ood ha solo 6 frasi: il controllo descrittivo rho(k, E_ood) è indicativo, non statistico.
- LoRA rank 8/16 non supportati (necessiterebbero modifiche al codice; rank 0 = full-parameter).
- I checkpoint a 25%/50% e gli sweep della dimensione di R_prot (N/4, N/2) sono stati esplorativi.
