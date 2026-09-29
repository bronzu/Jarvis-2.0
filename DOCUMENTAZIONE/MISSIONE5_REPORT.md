# Missione 5 (Sperimentale) — Report Finale

**Data:** 2026-09-28
**Worktree:** `verifica-per-invarianti-attribuzione-causale-via`
**Commit Missione 5:** `41aa11a` (branch `verifica-per-invarianti-attribuzione-causale-via`)
**Codice sperimentale consultabile in:** worktree/branch `verifica-per-invarianti-attribuzione-causale-via` @ commit `41aa11a`
**File codice (non mergiati in main):**
- `cognitive_layer/invariant_verifier.py`
- `cognitive_layer/causal_attribution.py`
- `cognitive_layer/__init__.py` (modificato per esportare i due moduli sopra)

---

## Sintesi Esecutiva

Entrambe le ipotesi sperimentali della Missione 5 sono risultate **non supportate** dall'evidenza empirica sul codice e sui modelli attuali. Questo è l'esito atteso di una missione esplorativa onesta: un risultato negativo ben documentato È il risultato, non un fallimento.

---

## Ipotesi A — Verificatore a Invarianti

> **Ipotesi:** Verificare le conseguenze necessarie di una soluzione (invarianti/vincoli dichiarati) è più affidabile che confrontare il testo di più rami di ragionamento tra loro (Missione 3).

### Evidenza raccolta: **CONTRO** (con dati disponibili)

**STEP A1 — Formato strutturato a 4 campi (CLAIM, INVARIANT, CONSEQUENCE, CONSTRAINT)**

- Implementato parser basato su regex per i 4 campi con template di prompt fisso incluso nel codice (`cognitive_layer/invariant_verifier.py`).
- Testato su **24 prompt** (soglia minima: 20) usando il `MicroTransformer` reale non addestrato via `NeuralModelInterface`.
- **Risultato: 0% aderenza sintattica** (0/24 generazioni rispettano la struttura dei 4 campi).
- **Soglia richiesta: 30%** — **NON SUPERATA**.

**Conseguenza:** Come prescritto dalla missione ("Se questa percentuale è troppo bassa (sotto il 30%) per rendere sensato il resto della missione, FERMARSI QUI e riportarlo come esito"), **gli STEP A2 e A3 non sono stati eseguiti**. Non ha senso costruire un motore di verifica su dati che il modello non sa produrre.

**Nota importante:** Il 0% è atteso per un modello non addestrato. L'ipotesi A rimane **indecidibile in generale** — non è stata confutata l'idea che *un modello addestrato* a produrre quel formato potrebbe rendere la verifica a invarianti più affidabile. Ciò che è dimostrato è che **con il modello attuale non addestrato l'approccio non è praticabile**.

---

## Ipotesi B — Attribuzione Causale via Linear Attention

> **Ipotesi:** Poiché `MultiHeadLinearAttention` accumula uno stato ricorrente come somma di contributi per-token (`kernel(q)·kernel(k)^T·v`, accumulato causalmente), è possibile calcolare il contributo esatto di un singolo token passato all'output finale rimuovendolo algebricamente dallo stato accumulato, SENZA rieseguire un forward pass completo.

### Evidenza raccolta: **CONTRO** — NON algebricamente valida sull'implementazione reale

**STEP B1 — Verifica preliminare di fattibilità (bloccante)**

Ispezionato il codice reale di `MultiHeadLinearAttention` in `neural_core/network.py:328-478`:

```
cumsum_phi_k = cumsum(phi_k, axis=2)                    # (B, H, T, d_k)
outer_prod = einsum('bhtd,bhtv->bhtdv', phi_k, v_heads) # (B, H, T, d_k, d_k)
cumsum_kv = cumsum(outer_prod, axis=2)                  # (B, H, T, d_k, d_k)
numerator = einsum('bhtd,bhtdv->bhtv', phi_q, cumsum_kv)
denominator = einsum('bhtd,bhtd->bht', phi_q, cumsum_phi_k)
output = numerator / (denominator[:, :, :, None] + eps)
```

**Problema fondamentale:** Il **denominatore** accoppia `phi_q[t]` (query del token corrente) con la **somma cumulativa di TUTTI i `phi_k` precedenti**. La divisione `numerator / (denominator + eps)` è una **non-linearità** che impedisce la decomposizione esatta per-token.

- `cumsum_phi_k` e `cumsum_kv` SONO somme lineari decomponibili per-token.
- MA il denominatore `phi_q[t] · Σ_{j≤t} phi_k[j]` mischia query corrente e TUTTE le key passate.
- Rimuovere token `i` algebricamente richiederebbe modificare `cumsum_phi_k[t]` per tutti `t ≥ i`, MA poi il nuovo denominatore `phi_q[t] · (cumsum_phi_k[t] - phi_k[i])` produce un risultato **diverso** dal forward pass reale senza token `i`, perché nel forward reale anche `phi_q[t]` per `t > i` cambierebbe (l'input a quei token è diverso se il contesto cambia).

**STEP B2 — Test numerico (10 sequenze, tolleranza dichiarata PRIMA: 1e-6)**

| Test | Max Absolute Error | Pass (≤1e-6) |
|------|-------------------|--------------|
| 1    | 1.63e-01          | ❌ FAIL      |
| 2    | 2.14e-01          | ❌ FAIL      |
| 3    | 1.88e-01          | ❌ FAIL      |
| 4    | 2.15e-01          | ❌ FAIL      |
| 5    | 1.24e-01          | ❌ FAIL      |
| 6    | 1.68e-01          | ❌ FAIL      |
| 7    | 1.11e-01          | ❌ FAIL      |
| 8    | 2.07e-01          | ❌ FAIL      |
| 9    | 1.81e-01          | ❌ FAIL      |
| 10   | 1.83e-01          | ❌ FAIL      |

**Risultato: 0/10 test passano.** Errore massimo misurato: **~2.15e-01** (due ordini di grandezza sopra la tolleranza).

**Conseguenza:** Come prescritto ("Se la scomposizione non è esatta a causa di una qualunque non-linearità presente nell'implementazione reale [...] DICHIARARLO SUBITO nel report come esito della missione e FERMARSI"), **lo STEP B2 non è stato implementato come attribuzione esatta** e **lo STEP B3 non è stato eseguito**.

---

## Se B fosse valida: cosa servirebbe per B3 (collegamento memoria simbolica)

*Sezione inclusa per completezza, sebbene B non sia valida.*

Per rendere B3 eseguibile e informativo (non solo smoke test) servirebbero:

1. **Fix architetturale:** Rimuovere la normalizzazione per denominatore cumulativo, o usare una formulazione di linear attention senza divisione query-dipendente (es. normalizzazione per-token fissata, o attention lineare non normalizzata con scaling appreso).
2. **Modello addestrato:** Un `MicroTransformer` che usa `MultiHeadLinearAttention` come attention principale (attualmente `TransformerBlock` usa `MultiHeadAttention` softmax-based).
3. **Iniezione K/V collegata:** Come rilevato in Missione 2, `symbolic_memory.py` inietta K/V ma `TransformerBlock` usa softmax attention, non linear attention. Servirebbe un branch dove l'iniezione avviene dentro `MultiHeadLinearAttention`.
4. **Dati di test con ground truth:** Sequenze dove si sa *a priori* quale token/nodo iniettato è rilevante per la domanda, per misurare se l'attribuzione lo rileva (contributo normalizzato > soglia) vs irrilevante (contributo ≈ 0).
5. **Soglia di significatività:** Definire metriche quantitative (es. contributo L2 normalizzato > 0.1 = significativo) invece di soglie arbitrarie.

---

## Cosa questa missione NON dimostta (elenco esplicito per evitare over-interpretazione)

### Riguardo Ipotesi A (Invarianti)
1. **NON dimostra** che la verifica a invarianti sia intrinsecamente meno affidabile del confronto tra rami — dimostra solo che **il modello attuale non addestrato non produce il formato richiesto**.
2. **NON dimostra** che un modello addestrato (fine-tuned) sul formato 4-campi fallirebbe — anzi, è plausibile che con training appropriato l'aderenza salga ben sopra 30%.
3. **NON confronta** effettivamente i due approcci su casi reali — lo STEP A3 non è stato eseguito per mancanza di dati validi.
4. **NON valida** il parser o il motore di verifica (STEP A2) — non sono stati testati su strutture valide.

### Riguardo Ipotesi B (Attribuzione Causale)
1. **NON dimostra** che l'attribuzione causale sia impossibile in principle — dimostra che **questa specifica implementazione di Linear Attention (con denominatore query-dipendente) non la supporta esattamente**.
2. **NON esclude** approssimazioni euristiche utili (es. attribution via gradienti, integrated gradients, o rimozione approssimata) — ma la missione richiedeva **esattezza algebrica**, non approssimazione.
3. **NON testa** varianti di linear attention senza denominatore (es. `phi(Q) * cumsum(phi(K) * V) / cumsum(phi(K))` con normalizzazione separata) — solo l'implementazione attuale in `network.py`.
4. **NON invalida** l'uso di linear attention per efficienza O(N) — l'implementazione è corretta per forward/backward, solo non decomponibile esattamente per attribuzione per-token.

### Riguardo la Missione 5 nel complesso
1. **NON è un fallimento del progetto** — è una missione *sperimentale* progettata per testare ipotesi che potevano reggere o no.
2. **NON autorizza** a rimuovere `cognitive_layer/` — i moduli `invariant_verifier.py` e `causal_attribution.py` documentano l'esperimento e possono base per lavoro futuro.
3. **NON implica** che Missioni 1-4 siano difettose — operano su assunti diversi (branch comparison, verifica sintattica/aritmetica) che funzionano sul modello attuale.

---

## File creati/modificati

| File | Stato | Note |
|------|-------|------|
| `cognitive_layer/invariant_verifier.py` | ✅ Creato | Parser 4-campi, motore verifica (A2), confronto (A3) — A2/A3 non eseguiti |
| `cognitive_layer/causal_attribution.py` | ✅ Creato | Test decomponibilità (B1), test attribuzione (B2) — B2 fallisce |
| `cognitive_layer/__init__.py` | ✅ Aggiornato | Esporta entrambi i moduli |

---

## Conclusione

| Ipotesi | Esito | Evidenza |
|---------|-------|----------|
| **A: Invarianti più affidabili** | **Indecidibile con dati attuali** | 0% aderenza sintattica (soglia 30%) — modello non addestrato |
| **B: Attribuzione esatta via Linear Attention** | **Falsa su implementazione reale** | Denominatore query-dipendente rompe decomposizione; errore ~0.2 vs tolleranza 1e-6 |

Entrambi i risultati negativi sono **onesti e documentati** come richiesto dalla natura sperimentale della missione. Nessun risultato è stato forzato, nessuna limitazione nascosta.

---

## Aggiornamento: Micro-Missione — Verifica attribuzione per-layer (denominatore incluso)

**Data:** 2026-09-29

### Risposta alla domanda preliminare

**La risposta è (a).**

Nel test B2 di Missione 5 (`cognitive_layer/causal_attribution.py`, funzione `test_attribution_single_token`, righe 219-231), la sottrazione del contributo del token `i` è stata applicata **allo stato accumulato E al denominatore separatamente, ricalcolando il rapporto DOPO aver tolto il token da entrambi**:

```python
# 1. Azzera contributo del token i in phi_k
phi_k_no_i = phi_k.copy()
phi_k_no_i[:, :, token_idx, :] = 0

# 2. Ricomputa ENTRAMBI gli accumulati cumulativi
cumsum_phi_k_no_i = np.cumsum(phi_k_no_i, axis=2)           # DENOMINATORE
outer_prod_no_i = np.einsum('bhtd,bhtv->bhtdv', phi_k_no_i, v_heads)
cumsum_kv_no_i = np.cumsum(outer_prod_no_i, axis=2)         # STATO (numeratore)

# 3. Ricalcola numeratore E denominatore con i NUOVI accumulati
numerator_no_i = np.einsum('bhtd,bhtdv->bhtv', phi_q, cumsum_kv_no_i)
denominator_no_i = np.einsum('bhtd,bhtd->bht', phi_q, cumsum_phi_k_no_i)

# 4. Ricalcola il RAPPORTO dopo aver tolto il token da ENTRAMBI
output_algebraic = numerator_no_i / (denominator_no_i[:, :, :, None] + 1e-6)
```

### Errore misurato nella formulazione (a)

**~0.2 (2e-1)** contro tolleranza **1e-6** — **0/10 test passano**.

### Conclusione netta della micro-missione

Poiché la risposta è **(a)** e l'errore era comunque **~0.2**, **questa micro-missione non ha nulla da testare**. L'ipotesi B è **falsa anche nella formulazione stretta** (sottrazione simultanea da stato E denominatore, ricalcolo del rapporto, singolo layer isolato).

Il fallimento non è dovuto alla propagazione multi-layer (il test B2 usa già un singolo `MultiHeadLinearAttention`), ma alla non-linearità intrinseca del rapporto `numerator/denominator` dove `phi_q[t]` (query del token corrente) accoppia la somma cumulativa in modo che la rimozione esatta per-token richiederebbe anche la modifica di `phi_q[t]` per `t ≥ i`, cosa che non è algebricamente fattibile senza rieseguire il forward.

**L'ipotesi B va dichiarata chiusa: falsa in ogni formulazione testata finora.**