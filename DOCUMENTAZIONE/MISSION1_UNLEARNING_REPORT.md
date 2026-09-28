# Missione 1 — Machine Unlearning Selettivo (singolo esempio)

**Data:** 2026-09-27
**Fase A:** completamento del protocollo gradient-based (canario, sweep esteso,
diagnostica del passo).
**Fase B:** braccio comparativo di unlearning causale (activation patching,
localizzazione, ablation mirata), stesso corpus, stesso modello, stesse metriche.
**Stato:** implementazione completa, entrambi i bracci misurati, **nessuno dei due
soddisfa i due criteri insieme** nella configurazione prescritta. Ogni numero qui
sotto viene da un'esecuzione reale, riportata per intero nei file di
`experiments/mission1_unlearning/results/`.

---

## 0. Sommario in cinque righe

1. Il protocollo gradient-based prescritto, eseguito come scritto, **non dimentica
   e non protegge**: 12/12 frasi correlate fuori dal 5%, soglia mai raggiunta.
2. Un canario di 6 frasi (A2) ferma l'ascent al **primo passo**, e da lì il
   criterio di collaterale risulta "rispettato" **perché il rollback ha riportato
   il modello a theta_0**, cioè perché non è successo niente.
3. La ragione è un difetto misurabile della formula di proiezione prescritta: il
   termine di protezione domina il gradiente di forget di 83–101 volte nei primi
   passi, con segno opposto, e **il passo non è un passo di ascent** (A4:
   11 passi su 15 con `delta_forget_1o` negativo sotto la formula prescritta,
   0 su 15 con la proiezione ortogonale esatta).
4. Con più passi e learning rate più alti (A3) il forgetting **si raggiunge**
   (z fino a −5.02 su LoRA, −8.20 su tutti i parametri) mantenendo la loss di
   retain entro il +5%: ma è la perdita **media** su 24 frasi a restare ferma,
   mentre la variazione per singola frase correlata arriva a 111009%.
5. Il braccio causale (Fase B) **localizza l'entità in modo netto e non
   selettivo**: il patching individua con esattezza le due posizioni di token
   dell'entità sostituita, ma nessun sottoinsieme di componenti la rimuove senza
   distruggere il collaterale. È la risposta esatta alla domanda di B6:
   rappresentazione diffusa, non localizzata.

---

## 1. Verifica di indipendenza dei fallimenti della suite globale (A1)

I 22 fallimenti e i 6 errori della suite globale **precedono questo lavoro**,
verificato per confronto diretto e non per ispezione.

Metodo: copia pulita di `HEAD` (dove nessuna delle modifiche di questa missione
è committata) estratta con `git archive` in una directory temporanea, e suite
completa lanciata lì. Nessuno `stash`, nessun `checkout`: il worktree non è stato
toccato.

```
baseline (HEAD, senza le modifiche della missione)
$ python3 -m pytest -q --tb=no -p no:randomly
22 failed, 1279 passed, 68 warnings, 6 errors, 6 subtests passed in 105.98s

worktree (con la missione)
$ python3 -m pytest TESTS/ -q --tb=no -p no:randomly
22 failed, 1326 passed, 73 warnings, 6 errors, 6 subtests passed in 265.05s
```

Il delta è **esattamente +47 test passati** (37 in `TESTS/test_unlearning.py`,
10 in `TESTS/test_causal_unlearning.py`). Il confronto delle liste dei nomi dei
test falliti ed in errore è **vuoto**:

```
$ diff <(grep -E '^(FAILED|ERROR)' baseline.txt | sort) <(... worktree ... | sort)
# nessuna differenza
```

**Dichiarazione:** i tempi di parete (105.98 s e 265.05 s) vengono dalla riga di
riassunto di pytest e **non sono archiviati** in `results/`, che contiene solo
l'output degli script della missione. Le liste `FAILED`/`ERROR` confrontate dal
`diff` qui sopra non sono anch'esse in `results/`: la verifica va ripetuta con i
due comandi sopra.

Causa di tutti i 22 fallimenti e dei 6 errori: artefatti assenti dal repository e
esclusi da `.gitignore`, non presenti neppure in `HEAD`.

```
AssertionError: missing neural_core/data/train.txt
AssertionError: missing neural_core/data/validation.txt
AssertionError: missing neural_core/data/test.txt
AssertionError: tokenizer.pkl not found
FileNotFoundError: ... /neural_core/data/model.pkl
```

**Un test è intermittente**, e lo era già prima: `TESTS/test_continual_learning.py::TestFisherLatency::test_fisher_latency_full_vs_micro`
è un test di **latenza** e compare nel baseline in alcune esecuzioni e non in
altre (nel baseline: 22 fallimenti in un giro, 23 in un altro). Non è correlato
con la missione e non è stato toccato; è dichiarato perché un conteggio di suite
che cambia di ±1 senza spiegazione è un fatto, non un rumore da ignorare.

---

## 2. Cosa è stato implementato

| File | Contenuto |
|---|---|
| `neural_core/unlearning.py` | `SelectiveUnlearner`, `UnlearningConfig`, `UnlearningResult`, `forgetting_success()`, `relative_probability_change()`, `single_vs_distribution()`, `fit_fisher()`. Solo NumPy + standard library. Dal 27/09 include il **canario di sicurezza** (`probe_tolerance`, `STOP_PROBE_DRIFT`). |
| `experiments/mission1_unlearning/corpus_unlearning.py` | Corpus reale e split disgiunti: X_forget, X_retain_proj, 12 correlate, 6 holdout mai viste, 4 fuori distribuzione, **6 frasi di probe/canario**, 6 controlli non protetti. |
| `experiments/mission1_unlearning/validate_unlearning.py` | Run unico del protocollo, valutazione dei due criteri, misura del canario, diagnostica A3 (sweep esteso) e A4 (segno del passo). |
| `experiments/mission1_unlearning/causal_unlearning.py` | **Fase B**: inventario architettura, coppia clean/corrupted, activation patching sul flusso residuo, ricerca del set minimo (vincolata e non), ablation, retest con le metriche della Fase A. |
| `TESTS/test_unlearning.py` | 37 test (31 iniziali + 6 su canario, A3, A4). |
| `TESTS/test_causal_unlearning.py` | 10 test sulla Fase B. |
| `.gitignore` | Unica modifica a un file esistente: eccezione `!experiments/mission1_unlearning/**`, sul modello di quella già presente per `phase4_continual_learning`. Senza, i numeri citati qui non sarebbero riproducibili da nessun altro. |

La **Fisher non è ricalcolata**: `SelectiveUnlearner` riusa
`Trainer.compute_fisher_matrix()` (`neural_core/trainer.py:536`), la stessa del
modulo Micro-EWC/LoRA già presente. `test_fisher_e_riusata_non_ricalcolata`
verifica per identità che `unlearner.fisher is trainer.fisher_matrix`.

Nomi dei moduli sperimentali: `corpus_unlearning` / `validate_unlearning` /
`causal_unlearning`, non `corpus` / `validate`. Motivo funzionale: nella stessa
sessione pytest `TESTS/test_continual_learning_criteria.py` importa i moduli
omonimi di `experiments/phase4_continual_learning`; con gli stessi nomi il secondo
import ritrova il modulo già in `sys.modules` e la pipeline misurerebbe il file
sbagliato. Rilevato eseguendo la suite globale.

---

## 3. Protocollo e scostamenti dichiarati

| Voce | Protocollo | Implementato | Scostamento |
|---|---|---|---|
| `g_forget`, `g_retain` | gradienti su X_forget e X_retain_proj | idem, normalizzati per token | nessuno |
| proiezione | `g_f - (g_f·F⁻¹g_r)/(g_r·F⁻¹g_r)·F⁻¹g_r` | idem (`projection_mode="prescribed"`, default) | nessuno |
| `F` | Micro-Fisher diagonale riusata | idem | nessuno |
| `F⁻¹` | elementwise | elementwise + pavimento relativo `fisher_floor=1e-3·max(F)` | **dichiarato**, §7.2 |
| `eta` iniziale | lr fine-tuning JARVIS 2.0 / 10 = 1e-3/10 = **1e-4** | idem (`STANDARD_FINETUNE_LEARNING_RATE = 1e-3`, da `train.py` e `17_MODEL_TRAINING_PROVENANCE.md` §5.2) | nessuno |
| dimezzamento eta | entro i primi 3 step, max 3 riavvii | idem (`halving_window=3`, `max_restarts=3`) | nessuno |
| STOP | retain +5% relativo, o 100 step | idem, più `degenerate_projection` e `probe_drift` | **dichiarato**, §7.3 e §5 |
| rollback | non menzionato | all'ULTIMO stato verificato conforme | **dichiarato**, §7.3 |
| criterio (1) | rientro di X_forget nella distribuzione OOD | log-likelihood **per token**, riferimento = 6 frasi mai viste | **dichiarato**, §7.4 |
| criterio (2) | variazione di probabilità entro 5% relativo sulle 10-15 correlate | idem su 12 frasi, con report separato del peggioramento | nessuno |

---

## 4. Dataset e modello

Fonte dei dati: `neural_core/data/jarvis_dialogue.txt`, già presente nel
repository (97 risposte dell'assistente). Nessuna generazione casuale; split
deterministici con seed fisso 42.

- **X_forget** (1 frase, inserita nel training 8 volte perché un esempio non
  memorizzato non ha nulla da togliere):
  *"Il mio creatore e' Riccardo Bronzuoli e mi ha progettato come assistente
  personale per l'uso quotidiano."*
- **X_retain_proj**: 24 frasi di addestramento, disgiunte dalle correlate.
- **12 frasi correlate**: entità condivisa con X_forget (JARVIS come assistente,
  il suo creatore, il Neural Core). Non in X_retain_proj.
- **6 frasi holdout mai viste**: riferimento del criterio (1).
- **4 frasi fuori distribuzione**: prosa di `DOCUMENTAZIONE/SPECIFICA_PROGETTO.md`.
- **6 frasi di probe/canario** (A2): frasi di addestramento disgiunte da
  X_retain_proj, dalle 12 correlate, dall'holdout, da fuori distribuzione e dal
  controllo non protetto. Non entrano mai nel gradiente.
- **6 controlli non protetti**: memorizzate come X_forget ma non protette;
  distinguono "dimenticato" da "solo degradato".

Corpus: 91 frasi + X_forget ×8 = 9996 caratteri, vocabolario 729 token, 46
finestre (context 64, stride 32).

**Sbilanciamento dichiarato.** X_forget ha 0 token fuori vocabolario per
costruzione; ogni frase di controllo "mai vista" ne ha fino a 10, perché in un
corpus di risposte indipendenti ogni frase contiene parole uniche. La direzione
dello sbilanciamento è **conservativa**: abbassare la media del riferimento rende
il criterio (1) più esigente, non meno. I conteggi sono stampati a ogni run.

**Modello**: `MicroTransformer`, d_model 64, 4 teste, d_ff 128, 2 layer, max_len
64, LoRA rank 4, seed 42, float64. **Non è il modello di produzione** (d_model
128, 4 layer, 2737806 parametri): i numeri qui non sono confrontabili con
`17_MODEL_TRAINING_PROVENANCE.md`.

Addestramento: Adam lr 0.01, batch 32, 400 epoche, 46 finestre. Spazio
addestrabile LoRA: **1536 parametri su 162649 (0,94%)**; ramo full-parameter:
161113 su 161113.

---

## 5. A2 — Il canario di sicurezza

**Che cosa è stato aggiunto.** Sei frasi di addestramento, disgiunte sia da
X_retain_proj sia dall'insieme di valutazione, usate **solo** come arresto
dell'ascent: se la variazione **massima** (non media) della probabilità di una
frase del canario supera il 5% rispetto a theta_0, l'ascent si ferma con
`stop_reason="probe_drift"`. Il canario non entra nel gradiente: è una rete di
sicurezza sul comportamento osservabile, non una protezione della loss.

Due scelte dichiarate, non nel testo del protocollo: lo STOP sul canario è un
STOP semplice e non un dimezzamento di eta con riavvio (è un segnale di
sicurezza, non una procedura di ottimizzazione); quando scatta si applica lo
stesso rollback allo stato conforme, che ora è l'ultimo stato che rispetta **sia**
la soglia di retain **sia** quella del canario.

### 5.1 LoRA — il canario cambia l'esito, e lo cambia in un modo che va detto

Grezzo: `results/output_lora.txt` (con canario) e
`results/output_lora_senza_probe.txt` (senza).

| | con canario | senza canario |
|---|---|---|
| STOP | `probe_drift` al passo **1** | `max_steps` dopo 100 |
| variazione massima sul canario al passo 1 | **19.653%** > 5% | — |
| NLL/token X_forget | 4.6834 → 4.6834 | 4.6834 → 4.8576 |
| z sul criterio (1) | 8.21 | 7.702 |
| variazione loss X_retain_proj | +0.000% | +3.227% |
| 12 correlate oltre il 5% | **0/12** | **12/12** (max 345.671%, peggioramento max 522.368%) |
| canario oltre il 5% alla fine | 0/6 | 6/6 (max 97.231%) |

Il criterio (2) passa da "NON RAGGIUNTO, 12/12" a "RAGGIUNTO, 0/12". **Non è un
miglioramento del metodo**: il rollback ha annullato l'unico passo eseguito e il
modello finale è indistinguibile da theta_0. Lo riportiamo perché è esattamente
il modo in cui un criterion può essere soddisfatto senza che sia successo nulla, e
perché nel report di questa missione quel numero non deve essere letto come un
successo. Il test `test_canario_che_ferma_primo_e_un_successo_vacuo` impedisce che
lo diventi.

### 5.2 Full-parameter — lo stesso, ma più netto

Grezzo: `results/output_full.txt` e `results/output_full_senza_probe.txt`.

Al primo passo, con eta = 1e-4, la perdita di retain va da **2.3343 a 19.9154** e
la variazione massima sul canario è **100.000%** (una frase del canario diventa
improbabile del tutto). Il termine di proiezione è **638200 volte** il gradiente
di forget (`||g_proj|| = 2.716e+06` contro `||g_forget|| = 4.256e+00`).

Senza canario, l'ascent si ferma comunque ma per la soglia di retain: 3 riavvii,
`eta 1e-4 → 1.25e-5`, STOP `retain_loss_increase` al passo 4 (retain a 10.3799
contro soglia 2.4510), rollback a theta_0. Anche qui il modello finale è
invariato: NLL/token 2.2035 → 2.2035, z 6.905, 0/12 correlate oltre soglia.

**Conclusione di A2:** il canario, come specificato, è efficace nel fermare
l'escalation, ma a eta = 1e-4 esso ferma **al passo 1** in entrambi i rami. Su
questo esperimento il protocollo, così come specificato, non è applicabile senza
il canario e non fa nulla senza il canario.

---

## 6. A4 — Il segno del primo passo dipende dalla formula di proiezione

Domanda: `first_order_forget_delta < 0` si presenta anche sotto `metric_orthogonal`?
Grezzo: `results/output_lora_first_order.txt`. Stessa theta_0, nessun canario,
nessun rollback, soglia di retain disattivata, prime 15 fasi.

**`delta_forget_1o` è la variazione prevista di primo ordine della NLL di
X_forget**: se è negativa, il passo non è un passo di ascent sulla loss che si
vuole far crescere.

Formula prescritta:

```
 step  delta_forget_1o ||g_proj||/||g_f||  NLL(X_forget)
    1       -2.464e-03          8.342e+01         4.6811
    2       -3.356e-03          8.926e+01         4.6781
    3       -4.248e-03          9.459e+01         4.6743
    4       -4.992e-03          9.867e+01         4.6699
    5       -5.414e-03          1.008e+02         4.6652
    6       -5.428e-03          1.008e+02         4.6607
    7       -5.070e-03          9.870e+01         4.6567
    8       -4.371e-03          9.409e+01         4.6534
    9       -3.366e-03          8.621e+01         4.6511
   10       -2.178e-03          7.528e+01         4.6497
   11       -9.691e-04          6.229e+01         4.6494
   12       +1.300e-04          4.826e+01         4.6499
   13       +9.834e-04          3.440e+01         4.6511
   14       +1.506e-03          2.244e+01         4.6527
   15       +1.755e-03          1.360e+01         4.6545
passi con delta_forget_1o NEGATIVO: 11/15
```

Proiezione ortogonale esatta:

```
 step  delta_forget_1o ||g_proj||/||g_f||  NLL(X_forget)
    1       +2.838e-03          9.997e-01         4.6862
    2       +2.852e-03          9.998e-01         4.6891
    3       +2.866e-03          9.998e-01         4.6919
    4       +2.880e-03          9.998e-01         4.6948
    5       +2.895e-03          9.998e-01         4.6977
    6       +2.909e-03          9.998e-01         4.7006
    7       +2.923e-03          9.998e-01         4.7036
    8       +2.937e-03          9.998e-01         4.7065
    9       +2.951e-03          9.998e-01         4.7095
   10       +2.965e-03          9.998e-01         4.7124
   11       +2.979e-03          9.999e-01         4.7154
   12       +2.993e-03          9.999e-01         4.7184
   13       +3.007e-03          9.999e-01         4.7214
   14       +3.021e-03          9.999e-01         4.7244
   15       +3.035e-03          9.999e-01         4.7275
passi con delta_forget_1o NEGATIVO: 0/15
```

**Risposta ad A4: il difetto è della formula prescritta, non del problema.**
Sotto `prescribed` 11 passi su 15 sono negativi, con il termine di protezione che
domina il gradiente di forget di 83–101 volte; la NLL di X_forget **scende** da
4.6834 a 4.6494 entro il passo 11, e solo dopo che la proiezione si è riallineata
il passo diventa un ascent vero. Sotto `metric_orthogonal` nessun passo è
negativo e il rapporto `||g_proj||/||g_forget||` è **0.9997**: la proiezione
ortogonale esatta, su questa Fisher, rimuove quasi nulla della direzione di
forget. Il ramo full-parameter conferma al passo 1: `delta_forget_1o = -6.702e+01`
con rapporto 6.382e+05.

---

## 7. Difetti misurati della specifica

### 7.1 La formula prescritta non è una proiezione ortogonale se F è anisotropa

Il termine rimosso è un multiplo di `F⁻¹g_r`, quindi
`<g_proj, g_r> = <g_f, g_r> − α·<F⁻¹g_r, g_r>` e la semiosfera di protezione non
si annulla se `F⁻¹ ≠ I`. La proiezione ortogonale esatta rispetto a
`u = F⁻¹g_r` richiede il denominatore `u·u`, non `g_r·F⁻¹·g_r`. Verificato
numericamente in `test_proiezione_prescritta_non_e_ortogonale_con_fisher_anisotropa`,
che fallisce volutamente se la formula viene "corretta" senza registrazione.

`neural_core/unlearning.py` implementa entrambe. Il default resta quello
prescritto; `projection_mode="metric_orthogonal"` implementa la proiezione
ortogonale esatta.

### 7.2 `F⁻¹` è instabile dove la Fisher è piccola

Su questo esperimento la Fisher grezza ha massimo 2.286e-02 e il pavimento
relativo 1e-3 applica un floors a 2.286e-05: il rapporto fra la voce più alta e
quella più bassa dell'inverso è ~1000. Il pavimento è una scelta ingegneristica
dichiarata, interpretabile come prior isotropico (ridge). Senza di esso la
divisione per zero su parametri mai osservati rende l'algoritmo inutilizzabile.
Non è neutra: cambia la metrica, e la tabella §8 mostra quanto.

### 7.3 Rollback in assenza di specifica, e cosa produce

Se lo STOP è provocato dalla soglia sulla loss di retain (o dal canario), il
modello si riporta all'**ultimo stato verificato conforme**, che all'inizio è
theta_0. Il testo del protocollo non dice nulla; lasciare il modello nello stato
che ha appena violato la soglia renderebbe incoerente ogni misura successiva.
`rollback_on_stop=False` mantiene lo stato offensivo e il risultato lo segnala in
`retain_threshold_violated`.

Effetto osservabile: nelle righe con eta grandi della tabella §8 l'ascent termina
con `NLL 4.683→4.683` e `coll max 0.00%`, cioè **non ha fatto nulla**: 3 riavvii,
nessun passo conforme, ritorno a theta_0. Comportamento corretto dato il
protocollo, ed è anche un modo di dire che a quei learning rate il protocollo non
è applicabile.

### 7.4 Confronto per token invece che per sequenza

Il criterio (1) è valutato sulla **log-likelihood media per token**. Sulla
log-likelihood totale, frasi di lunghezza diversa non sono confrontabili: nel primo
run di prova (20 epoche, modello non addestrato) X_forget, 16 token, aveva NLL
totale 80.56 contro 108.56 dell'holdout, cioè z = −3.5 *già dentro la
distribuzione* solo perché è più corta. Confronterebbe la lunghezza, non la
memorizzazione. **Dichiarazione:** quel run di prova è stato scartato (20 epoche,
modello non addestrato) e i suoi file grezzi non sono in `results/`: i due numeri
qui citati vengono dalla console al momento e non sono rintracciabili negli
artefatti salvati. Il run definitivo è 400 epoche ed è archiviato.

Il criterio è riportato in **due forme**: unilaterale (`z <= 1`, come scritto nel
protocollo) e bilaterale (`|z| <= 1`, rientro vero e proprio). La differenza non è
accademica: il criterio unilaterale è soddisfatto anche da un modello distrutto,
per il quale X_forget è molto *meno* probabile di qualunque testo mai visto.

### 7.5 Il criterio di STOP sulla loss di retain è quasi cieco al danno per frase

Questo emerge da A3 ed è il rilievo più utile sul protocollo. La soglia è sulla
loss **media per token** su 24 frasi; il criterio (2) è sulla variazione di
probabilità **per frase** su altre 12. Le due misure possono dissociarsi di ordini
di grandezza: nella riga `eta=3e-3, max_steps=500` della tabella §8.1 la loss di
retain si muove del **+0.48%** mentre 12/12 frasi correlate cambiano, con
variazione massima **725.25%**. Su un modello piccolo che ha già saturato il
corpus, la media su 24 frasi non si muove anche quando il singolooutput cambia
di tre ordini di grandezza.

---

## 8. A3 — Sweep esteso: il forgetting è irraggiungibile, o solo la ricerca finiva presto?

Domanda di A3: prima di concludere che il forgetting non è raggiungibile in
questa configurazione, aumentare `max_steps` a 300-500 e alzare `eta`. Solo
proiezione ortogonale esatta, canario **disattivato** (con il canario ogni
configurazione si fermerebbe al passo 1, misurato, e la domanda diventerebbe
irrilevante), 24 configurazioni per ramo, tutte da theta_0.

### 8.1 LoRA (1536 parametri addestrabili)

```
floor=0.001 eta=0.0001  max_steps=100  stop=max_steps          step=100 | NLL 4.683->5.030 z=    6.80 1-sided=no  2-sided=no  | coll max    568.21% nviol=11 | retain   +0.04%
floor=0.001 eta=0.0001  max_steps=300  stop=max_steps          step=300 | NLL 4.683->5.750 z=    3.96 1-sided=no  2-sided=no  | coll max  11162.59% nviol=12 | retain   +0.94%
floor=0.001 eta=0.0001  max_steps=500  stop=max_steps          step=500 | NLL 4.683->6.263 z=    2.21 1-sided=no  2-sided=no  | coll max  39079.68% nviol=12 | retain   +2.00%
floor=0.001 eta=0.0003  max_steps=100  stop=max_steps          step=100 | NLL 4.683->5.748 z=    3.97 1-sided=no  2-sided=no  | coll max  11118.75% nviol=12 | retain   +0.93%
floor=0.001 eta=0.0003  max_steps=300  stop=max_steps          step=300 | NLL 4.683->6.727 z=    0.81 1-sided=SI  2-sided=SI  | coll max  85335.85% nviol=12 | retain   +3.22%
floor=0.001 eta=0.0003  max_steps=500  stop=max_steps          step=500 | NLL 4.683->7.137 z=   -0.34 1-sided=SI  2-sided=SI  | coll max 111009.17% nviol=12 | retain   +4.35%
floor=0.001 eta=0.001   max_steps=100  stop=max_steps          step=100 | NLL 4.683->6.795 z=    0.61 1-sided=SI  2-sided=SI  | coll max  92463.73% nviol=11 | retain   +3.43%
floor=0.001 eta=0.001   max_steps=300  stop=retain_loss_increa step=201 | NLL 4.683->7.366 z=   -0.80 1-sided=SI  2-sided=SI  | coll max 119065.78% nviol=12 | retain   +5.00%
floor=0.001 eta=0.001   max_steps=500  stop=retain_loss_increa step=201 | NLL 4.683->7.366 z=   -0.80 1-sided=SI  2-sided=SI  | coll max 119065.78% nviol=12 | retain   +5.00%
floor=0.001 eta=0.003   max_steps=100  stop=retain_loss_increa step= 67 | NLL 4.683->7.358 z=   -0.80 1-sided=SI  2-sided=SI  | coll max 119241.93% nviol=12 | retain   +4.97%
floor=0.001 eta=0.003   max_steps=300  stop=retain_loss_increa step= 67 | NLL 4.683->7.358 z=   -0.80 1-sided=SI  2-sided=SI  | coll max 119241.93% nviol=12 | retain   +4.97%
floor=0.001 eta=0.003   max_steps=500  stop=retain_loss_increa step= 67 | NLL 4.683->7.358 z=   -0.80 1-sided=SI  2-sided=SI  | coll max 119241.93% nviol=12 | retain   +4.97%
floor=0.9   eta=0.0001  max_steps=100  stop=max_steps          step=100 | NLL 4.683->5.024 z=    6.83 1-sided=no  2-sided=no  | coll max    534.28% nviol=11 | retain   +0.02%
floor=0.9   eta=0.0001  max_steps=300  stop=max_steps          step=300 | NLL 4.683->5.583 z=    4.61 1-sided=no  2-sided=no  | coll max   4797.51% nviol=12 | retain   +0.04%
floor=0.9   eta=0.0001  max_steps=500  stop=max_steps          step=500 | NLL 4.683->5.935 z=    3.32 1-sided=no  2-sided=no  | coll max   9278.27% nviol=11 | retain   +0.06%
floor=0.9   eta=0.0003  max_steps=100  stop=max_steps          step=100 | NLL 4.683->5.583 z=    4.61 1-sided=no  2-sided=no  | coll max   4814.52% nviol=12 | retain   +0.05%
floor=0.9   eta=0.0003  max_steps=300  stop=max_steps          step=300 | NLL 4.683->6.390 z=    1.87 1-sided=no  2-sided=no  | coll max  14856.24% nviol=12 | retain   +0.10%
floor=0.9   eta=0.0003  max_steps=500  stop=max_steps          step=500 | NLL 4.683->6.737 z=    0.86 1-sided=SI  2-sided=SI  | coll max  19004.52% nviol=12 | retain   +0.14%
floor=0.9   eta=0.001   max_steps=100  stop=max_steps          step=100 | NLL 4.683->6.466 z=    1.64 1-sided=no  2-sided=no  | coll max  16073.22% nviol=12 | retain   +0.13%
floor=0.9   eta=0.001   max_steps=300  stop=max_steps          step=300 | NLL 4.683->7.342 z=   -0.91 1-sided=SI  2-sided=SI  | coll max  18571.42% nviol=12 | retain   +0.23%
floor=0.9   eta=0.001   max_steps=500  stop=max_steps          step=500 | NLL 4.683->7.663 z=   -1.70 1-sided=SI  2-sided=no  | coll max   8842.96% nviol=12 | retain   +0.25%
floor=0.9   eta=0.003   max_steps=100  stop=max_steps          step=100 | NLL 4.683->7.343 z=   -0.91 1-sided=SI  2-sided=SI  | coll max  19667.94% nviol=12 | retain   +0.30%
floor=0.9   eta=0.003   max_steps=300  stop=max_steps          step=300 | NLL 4.683->8.227 z=   -2.99 1-sided=SI  2-sided=no  | coll max   1991.65% nviol=12 | retain   +0.39%
floor=0.9   eta=0.003   max_steps=500  stop=max_steps          step=500 | NLL 4.683->9.153 z=   -5.02 1-sided=SI  2-sided=no  | coll max    725.25% nviol=12 | retain   +0.48%
```

### 8.2 Full-parameter (161113 parametri addestrabili)

```
floor=0.001 eta=0.0001  max_steps=100  stop=max_steps          step=100 | NLL 2.204->2.380 z=    6.62 1-sided=no  2-sided=no  | coll max     26.40% nviol= 4 | retain   +0.02%
floor=0.001 eta=0.0001  max_steps=300  stop=max_steps          step=300 | NLL 2.204->2.847 z=    6.06 1-sided=no  2-sided=no  | coll max     58.04% nviol= 9 | retain   +0.03%
floor=0.001 eta=0.0001  max_steps=500  stop=max_steps          step=500 | NLL 2.204->3.389 z=    5.53 1-sided=no  2-sided=no  | coll max     74.76% nviol= 9 | retain   +0.04%
floor=0.001 eta=0.0003  max_steps=100  stop=max_steps          step=100 | NLL 2.204->2.844 z=    6.06 1-sided=no  2-sided=no  | coll max     58.05% nviol= 9 | retain   +0.03%
floor=0.001 eta=0.0003  max_steps=300  stop=max_steps          step=300 | NLL 2.204->4.113 z=    4.88 1-sided=no  2-sided=no  | coll max     89.73% nviol=11 | retain   +0.09%
floor=0.001 eta=0.0003  max_steps=500  stop=max_steps          step=500 | NLL 2.204->6.326 z=    3.45 1-sided=no  2-sided=no  | coll max    128.05% nviol=12 | retain   +0.35%
floor=0.001 eta=0.001   max_steps=100  stop=max_steps          step=100 | NLL 2.204->4.274 z=    4.75 1-sided=no  2-sided=no  | coll max     91.68% nviol=11 | retain   +0.11%
floor=0.001 eta=0.001   max_steps=300  stop=max_steps          step=300 | NLL 2.204->14.067 z=   -0.53 1-sided=SI  2-sided=SI  | coll max    100.00% nviol=12 | retain   +1.29%
floor=0.001 eta=0.001   max_steps=500  stop=max_steps          step=500 | NLL 2.204->29.595 z=   -5.29 1-sided=SI  2-sided=no  | coll max   4975.78% nviol=12 | retain   +4.48%
floor=0.001 eta=0.003   max_steps=100  stop=max_steps          step=100 | NLL 2.204->13.844 z=   -0.43 1-sided=SI  2-sided=SI  | coll max    100.00% nviol=12 | retain   +1.36%
floor=0.001 eta=0.003   max_steps=300  stop=retain_loss_increa step=173 | NLL 2.204->30.491 z=   -5.40 1-sided=SI  2-sided=no  | coll max   6194.34% nviol=12 | retain   +4.94%
floor=0.001 eta=0.003   max_steps=500  stop=retain_loss_increa step=173 | NLL 2.204->30.491 z=   -5.40 1-sided=SI  2-sided=no  | coll max   6194.34% nviol=12 | retain   +4.94%
floor=0.9   eta=0.0001  max_steps=100  stop=max_steps          step=100 | NLL 2.204->2.380 z=    6.62 1-sided=no  2-sided=no  | coll max     26.40% nviol= 4 | retain   +0.01%
floor=0.9   eta=0.0001  max_steps=300  stop=max_steps          step=300 | NLL 2.204->2.847 z=    6.06 1-sided=no  2-sided=no  | coll max     58.04% nviol= 9 | retain   +0.00%
floor=0.9   eta=0.0001  max_steps=500  stop=max_steps          step=500 | NLL 2.204->3.389 z=    5.53 1-sided=no  2-sided=no  | coll max     74.77% nviol= 9 | retain   -0.01%
floor=0.9   eta=0.0003  max_steps=100  stop=max_steps          step=100 | NLL 2.204->2.844 z=    6.06 1-sided=no  2-sided=no  | coll max     58.05% nviol= 9 | retain   +0.01%
floor=0.9   eta=0.0003  max_steps=300  stop=max_steps          step=300 | NLL 2.204->4.113 z=    4.88 1-sided=no  2-sided=no  | coll max     89.73% nviol=11 | retain   -0.01%
floor=0.9   eta=0.0003  max_steps=500  stop=max_steps          step=500 | NLL 2.204->6.325 z=    3.45 1-sided=no  2-sided=no  | coll max    137.44% nviol=12 | retain   +0.15%
floor=0.9   eta=0.001   max_steps=100  stop=max_steps          step=100 | NLL 2.204->4.273 z=    4.75 1-sided=no  2-sided=no  | coll max     91.68% nviol=11 | retain   +0.00%
floor=0.9   eta=0.001   max_steps=300  stop=max_steps          step=300 | NLL 2.204->14.050 z=   -0.54 1-sided=SI  2-sided=SI  | coll max    114.11% nviol=12 | retain   +0.44%
floor=0.9   eta=0.001   max_steps=500  stop=max_steps          step=500 | NLL 2.204->29.427 z=   -5.29 1-sided=SI  2-sided=no  | coll max   5956.98% nviol=12 | retain   +1.49%
floor=0.9   eta=0.003   max_steps=100  stop=max_steps          step=100 | NLL 2.204->13.828 z=   -0.44 1-sided=SI  2-sided=SI  | coll max    117.78% nviol=12 | retain   +0.53%
floor=0.9   eta=0.003   max_steps=300  stop=max_steps          step=300 | NLL 2.204->75.279 z=   -8.20 1-sided=SI  2-sided=no  | coll max   3972.62% nviol=12 | retain   +3.01%
floor=0.9   eta=0.003   max_steps=500  stop=retain_loss_increa step=339 | NLL 2.204->106.687 z=   -7.36 1-sided=SI  2-sided=no  | coll max    100.00% nviol=12 | retain   +4.79%
```

### 8.3 Risposta ad A3

**Converge, e converge distruggendo le frasi correlate.** Il forgetting non era
irraggiungibile: era irraggiungibile *a 100 passi*. Con 300-500 passi e
`eta ≥ 3e-4` la soglia z ≤ 1 viene superata anche in forma bilaterale (z = −0.34
su LoRA, −5.29 su full-parameter, fino a −8.20). La perdita media su X_retain_proj
resta entro il +5% in tutte le 48 configurazioni.

Ma il criterio di STOP non scatta mai, e il motivo è il rilievo §7.5: la soglia è
sulla **media** su 24 frasi, il collaterale si misura **per frase** su altre 12.
Il prezzo del forgetting raggiunto è 12/12 frasi correlate oltre il 5% in ogni
riga, con variazione massima fino a **111009%** (LoRA) e **6194%**
(full-parameter).

Confronto fra i due rami, a parità di eta (1e-4) e di proiezione corretta: passare
da 1536 a 161113 parametri addestrabili riduce il collaterale massimo da 568% a
26% e le frasi oltre soglia da 11/12 a 4/12, con retain a +0.02%. È la misura più
netta del fatto che il collo di bottiglia è la **dimensione dello spazio
addestrabile condiviso**, non la formula di proiezione. Ma non basta: a 500 passi
anche il ramo full-parameter arriva a 74.76% di collaterale massimo.

Attenzione: nel ramo full-parameter la loss di addestramento arriva a 0.0020 — il
modello memorizza l'intero corpus di 46 finestre. È un regime di overfitting, e le
distribuzioni di controllo ne risentono (NLL/token holdout 11.83 contro 7.26 nel
ramo LoRA). Il confronto fra i due rami è indicativo sul meccanismo, non un
confronto di qualità fra configurazioni equivalenti.

---

## 9. Curva di compromesso sulla configurazione originale

Sweep 2 proiezioni × 2 pavimenti × 4 learning rate = 16 configurazioni, canario
disattivo come STOP ma misurato a fine run, tutte le righe da theta_0. Le 16
righe sono qui sotto con la colonna `probe` aggiunta e senza le colonne
`restart` e `peggiora`, che sono nel grezzo. Grezzo:
`results/output_lora_sweep.txt` e
`results/output_lora_sweep_con_canario.txt` (stesse righe con il canario attivo
come STOP).

```
prescribed            floor=0.001 eta=0.0001 stop=max_steps    step= 100 | NLL 4.683->4.858 z=  7.70 1-sided=no  2-sided=no  | coll max    345.67% nviol=12 | probe max   97.23% | retain  +3.23%
prescribed            floor=0.001 eta=0.001  stop=max_steps    step= 102 | NLL 4.683->5.010 z=  8.41 1-sided=no  2-sided=no  | coll max   2821.67% nviol=11 | probe max   99.99% | retain -14.94%
prescribed            floor=0.001 eta=0.01   stop=retain_loss  step=   5 | NLL 4.683->4.694 z=  8.16 1-sided=no  2-sided=no  | coll max    188.00% nviol=10 | probe max   93.82% | retain  +2.92%
prescribed            floor=0.001 eta=0.1    stop=retain_loss  step=   4 | NLL 4.683->4.683 z=  8.21 1-sided=no  2-sided=no  | coll max      0.00% nviol= 0 | probe max    0.00% | retain  +0.00%
prescribed            floor=0.9   eta=0.0001 stop=max_steps    step= 100 | NLL 4.683->4.721 z=  8.84 1-sided=no  2-sided=no  | coll max    213.69% nviol=12 | probe max   99.61% | retain  +3.30%
prescribed            floor=0.9   eta=0.001  stop=max_steps    step= 100 | NLL 4.683->4.848 z=  6.23 1-sided=no  2-sided=no  | coll max   1233.04% nviol=12 | probe max   99.28% | retain  -5.25%
prescribed            floor=0.9   eta=0.01   stop=max_steps    step= 103 | NLL 4.683->5.535 z=  4.83 1-sided=no  2-sided=no  | coll max    588.22% nviol=12 | probe max   98.72% | retain -10.29%
prescribed            floor=0.9   eta=0.1    stop=retain_loss  step=   4 | NLL 4.683->4.683 z=  8.21 1-sided=no  2-sided=no  | coll max      0.00% nviol= 0 | probe max    0.00% | retain  +0.00%
metric_orthogonal     floor=0.001 eta=0.0001 stop=max_steps    step= 100 | NLL 4.683->5.030 z=  6.80 1-sided=no  2-sided=no  | coll max    568.21% nviol=11 | probe max  341.74% | retain  +0.04%
metric_orthogonal     floor=0.001 eta=0.001  stop=max_steps    step= 100 | NLL 4.683->6.795 z=  0.61 1-sided=SI  2-sided=SI  | coll max  92463.73% nviol=11 | probe max   99.99% | retain  +3.43%
metric_orthogonal     floor=0.001 eta=0.01   stop=retain_loss  step=  21 | NLL 4.683->7.361 z= -0.83 1-sided=SI  2-sided=SI  | coll max 120658.21% nviol=12 | probe max  100.00% | retain  +4.99%
metric_orthogonal     floor=0.001 eta=0.1    stop=retain_loss  step=   6 | NLL 4.683->7.082 z= -0.41 1-sided=SI  2-sided=SI  | coll max 120596.87% nviol=12 | probe max  145.63% | retain  +4.34%
metric_orthogonal     floor=0.9   eta=0.0001 stop=max_steps    step= 100 | NLL 4.683->5.024 z=  6.83 1-sided=no  2-sided=no  | coll max    534.28% nviol=11 | probe max  328.53% | retain  +0.02%
metric_orthogonal     floor=0.9   eta=0.001  stop=max_steps    step= 100 | NLL 4.683->6.466 z=  1.64 1-sided=no  2-sided=no  | coll max  16073.22% nviol=12 | probe max   99.53% | retain  +0.13%
metric_orthogonal     floor=0.9   eta=0.01   stop=max_steps    step= 100 | NLL 4.683->8.440 z= -3.45 1-sided=SI  2-sided=no  | coll max   1748.13% nviol=12 | probe max   99.98% | retain  +0.67%
metric_orthogonal     floor=0.9   eta=0.1    stop=retain_loss  step=  35 | NLL 4.683->9.992 z= -7.11 1-sided=SI  2-sided=no  | coll max    615.70% nviol=12 | probe max  100.00% | retain  +4.54%
```

Lettura: il protocollo prescritto non dimentica in nessuna delle 16
configurazioni originali (miglior z = 4.83 fra le righe `prescribed floor=0.9
eta=0.01`); la proiezione corretta protegge molto meglio la loss di retain a parità
di eta (+0.04% contro +3.23% a eta 1e-4, 11 contro 12 frasi oltre soglia) e il
forgetting si può ottenere, ma solo al prezzo di un collaterale di ordini di
grandezza.

**Lo stesso sweep con il canario attivo** (`results/output_lora_sweep_con_canario.txt`)
degenere: 14 righe su 16 si fermano al passo 1 con `probe_drift` e riportano
`NLL 4.683→4.683`, `coll max 0.00%`. Le due righe che sopravvivono fino al passo
4 sono le due `metric_orthogonal` a eta 1e-4 (`coll max` 4.86% e 4.88%, `probe
max` 4.21% e 4.07%, z 8.17). È la misura compatta di §5.1: il canario, a eta
1e-4, è l'unica cosa che tiene insieme i due criteri — perché impedisce
all'ascent di accadere.

---

## 10. Fase B — Unlearning causale

Stesso corpus, stesso tokenizer, stesso X_forget, stesse 12 correlate, stesso
holdout, stesso modello e stesse funzioni di misura della Fase A
(`forgetting_success`, `relative_probability_change`, stesse soglie). Cambia il
metodo: invece di spostare tutti i pesi lungo una direzione di ascent, si cerca
**dove** l'informazione di X_forget è causalmente necessaria.

### 10.1 B0 — Architettura reale, letta dal codice

```
classe                : MicroTransformer
vocab_size            : 729
d_model               : 64
num_heads             : 4
head_dim (d_model/heads): 16
d_ff                  : 128
num_layers            : 2
max_len (context)     : 64
lora_rank             : 4
dtype                 : float64
parametri totali      : 162649
parametri addestrabili: 4 tensori, 1536 valori
tensori congelati     : 37
  # parametro                      forma  addestrabile
  0 layer0_lora_0_A            (4, 128)            si
  1 layer0_lora_0_B             (64, 4)            si
  2 layer1_lora_0_A            (4, 128)            si
  3 layer1_lora_0_B             (64, 4)            si
blocchi (pre-norm, residuo): x = x + Attn(LN(x)); x = x + FF(LN(x))
siti patchabili sul flusso residuo: 3
  sito 0: ingresso del blocco 0
  sito 1: ingresso del blocco 1
  sito 2: flusso residuo finale (ingresso della LayerNorm di uscita)
```

Questi numeri vengono dagli attributi dell'oggetto modello (`MicroTransformer.config()`
e la selezione dei parametri addestrabili del `Trainer`), non da uno schema
scritto. `test_b0_inventario_coerente_con_il_codice` verifica che l'inventario
restituisca esattamente `model.config()` e che i siti patchabili siano
`num_layers + 1`.

### 10.2 B1 — Coppia clean / corrupted

```
clean    : Il mio creatore e' Riccardo Bronzuoli e mi ha progettato come assistente personale per l'uso quotidiano.
corrupted: Il mio creatore e' Neural Core e mi ha progettato come assistente personale per l'uso quotidiano.
sostituzione: 'Riccardo Bronzuoli' -> 'Neural Core'
token    : clean 16 | corrupted 16 | stessa lunghezza: True
<UNK>    : clean 0 | corrupted 0
loglikelihood/token: clean -4.6834 | corrupted -5.2155 | delta -0.5322
```

**Perché "Neural Core" e non un altro nome.** Nel vocabolario di 729 token, fitto
solo sul corpus di addestramento, **non esiste alcun altro nome proprio di
persona**: nessuna alternativa del tipo "Lorenzo", "Rossi", "Bianchi", "Fabbri" è
presente (verificato, e il test `test_b1_entita_sostitutiva_e_davvero_nel_vocabolario`
lo fissa). Sostituire con un nome nuovo avrebbe introdotto token `<UNK>` e la
differenza misurata sarebbe stata **lessicale, non semantica**. "Neural Core" è un
entità già presente nel corpus, due token, entrambi nel vocabolario, sostituzione
grammaticalmente compatibile ("Il mio creatore e' Neural Core").

**Dichiarato**: la scelta rende la coppia clean/corrupted **più simile** di quanto
sarebbe stata con un nome nuovo, e quindi rende il patching più difficile da
rilevare, non più facile. La direzione è conservativa. Confronto lessicale
residuo: 14 dei 16 token sono identici.

### 10.3 B2 — Causal tracing per activation patching

Metrica: `score = NLL(continuazione pulita | run corrotto) − NLL(continuazione
pulita | run corrotto con attivazione pulita patchata)`, in NLL per token. Positivo
= il patching dell'attivazione pulita nel run corrotto ripristina la continuazione
pulita. È riportata anche la direzione inversa (run pulito con attivazione
corrotta): se le due direzioni concordano con segno opposto, la dipendenza è
reciproca e non un artefatto della sostituzione.

Riferimenti: NLL della continuazione pulita sul run corrotto = 5.2650 per token,
sul run pulito = 4.6834.

```
sito  pos    score_fwd    score_rev  NLL dopo patch fwd
   0    0       0.0000       0.0000              5.2650
   0    1       0.0000       0.0000              5.2650
   0    2       0.0000       0.0000              5.2650
   0    3       0.0000       0.0000              5.2650
   0    4       0.2458      -0.0599              5.0192
   0    5       0.5218      -0.3359              4.7433
   0    6..15    0.0000       0.0000              5.2650   (tutte)
   1    4       0.0732      -0.0883              5.1918
   1    5       0.4211      -0.4534              4.8439
   1    6       0.0400      -0.0361              5.2250
   1    7      -0.0343       0.0263              5.2993
   1    8      -0.0026      -0.0001              5.2676
   1    9       0.0234      -0.0345              5.2417
   1   10       0.0174      -0.0107              5.2476
   1   11      -0.0048       0.0045              5.2699
   1   12      -0.0116      -0.0000              5.2766
   1   13       0.0040      -0.0040              5.2611
   1   14       0.0125      -0.0137              5.2525
   1   15       0.0055      -0.0041              5.2595
   2    4       0.0976      -0.0976              5.1674
   2    5       0.3851      -0.3851              4.8800
   2    6       0.0359      -0.0359              5.2292
   2    7      -0.0222       0.0222              5.2872
   2    8       0.0100      -0.0100              5.2550
   2    9       0.0334      -0.0334              5.2317
   2   10       0.0160      -0.0160              5.2490
   2   11      -0.0039       0.0039              5.2690
   2   12       0.0158      -0.0158              5.2492
   2   13      -0.0119       0.0119              5.2769
   2   14       0.0142      -0.0142              5.2508
   2   15       0.0116      -0.0116              5.2534

punteggio massimo: sito 0 posizione 5 -> 0.5218
  sito 0: somma 0.7676 max 0.5218 media 0.0480
  sito 1: somma 0.5439 max 0.4211 media 0.0340
  sito 2: somma 0.5817 max 0.3851 media 0.0364
  concordanza fra le due direzioni: 46/48
```

**Il patching localizza l'entità con esattezza.** Le posizioni 4 e 5 della
finestra sono "Riccardo" e "Bronzuoli". Al sito 0 (ingresso del blocco 0, cioè
embedding + posizione) **tutte le altre 14 posizioni hanno punteggio esattamente
0.0000**: l'informazione sull'entità sostituita non è ancora presente nel flusso
residuo prima di arrivarci, e non lo è dopo che è passata. Il segnale compare
esattamente sulle due posizioni dell'entità, cala lungo i blocchi successivi
(0.5218 → 0.4211 → 0.3851) e la concordanza fra le due direzioni è 46/48 su 48
(posizioni 4 e 5 hanno segno opposto nelle due direzioni in entrambi i casi: le
due righe discordi sono entrambe posizioni di rumore del sito 1, con punteggi
in valore assoluto sotto 0.04, cioè una con score −0.0026 e una il cui score
inverso è negativo ma sotto 0.0001 in valore assoluto).

Al livello della singola coordinata: 881 componenti su 3072 (3 siti × 16
posizioni × 64 coordinate) danno un ripristino positivo; i primi 25 per
punteggio sono **tutti** alle posizioni 4 e 5 dell'entità, e i punteggi scendono
di un ordine di grandezza rispetto a quelli aggregati per posizione (0.3033
contro 0.5218 al sito 1), cioè l'informazione è distribuita su molte
coordinate delle due posizioni, non concentrata su poche:

```
 rank  sito  pos  coord        score
    1     1    5      3       0.3033
    2     1    5      5       0.2979
    3     1    5     39       0.2616
    4     1    5     61       0.1962
    5     1    5     25       0.1779
    6     0    5     22       0.1559
    7     0    5     61       0.1526
    8     0    5     39       0.1440
    9     0    5     28       0.1415
   10     0    5     30       0.1275
   ...  (25/25 alle posizioni 4 e 5)
```

Nel ramo full-parameter il quadro è lo stesso e più netto: picco a sito 0
posizione 4 con score 1.0695 (NLL 3.9331 → 2.8636), somme 1.9989 / 1.8379 /
1.7296, concordanza 47/48.

**Che cosa non è dimostrato da questi numeri.** Il patching misura l'effetto di
*sostituire* un'attivazione, quindi una dipendenza del comportamento
dall'attivazione in quel punto. Non distingue "informazione memorizzata lì" da
"informazione che lì passa". È dichiarato, non nascosto.

### 10.4 B3 — Set minimo di componenti

Pool di candidati a tre granularità, tutti ordinati per lo stesso punteggio
causale: **3123** componenti = 3072 a singola coordinata + 48 a posizione
intera + 3 a sito intero. Ablation: azzero delle sole coordinate indicate,
nient'altro. Soglia: la stessa del criterio (1), cioè
`NLL/token ≥ 6.9431` (holdout media 7.2567 − 1 deviazione standard 0.3136).
NLL iniziale 4.6834.

**B3, ricerca non vincolata al collaterale**:

```
   k   NLL/token     soglia  raggiunta                componente aggiunto  coord. attive
   1      6.5917     6.9431         no             sito 0 (stream intero)           1024
   2      6.5917     6.9431         no             sito 2 (stream intero)           2048
   3      6.5917     6.9431         no             sito 1 (stream intero)           3072
   ...  (NLL costante)
miglior NLL/token raggiunta con l'intero pool: 6.5917 (soglia 6.9431, NLL iniziale 4.6834, esplorati 5 candidati, poi saturazione rilevata)
```

La ricerca **satura**: azzerare l'intero flusso residuo rende costante l'uscita
del modello, quindi la NLL/token ha un **tetto di 6.5917** (quello di un
predittore costante su questo vocabolario) e la soglia di 6.9431 è **oltre quel
tetto**. Nessun sottoinsieme dei 3123 candidati la raggiunge, e non per mancanza
di candidati: il tetto è strutturale.

Lo stesso nel ramo full-parameter, con soglie diverse perché il modello è
sovraffittato: NLL iniziale 2.2035, soglia 10.4375 (holdout media 11.8319 − std
1.3944), **tetto dell'ablation 9.0269**, non raggiunto. In quel ramo la soglia del
criterio (1) è oltre la capacità di qualunque predittore del corpus: è un
limite del riferimento scelto, dichiarato, non del metodo.

**B3c, stessa ricerca vincolata al collaterale** (candidati che da soli violano
il 5% scartati, accumulo fermato al primo collaterale rotto):

```
candidati: 3123 | scartati come distruttivi: 3116 | ammessi: 7
   k   NLL/token     soglia  viol. coll.  raggiunta                   componente
   1      4.6854     6.9431            0         no        sito 2 pos 15 coord 5
   2      4.6860     6.9431            0         no       sito 2 pos 15 coord 49
   3      4.6886     6.9431            1         no       sito 2 pos 2 coord 1
collaterale rotto al primo k = 3 con NLL/token 4.6886 e 1 frasi oltre il 5%: la soglia di forgetting non e' ancora stata raggiunta
```

Ramo full-parameter: 3104 distruttivi su 3123, 19 ammessi, collaterale rotto al
k = 3 con NLL 2.1510.

### 10.5 B3b — Set minimo di parametri

Braccio parallelo a granularità di **parametro**, per rispondere alla domanda
"quanti parametri" del confronto finale. Non è causal tracing: è localizzazione
per effetto diretto, e viene riportata come tale.

```
[B3b] candidati valutati: 1536 | scartati come distruttivi: 1467 | ammessi: 69
[B3b] esempio di candidato distruttivo: layer0_lora_0_A[0] = 0.3567 -> NLL/token 4.7110 con 15 violazioni di collaterale
   k   NLL/token  viol. coll.  raggiunta
   1      4.6870            0         no
   2      4.6886            1         no
primo k con violazione di collaterale: 2 (NLL/token 4.6886, 1 frasi oltre il 5%). Il collaterale si rompe PRIMA che il forgetting sia raggiunto.
```

**Il vincolo di collaterale è indispensabile, e misurarlo cambia la risposta.**
Senza di esso la ricerca avrebbe risposto "k = 1": basta trovare l'unico valore di
LoRA la cui azzeramento rompe il modello. In un primo run, prima che il vincolo
fosse aggiunto, la ricerca ha restituito esattamente k = 1 con NLL/token 7.0023
(passaggio di soglia) e, nello stesso momento, **12/12 frasi correlate e 6/6 del
canario oltre il 5%** (canario max 15.277%). Il numero era un artefatto: non era
una localizzazione, era un guasto. Il vincolo scarta quei candidati a monte.
**Dichiarazione:** quel run preliminare è stato scartato e i suoi file grezzi non
sono stati archiviati in `results/`; i numeri qui citati vengono dalla console al
momento e non sono riproducibili dagli artefatti salvati. Il run che li ha
sostituiti è invece interamente in `results/output_causal_lora.txt`.

Nel ramo full-parameter il braccio a parametri **non è stato eseguito**: 161113
candidati superano il tetto di 100000 valutazioni individuali. È dichiarato come
un limite di costo, **non** come un'assenza di localizzazione, e non viene
riportato come numero.

### 10.6 B4/B5 — Retest con le metriche della Fase A

Nessun intervento è stato applicato, perché nessun set minimo è stato trovato. Il
retest riporta quindi solo il riferimento e i due bracci della Fase A, misurati
nello stesso file con le stesse funzioni:

```
intervento                           comp.  NLL/token   z(1s)  2-sided   coll max  nViol  probe max  nViol
Fase A protocollo (prescribed)        1536     4.8576    7.70       no    345.67%     12    97.23%      6
Fase A proiezione corretta            1536     5.0303    6.80       no    568.21%     11   341.74%      4
```

Ramo full-parameter:

```
intervento                           comp.  NLL/token   z(1s)  2-sided   coll max  nViol  probe max  nViol
Fase A protocollo (prescribed)      161113     2.2035    6.90       no      0.00%      0     0.00%      0
Fase A proiezione corretta          161113     2.3798    6.62       no     26.40%      4    17.40%      4
```

### 10.7 B6 — Verdetto sulla localizzazione

**Nessuna localizzazione utilizzabile, a nessuno dei due livelli.**

- Il **causal tracing localizza l'entità in modo netto e non selettivo**: il
  segnale è alle due posizioni di token dell'entità sostituita e a nessun'altra
  parte. Questo è un risultato positivo, e basta a stabilire dove l'informazione
  *passa*.
- Ma **nessun intervento selettivo su quel punto rimuove l'informazione**. Al
  livello delle attivazioni l'ablation satura a 6.5917 contro una soglia di
  6.9431, e dei 3123 candidati **3116 sono distruttivi se applicati da soli**:
  l'ablation che azzera il segnale causale azzera anche tutto il resto della
  rappresentazione a quella posizione. Dei 7 superstiti, l'accumulo rompe il
  collaterale al terzo.
- Al livello dei parametri, 1467 dei 1536 candidati sono distruttivi da soli e
  l'accumulo dei 69 superstiti rompe il collaterale al secondo, con una NLL
  (4.6886) ancora lontana dalla soglia (6.9431).

L'informazione di X_forget è **ridondante e distribuita**: l'entità è letta in un
punto preciso, ma non in un modo che si possa spegnere senza spegnere il resto.
Questo è quanto la missione chiedeva di riportare se il set minimo fosse risultato
"quasi tutti i parametri" — ed è un esito scientificamente valido, non
un'implementazione fallita. È anche tipico di un modello piccolo che sovraffitta
un corpus di poche decine di frasi: la ridondanza è una conseguenza
dell'overfitting. **Vale per questo corpus e per questo modello, non per il
modello di produzione.**

Una precisazione tecnica sull'intervento di attivazione, dichiarata perché ne
limita l'uso: esso è definito su **posizioni di token**, quindi non è applicabile
a input di lunghezza diversa senza perdere parte dell'effetto. Su una sequenza più
corta della posizione indicata l'ablation viene saltata. Anche se avesse trovato
un set minimo, sarebbe un intervento a runtime condizionato alla forma
dell'input, non una modifica del modello.

---

## 11. Confronto finale fra i due approcci

Stesso modello, stesso corpus, stesse funzioni di misura, stesse soglie
(`forgetting_success` e `relative_probability_change` della Fase A), stesso file di
uscita. Colonna "modificati": quante entità del modello l'intervento tocca.

| Braccio | Modificati | NLL/token X_forget | z (1-sided) | z (2-sided) | Frasi correlate oltre 5% | collaterale max | canario oltre 5% | canario max |
|---|---|---|---|---|---|---|---|---|
| **A** protocollo (prescribed), con canario | 0 (fermato al passo 1) | 4.6834 | 8.21 | no | 0/12 | 0.00% | 0/6 | 0.00% |
| **A** protocollo (prescribed), senza canario | 1536 (100% dello spazio LoRA) | 4.8576 | 7.70 | no | **12/12** | **345.67%** | 6/6 | 97.23% |
| **A** proiezione corretta, senza canario | 1536 (100%) | 5.0303 | 6.80 | no | 11/12 | 568.21% | 4/6 | 341.74% |
| **A** proiezione corretta, 300 passi, eta 3e-4 | 1536 (100%) | 6.727 | **0.81** | **sì** | 12/12 | 85335% | 6/6 | 99.99% |
| **B** attivazione, set minimo | non applicato (nessun set minimo) | — | — | — | — | — | — | — |
| **B** parametri, set minimo | non applicato (nessun set minimo) | — | — | — | — | — | — | — |

Ramo full-parameter (161113 parametri addestrabili, contro 1536):

| Braccio | Modificati | NLL/token | z (1-sided) | correlate oltre 5% | collaterale max | canario max |
|---|---|---|---|---|---|---|
| **A** protocollo, con canario | 0 (fermato al passo 1) | 2.2035 | 6.90 | 0/12 | 0.00% | 0.00% |
| **A** protocollo, senza canario | 161113 | 2.2035 (rollback) | 6.90 | 0/12 | 0.00% | 0.00% |
| **A** proiezione corretta, 100 passi | 161113 | 2.3798 | 6.62 | 4/12 | 26.40% | 17.40% |
| **A** proiezione corretta, 300 passi, eta 3e-3 | 161113 | 75.279 | −8.20 (sì) | 12/12 | 3972% | 21657% |
| **B** attivazione, set minimo | non applicato | — | — | — | — | — |

**Rapporto forgetting/collaterale dei due approcci.**

L'approccio gradient-based (A) può portare il forgetting a z ≤ 1: il prezzo è un
collaterale che va da 26% a 111009% della probabilità delle frasi correlate, e il
numero di parametri toccati è sempre **l'intero spazio addestrabile** (1536 o
161113): non c'è un modo per farlo con meno, perché l'ascent è una direzione
globale nello spazio dei pesi. Quando si alza `eta` abbastanza da proteggere il
retain (che è il caso a `eta = 0.1`: 0.00% di collaterale) l'intervento non fa
proprio niente, perché il rollback lo annulla.

L'approccio causale (B) identifica con precisione **dove** l'informazione passa,
ma non produce un intervento selettivo: il set minimo che azzera il segnale
causale azzera anche 3116 degli altri 3116 candidati sopravvissuti allo
scarto dei distruttivi, e la ricerca satura prima della soglia. Il numero di
componenti "modificati" che B produce è quindi **non applicabile**, non "pochi":
riportarlo come 1 senza dire che nello stesso passo il collaterale e il canario
esplodono sarebbe falso (i numeri di quel passo sono in §10.5).

**Nessuno dei due approcci soddisfa i due criteri insieme in questo
esperimento.** L'unico intervento che li soddisfa entrambi è non intervenire.

---

## 12. Test

`TESTS/test_unlearning.py` (37 test) e `TESTS/test_causal_unlearning.py`
(10 test). Esecuzione in isolamento:

```
$ python3 -m pytest TESTS/test_unlearning.py TESTS/test_causal_unlearning.py -q
47 passed in 106.00s (0:01:46)
```

Il tempo di parete (106.00 s) viene dalla riga di riassunto di pytest e non è
archiviato in `results/`: il conteggio dei 47 test, invece, è la cifra che
conta ed è riproducibile a ogni esecuzione.

- **Fase A, 22 test strutturali** (senza addestramento): le due formule di
  proiezione e le loro proprietà, criterio di successo unilaterale/bilaterale,
  variazione relativa di probabilità in log-spazio, `single_vs_distribution`,
  validazione dei parametri di `UnlearningConfig` (compreso `probe_tolerance`),
  riuso identico della Fisher, pavimento relativo e Fisher tutta zero, blocco dei
  parametri congelati sotto LoRA, STOP per proiezione degenere, rollback,
  dimezzamento di eta con riavvio da theta_0, disgiunza degli split **incluso il
  probe**.
- **Fase A, 9 test misurati**: precondizione di non-vacuita' (X_forget
  memorizzato), riproduzione della loss, protocollo prescritto, esito misurato del
  protocollo, difetto del primo passo, canario che ferma l'ascent, **canario come
  successo vacuo**, segno del passo sotto le due proiezioni, sweep esteso, stato
  addestrato.
- **Fase B, 6 test strutturali**: inventario coerente con `model.config()`, coppia
  clean/corrupted (stessa lunghezza, zero `<UNK>`, effettivamente diversa),
  sostituto nel vocabolario, ganci che ripristinano i metodi, azzeramento su
  sequenze più corte, pool ordinato per punteggio.
- **Fase B, 4 test misurati**: il patching individua le posizioni dell'entità e le
  due direzioni hanno segno opposto, il criterio di forgetting da solo non è
  soddisfacibile senza distruggere il modello, la ricerca vincolata non inventa
  localizzazioni, il riferimento è pulito e X_forget è memorizzato.

Esecuzione nella suite globale: `22 failed, 1326 passed, 6 errors`, con liste dei
test falliti identiche al baseline di §1 e +47 test passati.

Alcune asserzioni fissano **esiti negativi** (il protocollo prescritto non raggiunge
i criteri; nessuna localizzazione utilizzabile in B). Non sono test che "vogliono
fallire": impediscono a un cambiamento futuro di far passare sotto silenzio una
regressione, o di dichiarare un successo non misurato. Se un esito cambia, il
numero va rimisurato e questo report aggiornato.

---

## 13. Riproduzione

```bash
# Fase A, protocollo prescritto (con canario, esecuzione predefinita)
python3 experiments/mission1_unlearning/validate_unlearning.py
# ... e il confronto A2
python3 experiments/mission1_unlearning/validate_unlearning.py --no-probe
# diagnostica A4
python3 experiments/mission1_unlearning/validate_unlearning.py --first-order
# sweep A3
python3 experiments/mission1_unlearning/validate_unlearning.py --sweep-extended
# ramo full-parameter (aggiungere a qualsiasi comando)
python3 experiments/mission1_unlearning/validate_unlearning.py --no-lora

# Fase B
python3 experiments/mission1_unlearning/causal_unlearning.py
python3 experiments/mission1_unlearning/causal_unlearning.py --no-lora

# test
python3 -m pytest TESTS/test_unlearning.py TESTS/test_causal_unlearning.py -v
```

Output grezzi in `experiments/mission1_unlearning/results/`:

| file | contenuto |
|---|---|
| `output_lora.txt` / `unlearning_results.json` | protocollo LoRA con canario |
| `output_lora_senza_probe.txt` / `unlearning_results_senza_probe.json` | A2, confronto senza canario |
| `output_lora_first_order.txt` / `unlearning_results_first_order.json` | A4, segno del passo |
| `output_lora_sweep.txt` | sweep originale, tabella informativa di §9 |
| `output_lora_sweep_con_canario.txt` / `unlearning_results_sweep_con_canario.json` | stesso sweep con il canario attivo |
| `output_lora_sweep_esteso.txt` / `unlearning_results_esteso.json` | A3, 24 configurazioni LoRA |
| `output_full.txt` / `unlearning_results_full.json` | protocollo full-parameter con canario |
| `output_full_senza_probe.txt` / `unlearning_results_full_senza_probe.json` | A2, ramo full-parameter senza canario |
| `output_full_sweep.txt` | sweep originale, ramo full-parameter (non analizzato in §9) |
| `output_full_sweep_esteso.txt` / `unlearning_results_full_esteso.json` | A3, ramo full-parameter, 24 configurazioni |
| `output_causal_lora.txt` / `causal_unlearning_results.json` | Fase B completa, LoRA |
| `output_causal_full.txt` / `causal_unlearning_results_full.json` | Fase B completa, full-parameter |

`unlearning_results_esteso.json` contiene in testa la riga con canario alla
configurazione di default e poi `sweep_extended` con le 24 configurazioni; le
colonne `probe_*` del §11 sono lette da lì.

Tempo: ~50 s per run singola LoRA, ~35 s per quella full-parameter, ~2 min per
Fase B (CPU, NumPy float64). Tutti i numeri sono deterministici a parità di seed e
hardware.

---

## 14. Limiti dichiarati

1. **Test di fumo, non prova generale.** 97 frasi, un solo dominio (dialogo
   JARVIS in italiano), un modello da 162649 parametri (0,6% del modello di
   produzione), un seme, una sola frase da dimenticare.
2. **Non confrontabile con il modello di produzione.** Nessun esperimento sul
   modello da 2737806 parametri. I numeri di addestramento qui (loss 5.36 LoRA,
   0.002 full) non sono confrontabili con `17_MODEL_TRAINING_PROVENANCE.md`.
3. **Un solo esempio da dimenticare, una sola istanza.** Nessuna statistica su
   più esempi: non c'è una curva di rischio per frase, né un tasso di fallimento
   su un campione di richieste di cancellazione.
4. **Il ramo full-parameter è in overfitting** (loss 0.0020 su 46 finestre): le sue
   distribuzioni di controllo non sono confrontabili con quelle del ramo LoRA, e la
   soglia del criterio (1) in quel ramo (NLL ≥ 10.4375) è oltre il tetto di
   qualunque predittore del corpus (9.0269). Serve a isolare l'effetto "dimensione
   dello spazio addestrabile", non come alternativa migliore.
5. **Il controllo "mai visto" non è nella stessa distribuzione di X_forget.**
   X_forget ha 0 token fuori vocabolario, i controlli fino a 10. Lo sbilanciamento
   è conservativo ma non nullo.
6. **"Dimenticare" significa solo "portare la log-likelihood dentro la
   distribuzione delle frasi mai viste".** Non è cancellazione certificata: non si
   è provato che l'informazione sia irrecuperabile con ri-addestramento, prompt
   diversi o ispezione dei pesi.
7. **Con LoRA attiva l'unlearning tocca solo 1536 parametri.** Se
   l'informazione fosse memorizzata nei pesi congelati l'ascent non potrebbe
   rimuoverla. Qui la memorizzazione è negli adapter, quindi la misura è valida,
   ma la dipendenza è strutturale e va verificata caso per caso.
8. **Il pavimento sulla Fisher è una scelta, non una deduzione** (§7.2): cambia la
   metrica e quindi i numeri.
9. **Il rollback è una scelta non coperta dal testo del protocollo** (§7.3), e
   produce esiti "a vuoto" leggibili come successi se non si guarda lo `stop_reason`.
10. **La sostituzione clean/corrupted di Fase B usa un'entità già presente nel
    corpus**, perché il vocabolario non contiene altri nomi propri. Questo rende
    la coppia più simile e il patching più difficile da rilevare: direzione
    conservativa, dichiarata (§10.2).
11. **Il patching misura una dipendenza, non un luogo di memorizzazione**
    (§10.3). E l'ablation di Fase B è definita su posizioni di token, quindi non
    è applicabile a input di lunghezza diversa senza perdere effetto (§10.7).
12. **La ricerca del set minimo è greedy, non esaustiva**: 3123 candidati darebbero
    2^3123 configurazioni. Il risultato è un lower bound pragmatico, e lo è anche
    per la ricerca con il vincolo di collaterale.
13. **Il braccio a parametri di Fase B non è stato eseguito nel ramo
    full-parameter** (161113 valutazioni individuali superano il tetto di costo).
    È un limite di costo dichiarato, non un risultato.
14. **Il test statistico è indicativo**: `t` con 6 controlli e distribuzione
    normale al posto dei gradi di libertà esatti.
15. **Nessun confronto con framework esterni.** In questo file non è riportato
    alcun confronto con PyTorch, GPTQ, llama.cpp o altri: non è stato eseguito,
    quindi non è menzionato come confronto.
16. **L'unlearning concettuale è fuori scope**, come da consegna: non è definito
    nel dataset cosa costituisca un "concetto".

---

## 15. Conclusione, senza enfasi

Nella configurazione prescritta dalla missione — proiezione nella metrica di
Fisher con la formula data, eta = 1e-4, pavimento 1e-3, 100 passi, 3 riavvii —
**l'unlearning di un singolo esempio non ha successo su nessuno dei due criteri**,
e non è un caso limite: nessuna delle 16 configurazioni originali li soddisfa
insieme, e le 48 configurazioni dello sweep esteso che li soddisfano lo fanno solo
portando il collaterale a 10^4–10^5 per cento.

Cinque fatti misurati, non interpretazioni:

- **la formula di proiezione prescritta non annulla la componente di protezione
  con Fisher anisotropa**, e per gli ultimi 11 dei primi 15 passi produce un passo
  che peggiora la loss che dovrebbe far crescere (`delta_forget_1o` negativo,
  termine di proiezione 83–101 volte il gradiente di forget). Sotto la proiezione
  ortogonale esatta nessun passo è negativo;
- **il criterio di STOP della specifica è cieco al danno per frase**: la loss
  media su 24 frasi di retain si muove del +0.48% mentre 12/12 frasi correlate
  cambiano fino al 725%;
- **con più passi e learning rate più alti il forgetting si raggiunge** (z fino a
  −5.02 LoRA, −8.20 full-parameter) mantenendo il retain entro soglia: la
  conclusione "non raggiungibile" sarebbe stata falsa, e l'abbandono della
  ricerca al di qua di 100 passi era il vero limite della prima versione di
  questo lavoro;
- **un canario di 6 frasi con soglia sulla variazione massima tiene insieme i due
  criteri al passo 1**, e il motivo è che non lascia accadere niente: il rollback
  riporta il modello a theta_0. È una rete di sicurezza che funziona, non un
  metodo che funziona, e i due casi vanno tenuti distinti;
- **l'informazione sull'entità è localizzata in modo netto ma non selettivo**: il
  causal tracing individua con esattezza le due posizioni di token dell'entità e
  zero altrove, eppure 3116 componenti su 3123 sono distruttivi se ablate da
  soli, l'ablation satura sotto la soglia, e al livello dei parametri 1467
  candidati su 1536 lo sono. Su questo corpus l'informazione è ridondante e
  distribuita, ed è un risultato, non un fallimento.

L'unico intervento misurato che soddisfa entrambi i criteri è non intervenire. Se
questo report dovesse essere letto come "l'unlearning selettivo non funziona", la
lettura corretta è più stretta: **in questo setup, con questo modello, che ha
sovraffittato 46 finestre di un corpus di 91 frasi, rimuovere una frase senza
scotennare il resto non è riuscito con nessuno dei due metodi provati** — e la
prossima mossa utile non è una terza formula di proiezione, ma un corpus e una
scala in cui la memorizzazione non sia diffusa per costruzione.
