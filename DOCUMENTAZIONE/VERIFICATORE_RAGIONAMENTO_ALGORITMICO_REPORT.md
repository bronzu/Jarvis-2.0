# Verificatore di Ragionamento Algoritmico — Report

**Modulo:** `cognitive_layer/algorithmic_verifier.py`
**Test:** `TESTS/test_algorithmic_verifier.py`
**Data:** 2026-09-27
**Worktree:** `verificatore-di-ragionamento-algoritmico`

---

## 1. Cosa è stato implementato

Un organo che riceve una catena di ragionamento già prodotta dal modello e
restituisce una decisione strutturata. Non genera risposte nuove, salvo le
rigenerazioni di ramo previste dalla missione. Non genera output nuovo di suo.

| File | Righe | Contenuto |
|---|---|---|
| `cognitive_layer/__init__.py` | 16 | Pacchetto (non esisteva, vedi §8) |
| `cognitive_layer/algorithmic_verifier.py` | 1210 | STEP 1–5 |
| `TESTS/test_algorithmic_verifier.py` | 621 | STEP 6, 53 test |

### Integrazione con il codice esistente (nessuna modifica)

Il punto di innesto è l'agent loop, **prima** che la risposta torni all'utente:

```python
from cognitive_layer.algorithmic_verifier import verify

result = verify(input_text, model)   # model = NeuralModelInterface
if result.verification_failed:
    ...   # decisione dell'agent loop: FUORI scope di questa missione
return result.chosen_branch_text
```

Riusati così come sono, senza modificarli:

- `neural_core/model_interface.py:139` — `NeuralModelInterface.generate(prompt, **kwargs)`, il metodo di generazione esistente che ospita il `MicroTransformer`;
- `neural_core/model.py:156` — `Decoder.sample(logits, temperature, top_k, top_p, rng)`, la funzione di sampling già presente. **Non è stata creata una funzione di sampling nuova**;
- `neural_core/model_interface.py:160` — il valore `50` di `max_new_tokens`, riusato come `MAX_TOKENS_PER_BRANCH`;
- `config.yaml` → `audit.log_directory: ./audit_logs` e il formato JSON lines di `INFRASTRUCTURE/audit/audit_system.py` — stessa cartella, stesso formato per il log.

**Non toccati:** `neural_core/network.py`, `neural_core/trainer.py`, il meccanismo di generazione del modello, `INFRASTRUCTURE/`.

Dipendenze: solo libreria standard Python (`ast`, `re`, `dataclasses`, `typing`, `json`, `logging`, `pathlib`, `datetime`). Nessuna chiamata di rete. Nessun `eval()`/`exec()`: la valutazione aritmetica è fatta con un valutatore che cammina l'AST su una whitelist esplicita. NumPy è importato in modo differito, **solo** dentro `generate_branches`, per impostare il seed (vedi §8, punto 3).

---

## 2. Risultati dei 30 test della missione (STEP 6)

Il dataset dei 30 casi è fisso e incluso nel file di test, non generato a
runtime. Output grezso:

```
$ python3 -m pytest TESTS/test_algorithmic_verifier.py -v
============================= test session starts ==============================
platform darwin -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
collected 48 items

TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_01] PASSED [  2%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_02] PASSED [  4%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_03] PASSED [  6%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_04] PASSED [  8%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_05] PASSED [ 10%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_06] PASSED [ 12%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_07] PASSED [ 14%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_08] PASSED [ 16%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_09] PASSED [ 18%]
TESTS/test_algorithmic_verifier.py::test_arithmetic_sceglie_il_ramo_corretto[arith_10] PASSED [ 20%]
TESTS/test_algorithmic_verifier.py::test_codice_valido_supera_il_controllo_sintattico[code_valid_01] PASSED [ 22%]
TESTS/test_algorithmic_verifier.py::test_codice_valido_supera_il_controllo_sintattico[code_valid_02] PASSED [ 25%]
TESTS/test_algorithmic_verifier.py::test_codice_valido_supera_il_controllo_sintattico[code_valid_03] PASSED [ 27%]
TESTS/test_algorithmic_verifier.py::test_codice_valido_supera_il_controllo_sintattico[code_valid_04] PASSED [ 29%]
TESTS/test_algorithmic_verifier.py::test_codice_valido_supera_il_controllo_sintattico[code_valid_05] PASSED [ 31%]
TESTS/test_algorithmic_verifier.py::test_codice_non_valido_fallisce_il_controllo_sintattico[code_invalid_01] PASSED [ 33%]
TESTS/test_algorithmic_verifier.py::test_codice_non_valido_fallisce_il_controllo_sintattico[code_invalid_02] PASSED [ 35%]
TESTS/test_algorithmic_verifier.py::test_codice_non_valido_fallisce_il_controllo_sintattico[code_invalid_03] PASSED [ 37%]
TESTS/test_algorithmic_verifier.py::test_codice_non_valido_fallisce_il_controllo_sintattico[code_invalid_04] PASSED [ 39%]
TESTS/test_algorithmic_verifier.py::test_codice_non_valido_fallisce_il_controllo_sintattico[code_invalid_05] PASSED [ 41%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_01] PASSED [ 43%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_02] PASSED [ 45%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_03] PASSED [ 47%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_04] PASSED [ 50%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_05] PASSED [ 52%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_06] PASSED [ 54%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_07] PASSED [ 56%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_08] PASSED [ 58%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_09] PASSED [ 60%]
TESTS/test_algorithmic_verifier.py::test_ragionamento_aperto_e_othersempre_non_affidabile[open_10] PASSED [ 62%]
...
TESTS/test_algorithmic_verifier.py::test_integrazione_reale_con_microtransformer PASSED [ 97%]
TESTS/test_algorithmic_verifier.py::test_generazione_rumore_usa_davvero_i_seed PASSED [100%]

============================== 53 passed in 0.62s ==============================
```

### 2.1 Insieme A — 10 problemi aritmetici con soluzione nota

Criterio: `reliable=True` e `chosen_branch_text` contiene il risultato corretto. Ogni caso ha 3 rami: 2 corretti, 1 con riconciliazione fallita.

| # | Input | Risultato atteso | Esito |
|---|---|---|---|
| arith_01 | `Quanto fa 2 + 3?` | 5 | **pass** |
| arith_02 | `calcola 12 * 4` | 48 | **pass** |
| arith_03 | `Risolvi 7 - 2 =` | 5 | **pass** |
| arith_04 | `somma 15 + 27` | 42 | **pass** |
| arith_05 | `prodotto 6 * 7` | 42 | **pass** |
| arith_06 | `differenza 100 - 58` | 42 | **pass** |
| arith_07 | `quoziente 84 / 2` | 42 | **pass** |
| arith_08 | `esponente 3 ** 3` | 27 | **pass** |
| arith_09 | `2 + 3 * 4` | 14 | **pass** |
| arith_10 | `(4 + 6) / 2` | 5 | **pass** |

**10/10** (soglia richiesta: 8/10).

### 2.2 Insieme B — 10 snippet di codice

Criterio: `score_sintassi` discrimina correttamente. Ogni snippet è valutato isolatamente con `evaluate_branch(branch, "code")`.

| # | Snippet | Atteso | Esito |
|---|---|---|---|
| code_valid_01 | `def somma(a, b): return a + b` | `score_sintassi = 1` | **pass** |
| code_valid_02 | `class Punto: def __init__(self, x)` | `score_sintassi = 1` | **pass** |
| code_valid_03 | `import math` + `def area(r)` | `score_sintassi = 1` | **pass** |
| code_valid_04 | `for i in range(10): print(i)` | `score_sintassi = 1` | **pass** |
| code_valid_05 | `x = [1, 2, 3]` + `print(sum(x))` | `score_sintassi = 1` | **pass** |
| code_invalid_01 | `def somma(a, b)` (manca `:`) | `score_sintassi = 0` | **pass** |
| code_invalid_02 | `class Punto` (manca `:`) | `score_sintassi = 0` | **pass** |
| code_invalid_03 | `def f(:` (argomento malformato) | `score_sintassi = 0` | **pass** |
| code_invalid_04 | `x = = 5` | `score_sintassi = 0` | **pass** |
| code_invalid_05 | `if True` (manca `:`) | `score_sintassi = 0` | **pass** |

**10/10** (soglia richiesta: 9/10).

### 2.3 Insieme C — 10 prompt di ragionamento aperto / non deterministico

Criterio: il classificatore li marca `other` e l'output finale ha **sempre** `reliable=False`, indipendentemente dallo score di ramo. In tutti e 10 i casi lo score di ramo è `1.0` e il risultato resta `reliable=False`.

| # | Prompt | Dominio | Score ramo | `reliable` | Esito |
|---|---|---|---|---|---|
| open_01 | Quale musica è la migliore? | other | 1.0 | False | **pass** |
| open_02 | È giusto chiudere una fabbrica? | other | 1.0 | False | **pass** |
| open_03 | Cosa dire a un amico con parere sbagliato? | other | 1.0 | False | **pass** |
| open_04 | Qual è la cosa più bella immaginabile? | other | 1.0 | False | **pass** |
| open_05 | Perché si ha paura del buio da piccoli? | other | 1.0 | False | **pass** |
| open_06 | Cambiereesti professione domani? | other | 1.0 | False | **pass** |
| open_07 | È giusto che i genitori scelgano per i figli? | other | 1.0 | False | **pass** |
| open_08 | Descrivi la sensazione di svegliarsi presto d'estate | other | 1.0 | False | **pass** |
| open_09 | Capitano o più bravo: chi conta di più? | other | 1.0 | False | **pass** |
| open_10 | Ha senso lamentarsi della vita? | other | 1.0 | False | **pass** |

**10/10**.

### 2.4 Totali

| Insieme |Casi | Pass | Fail | Soglia | Esito |
|---|---|---|---|---|---|
| A — aritmetica | 10 | 10 | 0 | ≥ 8/10 | **pass** |
| B — codice | 10 | 10 | 0 | ≥ 9/10 | **pass** |
| C — ragionamento aperto | 10 | 10 | 0 | 10/10 | **pass** |
| **Totale missione** | **30** | **30** | **0** | — | **pass** |

Gli altri 23 test coprono i contratti del modulo: temperature e seed distinti, penalità fisse (-0.3 loop, -0.5 contraddizione, -1.0 riconciliazione), floor a 0.0, sintassi `None` su `other`, rigenerazione entro 2 round, rigenerazione dei soli rami scartati, tie-break sulla temperature più bassa, determinismo della decisione, contenuto del log JSONL, **assenza di effetti collaterali sullo stato RNG globale** (§3.1), e due test di integrazione con `MicroTransformer` reale.

### 2.5 Stato RNG globale

Cinque test dedicati, definiti in `TESTS/test_algorithmic_verifier.py`:

| Test | Cosa asserisce | Esito |
|---|---|---|
| `test_generate_branches_non_altera_lo_stato_rng_globale` | `np.random.get_state()` identico prima e dopo `generate_branches` (modello finto) | **pass** |
| `test_generate_branches_non_altera_lo_stato_rng_anche_su_modello_reale` | Stessa garanzia su `MicroTransformer` + `NeuralModelInterface` reali | **pass** |
| `test_stato_rng_ripristinato_anche_se_il_generatore_solleva` | Lo stato è ripristinato anche se `generate` solleva un'eccezione | **pass** |
| `test_stato_rng_ripristinato_anche_durante_le_rigenerazioni` | Stessa garanzia sui round di backtracking dello STEP 4 (verifica che i 9 round siano davvero avvenuti) | **pass** |
| `test_rami_restano_riproducibili_dopo_il_fix_rng` | Il ripristino non rompe la riproducibilità dei rami | **pass** |

**Verifica che i test mordano davvero**: con il fix temporaneamente rimosso, 4 dei 5 test falliscono. Il quinto fallirebbe anche senza fix in modo corretto (asserisce la riproducibilità, che deve valere in entrambi i casi).

```
$ python3 -m pytest TESTS/test_algorithmic_verifier.py -q     # fix rimosso
FAILED ...::test_generate_branches_non_altera_lo_stato_rng_globale
FAILED ...::test_generate_branches_non_altera_lo_stato_rng_anche_su_modello_reale
FAILED ...::test_stato_rng_ripristinato_anche_se_il_generatore_solleva
FAILED ...::test_stato_rng_ripristinato_anche_durante_le_rigenerazioni
4 failed, 49 passed in 0.64s
```

Implementazione (`_generate_one`, unico punto di generazione, usato sia da `generate_branches` sia dai round di rigenerazione):

```python
saved_state = np.random.get_state()
try:
    np.random.seed(seed)
    payload = model.generate(
        prompt,
        max_new_tokens=MAX_TOKENS_PER_BRANCH,
        temperature=temperature,
    )
finally:
    np.random.set_state(saved_state)
```

Il `finally` serve a coprire anche il caso di generatore che solleva: senza, un'eccezione lascerebbe il processo con il seed del ramo.

> **Avvertenza, ai sensi della regola 4 del progetto.** I 30 casi sono un **test di fumo del verificatore, non una prova generale**. Il dataset è di 30 esempi, fisso, e i rami sono testi scritti a mano, non generati dal modello. Ciò che è dimostrato è che la logica di punteggio, scarto e scelta seleziona il ramo corretto su questi 30 casi. **Non** è dimostrato che il modello `MicroTransformer` sappia risolvere aritmetica o scrivere codice, e il test non generalizza oltre questi casi.

---

## 3. Test in isolamento e nella suite globale

**Isolamento:** `53 passed in 0.62s`.

**Suite globale**, con i nuovi file:

```
$ python3 -m pytest TESTS -q
22 failed, 1332 passed, 76 warnings, 6 errors, 6 subtests passed in 142.99s (0:02:22)
```

**Baseline**, con i due file nuovi spostati fuori dal tree:

```
$ python3 -m pytest TESTS -q
22 failed, 1279 passed, 76 warnings, 6 errors, 6 subtests passed in 127.48s (0:02:07)
```

Confronto delle liste di fallimenti (ordinato e deduplicato):

```
$ comm -13 baseline.txt after_fix.txt   # SOLO DOPO (nuovi fallimenti)
$ comm -23 baseline.txt after_fix.txt   # SOLO PRIMA (risolti)
(nessuna output in entrambi i casi)
```

I due elenchi sono **identici**: nessuna regressione. `1332 - 1279 = 53` è esattamente il numero di test aggiunti.

Cause dei 22 fallimenti e dei 6 errori preesistenti, grezze:

```
 10 .../pathlib/__init__.py:771: FileNotFoundError
  7 TESTS/test_neural_core/test_gpu_parity.py:46: FileNotFoundError
  7 E   FileNotFoundError: ... /neural_core/data/model.pkl
  6 E   FileNotFoundError: No such file or directory: 'neural_core/data/train.txt'
  4 E   FileNotFoundError: ... /neural_core/data/train.txt
  3 TESTS/test_cross_split_regression.py:19: AssertionError
  1 TESTS/test_neural_core/test_dataset_leakage.py:28: AssertionError
  1 TESTS/test_kaggle_model_integration.py:64: AssertionError
  1 TESTS/test_kaggle_model_integration.py:55: AssertionError
  1 TESTS/test_kaggle_model_integration.py:40: FileNotFoundError
```

Sono tutti file di dataset e checkpoint assenti in questo worktree
(`neural_core/data/train.txt`, `validation.txt`, `test.txt`, `model.pkl`) e i
test GPU parity che non hanno una GPU. Nessuno riguarda `cognitive_layer`.

Nota di onestà: in una delle tre esecuzioni il conteggio è stato `23 failed`
invece di `22`. La differenza è `TESTS/test_continual_learning.py::TestFisherLatency::test_fisher_latency_full_vs_micro`, un test basato su soglie temporali e quindi instabile. Il confronto finale delle liste lo conferma: l'insieme dei fallimenti non cambia.

---

## 4. Dimostrazione end-to-end sul modello reale

`MicroTransformer` + `Tokenizer` + `NeuralModelInterface` reali, non addestrati,
sul path di log predefinito:

```
$ python3 -c "... verify('Quanto fa 2 + 3?', iface) ..."
LOG FILE: .../audit_logs/algorithmic_verifier.jsonl
dominio: arithmetic | reliable: False | chosen: branch_0 | score: 0.0 | failed: True | rounds: 2
temperature/seed: [(0.2, 20262927, 50), (0.6, 20262928, 50), (1.0, 20262929, 50)]

--- ultimo record JSONL ---
{
  "input_text": "Quanto fa 2 + 3?",
  "domain": "arithmetic",
  "reliable": false,
  "chosen_branch_id": "branch_0",
  "chosen_branch_score": 0.0,
  "discarded_branches": [
    { "id": "branch_0", "score": 0.0,
      "motivo_scarto": "Riconciliazione fallita: '3 + 5' = 8, ma il ramo dichiara 3." },
    { "id": "branch_1", "score": 0.0,
      "motivo_scarto": "Nessuna espressione aritmetica trovata nel ramo." },
    { "id": "branch_2", "score": 0.0,
      "motivo_scarto": "Nessuna espressione aritmetica trovata nel ramo." }
  ],
  "regeneration_rounds_used": 2,
  "verification_failed": true,
  "timestamp": "2026-09-27T19:20:07.303537+00:00"
}
```

Il modello non addestrato produce rumore, quindi tutti e 3 i rami vengono
scartati, si esauriscono 2 round di rigenerazione e il modulo espone
`verification_failed=true` **invece di restituire silenziosamente una risposta**.
È esattamente il comportamento richiesto dallo STEP 4. Le temperature sono
`[0.2, 0.6, 1.0]`, i seed sono distinti e presenti nel log, `token_count=50`
corrisponde a `MAX_TOKENS_PER_BRANCH`.

---

## 5. Domini su cui il verificatore È affidabile

Affidabilità qui significa: **il modulo è in grado di stabilire un errore
oggettivo con un controllo formale**, non che il modello risponda bene.

1. **Aritmetica di base su operatori whitelisted.** `+ - * / // % **` e
   unario `+ -`, costanti numeriche. Il controllo è forte perché non è
   euristico: l'espressione viene passata a `ast.parse` in modalità `eval`, si
   verifica che l'AST contenga **solo** nodi whitelisted (`Expression`,
   `Constant`, `BinOp`, `UnaryOp` e i relativi operatori — nessun `Call`,
   nessun `Name`, nessun `Attribute`), e il valore viene ricalcolato da un
   valutatore che cammina l'AST. Se il valore ricalcolato non coincide con
   quello dichiarato dal ramo, la penalità è -1.0 e il ramo viene scartato.
   Qui il modulo può affermare "questo numero è sbagliato" con fondamento.

2. **Sintassi del codice — solo la forma.** `ast.parse` sul codice estratto.
   Il modulo può affermare "questo codice non è parsabile" e basta. **Non**
   afferma nulla sulla correttezza semantica, sulla logica, o sul comportamento
   a runtime: un codo che fa il contrario di ciò che sembra ottiene
   `score_sintassi = 1`.

   **Semplificazione dichiarata:** l'esecuzione reale del codice generato è
   fuori scope per questa missione, come specificato nello STEP 3a. Verificare
   la correttezza semantica richiederebbe un sandbox di esecuzione, e potrebbe
   essere oggetto di una missione separata. Finché quel controllo non esiste,
   `reliable=True` sul dominio `code` significa soltanto "il codice è
   parsabile".

---

## 6. Domini su cui il verificatore NON è affidabile

Il principio: **`score_ramo` alto NON implica che il contenuto sia corretto.**

1. **Ragionamento in linguaggio naturale aperto.** Non esiste un parser che
   dica se "La musica rock è la migliore" sia vero. Il classificatore porta
   questi casi su `other`, dove `score_sintassi = None` e non penalizza: il
   punteggio di un ramo è quindi **1.0 per costruzione**, indipendentemente da
   quanto il testo sia sbagliato. Il risultato viene comunque marcato
   `reliable=False`.

2. **Giudizi di valore ed etici.** "È giusto chiudere una fabbrica che dà
   lavoro a cento persone?" non ha una risposta verificabile: la verifica
   richiederebbe fissare un sistema di valori, cosa che è una scelta etica, non
   un controllo algoritmico. Nessun parser può farlo. Il modulo non ci prova e
   restituisce `reliable=False`.

3. **Senso comune.** "Perché la gente ha paura del buio da piccoli?" richiede
   conoscenza del mondo. Una frase può essere grammaticalmente e
   sintatticamente perfetta, priva di ripetizioni e di contraddizioni, e
   comunque falsa rispetto ai fatti. Il modulo non ha accesso al mondo, solo
   alla forma del testo.

4. **Inferenza causale non formalizzabile.** "Se gli cambiassi mestiere domani,
   cosa farebbe?" richiede un modello causale del mondo. Il STEP 3b ricalcola
   solo l'aritmetica dichiarata nel testo: non valuta la catena causale.

5. **Qualsiasi dominio in cui la correttezza non è verificabile tramite un
   parser o un ricalcolo deterministico.** In generale: su questi domini i
   controlli di STEP 3b misurano **solo l'assenza di ripetizioni e di
   contraddizioni superficiali rilevate dalle regex**. È un **controllo di
   forma, non di verità**. Un ramo con `score_ramo = 1.0` su `other` non
   significa che la risposta sia giusta: significa che il testo non conteneva
   due righe identiche consecutive e nessuna delle coppie di marker
   contraddittori previste dalla lista.

Per questo il dominio `other` forza `reliable=False` **anche quando tutti e 3
i rami sopravvivono con punteggio massimo**, ed è esattamente ciò che verificano
i 10 casi dell'insieme C (score 1.0, `reliable=False`).

---

## 7. Limiti noti dell'euristica di STEP 3b

I controlli di coerenza tra step sono basati su regex e su una lista **fissa**
di marker lessicali. Limiti:

1. **Falsi negativi su sinonimi e riformulazioni.** È il limite principale. Una
   contraddizione espressa con parole non presenti in `CONTRADICTION_MARKER_PAIRS`
   non viene rilevata. Esempio: un ramo che dice "il risultato è 42" e poi
   "in effetti il conto non torna" **non** viene marcato come contraddizione,
   perché "non torna" non è nella lista. Lo stesso vale per qualsiasi
   riformulazione non prevista.
2. **Falsi positivi da sovrapposizione lessicale.** Le coppie nella lista
   possono sovrapporsi: "ma in realtà non" contiene "in realtà non". Il
   controllo deduplica per coppia `(step antecedente, step successivo)`, quindi
   la stessa contraddizione non viene contata due volte, ma due coppie diverse
   che puntano a step diversi **sì**. Una singola contraddizione può quindi
   costare più di -0.5 se il testo contiene entrambe le varianti.
3. **Il loop detection è solo locale.** Rileva due step **consecutivi**
   identici dopo normalizzazione. Non rileva un loop di 3 righe che si
   ripete su un ciclo più lungo, né la ripetizione di una singola parola. Due
   step identici separati da un passo diverso non vengono contati.
4. **La contraddizione sui valori finali confronta solo l'ultimo numero.** La
   regex `\d+(\.\d+)?` prende l'ultimo numero citato in uno step che contiene
   un marker. Non distingue un numero che è un risultato da un numero che è un
   dato di partenza: `quindi 42` viene trattato come un valore dichiarato
   anche se 42 fosse un anno.
5. **Nessun controllo di coerenza semantica tra step.** Due step possono
   essere logicamente incompatibili senza contenere nessuno dei marker previsti.
   Il modulo non lo rileva, perché non ha una rappresentazione del significato.
6. **La riconciliazione confronta solo l'ultima espressione.** Se un ramo
   contiene più calcoli, viene ricalcolato solo l'ultimo, e confrontato con
   l'ultimo numero citato. Un errore in un calcolo intermedio non viene
   rilevato.
7. **Se la riconciliazione non è possibile non c'è penalità.** Se non si
   trova un'espressione aritmetica ricalcolabile, o se il ramo non cita
   nessun valore, la penalità -1.0 **non** viene applicata: il controllo è
   saltato, non superato. Il log registra `riconciliazione_eseguita: false` con
   il motivo, quindi la distinzione resta ispezionabile, ma il punteggio non la
   riflette.
8. **Estrazione del codice grossolana.** `extract_code` prende il contenuto
   del blocco ``` se presente, altrimenti l'intero testo. Un ramo che
   avvolge il codice in prosa fallisce il controllo sintattico anche se il
   codice è corretto.
9. **Falsi negativi nella whitelist aritmetica.** Un ramo che contiene una
   chiamata a funzione anywhere viene marcato come "codice non aritmetico
   camuffato" e prende 0, anche se contiene anche un'espressione aritmetica
   valida. È una scelta conservativa: preferisce scartare un ramo ambiguo
   piuttosto che accettarne uno non verificabile.
10. **Il limite di sicurezza sull'esponente** (64) è una scelta ingegneristica
    non dettata dalla missione, necessaria per non esporre il processo a
    potenze enormi su testo non fidato. Espressioni con esponente legittimamente
    superiore a 64 non vengono ricalcolate e quindi non vengono penalizzate.

---

## 8. Punti da chiarire con l'utente

Nessuno di questi è stato deciso autonomamente dove la missione era silente;
quelli che ho dovuto scegliere per far funzionare il modulo sono marcati come
scelte, non come decisioni di specifica.

1. **Il path `cognitive_layer/` non esisteva.** `ARCHITETTURA.md` sez. 8 non
   definisce un layer "cognitive_layer": i layer sono NEURAL CORE, COGNITIVE
   CORE, EXECUTIVE SYSTEM, EXPANSION SYSTEM, INFRASTRUCTURE, INTERFACE. Il
   contenuto del modulo (verifica della risposta) ricade nel COGNITIVE CORE,
   che ha già `cognitive_core/`. Ho creato `cognitive_layer/` con `__init__.py`
   perché il path è scritto esplicitamente nella missione, ma serve conferma:
   **`cognitive_layer/` come layer nuovo, oppure il modulo dentro
   `cognitive_core/`?**

2. **Missione 1 e Missione 2 non esistono nel repository.** Verificato su
   tutti i worktree: nessun modulo, nessun riferimento. Lo STEP 5 chiedeva di
   "riusare la stessa cartella/formato" di M1/M2. In assenza di un modello,
   ho riusato l'unica convenzione JSONL già presente nel progetto:
   `audit_logs/algorithmic_verifier.jsonl`, stessa cartella di
   `config.yaml → audit.log_directory` e di `INFRASTRUCTURE/audit/audit_system.py`,
   stesso formato JSON lines con `ensure_ascii=False`. Se M1/M2 scelgono
   un'altra cartella, il percorso va riallineato.

3. **Il "metodo di generazione già esistente del MicroTransformer" non esiste.**
   `MicroTransformer` (`neural_core/network.py:1074`) non ha un metodo
   `generate`. Il punto d'ingresso esistente è
   `NeuralModelInterface.generate` (`neural_core/model_interface.py:139`), che
   ospita il `MicroTransformer` e campiona tramite `Decoder.sample`
   (`neural_core/model.py:156`). `generate_branches(prompt, model, ...)` riceve
   quindi l'`NeuralModelInterface`. Va bene, o serve un'altra firma?

4. **Il seed richiede NumPy, in tensione con "solo libreria standard".**
   `Decoder.sample` accetta un `rng`, ma `NeuralModelInterface.generate` non lo
   espone e usa il generatore globale NumPy. Per rendere i rami riproducibili
   **senza modificare** `network.py`, `trainer.py` o `model_interface.py`,
   `generate_branches` chiama `np.random.seed(seed)` prima di ogni
   generazione. NumPy è comunque il requisito matematico del progetto
   (regola 1). Va bene così, oppure è accettato un parametro `seed` in
   `NeuralModelInterface.generate`?

   **Stato RNG globale.** Il seeding dell'RNG globale è ora racchiuso: lo stato
   precedente viene salvato con `np.random.get_state()` e ripristinato con
   `np.random.set_state()` in un `finally` subito dopo la generazione, per
   ciascuno dei 3 rami e per ciascun round di rigenerazione. `generate_branches`
   non lascia quindi effetti collaterali sul processo al ritorno, e la
   riproducibilità dei rami resta garantita dal seed fisso registrato nel log.
   Coperto da 5 test dedicati (§2.5), di cui 4 verificati fallire senza il fix.

5. **`max_tokens_per_branch` non è una costante nel progetto**, ma un letterale
   `50` a `model_interface.py:160`. Ho riusato il valore e reso esplicito
   `MAX_TOKENS_PER_BRANCH = 50`. Conferma che sia il riferimento giusto.

6. **Casi non specificati, risolti con una scelta (da confermare):**
   - **Ramo `code` senza codice parsabile, o ramo `arithmetic` senza
     espressione** → `score_sintassi = 0`. Ho preferito non premiare un ramo
     non valutabile; l'alternativa (non penalizzare) lascerebbe passare
     risposte prive di contenuto verificabile.
   - **`discarded_branches`** contiene tutti i rami sotto soglia. Quando
     `verification_failed=True`, il `chosen_branch_id` punta a uno di essi:
     la sovrapposizione è intenzionale ed è ciò che la missione chiede
     ("restituisce il ramo con score più alto tra quelli scartati").
   - **`verification_failed` è esposto e non gestito**, come specificato.

7. **Falsi negativi noti della lista marker di STEP 2.** La lista
   `MATH_KEYWORDS` non contiene "di" né "e", quindi `prodotto di 3 e 4`
   classifica `other`. È una conseguenza diretta della lista fissa prescritta,
   non un bug: va bene, o la lista va estesa?

8. **`typing.Literal` non è usato da nessun modulo del repository** (0
   occorrenze; lo stile del progetto usa `class X(str, Enum)`). La firma
   `classify_domain` richiede però esplicitamente
   `Literal["arithmetic", "code", "other"]`. Ho seguito la missione. Conferma
   che `Literal` sia accettato in questo layer, o si preferisce un Enum per
   coerenza con `cognitive_core/runtime/verification.py:50`.

---

## 9. Conformità ai vincoli

| Vincolo | Stato |
|---|---|
| Solo libreria standard Python | OK, con l'eccezione NumPy in `generate_branches` (vedi §8.4) |
| Nessuna dipendenza esterna | OK |
| Nessuna chiamata di rete | OK, verificato per scansione |
| Nessun `eval()`/`exec()` su input non fidato | OK, valutatore AST su whitelist |
| Decisione deterministica | OK, test `test_decisione_deterministica_a_parita_di_input` |
| Nessun effetto collaterale sull'RNG globale | OK, 5 test dedicati (§2.5), 4 verificati fallire senza il fix |
| Non modifica `network.py`, `trainer.py`, generazione | OK, `git status` mostra solo 2 file nuovi |
| Riutilizzo della funzione di sampling esistente | OK, `Decoder.sample` via `NeuralModelInterface.generate` |
| Riutilizzo del valore `max_tokens` esistente | OK, 50 da `model_interface.py:160` |
| Riutilizzo della convenzione di logging | OK con riserva, `audit_logs` (§8.2) |
| `other` è il default esplicito | OK, `DEFAULT_DOMAIN = "other"` in un `return` esplicito |
| Test in pytest, isolamento + suite globale | OK, 48/48 e nessuna regressione |
| Nessun confronto con framework esterni | OK, nessun benchmark di confronto eseguito, nessun confronto dichiarato |
| Nessun linguaggio gonfiato | OK |
