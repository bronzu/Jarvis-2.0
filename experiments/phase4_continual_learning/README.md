# Phase 4 — Validazione Continual Learning (Micro-EWC con LoRA)

Misura i quattro criteri dichiarati nel protocollo di Continual Learning e
produce i numeri citati in
`DOCUMENTAZIONE/technical-acquisition-package/17_MODEL_TRAINING_PROVENANCE.md`
sezione 13.7.4.

## Corpus

I due task sono **reali e tratti dal repository**, non frasi sintetiche, e sono
deterministici (nessuna generazione casuale):

| | Task A | Task B |
|---|---|---|
| contenuto | prosa tecnica italiana | codice sorgente Python |
| sorgenti | 4 file in `DOCUMENTAZIONE/` | 8 file in `neural_core/` |
| parole | 16.953 | 11.917 |
| campioni | 1.059 (795 train / 264 val) | 744 (558 train / 186 val) |

Estrazione in `corpus.py`: il markdown viene ripulito di blocchi di codice,
tabelle e markup; il Python viene ripulito di commenti e docstring.

## Esecuzione

```bash
# criteri (a) memoria e (b) latenza — configurazione di produzione, ~2 min
python3 experiments/phase4_continual_learning/validate.py --criterion a

# criteri (c) ritenzione e (d) apprendimento — ~70 min su CPU
python3 experiments/phase4_continual_learning/validate.py --criterion cd

# diagnostica: il gap di (d) e' un limite di capacita' o di ottimizzazione?
python3 experiments/phase4_continual_learning/validate.py --criterion diag
```

Tutti i comandi accettano `--json <path>` per salvare i risultati strutturati.

## Cosa misura ogni criterio

- **(a) memoria** — `fisher_matrix` + `optimal_weights` nella versione
  full-parameter (2,7M parametri) contro Micro-EWC/LoRA (rank 4 e 16). Tre
  misure indipendenti: byte esatti degli array, picco `tracemalloc`, e **RSS di
  processo** con una variante per subprocesso pulito (`rss_probe.py`). Le tre
  danno risposte molto diverse ed e' la distinzione piu' importante del
  criterio: la riduzione del 99,78% vale per le strutture persistenti, non per
  la RAM del processo (~15%, intervallo 9,57-22,02% su 7 esecuzioni).
- **(b) latenza** — `compute_fisher_matrix` su 50 campioni, minimo di 3
  ripetizioni dopo warm-up.
- **(c) ritenzione** — addestramento su A, misura di loss e accuratezza sullo
  split di validazione di A, poi addestramento su B, poi rimisura. Eseguito
  con `ewc_lambda` 100 e con `ewc_lambda` 0, partendo dallo stesso checkpoint.
- **(d) apprendimento** — accuracy e loss sul task B con il solo adapter LoRA
  contro il fine-tuning completo, a parita' di dati, epoche, batch, seed e
  replay.

Gli scenari sono calcolati una volta sola e riusati: il run "LoRA + EWC
attivo" e' contemporaneamente il confronto di (c) e il lato LoRA di (d), perche'
ricalcolarlo produrrebbe due numeri diversi solo per rumore di schedulazione e
renderebbe il gap di (d) non confrontabile.

## Struttura

| file | ruolo |
|---|---|
| `corpus.py` | estrazione dei due task dal repository |
| `validate.py` | esecuzione e stampa dei criteri |
| `rss_probe.py` | sonda RSS di processo, una variante per esecuzione |
| `results/` | output grezzo delle esecuzioni |

## Limiti dichiarati

I criteri (c) e (d) **non sono raggiunti** sulla misura corrente: ritenzione in
accuracy −61,18% (soglia 2%) e gap di −17,71 punti percentuali sul task nuovo
(soglia 2 pp). I numeri e la loro interpretazione sono in 13.7.4 e 13.7.5.
