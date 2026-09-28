"""
Braccio comparativo di unlearning causale (Fase B della Missione 1).

Lo stesso corpus, lo stesso tokenizer, lo stesso X_forget, le stesse 12 frasi
correlate e lo stesso holdout della Fase A (gradient-based). Cambia il metodo:
invece di spostare tutti i pesi lungo una direzione di ascent, qui si cerca
DOVE l'informazione di X_forget e' causalmente necessaria, e si interviene solo
lì.

Pipeline:

  B0  inventario dell'architettura REALE, letta dal codice e non dal diagramma;
  B1  coppia clean/corrupted per X_forget;
  B2  causal tracing con activation patching sul flusso residuo, per sito e per
      posizione, e poi per singola coordinata;
  B3  set minimo di componenti, per accumulo greedy in ordine di punteggio;
  B4  ablation applicata SOLO sul set minimo (nessun'altra modifica ai pesi),
      piu' un braccio parallelo a granularita' di parametro per poter
      confrontare "quanti parametri" con la Fase A;
  B5  retest con le metriche identiche della Fase A piu' il canario;
  B6  verdetto sulla localizzazione.

Nessuna affermazione sulla causalita' e' possibile oltre a cioe' che segue:
il patching misura l'effetto di SOSTITUIRE un'attivazione, quindi una
dipendenza del comportamento dall'attivazione in quel punto. Non distingue
"informazione memorizzata li'" da "informazione che li' passa". E' dichiarato
nel report, non nascosto.

Solo NumPy + standard library, come nel resto della missione.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus_unlearning as corpus  # noqa: E402
import validate_unlearning as validate  # noqa: E402

from neural_core.unlearning import (  # noqa: E402
    SelectiveUnlearner, UnlearningConfig, forgetting_success,
    relative_probability_change,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Soglia del criterio (1): la stessa della Fase A, z <= 1 sul riferimento
#: holdout. Serve a rendere confrontabili le due braccia.
Z_TOLERANCE = 1.0
#: Soglia del criterio (2) e del canario, come in Fase A.
COLLATERAL_TOLERANCE = 0.05
#: Quanti passi consecutivi senza variazione della NLL contano come saturazione.
PLATEAU_WINDOW = 5
#: Tetto alla dimensione del set minimo cercato per accumulo greedy. Il pool
#: di candidati di Fase B contiene 3 siti x 16 posizioni x 64 coordinate = 3072
#: componenti, piu' 48 componenti a posizione e 3 a sito: 3123 in tutto. Il tetto
#: e' alto apposta, per poter concludere "non basta" su TUTTI i candidati e non
#: su un sottoinsieme arbitrario.
MAX_COMPONENTS = 4096


# --------------------------------------------------------------------------
# B0: inventario dell'architettura reale
# --------------------------------------------------------------------------

def architecture_inventory(model, trainer, log=print):
    """
    Inventario letto dagli oggetti del modello, non da una specifica scritta.

    Il numero di "siti" patchabili e' derivato dalla struttura effettiva: un
    sito per ogni blocco (ingresso del blocco) piu' il flusso residuo finale
    (ingresso della LayerNorm di uscita). Ogni sito e' una matrice
    `(batch, seq_len, d_model)`, cioe' l'attivazione residua a quel punto.
    """
    config = model.config()
    names = trainer._fisher_param_names()
    indices = trainer._trainable_indices()
    params = model.get_parameters()
    trainable = {int(i) for i in indices}

    log("=== B0: ARCHITETTURA REALE (da neural_core/network.py) ===")
    log(f"classe                : {type(model).__name__}")
    log(f"vocab_size            : {model.vocab_size}")
    log(f"d_model               : {config['d_model']}")
    log(f"num_heads             : {config['num_heads']}")
    log(f"head_dim (d_model/heads): {config['d_model'] // config['num_heads']}")
    log(f"d_ff                  : {config['d_ff']}")
    log(f"num_layers            : {config['num_layers']}")
    log(f"max_len (context)     : {config['max_len']}")
    log(f"lora_rank             : {config['lora_rank']}")
    log(f"dtype                 : {config['dtype']}")
    log(f"parametri totali      : {int(sum(np.asarray(p).size for p in params))}")
    log(f"parametri addestrabili: {len(indices)} tensori, "
        f"{int(sum(np.asarray(params[i]).size for i in indices))} valori")
    log("blocchi (pre-norm, residuo): x = x + Attn(LN(x)); x = x + FF(LN(x))")
    log("-" * 78)
    log(f"{'#':>3} {'parametro':22s} {'forma':>18s} {'addestrabile':>13s}")
    for slot, (name, idx) in enumerate(zip(names, indices)):
        shape = tuple(np.asarray(params[idx]).shape)
        log(f"{slot:3d} {name:22s} {str(shape):>18s} {'si':>13s}")
    n_frozen = len(params) - len(trainable)
    log(f"tensori congelati     : {n_frozen} "
        f"(embedding, norm, attenzione, proiezione di uscita: vedi rete)")

    sites = [("ingresso del blocco %d" % k) for k in range(config["num_layers"])]
    sites.append("flusso residuo finale (ingresso della LayerNorm di uscita)")
    log("")
    log(f"siti patchabili sul flusso residuo: {len(sites)}")
    for k, label in enumerate(sites):
        log(f"  sito {k}: {label}")

    return {
        "config": config,
        "head_dim": config["d_model"] // config["num_heads"],
        "n_parameters_total": int(sum(np.asarray(p).size for p in params)),
        "n_parameters_trainable": int(sum(np.asarray(params[i]).size for i in indices)),
        "n_trainable_tensors": len(indices),
        "trainable_names": list(names),
        "trainable_shapes": [tuple(np.asarray(params[i]).shape) for i in indices],
        "n_sites": len(sites),
        "site_labels": sites,
    }


# --------------------------------------------------------------------------
# B1: coppia clean / corrupted
# --------------------------------------------------------------------------

#: Entita' target di X_forget: il nome proprio del creatore.
TARGET_ENTITY = "Riccardo Bronzuoli"
#: Sostituto: due token del vocabolario, entrambi presenti, quindi zero token
#: fuori vocabolario. Nel vocabolario di 729 token NON esiste alcun altro nome
#: proprio di persona (verificato: nessuna alternativa "Lorenzo", "Rossi",
#: "Bianchi" e simili e' nel vocabolario), quindi un nome nuovo avrebbe
#: introdotto token <UNK> e la differenza misurata sarebbe stata lessicale,
#: non semantica. "Neural Core" e' un'entita' gia' presente nel corpus: e' una
#: scelta conservativa, perche' rende la coppia clean/corrupted PIU' simile e
#: quindi l'effetto del patching piu' difficile da rilevare, non piu' facile.
REPLACEMENT_ENTITY = "Neural Core"


def build_corrupted(clean, target=TARGET_ENTITY, replacement=REPLACEMENT_ENTITY):
    """Stessa frase con l'entita' target sostituita, resto identico."""
    if target not in clean:
        raise ValueError(f"L'entita' target {target!r} non e' in X_forget.")
    return clean.replace(target, replacement)


def validate_pair(unlearner, clean, corrupted, log=print):
    """
    Controlli sulla coppia: stesso numero di token, zero token fuori
    vocabolario su entrambe, e una differenza di log-likelihood non nulla
    (se fosse zero la coppia non misurerebbe nulla).
    """
    clean_ll = unlearner.loglikelihood(clean)
    corrupted_ll = unlearner.loglikelihood(corrupted)
    n_unk_clean = unlearner.tokenizer.count_unknown(clean)
    n_unk_corrupted = unlearner.tokenizer.count_unknown(corrupted)
    same_length = clean_ll["n_tokens"] == corrupted_ll["n_tokens"]
    log("")
    log("=== B1: COPPIA CLEAN / CORRUPTED ===")
    log(f"clean    : {clean}")
    log(f"corrupted: {corrupted}")
    log(f"sostituzione: {TARGET_ENTITY!r} -> {REPLACEMENT_ENTITY!r}")
    log(f"token    : clean {clean_ll['n_tokens']} | corrupted "
        f"{corrupted_ll['n_tokens']} | stessa lunghezza: {same_length}")
    log(f"<UNK>    : clean {n_unk_clean} | corrupted {n_unk_corrupted}")
    log(f"loglikelihood/token: clean {clean_ll['mean_loglikelihood']:.4f} | "
        f"corrupted {corrupted_ll['mean_loglikelihood']:.4f} | "
        f"delta {corrupted_ll['mean_loglikelihood'] - clean_ll['mean_loglikelihood']:+.4f}")
    if not same_length:
        raise AssertionError(
            "clean e corrupted devono avere lo stesso numero di token: "
            "altrimenti il confronto delle continuazioni non e' allineato."
        )
    if n_unk_clean or n_unk_corrupted:
        raise AssertionError(
            "La coppia non deve contenere token fuori vocabolario: la "
            "differenza misurata sarebbe lessicale, non semantica."
        )
    return {
        "clean": clean,
        "corrupted": corrupted,
        "n_tokens": clean_ll["n_tokens"],
        "unk_clean": n_unk_clean,
        "unk_corrupted": n_unk_corrupted,
        "clean_loglik_per_token": clean_ll["mean_loglikelihood"],
        "corrupted_loglik_per_token": corrupted_ll["mean_loglikelihood"],
    }


# --------------------------------------------------------------------------
# B2: activation patching sul flusso residuo
# --------------------------------------------------------------------------

class ResidualHooks:
    """
    Registra e sostituisce le attivazioni sul flusso residuo del modello.

    I ganci sono applicati ai metodi `forward` dei blocchi e della LayerNorm
    finale: sono il punto in cui l'attivazione residua esiste gia' come
    variabile autonoma, quindi sostituirla non richiede di toccare il codice del
    modello. Nulla in `neural_core/` viene modificato.

    `record[idx] = x` conserva l'attivazione di ogni sito a ogni esecuzione.
    `patch[idx] = y` sostituisce l'ingresso del sito idx con y.
    `zero[idx] = True` azzera l'attivazione residua del sito idx.
    `zero_coords[idx] = {(pos, coord)}` azzera solo le coordinate indicate.
    """

    def __init__(self, model):
        self.model = model
        self.record = {}
        self.patch = {}
        self.zero = {}
        self.zero_coords = {}
        self._originals = None

    def _wrapper(self, idx, original):
        def hook(x, *args, **kwargs):
            x = np.asarray(x)
            self.record[idx] = x.copy()
            if idx in self.zero:
                x = np.zeros_like(x)
            if idx in self.zero_coords:
                x = x.copy()
                width = x.shape[1]
                for (pos, coord) in self.zero_coords[idx]:
                    # Su una sequenza piu' corta della posizione indicata
                    # l'ablation non e' definita: viene semplicemente saltata,
                    # e il fatto e' dichiarato nel report (un intervento su
                    # posizioni di token non e' applicabile a input di lunghezza
                    # diverse senza perdere parte dell'effetto).
                    if pos < width:
                        x[:, pos, coord] = 0.0
            if idx in self.patch:
                x = np.asarray(self.patch[idx], dtype=x.dtype)
            return original(x, *args, **kwargs)
        return hook

    def __enter__(self):
        self._originals = []
        for idx, layer in enumerate(self.model.layers):
            original = layer.forward
            self._originals.append((layer, original))
            layer.forward = self._wrapper(idx, original)
        norm = self.model.norm
        original_norm = norm.forward
        self._originals.append((norm, original_norm))
        norm.forward = self._wrapper(len(self.model.layers), original_norm)
        return self

    def __exit__(self, *exc):
        if self._originals is not None:
            for target, original in self._originals:
                target.forward = original
        self._originals = None
        self.record.clear()
        self.patch.clear()
        self.zero.clear()
        self.zero_coords.clear()
        return False


def _nll_per_token(unlearner, inputs, targets):
    """NLL per token su una finestra esplicita, senza toccare i pesi."""
    total = unlearner._forward_nll_sum(inputs, targets)
    return total / float(np.asarray(targets).size)


def collect_activations(unlearner, inputs, hooks):
    """
    Esegue un forward pulito dentro il contesto di ganci gia' aperto e
    restituisce le attivazioni di ogni sito.

    Non apre un contesto proprio: i ganci vanno installati una volta e tenuti
    aperti per tutta la scansione, altrimenti ogni chiamata ripristinerebbe i
    metodioriginali mentre il chiamante si aspetta che restino istallati.
    """
    hooks.record.clear()
    unlearner._forward_nll_sum(inputs, inputs)
    return {k: v.copy() for k, v in hooks.record.items()}


def patch_scan_site_level(unlearner, clean_inputs, clean_targets,
                           corrupted_inputs, n_sites, log=print):
    """
    B2, livello sito x posizione: per ogni sito e ogni posizione di token si
    sostituisce l'attivazione CORROTTA con quella CLEAN e si misura quanto
    cala la NLL della continuazione pulita.

    Il segno e' definito in modo che POSITIVO = ripristino:
        score = NLL(continuazione pulita | run corrotto) - NLL(... | patch pulito)
    Viene riportata anche la direzione inversa (run pulito con attivazione
    corrotta): se le due direzioni concordano, la dipendenza e' reciproca e non
    un artefatto della sostituzione.
    """
    with ResidualHooks(unlearner.model) as hooks:
        clean_acts = collect_activations(unlearner, clean_inputs, hooks)
        corrupted_acts = collect_activations(unlearner, corrupted_inputs, hooks)
        n_positions = int(np.asarray(clean_inputs).shape[1])

        # Riferimenti: quanto peggio e' sul run corrotto, quanto meglio sul pulito.
        nll_corrupted = _nll_per_token(unlearner, corrupted_inputs, clean_targets)
        nll_clean = _nll_per_token(unlearner, clean_inputs, clean_targets)

        rows = []
        for site in range(n_sites):
            for pos in range(n_positions):
                patched = corrupted_acts[site].copy()
                patched[:, pos, :] = clean_acts[site][:, pos, :]
                hooks.patch[site] = patched
                nll_forward = _nll_per_token(unlearner, corrupted_inputs, clean_targets)
                hooks.patch.pop(site)

                patched_rev = clean_acts[site].copy()
                patched_rev[:, pos, :] = corrupted_acts[site][:, pos, :]
                hooks.patch[site] = patched_rev
                nll_reverse = _nll_per_token(unlearner, clean_inputs, clean_targets)
                hooks.patch.pop(site)

                rows.append({
                    "site": site,
                    "position": pos,
                    "score_forward": nll_corrupted - nll_forward,
                    "score_reverse": nll_clean - nll_reverse,
                    "nll_after_patch_forward": nll_forward,
                    "nll_after_patch_reverse": nll_reverse,
                })

    return {
        "nll_corrupted_run": nll_corrupted,
        "nll_clean_run": nll_clean,
        "n_positions": n_positions,
        "rows": rows,
    }


def patch_scan_coordinate_level(unlearner, clean_inputs, clean_targets,
                                corrupted_inputs, n_sites, log=print):
    """
    B2, livello coordinata: per ogni sito, posizione e coordinata d_model si
    sostituisce l'attivazione corrotta con quella pulita e si misura lo stesso
    ripristino. Serve a localizzare il vettore, non solo la posizione.
    """
    with ResidualHooks(unlearner.model) as hooks:
        clean_acts = collect_activations(unlearner, clean_inputs, hooks)
        corrupted_acts = collect_activations(unlearner, corrupted_inputs, hooks)
        n_positions = int(np.asarray(clean_inputs).shape[1])
        d_model = int(clean_acts[0].shape[2])
        nll_corrupted = _nll_per_token(unlearner, corrupted_inputs, clean_targets)

        rows = []
        for site in range(n_sites):
            for pos in range(n_positions):
                for coord in range(d_model):
                    patched = corrupted_acts[site].copy()
                    patched[:, pos, coord] = clean_acts[site][:, pos, coord]
                    hooks.patch[site] = patched
                    nll = _nll_per_token(unlearner, corrupted_inputs, clean_targets)
                    hooks.patch.pop(site)
                    rows.append({
                        "site": site,
                        "position": pos,
                        "coord": coord,
                        "score": nll_corrupted - nll,
                    })
    return {"nll_corrupted_run": nll_corrupted, "n_positions": n_positions,
            "d_model": d_model, "rows": rows}


# --------------------------------------------------------------------------
# B3: set minimo di componenti
# --------------------------------------------------------------------------

def build_candidate_pool(coord_rows, site_rows, n_sites, log=print):
    """
    Pool di candidati a tre granularita', tutti ordinati per lo stesso tipo di
    punteggio causale (ripristino della continuazione pulita patching
    l'attivazione):

      - `(site, pos, coord)`: una singola coordinata del flusso residuo;
      - `(site, pos)`: l'intero vettore residuo a una posizione;
      - `(site)`: l'intero flusso residuo del sito.

    Il greedy li tratta allo stesso modo perche' l'ablation e' sempre "azzera
    questo insieme di coordinate", con un insieme che puo' avere cardinalita'
    1, d_model o seq_len x d_model. Un candidato piu' grossolano con punteggio piu'
    alto viene quindi scelto per primo, ed e' il comportamento desiderato: non
    ha senso eliminare 64 coordinate una alla volta se una posizione intera ha
    gia' un punteggio maggiore.
    """
    pool = []
    for row in coord_rows:
        pool.append({"site": row["site"], "position": row["position"],
                     "coord": row["coord"], "granularity": "coordinata",
                     "score": row["score"]})
    for row in site_rows:
        pool.append({"site": row["site"], "position": row["position"],
                     "coord": None, "granularity": "posizione",
                     "score": row["score_forward"]})
    for site in range(n_sites):
        total = sum(r["score_forward"] for r in site_rows if r["site"] == site)
        pool.append({"site": site, "position": None, "coord": None,
                     "granularity": "sito", "score": float(total)})
    pool.sort(key=lambda c: c["score"], reverse=True)
    return pool


def greedy_minimal_set(unlearner, clean_inputs, clean_targets, ranked_components,
                       target_nll, n_sites, max_components=MAX_COMPONENTS, log=print):
    """
    Cerca il sottoinsieme MINIMO di componenti la cui ablation (azzero delle
    sole coordinate indicate) porta la NLL/token di X_forget alla soglia
    `target_nll`.

    Strategia greedy in ordine di punteggio causale: si aggiunge un componente
    alla volta e si misura l'effetto CUMULATIVO, non quello del singolo
    componente da solo. Il risultato e' un lower bound pragmatico, non il
    minimo esatto: con 3123 candidati una ricerca esaustiva sarebbe 2^3123 e
    non e' eseguibile. E' dichiarato.

    `ranked_components`: lista di {"site", "position", "coord", "score"} gia'
    ordinata per score decrescente (vedi `build_candidate_pool`).
    """
    zero_by_site = {site: set() for site in range(n_sites)}
    trace = []
    chosen = []
    reached = False
    saturated = False
    width = int(np.asarray(clean_inputs).shape[1])
    height = int(unlearner.model.d_model)
    with ResidualHooks(unlearner.model) as hooks:
        baseline = _nll_per_token(unlearner, clean_inputs, clean_targets)
        for rank, component in enumerate(ranked_components[:max_components], start=1):
            site = component["site"]
            if component["granularity"] == "coordinata":
                zero_by_site[site].add((component["position"], component["coord"]))
            elif component["granularity"] == "posizione":
                for coord in range(int(unlearner.model.d_model)):
                    zero_by_site[site].add((component["position"], coord))
            else:
                for pos in range(width):
                    for coord in range(height):
                        zero_by_site[site].add((pos, coord))
            for site_id, coords in zero_by_site.items():
                if coords:
                    hooks.zero_coords[site_id] = coords
                else:
                    hooks.zero_coords.pop(site_id, None)
            nll = _nll_per_token(unlearner, clean_inputs, clean_targets)
            chosen.append(component)
            trace.append({
                "k": rank,
                "nll_per_token": nll,
                "meets_threshold": bool(nll >= target_nll),
                "component": {k: component[k] for k in
                              ("site", "position", "coord", "granularity", "score")},
                "n_active_coordinates": int(sum(len(c) for c in zero_by_site.values())),
            })
            if nll >= target_nll:
                reached = True
                break
            # Saturazione: azzerare l'intero flusso residuo rende costante
            # l'uscita del modello, quindi la NLL si ferma a un massimo
            # raggiungibile. Continuare a valutare candidati su un pianoro non
            # cambia la risposta e costa tempo senza aggiungere informazione.
            recent = [e["nll_per_token"] for e in trace[-PLATEAU_WINDOW:]]
            if len(recent) == PLATEAU_WINDOW and max(recent) - min(recent) < 1e-9:
                saturated = True
                break
    return {
        "baseline_nll_per_token": baseline,
        "target_nll_per_token": target_nll,
        "reached": reached,
        "saturated": saturated,
        "ceiling_nll_per_token": max(e["nll_per_token"] for e in trace),
        "k": len(chosen) if reached else None,
        "explored": len(trace),
        "components": chosen if reached else [],
        "trace": trace,
    }


def _ablation_forward(unlearner, hooks, zero_by_site, inputs, targets):
    """Applica un insieme di coordinate azzerate e restituisce la NLL/token."""
    for site_id, coords in zero_by_site.items():
        if coords:
            hooks.zero_coords[site_id] = coords
        else:
            hooks.zero_coords.pop(site_id, None)
    return _nll_per_token(unlearner, inputs, targets)


def greedy_minimal_set_constrained(unlearner, splits, baseline, clean_inputs,
                                   clean_targets, candidate_pool, target_nll,
                                   n_sites, log=print):
    """
    Stessa ricerca greedy di `greedy_minimal_set`, ma VINCOLATA al collaterale.

    Serve perche' la versione non vincolata e' vacua: il criterio di
    forgetting da solo e' soddisfatto da qualunque intervento che distrugga il
    modello. Misurato: azzerare il flusso residuo a una sola posizione porta
    X_forget dentro la distribuzione di controllo (z bilaterale soddisfatto) e
    porta le 12 correlate a una variazione relativa del ordine delle 10^4
    percento. Non e' un dimenticamento, e' un guasto.

    Procedura: (a) si scartano i candidati la cui ablation DA SOLA viola la
    soglia di collaterale; (b) si accumula per punteggio causale e si ferma al
    primo punto in cui il collaterale si rompe, perche' oltre quel punto il
    confronto non ha piu' senso.
    """
    n_positions = int(np.asarray(clean_inputs).shape[1])
    d_model = int(unlearner.model.d_model)
    admitted = []
    n_destructive = 0
    with ResidualHooks(unlearner.model) as hooks:
        baseline_nll = _nll_per_token(unlearner, clean_inputs, clean_targets)
        for component in candidate_pool:
            zero_by_site = {site: set() for site in range(n_sites)}
            site = component["site"]
            if component["granularity"] == "coordinata":
                zero_by_site[site].add((component["position"], component["coord"]))
            elif component["granularity"] == "posizione":
                for coord in range(d_model):
                    zero_by_site[site].add((component["position"], coord))
            else:
                for pos in range(n_positions):
                    for coord in range(d_model):
                        zero_by_site[site].add((pos, coord))
            nll = _ablation_forward(unlearner, hooks, zero_by_site,
                                    clean_inputs, clean_targets)
            violations = _collateral_violations_unlearner(unlearner, splits,
                                                          baseline)
            if violations > 0:
                n_destructive += 1
            else:
                admitted.append(dict(component, solo_nll=nll))
        admitted.sort(key=lambda c: c["score"], reverse=True)

        trace = []
        chosen = []
        reached = False
        broken_at = None
        for rank, component in enumerate(admitted, start=1):
            chosen.append(component)
            zero_by_site = {site: set() for site in range(n_sites)}
            for spec in chosen:
                site = spec["site"]
                if spec["granularity"] == "coordinata":
                    zero_by_site[site].add((spec["position"], spec["coord"]))
                elif spec["granularity"] == "posizione":
                    for coord in range(d_model):
                        zero_by_site[site].add((spec["position"], coord))
                else:
                    for pos in range(n_positions):
                        for coord in range(d_model):
                            zero_by_site[site].add((pos, coord))
            nll = _ablation_forward(unlearner, hooks, zero_by_site,
                                    clean_inputs, clean_targets)
            violations = _collateral_violations_unlearner(unlearner, splits,
                                                          baseline)
            trace.append({"k": rank, "nll_per_token": nll,
                          "collateral_violations": violations,
                          "meets_threshold": bool(nll >= target_nll),
                          "component": {k: component[k] for k in
                                        ("site", "position", "coord",
                                         "granularity", "score")}})
            if violations > 0 and broken_at is None:
                broken_at = {"k": rank, "nll_per_token": nll,
                             "violations": violations}
                break
            if nll >= target_nll:
                reached = True
                break

    return {
        "baseline_nll_per_token": baseline_nll,
        "target_nll_per_token": target_nll,
        "reached": reached,
        "k": len(chosen) if reached else None,
        "collateral_broken_at": broken_at,
        "explored": len(trace),
        "n_candidates": len(candidate_pool),
        "n_destructive": n_destructive,
        "n_admitted": len(admitted),
        "components": chosen if reached else [],
        "trace": trace,
    }


def _collateral_violations_unlearner(unlearner, splits, baseline):
    """Come `_collateral_violations`, ma senza gli argomenti di ripristino."""
    n = 0
    for key in ("correlated", "probe"):
        after = np.array([unlearner.loglikelihood(t)["loglikelihood"]
                          for t in splits[key]])
        n += int(np.sum(relative_probability_change(np.array(baseline[key]),
                                                    after) > COLLATERAL_TOLERANCE))
    return n


def _collateral_violations(unlearner, splits, baseline, params, indices):
    """
    Numero di frasi (12 correlate + 6 del canario) oltre la soglia del 5%,
    con i pesi correnti. Usato come filtro: un candidato che da solo distrugge
    il modello non e' una localizzazione, e' un guasto.
    """
    n = 0
    for key in ("correlated", "probe"):
        after = np.array([unlearner.loglikelihood(t)["loglikelihood"]
                          for t in splits[key]])
        n += int(np.sum(relative_probability_change(np.array(baseline[key]),
                                                    after) > COLLATERAL_TOLERANCE))
    return n


def greedy_minimal_parameters(unlearner, trainer, splits, baseline,
                              clean_inputs, clean_targets, target_nll, log=print):
    """
    Braccio parallelo a granularita' di PARAMETRO, per rispondere alla domanda
    "quanti parametri" del confronto finale.

    Stessa procedura greedy, ma i candidati sono i valori dei tensori
    addestrabili e l'ablation e' l'azzeramento del valore nel peso. Non e'
    causal tracing: e' una localizzazione per effetto diretto, e viene
    riportata come tale. Il confronto con la Fase A e' perche' il conteggio e'
    espresso nella stessa unita'.

    VINCOLO DI COLLATERALE, necessario e dichiarato. Senza, la ricerca e'
    vacua: basta trovare l'unico parametro la cui azzeramento rompe il modello
    e il "set minimo" e' 1, senza che nessuna localizzazione sia stata
    trovata. Misurato su questo esperimento: azzerando un solo valore di LoRA la
    NLL di X_forget supera la soglia, e nello stesso tempo tutte le 12 correlate
    e tutte le 6 frasi del canario vanno oltre il 5%. Per questo i candidati
    vengono prima filtrati: sono ammessi solo quelli che, azzerati da soli, non
    violano la soglia di collaterale. Il numero di candidati scartati e' un
    risultato, non uno scarto.
    """
    names = trainer._fisher_param_names()
    indices = trainer._trainable_indices()
    params = unlearner.model.get_parameters()
    # Due copie distinte: `theta_pristine` non viene mai modificata ed e' la
    # base del ripristino; `theta0` e' la copia di lavoro che l'accumulo
    # azzera progressivamente. Confonderle lascerebbe il modello corrotto al
    # termine della ricerca.
    theta_pristine = [np.asarray(params[i]).copy() for i in indices]
    theta0 = [t.copy() for t in theta_pristine]

    def restore():
        for slot, idx in enumerate(indices):
            np.asarray(params[idx])[...] = theta_pristine[slot]

    def apply_working():
        for slot, idx in enumerate(indices):
            np.asarray(params[idx])[...] = theta0[slot]

    baseline_nll = _nll_per_token(unlearner, clean_inputs, clean_targets)
    baseline_violations = _collateral_violations(unlearner, splits, baseline,
                                                params, indices)
    log(f"[B3b] violazioni di collaterale sul modello non modificato: "
        f"{baseline_violations} (attese 0: il riferimento deve essere pulito)")

    ranking = []
    n_candidates = 0
    n_destructive = 0
    destructive_example = None
    for slot, idx in enumerate(indices):
        flat = theta0[slot].reshape(-1)
        for flat_index in range(flat.size):
            saved = flat[flat_index]
            if saved == 0.0:
                continue
            n_candidates += 1
            flat[flat_index] = 0.0
            np.asarray(params[idx])[...] = theta0[slot]
            nll = _nll_per_token(unlearner, clean_inputs, clean_targets)
            violations = _collateral_violations(unlearner, splits, baseline,
                                                params, indices)
            if violations > 0:
                n_destructive += 1
                if destructive_example is None:
                    destructive_example = {
                        "slot": slot, "index": flat_index, "name": names[slot],
                        "value": float(saved), "nll_per_token": nll,
                        "collateral_violations": violations,
                    }
            else:
                ranking.append({"slot": slot, "index": flat_index,
                                "name": names[slot], "value": float(saved),
                                "score": nll - baseline_nll})
            # Ripristino: theta0 deve restare il riferimento intatto, altrimenti
            # il candidato successivo partirebbe da un peso diverso.
            flat[flat_index] = saved
            restore()
    ranking.sort(key=lambda r: r["score"], reverse=True)
    log(f"[B3b] candidati valutati: {n_candidates} | scartati come distruttivi "
        f"(azzeramento da solo oltre la soglia di collaterale): {n_destructive} | "
        f"ammessi: {len(ranking)}")
    if destructive_example is not None:
        log(f"[B3b] esempio di candidato distruttivo: "
            f"{destructive_example['name']}[{destructive_example['index']}] = "
            f"{destructive_example['value']:.4f} -> NLL/token "
            f"{destructive_example['nll_per_token']:.4f} con "
            f"{destructive_example['collateral_violations']} violazioni di "
            "collaterale: soddisfarebbe il solo criterio di forgetting, e non "
            "e' una localizzazione")

    trace = []
    chosen = []
    reached = False
    collateral_broken_at = None
    for rank, item in enumerate(ranking[:MAX_COMPONENTS], start=1):
        chosen.append(item)
        for spec in chosen:
            theta0[spec["slot"]].reshape(-1)[spec["index"]] = 0.0
        apply_working()
        nll = _nll_per_token(unlearner, clean_inputs, clean_targets)
        violations = _collateral_violations(unlearner, splits, baseline,
                                            params, indices)
        trace.append({"k": rank, "nll_per_token": nll,
                      "collateral_violations": violations,
                      "meets_threshold": bool(nll >= target_nll)})
        if violations > 0 and collateral_broken_at is None:
            collateral_broken_at = {"k": rank, "nll_per_token": nll,
                                    "violations": violations}
        if nll >= target_nll:
            reached = True
            break
        if violations > 0:
            # Oltre questo punto l'accumulo non e' piu' confrontabile: il
            # modello si sta degradando, non dimenticando. Continuare
            # produrrebbe un "set minimo" costruito su un modello distrutto.
            break
    restore()
    return {
        "baseline_nll_per_token": baseline_nll,
        "baseline_collateral_violations": baseline_violations,
        "target_nll_per_token": target_nll,
        "reached": reached,
        "k": len(chosen) if reached else None,
        "collateral_broken_at": collateral_broken_at,
        "explored": len(trace),
        "n_candidates": n_candidates,
        "n_destructive": n_destructive,
        "n_admitted": len(ranking),
        "destructive_example": destructive_example,
        "top_ranked": ranking[:20],
        "components": chosen if reached else [],
        "trace": trace,
    }


# --------------------------------------------------------------------------
# B5: retest con le metriche della Fase A
# --------------------------------------------------------------------------

def evaluate_with_intervention(unlearner, splits, label, log=print):
    """
    Misura X_forget, holdout, 12 correlate e canario con l'intervento attivo.

    Riusa esattamente `forgetting_success` e `relative_probability_change` della
    Fase A, sulla stessa soglia del 5%: se i due bracci non usassero le stesse
    funzioni il confronto non significherebbe niente.
    """
    forget = unlearner.loglikelihood(splits["forget"])
    holdout = [unlearner.loglikelihood(t)["mean_loglikelihood"]
               for t in splits["unseen_holdout"]]
    correlated_before = [unlearner.loglikelihood(t)["loglikelihood"]
                         for t in splits["correlated"]]
    return {
        "label": label,
        "forget_nll_per_token": forget["nll_mean"],
        "forget_loglik_per_token": forget["mean_loglikelihood"],
        "forget_loglik_total": forget["loglikelihood"],
        "holdout": holdout,
        "correlated": correlated_before,
        "probe": [unlearner.loglikelihood(t)["loglikelihood"]
                  for t in splits["probe"]],
        "background": [unlearner.loglikelihood(t)["mean_loglikelihood"]
                       for t in splits["background_control"]],
    }


def _threshold_from_holdout(holdout_lls, tolerance=Z_TOLERANCE):
    """
    Soglia di NLL per token corrispondente a z <= tolerance sulla distribuzione
    dell'holdout: NLL/token >= -(media + tolerance * std).
    """
    arr = np.asarray(holdout_lls, dtype=float)
    return float(-(arr.mean() + tolerance * arr.std(ddof=1)))


def report_side_by_side(interventions, splits, log=print):
    """
    Confronto finale: stesso riferimento, stesse soglie, un blocco per
    intervento. Qui non si arrotonda e non si riassume: si stanno i numeri.
    """
    log("")
    log("=" * 100)
    log("CONFRONTO FINALE: componenti modificati e rapporto forgetting / collaterale")
    log("=" * 100)
    log(f"{'intervento':34s} {'comp.':>7s} {'NLL/token':>10s} {'z(1s)':>7s} "
        f"{'2-sided':>8s} {'coll max':>10s} {'nViol':>6s} {'probe max':>10s} "
        f"{'nViol':>6s}")
    for row in interventions:
        c = row["criteria"]
        log(f"{row['label']:34s} {row['n_components']:7d} "
            f"{c['forget_nll_per_token']:10.4f} {c['z_one_sided']:7.2f} "
            f"{'SI' if c['forgetting_two_sided'] else 'no':>8s} "
            f"{c['collateral_max_change'] * 100:9.2f}% {c['collateral_n_violations']:6d} "
            f"{c['probe_max_change'] * 100:9.2f}% {c['probe_n_violations']:6d}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build_baseline_measures(unlearner, splits):
    """
    Riferimento comune a tutti gli interventi: le stesse frasi, misurate una
    volta sola sul modello NON modificato. Ogni variazione e' riferita a qui,
    altrimenti "variazione relativa" non avrebbe un denominatore condiviso.
    """
    return {
        "correlated": [unlearner.loglikelihood(t)["loglikelihood"]
                       for t in splits["correlated"]],
        "probe": [unlearner.loglikelihood(t)["loglikelihood"]
                  for t in splits["probe"]],
        "holdout": [unlearner.loglikelihood(t)["mean_loglikelihood"]
                    for t in splits["unseen_holdout"]],
        "background": [unlearner.loglikelihood(t)["mean_loglikelihood"]
                       for t in splits["background_control"]],
    }


def criteria_for(unlearner, splits, baseline, label, n_components):
    """Stesse metriche e stesse soglie della Fase A, più il canario."""
    forget = unlearner.loglikelihood(splits["forget"])
    holdout_after = [unlearner.loglikelihood(t)["mean_loglikelihood"]
                     for t in splits["unseen_holdout"]]
    corr_after = np.array([unlearner.loglikelihood(t)["loglikelihood"]
                           for t in splits["correlated"]])
    probe_after = np.array([unlearner.loglikelihood(t)["loglikelihood"]
                            for t in splits["probe"]])

    success = forgetting_success(forget["mean_loglikelihood"], holdout_after,
                                two_sided=True)
    one_sided = forgetting_success(forget["mean_loglikelihood"], holdout_after)
    corr_change = relative_probability_change(np.array(baseline["correlated"]),
                                              corr_after)
    probe_change = relative_probability_change(np.array(baseline["probe"]),
                                               probe_after)
    n_corr_viol = int(np.sum(corr_change > COLLATERAL_TOLERANCE))
    n_probe_viol = int(np.sum(probe_change > COLLATERAL_TOLERANCE))
    return {
        "label": label,
        "n_components": n_components,
        "criteria": {
            "forget_nll_per_token": forget["nll_mean"],
            "forget_loglik_per_token": forget["mean_loglikelihood"],
            "z_one_sided": one_sided["z_score"],
            "forgetting_one_sided": one_sided["success_one_sided"],
            "forgetting_two_sided": success["success_two_sided"],
            "collateral_max_change": float(corr_change.max()),
            "collateral_mean_change": float(corr_change.mean()),
            "collateral_n_violations": n_corr_viol,
            "collateral_success": n_corr_viol == 0,
            "probe_max_change": float(probe_change.max()),
            "probe_mean_change": float(probe_change.mean()),
            "probe_n_violations": n_probe_viol,
            "probe_success": n_probe_viol == 0,
            "holdout_after_mean": success["control_mean"],
            "holdout_after_std": success["control_std"],
        },
        "correlated_per_sentence": [
            {"text": t, "before": float(b), "after": float(a), "change": float(c)}
            for t, b, a, c in zip(splits["correlated"], baseline["correlated"],
                                  corr_after, corr_change)
        ],
        "probe_per_sentence": [
            {"text": t, "before": float(b), "after": float(a), "change": float(c)}
            for t, b, a, c in zip(splits["probe"], baseline["probe"],
                                  probe_after, probe_change)
        ],
    }


def run_causal_unlearning(epochs=validate.DEFAULT_EPOCHS, lora_rank=None, log=print,
                          param_search_limit=100000):
    """Intera Fase B sullo stesso modello della Fase A."""
    lora_rank = validate.MODEL_CONFIG["lora_rank"] if lora_rank is None else lora_rank
    t_start = time.time()

    trained = validate.build_trained(epochs=epochs, lora_rank=lora_rank, log=log)
    trainer = trained["trainer"]
    splits = trained["splits"]
    unlearner = SelectiveUnlearner(trainer, config=UnlearningConfig(), log=log)
    results = {}

    # ---- B0 ----
    results["b0_architecture"] = architecture_inventory(trainer.model, trainer, log=log)
    n_sites = results["b0_architecture"]["n_sites"]

    # ---- B1 ----
    clean = splits["forget"]
    corrupted = build_corrupted(clean)
    results["b1_pair"] = validate_pair(unlearner, clean, corrupted, log=log)

    clean_inputs, clean_targets = unlearner.windows_from_text(clean)
    corrupted_inputs, corrupted_targets = unlearner.windows_from_text(corrupted)

    baseline = build_baseline_measures(unlearner, splits)
    target_nll = _threshold_from_holdout(baseline["holdout"])
    log("")
    log(f"[B3] soglia di NLL per token corrispondente a z <= {Z_TOLERANCE} "
        f"sull'holdout mai visto: NLL/token >= {target_nll:.4f}")
    log(f"[B3] NLL/token di X_forget sul modello non modificato: "
        f"{unlearner.loglikelihood(clean)['nll_mean']:.4f}")

    # ---- B2: sito x posizione ----
    t0 = time.time()
    site_scan = patch_scan_site_level(unlearner, clean_inputs, clean_targets,
                                      corrupted_inputs, n_sites, log=log)
    site_rows = site_scan["rows"]
    log("")
    log("=== B2: CAUSAL TRACING, sito x posizione (activation patching) ===")
    log(f"riferimento: NLL della continuazione pulita sul run corrotto = "
        f"{site_scan['nll_corrupted_run']:.4f} per token; sul run pulito = "
        f"{site_scan['nll_clean_run']:.4f}")
    log("score > 0 = il patching dell'attivazione PULITA nel run CORROTTO "
        "ripristina la continuazione pulita")
    log("")
    log(f"{'sito':>4} {'pos':>4} {'score_fwd':>12} {'score_rev':>12} "
        f"{'NLL dopo patch fwd':>19}")
    for row in site_rows:
        log(f"{row['site']:4d} {row['position']:4d} "
            f"{row['score_forward']:12.4f} {row['score_reverse']:12.4f} "
            f"{row['nll_after_patch_forward']:19.4f}")
    best_site_row = max(site_rows, key=lambda r: r["score_forward"])
    log("")
    log(f"punteggio massimo: sito {best_site_row['site']} posizione "
        f"{best_site_row['position']} -> {best_site_row['score_forward']:.4f}")
    per_site = {}
    for row in site_rows:
        per_site.setdefault(row["site"], []).append(row["score_forward"])
    for site in sorted(per_site):
        values = np.array(per_site[site])
        log(f"  sito {site}: somma {values.sum():.4f} max {values.max():.4f} "
            f"media {values.mean():.4f}")
    agree = sum(1 for r in site_rows
                if (r["score_forward"] > 0) == (r["score_reverse"] < 0))
    log(f"  concordanza fra le due direzioni del patching (segni opposti, "
        f"come deve essere se la dipendenza e' reciproca): {agree}/{len(site_rows)}")
    log(f"  durata scansione: {time.time() - t0:.2f}s")
    results["b2_site_scan"] = site_scan
    results["b2_site_summary"] = {str(s): {"sum": float(np.sum(v)),
                                           "max": float(np.max(v)),
                                           "mean": float(np.mean(v))}
                                  for s, v in per_site.items()}
    results["b2_direction_agreement"] = agree

    # ---- B2: coordinata ----
    t0 = time.time()
    coord_scan = patch_scan_coordinate_level(unlearner, clean_inputs, clean_targets,
                                             corrupted_inputs, n_sites, log=log)
    coord_rows = sorted(coord_scan["rows"], key=lambda r: r["score"], reverse=True)
    log("")
    log("=== B2: CAUSAL TRACING, sito x posizione x coordinata ===")
    log(f"candidati valutati: {len(coord_rows)} "
        f"({n_sites} siti x {coord_scan['n_positions']} posizioni x "
        f"{coord_scan['d_model']} coordinate)")
    log(f"{'rank':>5} {'sito':>5} {'pos':>4} {'coord':>6} {'score':>12}")
    for rank, row in enumerate(coord_rows[:25], start=1):
        log(f"{rank:5d} {row['site']:5d} {row['position']:4d} {row['coord']:6d} "
            f"{row['score']:12.4f}")
    positive = sum(1 for r in coord_rows if r["score"] > 0)
    log(f"componenti con ripristino positivo: {positive}/{len(coord_rows)}")
    log(f"  durata scansione: {time.time() - t0:.2f}s")
    results["b2_coord_scan"] = {
        "n_candidates": len(coord_rows),
        "n_positive": positive,
        "top25": coord_rows[:25],
    }

    # ---- B3: set minimo di componenti ----
    candidate_pool = build_candidate_pool(coord_rows, site_rows, n_sites, log=log)
    log("")
    log(f"[B3] pool di candidati: {len(candidate_pool)} "
        f"({len(coord_rows)} a singola coordinata, {len(site_rows)} a posizione "
        f"intero, {n_sites} a sito intero)")
    minimal = greedy_minimal_set(unlearner, clean_inputs, clean_targets,
                                 candidate_pool, target_nll, n_sites, log=log)
    log("")
    log("=== B3: SET MINIMO DI COMPONENTI (accumulo greedy) ===")
    log("l'ablazione azzera le sole coordinate elencate, nient'altro")
    log(f"{'k':>4} {'NLL/token':>11} {'soglia':>10} {'raggiunta':>10} "
        f"{'componente aggiunto':>34} {'coord. attive':>14}")
    for entry in minimal["trace"][:40]:
        comp = entry["component"]
        if comp["granularity"] == "coordinata":
            desc = "sito %d pos %d coord %d" % (comp["site"], comp["position"],
                                                comp["coord"])
        elif comp["granularity"] == "posizione":
            desc = "sito %d pos %d (vettore intero)" % (comp["site"], comp["position"])
        else:
            desc = "sito %d (stream intero)" % comp["site"]
        log(f"{entry['k']:4d} {entry['nll_per_token']:11.4f} "
            f"{target_nll:10.4f} {'SI' if entry['meets_threshold'] else 'no':>10s} "
            f"{desc:>34} {entry['n_active_coordinates']:14d}")
    if len(minimal["trace"]) > 40:
        log(f"... ({len(minimal['trace']) - 40} passi ulteriori omessi dalla "
            "tabella, tutti nel JSON)")
    log(f"risultato: {'raggiunto' if minimal['reached'] else 'NON raggiunto'} "
        f"con k = {minimal['k']} componenti su {len(candidate_pool)} candidati "
        f"esplorati = {minimal['explored']} "
        f"(NLL iniziale {minimal['baseline_nll_per_token']:.4f}, "
        f"soglia {target_nll:.4f})")
    if not minimal["reached"]:
        log(f"miglior NLL/token raggiunta con l'intero pool: "
            f"{minimal['ceiling_nll_per_token']:.4f} "
            f"(soglia {target_nll:.4f}, NLL iniziale "
            f"{minimal['baseline_nll_per_token']:.4f}, esplorati "
            f"{minimal['explored']} candidati"
            f"{', poi saturazione rilevata' if minimal['saturated'] else ''})")
    results["b3_minimal_components"] = minimal
    results["b3_candidate_pool_size"] = len(candidate_pool)

    # ---- B3: stessa ricerca, ma vincolata al collaterale ----
    t0 = time.time()
    minimal_constrained = greedy_minimal_set_constrained(
        unlearner, splits, baseline, clean_inputs, clean_targets, candidate_pool,
        target_nll, n_sites, log=log)
    log("")
    log("=== B3c: SET MINIMO DI COMPONENTI, VINCOLATO AL COLLATERALE ===")
    log("stessa ricerca, ma i candidati che da soli violano la soglia del 5% "
        "sono scartati, e l'accumulo si ferma al primo collaterale rotto")
    log(f"candidati: {len(candidate_pool)} | scartati come distruttivi: "
        f"{minimal_constrained['n_destructive']} | ammessi: "
        f"{minimal_constrained['n_admitted']}")
    log(f"{'k':>4} {'NLL/token':>11} {'soglia':>10} {'viol. coll.':>12} "
        f"{'raggiunta':>10} {'componente':>28}")
    for entry in minimal_constrained["trace"][:25]:
        comp = entry["component"]
        if comp["granularity"] == "coordinata":
            desc = "sito %d pos %d coord %d" % (comp["site"], comp["position"],
                                                comp["coord"])
        elif comp["granularity"] == "posizione":
            desc = "sito %d pos %d (vettore)" % (comp["site"], comp["position"])
        else:
            desc = "sito %d (stream)" % comp["site"]
        log(f"{entry['k']:4d} {entry['nll_per_token']:11.4f} {target_nll:10.4f} "
            f"{entry['collateral_violations']:12d} "
            f"{'SI' if entry['meets_threshold'] else 'no':>10s} {desc:>28}")
    broken = minimal_constrained["collateral_broken_at"]
    if broken is not None:
        log(f"collaterale rotto al primo k = {broken['k']} con NLL/token "
            f"{broken['nll_per_token']:.4f} e {broken['violations']} frasi oltre "
            f"il 5%: la soglia di forgetting ({target_nll:.4f}) non e' ancora "
            "stata raggiunta")
    log(f"risultato: {'raggiunto' if minimal_constrained['reached'] else 'NON raggiunto'} "
        f"con k = {minimal_constrained['k']} "
        f"(NLL iniziale {minimal_constrained['baseline_nll_per_token']:.4f})")
    log(f"  durata: {time.time() - t0:.2f}s")
    results["b3c_minimal_components_constrained"] = minimal_constrained

    # ---- B3/B4: braccio a granularita' di parametro ----
    n_trainable_values = int(sum(np.asarray(unlearner.model.get_parameters()[i]).size
                                 for i in trainer._trainable_indices()))
    run_param_search = n_trainable_values <= param_search_limit
    if not run_param_search:
        log("")
        log(f"=== B3b: SET MINIMO DI PARAMETRI ===")
        log(f"saltato: {n_trainable_values} candidati superano il tetto di "
            f"{param_search_limit} valutazioni. Il braccio a parametri e' "
            "un lower bound sul costo, non un'assenza di localizzazione: "
            "non viene riportato come numero.")

    minimal_params = None
    if run_param_search:
        t0 = time.time()
        minimal_params = greedy_minimal_parameters(unlearner, trainer, splits,
                                                   baseline, clean_inputs,
                                                   clean_targets, target_nll, log=log)
        log("")
        log("=== B3b: SET MINIMO DI PARAMETRI (azzeramento di valori addestrabili) ===")
        log("localizzazione per effetto diretto, non causal tracing")
        log(f"candidati con valore non nullo: {minimal_params['n_candidates']}")
        log(f"{'k':>4} {'NLL/token':>11} {'viol. coll.':>12} {'raggiunta':>10}")
        for entry in minimal_params["trace"]:
            log(f"{entry['k']:4d} {entry['nll_per_token']:11.4f} "
                f"{entry['collateral_violations']:12d} "
                f"{'SI' if entry['meets_threshold'] else 'no':>10s}")
        log(f"risultato: {'raggiunto' if minimal_params['reached'] else 'NON raggiunto'} "
            f"con k = {minimal_params['k']} parametri su "
            f"{minimal_params['n_admitted']} candidati ammessi "
            f"(NLL iniziale {minimal_params['baseline_nll_per_token']:.4f})")
        broken = minimal_params["collateral_broken_at"]
        if broken is not None:
            log(f"primo k con violazione di collaterale: {broken['k']} "
                f"(NLL/token {broken['nll_per_token']:.4f}, "
                f"{broken['violations']} frasi oltre il 5%). Il collaterale si "
                "rompe PRIMA che il forgetting sia raggiunto: azzerare i "
                "parametri degrada il modello nel suo complesso invece di "
                "rimuovere la frase.")
        log(f"  durata: {time.time() - t0:.2f}s")
        results["b3b_minimal_parameters"] = minimal_params

    # ---- B4/B5: interventi e retest ----
    log("")
    log("=== B4/B5: RETEST CON LE METRICHE DELLA FASE A ===")
    interventions = []
    reference = criteria_for(unlearner, splits, baseline,
                             "nessun intervento (riferimento)", 0)
    log("")
    log(f"[riferimento] NLL/token {reference['criteria']['forget_nll_per_token']:.4f} "
        f"| z {reference['criteria']['z_one_sided']:.2f} | "
        f"coll max {reference['criteria']['collateral_max_change'] * 100:.3f}% | "
        f"probe max {reference['criteria']['probe_max_change'] * 100:.3f}%")

    if minimal["reached"]:
        with ResidualHooks(unlearner.model) as hooks:
            zero_by_site = {}
            for comp in minimal["components"]:
                site = comp["site"]
                zero_by_site.setdefault(site, set())
                if comp["granularity"] == "coordinata":
                    zero_by_site[site].add((comp["position"], comp["coord"]))
                elif comp["granularity"] == "posizione":
                    for coord in range(int(unlearner.model.d_model)):
                        zero_by_site[site].add((comp["position"], coord))
                else:
                    for pos in range(int(np.asarray(clean_inputs).shape[1])):
                        for coord in range(int(unlearner.model.d_model)):
                            zero_by_site[site].add((pos, coord))
            for site, coords in zero_by_site.items():
                hooks.zero_coords[site] = coords
            row = criteria_for(unlearner, splits, baseline,
                               "B4 attivazione (non vincolata)",
                               minimal["k"])
        interventions.append(row)
        log(f"B4 attivazione non vincolata: applicata a {minimal['k']} "
            f"componenti su {len(candidate_pool)} candidati, a OGNI ingresso "
            "(non e' un caso speciale per X_forget). Il collaterale va letto "
            "prima di ogni altra cosa: vedi la riga corrispondente sopra.")
    else:
        log("[B4] nessun set minimo di componenti trovato sull'intero pool di "
            f"{len(candidate_pool)} candidati: l'intervento sull'attivazione "
            "NON viene applicato e NON viene riportato come un successo")

    if minimal_constrained["reached"]:
        with ResidualHooks(unlearner.model) as hooks:
            zero_by_site = {}
            for comp in minimal_constrained["components"]:
                site = comp["site"]
                zero_by_site.setdefault(site, set())
                if comp["granularity"] == "coordinata":
                    zero_by_site[site].add((comp["position"], comp["coord"]))
                elif comp["granularity"] == "posizione":
                    for coord in range(int(unlearner.model.d_model)):
                        zero_by_site[site].add((comp["position"], coord))
                else:
                    for pos in range(int(np.asarray(clean_inputs).shape[1])):
                        for coord in range(int(unlearner.model.d_model)):
                            zero_by_site[site].add((pos, coord))
            for site, coords in zero_by_site.items():
                hooks.zero_coords[site] = coords
            row = criteria_for(unlearner, splits, baseline,
                               "B4c attivazione (vincolata)",
                               minimal_constrained["k"])
        interventions.append(row)
    else:
        log("[B4c] nessun set minimo di componenti che rispetti anche il "
            "collaterale: l'intervento NON viene applicato e non viene "
            "riportato come un successo")

    if minimal_params is not None and minimal_params["reached"]:
        indices = trainer._trainable_indices()
        params = unlearner.model.get_parameters()
        theta_pristine = [np.asarray(params[i]).copy() for i in indices]
        working = [t.copy() for t in theta_pristine]
        for spec in minimal_params["components"]:
            working[spec["slot"]].reshape(-1)[spec["index"]] = 0.0
        for slot, idx in enumerate(indices):
            np.asarray(params[idx])[...] = working[slot]
        row = criteria_for(unlearner, splits, baseline,
                           "B4b pesi (parametri minimi)", minimal_params["k"])
        interventions.append(row)
        for slot, idx in enumerate(indices):
            np.asarray(params[idx])[...] = theta_pristine[slot]
    else:
        if minimal_params is not None:
            log("[B4b] nessun set minimo di parametri trovato entro il tetto di "
                f"{MAX_COMPONENTS}: l'intervento NON viene applicato")
        else:
            log("[B4b] braccio a parametri non eseguito su questo modello")

    # Fase A, stessa soglia, per il confronto a due braccia.
    log("")
    log("--- braccio di Fase A, misurato con le stesse funzioni ---")
    for label, config, eta in (
        ("Fase A protocollo (prescribed)", UnlearningConfig(projection_mode="prescribed"), None),
        ("Fase A proiezione corretta", UnlearningConfig(projection_mode="metric_orthogonal"), None),
    ):
        trained_theta0 = trained["theta0"]
        validate.restore_trainable(trainer, trained_theta0)
        phase_a = SelectiveUnlearner(trainer, config=config, log=lambda *a, **k: None)
        phase_a.forget(splits["forget"], splits["retain"], learning_rate=eta)
        row = criteria_for(phase_a, splits, baseline, label,
                           trainable_count(phase_a))
        interventions.append(row)
        log(f"{label:34s} NLL/token "
            f"{row['criteria']['forget_nll_per_token']:.4f} | z "
            f"{row['criteria']['z_one_sided']:.2f} | coll max "
            f"{row['criteria']['collateral_max_change'] * 100:.2f}% | probe max "
            f"{row['criteria']['probe_max_change'] * 100:.2f}%")
    validate.restore_trainable(trainer, trained["theta0"])

    report_side_by_side(interventions, splits, log=log)

    # ---- B6: verdetto sulla localizzazione ----
    n_components_found = minimal["k"] or 0
    n_params_found = minimal_params["k"] if minimal_params is not None else None
    n_trainable = results["b0_architecture"]["n_parameters_trainable"]
    log("")
    log("=== B6: VERDETTO SULLA LOCALIZZAZIONE ===")
    log(f"componenti di attivazione, ricerca NON vincolata: "
        f"{n_components_found} su {len(candidate_pool)} candidati "
        f"(esplorati {minimal['explored']})")
    log(f"componenti di attivazione, ricerca VINCOLATA al collaterale: "
        f"k = {minimal_constrained['k']} "
        f"(candidati distruttivi scartati: "
        f"{minimal_constrained['n_destructive']} su {len(candidate_pool)}; "
        f"collaterale rotto al k = "
        f"{(minimal_constrained['collateral_broken_at'] or {}).get('k')})")
    if n_params_found is None:
        if minimal_params is None:
            log(f"parametri addestrabili necessari: non misurato su questo "
                f"modello ({n_trainable} candidati, sopra il tetto della "
                "ricerca)")
        else:
            log(f"parametri addestrabili necessari entro il vincolo di "
                f"collaterale: NESSUN set minimo trovato. Candidati scartati "
                f"come distruttivi: {minimal_params['n_destructive']} su "
                f"{minimal_params['n_candidates']}; collaterale rotto al k = "
                f"{(minimal_params['collateral_broken_at'] or {}).get('k')} con "
                f"NLL/token "
                f"{(minimal_params['collateral_broken_at'] or {}).get('nll_per_token')}"
                f" contro una soglia di {target_nll:.4f}")
    else:
        log(f"parametri addestrabili necessari (azzeramento, entro il vincolo di "
            f"collaterale): {n_params_found} su {n_trainable} "
            f"({100.0 * (n_params_found or 0) / n_trainable:.2f}%)")
    diffuse_components = bool(minimal["explored"] >= len(candidate_pool)
                              and not minimal["reached"])
    diffuse_parameters = bool(n_params_found is not None
                              and n_params_found > 0.5 * n_trainable)
    if minimal_constrained["reached"] or n_params_found is not None:
        log("VERDETTO: esiste un set minimo che rispetta anche il collaterale. "
            "Il numero e' quello da confrontare con la Fase A.")
    elif minimal["reached"]:
        log("VERDETTO (A): il set minimo che soddisfa il SOLO criterio di "
            "forgetting esiste ed e' piccolo, ma lo stesso intervento porta il "
            "collaterale a variazioni di ordini di grandezza: soddisfa il "
            "criterio rompendo il modello, e questo non e' un dimenticamento.")
    else:
        log(f"VERDETTO (B): NESSUN sottoinsieme del pool di {len(candidate_pool)} "
            "componenti porta X_forget alla soglia. Il motivo e' misurabile e "
            "non e' la ricerca: l'ablation satura. Azzerare il flusso residuo "
            "rende costante l'uscita del modello, quindi la NLL/token ha un "
            f"tetto di {minimal['ceiling_nll_per_token']:.4f} contro una soglia "
            f"di {target_nll:.4f}, e la ricerca si ferma li' per definizione. "
            "Su questo modello l'informazione di X_forget non e' aggredibile "
            "da una soppressione di attivazione: o e' ridondante su molti canali, "
            "oppure la soglia di riferimento e' irraggiungibile per qualunque "
            "predittore di questo corpus.")
    if not minimal_constrained["reached"] and n_params_found is None:
        log("In ogni caso il vincolo di collaterale non e' mai stato "
            "soddisfatto insieme al forgetting: al livello delle attivazioni "
            f"{minimal_constrained['n_destructive']} candidati su "
            f"{len(candidate_pool)} sono distruttivi se applicati da soli, e "
            f"l'accumulo dei {minimal_constrained['n_admitted']} superstiti rompe "
            "il collaterale prima di arrivare alla soglia; al livello dei "
            f"parametri, idem. L'informazione di X_forget e' ridondante e "
            "distribuita. E' un risultato, non un fallimento, ed e' tipico di un "
            "modello piccolo che sovraffitta un corpus di poche decine di "
            "frasi: vale per questo corpus, non per il modello di produzione.")
    log(f"localizzazione diffusa: attivazione {diffuse_components} | "
        f"parametri {diffuse_parameters}")
    results["b6_verdict"] = {
        "n_components_needed_unconstrained": n_components_found,
        "n_components_needed_constrained": minimal_constrained["k"],
        "n_candidates_explored": len(candidate_pool),
        "components_explored": minimal["explored"],
        "activation_search_saturated": minimal["saturated"],
        "activation_ceiling_nll_per_token": minimal["ceiling_nll_per_token"],
        "n_destructive_components": minimal_constrained["n_destructive"],
        "n_parameters_needed": n_params_found,
        "n_trainable_parameters": n_trainable,
        "n_destructive_parameters": (minimal_params["n_destructive"]
                                     if minimal_params is not None else None),
        "param_search_ran": minimal_params is not None,
        "diffuse_components": bool(diffuse_components),
        "diffuse_parameters": diffuse_parameters,
        "usable_localization": bool(minimal_constrained["reached"]
                                    or n_params_found is not None),
    }

    results["interventions"] = interventions
    results["reference"] = reference
    results["settings"] = {
        "model": {**validate.MODEL_CONFIG, "lora_rank": lora_rank},
        "epochs": epochs,
        "z_tolerance": Z_TOLERANCE,
        "collateral_tolerance": COLLATERAL_TOLERANCE,
        "max_components": MAX_COMPONENTS,
        "target_nll_per_token": target_nll,
        "replacement_entity": REPLACEMENT_ENTITY,
        "target_entity": TARGET_ENTITY,
    }
    results["train"] = {
        "loss_initial": trained["train_loss_initial"],
        "loss_final": trained["train_loss_final"],
    }
    results["total_seconds"] = time.time() - t_start
    return results


def trainable_count(unlearner):
    """Numero di parametri che l'ascent della Fase A puo' toccare."""
    _, indices, params = unlearner._trainable_slots()
    return int(sum(np.asarray(params[i]).size for i in indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=validate.DEFAULT_EPOCHS)
    parser.add_argument("--no-lora", action="store_true",
                        help="Braccio causal su tutti i parametri invece che "
                             "sulla sola LoRA.")
    parser.add_argument("--param-search-limit", type=int, default=100000,
                        help="Tetto di candidati oltre il quale il braccio a "
                             "parametri viene saltato (161.113 in "
                             "full-parameter sono tropi).")
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "causal_unlearning_results.json"))
    args = parser.parse_args()

    results = run_causal_unlearning(epochs=args.epochs,
                                    lora_rank=0 if args.no_lora else None,
                                    param_search_limit=args.param_search_limit)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_name(path.stem + ("_full" if args.no_lora else "") + path.suffix)
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"risultati grezzi salvati in {path}")


if __name__ == "__main__":
    main()

