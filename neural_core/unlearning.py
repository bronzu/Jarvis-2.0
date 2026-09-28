"""
Machine unlearning selettivo a livello di singolo esempio (Missione 1).

L'obiettivo e' rimuovere l'informazione di UNA frase dal peso del modello senza
ri-addestrare da zero e senza degradare in modo misurabile le prestazioni sul
resto del corpus.

Algoritmo (gradient ascent con proiezione ortogonale nella metrica di Fisher):

    1. g_forget  = grad L(theta; X_forget)
    2. g_retain  = grad L(theta; X_retain_proj)
    3. g_forget_proj = g_forget
                    - (g_forget . F^-1 . g_retain) / (g_retain . F^-1 . g_retain)
                      * (F^-1 . g_retain)
    4. theta     = theta + eta * g_forget_proj

`F` e' la Micro-Fisher diagonale gia' calcolata da
`neural_core.trainer.Trainer.compute_fisher_matrix()` sugli stessi parametri
addestrabili (spazio LoRA quando la LoRA e' attiva, spazio full-parameter
altrimenti). Questo modulo NON ricalcola la Fisher: la riusa, e soltanto se
`None` delega al `Trainer` il compito di calcolarla.

In più del gradiente di retain esiste uno STOP di sicurezza secondario, attivabile
con `UnlearningConfig.probe_tolerance` e passando `probe_texts=`: un canario di
poche frasi, disgiunte sia da X_retain_proj sia dall'insieme di valutazione,
sulla cui variazione MASSIMA di probabilita' l'ascent si ferma. Il canario non
entra nel gradiente: protegge il comportamento osservabile, non la loss.

Perche' la proiezione: il gradiente di forget e quello di retain sono due vettori
nello spazio dei parametri; senza proiezione l'ascent sul primo degrada il
secondo in modo direttamente proporzionale al loro prodotto scalare. La
proiezione ortogonale in metrica F^-1 rimuove esattamente la componente di
`g_forget` che punta lungo `g_retain`, quindi il primo ordine di perturbazione
sulla loss di retain e' nullo. E' una garanzia di PRIMO ordine: gli ordini
superiori e il rumore dei dati non sono controllati, e su un numero limitato di
finestre di retain il gradiente di retain e' esso stesso una stima rumorosa.

LIMITAZIONI DICHIARATE (non nascondere)
---------------------------------------
- `F^-1` e' instabile dove la Fisher e' piccola: un parametro su cui il
  gradiente non e' mai osservato ha F ~ 0 e F^-1 ~ infinito. Si applica un
  pavimento relativo `fisher_floor` (default 1e-3 del massimo), interpretabile
  come un prior isotropico (ridge) sulla Fisher. Questa e' una scelta
  ingegneristica dichiarata, non un resultado teorico.
- Con LoRA attiva l'unlearning puo' toccare SOLO l'adapter: se l'informazione
  dell'esempio e' memorizzata nei pesi congelati (embedding, attenzione,
  output) l'ascent non la rimuove. Il modulo non puo' accorgersene da solo:
  spetta alla valutazione riportarlo.
- Il criterio di successo non e' "la log-likelihood e' piu' bassa": e' il
  rientro di X_forget dentro la distribuzione delle frasi mai viste. E la
  verifica di collaterale non e' "la loss non e' cambiata": e' la variazione di
  probabilita' entro il 5% relativo su un insieme di frasi correlate che NON
  fa parte di X_retain_proj.
- L'unlearning a livello di singolo esempio non implica che l'informazione non
  sia piu' recuperabile con altri mezzi (ri-addestramento, prompt diversi,
  ispezione dei pesi). Nessuna affermazione di cancellazione certificata e'
  supportata da questo modulo.
- L'unlearning concettuale (un intero concetto astratto) e' fuori scope: non e'
  definito nel dataset cosa costituisca un "concetto".

Nessuna API esterna: solo NumPy e standard library.
"""

from dataclasses import dataclass, field
from math import erfc, sqrt
from typing import List, Optional, Sequence

import numpy as np

# Learning rate di fine-tuning standard di JARVIS 2.0 (train.py, default e
# 17_MODEL_TRAINING_PROVENANCE.md sezione 5.2). Il protocollo di unlearning
# parte da un decimo di questo valore.
STANDARD_FINETUNE_LEARNING_RATE = 1e-3

# Motivi di arresto, riportati verbatim nel risultato.
STOP_RETAIN_LOSS = "retain_loss_increase"
STOP_MAX_STEPS = "max_steps"
STOP_DEGENERATE_PROJECTION = "degenerate_projection"
STOP_PROBE_DRIFT = "probe_drift"
STOP_FISHER_MISSING = "fisher_missing"


@dataclass
class UnlearningConfig:
    """
    Iperparametri del protocollo di unlearning. Tutti i default sono quelli
    del protocollo; ogni valore diverso va dichiarato nel report.
    """

    base_learning_rate: float = STANDARD_FINETUNE_LEARNING_RATE
    #: Soglia di arresto: STOP se la loss media su X_retain_proj supera
    #: `retain_loss_tolerance` in modo RELATIVO rispetto al valore pre-unlearning.
    retain_loss_tolerance: float = 0.05
    #: Numero massimo di step di ascent.
    max_steps: int = 100
    #: Finestra iniziale (in step) entro la quale il superamento della soglia
    #: provoca un dimezzamento di eta e un riavvio da theta_0 invece dello STOP.
    halving_window: int = 3
    #: Numero massimo di dimezzamenti/riavvii.
    max_restarts: int = 3
    #: Pavimento relativo applicato alla Fisher diagonale prima dell'inverso.
    fisher_floor: float = 1e-3
    #: Modalita' di proiezione:
    #:   "prescribed"       -> la formula esatta del protocollo;
    #:   "metric_orthogonal"-> proiezione ortogonale esatta rispetto a
    #:                        u = F^-1 g_retain (vedi le docstring dei due
    #:                        metodi statici per la differenza).
    projection_mode: str = "prescribed"
    #: Se True, quando lo STOP e' provocato dalla soglia sulla loss di retain i
    #: pesi vengono riportati all'ULTIMO stato verificato conforme alla soglia
    #: (che all'inizio e' theta_0), non semplicemente al passo precedente: il
    #: passo precedente puo' aver gia' violato la soglia, se il superamento e'
    #: avvenuto al primo passo dopo un riavvio. Il testo del protocollo non dice
    #: nulla su questo punto: lasciare il modello nello stato che ha appena
    #: violato la soglia renderebbe incoerente ogni misura successiva. Con
    #: False lo stato offensivo viene mantenuto e il risultato lo segnala.
    rollback_on_stop: bool = True
    #: Soglia dello STOP di sicurezza secondario sul probe set (canario). None
    #: = disattivo, e il protocollo resta identico a quello dichiarato. Se
    #: impostato, l'ascent si ferma quando la variazione RELATIVA MASSIMA della
    #: probabilita' di una frase del probe supera questa soglia, misurata rispetto
    #: allo stato theta_0. E' il criterio richiesto dalla missione: il danno
    #: collaterale si giudica sul caso peggiore, non sulla media. Il probe non
    #: entra mai nel gradiente e non partecipa a nessun criterio di esito.
    probe_tolerance: Optional[float] = None
    #: Se impostato, norma massima della direzione di ascent (None = disattivo,
    #: cioe' il protocollo non viene alterato).
    max_step_norm: Optional[float] = None
    #: Lunghezza della finestra di contesto usata per tokenizzare le frasi.
    context_size: int = 48
    stride: Optional[int] = None

    def __post_init__(self):
        if self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate deve essere > 0.")
        if self.max_steps < 1:
            raise ValueError("max_steps deve essere >= 1.")
        if self.retain_loss_tolerance < 0:
            raise ValueError("retain_loss_tolerance deve essere >= 0.")
        if self.probe_tolerance is not None and self.probe_tolerance < 0:
            raise ValueError("probe_tolerance deve essere >= 0 o None.")
        if self.max_restarts < 0:
            raise ValueError("max_restarts deve essere >= 0.")
        if self.halving_window < 0:
            raise ValueError("halving_window deve essere >= 0.")
        if not 0.0 < self.fisher_floor < 1.0:
            raise ValueError("fisher_floor deve essere in (0, 1).")
        if self.projection_mode not in ("prescribed", "metric_orthogonal"):
            raise ValueError(
                "projection_mode deve essere 'prescribed' o 'metric_orthogonal', "
                f"ricevuto: {self.projection_mode!r}"
            )
        if self.context_size < 2:
            raise ValueError("context_size deve essere >= 2.")


@dataclass
class UnlearningResult:
    """Esito dell'ascent: cosa e' successo, in modo ispezionabile."""

    stop_reason: str
    steps_executed: int
    learning_rate_initial: float
    learning_rate_final: float
    restarts: int
    #: Una voce per ogni dimezzamento: {"attempt", "step", "retain_loss",
    #: "learning_rate_before", "learning_rate_after"}.
    halving_events: List[dict] = field(default_factory=list)
    #: Cronologia per step: {"step", "retain_loss", "forget_nll", "grad_norm",
    #: "projection_alpha", "learning_rate", "attempt"}.
    history: List[dict] = field(default_factory=list)
    forget_nll_before: float = float("nan")
    forget_nll_after: float = float("nan")
    forget_loglik_before: float = float("nan")
    forget_loglik_after: float = float("nan")
    forget_tokens: int = 0
    retain_loss_before: float = float("nan")
    retain_loss_after: float = float("nan")
    retain_windows: int = 0
    fisher_source: str = ""
    fisher_floor_applied: float = float("nan")
    fisher_max: float = float("nan")
    trainable_parameters: int = 0
    projection_mode: str = "prescribed"
    #: True se lo stato finale supera la soglia sulla loss di retain (possibile
    #: solo con `rollback_on_stop=False`).
    retain_threshold_violated: bool = False
    #: Passi eseguiti il cui effetto e' stato annullato dal rollback.
    rolled_back_steps: int = 0
    projection_coefficients: List[float] = field(default_factory=list)
    #: Soglia dello STOP sul probe (None se disattivo) e numero di frasi usate.
    probe_tolerance: Optional[float] = None
    probe_sentences: int = 0
    #: Variazione relativa massima sul probe set allo stato finale, e la sua
    #: cronologia per step. La baseline e' theta_0, quindi al passo 1 la
    #: variazione e' piccola ma gia' non nulla.
    probe_max_relative_change_after: float = 0.0
    probe_max_relative_change_history: List[float] = field(default_factory=list)

    @property
    def retain_loss_relative_change(self):
        if not np.isfinite(self.retain_loss_before) or self.retain_loss_before == 0:
            return float("nan")
        return (self.retain_loss_after - self.retain_loss_before) / self.retain_loss_before


# --------------------------------------------------------------------------
# Funzioni statistiche di valutazione (criteri dichiarati nel protocollo)
# --------------------------------------------------------------------------

def forgetting_success(forget_loglik, unseen_logliks, tolerance_sigma: float = 1.0,
                       two_sided: bool = False):
    """
    Criterio di successo (1): la log-likelihood di X_forget dopo l'unlearning
    deve essere rientrata nella distribuzione di log-likelihood delle frasi mai
    viste.

    Operativamente: z = (ll_forget - media(se mai viste)) / std(se mai viste) e
    successo se z <= `tolerance_sigma`.

    Con `two_sided=True` si richiede invece |z| <= `tolerance_sigma`, cioe' un
    rientro vero e proprio nella distribuzione. La differenza non e' accademica:
    il criterio unilaterale e' soddisfatto anche da un modello distrutto, per
    il quale X_forget e' molto MENO probabile di qualunque testo mai visto. Un
    metodo che regge solo il criterio unilaterale ha solo spostato la soglia.

    `forget_loglik` e `unseen_logliks` devono essere quantita' PER TOKEN
    (`SelectiveUnlearner.loglikelihood()["mean_loglikelihood"]`). Con la
    log-likelihood totale il confronto misurerebbe la lunghezza delle frasi,
    non la loro memorizzazione: su frasi di lunghezza diversa il criterio
    sarebbe soddisfatto (o fallito) per ragioni di token count.

    E' una definizione operativa dichiarata, non un test ipotetico formale: con
    un numero ridotto di frasi di controllo la stima di media e deviazione
    standard e' rumorosa.

    Returns:
        dict con z, media, std, min/max del controllo e i due booleani di
        successo (unilaterale e bilaterale).
    """
    control = np.asarray(list(unseen_logliks), dtype=float)
    forget = float(forget_loglik)
    if control.size < 2:
        raise ValueError("Servono almeno 2 frasi di controllo per definire la distribuzione.")
    mean = float(control.mean())
    std = float(control.std(ddof=1))
    if std <= 0.0:
        # Distribuzione degenere: il confronto z non e' definito, si confronta
        # direttamente con la media e il flag resta comunque esplicito.
        z = 0.0 if abs(forget - mean) < 1e-9 else float("inf") * (1 if forget > mean else -1)
    else:
        z = (forget - mean) / std
    return {
        "forget_loglik": forget,
        "control_mean": mean,
        "control_std": std,
        "control_min": float(control.min()),
        "control_max": float(control.max()),
        "control_n": int(control.size),
        "z_score": float(z),
        "tolerance_sigma": float(tolerance_sigma),
        "two_sided": bool(two_sided),
        "success": bool(abs(z) <= tolerance_sigma) if two_sided else bool(z <= tolerance_sigma),
        "success_one_sided": bool(z <= tolerance_sigma),
        "success_two_sided": bool(abs(z) <= tolerance_sigma),
    }


def relative_probability_change(loglik_before, loglik_after):
    """
    Variazione RELATIVA di probabilita' della sequenza: |p' - p| / p con
    p = exp(loglik). Qui si usano le log-likelihood TOTALI, perche' la
    probabilita' di una frase e' proporzionale a exp(loglik totale) e non a
    exp(loglik per token).
    """
    before = np.asarray(list(loglik_before), dtype=float)
    after = np.asarray(list(loglik_after), dtype=float)
    if before.shape != after.shape:
        raise ValueError("loglik_before e loglik_after devono avere la stessa forma.")
    # Lavoro in log-spazio per non underfloware exp().
    with np.errstate(invalid="ignore", divide="ignore"):
        change = np.abs(np.expm1(after - before))
    return change


def single_vs_distribution(value, sample):
    """
    Confronto di un SINGOLO valore (X_forget) con una distribuzione campione
    (le frasi mai viste): t = (x - media) / (std / sqrt(n + 1)), con p-value
    per approssimazione normale. Solo NumPy + standard library.

    Serve a distinguere "X_forget rientra nella distribuzione delle frasi mai
    viste" da "X_forget e' ancora sistematicamente piu' probabile". Con 5-15
    frasi di controllo il test e' indicativo, non esatto: il p-value e'
    dichiarato come tale e va letto con cautela.
    """
    x = float(value)
    arr = np.asarray(list(sample), dtype=float)
    n = int(arr.size)
    if n < 2:
        raise ValueError("Servono almeno 2 elementi nel campione di riferimento.")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std <= 0.0:
        return {"value": x, "mean": mean, "std": std, "t": float("inf"),
                "p_value": 0.0, "n": n}
    t = float((x - mean) / (std / sqrt(n + 1)))
    p = float(erfc(abs(t) / sqrt(2.0)))
    return {"value": x, "mean": mean, "std": std, "t": t, "p_value": p, "n": n}


class SelectiveUnlearner:
    """
    Ascent selettivo su un singolo esempio, con proiezione di Fisher.

    Args:
        trainer: istanza di `neural_core.trainer.Trainer` che espone `model`,
            `tokenizer` e (opzionalmente) `fisher_matrix` già calcolata da
            `compute_fisher_matrix()`. Non viene creato alcun modello qui.
        config: `UnlearningConfig`. Se omesso usa il protocollo standard.
        fisher: dizionario nome -> array. Se omesso usa `trainer.fisher_matrix`.
        log: funzione di logging (default `print`).
    """

    def __init__(self, trainer, config: Optional[UnlearningConfig] = None,
                 fisher: Optional[dict] = None, log=print):
        if trainer is None or not hasattr(trainer, "model"):
            raise ValueError("trainer deve essere un'istanza di neural_core.trainer.Trainer.")
        self.trainer = trainer
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer
        self.config = config or UnlearningConfig()
        self._log = log
        self.fisher, self.fisher_source = self._resolve_fisher(fisher)

    # ------------------------------------------------------------------
    # Fisher
    # ------------------------------------------------------------------

    def _resolve_fisher(self, fisher):
        source = "argomento `fisher`"
        chosen = fisher
        if chosen is None:
            chosen = getattr(self.trainer, "fisher_matrix", None)
            source = "Trainer.fisher_matrix (Micro-Fisher/LoRA esistente)"
        if chosen is None:
            return None, "assente"
        if not isinstance(chosen, dict) or not chosen:
            raise ValueError("La Fisher deve essere un dizionario non vuoto nome -> array.")
        return chosen, source

    def _trainable_slots(self):
        """
        Restituisce (nomi, indici, parametri correnti) dei parametri
        addestrabili, riusando la stessa selezione del Trainer (spazio LoRA
        quando la LoRA e' attiva, spazio full-parameter altrimenti).
        """
        names = self.trainer._fisher_param_names()
        indices = self.trainer._trainable_indices()
        if not names or len(names) != len(indices):
            raise ValueError("Selezione dei parametri addestrabili incoerente.")
        params = self.model.get_parameters()
        return list(names), list(indices), params

    def fisher_arrays(self, names, indices, floor: Optional[float] = None):
        """
        Allinea la Fisher agli indici addestrabili e applica il pavimento
        relativo. Ritorna (liste F_eff, F_grezza, floor_effettivo).

        `floor` relativo: F_eff = max(F, floor * max(F)). Se la Fisher grezza e'
        tutta zero (parametri mai osservati) si restituisce 1.0: in quel caso
        l'inverso e' l'identita' e la proiezione diventa euclidea, che e' il
        comportamento ben definito.
        """
        floor = self.config.fisher_floor if floor is None else float(floor)
        raw = []
        for name, idx in zip(names, indices):
            if name not in self.fisher:
                raise KeyError(f"La Fisher non contiene il parametro addestrabile '{name}'.")
            array = np.asarray(self.fisher[name], dtype=float)
            if array.shape != np.asarray(self.model.get_parameters()[idx]).shape:
                raise ValueError(
                    f"Forma della Fisher incoerente per '{name}': {array.shape} "
                    f"atteso {np.asarray(self.model.get_parameters()[idx]).shape}."
                )
            raw.append(array)
        flat = np.concatenate([a.reshape(-1) for a in raw]) if raw else np.zeros(0)
        peak = float(flat.max()) if flat.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            effective = [np.ones_like(a) for a in raw]
            return effective, raw, 0.0
        threshold = floor * peak
        effective = [np.maximum(a, threshold) for a in raw]
        return effective, raw, float(threshold)

    # ------------------------------------------------------------------
    # Tokenizzazione e finestre
    # ------------------------------------------------------------------

    def windows_from_text(self, text: str):
        """
        Converte una frase in finestre (inputs, targets) con target shiftati di
        uno, troncata alla context window del modello.
        """
        ids = self.tokenizer.encode(text, add_end_of_text=True)
        if len(ids) < 2:
            raise ValueError(f"Testo troppo corto per produrre una finestra: {text!r}")
        window = min(int(self.config.context_size), int(self.model.max_len))
        # Input e target devono avere la STESSA lunghezza: n coppie (t, t+1).
        span = min(window, len(ids) - 1)
        inputs = np.asarray(ids[:span], dtype=np.int64).reshape(1, -1)
        targets = np.asarray(ids[1:span + 1], dtype=np.int64).reshape(1, -1)
        return inputs, targets

    def windows_from_texts(self, texts: Sequence[str]):
        """
        Finestre individuali (inputs, targets) per ogni frase, senza
        troncamento a lunghezza comune: ogni frase mantiene il suo contesto.
        """
        if not texts:
            raise ValueError("serve almeno una frase.")
        return [self.windows_from_text(text) for text in texts]

    # ------------------------------------------------------------------
    # Loss e gradienti (solo parametri addestrabili)
    # ------------------------------------------------------------------

    def _forward_nll_sum(self, inputs, targets):
        """Somma delle NLL su tutti i token della finestra (nessun aggiornamento)."""
        logits = self.model.forward(np.asarray(inputs), training=False)
        flat_logits = np.asarray(logits, dtype=float).reshape(-1, self.model.vocab_size)
        flat_targets = np.asarray(targets, dtype=int).reshape(-1)
        shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1))
        picked = shifted[np.arange(flat_targets.size), flat_targets] - logsumexp
        return float(-picked.sum())

    def loglikelihood(self, text: str):
        """
        Log-likelihood della frase, senza aggiornare alcun peso.

        Restituisce sia il totale sia la media per token. Il confronto
        fra frasi di lunghezza diversa va fatto sulla media per token: la
        log-likelihood TOTALE e' dominata dal numero di token, e usarla
        come criterio farebbe misurare la lunghezza della frase invece
        della sua memorizzazione.
        """
        inputs, targets = self.windows_from_text(text)
        logits = self.model.forward(inputs, training=False)
        flat_logits = np.asarray(logits, dtype=float).reshape(-1, self.model.vocab_size)
        flat_targets = np.asarray(targets, dtype=int).reshape(-1)
        shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1))
        nll = -(shifted[np.arange(flat_targets.size), flat_targets] - logsumexp)
        total = float(nll.sum())
        return {
            "loglikelihood": -total,
            "mean_loglikelihood": float(-nll.mean()),
            "total_nll": total,
            "nll_mean": float(nll.mean()),
            "n_tokens": int(flat_targets.size),
        }

    def loss_on_texts(self, texts: Sequence[str]):
        """Loss media per token su un insieme di frasi (X_retain_proj)."""
        total, count = 0.0, 0
        for inputs, targets in self.windows_from_texts(list(texts)):
            nll = self._forward_nll_sum(inputs, targets)
            total += nll
            count += int(np.asarray(targets).size)
        return total / count if count else float("inf")

    def trainable_gradients(self, inputs, targets, indices):
        """
        Gradiente della cross entropy sui parametri addestrabili, normalizzato
        per token (equivale al gradiente della loss media per token).

        `model.backward()` azzera gia' i gradienti dei parametri congelati
        quando la LoRA e' attiva, ma il filtro esplicito su `indices` rende
        il contratto indipendente da quel comportamento.
        """
        self.model.forward(np.asarray(inputs), training=False)
        gradients = self.model.backward(np.asarray(targets))
        n_tokens = int(np.asarray(targets).size)
        picked = [np.asarray(gradients[i], dtype=float) * n_tokens for i in indices]
        return [p / float(n_tokens) for p in picked]

    def trainable_gradients_on_texts(self, texts: Sequence[str], indices):
        """
        Gradiente della loss media per token sull'insieme di frasi indicato,
        pesato per il numero di token di ciascuna frase.
        """
        pairs = self.windows_from_texts(list(texts))
        total_tokens = sum(int(np.asarray(t).size) for _, t in pairs)
        accumulator = None
        for inputs, targets in pairs:
            part = self.trainable_gradients(inputs, targets, indices)
            accumulator = [a.copy() for a in part] if accumulator is None else [
                a + b for a, b in zip(accumulator, part)
            ]
        return [a / float(total_tokens) for a in accumulator]

    # ------------------------------------------------------------------
    # Proiezione
    # ------------------------------------------------------------------

    @staticmethod
    def project_in_fisher_metric(g_forget, g_retain, fisher_eff):
        """
        Proiezione nella metrica di Fisher, formula PRESCRITTA dal protocollo:

            g_proj = g_f - (g_f . F^-1 . g_r) / (g_r . F^-1 . g_r) * F^-1 . g_r

        Tutte le operazioni sono elementwise. Ritorna (g_proj, alpha, den)
        dove `alpha` e' il coefficiente della proiezione e `den` il prodotto
        scalare al denominatore. Se `den` e' nullo o non finito la proiezione e'
        degenerata e si restituisce `g_forget` invariato con alpha = 0: in quel
        caso il chiamante deve fermarsi invece di procedere senza protezione.

        LIMITAZIONE MISURATA E DICHIARATA: questa formula e' una proiezione
        ortogonale vera e propria SOLO se F^-1 e' l'identita'. Infatti il
        termine rimosso e' un multiplo di F^-1 g_r, quindi
        `<g_proj, g_r> = <g_f, g_r> - alpha <F^-1 g_r, g_r>` e la
        semiosfera di protezione non si annulla se F e' anisotropa. Con
        `projection_mode="metric_orthogonal"` si usa invece la proiezione
        ortogonale esatta rispetto a u = F^-1 g_r. Vedi
        `project_metric_orthogonal`.
        """
        inv_f = [1.0 / f for f in fisher_eff]
        numerator = 0.0
        denominator = 0.0
        for gf, gr, inv in zip(g_forget, g_retain, inv_f):
            numerator += float(np.sum(gf * inv * gr))
            denominator += float(np.sum(inv * gr * gr))
        if not np.isfinite(denominator) or denominator <= 0.0:
            return list(g_forget), 0.0, denominator
        alpha = numerator / denominator
        projected = [gf - alpha * inv * gr for gf, gr, inv in zip(g_forget, g_retain, inv_f)]
        return projected, float(alpha), float(denominator)

    @staticmethod
    def project_metric_orthogonal(g_forget, g_retain, fisher_eff):
        """
        Proiezione ortogonale ESATTA rispetto a u = F^-1 g_r:

            g_proj = g_f - (g_f . u) / (u . u) * u,   u = F^-1 . g_r

        A differenza della formula prescritta, questa garantisce
        `<g_proj, F^-1 g_r> = 0` per qualunque Fisher diagonale, anisotropa o
        no. E' la formulazione usata in letteratura come "orthogonal gradient
        descent" (proiezione del gradiente di forget fuori dalla direzione del
        gradiente di retain riscalato dalla Fisher).

        Ritorna (g_proj, alpha, ||u||^2).
        """
        inv_f = [1.0 / f for f in fisher_eff]
        u = [inv * gr for inv, gr in zip(inv_f, g_retain)]
        numerator = 0.0
        denominator = 0.0
        for gf, uu in zip(g_forget, u):
            numerator += float(np.sum(gf * uu))
            denominator += float(np.sum(uu * uu))
        if not np.isfinite(denominator) or denominator <= 0.0:
            return list(g_forget), 0.0, denominator
        alpha = numerator / denominator
        projected = [gf - alpha * uu for gf, uu in zip(g_forget, u)]
        return projected, float(alpha), float(denominator)

    # ------------------------------------------------------------------
    # Protocollo di unlearning
    # ------------------------------------------------------------------

    def forget(self, forget_text: str, retain_texts: Sequence[str],
               learning_rate: Optional[float] = None,
               probe_texts: Optional[Sequence[str]] = None) -> UnlearningResult:
        """
        Esegue l'ascent selettivo su `forget_text` proteggendo `retain_texts`.

        Sequenza: gradienti su X_forget e X_retain_proj -> proiezione in
        metrica di Fisher -> `theta += eta * g_forget_proj`, con dimezzamento di
        eta e riavvio da theta_0 se la loss di retain supera la soglia entro i
        primi `halving_window` step (massimo `max_restarts` riavvii), e STOP se
        la supera dopo quella finestra o al termine di `max_steps` step.

        `probe_texts` attiva lo STOP di sicurezza secondario: a ogni passo si
        confronta la variazione relativa di probabilita' di ogni frase del probe
        rispetto a theta_0 e, se la MAXIMUM supera `config.probe_tolerance`,
        l'ascent si ferma con `stop_reason="probe_drift"`. Il probe non entra
        nel gradiente: e' un canario, non una protezione.

        Due scelte dichiarate, non nel testo del protocollo:
        - lo STOP sul probe e' un STOP semplice, non un dimezzamento di eta con
          riavvio. Il canario e' un segnale di sicurezza, e rilanciare
          automaticamente da theta_0 con eta dimezzato trasformerebbe un
          segnale in una procedura di ottimizzazione;
        - quando lo STOP e' sul probe si esegue lo stesso rollback allo stato
          conforme di `theta_last_ok`, che ora e' l'ultimo stato che rispetta
          SIA la soglia di retain SIA quella del probe.
        """
        if self.fisher is None:
            raise ValueError(
                "Nessuna Fisher disponibile: calcolarla con "
                "Trainer.compute_fisher_matrix() (o passarla con `fisher=`). "
                "Questo modulo non la ricalcola da solo sul modello intero."
            )
        if not retain_texts:
            raise ValueError("X_retain_proj non puo' essere vuoto.")
        if probe_texts is not None and self.config.probe_tolerance is None:
            raise ValueError(
                "probe_texts richiede config.probe_tolerance: senza soglia il "
                "probe non puo' fermare l'ascent."
            )

        names, indices, params = self._trainable_slots()
        fisher_eff, fisher_raw, floor_eff = self.fisher_arrays(names, indices)
        n_trainable = int(sum(np.asarray(params[i]).size for i in indices))

        theta0 = [np.asarray(params[i]).copy() for i in indices]
        forget_inputs, forget_targets = self.windows_from_text(forget_text)
        retain_texts = list(retain_texts)

        forget_stats_before = self.loglikelihood(forget_text)
        retain_before = self.loss_on_texts(retain_texts)
        threshold = retain_before * (1.0 + self.config.retain_loss_tolerance)

        # Baseline del canario: log-likelihood TOTALE di ogni frase del probe
        # allo stato theta_0. Serve la totale perche' la soglia di collaterale e'
        # sulla probabilita' della frase, non sulla sua media per token.
        probe_texts = list(probe_texts) if probe_texts is not None else None
        probe_baseline = None
        if probe_texts is not None:
            if not probe_texts:
                raise ValueError("probe_texts non puo' essere una lista vuota.")
            probe_baseline = [self.loglikelihood(t)["loglikelihood"]
                              for t in probe_texts]

        eta0 = float(self.config.base_learning_rate / 10.0) if learning_rate is None else float(learning_rate)
        eta = eta0
        result = UnlearningResult(
            stop_reason=STOP_MAX_STEPS,
            steps_executed=0,
            learning_rate_initial=eta,
            learning_rate_final=eta0,
            restarts=0,
            forget_nll_before=forget_stats_before["nll_mean"],
            forget_loglik_before=forget_stats_before["loglikelihood"],
            forget_tokens=forget_stats_before["n_tokens"],
            retain_loss_before=retain_before,
            retain_windows=len(retain_texts),
            fisher_source=self.fisher_source,
            fisher_floor_applied=floor_eff,
            fisher_max=float(max(f.max() for f in fisher_raw)),
            trainable_parameters=n_trainable,
            projection_mode=self.config.projection_mode,
            probe_tolerance=self.config.probe_tolerance,
            probe_sentences=len(probe_texts) if probe_texts is not None else 0,
        )

        self._log(
            f"[unlearning] Fisher: {self.fisher_source}; parametri addestrabili "
            f"{n_trainable} su {int(sum(np.asarray(p).size for p in params))}; "
            f"pavimento F = {floor_eff:.3e} (max grezzo {result.fisher_max:.3e}); "
            f"proiezione = {self.config.projection_mode}"
        )
        self._log(
            f"[unlearning] pre-unlearning: NLL(X_forget) = "
            f"{result.forget_nll_before:.4f} su {result.forget_tokens} token; "
            f"loss(X_retain_proj) = {retain_before:.4f} su {len(retain_texts)} frasi; "
            f"soglia di STOP = {threshold:.4f}"
        )
        if probe_baseline is not None:
            self._log(
                f"[unlearning] probe/canario: {len(probe_texts)} frasi, soglia "
                f"sulla variazione relativa MASSIMA = "
                f"{self.config.probe_tolerance * 100:.1f}% (rispetto a theta_0); "
                "il probe non entra nel gradiente"
            )

        attempt = 0
        stop_reason = None
        while True:
            # Riavvio da theta_0 (identico a ogni tentativo).
            for slot, idx in enumerate(indices):
                np.asarray(params[idx])[...] = theta0[slot]
            step_in_attempt = 0
            restart_requested = False
            # Ultimo stato verificato conforme alla soglia: theta_0 all'inizio,
            # poi lo stato di ogni passo che l'ha rispettata. Serve al rollback.
            theta_last_ok = [np.asarray(p).copy() for p in theta0]
            steps_ok_in_attempt = 0
            while step_in_attempt < self.config.max_steps:
                g_forget = self.trainable_gradients(forget_inputs, forget_targets, indices)
                g_retain = self.trainable_gradients_on_texts(retain_texts, indices)
                if self.config.projection_mode == "metric_orthogonal":
                    g_proj, alpha, den = self.project_metric_orthogonal(
                        g_forget, g_retain, fisher_eff)
                else:
                    g_proj, alpha, den = self.project_in_fisher_metric(
                        g_forget, g_retain, fisher_eff)

                if den <= 0.0 or not np.isfinite(den):
                    stop_reason = STOP_DEGENERATE_PROJECTION
                    self._log(
                        "[unlearning] STOP: proiezione degenere "
                        f"(g_retain . F^-1 . g_retain = {den:.3e}); "
                        "nessuna direzione di protezione disponibile."
                    )
                    break

                direction = g_proj
                norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in g_proj)))
                forget_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in g_forget)))
                # Diagnostica di coerenza: se questo valore e' negativo, il
                # passo NON e' un passo di ascent sulla loss di forget, e la
                # proiezione ha fatto piu' danno che la sua stessa protezione.
                first_order = eta * float(sum(
                    float(np.sum(a * b)) for a, b in zip(g_forget, g_proj)))
                if self.config.max_step_norm and norm > self.config.max_step_norm:
                    scale = self.config.max_step_norm / norm
                    direction = [g * scale for g in g_proj]
                    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in direction)))

                for slot, idx in enumerate(indices):
                    np.asarray(params[idx])[...] = np.asarray(params[idx]) + eta * direction[slot]

                step_in_attempt += 1
                result.steps_executed += 1
                step = result.steps_executed

                retain_now = self.loss_on_texts(retain_texts)
                forget_now = self.loglikelihood(forget_text)
                # Canario: variazione relativa massima, non media. Sul caso peggio
                # una media su 6 frasi puo' nascondere che una frase e' andata a
                # pezzi mentre le altre cinque non si sono mosse.
                probe_max_change = 0.0
                if probe_baseline is not None:
                    probe_now = [self.loglikelihood(t)["loglikelihood"]
                                 for t in probe_texts]
                    probe_max_change = float(relative_probability_change(
                        probe_baseline, probe_now).max())
                    result.probe_max_relative_change_history.append(probe_max_change)
                result.history.append({
                    "step": step,
                    "attempt": attempt,
                    "step_in_attempt": step_in_attempt,
                    "retain_loss": retain_now,
                    "forget_nll": forget_now["nll_mean"],
                    "grad_norm": norm,
                    "forget_grad_norm": forget_norm,
                    "projection_ratio": (norm / forget_norm) if forget_norm > 0 else None,
                    "first_order_forget_delta": first_order,
                    "projection_alpha": alpha,
                    "learning_rate": eta,
                    "probe_max_relative_change": probe_max_change,
                })
                result.projection_coefficients.append(alpha)

                if step <= 5 or step % 10 == 0:
                    probe_txt = "" if probe_baseline is None else (
                        f" probe_max={probe_max_change * 100:7.2f}%")
                    self._log(
                        f"[unlearning] step {step:3d} (tentativo {attempt}) "
                        f"eta={eta:.2e} loss_retain={retain_now:.4f} "
                        f"NLL_forget={forget_now['nll_mean']:.4f} "
                        f"||g_proj||={norm:.3e} (||g_forget||={forget_norm:.3e}, "
                        f"rapporto {norm / forget_norm if forget_norm > 0 else float('nan'):.3e}) "
                        f"delta_forget_1o={first_order:+.3e} alpha={alpha:.3e}"
                        f"{probe_txt}"
                    )

                # --- STOP di sicurezza secondario sul canario ---
                # Ha precedenza sul rollback di retain solo in senso logico: e'
                # comunque uno STOP, e lo stato finale resta un theta verificato
                # conforme a entrambe le soglie.
                if probe_baseline is not None and \
                        probe_max_change > self.config.probe_tolerance:
                    stop_reason = STOP_PROBE_DRIFT
                    if self.config.rollback_on_stop:
                        rolled_back = step_in_attempt - steps_ok_in_attempt
                        for slot, idx in enumerate(indices):
                            np.asarray(params[idx])[...] = theta_last_ok[slot]
                        result.rolled_back_steps += rolled_back
                        self._log(
                            f"[unlearning] rollback: ripristino i pesi all'ultimo "
                            f"stato verificato conforme a entrambe le soglie "
                            f"({rolled_back} passo/i annullato/i)."
                        )
                    self._log(
                        f"[unlearning] STOP: probe/canario, variazione relativa "
                        f"massima {probe_max_change * 100:.3f}% > soglia "
                        f"{self.config.probe_tolerance * 100:.1f}% dopo lo step {step}"
                    )
                    break

                if retain_now > threshold:
                    if step_in_attempt <= self.config.halving_window and \
                            result.restarts < self.config.max_restarts:
                        result.restarts += 1
                        result.halving_events.append({
                            "attempt": attempt,
                            "step": step,
                            "retain_loss": retain_now,
                            "threshold": threshold,
                            "learning_rate_before": eta,
                            "learning_rate_after": eta / 2.0,
                        })
                        self._log(
                            f"[unlearning] loss_retain {retain_now:.4f} > soglia "
                            f"{threshold:.4f} entro i primi {self.config.halving_window} step: "
                            f"dimezzo eta {eta:.2e} -> {eta / 2.0:.2e} e riparto da theta_0 "
                            f"(riavvio {result.restarts}/{self.config.max_restarts})"
                        )
                        eta = eta / 2.0
                        attempt += 1
                        restart_requested = True
                        break
                    stop_reason = STOP_RETAIN_LOSS
                    if self.config.rollback_on_stop:
                        rolled_back = step_in_attempt - steps_ok_in_attempt
                        for slot, idx in enumerate(indices):
                            np.asarray(params[idx])[...] = theta_last_ok[slot]
                        result.rolled_back_steps += rolled_back
                        self._log(
                            f"[unlearning] rollback: ripristino i pesi all'ultimo "
                            f"stato verificato conforme alla soglia "
                            f"({rolled_back} passo/i annullato/i)."
                        )
                    else:
                        result.retain_threshold_violated = True
                    self._log(
                        f"[unlearning] STOP: loss_retain {retain_now:.4f} > soglia "
                        f"{threshold:.4f} (+{self.config.retain_loss_tolerance:.0%} "
                        f"relativo sul valore pre-unlearning {retain_before:.4f}) "
                        f"dopo lo step {step}"
                    )
                    break
                # Passo conforme a TUTTE le soglie attive: diventa il nuovo
                # riferimento per il rollback.
                theta_last_ok = [np.asarray(params[i]).copy() for i in indices]
                steps_ok_in_attempt += 1
            if stop_reason is not None:
                break
            if restart_requested:
                # Dimezzamento di eta gia' eseguito e theta_0 ripristinato:
                # si ricomincia un tentativo, non e' la fine dell'ascent.
                continue
            # max_steps raggiunti senza violare la soglia.
            stop_reason = STOP_MAX_STEPS
            self._log(
                f"[unlearning] STOP: max_steps={self.config.max_steps} raggiunti "
                f"senza superare la soglia sulla loss di retain"
            )
            break

        result.stop_reason = stop_reason
        result.learning_rate_final = eta

        forget_stats_after = self.loglikelihood(forget_text)
        result.forget_nll_after = forget_stats_after["nll_mean"]
        result.forget_loglik_after = forget_stats_after["loglikelihood"]
        result.retain_loss_after = self.loss_on_texts(retain_texts)
        if probe_baseline is not None:
            probe_final = [self.loglikelihood(t)["loglikelihood"]
                           for t in probe_texts]
            per_sentence = relative_probability_change(probe_baseline, probe_final)
            result.probe_max_relative_change_after = float(per_sentence.max())
        return result


def fit_fisher(trainer, dataset, num_samples: int = 100):
    """
    Calcola la Micro-Fisher con il codice gia' esistente del Trainer
    (`Trainer.compute_fisher_matrix`, ovvero la stessa usata dal
    Continual Learning/EWC sullo spazio LoRA) e la restituisce.

    Non e' una ricalcolazione su questo modulo: e' un wrapper esplicito, per
    dichiarare nella pipeline che la Fisher NON viene stimata da zero qui.
    """
    return trainer.compute_fisher_matrix(dataset, num_samples=num_samples)
