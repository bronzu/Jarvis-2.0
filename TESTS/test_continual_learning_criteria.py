"""
Test di regressione sui quattro criteri di validazione del Continual Learning.

I test non ricalcolano gli esperimenti A->B, che costano minuti: verificano
che le strutture che li rendono possibili restino corrette e che le soglie
dichiarate nella documentazione siano ancora sostenute dai test rapidi
disponibili.

La verifica end-to-end dei criteri (c) e (d) su corpus reali vive in
`experiments/phase4_continual_learning/validate.py`; qui si controlla che il
meccanismo che produce quei numeri non si rompa.

SOGLIE: i valori qui sotto sono i limiti dichiarati in
`DOCUMENTAZIONE/technical-acquisition-package/17_MODEL_TRAINING_PROVENANCE.md`
sezione 13.7. Se un test fallisce, o il meccanismo si e' rotto o la soglia
documentata non e' piu' vera: in entrambi i casi la documentazione va aggiornata.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "experiments" / "phase4_continual_learning"))

import corpus  # noqa: E402
import validate as V  # noqa: E402

from neural_core.network import MicroTransformer  # noqa: E402
from neural_core.tokenizer import Tokenizer  # noqa: E402
from neural_core.loss import CrossEntropyLoss  # noqa: E402
from neural_core.optimizer import Adam  # noqa: E402
from neural_core.trainer import Trainer  # noqa: E402

# Soglie dichiarate nella documentazione (sezione 13.7).
THRESHOLD_MEMORY_REDUCTION_PCT = 90.0
THRESHOLD_RETENTION_DEGRADATION_PCT = 2.0
THRESHOLD_NEW_TASK_ACCURACY_GAP_PP = 2.0


# --------------------------------------------------------------------------
# Corpus reali
# --------------------------------------------------------------------------

class TestRealCorpus:
    """I task A/B devono essere corpus reali del repository, non frasi sintetiche."""

    def test_task_a_e_prosa_italiana_reale(self):
        text, sources = corpus.load_task_a_text()
        assert len(sources) >= 3, f"task A troppo pochi sorgenti: {sources}"
        assert all(s.endswith(".md") for s in sources)
        words = len(text.split())
        assert words >= 5000, f"task A troppo corto ({words} parole): era un giocattolo"

    def test_task_b_e_codice_python_reale(self):
        text, sources = corpus.load_task_b_text()
        assert len(sources) >= 5, f"task B troppo pochi sorgenti: {sources}"
        assert all(s.endswith(".py") for s in sources)
        words = len(text.split())
        assert words >= 5000, f"task B troppo corto ({words} parole): era un giocattolo"

    def test_i_due_task_sono_distinguibili(self):
        """Prosa e codice non devono essere lo stesso segnale travestito."""
        text_a, _ = corpus.load_task_a_text()
        text_b, _ = corpus.load_task_b_text()
        a, b = set(text_a.split()), set(text_b.split())
        jaccard = len(a & b) / len(a | b)
        assert jaccard < 0.5, (
            f"i due task sono troppo simili (Jaccard {jaccard:.3f}): la ritenzione "
            f"sarebbe misurata su un segnale non distinto"
        )
        # il codice deve avere i suoi marker, la prosa non
        assert "def " in text_b, "il task B non contiene codice Python"
        assert "def " not in text_a.split("```")[0][:5000], "il task A contiene codice"

    def test_corpus_deterministico(self):
        a1, _ = corpus.load_task_a_text()
        a2, _ = corpus.load_task_a_text()
        b1, _ = corpus.load_task_b_text()
        b2, _ = corpus.load_task_b_text()
        assert a1 == a2 and b1 == b2, "il corpus non e' deterministico"


# --------------------------------------------------------------------------
# Criterio (a) — riduzione della memoria
# --------------------------------------------------------------------------

def _fisher_sizes(rank, samples=10):
    model = MicroTransformer(seed=V.SEED, lora_rank=rank, lora_alpha=V.LORA_ALPHA,
                             **V.PRODUCTION_CONFIG)
    trainer = Trainer(model=model,
                      tokenizer=Tokenizer(max_vocab_size=V.PRODUCTION_CONFIG["vocab_size"],
                                          mode="whitespace"),
                      loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.01),
                      seed=V.SEED)
    vocab = V.PRODUCTION_CONFIG["vocab_size"]
    rng = np.random.default_rng(V.SEED)
    ds = (rng.integers(0, vocab, size=(samples, V.CONTEXT_SIZE)),
          rng.integers(0, vocab, size=(samples, V.CONTEXT_SIZE)))
    trainer.compute_fisher_matrix(ds, num_samples=samples)
    total = (V.dict_bytes(trainer.fisher_matrix) + V.dict_bytes(trainer.optimal_weights))
    entries = sum(int(np.asarray(v).size) for v in trainer.fisher_matrix.values())
    return total, entries, model.parameter_count(), model.trainable_parameter_count()


class TestCriterionAMemory:
    def test_la_fisher_full_parameter_copre_tutti_i_parametri(self):
        total, entries, n_params, n_trainable = _fisher_sizes(0)
        assert entries == n_params == n_trainable, (
            f"la versione full-parameter deve penalizzare tutti i parametri: "
            f"entries {entries}, params {n_params}, addestrabili {n_trainable}")
        assert total == n_params * 8 * 2, (
            f"attesi fisher + optimal = 2 float64 per parametro: {total} != {n_params*16}")

    @pytest.mark.parametrize("rank", [4, 16])
    def test_la_fisher_lora_copre_solo_le_matrici_lora(self, rank):
        total, entries, n_params, n_trainable = _fisher_sizes(rank)
        assert entries == n_trainable, (
            f"la Micro-Fisher deve coprire solo i parametri addestrabili: "
            f"entries {entries} != addestrabili {n_trainable}")
        assert entries < n_params * 0.01, (
            f"rank {rank}: la Fisher e' ancora troppo grande ({entries:,} voci "
            f"su {n_params:,} parametri)")

    @pytest.mark.parametrize("rank", [4, 16])
    def test_riduzione_memoria_supera_la_soglia(self, rank):
        full_bytes, _, _, _ = _fisher_sizes(0)
        lora_bytes, _, _, _ = _fisher_sizes(rank)
        reduction = 100.0 * (1.0 - lora_bytes / full_bytes)
        assert reduction > THRESHOLD_MEMORY_REDUCTION_PCT, (
            f"rank {rank}: riduzione memoria {reduction:.4f}% sotto la soglia "
            f"documentata del {THRESHOLD_MEMORY_REDUCTION_PCT}%")

    def test_la_riduzione_e_verificata_sulle_strutture_persistenti(self):
        """
        La riduzione dichiarata riguarda fisher_matrix + optimal_weights.

        NON riguarda il picco di RSS del processo: quel valore e' dominato dal
        working set di forward/backward, identico nelle due varianti perche' il
        modello resta di 2,7M parametri. Il test fissa la distinzione perche' e'
        facile riportare il 99,78% come se fosse la RAM totale del processo.
        """
        import resource
        full_bytes, _, _, _ = _fisher_sizes(0)
        lora_bytes, _, _, _ = _fisher_sizes(16)
        assert lora_bytes < full_bytes / 100
        # il modello completo resta comunque grande: LoRA aggiunge, non toglie
        _, _, n_params, n_trainable = _fisher_sizes(16)
        assert n_params > 2_000_000, "la configurazione di produzione non e' piu' tale"
        assert n_trainable < n_params * 0.01
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
        assert peak > 0  # sanity: la sonda e' utilizzabile


# --------------------------------------------------------------------------
# Strutture per i criteri (c) e (d)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_task_data():
    tok, text_a, text_b, _, _ = V.build_tokenizer()
    a_in, a_tg = V.make_dataset(text_a, tok)
    b_in, b_tg = V.make_dataset(text_b, tok)
    a_tr, a_tg_tr, a_va, a_tg_va = V.split_train_val(a_in, a_tg)
    b_tr, b_tg_tr, b_va, b_tg_va = V.split_train_val(b_in, b_tg)
    return tok, (a_tr, a_tg_tr, a_va, a_tg_va, b_tr, b_tg_tr, b_va, b_tg_va)


class TestRetentionAndLearningHarness:
    def test_split_di_validazione_disgiunto_dall_addestramento(self, two_task_data):
        _, (a_tr, _, a_va, _, b_tr, _, b_va, _) = two_task_data
        assert len(a_va) > 0 and len(b_va) > 0
        assert len(a_va) == pytest.approx(len(a_tr) * 0.25 / 0.75, rel=0.2)
        # nessuna sovrapposizione: la ritenzione non puo' essere misurata su
        # campioni memorizzati durante l'addestramento
        assert len(set(map(tuple, a_tr.tolist())) & set(map(tuple, a_va.tolist()))) < \
            0.05 * len(a_va)

    def test_la_perdita_e_la_accuratezza_restano_legibili(self, two_task_data):
        """Interrompe l'esperimento se la loss resta a livello casuale."""
        tok, data = two_task_data
        a_tr, a_tg_tr, a_va, a_tg_va = data[:4]
        model = V.build_lora_model(len(tok.vocabulary))
        trainer = V.make_trainer(model, tok)
        before = trainer.evaluate(a_va, a_tg_va, batch_size=64)
        trainer.train(a_tr, a_tg_tr, epochs=3, batch_size=64)
        after = trainer.evaluate(a_va, a_tg_va, batch_size=64)
        assert after < before, (
            f"il task A non e' appreso: la loss di validazione non scende "
            f"({before:.4f} -> {after:.4f}); ritenzione e apprendimento non "
            f"sarebbero misurabili"
        )

    def test_la_ritenzione_e_misurata_su_dati_mai_visti(self, two_task_data):
        """
        Il perimetro di (c): la valutazione post-task-B usa lo split di
        validazione, non i dati di addestramento del task A.
        """
        tok, data = two_task_data
        a_tr, a_tg_tr, a_va, a_tg_va = data[:4]
        model = V.build_lora_model(len(tok.vocabulary))
        trainer = V.make_trainer(model, tok)
        trainer.train(a_tr, a_tg_tr, epochs=2, batch_size=64)
        # la loss restituita da evaluate su a_va non puo' coincidere con quella
        # su a_tr: se coincidessero la valutazione starebbe riusando i dati di
        # addestramento
        loss_val = trainer.evaluate(a_va, a_tg_va, batch_size=64)
        loss_tr = trainer.evaluate(a_tr, a_tg_tr, batch_size=64)
        assert loss_val != loss_tr, (
            "la valutazione di ritenzione riusa i dati di addestramento: "
            "la misura sarebbe tautologica")

    def test_il_confronto_per_il_criterio_d_e_a_parita_di_condizioni(self, two_task_data):
        """
        Lora e fine-tuning completo devono partire dagli stessi dati, dagli
        stessi iperparametri e dallo stesso seed: altrimenti il gap non
        significa nulla.
        """
        tok, data = two_task_data
        lora_model = V.build_lora_model(len(tok.vocabulary))
        full_model = V.build_full_model(len(tok.vocabulary))
        assert lora_model.trainable_parameter_count() < full_model.trainable_parameter_count()
        assert full_model.trainable_parameter_count() == full_model.parameter_count()
        a, b = V.make_trainer(lora_model, tok), V.make_trainer(full_model, tok)
        assert a.optimizer.learning_rate == b.optimizer.learning_rate
        assert a.seed == b.seed
        # lo stesso Task B alimenta entrambi i rami: il confronto di (d) e'
        # valido solo se i dati sono identici, non equivalenti
        b_in, b_tg = data[4], data[5]
        assert b_in.shape[0] > 0 and b_in.shape == b_tg.shape
        # una epoch basta a produrre due loss non nulle: questo test verifica
        # la confrontabilita' del setup, non le prestazioni (quelle stanno in
        # `validate.py`)
        lora_run = V._run_scenario(V.LORA_RANK, V.EWC_LAMBDA, False, tok, data, epochs=1)
        full_run = V._run_scenario(0, 0.0, False, tok, data, epochs=1)
        assert lora_run["b_loss"] > 0 and full_run["b_loss"] > 0, (
            "una delle due loss e' zero: il gap del criterio (d) non e' calcolabile")
