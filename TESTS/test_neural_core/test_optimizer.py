import numpy as np
import pytest

from neural_core.optimizer import Adam, LearningRateScheduler, SGD


def test_sgd_still_works():
    sgd = SGD(learning_rate=0.1)
    weights = [1.0, 2.0]
    bias = 0.5
    w_grads = [0.1, 0.2]
    b_grad = 0.05
    updated_w, updated_b = sgd.update(weights, bias, w_grads, b_grad)
    assert updated_w == [0.99, 1.98]
    assert updated_b == 0.495


def test_adam_update_changes_parameters():
    np.random.seed(42)
    params = [np.array([1.0, 2.0])]
    grads = [np.array([0.5, 0.5])]
    adam = Adam(learning_rate=0.1)
    original = params[0].copy()
    updated = adam.update(params, grads)
    assert np.any(updated[0] != original)


def test_adam_tracks_moments():
    np.random.seed(42)
    params = [np.array([1.0])]
    grads = [np.array([0.1])]
    adam = Adam(learning_rate=0.1)
    adam.update(params, grads)
    assert adam.m[0] is not None
    assert adam.v[0] is not None
    assert adam.t == 1


def test_adam_multiple_updates():
    np.random.seed(42)
    params = [np.array([1.0, 2.0])]
    grads = [np.array([0.1, 0.1])]
    adam = Adam(learning_rate=0.1)
    adam.update(params, grads)
    adam.update(params, grads)
    assert adam.t == 2


# ======================================================================
# ADAM: learning rate dinamico, weight decay, stato per il resume
# ======================================================================


def test_adam_set_learning_rate():
    adam = Adam(learning_rate=0.001)
    assert adam.set_learning_rate(0.05) == 0.05
    assert adam.learning_rate == 0.05

    with pytest.raises(ValueError):
        adam.set_learning_rate(-1)


def test_adam_rifiuta_gradienti_non_allineati():
    adam = Adam()
    with pytest.raises(ValueError):
        adam.update([np.zeros(2)], [np.zeros(2), np.zeros(2)])


def test_adam_weight_decay_solo_sulle_matrici():
    adam = Adam(learning_rate=0.1, weight_decay=0.5)
    matrice = np.ones((2, 2))
    bias = np.ones(2)
    zero_grad = [np.zeros((2, 2)), np.zeros(2)]
    original = matrice.copy()
    updated = adam.update([matrice, bias], zero_grad)

    assert np.any(updated[0] != original)
    assert np.allclose(updated[1], bias)


def test_adam_state_dict_permette_il_resume():
    adam = Adam(learning_rate=0.01)
    params = [np.array([1.0, 2.0])]
    grads = [np.array([0.3, -0.2])]
    adam.update(params, grads)
    adam.update(params, grads)

    state = adam.state_dict()
    ripristinato = Adam().load_state_dict(state)

    assert ripristinato.t == adam.t
    assert ripristinato.learning_rate == adam.learning_rate
    assert np.allclose(ripristinato.m[0], adam.m[0])
    assert np.allclose(ripristinato.v[0], adam.v[0])

    atteso = adam.update(params, grads)
    ottenuto = ripristinato.update(params, grads)
    assert np.allclose(atteso[0], ottenuto[0])


def test_adam_load_state_dict_rifiuta_valore_non_valido():
    with pytest.raises(ValueError):
        Adam().load_state_dict("stato")


def test_adam_reset_azzera_i_momenti():
    adam = Adam()
    adam.update([np.ones(2)], [np.ones(2)])
    adam.reset()

    assert adam.t == 0
    assert adam.m is None
    assert adam.v is None


# ======================================================================
# SCHEDULER: warmup e decadimento
# ======================================================================


def test_warmup_lineare_raggiunge_il_learning_rate_base():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=4, total_steps=100, schedule="constant"
    )

    warmup = [scheduler.learning_rate(step) for step in range(4)]

    assert warmup == sorted(warmup)
    assert np.isclose(warmup[0], 0.01 / 4)
    assert np.isclose(warmup[-1], 0.01)
    assert np.isclose(scheduler.learning_rate(4), 0.01)


def test_schedule_costante_dopo_il_warmup():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.005, warmup_steps=2, schedule="constant"
    )

    assert np.isclose(scheduler.learning_rate(2), 0.005)
    assert np.isclose(scheduler.learning_rate(500), 0.005)


def test_schedule_lineare_decade_fino_al_minimo():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=0, total_steps=10,
        schedule="linear", min_learning_rate=0.001,
    )

    assert np.isclose(scheduler.learning_rate(0), 0.01)
    assert np.isclose(scheduler.learning_rate(10), 0.001)
    assert scheduler.learning_rate(5) < scheduler.learning_rate(1)


def test_schedule_cosine_decade_fino_al_minimo():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.02, warmup_steps=0, total_steps=8,
        schedule="cosine", min_learning_rate=0.0,
    )

    valori = [scheduler.learning_rate(step) for step in range(9)]

    assert np.isclose(valori[0], 0.02)
    assert np.isclose(valori[-1], 0.0, atol=1e-12)
    assert valori == sorted(valori, reverse=True)


def test_scheduler_oltre_total_steps_resta_al_minimo():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, total_steps=5, schedule="cosine", min_learning_rate=0.002
    )

    assert np.isclose(scheduler.learning_rate(50), 0.002)


def test_scheduler_step_avanza_il_contatore():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=2, total_steps=10, schedule="linear"
    )

    primo = scheduler.step()
    secondo = scheduler.step()

    assert scheduler.last_step == 2
    assert secondo > primo


def test_scheduler_state_dict_roundtrip():
    scheduler = LearningRateScheduler(
        base_learning_rate=0.01, warmup_steps=3, total_steps=50,
        schedule="cosine", min_learning_rate=0.001,
    )
    scheduler.step()
    scheduler.step()

    ripristinato = LearningRateScheduler(base_learning_rate=0.1).load_state_dict(
        scheduler.state_dict()
    )

    assert ripristinato.base_learning_rate == 0.01
    assert ripristinato.warmup_steps == 3
    assert ripristinato.total_steps == 50
    assert ripristinato.schedule == "cosine"
    assert ripristinato.min_learning_rate == 0.001
    assert ripristinato.last_step == 2
    assert np.isclose(ripristinato.learning_rate(7), scheduler.learning_rate(7))


def test_scheduler_parametri_non_validi():
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0)
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0.01, warmup_steps=-1)
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0.01, total_steps=0)
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0.01, schedule="magica")
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0.01, min_learning_rate=1.0)
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0.01).learning_rate(-1)


def test_scheduler_load_state_dict_rifiuta_valore_non_valido():
    with pytest.raises(ValueError):
        LearningRateScheduler(base_learning_rate=0.01).load_state_dict(None)
