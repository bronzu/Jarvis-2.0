import numpy as np
import pytest

from neural_core.loss import CrossEntropyLoss, MSELoss


def test_mse_loss_still_works():
    loss = MSELoss()
    result = loss.forward(3.0, 1.0)
    assert result == 4.0


def test_cross_entropy_loss_forward():
    loss = CrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
    targets = [0, 1]
    result = loss.forward(logits, targets)
    assert isinstance(result, float)
    assert result > 0


def test_cross_entropy_loss_backward():
    loss = CrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.1]])
    targets = [0]
    grad = loss.backward(logits, targets)
    assert grad.shape == (1, 3)
    assert np.isclose(np.sum(grad), 0.0)


def test_cross_entropy_loss_perfect_prediction():
    loss = CrossEntropyLoss()
    logits = np.array([[10.0, -1.0, -2.0]])
    targets = [0]
    result = loss.forward(logits, targets)
    assert result < 0.01


# ======================================================================
# LOGITS 3D: modello autoregressivo (batch, seq_len, vocab)
# ======================================================================


def test_cross_entropy_su_logits_3d():
    loss = CrossEntropyLoss()
    logits = np.random.RandomState(0).randn(2, 4, 6)
    targets = np.array([[1, 2, 3, 4], [0, 1, 2, 3]])

    value = loss.forward(logits, targets)

    assert isinstance(value, float)
    assert value > 0


def test_cross_entropy_backward_su_logits_3d():
    loss = CrossEntropyLoss()
    logits = np.random.RandomState(1).randn(2, 3, 5)
    targets = np.array([[1, 2, 3], [0, 1, 2]])

    grad = loss.backward(logits, targets)

    assert grad.shape == logits.shape
    assert np.isclose(np.sum(grad), 0.0, atol=1e-9)


def test_cross_entropy_3d_equivale_al_2d_appiattito():
    loss = CrossEntropyLoss()
    logits = np.random.RandomState(2).randn(3, 4, 7)
    targets = np.random.RandomState(3).randint(0, 7, size=(3, 4))

    valore_3d = loss.forward(logits, targets)
    valore_2d = loss.forward(logits.reshape(-1, 7), targets.reshape(-1))

    assert np.isclose(valore_3d, valore_2d)


def test_cross_entropy_ignore_index_esclude_le_posizioni():
    loss = CrossEntropyLoss(ignore_index=-100)
    logits = np.array([[[5.0, 0.0], [0.0, 5.0]]])

    completo = loss.forward(logits, np.array([[0, 1]]))
    con_ignore = loss.forward(logits, np.array([[0, -100]]))
    solo_primo = loss.forward(logits[:, :1, :], np.array([[0]]))

    assert np.isclose(completo, con_ignore)
    assert np.isclose(con_ignore, solo_primo)


def test_cross_entropy_ignore_index_azzera_il_gradiente():
    loss = CrossEntropyLoss(ignore_index=-100)
    logits = np.random.RandomState(4).randn(1, 3, 4)
    targets = np.array([[1, -100, 2]])

    grad = loss.backward(logits, targets)

    assert np.allclose(grad[0, 1], 0.0)
    assert not np.allclose(grad[0, 0], 0.0)


def test_cross_entropy_tutti_target_ignorati():
    loss = CrossEntropyLoss(ignore_index=-100)
    logits = np.random.RandomState(5).randn(1, 2, 3)
    targets = np.array([[-100, -100]])

    assert loss.forward(logits, targets) == 0.0
    assert np.allclose(loss.backward(logits, targets), 0.0)


def test_cross_entropy_rifiuta_target_incoerenti():
    loss = CrossEntropyLoss()
    logits = np.random.RandomState(6).randn(2, 3, 4)

    with pytest.raises(ValueError):
        loss.forward(logits, np.array([[1, 2], [3, 0]]))


def test_cross_entropy_logits_1d():
    loss = CrossEntropyLoss()
    value = loss.forward(np.array([2.0, 1.0, 0.0]), [0])

    assert isinstance(value, float)
    assert value > 0
