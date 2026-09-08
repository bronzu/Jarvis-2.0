import numpy as np
import pytest

from neural_core.embeddings import Embeddings


def test_embeddings_creation():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    assert embeddings.vocab_size == 10
    assert embeddings.embedding_dim == 4


def test_embedding_vector_dimension():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    vector = embeddings.get(2)

    assert len(vector) == 4


def test_same_token_returns_same_embedding():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    vector_a = embeddings.get(3)
    vector_b = embeddings.get(3)

    assert vector_a == vector_b


def test_different_tokens_can_have_different_embeddings():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    embeddings.set(1, [1.0, 2.0, 3.0, 4.0])
    embeddings.set(2, [4.0, 3.0, 2.0, 1.0])

    assert embeddings.get(1) != embeddings.get(2)


def test_forward_returns_embedding_for_each_token():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    embeddings.set(1, [1.0, 2.0, 3.0, 4.0])
    embeddings.set(2, [4.0, 3.0, 2.0, 1.0])

    result = embeddings.forward([1, 2])

    assert len(result) == 2
    assert result[0] == [1.0, 2.0, 3.0, 4.0]
    assert result[1] == [4.0, 3.0, 2.0, 1.0]


def test_invalid_token_id():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    with pytest.raises(IndexError):
        embeddings.get(10)


def test_invalid_embedding_dimension():
    embeddings = Embeddings(vocab_size=10, embedding_dim=4)

    with pytest.raises(ValueError):
        embeddings.set(1, [1.0, 2.0])


def test_invalid_configuration():
    with pytest.raises(ValueError):
        Embeddings(vocab_size=0, embedding_dim=4)

    with pytest.raises(ValueError):
        Embeddings(vocab_size=10, embedding_dim=0)


# ======================================================================
# STORAGE NUMPY, INIZIALIZZAZIONE E BACKWARD
# ======================================================================


def test_inizializzazione_default_a_zero():
    embeddings = Embeddings(vocab_size=5, embedding_dim=3)

    assert embeddings.weights.shape == (5, 3)
    assert np.allclose(embeddings.weights, 0.0)
    assert embeddings.init_scale == 0.0


def test_inizializzazione_casuale_deterministica_con_seed():
    primo = Embeddings(vocab_size=6, embedding_dim=4, init_scale=0.02, seed=7)
    secondo = Embeddings(vocab_size=6, embedding_dim=4, init_scale=0.02, seed=7)

    assert np.allclose(primo.weights, secondo.weights)
    assert not np.allclose(primo.weights, 0.0)
    assert not np.allclose(primo.weights[0], primo.weights[1])


def test_init_scale_negativo_rifiutato():
    with pytest.raises(ValueError):
        Embeddings(vocab_size=4, embedding_dim=2, init_scale=-0.1)


def test_forward_matrix_supporta_batch_di_sequenze():
    embeddings = Embeddings(vocab_size=8, embedding_dim=3, init_scale=0.1, seed=1)
    token_ids = np.array([[1, 2, 3], [4, 5, 6]])

    matrix = embeddings.forward_matrix(token_ids)

    assert matrix.shape == (2, 3, 3)
    assert np.allclose(matrix[0, 0], embeddings.weights[1])


def test_forward_matrix_rifiuta_indici_fuori_vocabolario():
    embeddings = Embeddings(vocab_size=4, embedding_dim=2)

    with pytest.raises(IndexError):
        embeddings.forward_matrix(np.array([[0, 9]]))


def test_backward_accumula_i_gradienti_per_token_ripetuti():
    embeddings = Embeddings(vocab_size=5, embedding_dim=2)
    token_ids = np.array([[1, 1, 3]])
    grad_output = np.ones((1, 3, 2))

    grad_weights = embeddings.backward(token_ids, grad_output)

    assert grad_weights.shape == (5, 2)
    assert np.allclose(grad_weights[1], 2.0)
    assert np.allclose(grad_weights[3], 1.0)
    assert np.allclose(grad_weights[0], 0.0)


def test_forward_e_forward_matrix_coincidono():
    embeddings = Embeddings(vocab_size=6, embedding_dim=3, init_scale=0.05, seed=2)
    token_ids = [0, 2, 4]

    lista = embeddings.forward(token_ids)
    matrice = embeddings.forward_matrix(token_ids)

    assert np.allclose(np.asarray(lista), matrice)


def test_token_id_numpy_accettato():
    embeddings = Embeddings(vocab_size=4, embedding_dim=2)
    embeddings.set(np.int64(2), [1.0, 2.0])

    assert embeddings.get(np.int64(2)) == [1.0, 2.0]


def test_token_id_booleano_rifiutato():
    embeddings = Embeddings(vocab_size=4, embedding_dim=2)

    with pytest.raises(TypeError):
        embeddings.get(True)
