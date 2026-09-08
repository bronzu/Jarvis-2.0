import numpy as np
import pytest

from neural_core.loss import CrossEntropyLoss
from neural_core.model import causal_mask
from neural_core.network import (
    DEFAULT_CONTEXT_SIZE,
    MIN_CONTEXT_SIZE,
    SUPPORTED_CONTEXT_SIZES,
    FeedForward,
    LayerNorm,
    MicroTransformer,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerBlock,
)
from neural_core.tokenizer import Tokenizer


def _small_model(vocab_size=13, **overrides):
    config = {
        "vocab_size": vocab_size,
        "d_model": 16,
        "num_heads": 2,
        "d_ff": 32,
        "num_layers": 2,
        "max_len": MIN_CONTEXT_SIZE,
        "seed": 11,
    }
    config.update(overrides)
    return MicroTransformer(**config)


def test_micro_transformer_inizializzazione():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao JARVIS test")
    vocab_size = len(tokenizer.vocabulary)
    model = MicroTransformer(vocab_size=vocab_size)
    assert model is not None
    assert model.vocab_size == vocab_size
    assert model.d_model == 64
    assert len(model.layers) == 2


def test_micro_transformer_dimensione_input_output():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro")
    vocab_size = len(tokenizer.vocabulary)
    model = MicroTransformer(vocab_size=vocab_size)
    token_ids = tokenizer.encode("uno due")
    logits = model.forward(token_ids)
    assert logits.shape[0] == 1
    assert logits.shape[1] == len(token_ids)
    assert logits.shape[2] == vocab_size


def test_micro_transformer_forward_pass_singolo_token():
    tokenizer = Tokenizer()
    tokenizer.fit("ciao mondo")
    vocab_size = len(tokenizer.vocabulary)
    model = MicroTransformer(vocab_size=vocab_size)
    token_ids = [tokenizer.vocabulary["ciao"]]
    logits = model.forward(token_ids)
    assert logits.shape == (1, 1, vocab_size)


def test_micro_transformer_output_numerico_valido():
    tokenizer = Tokenizer()
    tokenizer.fit("prova test output")
    vocab_size = len(tokenizer.vocabulary)
    model = MicroTransformer(vocab_size=vocab_size)
    token_ids = tokenizer.encode("prova")
    logits = model.forward(token_ids)
    assert np.all(np.isfinite(logits))
    assert not np.any(np.isnan(logits))
    assert logits.shape[-1] == vocab_size


def test_micro_transformer_causal_mask_autoregressiva():
    tokenizer = Tokenizer()
    tokenizer.fit("uno due tre quattro cinque")
    vocab_size = len(tokenizer.vocabulary)
    model = MicroTransformer(vocab_size=vocab_size)
    token_ids = tokenizer.encode("uno due tre")
    logits = model.forward(token_ids)
    assert logits.shape == (1, 3, vocab_size)
    assert np.all(np.isfinite(logits))


# ======================================================================
# COMPONENTI: LayerNorm, FeedForward, MultiHeadAttention, PositionalEncoding
# ======================================================================


def test_layer_norm_normalizza_e_ha_parametri():
    norm = LayerNorm(4)
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    out = norm.forward(x)

    assert out.shape == x.shape
    assert np.isclose(np.mean(out), 0.0, atol=1e-6)
    assert np.isclose(np.std(out), 1.0, atol=1e-3)
    assert len(norm.parameters()) == 2


def test_layer_norm_backward_dimensioni_e_gradienti():
    norm = LayerNorm(4)
    x = np.random.RandomState(0).randn(2, 3, 4)
    norm.forward(x)
    grad_input = norm.backward(np.ones_like(x))

    assert grad_input.shape == x.shape
    assert norm.grad_gamma.shape == (4,)
    assert norm.grad_beta.shape == (4,)
    assert np.allclose(norm.grad_beta, 2 * 3)


def test_layer_norm_backward_richiede_forward():
    norm = LayerNorm(4)
    with pytest.raises(RuntimeError):
        norm.backward(np.ones((1, 1, 4)))


def test_feed_forward_dimensioni_e_backward():
    ff = FeedForward(8, 16, seed=1)
    x = np.random.RandomState(1).randn(2, 5, 8)
    out = ff.forward(x)
    grad_input = ff.backward(np.ones_like(out))

    assert out.shape == (2, 5, 8)
    assert grad_input.shape == (2, 5, 8)
    assert ff.grad_W1.shape == (8, 16)
    assert ff.grad_W2.shape == (16, 8)
    assert len(ff.parameters()) == len(ff.gradients()) == 4


def test_multi_head_attention_richiede_d_model_divisibile():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, num_heads=4)


def test_multi_head_attention_pesi_causali_e_normalizzati():
    attention = MultiHeadAttention(d_model=8, num_heads=2, seed=2)
    x = np.random.RandomState(2).randn(1, 4, 8)
    attention.forward(x, mask=causal_mask(4))
    weights = attention.attention_weights()

    assert weights.shape == (1, 2, 4, 4)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)

    future = np.triu(np.ones((4, 4), dtype=bool), 1)
    assert np.allclose(weights[..., future], 0.0)


def test_multi_head_attention_backward_dimensioni():
    attention = MultiHeadAttention(d_model=8, num_heads=2, seed=3)
    x = np.random.RandomState(3).randn(2, 4, 8)
    out = attention.forward(x, mask=causal_mask(4))
    grad_input = attention.backward(np.ones_like(out))

    assert out.shape == x.shape
    assert grad_input.shape == x.shape
    assert len(attention.parameters()) == len(attention.gradients()) == 8
    for parameter, gradient in zip(attention.parameters(), attention.gradients()):
        assert parameter.shape == gradient.shape


def test_transformer_block_residual_connection():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, seed=4)
    x = np.zeros((1, 3, 8))
    out = block.forward(x, mask=causal_mask(3))

    assert out.shape == x.shape
    grad_input = block.backward(np.ones_like(out))
    assert grad_input.shape == x.shape
    assert len(block.parameters()) == 16


def test_positional_encoding_e_limite_context_window():
    encoding = PositionalEncoding(8, max_len=16)
    x = np.zeros((1, 5, 8))
    out = encoding.forward(x)

    assert out.shape == x.shape
    assert not np.allclose(out[0, 0], out[0, 1])

    with pytest.raises(ValueError):
        encoding.forward(np.zeros((1, 17, 8)))


def test_positional_encoding_supporta_d_model_dispari():
    encoding = PositionalEncoding(7, max_len=8)
    out = encoding.forward(np.zeros((1, 4, 7)))
    assert out.shape == (1, 4, 7)
    assert np.all(np.isfinite(out))


# ======================================================================
# MODELLO: forward pass, dimensioni, batch, masking
# ======================================================================


def test_micro_transformer_forward_batch():
    model = _small_model()
    token_ids = np.array([[1, 2, 3], [4, 5, 6]])
    logits = model.forward(token_ids)

    assert logits.shape == (2, 3, model.vocab_size)
    assert np.all(np.isfinite(logits))


def test_micro_transformer_forward_accetta_token_singolo_intero():
    model = _small_model()
    logits = model.forward(3)
    assert logits.shape == (1, 1, model.vocab_size)


def test_micro_transformer_rifiuta_token_fuori_vocabolario():
    model = _small_model(vocab_size=5)
    with pytest.raises(IndexError):
        model.forward([0, 1, 99])


def test_micro_transformer_rifiuta_sequenza_vuota():
    model = _small_model()
    with pytest.raises(ValueError):
        model.forward([[]])


def test_micro_transformer_masking_impedisce_dipendenza_dal_futuro():
    """
    Il causal masking garantisce che i logits della posizione t non cambino
    se si modificano i token successivi a t.
    """
    model = _small_model(vocab_size=20)
    base = np.array([[1, 2, 3, 4, 5]])
    modified = np.array([[1, 2, 3, 17, 19]])

    logits_base = model.forward(base).copy()
    logits_modified = model.forward(modified).copy()

    assert np.allclose(logits_base[:, :3, :], logits_modified[:, :3, :])
    assert not np.allclose(logits_base[:, 3, :], logits_modified[:, 3, :])


def test_micro_transformer_attention_weights_causali_su_tutti_i_layer():
    model = _small_model(vocab_size=12, num_layers=3)
    model.forward([[1, 2, 3, 4]])

    future = np.triu(np.ones((4, 4), dtype=bool), 1)
    for layer_index in range(3):
        weights = model.attention_weights(layer_index)
        assert weights.shape == (1, model.num_heads, 4, 4)
        assert np.allclose(weights[..., future], 0.0)
        assert np.allclose(np.sum(weights, axis=-1), 1.0)


def test_micro_transformer_attention_weights_indice_non_valido():
    model = _small_model()
    model.forward([[1, 2]])
    with pytest.raises(IndexError):
        model.attention_weights(99)


# ======================================================================
# MODELLO: backward pass
# ======================================================================


def test_micro_transformer_backward_restituisce_gradienti_coerenti():
    model = _small_model()
    inputs = np.array([[1, 2, 3, 4]])
    targets = np.array([[2, 3, 4, 5]])

    model.forward(inputs, training=False)
    gradients = model.backward(targets)
    parameters = model.get_parameters()

    assert len(gradients) == len(parameters)
    for parameter, gradient in zip(parameters, gradients):
        assert np.asarray(parameter).shape == np.asarray(gradient).shape
        assert np.all(np.isfinite(gradient))


def test_micro_transformer_backward_richiede_forward():
    model = _small_model()
    with pytest.raises(RuntimeError):
        model.backward(np.array([[1, 2]]))


def test_micro_transformer_backward_rifiuta_target_incoerenti():
    model = _small_model()
    model.forward(np.array([[1, 2, 3]]))
    with pytest.raises(ValueError):
        model.backward(np.array([[1, 2, 3], [1, 2, 3]]))


def test_micro_transformer_gradiente_numerico_corretto():
    """
    Verifica analitica del backward: confronto con il gradiente numerico
    calcolato per differenze finite centrali.
    """
    np.random.seed(7)
    model = MicroTransformer(
        vocab_size=9, d_model=8, num_heads=2, d_ff=16, num_layers=2,
        max_len=MIN_CONTEXT_SIZE, seed=5,
    )
    loss_function = CrossEntropyLoss()
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    targets = np.array([[2, 3, 4], [5, 6, 7]])

    model.forward(inputs)
    gradients = model.backward(targets)
    parameters = model.get_parameters()

    epsilon = 1e-6
    rng = np.random.RandomState(0)
    checked = 0

    for parameter, gradient in zip(parameters, gradients):
        flat_parameter = np.asarray(parameter).reshape(-1)
        flat_gradient = np.asarray(gradient).reshape(-1)
        positions = rng.choice(flat_parameter.size, size=min(2, flat_parameter.size), replace=False)

        for position in positions:
            original = flat_parameter[position]

            flat_parameter[position] = original + epsilon
            loss_plus = loss_function.forward(model.forward(inputs), targets)

            flat_parameter[position] = original - epsilon
            loss_minus = loss_function.forward(model.forward(inputs), targets)

            flat_parameter[position] = original

            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            analytic = flat_gradient[position]
            scale = max(1e-8, abs(numeric) + abs(analytic))
            assert abs(numeric - analytic) / scale < 1e-3
            checked += 1

    assert checked > 0


def test_micro_transformer_un_passo_di_discesa_riduce_la_loss():
    np.random.seed(21)
    model = _small_model(vocab_size=10, seed=21)
    loss_function = CrossEntropyLoss()
    inputs = np.array([[1, 2, 3, 4]])
    targets = np.array([[2, 3, 4, 5]])

    loss_before = loss_function.forward(model.forward(inputs), targets)
    gradients = model.backward(targets)
    updated = [
        np.asarray(p) - 0.05 * np.asarray(g)
        for p, g in zip(model.get_parameters(), gradients)
    ]
    model.set_parameters(updated)
    loss_after = loss_function.forward(model.forward(inputs), targets)

    assert loss_after < loss_before


# ======================================================================
# MODELLO: parametri, configurazione, context window
# ======================================================================


def test_micro_transformer_get_set_parameters_roundtrip():
    model = _small_model()
    model.forward([[1, 2, 3]])
    original = [np.asarray(p).copy() for p in model.get_parameters()]

    model.set_parameters(original)
    for before, after in zip(original, model.get_parameters()):
        assert np.allclose(before, after)


def test_micro_transformer_set_parameters_rifiuta_numero_errato():
    model = _small_model()
    with pytest.raises(ValueError):
        model.set_parameters([np.zeros((2, 2))])


def test_micro_transformer_parameter_count_positivo():
    model = _small_model()
    assert model.parameter_count() > 0
    assert model.parameter_count() == sum(
        np.asarray(p).size for p in model.get_parameters()
    )


def test_micro_transformer_config_permette_ricostruzione():
    model = _small_model(vocab_size=14)
    clone = MicroTransformer(**model.config())

    assert clone.config() == model.config()
    assert len(clone.get_parameters()) == len(model.get_parameters())

    clone.set_parameters([np.asarray(p).copy() for p in model.get_parameters()])
    inputs = np.array([[1, 2, 3]])
    assert np.allclose(clone.forward(inputs), model.forward(inputs))


def test_context_window_minima_e_configurabile():
    assert MIN_CONTEXT_SIZE == 128
    assert DEFAULT_CONTEXT_SIZE in SUPPORTED_CONTEXT_SIZES
    assert SUPPORTED_CONTEXT_SIZES == (128, 256, 512, 1024)

    for context_size in SUPPORTED_CONTEXT_SIZES:
        model = _small_model(max_len=context_size)
        assert model.max_len == context_size
        assert model.context_size == context_size
        assert model.supports_context(context_size)
        assert not model.supports_context(context_size + 1)


def test_micro_transformer_default_rispetta_il_minimo():
    model = MicroTransformer(vocab_size=8)
    assert model.max_len >= MIN_CONTEXT_SIZE


def test_micro_transformer_forward_sequenza_lunga_128():
    model = _small_model(vocab_size=32, max_len=128)
    token_ids = np.arange(128).reshape(1, 128) % 32
    logits = model.forward(token_ids)

    assert logits.shape == (1, 128, 32)
    assert np.all(np.isfinite(logits))


def test_micro_transformer_rifiuta_sequenza_oltre_context_window():
    model = _small_model(max_len=128)
    token_ids = np.zeros((1, 129), dtype=int)
    with pytest.raises(ValueError):
        model.forward(token_ids)


def test_micro_transformer_dropout_attivo_solo_in_training():
    np.random.seed(3)
    model = _small_model(vocab_size=10, dropout=0.5, seed=6)
    inputs = np.array([[1, 2, 3, 4]])

    inference_a = model.forward(inputs, training=False).copy()
    inference_b = model.forward(inputs, training=False).copy()
    assert np.allclose(inference_a, inference_b)

    training_a = model.forward(inputs, training=True).copy()
    training_b = model.forward(inputs, training=True).copy()
    assert np.allclose(training_a, training_b)
    assert not np.allclose(training_a, inference_a)


def test_micro_transformer_parametri_non_validi():
    with pytest.raises(ValueError):
        MicroTransformer(vocab_size=0)
    with pytest.raises(ValueError):
        MicroTransformer(vocab_size=5, num_layers=0)
    with pytest.raises(ValueError):
        MicroTransformer(vocab_size=5, max_len=0)
    with pytest.raises(ValueError):
        MicroTransformer(vocab_size=5, dropout=1.0)


def test_causal_mask_forma_e_valori():
    mask = causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert np.all(mask[0, 0][np.tril_indices(4)] == 0.0)
    assert np.all(np.isneginf(mask[0, 0][np.triu_indices(4, 1)]))

    with pytest.raises(ValueError):
        causal_mask(0)


def test_micro_transformer_mask_cache_riusata():
    model = _small_model()
    model.forward([[1, 2, 3]])
    first = model._causal_mask(3)
    model.forward([[4, 5, 6]])
    assert model._causal_mask(3) is first
