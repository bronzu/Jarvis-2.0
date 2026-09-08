"""
Phase 2B — Theoretical parameter counting for the JARVIS decoder-only model.

The parameter count must be computed BOTH theoretically (here) and materially
(on the built model). If the two disagree, that is a hard stop condition.

The architecture (unchanged from the CPU reference ``neural_core.network``):

    embedding            : vocab_size * d_model
    per TransformerBlock :
        norm1            : 2 * d_model
        attention        : 4 * (d_model * d_model) + 4 * d_model
        norm2            : 2 * d_model
        feed-forward     : 2 * d_model * d_ff + d_model + d_ff
    final norm           : 2 * d_model
    output projection    : d_model * vocab_size + vocab_size

The number of heads does NOT change the parameter count (W_q/k/v/o are always
d_model x d_model); it only affects how d_model is split at attention time.
"""


def layer_parameter_count(d_model, d_ff):
    """Parameter count of a single TransformerBlock."""
    norm = 2 * d_model
    attention = 4 * (d_model * d_model) + 4 * d_model
    ffn = 2 * d_model * d_ff + d_model + d_ff
    return norm + attention + norm + ffn


def theoretical_breakdown(vocab_size, d_model, num_heads, d_ff, num_layers):
    """Return a dict with the full parameter breakdown (theoretical)."""
    embedding = vocab_size * d_model

    per_layer = layer_parameter_count(d_model, d_ff)
    attention_per_layer = 4 * (d_model * d_model) + 4 * d_model
    ffn_per_layer = 2 * d_model * d_ff + d_model + d_ff
    layernorm_per_layer = 2 * d_model + 2 * d_model

    attention = attention_per_layer * num_layers
    ffn = ffn_per_layer * num_layers
    layernorm = layernorm_per_layer * num_layers

    final_norm = 2 * d_model
    layernorm += final_norm

    output = d_model * vocab_size + vocab_size

    total = embedding + per_layer * num_layers + final_norm + output

    return {
        "total": int(total),
        "trainable": int(total),
        "non_trainable": 0,
        "embedding": int(embedding),
        "attention": int(attention),
        "ffn": int(ffn),
        "layernorm": int(layernorm),
        "output": int(output),
        "per_layer": int(per_layer),
    }


def theoretical_parameter_count(vocab_size, d_model, num_heads, d_ff, num_layers):
    """Total theoretical parameter count."""
    return theoretical_breakdown(vocab_size, d_model, num_heads, d_ff, num_layers)["total"]


def breakdown_from_config(config):
    """Theoretical breakdown from a :class:`ModelConfig`."""
    return theoretical_breakdown(
        config.vocab_size,
        config.d_model,
        config.num_heads,
        config.d_ff,
        config.num_layers,
    )
