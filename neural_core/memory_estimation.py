"""
Phase 2B — GPU memory estimation (THEORETICAL vs MEASURED).

Two distinct numbers are produced and MUST NOT be confused:

    * THEORETICAL ESTIMATE : derived from the parameter breakdown and a chosen
      batch/sequence shape. It is an analytical upper/lower bound, not a
      measurement.
    * MEASURED GPU MEMORY : read from the actual allocator (MPS/CUDA) around a
      forward / step. Only available where a real accelerator exists.

Theoretical components (all in bytes for the chosen ``bytes_per_param``,
typically 4 for float32 on GPU):

    parameter memory      : param_count * bytes
    gradient memory       : param_count * bytes
    adam first moment     : param_count * bytes
    adam second moment    : param_count * bytes
    activation memory     : per-layer saved tensors for backward + logits
    attention workspace   : the B*H*L*L score matrices (one per layer)
    overhead              : framework/allocator reserve (set per environment)

Context length (L) does NOT affect the parameter count but dominates the
attention workspace (B*H*L*L) and a large part of the activation memory.
"""

from typing import Optional


def _bytes_per_dtype(dtype: str) -> int:
    return 4 if dtype in ("float32", "fp32") else 8 if dtype in ("float64", "fp64") else 4


def theoretical_gpu_memory(
    param_count: int,
    vocab_size: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    bytes_per_param: int = 4,
    include_logits: bool = True,
    overhead_bytes: int = 0,
):
    """
    Analytical memory estimate (bytes). ``seq_len`` is the effective sequence
    length used for the activation estimate (usually ``min(max_len, seq_len)``).
    """
    L = int(seq_len)
    B = int(batch_size)
    H = int(num_heads)
    d = int(d_model)
    f = int(d_ff)
    V = int(vocab_size)

    parameter_mem = param_count * bytes_per_param
    gradient_mem = param_count * bytes_per_param
    adam_m1 = param_count * bytes_per_param
    adam_m2 = param_count * bytes_per_param

    # Activation elements per layer (saved for backward):
    #   x_in, norm1_x, q, k, v, attn_out, hidden, norm2_x, z1, ff_out
    per_layer_elems = 9 * B * L * d + B * H * L * L + 2 * B * L * f
    # embeddings input, final-norm input
    base_elems = 2 * B * L * d
    # logits forward + grad
    logits_elems = 2 * B * L * V if include_logits else 0
    activation_elems = num_layers * per_layer_elems + base_elems + logits_elems
    activation_mem = activation_elems * bytes_per_param

    # Attention workspace: the B*H*L*L score matrices (one per layer).
    attention_workspace = num_layers * B * H * L * L * bytes_per_param

    total = (
        parameter_mem
        + gradient_mem
        + adam_m1
        + adam_m2
        + activation_mem
        + attention_workspace
        + overhead_bytes
    )

    return {
        "parameter_memory_bytes": parameter_mem,
        "gradient_memory_bytes": gradient_mem,
        "adam_first_moment_bytes": adam_m1,
        "adam_second_moment_bytes": adam_m2,
        "activation_memory_bytes": activation_mem,
        "attention_workspace_bytes": attention_workspace,
        "overhead_bytes": overhead_bytes,
        "total_estimated_bytes": total,
        "bytes_per_param": bytes_per_param,
        "batch_size": B,
        "seq_len": L,
    }


def theoretical_gpu_memory_from_config(config, batch_size, seq_len, **kwargs):
    from neural_core.parameter_count import theoretical_parameter_count
    pc = theoretical_parameter_count(
        config.vocab_size, config.d_model, config.num_heads, config.d_ff, config.num_layers
    )
    eff_seq = min(int(seq_len), config.max_len)
    bpp = kwargs.pop("bytes_per_param", _bytes_per_dtype(config.dtype))
    return theoretical_gpu_memory(
        pc, config.vocab_size, config.d_model, config.d_ff,
        config.num_heads, config.num_layers, batch_size, eff_seq,
        bytes_per_param=bpp, **kwargs
    )


def measure_gpu_memory(model, batch_size, seq_len, device=None):
    """
    Measure allocated GPU memory around a forward and a training step.

    Returns ``None`` when no real accelerator (MPS/CUDA) is available, so callers
    can treat CUDA as "NOT MEASURED LOCALLY" rather than a failure.
    """
    import torch

    resolved = device or str(getattr(model, "device", "cpu"))
    is_accel = resolved in ("mps", "cuda") and torch.device(resolved).type in ("mps", "cuda")
    # Prefer an explicit capability check.
    if resolved == "cuda" and not torch.cuda.is_available():
        is_accel = False
    if resolved == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        is_accel = False

    if not is_accel:
        return None

    rng = __import__("numpy").random.RandomState(0)
    ids = torch.as_tensor(
        rng.randint(0, model.vocab_size, size=(batch_size, seq_len)),
        dtype=torch.long, device=resolved,
    )
    tgt = torch.as_tensor(
        rng.randint(0, model.vocab_size, size=(batch_size * seq_len)),
        dtype=torch.long, device=resolved,
    )

    def _alloc():
        if resolved == "cuda":
            return torch.cuda.memory_allocated()
        return torch.mps.current_allocated_memory()

    model.zero_grad()
    if resolved == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    base = _alloc()
    with torch.no_grad():
        logits = model.forward(ids)
    after_fwd = _alloc()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    out = model.forward(ids)
    loss = torch.nn.functional.cross_entropy(
        out.reshape(-1, model.vocab_size), tgt, ignore_index=-100, reduction="mean"
    )
    loss.backward()
    after_bwd = _alloc()
    opt.step()
    after_step = _alloc()

    result = {
        "device": resolved,
        "allocated_at_base_bytes": int(base),
        "allocated_after_forward_bytes": int(after_fwd),
        "allocated_after_backward_bytes": int(after_bwd),
        "allocated_after_step_bytes": int(after_step),
        "forward_delta_bytes": int(after_fwd - base),
        "step_delta_bytes": int(after_step - base),
    }

    if resolved == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        result["allocated_current_bytes"] = int(torch.cuda.memory_allocated())
        result["reserved_current_bytes"] = int(torch.cuda.memory_reserved())

    return result
