from executive_system.router import ExecutiveRouter


def test_router_handles_complex_prompt_with_internal_neural_core():
    router = ExecutiveRouter()

    prompt = (
        "Analizza in modo strutturato il piano di integrazione del nuovo modulo, "
        "valuta i rischi, e descrivi una strategia di rollout multi-step."
    )

    result = router.generate(prompt)

    assert "normalized_prompt" in result
    assert "response" in result
    assert result["route"] in {"reasoning", "local", "identity", "memory", "knowledge"}
    assert result["provider"] == "internal"
    assert result["internal"] is True


def test_router_never_uses_external_provider():
    """Il router non usa mai provider esterni (ollama/openai)."""
    router = ExecutiveRouter()
    result = router.generate(
        "Analizza un piano strategico multi-step con integrazione completa"
    )
    assert result["provider"] == "internal"
    assert result["route"] in {"reasoning", "local", "identity", "memory", "knowledge"}