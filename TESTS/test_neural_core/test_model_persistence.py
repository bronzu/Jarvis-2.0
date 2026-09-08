from neural_core.model import Neuron


def test_model_persistence(tmp_path):
    checkpoint_path = tmp_path / "model_checkpoint.json"

    original = Neuron(
        input_size=2,
        learning_rate=0.1
    )

    original.train_step(
        [1.0, 2.0],
        3.0
    )

    original.save_checkpoint(checkpoint_path)

    restored = Neuron(
        input_size=2,
        learning_rate=0.1
    )

    restored.load_checkpoint(checkpoint_path)

    assert restored.weights == original.weights
    assert restored.bias == original.bias