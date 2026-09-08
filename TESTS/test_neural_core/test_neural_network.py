from neural_core.model import Neuron


def test_neuron_training_reduces_loss():
    neuron = Neuron(
        input_size=2,
        learning_rate=0.1
    )

    inputs = [1.0, 1.0]
    target = 1.0

    _, initial_loss = neuron.train_step(
        inputs,
        target
    )

    for _ in range(20):
        neuron.train_step(
            inputs,
            target
        )

    _, final_loss = neuron.train_step(
        inputs,
        target
    )

    assert final_loss < initial_loss