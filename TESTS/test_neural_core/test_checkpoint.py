from neural_core.model import Neuron


def test_checkpoint_ripristina_parametri(tmp_path):
    checkpoint_path = tmp_path / "neuron_checkpoint.json"

    neuron = Neuron(input_size=2)

    neuron.train_step(
        [1.0, 2.0],
        1.0
    )

    weights_salvati = neuron.weights.copy()
    bias_salvato = neuron.bias

    neuron.save_checkpoint(checkpoint_path)

    nuovo_neuron = Neuron(input_size=2)

    nuovo_neuron.load_checkpoint(checkpoint_path)

    assert nuovo_neuron.weights == weights_salvati
    assert nuovo_neuron.bias == bias_salvato


def test_training_salva_checkpoint_automaticamente(tmp_path):
    checkpoint_path = tmp_path / "training_checkpoint.json"

    neuron = Neuron(input_size=2)

    neuron.train_step(
        [1.0, 2.0],
        1.0,
        checkpoint_path
    )

    assert checkpoint_path.exists()

    nuovo_neuron = Neuron(input_size=2)

    nuovo_neuron.load_checkpoint(checkpoint_path)

    assert nuovo_neuron.weights == neuron.weights
    assert nuovo_neuron.bias == neuron.bias