from neural_core.dataset import Dataset
from neural_core.embeddings import Embeddings
from neural_core.model import Neuron


def test_dataset_embeddings_neuron_training_cycle():
    tokens = [1, 2, 3, 4]

    dataset = Dataset(tokens)
    embeddings = Embeddings(vocab_size=10, embedding_dim=3)
    neuron = Neuron(input_size=3)

    initial_input, initial_target = dataset[0]

    initial_vector = embeddings.forward([initial_input])[0]
    initial_target_vector = embeddings.forward([initial_target])[0]

    _, initial_loss = neuron.train_step(
        initial_vector,
        sum(initial_target_vector)
    )

    for index in range(1, len(dataset)):
        input_token, target_token = dataset[index]

        input_vector = embeddings.forward([input_token])[0]
        target_vector = embeddings.forward([target_token])[0]

        neuron.train_step(
            input_vector,
            sum(target_vector)
        )

    final_input, final_target = dataset[0]

    final_vector = embeddings.forward([final_input])[0]
    final_target_vector = embeddings.forward([final_target])[0]

    _, final_loss = neuron.train_step(
        final_vector,
        sum(final_target_vector)
    )

    assert isinstance(initial_loss, (int, float))
    assert isinstance(final_loss, (int, float))
    assert final_loss >= 0