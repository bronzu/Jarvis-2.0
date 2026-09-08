from neural_core.dataset import Dataset
from neural_core.embeddings import Embeddings


def test_dataset_to_embeddings_integration():
    tokens = [0, 1, 2, 3]

    dataset = Dataset(tokens)
    embeddings = Embeddings(
        vocab_size=4,
        embedding_dim=3
    )

    embeddings.set(0, [0.1, 0.2, 0.3])
    embeddings.set(1, [0.4, 0.5, 0.6])
    embeddings.set(2, [0.7, 0.8, 0.9])
    embeddings.set(3, [1.0, 1.1, 1.2])

    inputs = dataset.get_inputs()
    targets = dataset.get_targets()

    input_vectors = embeddings.forward(inputs)
    target_vectors = embeddings.forward(targets)

    assert len(input_vectors) == len(inputs)
    assert len(target_vectors) == len(targets)

    assert input_vectors[0] == [0.1, 0.2, 0.3]
    assert target_vectors[0] == [0.4, 0.5, 0.6]

    for vector in input_vectors + target_vectors:
        assert len(vector) == 3