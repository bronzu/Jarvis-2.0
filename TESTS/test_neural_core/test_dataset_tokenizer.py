from neural_core.dataset import Dataset
from neural_core.tokenizer import Tokenizer


def test_tokenizer_to_dataset():
    tokenizer = Tokenizer()

    text = "ciao jarvis ciao mondo"

    tokenizer.fit(text)

    tokens = tokenizer.encode(text)

    dataset = Dataset(tokens)

    assert len(tokens) == 4
    assert len(dataset) == 3

    assert dataset[0] == (tokens[0], tokens[1])
    assert dataset[1] == (tokens[1], tokens[2])
    assert dataset[2] == (tokens[2], tokens[3])


def test_dataset_preserves_token_sequence():
    tokenizer = Tokenizer()

    text = "ciao jarvis"

    tokenizer.fit(text)

    tokens = tokenizer.encode(text)
    dataset = Dataset(tokens)

    assert dataset.get_inputs() == tokens[:-1]
    assert dataset.get_targets() == tokens[1:]