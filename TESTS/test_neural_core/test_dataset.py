import numpy as np
import pytest

from neural_core.dataset import Dataset
from neural_core.tokenizer import Tokenizer


def test_dataset_from_text_creates_training_and_validation():
    text = (
        "jarvis learns from real text data "
        "and validates what it learns from separate data"
    )

    tokenizer = Tokenizer()

    training_dataset, validation_dataset = Dataset.from_text(
        text,
        tokenizer,
        validation_split=0.2
    )

    assert isinstance(training_dataset, Dataset)
    assert isinstance(validation_dataset, Dataset)

    assert len(training_dataset) > 0
    assert len(validation_dataset) > 0


def test_dataset_from_text_uses_tokenizer():
    text = (
        "jarvis learns from real text data "
        "and validates what it learns"
    )

    tokenizer = Tokenizer()

    training_dataset, validation_dataset = Dataset.from_text(
        text,
        tokenizer,
        validation_split=0.2
    )

    assert len(tokenizer.vocabulary) > 1

    assert all(
        isinstance(token_id, int)
        for token_id in training_dataset.tokens
    )

    assert all(
        isinstance(token_id, int)
        for token_id in validation_dataset.tokens
    )


def test_dataset_from_text_keeps_training_and_validation_separate():
    text = (
        "jarvis learns from training data "
        "while validation checks unseen data"
    )

    tokenizer = Tokenizer()

    training_dataset, validation_dataset = Dataset.from_text(
        text,
        tokenizer,
        validation_split=0.25
    )

    assert training_dataset.tokens != validation_dataset.tokens


def test_dataset_from_text_rejects_empty_text():
    tokenizer = Tokenizer()

    try:
        Dataset.from_text(
            "",
            tokenizer,
            validation_split=0.2
        )
        assert False
    except ValueError:
        assert True


def test_dataset_from_text_rejects_invalid_split():
    text = "jarvis learns from real text data"

    tokenizer = Tokenizer()

    try:
        Dataset.from_text(
            text,
            tokenizer,
            validation_split=0
        )
        assert False
    except ValueError:
        assert True


def test_dataset_from_corpus_file(tmp_path):
    corpus = "alpha beta gamma delta epsilon zeta eta theta"
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(corpus, encoding="utf-8")

    tokenizer = Tokenizer()
    training_dataset, validation_dataset = Dataset.from_corpus_file(
        str(corpus_file), tokenizer, validation_split=0.25
    )

    assert isinstance(training_dataset, Dataset)
    assert isinstance(validation_dataset, Dataset)
    assert len(training_dataset) > 0
    assert len(validation_dataset) > 0
    assert all(isinstance(token_id, int) for token_id in training_dataset.tokens)
    assert all(isinstance(token_id, int) for token_id in validation_dataset.tokens)


# ======================================================================
# CONTEXT WINDOW: finestre autoregressive
# ======================================================================


def test_modalita_legacy_resta_a_token_singolo():
    dataset = Dataset([1, 2, 3, 4])

    assert dataset.is_windowed() is False
    assert dataset.context_size is None
    assert dataset.stride is None
    assert len(dataset) == 3
    assert dataset[0] == (1, 2)
    assert dataset.get_inputs() == [1, 2, 3]
    assert dataset.get_targets() == [2, 3, 4]


def test_finestre_autoregressive_con_target_shiftati():
    dataset = Dataset(list(range(10)), context_size=4)

    assert dataset.is_windowed() is True
    assert dataset.context_size == 4
    assert dataset.stride == 4
    assert len(dataset) == 2
    assert dataset[0] == ([0, 1, 2, 3], [1, 2, 3, 4])
    assert dataset[1] == ([4, 5, 6, 7], [5, 6, 7, 8])


def test_ogni_target_e_il_token_successivo():
    tokens = list(range(20))
    dataset = Dataset(tokens, context_size=5, stride=1)

    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        assert len(inputs) == len(targets) == 5
        for position in range(5):
            assert targets[position] == inputs[position] + 1


def test_stride_controlla_la_sovrapposizione():
    tokens = list(range(20))

    non_sovrapposto = Dataset(tokens, context_size=4)
    sovrapposto = Dataset(tokens, context_size=4, stride=1)

    assert len(sovrapposto) > len(non_sovrapposto)
    assert sovrapposto[1][0] == [1, 2, 3, 4]
    assert non_sovrapposto[1][0] == [4, 5, 6, 7]


def test_get_arrays_restituisce_matrici_coerenti():
    dataset = Dataset(list(range(33)), context_size=8)
    inputs, targets = dataset.get_arrays()

    assert inputs.shape == (len(dataset), 8)
    assert targets.shape == (len(dataset), 8)
    assert np.array_equal(targets[:, :-1], inputs[:, 1:])


def test_get_arrays_in_modalita_legacy():
    dataset = Dataset([5, 6, 7, 8])
    inputs, targets = dataset.get_arrays()

    assert inputs.shape == (3,)
    assert targets.shape == (3,)


def test_get_batches_in_modalita_sequenza():
    dataset = Dataset(list(range(41)), context_size=4, stride=4)
    batches = dataset.get_batches(3)

    assert sum(len(batch) for batch in batches) == len(dataset)
    first_input, first_target = batches[0][0]
    assert len(first_input) == 4
    assert len(first_target) == 4


def test_context_window_128_256_512():
    tokens = list(range(1200))

    for context_size in (128, 256, 512):
        dataset = Dataset(tokens, context_size=context_size)
        inputs, targets = dataset.get_arrays()
        assert inputs.shape[1] == context_size
        assert targets.shape[1] == context_size
        assert len(dataset) == (len(tokens) - 1 - context_size) // context_size + 1


def test_token_insufficienti_per_la_context_window():
    with pytest.raises(ValueError):
        Dataset(list(range(8)), context_size=16)


def test_context_size_non_valido():
    with pytest.raises(ValueError):
        Dataset([1, 2, 3], context_size=0)
    with pytest.raises(TypeError):
        Dataset([1, 2, 3], context_size=2.5)


def test_stride_non_valido():
    with pytest.raises(ValueError):
        Dataset(list(range(10)), context_size=4, stride=0)
    with pytest.raises(TypeError):
        Dataset(list(range(10)), context_size=4, stride=1.5)
    with pytest.raises(ValueError):
        Dataset(list(range(10)), stride=2)


def test_indice_fuori_intervallo_in_modalita_sequenza():
    dataset = Dataset(list(range(10)), context_size=4)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_from_text_propaga_la_context_window():
    text = " ".join(f"parola{i % 12}" for i in range(300))
    tokenizer = Tokenizer()

    training_dataset, validation_dataset = Dataset.from_text(
        text, tokenizer, validation_split=0.2, context_size=16
    )

    assert training_dataset.context_size == 16
    assert validation_dataset.context_size == 16
    assert training_dataset.get_arrays()[0].shape[1] == 16
    assert validation_dataset.get_arrays()[0].shape[1] == 16


def test_from_files_propaga_la_context_window(tmp_path):
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    test_file = tmp_path / "test.txt"
    train_file.write_text(" ".join(f"t{i % 9}" for i in range(200)), encoding="utf-8")
    val_file.write_text(" ".join(f"t{i % 9}" for i in range(80)), encoding="utf-8")
    test_file.write_text(" ".join(f"t{i % 9}" for i in range(80)), encoding="utf-8")

    tokenizer = Tokenizer()
    training_dataset, validation_dataset, test_dataset = Dataset.from_files(
        str(train_file), str(val_file), tokenizer,
        test_path=str(test_file), context_size=32, stride=16,
    )

    for dataset in (training_dataset, validation_dataset, test_dataset):
        assert dataset.context_size == 32
        assert dataset.stride == 16
        assert dataset.get_arrays()[0].shape[1] == 32


def test_from_corpus_file_propaga_la_context_window(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(" ".join(f"w{i % 7}" for i in range(200)), encoding="utf-8")

    tokenizer = Tokenizer()
    training_dataset, validation_dataset = Dataset.from_corpus_file(
        str(corpus_file), tokenizer, validation_split=0.25, context_size=16
    )

    assert training_dataset.context_size == 16
    assert validation_dataset.context_size == 16


def test_dataset_deduplicate_lines_removes_exact_duplicates():
    text = "alpha\nbeta\nalpha\ngamma\nbeta"
    deduped = Dataset.deduplicate_lines(text)
    assert deduped.count("alpha") == 1
    assert deduped.count("beta") == 1
    assert deduped.count("gamma") == 1


def test_dataset_deduplicate_lines_preserves_order():
    text = "first\nsecond\nfirst\nthird"
    deduped = Dataset.deduplicate_lines(text)
    assert deduped.splitlines().index("first") < deduped.splitlines().index("third")


def test_dataset_from_text_deduplicates_before_split():
    text = "alpha beta gamma\nalpha beta gamma\ndelta epsilon zeta\n" * 20
    tokenizer = Tokenizer()
    training_dataset, validation_dataset = Dataset.from_text(
        text,
        tokenizer,
        validation_split=0.5,
    )
    train_tokens = set(training_dataset.get_inputs())
    val_tokens = set(validation_dataset.get_inputs())
    overlap = train_tokens & val_tokens
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping tokens after deduplication"


def test_dataset_check_overlap_detects_shared_tokens():
    tokens_a = [1, 2, 3, 4, 5]
    tokens_b = [4, 5, 6, 7, 8]
    ds_a = Dataset(tokens_a)
    ds_b = Dataset(tokens_b)
    overlap = Dataset.check_overlap(ds_a, ds_b)
    assert overlap.get("dataset_1_vs_dataset_2") == 1


def test_dataset_audit_split_produces_report():
    tokens = list(range(20))
    ds_a = Dataset(tokens[:10])
    ds_b = Dataset(tokens[10:])
    report = Dataset.audit_split(ds_a, ds_b, names=["train", "test"])
    assert "sizes" in report
    assert report["sizes"]["train"] == 9
    assert report["sizes"]["test"] == 9
    assert report["overlap"]["dataset_1_vs_dataset_2"] == 0
