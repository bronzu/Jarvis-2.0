from neural_core.dataset import Dataset


def test_dataset_get_batches():
    dataset = Dataset([1, 2, 3, 4, 5])

    batches = dataset.get_batches(batch_size=2)

    assert len(batches) == 2

    assert batches[0] == [
        (1, 2),
        (2, 3)
    ]

    assert batches[1] == [
        (3, 4),
        (4, 5)
    ]


def test_dataset_get_batches_keeps_remainder():
    dataset = Dataset([1, 2, 3, 4, 5, 6])

    batches = dataset.get_batches(batch_size=4)

    assert len(batches) == 2

    assert len(batches[0]) == 4
    assert len(batches[1]) == 1


def test_dataset_get_batches_rejects_invalid_batch_size():
    dataset = Dataset([1, 2, 3])

    try:
        dataset.get_batches(0)
        assert False
    except ValueError:
        assert True


def test_dataset_get_batches_rejects_non_integer():
    dataset = Dataset([1, 2, 3])

    try:
        dataset.get_batches(2.5)
        assert False
    except TypeError:
        assert True