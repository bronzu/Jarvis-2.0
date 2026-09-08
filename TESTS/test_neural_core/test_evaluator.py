from neural_core.evaluator import ModelEvaluator


def test_evaluate_correct_predictions():
    evaluator = ModelEvaluator()

    predictions = [1.0, 2.0, 3.0]
    targets = [1.0, 2.0, 3.0]

    result = evaluator.evaluate(predictions, targets)

    assert result == 0.0


def test_evaluate_incorrect_predictions():
    evaluator = ModelEvaluator()

    predictions = [1.0, 2.0, 3.0]
    targets = [2.0, 2.0, 1.0]

    result = evaluator.evaluate(predictions, targets)

    assert result == 1.0


def test_evaluate_different_lengths():
    evaluator = ModelEvaluator()

    predictions = [1.0, 2.0]
    targets = [1.0]

    try:
        evaluator.evaluate(predictions, targets)
        assert False
    except ValueError:
        assert True


def test_evaluate_empty_lists():
    evaluator = ModelEvaluator()

    predictions = []
    targets = []

    try:
        evaluator.evaluate(predictions, targets)
        assert False
    except ValueError:
        assert True