from neural_core.model import Neuron
from neural_core.evaluator import ModelEvaluator


def test_learning_improves_model_evaluation():
    neuron = Neuron(
        input_size=2,
        learning_rate=0.1
    )

    evaluator = ModelEvaluator()

    inputs = [1.0, 1.0]
    target = 1.0

    initial_prediction = neuron.forward(inputs)

    initial_score = evaluator.evaluate(
        [initial_prediction],
        [target]
    )

    for _ in range(20):
        neuron.train_step(
            inputs,
            target
        )

    final_prediction = neuron.forward(inputs)

    final_score = evaluator.evaluate(
        [final_prediction],
        [target]
    )

    assert final_score <= initial_score