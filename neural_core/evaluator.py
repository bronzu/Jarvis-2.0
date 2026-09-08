class ModelEvaluator:
    """
    Valuta le predizioni di un modello confrontandole con i valori target.
    """

    def evaluate(self, predictions, targets):
        """
        Calcola l'errore medio assoluto (MAE).

        Args:
            predictions: valori prodotti dal modello.
            targets: valori corretti attesi.

        Returns:
            float: errore medio assoluto.
        """

        if len(predictions) != len(targets):
            raise ValueError(
                "Predictions e targets devono avere la stessa lunghezza."
            )

        if len(predictions) == 0:
            raise ValueError(
                "Predictions e targets non possono essere vuoti."
            )

        total_error = 0.0

        for prediction, target in zip(predictions, targets):
            total_error += abs(prediction - target)

        return total_error / len(predictions)