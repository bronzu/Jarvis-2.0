import numpy as np


class Embeddings:
    """
    Tabella di token embedding proprietaria del Neural Core di JARVIS.

    Lo storage interno è una matrice NumPy `(vocab_size, embedding_dim)` per
    permettere forward e backward vettorizzati sul Transformer.

    Compatibilità:
        - `Embeddings(vocab_size, embedding_dim)` inizializza **a zero**, come
          nella versione precedente;
        - `get()` e `forward()` restituiscono liste Python di `float`;
        - l'inizializzazione casuale è opt-in tramite `init_scale`.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, init_scale: float = 0.0, seed=None, dtype=np.float64):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if init_scale < 0:
            raise ValueError("init_scale must be greater than or equal to 0")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.init_scale = float(init_scale)
        self.dtype = np.dtype(dtype)

        if init_scale > 0:
            rng = np.random.RandomState(seed) if seed is not None else np.random
            self.weights = (rng.randn(vocab_size, embedding_dim) * init_scale).astype(self.dtype)
        else:
            self.weights = np.zeros((vocab_size, embedding_dim), dtype=self.dtype)

    # ------------------------------------------------------------------
    # API storica (liste Python)
    # ------------------------------------------------------------------

    def get(self, token_id: int) -> list:
        self._validate_token_id(token_id)
        return [float(value) for value in self.weights[int(token_id)]]

    def set(self, token_id: int, vector) -> None:
        self._validate_token_id(token_id)

        vector = list(vector)
        if len(vector) != self.embedding_dim:
            raise ValueError(
                "vector dimension does not match embedding_dim"
            )

        self.weights[int(token_id)] = np.asarray([float(value) for value in vector], dtype=np.float64)

    def forward(self, token_ids) -> list:
        return [self.get(token_id) for token_id in token_ids]

    # ------------------------------------------------------------------
    # API vettoriale (usata dal Transformer)
    # ------------------------------------------------------------------

    def forward_matrix(self, token_ids) -> np.ndarray:
        """
        Lookup vettoriale.

        Args:
            token_ids: array/lista di indici, di qualsiasi forma.

        Returns:
            np.ndarray di forma `token_ids.shape + (embedding_dim,)`.
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise IndexError("token_id is outside the vocabulary range")
        return self.weights[ids]

    def backward(self, token_ids, grad_output) -> np.ndarray:
        """
        Gradiente rispetto alla tabella di embedding.

        Args:
            token_ids: indici usati nel forward.
            grad_output: gradiente in uscita, forma `token_ids.shape + (embedding_dim,)`.

        Returns:
            np.ndarray `(vocab_size, embedding_dim)`.
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        grad_output = np.asarray(grad_output, dtype=np.float64)

        flat_ids = ids.reshape(-1)
        flat_grad = grad_output.reshape(flat_ids.shape[0], self.embedding_dim)

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, flat_ids, flat_grad)
        return grad_weights

    def _validate_token_id(self, token_id) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("token_id must be an integer")

        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError("token_id is outside the vocabulary range")
