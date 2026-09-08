from pathlib import Path

import numpy as np


class Dataset:
    """
    Dataset per il Neural Core di JARVIS 2.0.

    Due modalità, entrambe supportate dallo stesso oggetto.

    1) **Modalità token singolo** (`context_size=None`, comportamento storico)

       Riceve una sequenza di token e crea coppie::

           input -> target
           token[i] -> token[i + 1]

    2) **Modalità sequenza / context window** (`context_size=N`)

       Crea finestre autoregressive di `N` token con target shiftati di uno::

           input  = token[i     : i + N]
           target = token[i + 1 : i + N + 1]

       È la modalità richiesta dal Transformer proprietario
       (`neural_core.network.MicroTransformer`), che lavora con context window
       di almeno 128 token, configurabile a 256 o 512.

    Supporta:
    - accesso ai singoli campioni;
    - recupero di input e target;
    - creazione di dataset Training e Validation;
    - suddivisione dei dati in batch;
    - finestre con `stride` configurabile.
    """

    def __init__(self, tokens, context_size=None, stride=None):
        if tokens is None or len(tokens) == 0:
            raise ValueError("Il dataset non può essere vuoto.")

        if len(tokens) < 2:
            raise ValueError(
                "Il dataset deve contenere almeno due token."
            )

        self.tokens = list(tokens)
        self.context_size = self._validate_context_size(context_size)
        self.stride = self._validate_stride(stride)

        if self.context_size is not None:
            if len(self.tokens) < self.context_size + 1:
                raise ValueError(
                    "Il dataset deve contenere almeno context_size + 1 token "
                    f"({self.context_size + 1}), trovati {len(self.tokens)}."
                )

    # ------------------------------------------------------------------
    # Validazione
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_context_size(context_size):
        if context_size is None:
            return None

        if isinstance(context_size, bool) or not isinstance(context_size, (int, np.integer)):
            raise TypeError("context_size deve essere un intero.")

        if context_size <= 0:
            raise ValueError("context_size deve essere maggiore di zero.")

        return int(context_size)

    def _validate_stride(self, stride):
        if self.context_size is None:
            if stride is not None:
                raise ValueError(
                    "stride richiede che context_size sia specificato."
                )
            return None

        if stride is None:
            return self.context_size

        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
            raise TypeError("stride deve essere un intero.")

        if stride <= 0:
            raise ValueError("stride deve essere maggiore di zero.")

        return int(stride)

    # ------------------------------------------------------------------
    # Accesso
    # ------------------------------------------------------------------

    def is_windowed(self):
        """True se il dataset produce sequenze (context window)."""
        return self.context_size is not None

    def __len__(self):
        if self.context_size is None:
            return len(self.tokens) - 1

        usable = len(self.tokens) - 1 - self.context_size
        return usable // self.stride + 1

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                "Indice del dataset fuori intervallo."
            )

        if self.context_size is None:
            input_token = self.tokens[index]
            target_token = self.tokens[index + 1]
            return input_token, target_token

        start = index * self.stride
        end = start + self.context_size
        return self.tokens[start:end], self.tokens[start + 1:end + 1]

    def get_inputs(self):
        """Restituisce tutti gli input del dataset."""
        if self.context_size is None:
            return self.tokens[:-1]
        return [self[index][0] for index in range(len(self))]

    def get_targets(self):
        """Restituisce tutti i target del dataset."""
        if self.context_size is None:
            return self.tokens[1:]
        return [self[index][1] for index in range(len(self))]

    def get_arrays(self):
        """
        Restituisce input e target come array NumPy.

        - modalità token singolo: `(N,)`, `(N,)`
        - modalità sequenza:      `(N, context_size)`, `(N, context_size)`
        """
        return (
            np.asarray(self.get_inputs(), dtype=np.int64),
            np.asarray(self.get_targets(), dtype=np.int64),
        )

    def get_batches(self, batch_size):
        """
        Divide il dataset in batch.

        Ogni batch contiene una lista di coppie:
            (input, target)

        dove `input`/`target` sono token singoli oppure sequenze di
        `context_size` token.

        L'ultimo batch può contenere meno elementi.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError(
                "batch_size deve essere un intero."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size deve essere maggiore di zero."
            )

        batches = []

        for start in range(0, len(self), batch_size):
            end = start + batch_size

            batch = [
                self[index]
                for index in range(
                    start,
                    min(end, len(self))
                )
            ]

            batches.append(batch)

        return batches

    # ------------------------------------------------------------------
    # Costruzione da testo
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text,
        tokenizer,
        validation_split=0.2,
        test_split=0.0,
        shuffle_blocks=False,
        context_size=None,
        stride=None,
        seed=None,
    ):
        """
        Crea Training, Validation e (opzionalmente) Test Dataset
        a partire da testo reale.

        Args:
            text: Testo sorgente.
            tokenizer: Tokenizer da utilizzare. Viene addestrato sul
                training text dopo lo split.
            validation_split: Frazione per validation.
            test_split: Frazione per test.
            shuffle_blocks: Se True, mescola i blocchi di testo prima
                dello split per ridurre la correlazione train/val.
            context_size: Se specificato, i dataset generati producono
                finestre autoregressive di `context_size` token.
            stride: Passo tra finestre consecutive (default `context_size`).
        """

        if not isinstance(text, str):
            raise TypeError(
                "Il testo del dataset deve essere una stringa."
            )

        if not text.strip():
            raise ValueError(
                "Il testo del dataset non può essere vuoto."
            )

        if not 0 < validation_split < 1:
            raise ValueError(
                "validation_split deve essere compreso tra 0 e 1."
            )

        if test_split < 0 or test_split >= 1:
            raise ValueError(
                "test_split deve essere compreso tra 0 e 1, escluso 1."
            )

        if test_split > 0 and validation_split + test_split >= 1:
            raise ValueError(
                "La somma di validation_split e test_split deve essere minore di 1."
            )

        cleaned_text = cls.deduplicate_lines(text)
        tokens = (" ".join(cleaned_text.split())).split()

        total_tokens = len(tokens)

        if total_tokens < 6:
            raise ValueError(
                "Il testo deve contenere almeno sei token per "
                "supportare train, validation e test."
            )

        if test_split > 0:
            test_size = max(2, int(total_tokens * test_split))
            remaining = total_tokens - test_size
            val_size = max(2, int(remaining * validation_split))
            train_size = remaining - val_size

            if train_size < 2:
                raise ValueError(
                    "Il Training Dataset deve contenere almeno due token."
                )
            if val_size < 2:
                raise ValueError(
                    "Il Validation Dataset deve contenere almeno due token."
                )
            if test_size < 2:
                raise ValueError(
                    "Il Test Dataset deve contenere almeno due token."
                )

            training_tokens = tokens[:train_size]
            validation_tokens = tokens[train_size:train_size + val_size]
            test_tokens = tokens[train_size + val_size:]
        else:
            if shuffle_blocks:
                blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
                if len(blocks) <= 1:
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    block_size = max(10, len(lines) // 100)
                    blocks = ["\n".join(lines[i:i + block_size]) for i in range(0, len(lines), block_size)]
                    blocks = [b for b in blocks if b.strip()]

                if len(blocks) > 1:
                    import random
                    rng = random.Random(seed) if seed is not None else random.Random(42)
                    rng.shuffle(blocks)
                    shuffled_text = "\n\n".join(blocks)
                    tokens = (" ".join(shuffled_text.split())).split()

            split_index = int(total_tokens * (1 - validation_split))

            if split_index < 2:
                raise ValueError(
                    "Il Training Dataset deve contenere almeno due token."
                )

            if total_tokens - split_index < 2:
                raise ValueError(
                    "Il Validation Dataset deve contenere almeno due token."
                )

            training_tokens = tokens[:split_index]
            validation_tokens = tokens[split_index:]
            test_tokens = []

        training_text = " ".join(training_tokens)
        validation_text = " ".join(validation_tokens)

        tokenizer.fit(training_text)

        training_ids = tokenizer.encode(training_text)
        validation_ids = tokenizer.encode(validation_text)

        training_dataset = cls(training_ids, context_size=context_size, stride=stride)
        validation_dataset = cls(validation_ids, context_size=context_size, stride=stride)

        datasets = [training_dataset, validation_dataset]
        names = ["training", "validation"]

        if test_tokens:
            test_text = " ".join(test_tokens)
            test_ids = tokenizer.encode(test_text)
            test_dataset = cls(test_ids, context_size=context_size, stride=stride)
            datasets.append(test_dataset)
            names.append("test")
            overlap = cls.check_overlap(*datasets)
            if any(v > 0 for v in overlap.values()):
                import warnings
                warnings.warn(
                    f"Rilevata sovrapposizione tra split: {overlap}. "
                    "Usare shuffle_blocks=True o verificare il sorgente."
                )
            return training_dataset, validation_dataset, test_dataset

        overlap = cls.check_overlap(*datasets)
        if any(v > 0 for v in overlap.values()):
            import warnings
            warnings.warn(
                f"Rilevata sovrapposizione tra split: {overlap}. "
                "Usare shuffle_blocks=True o verificare il sorgente."
            )

        return training_dataset, validation_dataset

    @staticmethod
    def deduplicate_lines(text):
        """Rimuove righe duplicate esatte dal testo, preservando l'ordine."""
        seen = set()
        lines = []
        for line in text.splitlines():
            normalized = " ".join(line.split())
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            lines.append(normalized)
        return "\n".join(lines)

    @staticmethod
    def check_overlap(*datasets):
        """Verifica sovrapposizione di token tra più dataset."""
        if len(datasets) < 2:
            return {}
        token_sets = []
        for ds in datasets:
            if hasattr(ds, "get_inputs"):
                tokens = ds.get_inputs()
            else:
                tokens = list(ds)
            if isinstance(tokens[0], (list, tuple)):
                flat = []
                for seq in tokens:
                    flat.extend(seq)
                token_sets.append(set(flat))
            else:
                token_sets.append(set(tokens))
        overlap = {}
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                key = f"dataset_{i+1}_vs_dataset_{j+1}"
                overlap[key] = len(token_sets[i] & token_sets[j])
        return overlap

    @staticmethod
    def audit_split(*datasets, names=None):
        """Produce un report di audit sulla sovrapposizione tra split."""
        if names is None:
            names = [f"split_{i+1}" for i in range(len(datasets))]
        report = {
            "sizes": {name: len(ds) for name, ds in zip(names, datasets)},
            "overlap": Dataset.check_overlap(*datasets),
        }
        for name, ds in zip(names, datasets):
            report[name] = {
                "samples": len(ds),
                "context_window": getattr(ds, "context_size", None),
            }
        return report


    @classmethod
    def from_corpus_file(
        cls,
        corpus_path,
        tokenizer,
        validation_split=0.2,
        test_split=0.0,
        context_size=None,
        stride=None,
        seed=None,
    ):
        text = Path(corpus_path).read_text(encoding="utf-8")
        return cls.from_text(
            text,
            tokenizer,
            validation_split,
            test_split,
            context_size=context_size,
            stride=stride,
            seed=seed,
        )

    @classmethod
    def from_files(
        cls,
        train_path,
        val_path,
        tokenizer,
        test_path=None,
        context_size=None,
        stride=None,
    ):
        """
        Crea dataset da file di testo pre-split.

        Il tokenizer viene addestrato solo sul file di training.
        """
        train_text = Path(train_path).read_text(encoding="utf-8")
        val_text = Path(val_path).read_text(encoding="utf-8")

        tokenizer.fit(train_text)

        train_ids = tokenizer.encode(train_text)
        val_ids = tokenizer.encode(val_text)

        training_dataset = cls(train_ids, context_size=context_size, stride=stride)
        validation_dataset = cls(val_ids, context_size=context_size, stride=stride)

        if test_path is not None:
            test_text = Path(test_path).read_text(encoding="utf-8")
            test_ids = tokenizer.encode(test_text)
            test_dataset = cls(test_ids, context_size=context_size, stride=stride)
            return training_dataset, validation_dataset, test_dataset

        return training_dataset, validation_dataset
