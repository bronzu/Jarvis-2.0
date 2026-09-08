"""
Tokenizer proprietario del Neural Core di JARVIS.

Nessun tokenizer pre-addestrato, nessun vocabolario scaricato: il vocabolario
viene costruito esclusivamente dal corpus interno tramite `fit()`.

Due modalità di pre-tokenizzazione:

- ``"whitespace"`` (default, **comportamento storico**): `text.split()`.
- ``"technical"``: pre-tokenizzazione a regole progettata per
  **italiano moderno**, **italiano tecnico**, **codice**, **matematica** e
  **scienza**. Separa punteggiatura, elisioni (``dell'`` + ``acqua``), numeri
  (``3.14``, ``1,5``, ``1e-9``, ``0xFF``), operatori multi-carattere
  (``==``, ``!=``, ``->``, ``**``, ``//``, ``:=`` ...) e simboli
  matematico-scientifici (``≤``, ``√``, ``∑``, ``π``, ``°`` ...).

La modalità di default è invariata per garantire la compatibilità con i
checkpoint e i tokenizer già serializzati.
"""

import json
import re
from collections import Counter
from pathlib import Path


class Tokenizer:
    UNK_TOKEN = "<UNK>"
    END_OF_TEXT_TOKEN = "<|endoftext|>"

    MODE_WHITESPACE = "whitespace"
    MODE_TECHNICAL = "technical"
    SUPPORTED_MODES = (MODE_WHITESPACE, MODE_TECHNICAL)

    #: Operatori e sequenze multi-carattere di codice/matematica.
    OPERATORS = (
        "**=", "//=", "<<=", ">>=", "===", "!==", "<=>", "...", "|>",
        "->", "=>", "::", ":=", "==", "!=", "<=", ">=", "&&", "||",
        "**", "//", "++", "--", "+=", "-=", "*=", "/=", "%=", "|=",
        "&=", "^=", "<<", ">>", "??", "?.",
    )

    #: Lettere latine con diacritici (esclusi i simboli ``×`` e ``÷``).
    _LETTER = r"A-Za-z_\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F"

    #: Token che non vogliono spazio prima nella ricostruzione del testo.
    NO_SPACE_BEFORE = frozenset(
        [",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "…", "'", "’",
         "”", "»", "″", "′", "°", "ˆ"]
    )

    #: Token che non vogliono spazio dopo nella ricostruzione del testo.
    NO_SPACE_AFTER = frozenset(["(", "[", "{", "“", "«", "#", "@", "$", "¿", "¡", "°"])

    def __init__(self, max_vocab_size=None, mode=MODE_WHITESPACE, lowercase=False):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"mode deve essere uno di {self.SUPPORTED_MODES}, ricevuto: {mode!r}"
            )

        self.vocabulary = {
            self.UNK_TOKEN: 0,
            self.END_OF_TEXT_TOKEN: 1,
        }
        self.reverse_vocabulary = {
            0: self.UNK_TOKEN,
            1: self.END_OF_TEXT_TOKEN,
        }
        self.max_vocab_size = max_vocab_size
        self.mode = mode
        self.lowercase = bool(lowercase)
        self.special_tokens = [self.UNK_TOKEN, self.END_OF_TEXT_TOKEN]
        self._pattern = None

    # ------------------------------------------------------------------
    # Proprietà
    # ------------------------------------------------------------------

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    @property
    def unk_id(self):
        return self.vocabulary[self.UNK_TOKEN]

    @property
    def end_of_text_id(self):
        return self.vocabulary[self.END_OF_TEXT_TOKEN]

    # ------------------------------------------------------------------
    # Pre-tokenizzazione
    # ------------------------------------------------------------------

    def _build_pattern(self):
        specials = sorted(self.special_tokens, key=len, reverse=True)
        operators = sorted(self.OPERATORS, key=len, reverse=True)

        parts = []
        if specials:
            parts.append("|".join(re.escape(token) for token in specials))

        parts.extend([
            # numeri esadecimali (codice)
            r"0[xX][0-9a-fA-F]+",
            # numeri decimali, migliaia e notazione scientifica
            r"\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?",
            # operatori multi-carattere
            "|".join(re.escape(operator) for operator in operators),
            # parole italiane e identificatori di codice, con elisione finale
            rf"[{self._LETTER}][{self._LETTER}0-9]*['\u2019]?",
            # qualunque altro carattere non-spazio: punteggiatura, simboli,
            # lettere greche, operatori matematici unicode
            r"[^\s]",
        ])

        return re.compile("|".join(f"(?:{part})" for part in parts))

    def _get_pattern(self):
        if getattr(self, "_pattern", None) is None:
            self._pattern = self._build_pattern()
        return self._pattern

    def pre_tokenize(self, text):
        """
        Divide il testo grezzo nei token di superficie, senza usare il
        vocabolario.

        Returns:
            list[str]
        """
        if text is None:
            raise ValueError("text non può essere None")

        text = str(text)
        if self.lowercase:
            text = text.lower()

        if self.mode == self.MODE_WHITESPACE:
            return text.split()

        return self._get_pattern().findall(text)

    # ------------------------------------------------------------------
    # Vocabolario
    # ------------------------------------------------------------------

    def add_special_token(self, token):
        """
        Registra un token speciale, garantendo che resti atomico durante la
        pre-tokenizzazione.
        """
        if not token:
            raise ValueError("Il token speciale non può essere vuoto.")

        if token not in self.special_tokens:
            self.special_tokens.append(token)
            self._pattern = None

        return self.add_token(token)

    def add_token(self, token):
        """Aggiunge un token al vocabolario e restituisce il suo id."""
        if token in self.vocabulary:
            return self.vocabulary[token]

        token_id = len(self.vocabulary)
        self.vocabulary[token] = token_id
        self.reverse_vocabulary[token_id] = token
        return token_id

    def fit(self, text, max_vocab_size=None):
        """
        Costruisce (o estende) il vocabolario dal testo fornito.

        Se è impostato un limite di vocabolario, vengono mantenuti solo i token
        più frequenti, riservando spazio ai token speciali.
        """
        max_size = max_vocab_size if max_vocab_size is not None else self.max_vocab_size
        tokens = self.pre_tokenize(text)

        if max_size is not None:
            reserved = len(self.special_tokens)
            budget = max(0, int(max_size) - reserved)
            freq = Counter(tokens)
            keep = {token for token, _ in freq.most_common(budget)}
        else:
            keep = None

        for token in tokens:
            if token not in self.vocabulary:
                if keep is not None and token not in keep:
                    continue
                self.add_token(token)

        return self

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text, add_end_of_text=False):
        tokens = self.pre_tokenize(text)

        unk_id = self.vocabulary[self.UNK_TOKEN]
        token_ids = [self.vocabulary.get(token, unk_id) for token in tokens]

        if add_end_of_text:
            token_ids.append(self.vocabulary[self.END_OF_TEXT_TOKEN])

        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        tokens = [
            self.reverse_vocabulary.get(
                int(token_id),
                self.UNK_TOKEN
            )
            for token_id in token_ids
        ]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]

        if self.mode == self.MODE_WHITESPACE:
            return " ".join(tokens)

        return self.detokenize(tokens)

    # ------------------------------------------------------------------
    # Analisi della copertura e dei token sconosciuti (<UNK>)
    # ------------------------------------------------------------------

    def count_unknown(self, text):
        """
        Conta i token del `text` mappati a `<UNK>` dopo la pre-tokenizzazione.

        Utile per misurare la copertura del vocabolario prima e dopo aver
        aumentato `max_vocab_size`.
        """
        tokens = self.pre_tokenize(text)
        unk_id = self.vocabulary[self.UNK_TOKEN]
        return sum(1 for token in tokens if self.vocabulary.get(token, unk_id) == unk_id)

    def unknown_token_rate(self, text):
        """
        Frazione di token di `text` mappati a `<UNK>`.

        Returns:
            float in [0, 1]. Restituisce ``0.0`` per testo vuoto.
        """
        tokens = self.pre_tokenize(text)
        if not tokens:
            return 0.0
        return self.count_unknown(text) / len(tokens)

    def token_coverage(self, text):
        """
        Frazione di **parole distinte** di `text` presenti nel vocabolario.

        A differenza di `unknown_token_rate`, conta una parola una sola volta,
        evidenziando quanto del lessico del corpus sia coperto dal vocabolario.
        """
        words = self.pre_tokenize(text)
        if not words:
            return 0.0
        distinct = set(words)
        covered = sum(1 for word in distinct if word in self.vocabulary)
        return covered / len(distinct)

    def detokenize(self, tokens):
        """
        Ricostruisce il testo dai token di superficie applicando le regole di
        spaziatura dell'italiano scritto (punteggiatura attaccata, elisioni
        unite, parentesi aderenti).
        """
        pieces = []
        previous = None

        for token in tokens:
            if previous is None:
                pieces.append(token)
                previous = token
                continue

            attach = (
                token in self.NO_SPACE_BEFORE
                or previous in self.NO_SPACE_AFTER
                or previous.endswith("'")
                or previous.endswith("\u2019")
            )

            if attach:
                pieces.append(token)
            else:
                pieces.append(" ")
                pieces.append(token)

            previous = token

        return "".join(pieces)

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def state_dict(self):
        """Stato serializzabile del tokenizer."""
        return {
            "format": "jarvis-tokenizer",
            "version": 2,
            "mode": self.mode,
            "lowercase": self.lowercase,
            "max_vocab_size": self.max_vocab_size,
            "special_tokens": list(self.special_tokens),
            "vocabulary": dict(self.vocabulary),
        }

    def load_state_dict(self, state):
        """Ripristina il tokenizer da uno stato prodotto da `state_dict()`."""
        if not isinstance(state, dict) or "vocabulary" not in state:
            raise ValueError("Stato del tokenizer non valido.")

        mode = state.get("mode", self.MODE_WHITESPACE)
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode non supportato nello stato: {mode!r}")

        self.mode = mode
        self.lowercase = bool(state.get("lowercase", False))
        self.max_vocab_size = state.get("max_vocab_size")
        self.special_tokens = list(
            state.get("special_tokens", [self.UNK_TOKEN, self.END_OF_TEXT_TOKEN])
        )
        self.vocabulary = {token: int(token_id) for token, token_id in state["vocabulary"].items()}
        self.reverse_vocabulary = {token_id: token for token, token_id in self.vocabulary.items()}
        self._pattern = None
        return self

    def save(self, path):
        """Salva il tokenizer in formato JSON (portabile e ispezionabile)."""
        target = Path(path)
        if target.parent and str(target.parent):
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.state_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)

    @classmethod
    def load(cls, path):
        """Carica un tokenizer salvato con `save()`."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer non trovato: {path}")

        state = json.loads(source.read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.load_state_dict(state)
        return tokenizer

    # ------------------------------------------------------------------
    # Compatibilità pickle (tokenizer salvati dalle versioni precedenti)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_pattern", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.vocabulary = dict(state.get("vocabulary", {self.UNK_TOKEN: 0, self.END_OF_TEXT_TOKEN: 1}))
        self.reverse_vocabulary = dict(
            state.get("reverse_vocabulary")
            or {token_id: token for token, token_id in self.vocabulary.items()}
        )
        self.max_vocab_size = state.get("max_vocab_size")
        self.mode = state.get("mode", self.MODE_WHITESPACE)
        self.lowercase = bool(state.get("lowercase", False))
        self.special_tokens = list(
            state.get("special_tokens") or [self.UNK_TOKEN, self.END_OF_TEXT_TOKEN]
        )
        self._pattern = None

    # ------------------------------------------------------------------
    # Costruzione da corpus
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus_file(cls, corpus_path, max_vocab_size=None, mode=MODE_WHITESPACE, lowercase=False):
        tokenizer = cls(max_vocab_size=max_vocab_size, mode=mode, lowercase=lowercase)
        text = Path(corpus_path).read_text(encoding="utf-8")
        tokenizer.fit(text)
        return tokenizer
