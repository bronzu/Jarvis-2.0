"""
Algorithmic Verifier
====================

Verificatore di ragionamento algoritmico: opera sulle catene di ragionamento
(CoT) gia' prodotte dal modello e restituisce una decisione. Non genera
risposte nuove, salvo le rigenerazioni di ramo esplicitamente previste.

Integrazione nell'agent loop:

    MODELLO produce CoT
        -> AlgorithmicVerifier.verify(...)  <-- questo modulo
        -> agent loop decide cosa mostrare all'utente

    Dopo la risposta verificata il modulo restituisce ``VerificationResult``.
    Il campo ``verification_failed`` e' esposto ma NON gestito qui: decidere
    cosa fare con un fallimento della verifica spetta all'agent loop.

Responsabilita':
    - generare 3 rami parziali variando solo la temperature (STEP 1);
    - classificare il dominio dell'input con regole esplicite (STEP 2);
    - valutare ogni ramo con controlli deterministici (STEP 3);
    - scartare, rigenerare e scegliere (STEP 4);
    - registrare tutto in JSON lines (STEP 5).

Cosa questo modulo NON fa:
    - non esegue il codice prodotto dal modello: il controllo sintattico
      verifica solo che il codice sia parsabile, non che sia corretto;
    - non valuta la verita' di un'affermazione in linguaggio naturale: sui
      domini non formalizzabili restituisce sempre ``reliable=False``.

Dipendenze: solo libreria standard Python. NumPy viene importato in modo
differito e solo dentro ``generate_branches``, per impostare il seed del
sampler gia' esistente del Neural Core (vedi la nota sul seed piu' in basso).
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

logger = logging.getLogger("JARVIS.AlgorithmicVerifier")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Project root, resolved from this file so the log path does not depend on CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Log directory. Same folder already used by the project's audit system
#: (``config.yaml`` -> ``audit.log_directory: ./audit_logs`` and
#: ``INFRASTRUCTURE/audit/audit_system.py``). No new folder is introduced.
LOG_DIR = PROJECT_ROOT / "audit_logs"

#: One JSON lines record per verification call, same format as the audit system.
LOG_FILE = LOG_DIR / "algorithmic_verifier.jsonl"

#: Fixed temperatures for the 3 branches. Only this parameter varies.
BRANCH_TEMPERATURES: Tuple[float, ...] = (0.2, 0.6, 1.0)

#: Safety limit on generated length per branch. Reuses the value already used
#: for standard generation in ``neural_core/model_interface.py`` (line 160:
#: ``max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_length", 50))``).
MAX_TOKENS_PER_BRANCH = 50

#: Base seed for the per-branch seeds. Fixed so that a given (input, branches)
#: pair always produces the same decision. Overridable per call.
DEFAULT_BASE_SEED = 20260927

#: A branch scoring strictly below this value is discarded.
DISCARD_THRESHOLD = 0.5

#: Total regeneration rounds for the whole call (not per branch).
MAX_REGENERATION_ROUNDS = 2

#: Step score floor: a branch can never go below zero.
SCORE_FLOOR = 0.0

#: Penalty for a pair of consecutive identical normalized steps (loop).
LOOP_PENALTY = 0.3

#: Penalty for each detected lexical contradiction.
CONTRADICTION_PENALTY = 0.5

#: Penalty when the branch's final value contradicts the recomputed arithmetic.
RECONCILIATION_PENALTY = 1.0

#: Score assigned to the syntax check when the domain is ``other``.
NOT_APPLICABLE = None

#: Domain returned by default. This is the explicit default branch of
#: ``classify_domain``, not an implicit leftover.
DEFAULT_DOMAIN = "other"


# ---------------------------------------------------------------------------
# STEP 2 — domain classification rules
# ---------------------------------------------------------------------------

Domain = Literal["arithmetic", "code", "other"]

#: Characters allowed in an arithmetic input: digits, the listed math
#: operators, grouping and comparison punctuation.
ARITHMETIC_ALLOWED_CHARS = frozenset("0123456789+-*/%().,=<>^ \t\n")

#: Sentence punctuation removed before the arithmetic check. These characters
#: are treated as token delimiters, NOT as allowed math characters: a natural
#: language question such as "Quanto fa 2 + 3?" must still classify as
#: arithmetic. The list is fixed and exhaustive. Grouping characters ``(`` and
#: ``)`` are deliberately absent: they are already allowed math characters.
ARITHMETIC_SENTENCE_PUNCTUATION = frozenset("?!;:\"'«»")

#: Predefined math keywords. An arithmetic input may be built out of these
#: words plus the allowed characters. The list is fixed and exhaustive.
MATH_KEYWORDS: Tuple[str, ...] = (
    "somma",
    "prodotto",
    "differenza",
    "quoziente",
    "risolvi",
    "calcola",
    "equazione",
    "quanto fa",
    "quanto e",
    "moltiplicazione",
    "divisione",
    "addizione",
    "sottrazione",
    "esponente",
    "radice",
    "media",
)

#: Minimum number of distinct code markers for the ``code`` domain.
CODE_MARKER_THRESHOLD = 2

#: Code markers. Fenced code blocks are detected separately, by the backticks.
CODE_KEYWORDS: Tuple[str, ...] = (
    "def",
    "class",
    "import",
    "function",
    "return",
    "print",
    "{",
    "}",
    ";",
)

#: Fenced code block delimiter.
CODE_FENCE = "```"


# ---------------------------------------------------------------------------
# STEP 3b — coherence rules
# ---------------------------------------------------------------------------

#: Contradictory marker pairs. A contradiction is counted when the first
#: marker appears in a step and the second marker appears in a LATER step
#: of the same branch.
CONTRADICTION_MARKER_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("quindi", "ma in realtà non"),
    ("quindi", "in realtà non"),
    ("però", "ma in realtà"),
    ("la risposta è", "la risposta non è"),
    ("il risultato è", "il risultato non è"),
    ("è corretto", "non è corretto"),
    ("è vero", "non è vero"),
    ("confermo", "smentisco"),
)

#: Markers that introduce a claimed final value in a step.
ANSWER_MARKERS: Tuple[str, ...] = ("risposta", "risultato", "quindi")

#: Numbers cited in a step, as required by the mission specification.
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

#: Candidate arithmetic substrings inside a branch text. Deliberately
#: permissive; every candidate is then validated against the AST whitelist.
_ARITHMETIC_CANDIDATE_RE = re.compile(r"[\d(][\d\s.,+\-*/%()^]*[\d)]")

#: Call-shaped and attribute-shaped substrings. These are what a non
#: arithmetic snippet disguised as arithmetic looks like (``len(abc)``,
#: ``os.remove``); they are collected separately because they carry no digit
#: and would be filtered out by the arithmetic candidate rules.
_CODE_SHAPED_CANDIDATE_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*(?:\s*\.\s*[A-Za-z_][A-Za-z_0-9]*)*\s*\([^()]*\)"
    r"|[A-Za-z_][A-Za-z_0-9]*(?:\s*\.\s*[A-Za-z_][A-Za-z_0-9]*)+"
)

#: Arithmetic operator characters, used to reject trivial candidates.
_ARITHMETIC_OPERATOR_CHARS = frozenset("+-*/%^")

#: AST node types accepted for an arithmetic expression. Compared by name to
#: stay clear of deprecated aliases (``ast.Num`` no longer exists on 3.12+).
_WHITELISTED_AST_NODES = frozenset(
    {
        "Expression",
        "Constant",
        "Num",
        "BinOp",
        "UnaryOp",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "FloorDiv",
        "Mod",
        "Pow",
        "USub",
        "UAdd",
        "Load",
    }
)

#: Numeric constant types accepted inside an arithmetic expression.
_NUMERIC_CONSTANTS = (int, float, bool, complex)

#: Guard against trivially large exponentiation on untrusted branch text.
_MAX_POW_EXPONENT = 64


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class Branch:
    """A single generated reasoning branch.

    Attributes:
        id: Stable identifier, e.g. ``"branch_0"``.
        temperature: Sampling temperature used for this branch.
        seed: Explicit seed applied before generation, for reproducibility.
        testo_completo: Full text produced by the model for this branch.
        step_intermedi: The chain of thought split into steps (one per line).
        token_count: Number of generated tokens, as reported by the generator.
    """

    id: str
    temperature: float
    seed: int
    testo_completo: str
    step_intermedi: List[str] = field(default_factory=list)
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation, used by the JSON lines log."""
        return {
            "id": self.id,
            "temperature": self.temperature,
            "seed": self.seed,
            "testo_completo": self.testo_completo,
            "step_intermedi": list(self.step_intermedi),
            "token_count": self.token_count,
        }


@dataclass
class BranchEvaluation:
    """Intermediate scores of a single branch (STEP 3).

    Every individual control is recorded, not only the final score, so the log
    can explain why a branch was kept or discarded.

    Attributes:
        branch_id: Identifier of the evaluated branch.
        temperature: Temperature of the evaluated branch.
        seed: Seed of the evaluated branch.
        round_index: Generation round this evaluation belongs to (0 = first).
        score_sintassi: Syntax check result: 1.0 pass, 0.0 fail, ``None`` when
            not applicable (domain ``other``).
        sintassi_dettaglio: Human-readable explanation of ``score_sintassi``.
        loop_rilevati: Number of consecutive identical step pairs found.
        contraddizioni_rilevate: Number of lexical contradictions found.
        contraddizioni_dettaglio: Description of each contradiction found.
        riconciliazione_eseguita: Whether a recomputation was actually performed.
        riconciliazione_dettaglio: Explanation of the recomputation outcome.
        score_ramo: Final branch score, floored at ``SCORE_FLOOR``.
        scarto: ``True`` when ``score_ramo < DISCARD_THRESHOLD``.
        motivo_scarto: Human-readable discard reason, empty when kept.
    """

    branch_id: str
    temperature: float
    seed: int
    round_index: int
    score_sintassi: Optional[float]
    sintassi_dettaglio: str
    loop_rilevati: int
    contraddizioni_rilevate: int
    contraddizioni_dettaglio: List[str]
    riconciliazione_eseguita: bool
    riconciliazione_dettaglio: str
    score_ramo: float
    scarto: bool
    motivo_scarto: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation, used by the JSON lines log."""
        return {
            "branch_id": self.branch_id,
            "temperature": self.temperature,
            "seed": self.seed,
            "round_index": self.round_index,
            "score_sintassi": self.score_sintassi,
            "sintassi_dettaglio": self.sintassi_dettaglio,
            "loop_rilevati": self.loop_rilevati,
            "contraddizioni_rilevate": self.contraddizioni_rilevate,
            "contraddizioni_dettaglio": list(self.contraddizioni_dettaglio),
            "riconciliazione_eseguita": self.riconciliazione_eseguita,
            "riconciliazione_dettaglio": self.riconciliazione_dettaglio,
            "score_ramo": self.score_ramo,
            "scarto": self.scarto,
            "motivo_scarto": self.motivo_scarto,
        }


@dataclass
class VerificationResult:
    """Outcome of one verification call (STEP 5).

    Attributes:
        input_text: The original input that was verified.
        domain: One of ``"arithmetic"``, ``"code"``, ``"other"``.
        reliable: Whether the chosen branch can be trusted. Always ``False``
            on the ``other`` domain, where no formal check is possible.
        chosen_branch_id: Identifier of the selected branch, ``None`` only if
            no branch was produced at all.
        chosen_branch_text: Text of the selected branch.
        chosen_branch_score: Score of the selected branch.
        discarded_branches: Every branch whose score fell below the threshold,
            each with its id, score and discard reason.
        regeneration_rounds_used: Regeneration rounds actually performed.
        verification_failed: ``True`` when even the regeneration rounds left no
            usable branch. Exposed only: the agent loop decides what to do.
        timestamp: ISO-8601 UTC timestamp of the call.
        branches: Full branch payloads, one per generated branch.
        evaluations: Intermediate scores of STEP 3 for every branch, not only
            the chosen one.
    """

    input_text: str
    domain: str
    reliable: bool
    chosen_branch_id: Optional[str]
    chosen_branch_text: Optional[str]
    chosen_branch_score: Optional[float]
    discarded_branches: List[Dict[str, Any]] = field(default_factory=list)
    regeneration_rounds_used: int = 0
    verification_failed: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    branches: List[Dict[str, Any]] = field(default_factory=list)
    evaluations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation, written as one JSON lines record."""
        return {
            "input_text": self.input_text,
            "domain": self.domain,
            "reliable": self.reliable,
            "chosen_branch_id": self.chosen_branch_id,
            "chosen_branch_text": self.chosen_branch_text,
            "chosen_branch_score": self.chosen_branch_score,
            "discarded_branches": list(self.discarded_branches),
            "regeneration_rounds_used": self.regeneration_rounds_used,
            "verification_failed": self.verification_failed,
            "timestamp": self.timestamp,
            "branches": list(self.branches),
            "evaluations": list(self.evaluations),
        }


class VerifierError(RuntimeError):
    """Raised when the verifier receives invalid input."""


# ---------------------------------------------------------------------------
# STEP 1 — branch generation
# ---------------------------------------------------------------------------


def _derive_branch_seed(base_seed: int, branch_index: int, round_index: int) -> int:
    """Derive a stable, distinct seed for one branch in one round."""
    return int(base_seed) + int(round_index) * 1000 + int(branch_index)


def _split_steps(text: str) -> List[str]:
    """Split a chain of thought into steps, one per non-empty line."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_generated_text(payload: Any) -> str:
    """Read the generated text out of whatever the generator returned.

    ``NeuralModelInterface.generate`` returns a dict carrying ``"text"``; a
    plain string is accepted too so the verifier is not tied to one generator.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        value = payload.get("text")
        return value if isinstance(value, str) else str(value or "")
    return str(payload)


def _extract_token_count(payload: Any, text: str) -> int:
    """Token count reported by the generator, with a deterministic fallback.

    ``NeuralModelInterface.generate`` reports ``"generated_tokens"``. When that
    is absent the count falls back to whitespace-separated tokens, so the value
    stays deterministic for the same (input, branches) pair.
    """
    if isinstance(payload, dict):
        reported = payload.get("generated_tokens")
        if isinstance(reported, int) and reported >= 0:
            return reported
    return len(text.split())


def _generate_one(
    model: Any,
    prompt: str,
    temperature: float,
    seed: int,
) -> Tuple[str, int]:
    """Generate a single branch through the project's existing generator.

    The sampling itself is performed by ``Decoder.sample`` inside
    ``NeuralModelInterface.generate`` (``neural_core/model_interface.py``), which
    is reused as is: ``neural_core/network.py`` and ``neural_core/trainer.py``
    are not touched.

    ``Decoder.sample`` falls back to the global NumPy generator when no explicit
    ``rng`` is passed, and ``NeuralModelInterface.generate`` does not expose one.
    The only way to make a branch reproducible without modifying those files is
    therefore to seed that global generator right before the call.

    Seeding the global generator would otherwise be a side effect on the whole
    process: the seed and the draws made for this branch would survive the call.
    The previous state is therefore saved with ``np.random.get_state()`` and
    restored with ``np.random.set_state()`` right after the generation, in a
    ``finally`` block so that the restoration also happens if the generator
    raises. Each branch, and each regeneration round, is scoped in the same way,
    so ``generate_branches`` leaves the global RNG state untouched on return.
    """
    import numpy as np

    saved_state = np.random.get_state()
    try:
        np.random.seed(seed)
        payload = model.generate(
            prompt,
            max_new_tokens=MAX_TOKENS_PER_BRANCH,
            temperature=temperature,
        )
    finally:
        np.random.set_state(saved_state)
    text = _extract_generated_text(payload)
    return text, _extract_token_count(payload, text)


def generate_branches(
    prompt: str,
    model: Any,
    n_branches: int = 3,
    base_seed: int = DEFAULT_BASE_SEED,
) -> List[Branch]:
    """Generate ``n_branches`` partial reasoning branches (STEP 1).

    The branches differ only in the sampling temperature, taken in order from
    :data:`BRANCH_TEMPERATURES`. Each branch gets an explicit, distinct seed,
    stored on the branch and in the log, so the same call is reproducible.

    Args:
        prompt: The input handed to the model.
        model: Object exposing ``generate(prompt, **kwargs)``. In this project
            that is ``NeuralModelInterface``, which wraps ``MicroTransformer``
            and performs the sampling through ``Decoder.sample``.
        n_branches: How many branches to generate, capped at the number of
            fixed temperatures available.
        base_seed: Base for the per-branch seeds.

    Returns:
        The generated branches, ordered by ascending temperature.

    Raises:
        VerifierError: If the prompt is not a string, the branch count is out
            of range, or the model exposes no ``generate`` method.
    """
    if not isinstance(prompt, str):
        raise VerifierError("prompt deve essere una stringa.")
    if not isinstance(n_branches, int) or isinstance(n_branches, bool):
        raise VerifierError("n_branches deve essere un intero.")
    if not 1 <= n_branches <= len(BRANCH_TEMPERATURES):
        raise VerifierError(
            f"n_branches deve essere compreso tra 1 e {len(BRANCH_TEMPERATURES)}."
        )
    if model is None or not callable(getattr(model, "generate", None)):
        raise VerifierError(
            "model deve esporre un metodo generate(prompt, **kwargs)."
        )

    branches: List[Branch] = []
    for index in range(n_branches):
        temperature = BRANCH_TEMPERATURES[index]
        seed = _derive_branch_seed(base_seed, index, 0)
        text, token_count = _generate_one(model, prompt, temperature, seed)
        branch = Branch(
            id=f"branch_{index}",
            temperature=temperature,
            seed=seed,
            testo_completo=text,
            step_intermedi=_split_steps(text),
            token_count=token_count,
        )
        branches.append(branch)
        logger.debug(
            "Ramo %s generato: temperature=%s seed=%s token=%s",
            branch.id,
            branch.temperature,
            branch.seed,
            branch.token_count,
        )
    return branches


# ---------------------------------------------------------------------------
# STEP 2 — domain classification
# ---------------------------------------------------------------------------


def _strip_arithmetic_prose(input_text: str) -> str:
    """Remove the parts of an input that are prose rather than math.

    Sentence punctuation and every occurrence of a predefined math keyword are
    dropped, leaving only what the arithmetic character check must inspect.
    """
    remaining = "".join(
        character
        for character in input_text
        if character not in ARITHMETIC_SENTENCE_PUNCTUATION
    )
    for keyword in MATH_KEYWORDS:
        remaining = remaining.replace(keyword, " ")
    return remaining


def _looks_arithmetic(input_text: str) -> bool:
    """True when only allowed math characters survive the prose stripping."""
    remaining = _strip_arithmetic_prose(input_text)
    return bool(remaining) and all(
        character in ARITHMETIC_ALLOWED_CHARS for character in remaining
    )


def _has_letters(input_text: str) -> bool:
    """True when the text carries at least one alphabetic character."""
    return any(character.isalpha() for character in input_text)


def _has_math_keyword(input_text: str) -> bool:
    """True when at least one predefined math keyword is present."""
    return any(keyword in input_text for keyword in MATH_KEYWORDS)


def _has_digit(input_text: str) -> bool:
    """True when at least one digit is present."""
    return any(character.isdigit() for character in input_text)


def _has_code_fence(input_text: str) -> bool:
    """True when the text contains a fenced code block."""
    return CODE_FENCE in input_text


def _count_code_markers(input_text: str) -> int:
    """Number of distinct code markers present in the text."""
    found = {keyword for keyword in CODE_KEYWORDS if keyword in input_text}
    return len(found)


def classify_domain(input_text: str) -> Domain:
    """Classify the domain of an input using fixed rules only (STEP 2).

    The rules, in order:

    1. ``arithmetic`` when the input is non-empty, contains at least one digit,
       every character outside :data:`ARITHMETIC_SENTENCE_PUNCTUATION` belongs
       to :data:`ARITHMETIC_ALLOWED_CHARS`, and either the input carries no
       letters at all (a bare expression such as ``2 + 3``) or it contains at
       least one predefined math keyword from :data:`MATH_KEYWORDS`.
    2. ``code`` when the text contains a fenced block delimited by ``` or at
       least :data:`CODE_MARKER_THRESHOLD` distinct code markers.
    3. ``other`` in every remaining case. This is the explicit default.

    The domain only declares declared reliability. It never blocks
    verification: the verifier still runs on ``other`` and the result is
    marked ``reliable=False``.

    Args:
        input_text: The input to classify.

    Returns:
        ``"arithmetic"``, ``"code"`` or ``"other"``.
    """
    if not isinstance(input_text, str) or not input_text.strip():
        return DEFAULT_DOMAIN

    stripped = input_text.strip()
    lowered = stripped.lower()

    if (
        _looks_arithmetic(lowered)
        and _has_digit(lowered)
        and (not _has_letters(lowered) or _has_math_keyword(lowered))
    ):
        return "arithmetic"

    if _has_code_fence(stripped) or _count_code_markers(lowered) >= CODE_MARKER_THRESHOLD:
        return "code"

    return DEFAULT_DOMAIN


# ---------------------------------------------------------------------------
# STEP 3a — syntax check
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str:
    """Extract the code out of a branch text.

    The content of a fenced block is used when present, otherwise the whole
    text. A branch that wraps its code in prose will therefore usually fail the
    syntax check; that limitation is documented in the mission report.
    """
    if CODE_FENCE in text:
        parts = text.split(CODE_FENCE)
        if len(parts) >= 3:
            return parts[1]
        return parts[-1]
    return text


def _find_arithmetic_expressions(text: str) -> List[str]:
    """Candidate arithmetic expressions contained in a text."""
    candidates: List[str] = []
    for raw in _ARITHMETIC_CANDIDATE_RE.findall(text):
        candidate = raw.strip().rstrip("+-*/%^.,")
        if not candidate or not any(char.isdigit() for char in candidate):
            continue
        if not any(char in _ARITHMETIC_OPERATOR_CHARS for char in candidate):
            continue
        candidates.append(candidate)
    return candidates


def _find_code_shaped_expressions(text: str) -> List[str]:
    """Call-shaped and attribute-shaped substrings contained in a text."""
    return [match.strip() for match in _CODE_SHAPED_CANDIDATE_RE.findall(text)]


def _offending_ast_nodes(tree: ast.AST) -> List[str]:
    """Names of AST node types that are not whitelisted for arithmetic.

    Any ``Call``, ``Name`` or ``Attribute`` shows up here, which is how a
    non-arithmetic snippet disguised as arithmetic is detected.
    """
    return sorted(
        {
            type(node).__name__
            for node in ast.walk(tree)
            if type(node).__name__ not in _WHITELISTED_AST_NODES
        }
    )


def _parse_arithmetic(expression: str) -> Tuple[Optional[ast.Expression], List[str]]:
    """Parse an expression in eval mode and check the node whitelist.

    Returns:
        ``(tree, offending_nodes)``. ``tree`` is ``None`` when the expression
        does not parse; ``offending_nodes`` is empty when the expression is
        fully whitelisted.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None, []

    for node in ast.walk(tree):
        if type(node).__name__ == "Constant" and not isinstance(
            node.value, _NUMERIC_CONSTANTS
        ):
            return tree, sorted({type(node).__name__ for node in ast.walk(tree)})

    return tree, _offending_ast_nodes(tree)


def _evaluate_arithmetic(node: ast.AST) -> Any:
    """Evaluate a whitelisted arithmetic AST without eval/exec.

    Only the node types in :data:`_WHITELISTED_AST_NODES` are handled. Anything
    else raises ``ValueError``. Exponents are capped at
    :data:`_MAX_POW_EXPONENT` so untrusted branch text cannot force an
    unbounded computation.
    """
    name = type(node).__name__

    if name == "Expression":
        return _evaluate_arithmetic(node.body)
    if name == "Constant":
        return node.value
    if name == "UnaryOp":
        operand = _evaluate_arithmetic(node.operand)
        if name_of(node.op) == "USub":
            return -operand
        if name_of(node.op) == "UAdd":
            return +operand
        raise ValueError(f"Operatore unario non consentito: {name_of(node.op)}")
    if name == "BinOp":
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        operator = name_of(node.op)
        if operator == "Add":
            return left + right
        if operator == "Sub":
            return left - right
        if operator == "Mult":
            return left * right
        if operator == "Div":
            return left / right
        if operator == "FloorDiv":
            return left // right
        if operator == "Mod":
            return left % right
        if operator == "Pow":
            if abs(complex(right).real if isinstance(right, complex) else right) > _MAX_POW_EXPONENT:
                raise ValueError("Esponente oltre il limite di sicurezza.")
            return left ** right
        raise ValueError(f"Operatore binario non consentito: {operator}")

    raise ValueError(f"Nodo non consentito: {name}")


def name_of(node: ast.AST) -> str:
    """Name of an AST node, used to avoid deprecated isinstance checks."""
    return type(node).__name__


def _last_arithmetic_result(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Recompute the last whitelisted arithmetic expression found in a text.

    Returns:
        ``(value, expression)``, both ``None`` when no arithmetic expression
        could be validated and evaluated.
    """
    for expression in reversed(_find_arithmetic_expressions(text)):
        tree, offending = _parse_arithmetic(expression)
        if tree is None or offending:
            continue
        try:
            value = _evaluate_arithmetic(tree)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
            continue
        return float(value), expression
    return None, None


def _last_cited_number(text: str) -> Optional[float]:
    """Last number cited in a text, per the ``\\d+(\\.\\d+)?`` rule."""
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _last_answer_number(step: str) -> Optional[float]:
    """Last number cited in a step that carries an answer marker."""
    normalized = step.lower()
    if not any(marker in normalized for marker in ANSWER_MARKERS):
        return None
    return _last_cited_number(step)


def _check_code_syntax(text: str) -> Tuple[Optional[float], str]:
    """Syntax check for the ``code`` domain.

    Scope: this verifies only the SHAPE of the code (is it parsable). It says
    nothing about semantic correctness or runtime behaviour. Real execution
    verification is out of scope for this mission.
    """
    code = extract_code(text)
    if not code.strip():
        return 0.0, "Nessun codice trovato nel ramo."
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        return 0.0, f"ast.parse fallito: {error}"
    return 1.0, "Codice parsabile (solo forma, non semantica)."


def _check_arithmetic_syntax(text: str) -> Tuple[Optional[float], str]:
    """Syntax check for the ``arithmetic`` domain.

    The expression is validated with ``ast.parse`` in eval mode and the AST
    must contain exclusively whitelisted nodes: no ``Call``, no ``Name``, no
    ``Attribute``. A call-shaped or attribute-shaped expression found anywhere
    in the branch is reported as disguised non-arithmetic code.
    """
    disguised = _find_disguised_code(text)
    if disguised is not None:
        expression, offending = disguised
        logger.warning(
            "Codice non aritmetico camuffato, nodi non whitelisted %s: %r",
            offending,
            expression,
        )
        return 0.0, (
            "Tentativo di codice non aritmetico camuffato: nodi "
            f"{offending} in {expression!r}"
        )

    candidates = _find_arithmetic_expressions(text)
    if not candidates:
        return 0.0, "Nessuna espressione aritmetica trovata nel ramo."

    for expression in reversed(candidates):
        tree, offending = _parse_arithmetic(expression)
        if tree is None:
            continue
        if offending:
            logger.warning(
                "Espressione aritmetica con nodi non whitelisted %s: %r",
                offending,
                expression,
            )
            return 0.0, (
                "Tentativo di codice non aritmetico camuffato: nodi "
                f"{offending} in {expression!r}"
            )
        return 1.0, f"Espressione aritmetica whitelisted: {expression!r}"

    return 0.0, "Nessuna espressione aritmetica parsabile nel ramo."


def _find_disguised_code(text: str) -> Optional[Tuple[str, List[str]]]:
    """Find a call-shaped or attribute-shaped expression in a branch.

    Returns the offending expression together with the non-whitelisted AST node
    types it contains, or ``None`` when there is none.
    """
    for expression in _find_code_shaped_expressions(text):
        tree, offending = _parse_arithmetic(expression)
        if tree is not None and offending:
            return expression, offending
    return None


# ---------------------------------------------------------------------------
# STEP 3b — coherence between steps
# ---------------------------------------------------------------------------


def _normalize_step(step: str) -> str:
    """Normalize a step for comparison: lowercase, whitespace reduced."""
    return re.sub(r"\s+", " ", step.strip().lower())


def _count_loops(steps: Sequence[str]) -> int:
    """Count pairs of consecutive identical normalized steps."""
    normalized = [_normalize_step(step) for step in steps]
    return sum(
        1
        for index in range(1, len(normalized))
        if normalized[index] and normalized[index] == normalized[index - 1]
    )


def _find_contradictions(steps: Sequence[str]) -> List[str]:
    """Find lexical contradictions inside a single branch.

    Two independent checks, both reported:

    1. Contradictory marker pairs, where the second marker appears in a later
       step than the first.
    2. Two different final values claimed on the same branch, extracted with
       ``\\d+(\\.\\d+)?`` from the last number cited by a step carrying a
       "risposta" / "risultato" / "quindi" marker.
    """
    found: List[str] = []
    normalized = [_normalize_step(step) for step in steps]

    # Overlapping marker pairs would otherwise count the same clash twice
    # (e.g. "ma in realtà non" also contains "in realtà non"). A given
    # (earlier step, later step) pair is therefore counted at most once.
    seen_pairs = set()
    for first, second in CONTRADICTION_MARKER_PAIRS:
        for index, step in enumerate(normalized):
            if first not in step:
                continue
            for later_index in range(index + 1, len(normalized)):
                if second in normalized[later_index]:
                    if (index, later_index) in seen_pairs:
                        break
                    seen_pairs.add((index, later_index))
                    found.append(
                        f"marker contraddittori {first!r} (step {index}) "
                        f"poi {second!r} (step {later_index})"
                    )
                    break

    claimed: List[Tuple[int, float]] = []
    for index, step in enumerate(normalized):
        value = _last_answer_number(step)
        if value is not None:
            claimed.append((index, value))

    for position in range(1, len(claimed)):
        if claimed[position][1] != claimed[0][1]:
            found.append(
                "valori finali contraddittori: "
                f"{claimed[0][1]} (step {claimed[0][0]}) vs "
                f"{claimed[position][1]} (step {claimed[position][0]})"
            )

    return found


def _check_reconciliation(text: str, domain: str) -> Tuple[bool, bool, str]:
    """Recompute the arithmetic and compare it with the final cited value.

    Applies only to the ``arithmetic`` and ``code`` domains. The recomputation
    is always the ARITHMETIC calculation, never the execution of the code:
    running generated code is out of scope for this mission.

    Returns:
        ``(eseguito, coincide, dettaglio)``. When ``eseguito`` is ``False`` the
        comparison was impossible and no penalty may be applied.
    """
    if domain not in ("arithmetic", "code"):
        return False, False, "Dominio non formalizzabile: riconciliazione non applicabile."

    recomputed, expression = _last_arithmetic_result(text)
    if recomputed is None or expression is None:
        return False, False, "Nessun calcolo aritmetico riconciliabile nel ramo."

    claimed = _last_cited_number(text)
    if claimed is None:
        return False, False, "Nessun valore finale citato nel ramo: riconciliazione impossibile."

    if abs(claimed - recomputed) <= 1e-9:
        return True, True, (
            f"Riconciliato: {expression!r} = {recomputed:g}, valore citato {claimed:g}."
        )

    return True, False, (
        f"Riconciliazione fallita: {expression!r} = {recomputed:g}, "
        f"ma il ramo dichiara {claimed:g}."
    )


# ---------------------------------------------------------------------------
# STEP 3 — branch evaluation
# ---------------------------------------------------------------------------


def evaluate_branch(
    branch: Branch,
    domain: str,
    round_index: int = 0,
) -> BranchEvaluation:
    """Evaluate a single branch and record every individual control (STEP 3).

    The controls are applied in this order: syntax, then step coherence, then
    the final score with its floor.

    Args:
        branch: The branch to evaluate.
        domain: The domain of the input, from :func:`classify_domain`.
        round_index: Generation round this branch belongs to.

    Returns:
        The :class:`BranchEvaluation` with all intermediate scores.
    """
    text = branch.testo_completo
    steps = branch.step_intermedi or _split_steps(text)

    if domain == "code":
        score_sintassi, sintassi_dettaglio = _check_code_syntax(text)
    elif domain == "arithmetic":
        score_sintassi, sintassi_dettaglio = _check_arithmetic_syntax(text)
    else:
        score_sintassi = NOT_APPLICABLE
        sintassi_dettaglio = "Controllo sintattico non applicabile al dominio 'other'."

    loops = _count_loops(steps)
    contradictions = _find_contradictions(steps)
    riconciliato, riconciliato_ok, riconciliazione_dettaglio = _check_reconciliation(
        text, domain
    )

    score = 1.0 if score_sintassi is None else float(score_sintassi)
    score -= LOOP_PENALTY * loops
    score -= CONTRADICTION_PENALTY * len(contradictions)
    if riconciliato and not riconciliato_ok:
        score -= RECONCILIATION_PENALTY
    score = max(score, SCORE_FLOOR)

    scarto = score < DISCARD_THRESHOLD
    reasons: List[str] = []
    if score_sintassi is not None and score_sintassi < 1.0:
        reasons.append(sintassi_dettaglio)
    if loops:
        reasons.append(f"{loops} loop di step consecutivi identici")
    if contradictions:
        reasons.append(f"{len(contradictions)} contraddizioni lessicali")
    if riconciliato and not riconciliato_ok:
        reasons.append(riconciliazione_dettaglio)

    return BranchEvaluation(
        branch_id=branch.id,
        temperature=branch.temperature,
        seed=branch.seed,
        round_index=round_index,
        score_sintassi=score_sintassi,
        sintassi_dettaglio=sintassi_dettaglio,
        loop_rilevati=loops,
        contraddizioni_rilevate=len(contradictions),
        contraddizioni_dettaglio=contradictions,
        riconciliazione_eseguita=riconciliato,
        riconciliazione_dettaglio=riconciliazione_dettaglio,
        score_ramo=score,
        scarto=scarto,
        motivo_scarto="; ".join(reasons),
    )


# ---------------------------------------------------------------------------
# STEP 4 — decision
# ---------------------------------------------------------------------------


def _select_best(
    branches: Sequence[Branch],
    evaluations: Sequence[BranchEvaluation],
) -> Optional[BranchEvaluation]:
    """Pick the highest scoring branch, ties broken by lowest temperature."""
    by_id = {branch.id: branch for branch in branches}
    usable = [evaluation for evaluation in evaluations if not evaluation.scarto]

    def sort_key(evaluation: BranchEvaluation) -> Tuple[float, float]:
        temperature = by_id[evaluation.branch_id].temperature
        return (-evaluation.score_ramo, temperature)

    if not usable:
        return None
    return sorted(usable, key=sort_key)[0]


# ---------------------------------------------------------------------------
# STEP 5 — logging
# ---------------------------------------------------------------------------


def _write_log(result: VerificationResult, log_path: Optional[Path]) -> None:
    """Append one JSON lines record for this call."""
    target = Path(log_path) if log_path is not None else LOG_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    except OSError as error:
        logger.error("Impossibile scrivere il log di verifica su %s: %s", target, error)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def verify(
    input_text: str,
    model: Any,
    base_seed: int = DEFAULT_BASE_SEED,
    log_path: Optional[Path] = None,
) -> VerificationResult:
    """Verify a model response end to end (STEP 1 to STEP 5).

    The pipeline is: classify the input, generate the branches, evaluate them,
    discard and regenerate the discarded ones for at most
    :data:`MAX_REGENERATION_ROUNDS` rounds, then choose.

    Args:
        input_text: The input to verify.
        model: Object exposing ``generate(prompt, **kwargs)``, i.e.
            ``NeuralModelInterface``.
        base_seed: Base for the per-branch seeds.
        log_path: Override for the JSON lines log destination.

    Returns:
        The :class:`VerificationResult` for this call. It is also written to
        the log.
    """
    if not isinstance(input_text, str):
        raise VerifierError("input_text deve essere una stringa.")

    domain = classify_domain(input_text)

    branches = generate_branches(
        input_text, model, n_branches=len(BRANCH_TEMPERATURES), base_seed=base_seed
    )
    branches_by_id: Dict[str, Branch] = {branch.id: branch for branch in branches}
    evaluations = [evaluate_branch(branch, domain) for branch in branches]

    rounds_used = 0
    for round_index in range(1, MAX_REGENERATION_ROUNDS + 1):
        discarded_slots = [evaluation for evaluation in evaluations if evaluation.scarto]
        if len(discarded_slots) < 2:
            break

        rounds_used = round_index
        for evaluation in discarded_slots:
            slot_index = int(evaluation.branch_id.rsplit("_", 1)[-1])
            temperature = BRANCH_TEMPERATURES[slot_index]
            seed = _derive_branch_seed(base_seed, slot_index, round_index)
            text, token_count = _generate_one(model, input_text, temperature, seed)
            branch = Branch(
                id=evaluation.branch_id,
                temperature=temperature,
                seed=seed,
                testo_completo=text,
                step_intermedi=_split_steps(text),
                token_count=token_count,
            )
            branches_by_id[branch.id] = branch
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.branch_id != branch.id
            ]
            evaluations.append(evaluate_branch(branch, domain, round_index=round_index))

    chosen = _select_best(list(branches_by_id.values()), evaluations)
    verification_failed = chosen is None

    if chosen is None:
        surviving = [evaluation for evaluation in evaluations if not evaluation.scarto]
        pool = surviving or list(evaluations)
        if pool:
            by_id = branches_by_id
            chosen = sorted(
                pool,
                key=lambda evaluation: (
                    -evaluation.score_ramo,
                    by_id[evaluation.branch_id].temperature,
                ),
            )[0]

    chosen_branch = branches_by_id.get(chosen.branch_id) if chosen else None
    reliable = bool(chosen is not None and not verification_failed and domain != "other")

    discarded = [
        {
            "id": evaluation.branch_id,
            "score": evaluation.score_ramo,
            "motivo_scarto": evaluation.motivo_scarto,
        }
        for evaluation in sorted(evaluations, key=lambda item: item.branch_id)
        if evaluation.scarto
    ]

    result = VerificationResult(
        input_text=input_text,
        domain=domain,
        reliable=reliable,
        chosen_branch_id=chosen.branch_id if chosen else None,
        chosen_branch_text=chosen_branch.testo_completo if chosen_branch else None,
        chosen_branch_score=chosen.score_ramo if chosen else None,
        discarded_branches=discarded,
        regeneration_rounds_used=rounds_used,
        verification_failed=verification_failed,
        branches=[
            branches_by_id[branch_id].to_dict()
            for branch_id in sorted(branches_by_id)
        ],
        evaluations=[
            evaluation.to_dict()
            for evaluation in sorted(evaluations, key=lambda item: item.branch_id)
        ],
    )

    _write_log(result, log_path)
    return result
