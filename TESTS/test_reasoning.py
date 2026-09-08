import pytest

from cognitive_core.reasoning.decomposer import Decomposer
from cognitive_core.reasoning.hypothesis import HypothesisGenerator
from cognitive_core.reasoning.reasoner import Reasoner


def test_decomposer_separates_problem_components():
    problem = {
        "facts": ["piove", "la strada è esposta"],
        "rules": [
            {
                "if": ["piove", "la strada è esposta"],
                "then": "la strada è bagnata",
            }
        ],
        "goal": "la strada è bagnata",
    }

    result = Decomposer().decompose(problem)

    assert result == problem
    assert result is not problem
    assert result["facts"] is not problem["facts"]
    assert result["rules"] is not problem["rules"]


def test_decomposer_normalizes_statements():
    problem = {
        "facts": ["  piove  "],
        "rules": [{"if": [" piove "], "then": " strada bagnata "}],
        "goal": " strada bagnata ",
    }

    result = Decomposer().decompose(problem)

    assert result["facts"] == ["piove"]
    assert result["rules"] == [{"if": ["piove"], "then": "strada bagnata"}]
    assert result["goal"] == "strada bagnata"


@pytest.mark.parametrize(
    "problem",
    [
        {},
        {"facts": ["x"], "rules": [], "goal": ""},
        {"facts": ["x"], "rules": [{"if": [], "then": "y"}], "goal": "y"},
    ],
)
def test_decomposer_rejects_invalid_problem(problem):
    with pytest.raises(ValueError):
        Decomposer().decompose(problem)


def test_hypothesis_generator_creates_candidates_from_rules():
    rules = [
        {"if": ["piove"], "then": "strada bagnata"},
        {"if": ["strada bagnata"], "then": "strada scivolosa"},
    ]

    hypotheses = HypothesisGenerator().generate(rules)

    assert hypotheses == [
        {"conditions": ["piove"], "conclusion": "strada bagnata"},
        {
            "conditions": ["strada bagnata"],
            "conclusion": "strada scivolosa",
        },
    ]


def test_hypothesis_validation_requires_all_conditions():
    generator = HypothesisGenerator()
    hypothesis = {
        "conditions": ["piove", "strada esposta"],
        "conclusion": "strada bagnata",
    }

    assert generator.validate(hypothesis, ["piove", "strada esposta"])
    assert not generator.validate(hypothesis, ["piove"])


def test_hypothesis_generator_rejects_invalid_rule():
    with pytest.raises(ValueError):
        HypothesisGenerator().generate([{"if": [], "then": "y"}])


def test_hypothesis_validation_rejects_invalid_input():
    with pytest.raises(TypeError):
        HypothesisGenerator().validate(
            {"conditions": ["x"], "conclusion": "y"},
            {"x"},
        )


def test_reasoner_solves_deductive_problem():
    problem = {
        "facts": ["piove", "strada esposta"],
        "rules": [
            {
                "if": ["piove", "strada esposta"],
                "then": "strada bagnata",
            }
        ],
        "goal": "strada bagnata",
    }

    result = Reasoner().reason(problem)

    assert result["proved"]
    assert result["conclusion"] == "strada bagnata"
    assert result["steps"] == [
        {
            "derived": "strada bagnata",
            "from": ["piove", "strada esposta"],
        }
    ]


def test_reasoner_resolves_multi_step_deduction():
    problem = {
        "facts": ["piove"],
        "rules": [
            {"if": ["piove"], "then": "strada bagnata"},
            {"if": ["strada bagnata"], "then": "strada scivolosa"},
        ],
        "goal": "strada scivolosa",
    }

    result = Reasoner().reason(problem)

    assert result["proved"]
    assert result["known_facts"] == [
        "piove",
        "strada bagnata",
        "strada scivolosa",
    ]
    assert len(result["steps"]) == 2


def test_reasoner_reports_goal_that_cannot_be_proved():
    problem = {
        "facts": ["piove"],
        "rules": [{"if": ["sole"], "then": "strada asciutta"}],
        "goal": "strada asciutta",
    }

    result = Reasoner().reason(problem)

    assert not result["proved"]
    assert result["conclusion"] is None
    assert result["steps"] == []


def test_reasoner_does_not_duplicate_known_conclusions():
    problem = {
        "facts": ["piove", "strada bagnata"],
        "rules": [{"if": ["piove"], "then": "strada bagnata"}],
        "goal": "strada bagnata",
    }

    result = Reasoner().reason(problem)

    assert result["proved"]
    assert result["known_facts"] == ["piove", "strada bagnata"]
    assert result["steps"] == []
