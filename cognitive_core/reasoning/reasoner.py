"""Motore centrale per il ragionamento deduttivo strutturato."""

from cognitive_core.reasoning.decomposer import Decomposer
from cognitive_core.reasoning.hypothesis import HypothesisGenerator
from cognitive_core.world_model.world_model import WorldModel


class Reasoner:
    """Deriva conclusioni applicando regole ai fatti disponibili."""

    def __init__(self, decomposer=None, hypothesis_generator=None, world_model=None):
        self.decomposer = decomposer or Decomposer()
        self.hypothesis_generator = (
            hypothesis_generator or HypothesisGenerator()
        )
        self.world_model = world_model or WorldModel()

    def _tree_of_thought_branches(self, components, known_facts):
        candidate_rules = []
        for rule in components["rules"]:
            if rule["then"] in known_facts:
                continue
            coverage = len(set(rule["if"]) & set(known_facts))
            score = coverage / max(1, len(rule["if"]))
            candidate_rules.append({
                "conclusion": rule["then"],
                "score": score,
                "conditions": list(rule["if"]),
            })

        if not candidate_rules:
            return []

        best = sorted(candidate_rules, key=lambda item: item["score"], reverse=True)
        return best

    def reason(self, problem):
        """Verifica se l'obiettivo può essere dedotto dal problema.

        Le regole sono applicate fino a quando non vengono prodotti nuovi
        fatti. Ogni derivazione viene salvata in una traccia verificabile.
        """
        components = self.decomposer.decompose(problem)
        known_facts = list(components["facts"])
        steps = []
        hypotheses = self.hypothesis_generator.generate(
            components["rules"]
        )

        changed = True
        while changed:
            changed = False

            for hypothesis in hypotheses:
                conclusion = hypothesis["conclusion"]

                if conclusion in known_facts:
                    continue

                if self.hypothesis_generator.validate(
                    hypothesis,
                    known_facts,
                ):
                    known_facts.append(conclusion)
                    steps.append({
                        "derived": conclusion,
                        "from": list(hypothesis["conditions"]),
                    })
                    changed = True

        goal = components["goal"]
        proved = goal in known_facts
        result = {
            "goal": goal,
            "proved": proved,
            "conclusion": goal if proved else None,
            "known_facts": known_facts,
            "steps": steps,
        }

        branches = self._tree_of_thought_branches(components, known_facts)
        if branches:
            result["tree_branches"] = branches
            result["best_branch"] = branches[0]
        return result

    def reason_tree(self, problem):
        """Esposizione esplicita del ragionamento a rami alternativi."""
        result = self.reason(problem)
        result["tree_of_thought"] = result.get("tree_branches", [])
        return result

    def induce(self, examples):
        """Induzione: generalizza una regola da esempi positivi e negativi.

        Args:
            examples: lista di dict con chiavi 'features' (list) e 'label' (str).

        Returns:
            dict con 'rule', 'confidence', 'examples_used'.
        """
        if not isinstance(examples, list) or not examples:
            raise ValueError("examples deve essere una lista non vuota")

        positives = [ex for ex in examples if ex.get("label") == "positive"]
        negatives = [ex for ex in examples if ex.get("label") == "negative"]

        if not positives:
            return {"rule": None, "confidence": 0.0, "examples_used": len(examples)}

        # Generalizza prendendo l'intersezione delle features comuni ai positivi
        common = set(positives[0].get("features", []))
        for ex in positives[1:]:
            common &= set(ex.get("features", []))

        rule = {"if": sorted(common), "then": "positive"} if common else None
        confidence = len(positives) / max(1, len(examples))
        return {
            "rule": rule,
            "confidence": confidence,
            "examples_used": len(examples),
        }

    def classify(self, item, categories):
        """Classificazione: assegna item alla categoria più simile.

        Args:
            item: dict con features da classificare.
            categories: dict {nome_categoria: [features]}.

        Returns:
            dict con 'category', 'score', 'reason'.
        """
        if not isinstance(item, dict) or not isinstance(categories, dict):
            raise TypeError("item e categories devono essere dict")

        item_features = set(item.get("features", []))
        best_category = None
        best_score = -1.0
        best_intersection = 0

        for category, features in categories.items():
            cat_features = set(features)
            intersection = len(item_features & cat_features)
            union = len(item_features | cat_features)
            score = intersection / max(1, union)
            if score > best_score or (score == best_score and intersection > best_intersection):
                best_score = score
                best_category = category
                best_intersection = intersection

        return {
            "category": best_category,
            "score": best_score,
            "reason": f"intersection={best_intersection}",
        }

    def compare(self, a, b, criteria):
        """Confronto multi-criterio tra due alternative.

        Args:
            a, b: dict di alternative.
            criteria: lista di dict con 'name', 'weight', 'higher_is_better'.

        Returns:
            dict con 'winner', 'scores', 'details'.
        """
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise TypeError("a e b devono essere dict")
        if not criteria:
            raise ValueError("criteria non può essere vuoto")

        scores = {"a": 0.0, "b": 0.0}
        details = []

        for criterion in criteria:
            name = criterion["name"]
            weight = float(criterion.get("weight", 1.0))
            higher_is_better = bool(criterion.get("higher_is_better", True))

            value_a = float(a.get(name, 0.0))
            value_b = float(b.get(name, 0.0))

            if value_a == value_b == 0.0:
                score_a = 0.0
                score_b = 0.0
            elif value_a == 0.0:
                score_a = 0.0
                score_b = 1.0
            elif value_b == 0.0:
                score_a = 1.0
                score_b = 0.0
            else:
                total = value_a + value_b
                score_a = value_a / total if higher_is_better else value_b / total
                score_b = value_b / total if higher_is_better else value_a / total

            scores["a"] += score_a * weight
            scores["b"] += score_b * weight
            details.append({
                "criterion": name,
                "a": value_a,
                "b": value_b,
                "score_a": score_a,
                "score_b": score_b,
            })

        winner = "a" if scores["a"] >= scores["b"] else "b"
        return {
            "winner": winner,
            "scores": scores,
            "details": details,
        }

    def solve_math(self, expression: str):
        """Problem solving matematico limitato a espressioni numeriche sicure.

        Restituisce un risultato simbolico e numerico quando possibile.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression è obbligatoria")

        allowed = set("0123456789+-*/().= ")
        if not all(ch in allowed for ch in expression):
            raise ValueError("Espressione non supportata dal solver sicuro")

        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {
                "expression": expression,
                "result": result,
                "type": "numeric",
            }
        except Exception as e:
            return {
                "expression": expression,
                "result": None,
                "type": "error",
                "error": str(e),
            }

    def explain_causal(self, cause, observed_effect):
        """Spiega una relazione causale usando il world model.

        Returns:
            dict con 'predicted', 'matches', 'alternative_causes'.
        """
        if self.world_model is None:
            raise RuntimeError("WorldModel non disponibile")
        predicted = self.world_model.predict(cause)
        return {
            "cause": cause,
            "observed_effect": observed_effect,
            "predicted": predicted,
            "matches": observed_effect in predicted,
            "alternative_causes": [k for k, effects in self.world_model._causes.items() if observed_effect in effects and k != cause],
        }

    def analogical_reason(self, source_case, target_case):
        """Ragionamento per analogia: trasferisce soluzioni da un caso noto a un caso nuovo.

        Calcola la similarità tra source_case e target_case su feature condivise
        e restituisce una confidenza nel trasferimento.
        """
        if not isinstance(source_case, dict) or not isinstance(target_case, dict):
            raise TypeError("source_case e target_case devono essere dict")

        source_features = set(source_case.get("features", []))
        target_features = set(target_case.get("features", []))
        if not source_features or not target_features:
            return {
                "source": source_case,
                "target": target_case,
                "similarity": 0.0,
                "confidence": 0.0,
                "transferred_solution": None,
            }

        intersection = source_features & target_features
        union = source_features | target_features
        similarity = len(intersection) / len(union)
        confidence = similarity * source_case.get("confidence", 1.0)
        transferred = source_case.get("solution") if similarity >= 0.5 else None

        return {
            "source": source_case,
            "target": target_case,
            "similarity": similarity,
            "confidence": confidence,
            "transferred_solution": transferred,
            "shared_features": list(intersection),
            "source_only": list(source_features - target_features),
            "target_only": list(target_features - source_features),
        }

    def probabilistic_reason(self, hypotheses, evidence):
        """Ragionamento probabilistico: calcola probabilità a posteriori di ipotesi.

        hypotheses: lista di dict con 'name' e 'prior'.
        evidence: lista di dict con 'hypothesis', 'likelihood'.

        Restituisce probabilità a posteriori normalizzate.
        """
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError("hypotheses deve essere una lista non vuota")
        if not isinstance(evidence, list):
            raise ValueError("evidence deve essere una lista")

        prior_sum = sum(max(0.0, min(1.0, h.get("prior", 0.0))) for h in hypotheses)
        if prior_sum == 0:
            return {
                "hypotheses": [
                    {**h, "posterior": 1.0 / len(hypotheses)} for h in hypotheses
                ],
                "evidence_used": evidence,
                "normalization": "uniform",
            }

        posteriors = []
        for hypothesis in hypotheses:
            prior = max(0.0, min(1.0, hypothesis.get("prior", 0.0)))
            likelihood = 1.0
            for ev in evidence:
                if ev.get("hypothesis") == hypothesis.get("name"):
                    likelihood *= max(0.0, min(1.0, ev.get("likelihood", 1.0)))
            posteriors.append(prior * likelihood)

        total = sum(posteriors)
        if total == 0:
            normalized = [1.0 / len(hypotheses)] * len(hypotheses)
        else:
            normalized = [p / total for p in posteriors]

        return {
            "hypotheses": [
                {**h, "posterior": round(p, 6)} for h, p in zip(hypotheses, normalized)
            ],
            "evidence_used": evidence,
            "normalization": "bayesian",
        }

    def bayesian_update(self, prior, likelihood, evidence_weight=1.0):
        """Aggiornamento Bayesiano di una probabilità.

        Restituisce la probabilità a posteriori data una prior e una likelihood.
        """
        prior = max(0.0, min(1.0, float(prior)))
        likelihood = max(0.0, min(1.0, float(likelihood)))
        evidence_weight = max(0.0, min(1.0, float(evidence_weight)))

        if prior == 0.0:
            return 0.0
        if prior == 1.0:
            return 1.0

        weighted_likelihood = likelihood * evidence_weight + (1.0 - evidence_weight) * 0.5
        numerator = weighted_likelihood * prior
        denominator = numerator + (1.0 - weighted_likelihood) * (1.0 - prior)
        if denominator == 0:
            return prior
        return numerator / denominator

    def confidence_score(self, components):
        """Calcola un confidence score composito.

        components: lista di dict con 'weight' e 'score' (0..1).
        """
        if not components:
            return 0.0
        total_weight = sum(max(0.0, c.get("weight", 0.0)) for c in components)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(c.get("weight", 0.0) * max(0.0, min(1.0, c.get("score", 0.0))) for c in components)
        return weighted_sum / total_weight
