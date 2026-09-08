"""
Training strategy derivation from weaknesses for JARVIS 2.0.

Converts weakness detections and benchmark failures into concrete
training strategies that can be executed by the existing Trainer.
"""

from typing import Any, Dict, List, Optional


class TrainingStrategy:
    """Concrete training strategy derived from a weakness."""

    def __init__(self, strategy_id: str, weakness_type: str, parameters: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.weakness_type = weakness_type
        self.parameters = dict(parameters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "weakness_type": self.weakness_type,
            "parameters": dict(self.parameters),
        }


class TrainingStrategyBuilder:
    """Trasforma weakness e failure in strategie di training concrete."""

    WEAKNESS_STRATEGIES = {
        "reasoning_failure": {
            "dataset": "reasoning_train",
            "objective": "next_token",
            "epochs": 3,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "context_size": 128,
            "description": "Aumenta esposizione a esempi di reasoning multi-step",
        },
        "memory_failure": {
            "dataset": "memory_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0003,
            "batch_size": 32,
            "context_size": 256,
            "description": "Rafforza memorizzazione e recupero",
        },
        "instruction_following_failure": {
            "dataset": "instruction_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "context_size": 128,
            "description": "Migliora comprensione e aderenza a istruzioni",
        },
        "planning_failure": {
            "dataset": "planning_train",
            "objective": "next_token",
            "epochs": 3,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "context_size": 256,
            "description": "Migliora decomposizione e ordinamento passi",
        },
        "tool_selection_failure": {
            "dataset": "tool_selection_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "context_size": 128,
            "description": "Migliora selezione e utilizzo strumenti",
        },
        "tool_execution_failure": {
            "dataset": "tool_execution_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "context_size": 128,
            "description": "Migliora esecuzione e gestione errori tool",
        },
        "error_recovery_failure": {
            "dataset": "error_recovery_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "context_size": 128,
            "description": "Migliora recupero da errori",
        },
        "safety_failure": {
            "dataset": "safety_train",
            "objective": "next_token",
            "epochs": 1,
            "learning_rate": 0.0002,
            "batch_size": 16,
            "context_size": 128,
            "description": "Rafforza conformità sicurezza",
        },
        "performance_regression": {
            "dataset": "performance_train",
            "objective": "next_token",
            "epochs": 1,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "context_size": 128,
            "description": "Ottimizza performance",
        },
        "timeout_failure": {
            "dataset": "performance_train",
            "objective": "next_token",
            "epochs": 1,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "context_size": 64,
            "description": "Riduce latenza",
        },
        "model_problem": {
            "dataset": "general_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "context_size": 128,
            "description": "Migliora comportamento generale modello",
        },
        "prompt_problem": {
            "dataset": "instruction_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "context_size": 128,
            "description": "Migliora comprensione prompt",
        },
        "planner_problem": {
            "dataset": "planning_train",
            "objective": "next_token",
            "epochs": 3,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "context_size": 256,
            "description": "Migliora pianificazione",
        },
        "tool_use_problem": {
            "dataset": "tool_selection_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "context_size": 128,
            "description": "Migliora utilizzo tool",
        },
        "memory_problem": {
            "dataset": "memory_train",
            "objective": "next_token",
            "epochs": 2,
            "learning_rate": 0.0003,
            "batch_size": 32,
            "context_size": 256,
            "description": "Migliora memoria",
        },
        "infrastructure_problem": {
            "dataset": "general_train",
            "objective": "next_token",
            "epochs": 1,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "context_size": 128,
            "description": "Migliora stabilità infrastruttura",
        },
    }

    def build_from_weakness(self, weakness: Dict[str, Any], candidate_id: str, parent_version: str) -> TrainingStrategy:
        weakness_type = weakness.get("type", "model_problem")
        base = dict(self.WEAKNESS_STRATEGIES.get(weakness_type, self.WEAKNESS_STRATEGIES["model_problem"]))
        parameters = {
            **base,
            "candidate_id": candidate_id,
            "parent_version": parent_version,
            "weakness_type": weakness_type,
            "weakness_metric": weakness.get("metric"),
            "weakness_severity": weakness.get("severity"),
            "weakness_value": weakness.get("value"),
            "weakness_threshold": weakness.get("threshold"),
        }
        strategy_id = f"strategy-{weakness_type}-{candidate_id[:8]}"
        return TrainingStrategy(strategy_id=strategy_id, weakness_type=weakness_type, parameters=parameters)

    def build_from_failure(self, failure: Dict[str, Any], candidate_id: str, parent_version: str) -> TrainingStrategy:
        failure_type = failure.get("failure_type", "unexpected_error")
        weakness_type = failure_type
        base = dict(self.WEAKNESS_STRATEGIES.get(weakness_type, self.WEAKNESS_STRATEGIES["model_problem"]))
        parameters = {
            **base,
            "candidate_id": candidate_id,
            "parent_version": parent_version,
            "failure_type": failure_type,
            "failure_category": failure.get("category"),
            "failure_error": failure.get("error"),
            "weakness_type": weakness_type,
        }
        strategy_id = f"strategy-{failure_type}-{candidate_id[:8]}"
        return TrainingStrategy(strategy_id=strategy_id, weakness_type=weakness_type, parameters=parameters)

    def available_strategies(self) -> List[str]:
        return list(self.WEAKNESS_STRATEGIES.keys())
