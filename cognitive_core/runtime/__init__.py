"""JARVIS Agent Runtime package.

Provides:
- Task lifecycle (TaskLifecycle, TaskStatus, transitions)
- Agent state (AgentState — serializable, checkpointable)
- Agent runtime (AgentRuntime — orchestrates the cognitive loop)
- Recovery policy (RecoveryPolicy — retry/reparam/different_tool/different_plan/user_escalation)

Architectural role
-------------------
The Agent Runtime is the *primary* orchestrator of JARVIS. It does NOT
replace the existing CognitiveCoordinator: it sits on top of it and uses
it as a delegate for backward-compatible one-shot processing, while
introducing an explicit observe -> act -> verify -> replan loop with a
proper task lifecycle.

This is part of the JARVIS 2.0 -> 3.0 transformation, Phase 1 + Phase 3.
"""

from .task_lifecycle import TaskStatus, TaskLifecycle, LifecycleError
from .agent_state import AgentState, StateError
from .agent_runtime import AgentRuntime, AgentLoopResult
from .action import (
    ActionStatus,
    ActionRisk,
    ActionError,
    create_action,
    validate_action,
    ActionAuthorizer,
)
from .verification import (
    VerificationStatus,
    VerificationError,
    VerificationResult,
    DefaultVerifier,
    SideEffectVerifier,
)
from .recovery import RecoveryPolicy

__all__ = [
    "TaskStatus",
    "TaskLifecycle",
    "LifecycleError",
    "AgentState",
    "StateError",
    "AgentRuntime",
    "AgentLoopResult",
    "ActionStatus",
    "ActionRisk",
    "ActionError",
    "create_action",
    "validate_action",
    "ActionAuthorizer",
    "VerificationStatus",
    "VerificationError",
    "VerificationResult",
    "DefaultVerifier",
    "SideEffectVerifier",
    "RecoveryPolicy",
]
