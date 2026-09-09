import os
import pickle
import re
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from INFRASTRUCTURE.audit.audit_system import AuditSystem
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from cognitive_core.vision.vision_system import VisionSystem
from neural_core.tokenizer import Tokenizer
from neural_core.model_interface import NeuralModelInterface
from neural_core.trainer import Trainer, NextTokenModel, load_model_from_weights
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam
from cognitive_core.identity.identity_resolver import IdentityResolver
from cognitive_core.coordinator import CognitiveCoordinator
from cognitive_core.memory.unified_memory import UnifiedMemory
from executive_system.tools.tool_system import ToolSystem

logger = logging.getLogger("JARVIS.ExecutiveRouter")


class ExecutiveRouter:
    """
    Router JARVIS: coordina i sistemi interni senza conoscere modelli esterni.
    """

    def __init__(self, permission_guard=None, config_path=None,
                 identity_resolver=None, neural_interface=None,
                 coordinator=None, agent_runtime=None, auto_approve_tools=True):
        self.permission_guard = permission_guard or PermissionGuard()
        self.config_path = config_path or Path(__file__).resolve().parents[1] / "config.yaml"
        self.config = self._load_config(self.config_path)

        audit_cfg = self.config.get("audit", {})
        self.audit_system = AuditSystem(
            enabled=audit_cfg.get("enabled", True),
            log_directory=audit_cfg.get("log_directory", "./audit_logs"),
            log_level=audit_cfg.get("log_level", "INFO"),
            max_file_size=int(audit_cfg.get("max_file_size", 10 * 1024 * 1024)),
            backup_count=int(audit_cfg.get("backup_count", 5)),
        )
        self.security_layer = SecurityLayer(self.permission_guard, audit_system=self.audit_system)

        vision_cfg = self.config.get("vision", {})
        self.vision_system = VisionSystem(config=vision_cfg) if vision_cfg.get("enabled", False) else None

        model_dir = Path(__file__).resolve().parents[1] / "neural_core" / "data"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = model_dir / "tokenizer.pkl"
        model_path = model_dir / "model.pkl"

        self.tokenizer = self._load_tokenizer(tokenizer_path)
        model = self._load_model(model_path)

        self.neural_interface = neural_interface or NeuralModelInterface(
            model=model,
            tokenizer=self.tokenizer,
        )

        self.memory = UnifiedMemory()
        self.tool_system = ToolSystem(security_layer=self.security_layer, vision_system=self.vision_system)
        self.identity_resolver = identity_resolver or IdentityResolver()
        self.coordinator = coordinator or CognitiveCoordinator(
            reasoner=None,
            memory=self.memory,
            neural_interface=self.neural_interface,
            identity_resolver=self.identity_resolver,
            tool_system=self.tool_system,
            security_layer=self.security_layer,
            vision_system=self.vision_system,
            auto_approve_tools=auto_approve_tools,
        )
        self.agent_runtime = agent_runtime
        self.auto_approve_tools = bool(auto_approve_tools)

    def _load_config(self, path):
        config = {"router": {"primary_system": "neural-core"}}
        if not os.path.exists(path):
            return config
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except Exception:
            loaded = {}
        if not isinstance(loaded, dict):
            return config
        for section in ("router", "vision", "audit"):
            if section in loaded and isinstance(loaded[section], dict):
                config[section] = {key: value for key, value in loaded[section].items() if value is not None}
        return config

    def _load_tokenizer(self, path):
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                return Tokenizer.load(str(json_path))
            except Exception:
                pass

        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

        tokenizer = Tokenizer()
        corpus_path = Path(__file__).resolve().parents[1] / "corpus_addestramento.txt"
        if corpus_path.exists():
            text = corpus_path.read_text(encoding="utf-8")
            tokenizer.fit(text)
        return tokenizer

    def _load_model(self, path):
        """
        Carica il modello addestrato dal Neural Core.

        Supporta sia `NextTokenModel` sia `MicroTransformer` e sia il formato
        corrente sia quello storico dei pesi. Se il caricamento non è possibile
        viene creato un modello nuovo coerente con il vocabolario corrente.
        """
        vocab_size = len(self.tokenizer.vocabulary)

        if path.exists():
            try:
                return load_model_from_weights(str(path), vocab_size=vocab_size)
            except Exception:
                pass

        return NextTokenModel(vocab_size=vocab_size, embedding_dim=64)

    def normalize_prompt(self, prompt):
        if prompt is None:
            raise ValueError("prompt non può essere None")
        text = str(prompt).strip()
        if not text:
            raise ValueError("prompt non può essere vuoto")
        return text

    def _internal_response(self, prompt):
        result = self.neural_interface.generate(prompt)
        return result.get("text", str(result))

    def _decide_route(self, prompt):
        if isinstance(prompt, dict):
            return "local"
        lowered = prompt.lower()
        if self.identity_resolver.is_identity_question(prompt):
            return "identity"
        if re.search(r"(ricorda|remember|memoria|memory|ricordi)", lowered):
            return "memory"
        if re.search(r"(conoscenza|knowledge|sapere|informazione)", lowered):
            return "knowledge"
        if re.search(
            r"(analizza|piano|strategia|progetto|multi|integrazione|architettura|"
            r"analisi|decisione|ottimizza|ragiona|reason)",
            lowered,
        ):
            return "reasoning"
        return "local"

    def generate(self, prompt, model=None):
        if prompt is None:
            raise ValueError("prompt non può essere None")

        is_vision = isinstance(prompt, dict) and prompt.get("type") == "vision"
        decision = self._decide_route(prompt)

        if is_vision:
            try:
                result = self.coordinator.process(prompt, session_id=self.audit_system.get_session_id())
                if isinstance(result, dict):
                    response = result.get("response", str(result))
                else:
                    response = str(result)
            except Exception as e:
                response = f"Errore elaborazione input visivo: {e}"
            return {
                "route": "vision",
                "provider": "internal",
                "model": "vision-system",
                "response": response,
                "fallback": False,
                "internal": True,
                "normalized_prompt": str(prompt),
            }

        if decision == "identity":
            identity_answer = self.identity_resolver.resolve(prompt)
            return {
                "route": "identity",
                "provider": "jarvis",
                "model": "jarvis-identity",
                "response": identity_answer,
                "fallback": False,
                "is_identity_question": True,
            }

        if self.agent_runtime is not None:
            try:
                loop_result = self._run_through_agent_runtime(prompt)
                return {
                    "route": decision,
                    "provider": "internal",
                    "model": "neural-core",
                    "response": loop_result.response or loop_result.final_state.get("working_memory", {}).get("last_verified_result", {}).get("delegate_result", {}).get("response", "ok") if isinstance(loop_result.final_state, dict) else "ok",
                    "fallback": False,
                    "internal": True,
                    "normalized_prompt": prompt,
                    "agent_runtime": {
                        "status": loop_result.status,
                        "iterations": loop_result.iterations,
                        "termination_reason": loop_result.termination_reason,
                    },
                }
            except Exception as e:
                logger.warning("AgentRuntime path failed, falling back to coordinator: %s", e)

        try:
            result = self.coordinator.process(prompt, session_id=self.audit_system.get_session_id())
            if isinstance(result, dict):
                response = result.get("response", str(result))
            else:
                response = str(result)
        except Exception as e:
            response = self._internal_response(prompt)

        return {
            "route": decision,
            "provider": "internal",
            "model": "neural-core",
            "response": response,
            "fallback": False,
            "internal": True,
            "normalized_prompt": prompt,
        }

    def _run_through_agent_runtime(self, prompt: str) -> Any:
        from cognitive_core.runtime import AgentRuntime
        if self.agent_runtime is None:
            raise RuntimeError("AgentRuntime not configured")
        task_id = self.agent_runtime.create_task(
            goal=prompt,
            context={"source": "ExecutiveRouter"},
        )
        return self.agent_runtime.run(
            task_id,
            action_producer=self._make_action_producer(task_id),
        )

    def _make_action_producer(self, task_id: str) -> Callable[[Any, Dict[str, Any]], Optional[Dict[str, Any]]]:
        coordinator = self.coordinator
        tool_system = self.tool_system
        audit = self.audit_system
        _producer_called = False

        def producer(state: AgentState, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            nonlocal _producer_called
            if coordinator is None:
                return None
            if _producer_called:
                return None
            _producer_called = True
            try:
                result = coordinator.process(state.goal, session_id=task_id)
                if isinstance(result, dict):
                    response = result.get("response", str(result))
                else:
                    response = str(result)
                return {
                    "name": "coordinator.process",
                    "capability": "read",
                    "args": {"goal": state.goal},
                    "result": {
                        "status": "delegated",
                        "delegate_result": result,
                        "response": response,
                    },
                }
            except Exception as exc:
                return {
                    "name": "coordinator.error",
                    "capability": "read",
                    "args": {"goal": state.goal},
                    "result": {"error": str(exc), "verified": False},
                }
        return producer

    def route(self, prompt):
        if prompt is None:
            raise ValueError("prompt non può essere None")
        decision = self._decide_route(prompt)
        if decision == "identity":
            return {"route": "identity", "provider": "jarvis", "model": "jarvis-identity", "is_identity_question": True}
        return {"route": decision, "provider": "internal", "model": "neural-core"}

    def execute_generated_code(self, code, approved=False):
        self.permission_guard.require_user_approval(approved)
        if code is None:
            raise ValueError("Il codice generato non può essere None")
        return str(code)

    def route_and_execute(self, prompt, code=None, approved=False):
        route = self.route(prompt)
        if code is not None:
            execution_output = self.execute_generated_code(code, approved=approved)
            route["execution_output"] = execution_output
        return route

    def get_identity_resolver(self):
        return self.identity_resolver

    def get_neural_interface(self):
        return self.neural_interface

    def run_self_improvement(self, trigger: str = "manual", benchmark_data=None, approved: bool = False) -> dict:
        if self.coordinator is None:
            return {"status": "disabled", "reason": "coordinator_not_configured"}
        return self.coordinator.run_self_improvement_cycle(trigger=trigger, benchmark_data=benchmark_data, approved=approved)
