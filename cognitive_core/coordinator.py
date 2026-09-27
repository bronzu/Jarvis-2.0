import logging
import re
import threading
from typing import Any, Dict, Optional
from cognitive_core.identity.identity_resolver import IdentityResolver
from cognitive_core.reasoning.reasoner import Reasoner
from cognitive_core.memory.episodic_memory import EpisodicMemory
from cognitive_core.perception import PerceptEvent, PerceptionLayer
from cognitive_core.understanding import Understanding
from cognitive_core.planner import Planner
from neural_core.model_interface import NeuralModelInterface


logger = logging.getLogger("JARVIS.CognitiveCoordinator")


class CognitiveCoordinator:
    """
    Coordina tutti i sistemi cognitivi di JARVIS.

    Supporta due modalità per retrocompatibilità:
    1. Vecchia modalità (neural_model): usa solo Neural Core, Reasoner, EpisodicMemory
    2. Nuova modalità (neural_interface + componenti completi): usa memoria, tool, reasoning, RAG, security
    """

    def __init__(self, reasoner=None, episodic_memory=None, neural_model=None,
                 identity_resolver=None, neural_interface=None, memory=None,
                 tool_system=None, security_layer=None, vision_system=None,
                 goal_manager=None, observation_feedback=None,
                 understanding=None, planner=None, improvement_cycle=None,
                 perception_layer=None, auto_approve_tools=True):
        self.reasoner = reasoner or Reasoner()
        self.memory = memory
        if memory is not None and episodic_memory is None:
            self.episodic_memory = memory.episodic
        else:
            self.episodic_memory = episodic_memory or EpisodicMemory()
        self.neural_model = neural_model
        self.identity_resolver = identity_resolver or IdentityResolver()
        self.neural_interface = neural_interface
        self.tool_system = tool_system
        self.security_layer = security_layer
        self.vision_system = vision_system
        self.goal_manager = goal_manager
        self.observation_feedback = observation_feedback
        self.understanding = understanding or Understanding()
        self.perception_layer = perception_layer or PerceptionLayer()
        self.planner = planner or Planner(
            reasoner=self.reasoner,
            memory=self.memory,
            tool_selector=getattr(self, 'tool_selector', None),
            autonomy_manager=getattr(self, 'autonomy_manager', None),
            permission_guard=getattr(self, 'permission_guard', None),
        )
        self.improvement_cycle = improvement_cycle
        self._errors = []
        self._neural_priority = True
        self._model_lock = threading.RLock()
        self.auto_approve_tools = bool(auto_approve_tools)

    def process(self, problem, session_id=None):
        if problem is None:
            raise ValueError("Il problema non può essere None")

        if isinstance(problem, PerceptEvent):
            percept_event = problem
        else:
            percept_event = self.perception_layer.process(problem)

        if percept_event.input_type == "vision":
            return self._process_vision_from_percept(percept_event, session_id=session_id)

        normalized = percept_event.normalized_text
        identity_answer = self.identity_resolver.resolve(normalized)
        if identity_answer is not None:
            self.episodic_memory.record(
                percept_event.raw_input,
                {
                    "result": identity_answer,
                    "source": "identity",
                    "is_identity_question": True,
                }
            )
            return identity_answer

        if getattr(self, 'neural_model', None) is not None and self.neural_interface is None:
            return self._process_legacy(percept_event, session_id=session_id)

        return self._process_full(percept_event, session_id=session_id)

    def _process_legacy(self, percept_event, session_id=None):
        result = None
        used_source = None
        text_query = percept_event.normalized_text
        if percept_event.input_type == "structured":
            text_query = percept_event.raw_input

        if self._neural_priority and self.neural_model is not None:
            try:
                result = self._invoke_neural_core(text_query)
                used_source = "neural_core"
            except Exception as exc:
                logger.warning("Neural Core fallback: %s", exc)
                result = None

        if result is None:
            try:
                result = self.reasoner.reason(text_query)
                used_source = "reasoner"
            except Exception as exc:
                logger.warning("Reasoner fallback: %s", exc)
                result = None

        if result is None:
            result = self._procedural_fallback(text_query)
            used_source = "procedural_fallback"

        self.episodic_memory.record(
            percept_event.raw_input,
            {
                "result": result,
                "source": used_source,
                "neural_model_used": used_source == "neural_core",
                "is_identity_question": False,
            }
        )

        return result

    def _process_full(self, percept_event, session_id=None):
        used_source = None
        context_parts = []
        tool_result = None
        reasoning_result = None
        goal_id = None

        try:
            if self.security_layer is not None:
                self.security_layer.log_permission_event(
                    "USER_INPUT", component="CognitiveCoordinator", action="process", status="started", session_id=session_id
                )
        except Exception as exc:
            self._errors.append(f"security_layer log failed: {exc}")

        text_query = percept_event.normalized_text
        if percept_event.input_type == "structured":
            text_query = percept_event.raw_input
        understanding = None
        try:
            understanding = self.understanding.understand(text_query)
            if understanding and understanding.get("confidence", 0) > 0:
                context_parts.append(f"Intent: {understanding['intent']}")
                if understanding.get("entities"):
                    context_parts.append("Entità: " + ", ".join(e["value"] for e in understanding["entities"][:5]))
                if understanding.get("goal"):
                    context_parts.append(f"Goal: {understanding['goal']}")
                if understanding.get("missing_information"):
                    context_parts.append("Informazioni mancanti: " + ", ".join(understanding["missing_information"][:3]))
        except Exception as exc:
            logger.warning("Understanding failed: %s", exc)

        if understanding and understanding.get("intent") == Understanding.INTENT_COMMAND:
            if self.goal_manager is not None:
                try:
                    goal_id = self.goal_manager.add(
                        understanding.get("goal") or understanding.get("raw", text_query),
                        steps=None,
                        metadata={"source": "understanding", "session_id": session_id, "intent": understanding["intent"]},
                    )
                    self.goal_manager.start(goal_id)
                    context_parts.append(f"Obiettivo attivo: {understanding.get('goal') or understanding.get('raw')}")
                except Exception as exc:
                    self._errors.append(f"goal_manager failed: {exc}")
                    goal_id = None

        try:
            if self.memory is not None:
                memory_hits = self.memory.search_semantic(text_query)
                if memory_hits:
                    context_parts.append("Contesto memoria precedente: " + "; ".join(
                        hit["payload"].get("text", "") for hit in memory_hits[:3]
                    ))
        except Exception as exc:
            self._errors.append(f"memory search failed: {exc}")

        tool_result = self._try_execute_tool(text_query, session_id=session_id)
        if tool_result is not None:
            context_parts.append(f"Risultato tool: {tool_result}")
            used_source = "tool"

        try:
            if self.memory is not None:
                related = self.memory.related(text_query)
                if related:
                    context_parts.append("Conoscenze correlate: " + "; ".join(
                        f"{item['relation']} {item['target']}" for item in related[:5]
                    ))
        except Exception as exc:
            self._errors.append(f"memory related failed: {exc}")

        if self._is_reasoning_request(text_query):
            try:
                reasoning_result = self._run_reasoner(text_query)
                if reasoning_result and reasoning_result.get("proved"):
                    context_parts.append(f"Ragionamento: {reasoning_result['goal']} è dimostrato")
                    if used_source is None:
                        used_source = "reasoner"
            except Exception as exc:
                logger.warning("Reasoning request failed: %s", exc)
                reasoning_result = None

        enriched_prompt = str(text_query)
        if context_parts:
            enriched_prompt = enriched_prompt + "\n\n" + "\n".join(context_parts)

        if self._neural_priority and self.neural_interface is not None:
            try:
                with self._model_lock:
                    neural_result = self.neural_interface.generate(enriched_prompt)
                response = neural_result.get("text", str(neural_result))
                if used_source is None:
                    used_source = "neural_core"
            except Exception as exc:
                logger.warning("Neural interface generation failed: %s", exc)
                if reasoning_result is not None and used_source == "reasoner":
                    response = f"Ragionamento completato: {reasoning_result.get('goal', text_query)}"
                elif tool_result is not None:
                    response = f"Tool eseguito: {tool_result}"
                else:
                    response = self._procedural_fallback(enriched_prompt).get("default_response", "")
                    if used_source is None:
                        used_source = "procedural_fallback"
        else:
            if reasoning_result is not None:
                response = f"Ragionamento completato: {reasoning_result.get('goal', text_query)}"
                if used_source is None:
                    used_source = "reasoner"
            elif tool_result is not None:
                response = f"Tool eseguito: {tool_result}"
            else:
                response = self._procedural_fallback(enriched_prompt).get("default_response", "")
                if used_source is None:
                    used_source = "procedural_fallback"

        if tool_result is not None:
            response = f"{response}\n\n[Risultato tool: {tool_result}]"

        outcome = {
            "result": response,
            "source": used_source,
            "tool_used": tool_result is not None,
            "reasoning_used": reasoning_result is not None,
        }
        if reasoning_result is not None:
            outcome["reasoning_result"] = reasoning_result

        if goal_id is not None and self.goal_manager is not None:
            try:
                step = self.goal_manager.next_pending_step(goal_id)
                if step:
                    self.goal_manager.complete_step(goal_id, step["id"], outcome)
                self.goal_manager.complete(goal_id)
            except Exception as exc:
                logger.warning("Goal manager completion failed: %s", exc)

        if self.observation_feedback is not None:
            try:
                self.observation_feedback.record(
                    action=percept_event.raw_input,
                    result=response,
                    outcome="success" if used_source != "procedural_fallback" else "failure",
                    session_id=session_id,
                    metadata={"source": used_source, "goal_id": goal_id},
                )
            except Exception as exc:
                logger.warning("Observation feedback recording failed: %s", exc)

        self.episodic_memory.record(percept_event.raw_input, outcome)

        if self.memory is not None:
            try:
                self.memory.remember_working({
                    "input": percept_event.raw_input,
                    "response": response,
                    "source": used_source,
                })
            except Exception as exc:
                logger.warning("Working memory update failed: %s", exc)

        result = {
            "response": response,
            "source": used_source,
            "tool_result": tool_result,
            "reasoning_result": reasoning_result,
        }
        if reasoning_result is not None:
            result.update(reasoning_result)
        if used_source == "procedural_fallback":
            result["fallback"] = True

        try:
            if self.security_layer is not None:
                self.security_layer.log_permission_event(
                    "MODEL_RESPONSE", component="CognitiveCoordinator", action="process", status="completed", session_id=session_id
                )
        except Exception as exc:
            logger.warning("Security logging failed: %s", exc)

        return result

    def _process_vision_from_percept(self, percept_event, session_id=None):
        if self.vision_system is None:
            return {
                "response": "Vision system non configurato.",
                "source": "vision",
                "vision_result": None,
            }
        image_path = percept_event.metadata.get("image_path")
        if not image_path:
            return {
                "response": "Percorso immagine mancante.",
                "source": "vision",
                "vision_result": None,
            }
        try:
            if self.security_layer is not None:
                self.security_layer.check_filesystem_access(image_path, "read", session_id=session_id)
        except Exception as exc:
            logger.warning("Vision filesystem access check failed: %s", exc)
        try:
            if self.security_layer is not None:
                self.security_layer.log_permission_event(
                    "VISION_ANALYSIS_START", component="CognitiveCoordinator", action="process_vision", status="started", session_id=session_id
                )
            vision_result = self.vision_system.analyze_image(image_path)
            if self.security_layer is not None:
                self.security_layer.log_permission_event(
                    "VISION_ANALYSIS_END", component="CognitiveCoordinator", action="process_vision", status="completed", session_id=session_id
                )
        except Exception as e:
            if self.security_layer is not None:
                self.security_layer.log_permission_event(
                    "VISION_ERROR", component="CognitiveCoordinator", action="process_vision", status="error", error=str(e), session_id=session_id
                )
            return {
                "response": f"Errore analisi immagine: {e}",
                "source": "vision",
                "vision_result": None,
            }
        response = vision_result.get("description", "Analisi immagine completata.")
        return {
            "response": response,
            "source": "vision",
            "vision_result": vision_result,
        }

    def _invoke_neural_core(self, problem):
        with self._model_lock:
            if self.neural_model is None:
                raise ValueError("Neural Core non disponibile")
            input_vector = self._problem_to_vector(problem)
            output = self.neural_model.forward(input_vector)
        result = self._decode_neural_output(output)
        return result

    def _problem_to_vector(self, problem):
        if isinstance(problem, str):
            hash_val = hash(problem)
            return [float(hash_val % 256) / 256.0 for _ in range(64)]
        elif isinstance(problem, (list, tuple)):
            return [float(x) for x in problem]
        else:
            return [0.5] * 64

    def _decode_neural_output(self, output):
        if isinstance(output, (float, int)):
            return {"neural_score": output, "decision": "accepted" if output > 0.5 else "rejected"}
        elif isinstance(output, (list, tuple)):
            return {"neural_output": list(output), "magnitude": sum(float(x) ** 2 for x in output) ** 0.5}
        else:
            return {"raw_output": str(output)}

    def _procedural_fallback(self, problem):
        return {
            "fallback": True,
            "problem_hash": hash(problem),
            "default_response": "Utilizzo del fallback procedurale",
        }

    def _try_execute_tool(self, problem, session_id=None):
        if self.tool_system is None:
            return None
        capability = self._detect_tool_capability(problem)
        if capability is None:
            return None
        try:
            if capability == "filesystem":
                return self._execute_filesystem_tool(problem, session_id=session_id)
            elif capability == "web":
                return self._execute_web_tool(problem, session_id=session_id)
            elif capability == "internet":
                return self._execute_internet_tool(problem, session_id=session_id)
            elif capability == "terminal":
                return self._execute_terminal_tool(problem, session_id=session_id)
            elif capability == "python":
                return self._execute_python_tool(problem, session_id=session_id)
        except Exception as e:
            return f"Errore esecuzione tool {capability}: {e}"
        return None

    def _detect_tool_capability(self, problem):
        if isinstance(problem, dict):
            return problem.get("capability")
        lowered = str(problem).lower()
        if any(k in lowered for k in ["cerca", "search", "web", "naviga", "google", "duckduckgo"]):
            return "web"
        if any(k in lowered for k in ["http://", "https://", "url", "link", "pagina"]):
            return "internet"
        if any(k in lowered for k in ["file", "cartella", "directory", "leggi", "scrivi", "crea file", "crea il file", "lista file", "list", "lista"]):
            return "filesystem"
        if any(k in lowered for k in ["esegui", "run", "terminal", "comando", "shell"]):
            return "terminal"
        if any(k in lowered for k in ["python", "codice", "script", "esegui codice"]):
            return "python"
        return None

    def _execute_filesystem_tool(self, problem, session_id=None):
        lowered = str(problem).lower()
        if any(k in lowered for k in ["scrivi", "write", "crea file", "crea il file"]):
            path = self._extract_path(problem)
            content = self._extract_content(problem)
            return self.tool_system.execute("filesystem", operation="write", path=path, content=content, approved=self.auto_approve_tools, session_id=session_id)
        if any(k in lowered for k in ["leggi", "read", "contenuto"]):
            path = self._extract_path(problem)
            return self.tool_system.execute("filesystem", operation="read", path=path, approved=self.auto_approve_tools, session_id=session_id)
        if any(k in lowered for k in ["lista", "list", "elenco"]):
            path = self._extract_path(problem) or "."
            return self.tool_system.execute("filesystem", operation="list", path=path, approved=self.auto_approve_tools, session_id=session_id)
        return None

    def _execute_web_tool(self, problem, session_id=None):
        query = re.sub(r"(cerca|search|web|naviga|google|duckduckgo)", "", str(problem), flags=re.IGNORECASE).strip()
        return self.tool_system.execute("web", query=query, approved=self.auto_approve_tools, session_id=session_id)

    def _execute_internet_tool(self, problem, session_id=None):
        url = self._extract_url(str(problem))
        if not url:
            return None
        return self.tool_system.execute("internet", url=url, extract_text=True, approved=self.auto_approve_tools, session_id=session_id)

    def _execute_terminal_tool(self, problem, session_id=None):
        command = re.sub(r"(esegui|run|terminal|comando|shell)", "", str(problem), flags=re.IGNORECASE).strip()
        if not command:
            return None
        return self.tool_system.execute("terminal", command=command, approved=self.auto_approve_tools, session_id=session_id)

    def _execute_python_tool(self, problem, session_id=None):
        code = re.sub(r"(python|codice|script|esegui codice)", "", str(problem), flags=re.IGNORECASE).strip()
        if not code:
            return None
        return self.tool_system.execute("python", code=code, approved=self.auto_approve_tools, session_id=session_id)

    def _extract_path(self, text):
        file_match = re.search(r"file\s+([^\s]+)", text, re.IGNORECASE)
        dir_match = re.search(r"(?:in|nella|sulla|sullo|sul|nello|nella|del|dello|della)\s+(?:cartella|directory|folder|path|percorso)\s+([^\s]+)", text, re.IGNORECASE)
        if not dir_match:
            dir_match = re.search(r"(?:in|nella|sulla|sullo|sul|nello|nella|del|dello|della)\s+([^\s]+)", text, re.IGNORECASE)
        if file_match and dir_match:
            return f"{dir_match.group(1)}/{file_match.group(1)}"
        if file_match:
            return file_match.group(1)
        if dir_match:
            return dir_match.group(1)
        return "."

    def _extract_content(self, text):
        match = re.search(r"contenuto\s+([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("'\"")
        match = re.search(r"con\s+([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("'\"")
        return ""

    def _extract_url(self, text):
        match = re.search(r"(https?://[^\s]+)", text)
        if match:
            return match.group(1)
        return None

    def _is_reasoning_request(self, problem):
        if isinstance(problem, dict):
            return True
        lowered = str(problem).lower()
        keywords = ["analizza", "ragiona", "dimostra", "prova che", "reason", "analisi", "perché", "perche", "causa", "motivo"]
        return any(k in lowered for k in keywords)

    def _run_reasoner(self, problem):
        if not isinstance(problem, dict):
            facts = [problem]
            rules = []
            goal = problem
            problem_dict = {"facts": facts, "rules": rules, "goal": goal}
        else:
            problem_dict = problem
        return self.reasoner.reason(problem_dict)

    def set_neural_priority(self, enabled):
        self._neural_priority = bool(enabled)

    def is_neural_priority_enabled(self):
        return self._neural_priority

    def get_identity_resolver(self):
        return self.identity_resolver

    def run_self_improvement_cycle(self, trigger: str = "manual", benchmark_data: Optional[Dict[str, Any]] = None, approved: bool = False) -> Dict[str, Any]:
        if self.improvement_cycle is None:
            return {"status": "disabled", "reason": "improvement_cycle_not_configured"}
        try:
            self.security_layer.log_permission_event(
                "SELF_IMPROVEMENT_CYCLE",
                component="CognitiveCoordinator",
                action="run_self_improvement_cycle",
                status="started",
                metadata={"trigger": trigger},
            )
        except Exception:
            pass
        result = self.improvement_cycle.run_cycle(trigger=trigger, benchmark_data=benchmark_data, approved=approved)
        try:
            self.security_layer.log_permission_event(
                "SELF_IMPROVEMENT_CYCLE",
                component="CognitiveCoordinator",
                action="run_self_improvement_cycle",
                status="completed",
                metadata={"trigger": trigger, "result": result.get("status")},
            )
        except Exception:
            pass
        return result
