"""
Identity Resolver - Cognitive Core component for identity questions
====================================================================

This component bridges CORE/system/identity.py with the cognitive core.
It's responsible for:

1. Identifying when a query is about JARVIS identity
2. Routing identity questions to the identity system
3. Ensuring identity answers are consistent across the entire system
4. Distinguishing JARVIS from external providers in responses

Identity questions should NEVER be routed to external providers like Ollama.
This is the architectural enforcement point.
"""

from typing import Optional, Dict, Any
from CORE.system.identity import (
    is_jarvis_identity_query,
    answer_identity_question,
    get_canonical_identity,
    get_identity_context,
)


class IdentityResolver:
    """
    Resolves identity-related queries at the cognitive core level.
    
    This component sits between the router/coordinator and the external
    providers, ensuring that identity questions are answered by JARVIS
    itself and NOT delegated to external LLMs like Ollama.
    
    CRITICAL RULE:
        Identity questions MUST be answered by IdentityResolver.
        They must NEVER reach Ollama or external providers.
    """
    
    def __init__(self):
        """Initialize the identity resolver."""
        self.identity = get_canonical_identity()
        self._query_history: list = []
    
    def is_identity_question(self, query: str) -> bool:
        """
        Check if a query is asking about JARVIS identity.
        
        Args:
            query: The user's query
            
        Returns:
            bool: True if this is an identity question
        """
        return is_jarvis_identity_query(query)
    
    def resolve(self, query: str) -> Optional[str]:
        """
        Resolve an identity question.
        
        If the query is an identity question, this method returns the
        appropriate answer. If it's not an identity question, it returns None.
        
        Args:
            query: The user's query
            
        Returns:
            str: The answer if this is an identity question, None otherwise
            
        Raises:
            ValueError: If the query appears to be identity-related but
                       cannot be answered
        """
        if not self.is_identity_question(query):
            return None
        
        try:
            answer = answer_identity_question(query)
            self._record_identity_query(query, answer)
            return answer
        except Exception as e:
            # Log the error but provide a fallback response
            fallback_answer = self._get_fallback_identity_response()
            self._record_identity_query(query, fallback_answer, error=str(e))
            return fallback_answer
    
    def get_identity_context(self) -> Dict[str, Any]:
        """
        Get full identity context for system prompts or responses.
        
        Returns:
            Dict[str, Any]: Complete identity information
        """
        return get_identity_context()
    
    def get_creator(self) -> str:
        """
        Get the creator's name.
        
        Returns:
            str: Creator name
        """
        return self.identity.creator
    
    def get_name(self) -> str:
        """
        Get JARVIS's name.
        
        Returns:
            str: JARVIS name
        """
        return self.identity.name
    
    def get_version(self) -> str:
        """
        Get JARVIS version.
        
        Returns:
            str: Version string
        """
        return self.identity.version
    
    def inject_identity_in_prompt(self, prompt: str) -> str:
        """
        Inject identity context into a system prompt.
        
        This is useful for external calls to ensure the model
        knows it's communicating with JARVIS and not being asked
        directly about JARVIS.
        
        Args:
            prompt: The original system prompt
            
        Returns:
            str: Prompt with identity context injected
        """
        context = self.get_identity_context()
        identity_context = (
            f"\n\nSYSTEM CONTEXT:\n"
            f"You are assisting {context['name']}, created by {context['creator']}.\n"
            f"Do NOT confuse {context['name']} with any LLM provider.\n"
            f"{context['name']} is a {context['system_type']}.\n"
            f"If asked about {context['name']}'s creator or identity, refer to the creator information provided.\n"
        )
        return prompt + identity_context
    
    def verify_identity(self, claimed_name: str) -> bool:
        """
        Verify that a claimed identity matches the canonical identity.
        
        Args:
            claimed_name: The name being claimed
            
        Returns:
            bool: True if the claimed name matches JARVIS's identity
        """
        return self.identity.verify_identity(claimed_name)
    
    def _record_identity_query(self, query: str, answer: str, error: Optional[str] = None):
        """
        Record an identity query for logging/debugging.
        
        Args:
            query: The identity question
            answer: The answer provided
            error: Optional error message if one occurred
        """
        record = {
            "query": query,
            "answer": answer,
            "resolved": error is None,
            "error": error,
        }
        self._query_history.append(record)
    
    def _get_fallback_identity_response(self) -> str:
        """
        Get a fallback response if identity resolution fails.
        
        Returns:
            str: A safe fallback identity response
        """
        return (
            f"I am {self.identity.name}, created by {self.identity.creator}. "
            f"I am an integrated cognitive system, not an external model."
        )
    
    def get_query_history(self) -> list:
        """
        Get history of identity queries (for debugging/testing).
        
        Returns:
            list: List of identity query records
        """
        return self._query_history.copy()
    
    def clear_history(self):
        """Clear the identity query history."""
        self._query_history.clear()
