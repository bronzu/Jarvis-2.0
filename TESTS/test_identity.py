"""
Test Identity System - CRITICAL TESTS FOR JARVIS IDENTITY
==========================================================

These tests ensure that:
1. JARVIS correctly identifies itself
2. JARVIS is never confused with external providers like Ollama
3. Identity questions are answered by the identity system, not external providers
4. The creator information is always correct
"""

import pytest
from CORE.system.identity import (
    JarvisIdentity,
    get_canonical_identity,
    get_identity_context,
    is_jarvis_identity_query,
    answer_identity_question,
)
from cognitive_core.identity.identity_resolver import IdentityResolver


# ==============================================================================
# TESTS: CORE IDENTITY SYSTEM (CORE/system/identity.py)
# ==============================================================================

class TestCoreIdentitySystem:
    """Test the core identity system in CORE/system/identity.py"""
    
    def test_jarvis_identity_has_correct_name(self):
        """JARVIS must have the correct name."""
        identity = get_canonical_identity()
        assert identity.name == "JARVIS"
        assert identity.name != "Ollama"
        assert identity.name != "LLM Model"
    
    def test_jarvis_identity_has_correct_creator(self):
        """JARVIS must have the correct creator - NOT Ollama."""
        identity = get_canonical_identity()
        assert identity.creator == "Riccardo Bronzuoli"
        assert identity.creator != "Ollama"
        assert identity.creator != "Open source project"
    
    def test_jarvis_has_correct_version(self):
        """JARVIS must have a version."""
        identity = get_canonical_identity()
        assert identity.version == "2.0.0"
        assert identity.version is not None
    
    def test_canonical_identity_is_singleton(self):
        """The canonical identity should be a singleton."""
        identity1 = get_canonical_identity()
        identity2 = get_canonical_identity()
        assert identity1 is identity2
    
    def test_identity_to_dict_conversion(self):
        """Identity must be convertible to dictionary."""
        identity = get_canonical_identity()
        identity_dict = identity.to_dict()
        
        assert isinstance(identity_dict, dict)
        assert identity_dict["name"] == "JARVIS"
        assert identity_dict["creator"] == "Riccardo Bronzuoli"
        assert identity_dict["version"] == "2.0.0"
        assert "neural_core" in identity_dict
        assert "cognitive_core" in identity_dict
        assert "executive_system" in identity_dict
    
    def test_identity_has_fundamental_principles(self):
        """JARVIS identity must include fundamental principles."""
        identity = get_canonical_identity()
        assert identity.fundamental_principles is not None
        assert len(identity.fundamental_principles) > 0
        
        # Check that principles distinguish JARVIS from Ollama
        principles_text = " ".join(identity.fundamental_principles).lower()
        assert "ollama" in principles_text or "external" in principles_text
    
    def test_identity_context_retrieval(self):
        """Identity context must be retrievable."""
        context = get_identity_context()
        
        assert context["name"] == "JARVIS"
        assert context["creator"] == "Riccardo Bronzuoli"
        assert context["version"] == "2.0.0"


# ==============================================================================
# TESTS: IDENTITY QUESTION DETECTION
# ==============================================================================

class TestIdentityQuestionDetection:
    """Test that identity questions are correctly identified."""
    
    def test_creator_questions_are_identified(self):
        """Creator questions must be identified as identity questions."""
        creator_questions = [
            "Chi ti ha creato?",
            "Who created you?",
            "Who is your creator?",
            "Who made you?",
            "Who built you?",
            "Chi è il tuo creatore?",
        ]
        
        for question in creator_questions:
            assert is_jarvis_identity_query(question), f"Failed to identify: {question}"
    
    def test_name_questions_are_identified(self):
        """Name questions must be identified as identity questions."""
        name_questions = [
            "Who are you?",
            "What is your name?",
            "What are you called?",
            "Qual è il tuo nome?",
            "Chi sei?",
        ]
        
        for question in name_questions:
            assert is_jarvis_identity_query(question), f"Failed to identify: {question}"
    
    def test_version_questions_are_identified(self):
        """Version questions must be identified as identity questions."""
        version_questions = [
            "What is your version?",
            "What version are you?",
            "Qual è la tua versione?",
        ]
        
        for question in version_questions:
            assert is_jarvis_identity_query(question), f"Failed to identify: {question}"
    
    def test_non_identity_questions_are_not_identified(self):
        """Non-identity questions must NOT be identified as identity questions."""
        non_identity_questions = [
            "What is the weather?",
            "Calculate 2 + 2",
            "Tell me a joke",
            "Help me with Python",
            "What's the capital of France?",
        ]
        
        for question in non_identity_questions:
            assert not is_jarvis_identity_query(question), f"Wrongly identified as identity: {question}"


# ==============================================================================
# TESTS: IDENTITY ANSWERS (CRITICAL - Must NOT return Ollama)
# ==============================================================================

class TestIdentityAnswers:
    """Test that identity questions are answered correctly."""
    
    def test_creator_question_returns_correct_creator(self):
        """CRITICAL: 'Who created you?' must return the correct creator, NOT Ollama."""
        question = "Who created you?"
        answer = answer_identity_question(question)
        
        assert "Riccardo Bronzuoli" in answer
        assert "Ollama" not in answer.lower()
        assert answer.lower() != "Ollama"
    
    def test_chi_ti_ha_creato_returns_correct_creator(self):
        """CRITICAL: Italian 'Chi ti ha creato?' must return correct creator."""
        question = "Chi ti ha creato?"
        answer = answer_identity_question(question)
        
        assert "Riccardo Bronzuoli" in answer
        assert "Ollama" not in answer.lower()
    
    def test_name_question_returns_jarvis(self):
        """'Who are you?' must return JARVIS identity, not external provider."""
        question = "Who are you?"
        answer = answer_identity_question(question)
        
        assert "JARVIS" in answer
        assert "Ollama" not in answer
    
    def test_version_question_returns_correct_version(self):
        """Version question must return 2.0.0."""
        question = "What is your version?"
        answer = answer_identity_question(question)
        
        assert "2.0.0" in answer
    
    def test_identity_questions_never_return_ollama(self):
        """CRITICAL: No identity answer should ever return 'Ollama' as the creator."""
        identity_questions = [
            "Chi ti ha creato?",
            "Who created you?",
            "Who is your creator?",
            "Who are you?",
            "What is your name?",
        ]
        
        for question in identity_questions:
            answer = answer_identity_question(question)
            assert "Ollama" not in answer, f"Answer contains 'Ollama': {answer}"
    
    def test_identity_answer_explains_jarvis_is_integrated_system(self):
        """Identity answers should explain that JARVIS is an integrated system."""
        question = "Who are you?"
        answer = answer_identity_question(question)
        
        # Should mention that JARVIS is an integrated system
        assert any(word in answer.lower() for word in [
            "integrated", "system", "cognitive", "architecture"
        ])


# ==============================================================================
# TESTS: IDENTITY RESOLVER (cognitive_core/identity/identity_resolver.py)
# ==============================================================================

class TestIdentityResolver:
    """Test the IdentityResolver component in cognitive_core."""
    
    def test_identity_resolver_initialization(self):
        """IdentityResolver must initialize without errors."""
        resolver = IdentityResolver()
        assert resolver is not None
        assert resolver.get_name() == "JARVIS"
    
    def test_resolver_identifies_identity_questions(self):
        """Resolver must identify identity questions."""
        resolver = IdentityResolver()
        assert resolver.is_identity_question("Who created you?")
        assert resolver.is_identity_question("Chi ti ha creato?")
        assert not resolver.is_identity_question("What's the weather?")
    
    def test_resolver_resolves_creator_question(self):
        """CRITICAL: Resolver must resolve creator question correctly."""
        resolver = IdentityResolver()
        answer = resolver.resolve("Who created you?")
        
        assert answer is not None
        assert "Riccardo Bronzuoli" in answer
        assert "Ollama" not in answer.lower()
    
    def test_resolver_resolves_identity_question(self):
        """Resolver must resolve 'Who are you?' correctly."""
        resolver = IdentityResolver()
        answer = resolver.resolve("Who are you?")
        
        assert answer is not None
        assert "JARVIS" in answer
    
    def test_resolver_returns_none_for_non_identity_questions(self):
        """Resolver must return None for non-identity questions."""
        resolver = IdentityResolver()
        answer = resolver.resolve("What's 2 + 2?")
        assert answer is None
    
    def test_resolver_provides_identity_context(self):
        """Resolver must provide complete identity context."""
        resolver = IdentityResolver()
        context = resolver.get_identity_context()
        
        assert context["name"] == "JARVIS"
        assert context["creator"] == "Riccardo Bronzuoli"
        assert context["version"] == "2.0.0"
    
    def test_resolver_can_verify_identity(self):
        """Resolver must be able to verify JARVIS's identity."""
        resolver = IdentityResolver()
        
        assert resolver.verify_identity("JARVIS")
        assert resolver.verify_identity("jarvis")
        assert not resolver.verify_identity("Ollama")
        assert not resolver.verify_identity("GPT")
    
    def test_resolver_records_query_history(self):
        """Resolver must record identity query history."""
        resolver = IdentityResolver()
        
        # Make some identity queries
        resolver.resolve("Who created you?")
        resolver.resolve("What is your name?")
        
        # Check history
        history = resolver.get_query_history()
        assert len(history) == 2
        assert history[0]["resolved"] is True
        assert history[1]["resolved"] is True
    
    def test_resolver_injects_identity_in_prompt(self):
        """Resolver must be able to inject identity context in prompts."""
        resolver = IdentityResolver()
        prompt = "Respond to the user's question."
        injected = resolver.inject_identity_in_prompt(prompt)
        
        assert prompt in injected
        assert "JARVIS" in injected or "SYSTEM CONTEXT" in injected
        assert "Riccardo Bronzuoli" in injected or "creator" in injected.lower()


# ==============================================================================
# TESTS: INTEGRATION - Ensuring JARVIS ≠ Ollama
# ==============================================================================

class TestJarvisDistinctFromOllama:
    """
    CRITICAL TESTS: Verify that JARVIS is always distinct from Ollama.
    
    This is a fundamental architectural requirement.
    """
    
    def test_jarvis_is_not_ollama(self):
        """JARVIS identity must be distinct from Ollama."""
        identity = get_canonical_identity()
        
        assert identity.name != "Ollama"
        assert identity.creator != "Ollama"
        assert identity.system_type != "Ollama"
    
    def test_creator_is_not_ollama_project(self):
        """Creator must never be Ollama or an open-source project."""
        identity = get_canonical_identity()
        
        assert identity.creator == "Riccardo Bronzuoli"
        assert "Ollama" not in identity.creator
        assert "open" not in identity.creator.lower()
    
    def test_identity_resolver_distinguishes_jarvis_from_ollama(self):
        """IdentityResolver must clearly distinguish JARVIS from Ollama."""
        resolver = IdentityResolver()
        
        answer = resolver.resolve("Who created you?")
        assert answer is not None
        
        # The answer should explain that JARVIS was created by a human
        assert "Riccardo Bronzuoli" in answer
        
        # The answer should NOT attribute JARVIS to Ollama
        assert "Ollama" not in answer
    
    def test_principles_exclude_ollama_as_jarvis_creator(self):
        """Fundamental principles must exclude Ollama as JARVIS creator."""
        identity = get_canonical_identity()
        principles_text = " ".join(identity.fundamental_principles).lower()
        
        # Check that principles establish JARVIS as integrated system
        assert "integrated" in principles_text or "integrated" in principles_text
        assert "jarvis" in principles_text.lower()


# ==============================================================================
# EDGE CASES AND ERROR HANDLING
# ==============================================================================

class TestIdentityEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query_not_identified_as_identity(self):
        """Empty query must not be identified as identity question."""
        assert not is_jarvis_identity_query("")
    
    def test_case_insensitive_identity_detection(self):
        """Identity detection must be case-insensitive."""
        assert is_jarvis_identity_query("WHO CREATED YOU?")
        assert is_jarvis_identity_query("who created you?")
        assert is_jarvis_identity_query("Who Created You?")
    
    def test_resolver_handles_invalid_query_gracefully(self):
        """Resolver must handle invalid queries gracefully."""
        resolver = IdentityResolver()
        
        # Non-identity question should return None
        answer = resolver.resolve("What is Python?")
        assert answer is None
    
    def test_identity_question_with_extra_text(self):
        """Identity questions with extra text must be detected."""
        questions = [
            "Hey, who created you?",
            "Can you tell me who is your creator?",
            "I'd like to know your name",
            "Hi, who are you?",
        ]
        
        for question in questions:
            assert is_jarvis_identity_query(question)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
