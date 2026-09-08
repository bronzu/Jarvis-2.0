import pytest

from executive_system.observation_feedback import ObservationFeedback


class TestObservationFeedback:
    def test_record_legacy_two_args(self):
        feedback = ObservationFeedback()
        feedback.record("filesystem", "ok")
        recorded = feedback.all()
        assert recorded == [{"action": "filesystem", "result": "ok"}]

    def test_record_with_outcome(self):
        feedback = ObservationFeedback()
        entry = feedback.record("tool", "PermissionError negato", outcome="failure")
        assert entry["outcome"] == "failure"
        assert entry["suggestion"] == "richiedi_approvazione_utente"

    def test_record_classifies_failure(self):
        feedback = ObservationFeedback()
        entry = feedback.record("tool", "PermissionError negato", outcome="failure")
        assert entry["outcome"] == "failure"

    def test_success_rate(self):
        feedback = ObservationFeedback()
        feedback.record("a", "ok", outcome="success")
        feedback.record("b", "errore", outcome="failure")
        assert abs(feedback.success_rate() - 0.5) < 1e-6

    def test_metrics_counts(self):
        feedback = ObservationFeedback()
        feedback.record("a", "ok", outcome="success")
        feedback.record("b", "errore", outcome="failure")
        metrics = feedback.metrics()
        assert metrics["total_actions"] == 2
        assert metrics["successes"] == 1
        assert metrics["failures"] == 1