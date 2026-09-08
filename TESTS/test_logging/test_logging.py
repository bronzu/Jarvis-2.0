"""
Test del sistema di logging di JARVIS 2.0.
"""

from INFRASTRUCTURE.logging import crea_logger


def test_crea_logger():
    logger = crea_logger()

    assert logger is not None
    assert logger.name == "JARVIS"
    assert logger.level != 0