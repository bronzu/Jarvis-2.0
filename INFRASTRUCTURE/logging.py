"""
Sistema di logging di JARVIS 2.0.

Gestisce la registrazione degli eventi del sistema.
"""

import logging


def crea_logger(nome: str = "JARVIS") -> logging.Logger:
    """
    Crea e restituisce il logger principale di JARVIS.
    """

    logger = logging.getLogger(nome)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formato = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formato)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger