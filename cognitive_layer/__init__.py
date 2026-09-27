"""
Cognitive Layer
===============

Layer cognitivo aggiuntivo di JARVIS 2.0.

Ogni modulo in questo pacchetto espone una sola responsabilita' e non modifica
il comportamento degli organi gia' esistenti: riceve input gia' prodotti da un
altro organo e restituisce una decisione strutturata.

Moduli:
    - ``algorithmic_verifier``: verifica euristica dei rami di ragionamento
      prodotti dal modello, prima che la risposta venga restituita all'utente.
"""

__all__ = ["algorithmic_verifier"]
