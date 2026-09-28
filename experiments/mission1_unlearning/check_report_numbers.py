#!/usr/bin/env python3
"""Verifica numerica del report di Missione 1 contro gli artefatti grezzi.

Solo standard library. Estrae dal report ogni numero con almeno due decimali e
verifica se lo stesso valore compaia nei file di
`experiments/mission1_unlearning/results/`. Stampa i numeri assenti con il numero
di riga del report in cui compaiono e il testo della riga.

Il confronto accetta due casi, per non produrre falsi positivi:

1. il numero nei grezzi ha più cifre decimali di quante riportate nel report
   (`4.6834` == 4.683367902586001 arrotondato a 4 decimali);
2. il numero nei grezzi ha meno cifre decimali di quante riportate nel report
   (`2.2035` non viene prodotto, ma `2.20350828` arrotondato sì).

**Il punto come separatore delle migliaia non è tollerato.** Un token come
`111.009` viene letto come il decimale 111.009 e non come 111009: se il report
scrive un intero con il punto come separatore delle migliaia, il numero non viene
trovato e la riga viene segnalata. È il comportamento voluto: nel documento in
italiano quella scrittura è ambigua e il checker pretende la forma non ambigua.

Nessuna tolleranza esplicita: il confronto è per arrotondamento al numero di
decimali del token del report.

Uso:
    python3 experiments/mission1_unlearning/check_report_numbers.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT = os.path.join(REPO_ROOT, "DOCUMENTAZIONE", "MISSION1_UNLEARNING_REPORT.md")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Numero con almeno due decimali: 4.6834, 2.286e-02, -2.464e-03.
# I lookbehind/lookahead impediscono di estrarre un prefisso di un numero più
# lungo (`2.737806` non deve produrre `2.737`) e di perdere il segno. Il report
# usa il segno meno tipografico U+2212, i grezzi il trattino ASCII: vengono
# normalizzati in `normalize`.
REPORT_NUMBER = re.compile(r"(?<![\w.])[-−+]?\d+\.\d{2,}(?:[eE][-+]?\d+)?(?![\d])(?!\.\d)")
# Qualunque numerico presente nei grezzi, intero o decimale, con esponente
RAW_NUMBER = re.compile(r"(?<![\w.])[-−+]?\d+\.?\d*(?:[eE][-+]?\d+)?(?![\d])(?!\.\d)")


def normalize(token: str) -> str:
    """Segno meno tipografico -> trattino ASCII."""
    return token.replace("−", "-")


def load_raw_values() -> tuple[set[float], list[str]]:
    """Tutti i valori numerici presenti nei file di results/."""
    values: set[float] = set()
    names: list[str] = []
    if not os.path.isdir(RESULTS_DIR):
        return values, names
    for name in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, name)
        if not os.path.isfile(path):
            continue
        names.append(name)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        for token in RAW_NUMBER.findall(text):
            try:
                values.add(float(normalize(token)))
            except ValueError:
                continue
    return values, names


def is_present(token: str, values: set[float]) -> bool:
    """Il token e' presente se coincide esattamente o arrotondato ai suoi decimali.

    Non esiste piu' alcuna interpretazione alternativa: in particolare il punto non
    viene mai letto come separatore delle migliaia.
    """
    mantissa = normalize(token).split("e")[0].split("E")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    try:
        value = float(normalize(token))
    except ValueError:
        return False
    for raw in values:
        try:
            if raw == value or round(raw, decimals) == value:
                return True
        except (OverflowError, ValueError):
            continue
    return False


def main() -> int:
    raw_values, files = load_raw_values()
    if not files:
        print(f"ERRORE: nessun file in {RESULTS_DIR}", file=sys.stderr)
        return 2

    with open(REPORT, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    missing: dict[tuple[str, str], list[int]] = defaultdict(list)
    total = 0
    for number, line in enumerate(lines, 1):
        for token in REPORT_NUMBER.findall(line):
            total += 1
            if is_present(token, raw_values):
                continue
            missing[(token, line.strip())].append(number)

    print(f"file grezzi letti            : {len(files)}")
    print(f"valori numerici distinti    : {len(raw_values)}")
    print(f"numeri con >=2 decimali     : {total} (occorrenze, {len({t for t, _ in missing})} distinti)")
    print(f"occorrenze non rintracciate : {sum(len(v) for v in missing.values())}")
    print(f"righe di report coinvolte   : {len({n for v in missing.values() for n in v})}")
    print()

    for (token, line), where in sorted(missing.items(), key=lambda kv: kv[1][0]):
        print(f"riga {where[0]}"
              + (f" (ripetuto alle righe {', '.join(str(n) for n in where[1:])})" if len(where) > 1 else ""))
        print(f"  numero : {token}")
        print(f"  riga   : {line}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
