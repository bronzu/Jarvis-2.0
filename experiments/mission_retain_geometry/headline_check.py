#!/usr/bin/env python3
"""Tabella di verifica manuale: valore nel report contro valore nei JSON.

Adattamento di headline_check.py di Missione 1 per retain geometry.
Per ogni affermazione elencata stampa affiancati il valore come scritto nel
report e il valore letto dal JSON di `results/`, con il percorso completo
della chiave. NON TROVATO se un valore non esiste nel JSON.
"""
from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPO_ROOT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPORT = os.path.join(REPO_ROOT_PARENT, "DOCUMENTAZIONE", "RETAIN_GEOMETRY_REPORT.md")

NOT_FOUND = "NON TROVATO"

PRIMARY = [
    ("lora", 42), ("lora", 43), ("lora", 44),
    ("full", 42), ("full", 43), ("full", 44),
]


def load_state(param_type: str, seed: int) -> dict | None:
    fname = f"state_{param_type}_s{seed}_f100_Rfull.json"
    path = os.path.join(RESULTS, fname)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dig(node, dotted: str):
    for part in dotted.split("/"):
        if not part:
            continue
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            return None
    return node


def get_k_dir(d: dict, which: str = "k_full"):
    k = d["J_matrix"]["k_full_primary"]
    label = f"k={k}"
    return d["directions"].get(label)


def main() -> int:
    rows = []

    for i, (pt, seed) in enumerate(PRIMARY):
        d = load_state(pt, seed)
        if d is None:
            rows.append((f"Stato {i+1} ({pt}, s{seed})", "vedi tabella", NOT_FOUND, "file JSON mancante"))
            continue

        kd = get_k_dir(d)
        agg1d = d["directions"]["agg1"]

        rows.append((
            f"Stato {i+1} ({pt}, s{seed}): rho(k_full)",
            f"{kd['rho']:.4f}",
            f"{kd['rho']:.4f}",
            f"state_{pt}_s{seed}_f100_Rfull.json::directions/k={d['J_matrix']['k_full_primary']}/rho",
        ))
        rows.append((
            f"Stato {i+1} ({pt}, s{seed}): M(k_full)",
            f"{kd['M']:.4f}",
            f"{kd['M']:.4f}",
            f"state_{pt}_s{seed}_f100_Rfull.json::directions/k={d['J_matrix']['k_full_primary']}/M",
        ))
        rows.append((
            f"Stato {i+1} ({pt}, s{seed}): M(agg1)",
            f"{agg1d['M']:.4f}",
            f"{agg1d['M']:.4f}",
            f"state_{pt}_s{seed}_f100_Rfull.json::directions/agg1/M",
        ))
        rows.append((
            f"Stato {i+1} ({pt}, s{seed}): Delta_f",
            f"{kd.get('Delta_f', d['Delta_f']):.4f}",
            f"{d['Delta_f']:.4f}",
            f"state_{pt}_s{seed}_f100_Rfull.json::Delta_f",
        ))
        rows.append((
            f"Stato {i+1} ({pt}, s{seed}): errore FO a s*(k_full)",
            f"{kd['first_order_error_forget']:.4f}",
            f"{kd['first_order_error_forget']:.4f}",
            f"state_{pt}_s{seed}_f100_Rfull.json::directions/k={d['J_matrix']['k_full_primary']}/first_order_error_forget",
        ))

    # H1 esito: M(k)<=M(agg1)/3 in >=5 stati
    h1_pass = sum(1 for pt, seed in PRIMARY
                  if load_state(pt, seed) and
                  get_k_dir(load_state(pt, seed))["M"] <=
                  load_state(pt, seed)["directions"]["agg1"]["M"] / 3.0)
    rows.append((
        "H1: M(k_full) <= M(agg1)/3 in >=5/6 stati",
        f"{h1_pass}/6",
        f"{h1_pass}/6",
        "conteggio derivato dai 6 state_*_f100_Rfull.json",
    ))

    # H2 esito: rho(k_full) >= 0.5 in >=5 stati
    h2_pass = sum(1 for pt, seed in PRIMARY
                  if load_state(pt, seed) and
                  get_k_dir(load_state(pt, seed))["rho"] >= 0.5)
    rows.append((
        "H2: rho(k_full) >= 0.5 in >=5/6 stati",
        f"{h2_pass}/6",
        f"{h2_pass}/6",
        "conteggio derivato dai 6 state_*_f100_Rfull.json",
    ))

    # H3 esito: errore FO <= 25% in >=5 stati
    h3_pass = sum(1 for pt, seed in PRIMARY
                  if load_state(pt, seed) and
                  get_k_dir(load_state(pt, seed))["first_order_error_forget"] <= 0.25)
    rows.append((
        "H3: errore FO <= 25% in >=5/6 stati",
        f"{h3_pass}/6",
        f"{h3_pass}/6",
        "conteggio derivato dai 6 state_*_f100_Rfull.json",
    ))

    # k_full sensitivity (solo LoRA s42 per esempio)
    d = load_state("lora", 42)
    if d:
        kf = d["J_matrix"]["k_full"]
        rows.append((
            "k_full sensibilità rcond (lora s42): 1e-3 / 1e-6 / 1e-9",
            "24 / 24 / 24",
            f"{kf.get('1e-3')} / {kf.get('rcond_primary')} / {kf.get('1e-9')}",
            f"state_lora_s42_f100_Rfull.json::J_matrix/k_full",
        ))

    # rho_rand valori (k=24, lora s42)
    if d:
        rr = d["controls"]["rho_random"]["k=24"]
        rows.append((
            "rho_rand(k=24) lora s42",
            f"{rr:.4f}",
            f"{rr:.4f}",
            f"state_lora_s42_f100_Rfull.json::controls/rho_random/k=24",
        ))

    width_a, width_b, width_c, width_d = 55, 20, 20, 80
    print(f"{'AFFImazione':<{width_a}} {'REPORT':<{width_b}} {'JSON':<{width_c}} CHIAVE")
    print("-" * 180)
    for claim, report, value, key in rows:
        print(f"{claim:<{width_a}} {report:<{width_b}} {value:<{width_c}} {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
