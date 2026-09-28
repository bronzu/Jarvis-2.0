#!/usr/bin/env python3
"""Tabella di verifica manuale: valore nel report contro valore nei JSON.

Solo standard library. Per ogni affermazione elencata nella missione stampa
affiancati il valore come scritto nel report e il valore letto dal JSON di
`results/`, con il percorso completo della chiave. Non interpreta, non giudica,
non arrotonda: se un valore non esiste nel JSON lo dice scrivendo NON TROVATO.

Uso:
    python3 experiments/mission1_unlearning/headline_check.py
"""

from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ESTESO_LORA = "unlearning_results_esteso.json"
ESTESO_FULL = "unlearning_results_full_esteso.json"
CAUSALE_LORA = "causal_unlearning_results.json"
CAUSALE_FULL = "causal_unlearning_results_full.json"

NOT_FOUND = "NON TROVATO"


def load(name: str):
    path = os.path.join(RESULTS, name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dig(node, dotted: str):
    """Risolve `a/b/c` dentro dizionari e liste; None se il percorso non esiste."""
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


def sweep_row(doc, floor, eta, steps):
    """Riga di `sweep_extended` identificata da pavimento, learning rate e passi."""
    if doc is None:
        return None, None
    for i, row in enumerate(doc.get("sweep_extended", [])):
        if (row["fisher_floor"] == floor and row["learning_rate"] == eta
                and row["max_steps"] == steps):
            return row, f"/sweep_extended[{i}]"
    return None, None


def main() -> int:
    esteso_lora = load(ESTESO_LORA)
    esteso_full = load(ESTESO_FULL)
    causale_lora = load(CAUSALE_LORA)
    causale_full = load(CAUSALE_FULL)

    row_full, path_full = sweep_row(esteso_full, 0.001, 0.0001, 100)

    rows = []

    def add(claim: str, report: str, doc, dotted: str, fmt):
        if doc is None:
            rows.append((claim, report, "NON TROVATO", f"{'?'}::{dotted}"))
            return
        value = dig(doc, dotted)
        if value is None:
            rows.append((claim, report, NOT_FOUND, f"{doc.get('__name__', '?')}::{dotted}"))
            return
        rows.append((claim, report, fmt(value), f"{dotted}  [{ESTESO_FULL if doc is esteso_full else CAUSALE_LORA if doc is causale_lora else ESTESO_LORA}]"))

    # 1-3. metric_orthogonal, ramo full-parameter, 100 passi, eta 1e-4, floor 0.001
    if row_full is not None:
        src = f"{ESTESO_FULL}::{path_full}"
        rows.append((
            "z del forgetting, metric_orthogonal full-parameter (100 passi, eta 1e-4)",
            "6.62", f"{row_full['z_one_sided']!r}", src + "/z_one_sided"))
        rows.append((
            "variazione del retain, stessa riga",
            "+0.02%",
            f"{row_full['retain_loss_relative_change']!r}  (x100 = {row_full['retain_loss_relative_change'] * 100:.4f}%)",
            src + "/retain_loss_relative_change"))
        rows.append((
            "collaterale massimo, stessa riga",
            "26.40%",
            f"{row_full['collateral_max_change']!r}  (x100 = {row_full['collateral_max_change'] * 100:.4f}%)",
            src + "/collateral_max_change"))
        rows.append((
            "frasi oltre il 5%, stessa riga",
            "4/12",
            f"{row_full['collateral_n_violations']} violazioni su "
            f"{dig(esteso_full, '/dataset/n_correlated')} frasi correlate",
            src + "/collateral_n_violations"))
    else:
        for claim, report in (("z del forgetting, metric_orthogonal full-parameter", "6.62"),
                              ("variazione del retain, stessa riga", "+0.02%"),
                              ("collaterale massimo, stessa riga", "26.40%"),
                              ("frasi oltre il 5%, stessa riga", "4/12")):
            rows.append((claim, report, NOT_FOUND, "riga di sweep non trovata"))

    # 4. set minimo causale
    for label, doc, name in (("LoRA", causale_lora, CAUSALE_LORA),
                             ("full-parameter", causale_full, CAUSALE_FULL)):
        for key in ("/b3c_minimal_components_constrained/k", "/b3_minimal_components/k",
                    "/b6_verdict/n_components_needed_constrained"):
            value = dig(doc, key) if doc else None
            rows.append((
                f"dimensione del set minimo causale ({label}) {key.rsplit('/', 1)[-1]}",
                "non applicato (nessun set minimo)",
                NOT_FOUND if value is None else repr(value),
                f"{name}::{key}"))
        pool = dig(doc, "/b3_candidate_pool_size") if doc else None
        rows.append((
            f"pool di candidati esplorati ({label})",
            "3123", NOT_FOUND if pool is None else repr(pool),
            f"{name}::/b3_candidate_pool_size"))

    # 5. esito dello sweep esteso
    for label, doc, name in (("LoRA", esteso_lora, ESTESO_LORA),
                             ("full-parameter", esteso_full, ESTESO_FULL)):
        sweep = dig(doc, "/sweep_extended") if doc else None
        if not sweep:
            rows.append((f"sweep esteso ({label})", "vedi §8", NOT_FOUND, f"{name}::/sweep_extended"))
            continue
        worst_retain = max(sweep, key=lambda r: r["retain_loss_relative_change"])
        worst_coll = max(sweep, key=lambda r: r["collateral_max_change"])
        best_z = min(sweep, key=lambda r: r["z_one_sided"])
        reached = [r for r in sweep if r["forgetting_one_sided"]]
        completed = [r for r in sweep if r["stop_reason"] == "max_steps"]
        worst_coll_completed = max(completed, key=lambda r: r["collateral_max_change"])
        rows.append((
            f"sweep esteso ({label}): ritenzione peggiore",
            "+5.00%",
            f"{worst_retain['retain_loss_relative_change'] * 100:.4f}%  "
            f"(eta={worst_retain['learning_rate']}, {worst_retain['max_steps']} passi)",
            f"{name}::/sweep_extended/retain_loss_relative_change (max)"))
        rows.append((
            f"sweep esteso ({label}): collaterale massimo su tutte le righe",
            "111009% LoRA / 6194% full",
            f"{worst_coll['collateral_max_change'] * 100:.4f}%  "
            f"(eta={worst_coll['learning_rate']}, {worst_coll['max_steps']} passi, "
            f"stop={worst_coll['stop_reason']})",
            f"{name}::/sweep_extended/collateral_max_change (max)"))
        rows.append((
            f"sweep esteso ({label}): collaterale massimo fra le righe a fine (stop=max_steps)",
            "111009% LoRA / 6194% full",
            f"{worst_coll_completed['collateral_max_change'] * 100:.4f}%  "
            f"(eta={worst_coll_completed['learning_rate']}, "
            f"{worst_coll_completed['max_steps']} passi)",
            f"{name}::/sweep_extended/collateral_max_change (max su stop=max_steps)"))
        rows.append((
            f"sweep esteso ({label}): z minimo (forgetting)",
            "-5.02 LoRA / -8.20 full",
            f"{best_z['z_one_sided']!r}  (eta={best_z['learning_rate']}, "
            f"{best_z['max_steps']} passi)",
            f"{name}::/sweep_extended/z_one_sided (min)"))
        rows.append((
            f"sweep esteso ({label}): righe che raggiungono il forgetting",
            "sì, distrutte",
            f"{len(reached)} su {len(sweep)}",
            f"{name}::/sweep_extended/forgetting_one_sided"))

    # 6. concordanza dei segni e posizioni 4-5
    for label, doc, name, reported in (("LoRA", causale_lora, CAUSALE_LORA, "46/48"),
                                       ("full-parameter", causale_full, CAUSALE_FULL, "47/48")):
        agree = dig(doc, "/b2_direction_agreement") if doc else None
        rows_scan = dig(doc, "/b2_site_scan") if doc else None
        n_rows = len(rows_scan.get("rows", [])) if isinstance(rows_scan, dict) else None
        rows.append((
            f"concordanza dei segni fra le due direzioni ({label})",
            reported, NOT_FOUND if agree is None else f"{agree} su {n_rows} righe",
            f"{name}::/b2_direction_agreement"))
    pos = dig(causale_lora, "/b2_coord_scan") if causale_lora else None
    rows.append((
        "quota di componenti coordinate sulle posizioni 4 e 5 (25/25)",
        "25/25",
        NOT_FOUND if pos is None else
        f"il JSON conserva solo n_candidates={pos.get('n_candidates')} e "
        f"n_positive={pos.get('n_positive')}: la classifica dei primi 25 e' "
        f"solo in output_causal_lora.txt",
        f"{CAUSALE_LORA}::/b2_coord_scan"))

    # 7. canary nello sweep esteso: valore della riga citata dal report e massimo assoluto
    canary_rows = (
        ("LoRA", esteso_lora, ESTESO_LORA, "99.99%", 0.001, 0.0003, 300),
        ("full-parameter", esteso_full, ESTESO_FULL, "21657%", 0.9, 0.003, 300),
    )
    for label, doc, name, reported, floor, eta, steps in canary_rows:
        cited, cited_path = sweep_row(doc, floor, eta, steps)
        if cited is None:
            rows.append((f"canary, riga citata dal report ({label})", reported, NOT_FOUND,
                         f"{name}::riga sweep floor={floor} eta={eta} steps={steps} assente"))
        else:
            rows.append((
                f"canary, riga citata dal report ({label})",
                reported,
                f"{cited['probe_max_change']!r}  (x100 = {cited['probe_max_change'] * 100:.4f}%, "
                f"{cited['probe_n_violations']}/{6} oltre soglia)",
                f"{name}::{cited_path}/probe_max_change"))
        sweep = dig(doc, "/sweep_extended") if doc else None
        if not sweep:
            rows.append((f"canary massimo su tutto lo sweep ({label})", "n/a", NOT_FOUND,
                         f"{name}::/sweep_extended"))
            continue
        worst = max(sweep, key=lambda r: r["probe_max_change"])
        rows.append((
            f"canary massimo su tutto lo sweep ({label})",
            "n/a (il report non lo dichiara)",
            f"{worst['probe_max_change']!r}  (x100 = {worst['probe_max_change'] * 100:.4f}%, "
            f"{worst['probe_n_violations']}/{6} oltre soglia; eta={worst['learning_rate']}, "
            f"{worst['max_steps']} passi)",
            f"{name}::/sweep_extended/probe_max_change (max)"))

    width_a, width_b, width_c = 62, 26, 22
    print(f"{'AFFERMAZIONE':<{width_a}} {'REPORT':<{width_b}} {'JSON':<{width_c}} CHIAVE")
    print("-" * 150)
    for claim, report, value, key in rows:
        print(f"{claim:<{width_a}} {report:<{width_b}} {value:<{width_c}} {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
