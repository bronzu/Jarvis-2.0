#!/usr/bin/env python3
"""Generatore del report RETAIN_GEOMETRY_REPORT.md dai JSON di results/.

Solo standard library. Legge i 6 stati primari + controlli descrittivi e produce
il report in DOCUMENTAZIONE/RETAIN_GEOMETRY_REPORT.md.
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPORT_PATH = os.path.join(REPO_ROOT, "DOCUMENTAZIONE", "RETAIN_GEOMETRY_REPORT.md")


def load_state(lora_rank: int, seed: int) -> dict:
    param_type = "lora" if lora_rank > 0 else "full"
    fname = f"state_{param_type}_s{seed}_f100_Rfull.json"
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(d: dict):
    """Estrae le metriche chiave dal JSON per il report."""
    k_full = d["J_matrix"]["k_full_primary"]
    k_label = f"k={k_full}"
    directions = d["directions"]

    k_dir = directions.get(k_label, {})
    agg1_dir = directions.get("agg1", {})

    metrics = {
        "param_type": d["state"]["param_type"],
        "lora_rank": d["state"]["lora_rank"],
        "seed": d["state"]["seed"],
        "d": d["state"]["d"],
        "theta_norm": d["state"]["theta_norm"],
        "k_full": k_full,
        "k_full_rcond": d["J_matrix"]["k_full"],
        "Delta_f": d["Delta_f"],
        "Delta_f_status": d["Delta_f_status"],
        "rho_k": k_dir.get("rho"),
        "M_k": k_dir.get("M"),
        "s_star_k": k_dir.get("s_star"),
        "n_over_5pct_k": k_dir.get("n_over_5pct"),
        "n_total_corr_k": k_dir.get("n_total_corr"),
        "censored_k": k_dir.get("censored"),
        "M_agg1": agg1_dir.get("M"),
        "rho_agg1": agg1_dir.get("rho"),
        "n_over_5pct_agg1": agg1_dir.get("n_over_5pct"),
        "n_over_5pct_canary": k_dir.get("canary_n_over_5pct"),
        "n_over_5pct_retain": k_dir.get("retain_n_over_5pct"),
        "n_total_retain": len(k_dir.get("retain_rel_changes", [])),
        "n_total_canary": len(k_dir.get("canary_rel_changes", [])),
        "first_order_error_forget": k_dir.get("first_order_error_forget"),
        "first_order_pred_forget": k_dir.get("first_order_pred_forget"),
        "first_order_measured_forget": k_dir.get("first_order_measured_forget"),
        "forget_nll_before": d["g_forget"]["nll_before"],
        "g_forget_norm": d["g_forget"]["norm"],
        "ood_mean_nll": d["E_ood"]["mean_nll"],
        "train_loss_initial": d["training"]["train_loss_initial"],
        "train_loss_final": d["training"]["train_loss_final"],
        "rho_random": d["controls"]["rho_random"],
        "rho_with_ood": d["controls"]["rho_with_ood_sentence"],
    }
    return metrics


def check_h1(m: dict) -> tuple[str, str]:
    """H1: M(k_full) <= M(agg1) / 3. Dati censurati se Delta_f non applicabile."""
    if m["Delta_f_status"] != "applicabile":
        return "NON APPLICABILE", f"Delta_f={m['Delta_f']:.4f} {m['Delta_f_status']}"
    if m["censored_k"] is not None:
        return "DATO CENSURATO", m["censored_k"]
    M_k = m["M_k"]
    M_agg1 = m["M_agg1"]
    ratio = M_k / M_agg1 if M_agg1 else float("inf")
    passed = M_k <= M_agg1 / 3.0
    return ("VERO" if passed else "FALSO",
            f"M(k)={M_k:.4f}, M(agg1)={M_agg1:.4f}, rapporto={ratio:.4f}, soglia={M_agg1/3:.4f}")


def check_h2(m: dict) -> tuple[str, str]:
    """H2: rho(k_full) >= 0.5."""
    if m["Delta_f_status"] != "applicabile":
        return "NON APPLICABILE", f"Delta_f={m['Delta_f']:.4f} {m['Delta_f_status']}"
    if m["censored_k"] is not None:
        return "DATO CENSURATO", m["censored_k"]
    rho = m["rho_k"]
    passed = rho >= 0.5
    return ("VERO" if passed else "FALSO",
            f"rho={rho:.4f}, soglia=0.5000")


def check_h3(m: dict) -> tuple[str, str]:
    """H3: errore primo ordine su forget a s*(k_full) <= 25%."""
    if m["Delta_f_status"] != "applicabile":
        return "NON APPLICABILE", f"Delta_f={m['Delta_f']:.4f} {m['Delta_f_status']}"
    if m["censored_k"] is not None:
        return "DATO CENSURATO", m["censored_k"]
    err = m["first_order_error_forget"]
    passed = err <= 0.25
    return ("VERO" if passed else "FALSO",
            f"errore={err:.4f}, soglia=0.2500")


def main():
    states = []
    for lora_rank in [4, 0]:
        for seed in [42, 43, 44]:
            d = load_state(lora_rank, seed)
            if d is None:
                print(f"WARNING: state lora_rank={lora_rank} seed={seed} not found", file=sys.stderr)
                continue
            m = extract_metrics(d)
            states.append(m)

    h1_pass = sum(1 for s in states if check_h1(s)[0] == "VERO")
    h1_cens = sum(1 for s in states if check_h1(s)[0] == "DATO CENSURATO")
    h1_na = sum(1 for s in states if check_h1(s)[0] == "NON APPLICABILE")
    h2_pass = sum(1 for s in states if check_h2(s)[0] == "VERO")
    h2_cens = sum(1 for s in states if check_h2(s)[0] == "DATO CENSURATO")
    h2_na = sum(1 for s in states if check_h2(s)[0] == "NON APPLICABILE")
    h3_pass = sum(1 for s in states if check_h3(s)[0] == "VERO")
    h3_cens = sum(1 for s in states if check_h3(s)[0] == "DATO CENSURATO")
    h3_na = sum(1 for s in states if check_h3(s)[0] == "NON APPLICABILE")

    lines = []
    lines.append("# Report — Protezione per-frase vs protezione aggregata: geometria al primo ordine")
    lines.append("")
    lines.append("Esperimento pre-registrato in `experiments/mission_retain_geometry/PREREGISTRATION.md`.")
    lines.append("Definizioni, ipotesi e soglie non modificate.")
    lines.append("")
    lines.append("## 1. Configurazione")
    lines.append("")
    lines.append("| Parametro | Valore |")
    lines.append("|---|---|")
    lines.append("| Epochs | 400 |")
    lines.append("| LR | 0.01 |")
    lines.append("| Batch size | 32 |")
    lines.append("| Context size | 64 |")
    lines.append("| Stride | 32 |")
    lines.append("| d_model | 64 |")
    lines.append("| num_heads | 4 |")
    lines.append("| d_ff | 128 |")
    lines.append("| num_layers | 2 |")
    lines.append("| LoRA rank | 4 / 0 (full) |")
    lines.append("| LoRA alpha | 16.0 |")
    lines.append("| Seeds | 42, 43, 44 |")
    lines.append("| rcond (primario) | 1e-6 |")
    lines.append("| rcond (sensibilità) | 1e-3, 1e-9 |")
    lines.append("| ALPHAS | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 |")
    lines.append("| FISHER_SAMPLES | 100 |")
    lines.append("| Checkpoint fractions | 0.25, 0.5, 1.0 |")
    lines.append("")
    lines.append("## 2. Stati primari (checkpoint 100%, R_prot completo)")
    lines.append("")
    lines.append("| # | Tipo | Seed | d | k_full | Delta_f | rho(k_full) | M(k_full) | M(agg1) | s*(k) | Errore FO | Stato |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(states):
        h1, _ = check_h1(m)
        h2, _ = check_h2(m)
        h3, _ = check_h3(m)
        status = "OK" if all(x == "VERO" for x in [h1, h2, h3]) else f"H1={h1} H2={h2} H3={h3}"
        lines.append(
            f"| {i+1} | {m['param_type']} | {m['seed']} | {m['d']} | {m['k_full']} | "
            f"{m['Delta_f']:.4f} ({m['Delta_f_status']}) | {m['rho_k']:.4f} | "
            f"{m['M_k']:.4f} | {m['M_agg1']:.4f} | {m['s_star_k']:.4f} | "
            f"{m['first_order_error_forget']:.4f} | {status} |"
        )
    lines.append("")

    lines.append("## 3. Esito delle ipotesi (6 stati primari)")
    lines.append("")
    lines.append("| Ipotesi | Soglia | Passa | Dato censurato | Non applicabile | Esito |")
    lines.append("|---|---|---|---|---|---|")
    n_states = len(states)
    h1_fail = n_states - h1_pass - h1_cens - h1_na
    h2_fail = n_states - h2_pass - h2_cens - h2_na
    h3_fail = n_states - h3_pass - h3_cens - h3_na
    lines.append(f"| H1: M(k_full) <= M(agg1)/3 | >=5/6 | {h1_pass}/{n_states} | {h1_cens} | {h1_na} | "
                 f"{'VERO' if h1_pass >= 5 else 'FALSO' if n_states == 6 and h1_pass < 5 else 'INCONCLUSIVO (troppi stati censurati/NA)'} |")
    lines.append(f"| H2: rho(k_full) >= 0.5 | >=5/6 | {h2_pass}/{n_states} | {h2_cens} | {h2_na} | "
                 f"{'VERO' if h2_pass >= 5 else 'FALSO' if n_states == 6 and h2_pass < 5 else 'INCONCLUSIVO (troppi stati censurati/NA)'} |")
    lines.append(f"| H3: errore FO <= 25% a s*(k_full) | >=5/6 | {h3_pass}/{n_states} | {h3_cens} | {h3_na} | "
                 f"{'VERO' if h3_pass >= 5 else 'FALSO' if n_states == 6 and h3_pass < 5 else 'INCONCLUSIVO (troppi stati censurati/NA)'} |")
    lines.append("")

    lines.append("## 4. Conteggi dettagliati (6 stati primari)")
    lines.append("")
    lines.append("| # | Tipo | Seed | M(k) frasi oltre 5% | M(agg1) frasi oltre 5% | Canary oltre 5% | Retain oltre 5% |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, m in enumerate(states):
        lines.append(
            f"| {i+1} | {m['param_type']} | {m['seed']} | "
            f"{m['n_over_5pct_k']}/{m['n_total_corr_k']} | "
            f"{m['n_over_5pct_agg1']}/{m['n_total_corr_k']} | "
            f"{m['n_over_5pct_canary']}/{m['n_total_canary']} | "
            f"{m['n_over_5pct_retain']}/{m['n_total_retain']} |"
        )
    lines.append("")

    lines.append("## 5. Stati censurati / non applicabili")
    lines.append("")
    censored = [(i+1, s) for i, s in enumerate(states)
                if s["Delta_f_status"] != "applicabile" or s["censored_k"] is not None]
    if censored:
        for idx, m in censored:
            lines.append(f"- **Stato {idx}** ({m['param_type']}, seed={m['seed']}): "
                         f"Delta_f={m['Delta_f']:.4f} ({m['Delta_f_status']}), "
                         f"censurato: {m['censored_k']}")
    else:
        lines.append("Nessuno stato censurato o non applicabile.")
    lines.append("")

    lines.append("## 6. Controlli descrittivi")
    lines.append("")
    lines.append("### 6.1 rho_rand(k) = sqrt((d - k) / d)")
    lines.append("")
    lines.append("| # | Tipo | Seed | d | k=1 | k=2 | k=4 | k=8 | k=16 | k=24 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(states):
        rr = m["rho_random"]
        vals = [f"{rr.get(f'k={k}', 'N/A'):.4f}" for k in [1, 2, 4, 8, 16, 24]]
        lines.append(f"| {i+1} | {m['param_type']} | {m['seed']} | {m['d']} | {' | '.join(vals)} |")
    lines.append("")

    lines.append("### 6.2 rho(k) con E_ood al posto di X_forget (controllo)")
    lines.append("")
    lines.append("| # | Tipo | Seed | k_full | rho(k, E_ood) | rho(k, X_forget) |")
    lines.append("|---|---|---|---|---|---|")
    for i, m in enumerate(states):
        rood = m["rho_with_ood"].get(f"k={m['k_full']}", m["rho_with_ood"].get(str(m["k_full"])))
        rood_str = f"{rood:.4f}" if rood is not None else "N/A"
        lines.append(
            f"| {i+1} | {m['param_type']} | {m['seed']} | {m['k_full']} | "
            f"{rood_str} | {m['rho_k']:.4f} |"
        )
    lines.append("")

    lines.append("### 6.3 Sensibilità rcond (k_full)")
    lines.append("")
    lines.append("| # | Tipo | Seed | rcond=1e-3 | rcond=1e-6 (primario) | rcond=1e-9 |")
    lines.append("|---|---|---|---|---|---|")
    for i, m in enumerate(states):
        kr = m["k_full_rcond"]
        lines.append(
            f"| {i+1} | {m['param_type']} | {m['seed']} | "
            f"{kr.get('1e-3', 'N/A')} | {kr.get('rcond_primary', 'N/A')} | "
            f"{kr.get('1e-9', 'N/A')} |"
        )
    lines.append("")

    lines.append("### 6.4 Verifica Missione 1: u = F^-1 * g_r (singola direzione)")
    lines.append("")
    lines.append("```")
    from geometry import verify_mission1_single_vector
    v1 = verify_mission1_single_vector()
    for k, val in v1.items():
        lines.append(f"{k}: {val}")
    lines.append("```")
    lines.append("")

    lines.append("## 7. Training metadata")
    lines.append("")
    lines.append("| # | Tipo | Seed | Loss iniziale | Loss finale | Epochs |")
    lines.append("|---|---|---|---|---|---|")
    for i, m in enumerate(states):
        li = f"{m['train_loss_initial']:.4f}" if m['train_loss_initial'] else "N/A"
        lf = f"{m['train_loss_final']:.4f}" if m['train_loss_final'] else "N/A"
        lines.append(
            f"| {i+1} | {m['param_type']} | {m['seed']} | "
            f"{li} | {lf} | 400 |"
        )
    lines.append("")

    lines.append("## 8. Deviazioni")
    lines.append("")
    lines.append("Modifiche a `geometry.py` successive al primo run sperimentale (prima del re-run):")
    lines.append("")
    lines.append("1. **save_checkpoint** (linea ~494): ora salva `{**MODEL_CONFIG, 'lora_rank': lora_rank}` invece di `MODEL_CONFIG`.")
    lines.append("   **Impatto**: Sì. I checkpoint full-parameter ora ricostruiscono correttamente il modello senza LoRA adapter.")
    lines.append("   Prima della correzione, `load_checkpoint` falliva con `ValueError: MicroTransformer richiede 41 parametri`.")
    lines.append("2. **load_checkpoint** (linea ~515): aggiunto ritorno della `train_history`. Ora restituisce 5 valori.")
    lines.append("   **Impatto**: Sì — `ensure_checkpoint` e `compute_geometry_for_state` aggiornati per unpack a 5 elementi.")
    lines.append("3. **ensure_checkpoint** (linea ~591): salva checkpoint a tutte le frazioni (25%, 50%, 100%), non solo 100%.")
    lines.append("   **Impatto**: Sì — i checkpoint a 25%/50% ora vengono persistiti e riutilizzati.")
    lines.append("4. **compute_geometry_for_state** (linea ~743): `train_loss_initial`, `train_loss_final`, `history` ora popolati da checkpoint.")
    lines.append("   **Impatto**: Sì — i metadati di training nei JSON sono ora popolati invece di essere `None`.")
    lines.append("")
    lines.append("**Re-run post-correzione**: identico per tutti i valori geometrici (rho, M, s*, errore FO, Delta_f).")
    lines.append("Solo i campi di tempo (`fisher.seconds`, `compute_seconds`, `train_seconds`) differiscono (non deterministici).")
    lines.append("")

    lines.append("## 9. Limiti")
    lines.append("")
    lines.append("- I risultati sono a livello di un singolo modello micro-trasformers (d_model=64, 2 layer) su un corpus di prova.")
    lines.append("- La geometria è valutata al primo ordine: la linearizzazione può non valere per passi grandi (s >> 1).")
    lines.append("- k_full = 24 in tutti gli stati primari: J è pienamente rangiato, quindi non c'è margine per testare la")
    lines.append("  transizione tra sottospazi di diversa dimensionalità.")
    lines.append("- E_ood ha solo 6 frasi: il controllo descrittivo rho(k, E_ood) è indicativo, non statistico.")
    lines.append("- LoRA rank 8/16 non supportati (necessiterebbero modifiche al codice; rank 0 = full-parameter).")
    lines.append("- I checkpoint a 25%/50% e gli sweep della dimensione di R_prot (N/4, N/2) sono stati esplorativi.")
    lines.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report scritto: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
