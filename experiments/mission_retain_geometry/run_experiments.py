"""
Orchestratore experiment per mission_retain_geometry.

Esegue:
1. Training di tutti gli stati primari (lora_rank in {4, 0} × seed in {42, 43, 44})
   con checkpoint ai 25%, 50%, 100%.
2. Calcolo geometria per ogni stato primario (100%).
3. Calcolo geometria per stati esplorativi: 25%, 50% per ogni (lora_rank, seed).
4. Sweep dimensione R_prot a N/4, N/2, N per ogni stato primario (100%).
5. Sensibilita rcond: 1e-3, 1e-9 (calcolato dentro compute_geometry_for_state).

Risultati: JSON in experiments/mission_retain_geometry/results/
"""
import json
import sys
import time
from pathlib import Path

from experiments.mission_retain_geometry import geometry as geo
from experiments.mission_retain_geometry.geometry import (
    DEFAULT_EPOCHS,
    LORA_RANKS,
    SEEDS,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    compute_geometry_for_state,
    train_with_checkpoints,
)

PIPELINE = geo.corpus.build_pipeline()


def train_and_save(lora_rank, seed):
    tokenizer = PIPELINE["tokenizer"]
    t0 = time.time()
    info = train_with_checkpoints(tokenizer, lora_rank, seed)
    info["train_seconds"] = time.time() - t0
    param_type = "lora" if lora_rank > 0 else "full"
    meta_path = RESULTS_DIR / f"training_meta_{param_type}_s{seed}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[done] training {param_type} seed={seed} in {info['train_seconds']:.1f}s")
    return info


def compute_and_save(lora_rank, seed, fraction, r_prot_size, suffix=""):
    t0 = time.time()
    result = compute_geometry_for_state(
        lora_rank=lora_rank,
        seed=seed,
        checkpoint_fraction=fraction,
        epochs=DEFAULT_EPOCHS,
        r_prot_size=r_prot_size,
    )
    result["compute_seconds"] = time.time() - t0
    param_type = "lora" if lora_rank > 0 else "full"
    frac_str = f"f{int(fraction*100)}"
    rprot_str = f"R{r_prot_size}" if r_prot_size else "Rfull"
    name = f"state_{param_type}_s{seed}_{frac_str}_{rprot_str}{suffix}"
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[done] geometry {param_type} seed={seed} frac={fraction} "
          f"R_prot={r_prot_size} -> {out_path.name} ({t0:.1f}s)")
    return str(out_path)


def main():
    print("=" * 70)
    print("Mission Retain-Geometry: Full Experiment Suite")
    print(f"  LORA_RANKS={LORA_RANKS}, SEEDS={SEEDS}, EPOCHS={DEFAULT_EPOCHS}")
    print(f"  CHECKPOINT_DIR={CHECKPOINT_DIR}")
    print(f"  RESULTS_DIR={RESULTS_DIR}")
    print("=" * 70)

    # --- Phase 1: Train all primary states ---
    print("\n--- Phase 1: Training ---")
    for lora_rank in LORA_RANKS:
        for seed in SEEDS:
            train_and_save(lora_rank, seed)

    # --- Phase 2: Primary geometry (100% checkpoint, full R_prot) ---
    print("\n--- Phase 2: Primary geometry (frac=1.0, R_prot=full) ---")
    primary_files = []
    for lora_rank in LORA_RANKS:
        for seed in SEEDS:
            f = compute_and_save(lora_rank, seed, fraction=1.0, r_prot_size=None)
            primary_files.append(f)

    # --- Phase 3: Exploratory checkpoints (25%, 50%) ---
    print("\n--- Phase 3: Exploratory checkpoints ---")
    for lora_rank in LORA_RANKS:
        for seed in SEEDS:
            for frac in [0.25, 0.5]:
                compute_and_save(lora_rank, seed, fraction=frac, r_prot_size=None)

    # --- Phase 4: R_prot size sweep (100% checkpoint) ---
    print("\n--- Phase 4: R_prot size sweep ---")
    for lora_rank in LORA_RANKS:
        for seed in SEEDS:
            for r_prot_size in [6, 12, 24]:
                compute_and_save(lora_rank, seed, fraction=1.0, r_prot_size=r_prot_size,
                                 suffix=f"sweep_r{r_prot_size}")

    # --- Phase 5: rcond sensitivity for primary states ---
    print("\n--- Phase 5: rcond sensitivity ---")
    for lora_rank in LORA_RANKS:
        for seed in SEEDS:
            for rcond in [1e-3, 1e-9]:
                result = compute_geometry_for_state(
                    lora_rank=lora_rank, seed=seed,
                    checkpoint_fraction=1.0, r_prot_size=None,
                    rcond=rcond)
                param_type = "lora" if lora_rank > 0 else "full"
                suffix = f"rcond{rcond:.0e}"
                out_path = RESULTS_DIR / f"sensitivity_{param_type}_s{seed}_{suffix}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"[done] rcond sensitivity {param_type} s={seed} rcond={rcond}")

    print("\n" + "=" * 70)
    print("All experiments complete.")
    print(f"Results in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
