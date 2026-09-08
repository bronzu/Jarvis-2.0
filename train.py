"""
Entry point di addestramento del Neural Core di JARVIS.

Questo script **prepara e avvia** il training su richiesta esplicita: non viene
eseguito nulla al semplice import. Supporta entrambe le architetture
proprietarie:

    --architecture next_token    modello a contesto singolo (leggero)
    --architecture transformer   Transformer autoregressivo (context window)

Esempi::

    # smoke test brevissimo, solo per verificare la pipeline
    python train.py --architecture transformer --context-size 128 \
        --epochs 1 --batch-size 8 --max-train-windows 32 --dry-run

    # training reale (lungo) sul corpus completo
    python train.py --architecture transformer --context-size 256 --epochs 8

Nessun modello, tokenizer o embedding esterno viene utilizzato.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.model_interface import NeuralModelInterface
from neural_core.optimizer import Adam, LearningRateScheduler
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import Trainer, build_model

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "neural_core" / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def build_training_setup(
    train_path,
    val_path,
    test_path=None,
    architecture="transformer",
    context_size=None,
    stride=None,
    max_vocab_size=8000,
    tokenizer_mode=Tokenizer.MODE_WHITESPACE,
    embedding_dim=64,
    hidden_dim=64,
    dropout=0.1,
    d_model=128,
    num_heads=4,
    d_ff=256,
    num_layers=4,
    learning_rate=0.001,
    warmup_steps=0,
    total_steps=None,
    schedule="constant",
    gradient_accumulation_steps=1,
    max_grad_norm=None,
    seed=None,
    dialogue_path=None,
):
    """
    Prepara tokenizer, dataset, modello e trainer **senza addestrare**.

    Se ``dialogue_path`` punta a un file di dialoghi JARVIS, il tokenizer viene
    addestrato sul testo **combinato** (train + dialoghi) così da coprire i
    token conversazionali e ridurre i ``<UNK>`` nel dataset misto. Il dataset
    dei dialoghi viene esposto come ``trainer.dialogue_dataset`` per il
    calcolo della *dialogue loss*.

    Returns:
        (trainer, training_dataset, validation_dataset, test_dataset|None)
    """
    tokenizer = Tokenizer(max_vocab_size=max_vocab_size, mode=tokenizer_mode)

    train_text = Path(train_path).read_text(encoding="utf-8")
    val_text = Path(val_path).read_text(encoding="utf-8")

    dialogue_text = ""
    if dialogue_path is not None:
        dialogue_file = Path(dialogue_path)
        if dialogue_file.exists():
            dialogue_text = dialogue_file.read_text(encoding="utf-8")

    # Fit prioritario sui dialoghi: i token conversazionali vengono fissati
    # nel vocabolario PRIMA del testo generale. Poiché il corpus generale è
    # molto più ampio, un fit combinato farebbe cadere i token dei dialoghi
    # sotto il limite di vocabolario; fissarli prima riduce drasticamente i
    # <UNK> nelle risposte di JARVIS.
    if dialogue_text:
        tokenizer.fit(dialogue_text)
    tokenizer.fit(train_text)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    training_dataset = Dataset(train_ids, context_size=context_size, stride=stride)
    validation_dataset = Dataset(val_ids, context_size=context_size, stride=stride)

    test_dataset = None
    if test_path is not None:
        test_ids = tokenizer.encode(Path(test_path).read_text(encoding="utf-8"))
        test_dataset = Dataset(test_ids, context_size=context_size, stride=stride)

    dialogue_dataset = None
    if dialogue_text:
        dialogue_ids = tokenizer.encode(dialogue_text)
        # Il dataset dei dialoghi viene costruito solo se contiene abbastanza
        # token per almeno una finestra; in caso contrario si salta (il
        # training resta sul testo generale).
        if context_size is None or len(dialogue_ids) >= context_size + 1:
            dialogue_dataset = Dataset(
                dialogue_ids, context_size=context_size, stride=stride
            )

    vocab_size = len(tokenizer.vocabulary)

    if str(architecture).lower() in {"transformer", "microtransformer"}:
        model = build_model(
            "transformer",
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=context_size or 256,
            dropout=dropout,
            seed=seed,
        )
    else:
        model = build_model(
            "next_token",
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    scheduler = None
    if warmup_steps > 0 or total_steps is not None or schedule != "constant":
        scheduler = LearningRateScheduler(
            base_learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule=schedule,
        )

    trainer = Trainer(
        model,
        tokenizer,
        CrossEntropyLoss(),
        Adam(learning_rate=learning_rate),
        scheduler=scheduler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
    )

    # Espone il dataset dei dialoghi e i testi per le metriche di training
    # (unknown token rate e dialogue loss) senza alterare l'interfaccia di
    # ritorno usata dagli altri componenti.
    trainer.dialogue_dataset = dialogue_dataset
    trainer.training_text = train_text
    trainer.dialogue_text = dialogue_text

    return trainer, training_dataset, validation_dataset, test_dataset


def train_and_save(
    train_path,
    val_path,
    output_dir,
    epochs=8,
    embedding_dim=64,
    hidden_dim=64,
    dropout=0.1,
    batch_size=4096,
    patience=4,
    max_vocab_size=8000,
    test_path=None,
    max_train_samples=None,
    architecture="transformer",
    context_size=None,
    stride=None,
    tokenizer_mode=Tokenizer.MODE_WHITESPACE,
    d_model=128,
    num_heads=4,
    d_ff=256,
    num_layers=4,
    learning_rate=0.001,
    warmup_steps=0,
    schedule="constant",
    gradient_accumulation_steps=1,
    max_grad_norm=None,
    resume_path=None,
    seed=None,
    verbose=True,
    dialogue_path=None,
):
    output_dir = Path(output_dir)

    if output_dir == DATA_DIR:
        output_dir = EXPERIMENTS_DIR / "default"

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "config"
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    reports_dir = output_dir / "reports"
    for directory in (config_dir, checkpoints_dir, metrics_dir, logs_dir, reports_dir):
        directory.mkdir(exist_ok=True)

    trainer, training_dataset, validation_dataset, test_dataset = build_training_setup(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        architecture=architecture,
        context_size=context_size,
        stride=stride,
        max_vocab_size=max_vocab_size,
        tokenizer_mode=tokenizer_mode,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        schedule=schedule,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        dialogue_path=dialogue_path,
    )

    tokenizer = trainer.tokenizer
    inputs, targets = training_dataset.get_arrays()
    val_inputs, val_targets = validation_dataset.get_arrays()

    # ------------------------------------------------------------------
    # Integrazione del dataset conversazionale (dataset misto)
    # ------------------------------------------------------------------
    # Il training avviene sul testo generale + le conversazioni JARVIS. La
    # validation resta sul testo generale ("validation loss generale"), mentre
    # il dataset dei dialoghi serve a calcolare la "dialogue loss".
    dialogue_inputs = None
    dialogue_targets = None
    dialogue_loss = None
    if getattr(trainer, "dialogue_dataset", None) is not None:
        dialogue_inputs, dialogue_targets = trainer.dialogue_dataset.get_arrays()
        inputs = np.concatenate([inputs, dialogue_inputs], axis=0)
        targets = np.concatenate([targets, dialogue_targets], axis=0)

    # Tasso di token sconosciuti (<UNK>) sul testo di training e sui dialoghi.
    unknown_token_rate_train = tokenizer.unknown_token_rate(
        getattr(trainer, "training_text", "")
    )
    unknown_token_rate_dialogue = tokenizer.unknown_token_rate(
        getattr(trainer, "dialogue_text", "")
    )

    if max_train_samples is not None and len(inputs) > max_train_samples:
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(inputs))[:max_train_samples]
        inputs = inputs[indices]
        targets = targets[indices]

    if verbose:
        print(f"Architettura: {type(trainer.model).__name__}", flush=True)
        print(f"Vocabolario: {len(tokenizer.vocabulary)} token", flush=True)
        print(f"Context window: {getattr(trainer.model, 'max_len', None)}", flush=True)
        print(f"Parametri: {trainer.model.parameter_count()}", flush=True)
        print(f"Training: {len(inputs)} campioni", flush=True)
        print(f"Validation: {len(val_inputs)} campioni", flush=True)
        print(f"batch_size={batch_size}, accumulation={gradient_accumulation_steps}", flush=True)

    checkpoint_path = str(checkpoints_dir / "best_model.json")
    last_checkpoint_path = str(checkpoints_dir / "last_training_state.pkl")

    def on_epoch_end(epoch, train_loss, val_loss, trainer, is_best, patience_counter):
        if not verbose:
            return
        d_loss = None
        if dialogue_inputs is not None and dialogue_targets is not None:
            d_loss = float(trainer.evaluate(dialogue_inputs, dialogue_targets, batch_size=batch_size))
        unk_train = float(tokenizer.unknown_token_rate(getattr(trainer, "training_text", "")))
        unk_dialogue = float(tokenizer.unknown_token_rate(getattr(trainer, "dialogue_text", "")))

        best_marker = " [BEST]" if is_best else ""
        print(
            f"Epoca {epoch} - "
            f"Training loss: {train_loss:.4f} - "
            f"Validation loss: {val_loss:.4f}"
            f"{best_marker}",
            flush=True,
        )
        if d_loss is not None:
            print(f"         Dialogue loss: {d_loss:.4f}", flush=True)
        print(
            f"         UNK rate train: {unk_train:.4f} | UNK rate dialogo: {unk_dialogue:.4f}",
            flush=True,
        )

    history = trainer.train_with_validation(
        inputs,
        targets,
        val_inputs,
        val_targets,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        resume_path=resume_path,
        on_epoch_end=on_epoch_end,
    )

    best_val_loss = min(entry["validation_loss"] for entry in history)

    # ------------------------------------------------------------------
    # Metrica aggiuntiva: dialogue loss sul dataset conversazionale
    # ------------------------------------------------------------------
    if dialogue_inputs is not None and dialogue_targets is not None:
        dialogue_loss = float(
            trainer.evaluate(dialogue_inputs, dialogue_targets, batch_size=batch_size)
        )

    metrics_path = metrics_dir / "metrics.json"
    metrics = {
        "unknown_token_rate_train": float(unknown_token_rate_train),
        "unknown_token_rate_dialogue": float(unknown_token_rate_dialogue),
        "dialogue_loss": dialogue_loss,
        "validation_loss": best_val_loss,
        "mixed_training_samples": int(len(inputs)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tokenizer_path = checkpoints_dir / "tokenizer.pkl"
    tokenizer_json_path = checkpoints_dir / "tokenizer.json"
    model_path = checkpoints_dir / "model.pkl"

    trainer.save_tokenizer(str(tokenizer_path))
    tokenizer.save(str(tokenizer_json_path))
    trainer.save_weights(str(model_path))

    config_path = config_dir / "training_config.json"
    config_payload = {
        "architecture": architecture,
        "context_size": context_size,
        "stride": stride,
        "max_vocab_size": max_vocab_size,
        "tokenizer_mode": tokenizer_mode,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "schedule": schedule,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
        "dialogue_path": dialogue_path,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    if verbose:
        print(f"Miglior validation loss: {best_val_loss:.4f}", flush=True)
        print(
            f"Tasso token <UNK> (training): {unknown_token_rate_train:.4f}",
            flush=True,
        )
        print(
            f"Tasso token <UNK> (dialoghi): {unknown_token_rate_dialogue:.4f}",
            flush=True,
        )
        if dialogue_loss is not None:
            print(f"Dialogue loss: {dialogue_loss:.4f}", flush=True)
        print(f"Tokenizer salvato in: {tokenizer_path}", flush=True)
        print(f"Tokenizer JSON salvato in: {tokenizer_json_path}", flush=True)
        print(f"Modello salvato in: {model_path}", flush=True)
        print(f"Best checkpoint salvato in: {checkpoint_path}", flush=True)
        print(f"Stato training (resume) salvato in: {last_checkpoint_path}", flush=True)

    if test_dataset is not None:
        test_inputs, test_targets = test_dataset.get_arrays()
        test_loss = trainer.evaluate(test_inputs, test_targets, batch_size=batch_size)
        metrics["test_loss"] = float(test_loss)
        if verbose:
            print(f"Test loss: {test_loss:.4f}", flush=True)

    interface = NeuralModelInterface(model=trainer.model, tokenizer=trainer.tokenizer)
    if verbose:
        print("Stato Neural Core:", interface.status(), flush=True)

    return history, best_val_loss, metrics


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Addestramento del Neural Core di JARVIS")
    parser.add_argument("--architecture", default="transformer", choices=["next_token", "transformer"])
    parser.add_argument("--context-size", type=int, default=None,
                        help="Context window (128, 256 o 512) per il Transformer")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument(
        "--dialogue-path",
        default=str(DATA_DIR / "jarvis_dialogue.txt"),
        help="File di dialoghi JARVIS da integrare nel dataset misto di training.",
    )
    parser.add_argument("--tokenizer-mode", default=Tokenizer.MODE_WHITESPACE,
                        choices=list(Tokenizer.SUPPORTED_MODES))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--schedule", default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint completo da cui riprendere")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepara la pipeline e stampa la configurazione senza addestrare")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)

    train_file = DATA_DIR / "train.txt"
    val_file = DATA_DIR / "validation.txt"
    test_file = DATA_DIR / "test.txt"

    if args.dry_run:
        trainer, training_dataset, validation_dataset, _ = build_training_setup(
            train_path=train_file,
            val_path=val_file,
            test_path=None,
            architecture=args.architecture,
            context_size=args.context_size,
            stride=args.stride,
            max_vocab_size=args.max_vocab_size,
            tokenizer_mode=args.tokenizer_mode,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            schedule=args.schedule,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            dialogue_path=args.dialogue_path,
        )
        print("=== DRY RUN: nessun addestramento eseguito ===")
        print(f"Architettura: {type(trainer.model).__name__}")
        print(f"Parametri: {trainer.model.parameter_count()}")
        print(f"Vocabolario: {len(trainer.tokenizer.vocabulary)}")
        print(f"Context window: {getattr(trainer.model, 'max_len', None)}")
        print(f"Campioni di training: {len(training_dataset)}")
        print(f"Campioni di validation: {len(validation_dataset)}")
        return 0

    train_and_save(
        train_path=train_file,
        val_path=val_file,
        output_dir=DATA_DIR,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        patience=args.patience,
        max_vocab_size=args.max_vocab_size,
        test_path=test_file,
        max_train_samples=args.max_train_samples,
        architecture=args.architecture,
        context_size=args.context_size,
        stride=args.stride,
        tokenizer_mode=args.tokenizer_mode,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        schedule=args.schedule,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        resume_path=args.resume,
        seed=args.seed,
        dialogue_path=args.dialogue_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
