"""Command line for isolated V4EA-synth experiments."""
import argparse
import json
from pathlib import Path

from . import OUTPUT


def main(argv=None):
    parser = argparse.ArgumentParser(description="V4EA full-track percussion generation")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    tokenizer = commands.add_parser("train-tokenizer")
    tokenizer.add_argument("--source-output", type=Path)
    tokenizer.add_argument("--vocab-size", type=int, default=16384)
    tokenizer.add_argument("--dry-run", action="store_true")
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source-output", type=Path)
    preparation.add_argument("--dry-run", action="store_true")
    commands.add_parser("validate")
    training = commands.add_parser("train")
    training.add_argument("--resume-checkpoint")
    training.add_argument("--epochs", type=int, default=12)
    training.add_argument("--examples-per-epoch", type=int, default=268)
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--accumulation", type=int, default=8)
    training.add_argument("--save-steps", type=int, default=10)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--seed", type=int, default=415)
    training.add_argument("--cpu", action="store_true")
    training.add_argument("--dry-run", action="store_true")
    training.add_argument("--progress-seconds", type=int, default=180)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--partition", choices=("validation20", "validation", "test"),
                            default="validation20")
    evaluation.add_argument("--cpu", action="store_true")
    generation = commands.add_parser("generate")
    generation.add_argument("--input", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--target-note-count", type=int, required=True)
    generation.add_argument("--max-new-tokens", type=int, default=3072)
    generation.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "train-tokenizer":
            from .pipeline import train_tokenizer
            result = train_tokenizer(args.output_dir, args.source_output,
                                     args.vocab_size, dry_run=args.dry_run)
        elif args.command == "prepare":
            from .pipeline import prepare
            result = prepare(args.output_dir, args.source_output, dry_run=args.dry_run)
        elif args.command == "validate":
            from .pipeline import verify_dataset
            manifest, groups = verify_dataset(args.output_dir)
            result = {"version": manifest["version"], "pairs":
                      {split: len(rows) for split, rows in groups.items()}}
        elif args.command == "train":
            from .trainer import train
            result = train(args.output_dir, resume_checkpoint=args.resume_checkpoint,
                           epochs=args.epochs, examples_per_epoch=args.examples_per_epoch,
                           learning_rate=args.learning_rate, accumulation=args.accumulation,
                           save_steps=args.save_steps, patience=args.patience, seed=args.seed,
                           cpu=args.cpu, dry_run=args.dry_run,
                           progress_seconds=args.progress_seconds)
        elif args.command == "evaluate":
            from .inference import evaluate
            result = evaluate(args.output_dir, args.partition, args.cpu)
        else:
            from .inference import generate
            result = generate(args.output_dir, args.input, args.output,
                              args.target_note_count, args.max_new_tokens, args.cpu)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        parser.exit(2, f"V4EA: {exc}\n")
    print(json.dumps(result, indent=2, default=str))
    return 0
