"""Command line for isolated V4EA-synth experiments."""
import argparse
import json
from pathlib import Path

from . import OUTPUT


def main(argv=None):
    parser = argparse.ArgumentParser(description="V4EA percussion generation")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    tokenizer = commands.add_parser("train-tokenizer")
    tokenizer.add_argument("--source-output", type=Path)
    tokenizer.add_argument("--vocab-size", type=int, default=16384)
    tokenizer.add_argument("--dry-run", action="store_true")
    tokenizer.add_argument("--target-format", choices=("positions-v1", "musical-v2"),
                           default="positions-v1")
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source-output", type=Path)
    preparation.add_argument("--dry-run", action="store_true")
    preparation.add_argument("--target-format", choices=("positions-v1", "musical-v2"),
                             default="positions-v1")
    commands.add_parser("validate")
    training = commands.add_parser("train")
    training.add_argument("--resume-checkpoint")
    training.add_argument("--epochs", type=int, help="legacy sampler only; coverage derives epochs from passes")
    training.add_argument("--examples-per-epoch", type=int)
    training.add_argument("--dataset-passes", type=int, default=2)
    training.add_argument("--sampler", choices=("coverage", "legacy"), default="coverage")
    training.add_argument("--lora-learning-rate", type=float)
    training.add_argument("--embedding-learning-rate", type=float)
    training.add_argument("--warmup-fraction", type=float, default=0.1)
    training.add_argument("--accumulation", type=int, default=8)
    training.add_argument("--save-steps", type=int, default=0,
                          help="periodic optimizer-step saves; 0 saves complete state only at each epoch")
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--seed", type=int, default=415)
    training.add_argument("--cpu", action="store_true")
    training.add_argument("--dry-run", action="store_true")
    training.add_argument("--progress-seconds", type=int, default=180)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--partition", choices=("validation20", "validation", "test"),
                            default="validation20")
    evaluation.add_argument("--cpu", action="store_true")
    evaluation.add_argument("--constrained", action="store_true")
    evaluation.add_argument("--checkpoint-run", choices=("auto", "coverage", "legacy"),
                            default="auto")
    generation = commands.add_parser("generate")
    generation.add_argument("--input", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--target-note-count", type=int, required=True)
    generation.add_argument("--max-new-tokens", type=int, default=512)
    generation.add_argument("--cpu", action="store_true")
    generation.add_argument("--constrained", action="store_true")
    generation.add_argument("--checkpoint-run", choices=("auto", "coverage", "legacy"),
                            default="auto")
    comparison = commands.add_parser("compare", help="combine three saved evaluation reports")
    comparison.add_argument("--reports", type=Path, nargs=3, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train-tokenizer":
            from .pipeline import train_tokenizer
            result = train_tokenizer(args.output_dir, args.source_output,
                                     args.vocab_size, dry_run=args.dry_run,
                                     target_format=args.target_format)
        elif args.command == "prepare":
            from .pipeline import prepare
            result = prepare(args.output_dir, args.source_output, dry_run=args.dry_run,
                             target_format=args.target_format)
        elif args.command == "validate":
            from .pipeline import verify_dataset
            manifest, groups = verify_dataset(args.output_dir)
            result = {"version": manifest["version"], "pairs":
                      {split: len(rows) for split, rows in groups.items()}}
        elif args.command == "train":
            from .trainer import train
            result = train(args.output_dir, resume_checkpoint=args.resume_checkpoint,
                           epochs=args.epochs, examples_per_epoch=args.examples_per_epoch,
                           dataset_passes=args.dataset_passes, sampler=args.sampler,
                           lora_learning_rate=args.lora_learning_rate,
                           embedding_learning_rate=args.embedding_learning_rate,
                           warmup_fraction=args.warmup_fraction,
                           accumulation=args.accumulation,
                           save_steps=args.save_steps, patience=args.patience, seed=args.seed,
                           cpu=args.cpu, dry_run=args.dry_run,
                           progress_seconds=args.progress_seconds)
        elif args.command == "evaluate":
            from .inference import evaluate
            result = evaluate(args.output_dir, args.partition, args.cpu,
                              checkpoint_run=args.checkpoint_run, constrained=args.constrained)
        elif args.command == "generate":
            from .inference import generate
            result = generate(args.output_dir, args.input, args.output,
                              args.target_note_count, args.max_new_tokens, args.cpu,
                              checkpoint_run=args.checkpoint_run, constrained=args.constrained)
        else:
            from .inference import compare_reports
            result = compare_reports(args.reports, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        parser.exit(2, f"V4EA: {exc}\n")
    print(json.dumps(result, indent=2, default=str))
    return 0
