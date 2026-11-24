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
    plugin = commands.add_parser("plugin-serve", help="start localhost Tweeter feedback bridge")
    plugin.add_argument("--feedback-dir", type=Path)
    plugin.add_argument("--dpo-root", type=Path)
    plugin.add_argument("--port", type=int, default=19785)
    plugin.add_argument("--cpu", action="store_true")
    stats = commands.add_parser("feedback-stats")
    stats.add_argument("--feedback-dir", type=Path)
    stats.add_argument("--target", type=int, default=80)
    backup = commands.add_parser("feedback-backup")
    backup.add_argument("--feedback-dir", type=Path)
    backup.add_argument("--destination", type=Path, required=True)
    export = commands.add_parser("feedback-export")
    export.add_argument("--feedback-dir", type=Path)
    export.add_argument("--destination", type=Path, required=True)
    recovery = commands.add_parser("feedback-recover")
    recovery.add_argument("--feedback-dir", type=Path)
    recovery.add_argument("--group-id")
    prepare_dpo = commands.add_parser("dpo-prepare")
    prepare_dpo.add_argument("--feedback-dir", type=Path)
    prepare_dpo.add_argument("--run-dir", type=Path, required=True)
    prepare_dpo.add_argument("--minimum-rating-difference", type=int, default=2)
    prepare_dpo.add_argument("--validation-fraction", type=float, default=.2)
    prepare_dpo.add_argument("--seed", type=int, default=415)
    reference = commands.add_parser("dpo-reference")
    reference.add_argument("--run-dir", type=Path, required=True)
    reference.add_argument("--cpu", action="store_true")
    dpo = commands.add_parser("dpo-train")
    dpo.add_argument("--run-dir", type=Path, required=True)
    dpo.add_argument("--policy-adapter", type=Path)
    dpo.add_argument("--resume-checkpoint")
    dpo.add_argument("--epochs", type=int, default=1)
    dpo.add_argument("--accumulation", type=int, default=4)
    dpo.add_argument("--learning-rate", type=float, default=5e-6)
    dpo.add_argument("--beta", type=float, default=.1)
    dpo.add_argument("--save-steps", type=int, default=10)
    dpo.add_argument("--save-total-limit", type=int, default=3)
    dpo.add_argument("--save-on-interrupt", action="store_true", default=True)
    dpo.add_argument("--max-optimizer-steps", type=int)
    dpo.add_argument("--seed", type=int, default=415)
    dpo.add_argument("--patience", type=int, default=3)
    dpo.add_argument("--cpu", action="store_true")
    dpo.add_argument("--dry-run", action="store_true")
    pause_dpo = commands.add_parser("dpo-pause")
    pause_dpo.add_argument("--run-dir", type=Path, required=True)
    checkpoints = commands.add_parser("dpo-checkpoints")
    checkpoints.add_argument("--run-dir", type=Path, required=True)
    recover_dpo = commands.add_parser("dpo-recover")
    recover_dpo.add_argument("--run-dir", type=Path, required=True)
    approve = commands.add_parser("dpo-approve")
    approve.add_argument("--checkpoint", type=Path, required=True)
    approve.add_argument("--dpo-root", type=Path)
    dpo_evaluation = commands.add_parser("dpo-evaluate")
    dpo_evaluation.add_argument("--adapter", type=Path, required=True)
    dpo_evaluation.add_argument("--partition", choices=("validation20", "validation", "test"), default="validation20")
    dpo_evaluation.add_argument("--seed", type=int, default=415)
    dpo_evaluation.add_argument("--temperature", type=float, default=.8)
    dpo_evaluation.add_argument("--top-p", type=float, default=.95)
    dpo_evaluation.add_argument("--cpu", action="store_true")
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
        elif args.command == "plugin-serve":
            from .plugin_bridge import Bridge, serve
            serve(Bridge(args.output_dir, args.feedback_dir, args.dpo_root, args.cpu), args.port)
            return 0
        elif args.command in ("feedback-stats", "feedback-backup", "feedback-export", "feedback-recover"):
            from .plugin_store import FeedbackStore
            store = FeedbackStore(args.feedback_dir)
            if args.command == "feedback-stats":
                result = {"feedback_dir": str(store.root), **store.statistics(args.target)}
            elif args.command == "feedback-backup":
                result = store.backup(args.destination)
            elif args.command == "feedback-export":
                result = store.export(args.destination)
            else:
                result = ({"group": store.get_group(args.group_id),
                           "ratings": store.current_ratings(args.group_id)} if args.group_id else
                          {"incomplete_groups": store.incomplete_groups()})
        elif args.command == "dpo-prepare":
            from .pipeline import verify_dataset
            from .plugin_preference import make_snapshot
            from .plugin_store import FeedbackStore
            manifest, _ = verify_dataset(args.output_dir)
            adapter = args.output_dir / "checkpoints/rhythm_coverage_v1/best_adapter"
            result = make_snapshot(FeedbackStore(args.feedback_dir), args.run_dir,
                                   manifest, adapter, args.minimum_rating_difference,
                                   args.validation_fraction, args.seed)
        elif args.command == "dpo-reference":
            from .plugin_preference import precompute_reference
            result = precompute_reference(args.run_dir, args.output_dir, args.cpu)
        elif args.command == "dpo-train":
            from .plugin_dpo import train_dpo
            result = train_dpo(args.run_dir, args.output_dir,
                               policy_adapter=args.policy_adapter,
                               resume_checkpoint=args.resume_checkpoint, epochs=args.epochs,
                               accumulation=args.accumulation, learning_rate=args.learning_rate,
                               beta=args.beta, save_steps=args.save_steps,
                               save_total_limit=args.save_total_limit, seed=args.seed,
                               patience=args.patience, cpu=args.cpu, dry_run=args.dry_run,
                               max_optimizer_steps=args.max_optimizer_steps,
                               save_on_interrupt=args.save_on_interrupt)
        elif args.command == "dpo-pause":
            from .plugin_store import timestamp
            path = args.run_dir / "pause.request"
            path.write_text(timestamp(), encoding="utf-8")
            result = {"pause_requested": str(path)}
        elif args.command in ("dpo-checkpoints", "dpo-recover"):
            from .plugin_checkpoints import inspect, recover
            root = args.run_dir / "checkpoints"
            if args.command == "dpo-checkpoints":
                result = {"checkpoints": inspect(root)}
            else:
                path, state, failures = recover(root)
                result = {"recovered_checkpoint": str(path), "global_step": state["global_step"],
                          "cursor": state["cursor"], "invalid_newer": failures}
        elif args.command == "dpo-approve":
            from .plugin_checkpoints import verify
            from .plugin_dpo import _atomic_progress
            checkpoint = args.checkpoint.resolve()
            state = verify(checkpoint)
            root = args.dpo_root or args.output_dir.parent / "DPOPreferenceRuns"
            if (not checkpoint.is_relative_to(root.resolve()) or
                    checkpoint.parent.name not in ("best", "final", "epochs") or
                    not (checkpoint / "adapter").is_dir()):
                raise ValueError("approval requires a verified non-prunable DPO checkpoint under DPOPreferenceRuns")
            _atomic_progress(root / "approved.json", {"adapter": str(checkpoint / "adapter"),
                "checkpoint": str(checkpoint), "global_step": state["global_step"]})
            result = {"approved_adapter": str(checkpoint / "adapter"),
                      "global_step": state["global_step"]}
        elif args.command == "dpo-evaluate":
            from .plugin_evaluate import evaluate_policies
            result = evaluate_policies(args.output_dir, args.adapter, args.partition,
                                       args.cpu, args.seed, args.temperature, args.top_p)
        else:
            from .inference import compare_reports
            result = compare_reports(args.reports, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        parser.exit(2, f"V4EA: {exc}\n")
    print(json.dumps(result, indent=2, default=str))
    return 0
