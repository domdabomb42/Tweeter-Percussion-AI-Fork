"""Independent V3 preparation, training, inference and comparison commands."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

from .data import ROOT, V1_OUTPUT, V2_OUTPUT, prepare


def counts_argument(text):
    counts = [int(part) for part in text.split(",")]
    if not counts or any(not 1 <= count <= 64 for count in counts):
        raise ValueError("counts must be comma-separated integers 1..64")
    return counts


def overnight(args):
    """Baseline fine-tuning, then an independently saved decoding experiment."""
    from .evaluation import compare, evaluate_v3
    from .model import train

    root = args.output_dir
    if not (root / "manifest.json").is_file():
        prepare(root)
    baseline = root / "checkpoints/rhythm/best_adapter"
    constrained = root / "checkpoints/constrained/best_adapter"
    if args.dry_run and (baseline.exists() or constrained.exists()):
        return {"status": "existing checkpoints; a fresh overnight run would not overwrite them",
                "baseline_checkpoint": str(baseline) if baseline.exists() else None,
                "constrained_checkpoint": str(constrained) if constrained.exists() else None}
    if constrained.exists():
        raise FileExistsError(f"constrained experiment already exists: {constrained}")
    history = train(root, "rhythm", epochs=args.epochs,
                    learning_rate=args.learning_rate, accumulation=args.accumulation,
                    patience=args.patience, min_delta=args.min_delta,
                    progress_seconds=args.progress_seconds, cpu=args.cpu,
                    resume=args.resume, dry_run=args.dry_run,
                    profile=getattr(args, "profile", False))
    if args.dry_run:
        return {"baseline_training_plan": history,
                "next_stage": "copy best baseline adapter, then run constrained-decoding evaluation",
                "constrained_checkpoint": str(constrained)}
    if not (baseline / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("baseline training finished without a best adapter")
    constrained.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(baseline, constrained)
    config = {"kind": "decoding experiment, not a second fine-tuning run",
              "source_adapter": str(baseline), "copied_adapter": str(constrained),
              "strategy": "greedy logits restricted to ascending tracker steps, exact requested count, end marker and EOS",
              "same_weights_as_baseline": True,
              "counts": args.counts, "partition": "test", "max_new_tokens": args.max_new_tokens}
    config_path = constrained.parent / "decoding_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved separate constrained adapter at {constrained}", flush=True)
    # A failed baseline benchmark must not prevent the requested second stage.
    try:
        baseline_eval = evaluate_v3(root, False, "test", args.cap, args.counts,
                                    args.max_new_tokens, args.cpu)
    except Exception as error:
        baseline_eval = {"error": str(error)}
        print(f"baseline evaluation failed: {error}; continuing to constrained stage", flush=True)
    constrained_eval = evaluate_v3(root, True, "test", args.cap, args.counts,
                                   args.max_new_tokens, args.cpu)
    cd2_eval = evaluate_v3(root, False, "test", args.cap, args.counts,
                           args.max_new_tokens, args.cpu, cd2=True)
    if V1_OUTPUT.is_dir() and V2_OUTPUT.is_dir():
        try:
            comparison = compare(root, "test", args.cap, args.counts,
                                 args.max_new_tokens, args.cpu)
        except Exception as error:
            comparison = {"error": str(error)}
            print(f"V1/V2/V3 comparison failed: {error}", flush=True)
    else:
        comparison = {"status": "unavailable", "reason": "V1/V2 source folders are absent"}
    result = {"best_epoch": history.get("best_epoch"),
              "best_validation_loss": history.get("best_validation_loss"),
              "baseline_checkpoint": str(baseline),
              "constrained_checkpoint": str(constrained),
              "constrained_decoding_config": str(config_path),
              "baseline_evaluation": baseline_eval,
              "constrained_evaluation": constrained_eval,
              "cd2_evaluation": cd2_eval,
              "v1_v2_v3_comparison": comparison}
    result_path = root / "overnight_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="V3 Tweeter position-only percussion generation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--encoding", choices=("lossless-v2", "legacy-v1", "compact"),
                                default="lossless-v2")
    prepare_parser.add_argument("--from-output", type=Path,
                                help="copy an existing verified augmented pool and add fixed validation")
    sub.add_parser("validate-splits")
    codec_compare = sub.add_parser("compare-codecs")
    codec_compare.add_argument("--from-output", type=Path)
    benchmark = sub.add_parser("benchmark-inference")
    benchmark.add_argument("--report", type=Path, default=ROOT / "benchmarks/v3_inference.json")
    benchmark.add_argument("--partition", choices=("validation", "test"), default="validation")
    benchmark.add_argument("--cap", type=int, default=1)
    benchmark.add_argument("--counts", type=counts_argument, default=[5, 8])
    benchmark.add_argument("--max-new-tokens", type=int, default=16)
    benchmark.add_argument("--cpu", action="store_true")
    train_parser = sub.add_parser("train")
    overnight_parser = sub.add_parser("overnight")
    for command in (train_parser, overnight_parser):
        command.add_argument("--epochs", type=int, default=None if command is train_parser else 20)
        command.add_argument("--learning-rate", type=float,
                             default=None if command is train_parser else 0.00005)
        command.add_argument("--accumulation", type=int, default=8)
        command.add_argument("--patience", type=int, default=None if command is train_parser else 3)
        command.add_argument("--min-delta", type=float, default=0.005)
        command.add_argument("--progress-seconds", type=int, default=180)
        command.add_argument("--cpu", action="store_true")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--profile", action="store_true")
    train_parser.add_argument("--init-adapter", type=Path)
    train_parser.add_argument("--examples-per-epoch", type=int)
    train_parser.add_argument("--save-steps", type=int)
    train_parser.add_argument("--resume-checkpoint", type=str)
    train_parser.add_argument("--seed", type=int, default=415)
    overnight_parser.add_argument("--cap", type=int, default=5)
    overnight_parser.add_argument("--counts", type=counts_argument, default=[5, 8, 10, 12, 16, 20])
    overnight_parser.add_argument("--max-new-tokens", type=int, default=512)
    generate = sub.add_parser("generate")
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--target-note-count", type=int, required=True)
    decoder_options = generate.add_mutually_exclusive_group()
    decoder_options.add_argument("--constrained", action="store_true")
    decoder_options.add_argument("--cd2", action="store_true",
                                 help="exact count and no duplicate positions, with no ordering restriction")
    generate.add_argument("--loop-ms", type=int)
    generate.add_argument("--reference-track-id", type=int)
    generate.add_argument("--target-track-id", type=int)
    generate.add_argument("--preset", type=Path)
    generate.add_argument("--max-new-tokens", type=int, default=3072)
    generate.add_argument("--cpu", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate_decoders = evaluate.add_mutually_exclusive_group()
    evaluate_decoders.add_argument("--constrained", action="store_true")
    evaluate_decoders.add_argument("--cd2", action="store_true")
    evaluate.add_argument("--partition", choices=("validation", "test"), default="test")
    evaluate.add_argument("--cap", type=int, default=5)
    evaluate.add_argument("--counts", type=counts_argument, default=[5, 8, 10, 12, 16, 20])
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument("--cpu", action="store_true")
    evaluate.add_argument("--no-shared-prefix-cache", action="store_true")
    evaluate.add_argument("--profile", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--partition", choices=("validation", "test"), default="test")
    compare.add_argument("--cap", type=int, default=5)
    compare.add_argument("--counts", type=counts_argument, default=[5, 8, 10, 12, 16, 20])
    compare.add_argument("--max-new-tokens", type=int, default=512)
    compare.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        for variant in ("smollm135", "smollm360", "smollm1b"):
            other = ROOT.parent.parent / variant
            if other != ROOT.parent and args.output_dir.resolve().is_relative_to(other.resolve()):
                raise ValueError("output directory belongs to another model experiment")
        if args.command == "prepare":
            if args.encoding == "lossless-v2":
                from .lossless_dataset import process
                result = process(args.output_dir, prepare=True, source_output=args.from_output)
            elif args.from_output:
                from .data import prepare_epoch7_experiment
                result = prepare_epoch7_experiment(args.output_dir, args.from_output)
            else:
                result = prepare(args.output_dir)
        elif args.command == "validate-splits":
            from .data import validate_epoch7_splits
            result = validate_epoch7_splits(args.output_dir)
        elif args.command == "compare-codecs":
            from .lossless_dataset import process
            if args.output_dir.resolve() == (ROOT / "output").resolve():
                args.output_dir = ROOT / "codec_benchmark_v2"
            report = process(args.output_dir, source_output=args.from_output)
            result = {"benchmark": str(args.output_dir / "codec_benchmark.json"),
                      "examples": report["example_count"],
                      "reduction_percent": report["reduction_percent"]}
        elif args.command == "benchmark-inference":
            from .profiling import benchmark_inference
            result = benchmark_inference(args.output_dir, args.report,
                                         args.partition, args.cap, args.counts,
                                         args.max_new_tokens, args.cpu)
        elif args.command == "train":
            from .model import train
            rotating = bool(args.init_adapter or args.resume_checkpoint or
                            args.examples_per_epoch is not None or args.save_steps is not None)
            result = train(args.output_dir, "rhythm",
                           epochs=args.epochs if args.epochs is not None else (None if rotating else 20),
                           learning_rate=args.learning_rate if args.learning_rate is not None else (None if rotating else 5e-5),
                           accumulation=args.accumulation, patience=args.patience,
                           seed=args.seed, cpu=args.cpu, dry_run=args.dry_run,
                           resume=args.resume, min_delta=args.min_delta,
                           progress_seconds=args.progress_seconds,
                           init_adapter=args.init_adapter,
                           examples_per_epoch=args.examples_per_epoch,
                           save_steps=args.save_steps,
                           resume_checkpoint=args.resume_checkpoint,
                           profile=args.profile)
        elif args.command == "overnight":
            result = overnight(args)
        elif args.command == "generate":
            from .inference import generate_file
            result = generate_file(args.input, args.output, args.target_note_count,
                                   args.output_dir, args.constrained, args.max_new_tokens,
                                   args.cpu, args.loop_ms, args.reference_track_id,
                                   args.target_track_id, args.preset, args.cd2)
        elif args.command == "evaluate":
            from .evaluation import evaluate_v3
            result = evaluate_v3(args.output_dir, args.constrained, args.partition,
                                 args.cap, args.counts, args.max_new_tokens, args.cpu, args.cd2,
                                 shared_prefix_cache=not args.no_shared_prefix_cache,
                                 profile=args.profile)
        else:
            from .evaluation import compare
            result = compare(args.output_dir, args.partition, args.cap,
                             args.counts, args.max_new_tokens, args.cpu)
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        if isinstance(error, RuntimeError) and any(marker in str(error).upper()
                                                   for marker in ("CUDA", "CUBLAS")):
            traceback.print_exc()
        print(f"V3: {error}", file=sys.stderr)
        return 2
