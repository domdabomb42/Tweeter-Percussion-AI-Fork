import argparse
import json
from pathlib import Path
import sys
from .data_loader import load_sessions, dataset_summary, split_sessions
from .prepare import prepare, load_prepared, tokenizer_for_model
from .train import train, evaluate_original, cpu_smoke
from .evaluate import evaluate, evaluate_base_features, compare, select_feature_queries, complete_value
from .report import render as render_report


BASE = Path(__file__).resolve().parents[3]
DEFAULT_SESSIONS = BASE / "L2Ork-Tweeter" / "sessions"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "training_output"


def main(argv=None):
    parser = argparse.ArgumentParser(description="SmolLM2 causal LM experiment on CustomExtractor sessions")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare", "baseline", "train", "evaluate", "compare", "complete", "smoke", "report"):
        sub.add_parser(name)
    for name in ("inspect", "prepare"):
        cmd = sub.choices[name]
        cmd.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
        cmd.add_argument("--seed", type=int, default=415)
    sub.choices["prepare"].add_argument("--max-length", type=int, default=4096)
    sub.choices["prepare"].add_argument("--local-only", action="store_true")
    sub.choices["baseline"].add_argument("--cpu", action="store_true")
    cmd = sub.choices["train"]
    cmd.add_argument("--dry-run", action="store_true")
    cmd.add_argument("--resume", action="store_true", help="continue from the saved best adapter; --epochs is the total count")
    cmd.add_argument("--cpu", action="store_true")
    cmd.add_argument("--epochs", type=int, default=5)
    cmd.add_argument("--learning-rate", type=float, default=1e-4)
    cmd.add_argument("--gradient-accumulation", type=int, default=8)
    cmd.add_argument("--patience", type=int, default=2)
    cmd.add_argument("--seed", type=int, default=415)
    cmd = sub.choices["evaluate"]
    cmd.add_argument("--cpu", action="store_true")
    cmd.add_argument("--cap-per-feature", type=int, default=6)
    cmd.add_argument("--seed", type=int, default=415)
    cmd = sub.choices["complete"]
    cmd.add_argument("--partition", choices=("validation", "test"), default="validation")
    cmd.add_argument("--feature", help="Dotted feature name, e.g. synth.carrier or note.pitch")
    cmd.add_argument("--index", type=int, default=0)
    cmd.add_argument("--original", action="store_true")
    cmd.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    root = args.output
    dataset = root / "dataset"
    evaluation = root / "evaluation"
    checkpoints = root / "checkpoints"
    try:
        if args.command == "inspect":
            sessions = load_sessions(args.sessions)
            report = dataset_summary(sessions)
            _, manifest = split_sessions(sessions, args.seed)
            tokenizer = tokenizer_for_model()
            from .serializer import serialize
            counts = {s.name: len(tokenizer.encode(serialize(s.composition), add_special_tokens=False)) + 1 for s in sessions}
            report["tokens_per_composition"] = counts
            report["total_available_tokens"] = sum(counts.values())
            report["split"] = manifest
        elif args.command == "prepare":
            report = prepare(args.sessions, dataset, args.seed, args.max_length, args.local_only)
        elif args.command == "baseline":
            report = evaluate_original(dataset, evaluation / "original_validation.json", args.cpu)
        elif args.command == "train":
            report = train(dataset, checkpoints, args.epochs, args.learning_rate,
                           args.gradient_accumulation, args.patience, args.seed, args.cpu, args.dry_run,
                           args.resume)
        elif args.command == "smoke":
            report = cpu_smoke(dataset)
        elif args.command == "report":
            report = {"path": str(render_report(root, Path(__file__).resolve().parents[1] / "REPORT.md"))}
        elif args.command == "evaluate":
            report = evaluate(dataset, checkpoints / "best_adapter", evaluation / "fine_tuned.json",
                              cap_per_feature=args.cap_per_feature, cpu=args.cpu, seed=args.seed)
            # Test the original only after the checkpoint has already been selected.
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            baseline = evaluate_base_features(dataset, evaluation / "original_full.json",
                                              cap_per_feature=args.cap_per_feature, cpu=args.cpu, seed=args.seed)
            report = {"fine_tuned": report, "original": baseline}
        elif args.command == "compare":
            report = compare(evaluation / "original_full.json", evaluation / "fine_tuned.json",
                             checkpoints / "history.json", evaluation / "comparison.json")
        else:
            from .train import load_model
            rows = load_prepared(dataset, args.partition)
            queries = select_feature_queries(rows, cap=100000)
            if args.feature:
                queries = [q for q in queries if q["name"] == args.feature]
            if not 0 <= args.index < len(queries):
                raise ValueError(f"completion index {args.index} out of {len(queries)} matching queries")
            q = queries[args.index]
            model, device, dtype = load_model(None if args.original else checkpoints / "best_adapter", args.cpu)
            prediction, fragment = complete_value(model, tokenizer_for_model(local_only=True), device, q["prefix"])
            report = {"source": q["source"], "feature": q["name"], "path": q["path"],
                      "preceding_context_tail": q["prefix"][-800:], "expected": q["value"],
                      "prediction": prediction, "generated_fragment": fragment[:200],
                      "model": "original" if args.original else "fine_tuned"}
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"custom_training: {error}", file=sys.stderr)
        return 2
