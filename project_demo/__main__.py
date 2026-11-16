"""Consistent CLI. Run `python -m project_demo --help`."""

import argparse, json, sys
from pathlib import Path
from . import core
from .runtime import save_json, save_jsonl, read_jsonl, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=core.TITLE)
    parser.add_argument("command", choices=["prepare", "demo", "evaluate", "design"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    config = dict(core.DEFAULT_CONFIG)
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    output = args.output_dir
    existed_before = (output / "run_manifest.json").exists()
    try:
        if args.command == "evaluate":
            rows = read_jsonl(output / "predictions.jsonl")
            meta = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            result = core.evaluate(rows, meta["config"])
            result["result_kind"] = meta["result_kind"]
            save_json(output / "metrics.json", result)
        elif args.command == "design":
            result = core.design(config, args.mode, output)
        else:
            if (output / "run_manifest.json").exists():
                raise FileExistsError(
                    "Choose a new --output-dir; completed run evidence is never overwritten"
                )
            output.mkdir(parents=True, exist_ok=True)
            rows = core.prepare(config, args.mode, output)
            save_jsonl(output / "dataset.jsonl", rows)
            meta = manifest(config, args.mode, rows, "prepared")
            save_json(output / "run_manifest.json", meta)
            if args.command == "demo":
                predictions = core.run(config, rows, args.mode, output)
                save_jsonl(output / "predictions.jsonl", predictions)
                meta["result_kind"] = (
                    "authored_fixture_no_model_inference"
                    if args.mode == "fixture"
                    else "actual_nim_calls"
                )
                result = core.evaluate(predictions, config)
                result["result_kind"] = meta["result_kind"]
                save_json(output / "metrics.json", result)
                meta["status"] = (
                    "completed_with_errors"
                    if any(r.get("error") or r.get("judge_errors") for r in predictions)
                    else "completed"
                )
                save_json(output / "run_manifest.json", meta)
            else:
                result = {"prepared": len(rows), "result_kind": meta["result_kind"]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        if not existed_before:
            save_json(
                output / "failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "mode": args.mode,
                    "command": args.command,
                    "no_fixture_fallback": True,
                },
            )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
