"""Controlled algorithmic reasoning tasks with deterministic answer oracles."""

from __future__ import annotations
from collections import defaultdict
import random, time
from .runtime import ask, client_for, parse_object, wilson, save_json
from .tokenization import get_tokenizer, pad_exact

TITLE = "LLM Complexity with NVIDIA NIM"
DEFAULT_CONFIG = {
    "seed": 42,
    "tasks": ["pointer", "arithmetic", "brackets"],
    "difficulties": [1, 3, 5],
    "lengths": [1024, 2048],
    "replicates": 2,
    "models": ["nvidia/nemotron-3.5-lightning-30b-a3b"],
    "thinking": [False],
    "base_url": "https://integrate.api.nvidia.com/v1",
    "max_requests": 100,
    "max_tokens": 1024,
    "temperature": 0,
    "tokenizer_model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
}


def oracle(task, payload):
    if task == "pointer":
        value = payload["start"]
        for _ in range(payload["steps"]):
            value = payload["links"][value]
        return value
    if task == "arithmetic":
        value = payload["start"]
        for operation, operand in payload["operations"]:
            value = (value + operand if operation == "add" else value * operand) % 97
        return str(value)
    if task == "brackets":
        stack = []
        for char in payload["sequence"]:
            if char in "([":
                stack.append(char)
            elif not stack or stack.pop() != {")": "(", "]": "["}[char]:
                return "invalid"
        return "valid" if not stack else "invalid"
    raise ValueError("Unknown task")


def make_problem(task, difficulty, seed):
    if not 1 <= difficulty <= 20:
        raise ValueError("difficulty must be 1..20")
    rng = random.Random(seed)
    if task == "pointer":
        # Number of displayed links stays fixed while requested chain depth changes.
        labels = [f"node_{i:02d}" for i in range(48)]
        shuffled = labels[:]
        rng.shuffle(shuffled)
        links = dict(zip(shuffled, shuffled[1:] + shuffled[:1]))
        payload = {"links": links, "start": shuffled[0], "steps": difficulty}
        displayed = list(links.items())
        rng.shuffle(displayed)
        prompt = (
            f"Follow exactly {difficulty} links from {payload['start']}. Links: "
            + ", ".join(f"{a}->{b}" for a, b in displayed)
        )
    elif task == "arithmetic":
        operations = [
            (rng.choice(["add", "multiply"]), rng.randint(2, 12)) for _ in range(20)
        ]
        payload = {"start": rng.randint(1, 30), "operations": operations[:difficulty]}
        prompt = (
            f"Start at {payload['start']}. Execute only the first {difficulty} operations, reducing modulo 97 after each. Operations: "
            + ", ".join(f"{op} {value}" for op, value in operations)
        )
    elif task == "brackets":
        # Nesting depth varies while the sequence always has 40 characters.
        opening = "".join(rng.choice("([") for _ in range(difficulty))
        closing = "".join({"(": ")", "[": "]"}[c] for c in reversed(opening))
        sequence = opening + closing + "()" * (20 - difficulty)
        if seed % 2:
            index = difficulty
            sequence = (
                sequence[:index]
                + ("]" if sequence[index] == ")" else ")")
                + sequence[index + 1 :]
            )
        payload = {"sequence": sequence}
        prompt = (
            "Is this bracket sequence correctly balanced and type-matched? Reply valid or invalid. Sequence: "
            + sequence
        )
    else:
        raise ValueError("Unknown task")
    return payload, prompt


def prepare(config, mode, output):
    tokenizer = get_tokenizer(mode, config)
    rows = []
    for task in config["tasks"]:
        for repetition in range(config["replicates"]):
            seed = config["seed"] + repetition
            for difficulty in config["difficulties"]:
                payload, problem = make_problem(task, difficulty, seed)
                if config.get("variants_path"):
                    from .designer import load_approved

                    templates = load_approved(config["variants_path"], "paraphrase")
                    problem = templates[repetition % len(templates)].replace(
                        "{TASK}", problem
                    )
                for length in config["lengths"]:
                    text = (
                        'Return one JSON object {"answer": "..."}. Ignore the padding after END TASK.\n'
                        + problem
                        + "\nEND TASK\nPadding:"
                    )
                    padded = pad_exact(text, length, tokenizer)
                    rows.append(
                        {
                            "id": f"{task}-r{repetition}-d{difficulty}-n{length}",
                            "task": task,
                            "difficulty": difficulty,
                            "length": length,
                            "seed": seed,
                            "payload": payload,
                            "prompt": padded,
                            "answer": oracle(task, payload),
                            "tokenizer": tokenizer.label,
                            "actual_length": len(tokenizer.encode(padded)),
                        }
                    )
    return rows


def run(config, rows, mode, output):
    client = client_for(config, output) if mode == "live" else None
    results = []
    if mode == "live" and any(
        m != "nvidia/nemotron-3.5-lightning-30b-a3b" for m in config["models"]
    ):
        raise ValueError("Validate a tokenizer mapping before changing the live model")
    for model in config["models"]:
        for thinking in config["thinking"]:
            for row in rows:
                started = time.perf_counter()
                error = None
                raw = ""
                predicted = ""
                try:
                    raw = (
                        '{"answer":'
                        + __import__("json").dumps(oracle(row["task"], row["payload"]))
                        + "}"
                        if mode == "fixture"
                        else ask(
                            client,
                            "Solve the supplied algorithmic task. Return a final JSON object.",
                            row["prompt"],
                            model,
                            config,
                            thinking,
                        )
                    )
                    predicted = str(parse_object(raw)["answer"]).strip()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                results.append(
                    {
                        **row,
                        "model": model,
                        "thinking": thinking,
                        "raw": raw,
                        "prediction": predicted,
                        "correct": predicted == row["answer"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": error,
                        "mode": mode,
                    }
                )
    return results


def evaluate(rows, config):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["model"],
                row["thinking"],
                row["task"],
                row["difficulty"],
                row["length"],
            )
        ].append(row)
    metrics = []
    for key, items in sorted(groups.items()):
        correct = sum(
            row["prediction"] == row["answer"] and not row.get("error") for row in items
        )
        metrics.append(
            dict(zip(["model", "thinking", "task", "difficulty", "length"], key))
            | {
                "n": len(items),
                "correct": correct,
                "accuracy": correct / len(items),
                "wilson95": wilson(correct, len(items)),
                "mean_seconds": sum(r["elapsed_seconds"] for r in items) / len(items),
                "errors": sum(bool(r.get("error")) for r in items),
            }
        )
    return {
        "conditions": metrics,
        "n_predictions": len(rows),
        "interpretation": "Fixture scores validate the oracle and harness only. Live metrics estimate task performance; they are not a complexity-theoretic lower bound.",
    }


def design(config, mode, output):
    from .designer import generate_variants

    return generate_variants(config, mode, output, "paraphrase")
