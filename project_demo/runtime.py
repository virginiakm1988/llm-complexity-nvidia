"""Small shared evidence utilities; no live-to-fixture fallback."""

from __future__ import annotations
import hashlib, json, math, os, platform
from datetime import datetime, timezone
from pathlib import Path
from .nvidia_demo_support import NIMClient


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def parse_object(text):
    """Accept one JSON object or a fenced object; never mine a letter from prose."""
    text = str(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, end = decoder.raw_decode(text[index:])
                if isinstance(value, dict) and not text[index + end :].strip():
                    return value
            except ValueError:
                pass
    raise ValueError("Model response has no final JSON object")


def wilson(correct, total):
    if not total:
        return [None, None]
    z, p = 1.96, correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0, center - half), min(1, center + half)]


def client_for(config, directory):
    return NIMClient(
        base_url=os.environ.get("NVIDIA_BASE_URL", config["base_url"]),
        output_dir=Path(directory) / "api",
        max_requests=config.get("max_requests", 100),
        timeout=config.get("timeout", 90),
    )


def ask(client, system, prompt, model, config, thinking=False):
    return client.chat_text(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=model,
        temperature=config.get("temperature", 0),
        max_tokens=config.get("max_tokens", 1024),
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )


def manifest(config, mode, rows, status="completed"):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": status,
        "result_kind": (
            "prepared_inputs_no_model_inference"
            if status == "prepared"
            else (
                "authored_fixture_no_model_inference"
                if mode == "fixture"
                else "actual_nim_calls"
            )
        ),
        "python": platform.python_version(),
        "config": config,
        "dataset_sha256": content_hash(rows),
        "n_rows": len(rows),
    }
