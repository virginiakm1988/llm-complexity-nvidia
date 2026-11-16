"""Real NeMo Data Designer DAG for reviewed task and distractor templates."""

from __future__ import annotations
import importlib.metadata, json, os
from pathlib import Path
from .runtime import save_json, save_jsonl, read_jsonl

SDK_VERSION = "0.9.3"


def validate_template(template, kind):
    if not isinstance(template, str) or not 15 <= len(template) <= 1600:
        return False
    if kind == "paraphrase":
        meta_instruction = any(
            word in template.lower()
            for word in (
                "placeholder",
                "schema",
                "code fence",
                "task wrapper",
                "write an english",
            )
        )
        return (
            template.count("{TASK}") == 1
            and "{ANSWER}" not in template
            and not meta_instruction
        )
    return (
        template.count("{ENTITY}") == 1
        and template.count("{WRONG}") == 1
        and "{ANSWER}" not in template
        and "unverified" in template.lower()
    )


def load_approved(path, kind):
    rows = read_jsonl(path)
    templates = [
        r["template"]
        for r in rows
        if r.get("kind") == kind
        and r.get("review_status") == "approved"
        and validate_template(r.get("template"), kind)
    ]
    if not templates:
        raise ValueError("No valid human-approved templates for this workflow")
    return templates


def build_designer(config, kind, artifact_path):
    import pandas as pd
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    if importlib.metadata.version("data-designer") != SDK_VERSION:
        raise RuntimeError(
            "Use data-designer==0.9.3 to preserve the reviewed request bound"
        )
    seeds = [
        {"record_id": f"{kind}-{i}", "style": style}
        for i, style in enumerate(["concise", "formal", "plain language"])
    ]
    model = dd.ModelConfig(
        alias="template-writer",
        model=config.get("model", config.get("models", [None])[0]),
        provider="nvidia",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            max_parallel_requests=1,
            timeout=90,
            temperature=0.7,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model])
    builder.with_seed_dataset(
        dd.DataFrameSeedSource(df=pd.DataFrame(seeds)),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )
    task = (
        "Produce a short {{ style }} sentence that asks a reader to solve a supplied task and return JSON. The sentence must contain {TASK} exactly once. Example template: Solve {TASK} and return JSON with an answer field. Write only the requested sentence as the template value, not these generation instructions. Do not solve a task or mention placeholders, templates, schemas or code fences inside the template value."
        if kind == "paraphrase"
        else "Produce ONE short {{ style }} sentence about an unverified delivery-code record, at most 110 characters. Example: Unverified draft: {ENTITY} has delivery code {WRONG}. Change the wording for the requested style. Keep each brace-delimited placeholder exactly once in the output sentence. Never repeat the entity. No heading, explanation, additional facts, or extra sentence. Return this sentence as the template field value."
    )
    task += " Return the requested object inside a fenced JSON block (```json followed by the object and closing ```), matching the supplied JSON schema."
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="generated",
            model_alias="template-writer",
            prompt=task,
            output_format={
                "type": "object",
                "properties": {"template": {"type": "string"}},
                "required": ["template"],
                "additionalProperties": False,
            },
        )
    )
    designer = DataDesigner(
        artifact_path=str(artifact_path),
        model_providers=[
            dd.ModelProvider(
                name="nvidia",
                endpoint=os.environ.get("NVIDIA_BASE_URL", config["base_url"]),
                provider_type="openai",
                api_key="NVIDIA_API_KEY",
            )
        ],
    )
    designer.set_run_config(
        dd.RunConfig(
            buffer_size=3,
            max_concurrent_row_groups=1,
            max_in_flight_tasks=1,
            max_conversation_restarts=0,
            max_conversation_correction_steps=0,
            otel_metrics_port=None,
        )
    )
    return designer, builder, seeds


def generate_variants(config, mode, output, kind):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "designer-attempt.json"
    # Exclusive marker persists even on errors: re-running cannot silently spend again.
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "kind": kind,
                "mode": mode,
                "status": "started",
                "sdk": SDK_VERSION,
                "reserved_http_attempts": 12 if mode == "live" else 0,
            },
            handle,
        )
    if mode == "fixture":
        templates = (
            [
                "Solve this task: {TASK}. Return JSON with the answer key.",
                "Please complete {TASK}. Reply with a JSON answer.",
                "Analyze the task {TASK}. Give the answer as JSON.",
            ]
            if kind == "paraphrase"
            else [
                "Unverified draft: {ENTITY} uses delivery code {WRONG}.",
                "An unverified note assigns {WRONG} to {ENTITY}.",
                "Unverified delivery record for {ENTITY}: {WRONG}.",
            ]
        )
        generated = [
            {"record_id": f"{kind}-{i}", "generated": {"template": t}}
            for i, t in enumerate(templates)
        ]
    else:
        if not os.environ.get("NVIDIA_API_KEY"):
            raise RuntimeError("Set NVIDIA_API_KEY securely before generation")
        if config.get("designer_max_requests", 12) < 12:
            raise ValueError(
                "Three rows reserve 12 HTTP attempts (SDK retry upper bound)"
            )
        designer, builder, seeds = build_designer(config, kind, output / "sdk")
        designer.validate(builder)
        results = designer.create(
            builder, num_records=len(seeds), dataset_name="reviewed_templates"
        )
        generated = json.loads(
            results.load_dataset().to_json(orient="records", force_ascii=False)
        )
    rows = []
    for row in generated:
        value = row.get("generated", {})
        if isinstance(value, str):
            value = json.loads(value)
        template = value.get("template", "")
        rows.append(
            {
                "record_id": row.get("record_id"),
                "kind": kind,
                "template": template,
                "automatic_validation": validate_template(template, kind),
                "review_status": "pending",
                "source": (
                    "authored_fixture"
                    if mode == "fixture"
                    else "nemo_data_designer_nim"
                ),
            }
        )
    save_jsonl(output / "templates.jsonl", rows)
    save_json(
        marker,
        {
            "kind": kind,
            "mode": mode,
            "status": "completed",
            "sdk": SDK_VERSION,
            "rows": len(rows),
            "reserved_http_attempts": 12 if mode == "live" else 0,
            "note": "Reserved bound, not a measured count. Inspect SDK artifacts for actual attempts.",
        },
    )
    return {
        "generated": len(rows),
        "automatically_valid": sum(r["automatic_validation"] for r in rows),
        "review": "Review template meaning and mark accepted rows approved before setting variants_path",
        "path": str(output / "templates.jsonl"),
    }
