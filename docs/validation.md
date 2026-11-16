# Execution and validation

Validation date: 2026-11-16. Separate fixture, CPU preparation and provider execution evidence are retained.

- 8 offline unit tests passed; see [test output](../results/test-output.txt).
- Both interactive notebooks executed without cell errors using Python 3.12, ipywidgets 8.1.7 and nbclient 0.10.2. The saved quickstart uses explicit fixture mode; output is not presented as model inference. See [validation record](../results/validation.json).
- The package is independently installable; use one environment per repository because all teaching packages use the same import name `project_demo`.
- Live-mode CPU preparation passed with the real NVIDIA Lightning tokenizer pinned at `d468880b6ad3c6e0d21377ce7242adaea4cc884d`; file SHA-256 `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7`. This step makes no model calls and is labeled prepared inputs.
- Real Data Designer 0.9.3 + Lightning template generation completed: 3 rows, 3 structurally valid candidates (2 unique valid templates). All candidates remain pending human review. See [actual SDK output](../results/designer-live-v2/templates.jsonl); earlier failed checks are retained in results/designer-live. These generated candidates were not used in the original NIM experiment, which predates their review.

## Real NIM small run

[Saved provider run](../results/live-smoke/run_manifest.json): 12 real requests, 8 correct answers, no transport/parser errors, using `nvidia/nemotron-3.5-lightning-30b-a3b`. The experiment crosses three task families, two difficulty levels and two body token lengths. Each condition has one sample; these numbers are teaching evidence with substantial uncertainty, not a benchmark ranking or complexity proof. Per-case outputs, timings and provider response records are retained.

## Remaining validation scope

Hosted NIM inference does not require a local GPU. No GPU training is part of these three repositories. SDK compatibility on another account/runtime must be confirmed from real executions. Failures are retained and never replaced with fixtures. Independent review is recorded separately in `REVIEW.md`.

Python wheel build passed; the distribution includes its license files.
