# LLM Complexity minimum demo playbook

**Audience:** research engineers and academic collaborators. **Goal:** run the core intervention and inspect reproducible evidence, not reproduce full-paper headline scores.

1. Read the paper/project links in README and the limitations. Install the indicated extra in a fresh environment.
2. Run the fixture quickstart. Inspect three representative tasks/scenarios and explain why the result is an authored fixture.
3. Change one experimental factor at a time. Keep the data revision, seed, prompt and scorer fixed for comparisons.
4. Configure `NVIDIA_API_KEY` privately. Check the model and endpoint, request budget, dataset size, and output directory.
5. Run the live-smoke configuration. Inspect every failure before averaging. Missing or malformed answers count as wrong when the metric defines correctness; no-answer grading stays explicitly ungraded.
6. Recompute metrics with `evaluate`. Compare per-condition results and uncertainty rather than selecting only successful examples.
7. Archive configuration, input hash, raw responses, model name, tokenizer revision where used, and metric output. Keep secrets out of shared files.

**Expected evidence:** `run_manifest.json`, `dataset.jsonl`, `predictions.jsonl`, `metrics.json`, and live API attempt records. A successful API call alone is not evidence of research quality.

**Three discussion prompts:** Which variable changed? What stayed fixed? Which claim is supported by these measured outputs?

The tasks are synthetic. Success or failure does not prove a complexity class limitation. Small runs have wide uncertainty intervals; the fixed tokenizer controls body tokens, not hidden server chat-template overhead.
