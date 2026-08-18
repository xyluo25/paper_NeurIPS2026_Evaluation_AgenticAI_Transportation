# Agentic RealTwin Evaluation Scripts

This folder contains the reproducible benchmark replay used by
`paper_eval_agents/eval_agents.tex`.

Run from the repository root:

```powershell
& C:\Users\xh8\AppData\Local\miniforge3\envs\rt_deepagent\python.exe .\script_eval_agents\evaluate_agentic_realtwin.py
```

Linux/macOS with the project conda environment:

```bash
conda run -n rt_deepagent python script_eval_agents/evaluate_agentic_realtwin.py
```

The script writes generated outputs to `script_eval_agents/outputs/`:

- `benchmark_tasks.json`: the 40-task benchmark suite.
- `task_results.csv`: one row per task and system configuration.
- `summary_results.csv`: aggregated results by system.
- `agentic_realtwin_difficulty.csv`: Agentic RealTwin results by difficulty.
- `model_task_results.csv`: one row per task and selected Agentic RealTwin model.
- `model_summary_results.csv`: aggregated results for each selected model.
- `model_provider_status.csv`: non-secret provider/API-key readiness metadata.
- `cross_validation_ranking.csv`: final ranked cross-validation across selected models.
- `cross_validation_results.csv`: pairwise model cross-validation deltas.
- `models/<model_id>/`: per-model task, summary, difficulty, and metadata outputs.
- `summary_results.tex`: LaTeX tables inserted into the manuscript.
- `reproducibility_manifest.json`: source hashes and deterministic-run metadata.

The benchmark uses `deepeval` for agentic evaluation. Tool-use quality is scored
with `deepeval.metrics.ToolCorrectnessMetric`, and end-to-end completion is
scored with a deterministic `deepeval.metrics.BaseMetric` subclass over
task-replay metadata.

The benchmark intentionally avoids live LLM calls, network requests, random
seeds, and simulator execution by default. It parses the Agentic RealTwin source
tree to recover the exposed tool registry and human-in-the-loop tool list, then
scores deterministic task traces against those implementation-derived
capabilities. Selected models are read from `selected_models.txt`; `gpt-5.5` is
included as the current OpenAI implementation, Claude/Gemini models use their
provider keys, and all other selected models are routed as Ollama cloud models
using `OLLAMA_API_KEY`.
