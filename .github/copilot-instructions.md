# GitHub Copilot Instructions for Stateful Projan (TrojanZoo)

This document provides concise, actionable guidance for GitHub Copilot and contributors working on the Stateful Projan codebase (a TrojanZoo extension for backdoor/backdoor-research and defenses).

**Use this file to know where to find key components, how to run experiments, and which conventions to follow when making changes.**

## Project Overview

- Elevator pitch: Stateful Projan is a research-oriented framework for implementing and evaluating stateful backdoor attacks and defenses on vision models. It targets researchers and engineers working on adversarial ML, backdoor attacks, and defenses who need reproducible training, evaluation, and interpretability tools.
- Primary goals: reproducible attack training, modular marks/triggers, partitioner-based stateful logic, and easy experiment orchestration via CLI scripts in `examples/`.

## Tech Stack

- Language: Python 3.8+ (check `pyproject.toml` / `requirements.txt` for exact constraints)
- Deep learning: PyTorch (versions in `requirements.txt` -- typically 1.10+ / 1.11+)
- Utilities: torchvision, tensorboard
- Optional: Captum (for XAI), NumPy, tqdm
- Packaging: setuptools (`setup.py`, `pyproject.toml`)

Note: Inspect `requirements.txt` and `pyproject.toml` for exact pinned versions when reproducing environments.

## Coding Guidelines

- Naming: use `snake_case` for functions and variables, `CamelCase` for classes. Prefer descriptive names over one-letter variables.
- Indentation: 4 spaces per indent. Do not use tabs.
- Type hints: Add type annotations for public functions and methods where practical (e.g., argument and return types). Use `typing.TYPE_CHECKING` to avoid import cycles.
- Logging/output: Use `trojanzoo.utils.output.prints` for CLI messages and `trojanzoo.utils.logger.MetricLogger` for training metric logging. Avoid raw `print()` calls in library code.
- Batch-norm handling: When disabling batch norm in experiments, use provided helpers `_disable_batch_norm` / `_enable_batch_norm` and set momentum via helper `_set_batchnorm_momentum`.
- Determinism: Use the `_set_deterministic_state(seed)` helper when reproducibility is required. Default seed used in the repo is `1228` in some attack implementations.
- Argument parsing: Add CLI flags via the `add_argument(group)` class method found in attacks, models, datasets, and environment modules.
- Tests: The repo uses custom test scripts under `tests/`. Run them manually. Keep tests small and focused.

Best practices
- Keep changes small and focused. Follow existing project patterns (factory `create()` functions, `env` for configuration, `MetricLogger` for progress reporting).
- When modifying logging/printing behavior, prefer changes in `trojanzoo.utils.output` to keep callers unchanged.

## Project Structure (key folders)

- `trojanzoo/` — Core library: environment management, trainer helpers, configs, utils.
	- See `trojanzoo/environ.py` for the global `env` singleton.
	- See `trojanzoo/utils/output.py` for CLI printing helpers and `trojanzoo/utils/logger.py` for `MetricLogger`.
- `trojanvision/` — Computer-vision specific code and attacks.
	- `trojanvision/attacks/` — Attack implementations.
	- `trojanvision/attacks/backdoor/prob/stateful_prob.py` — Stateful Projan implementation to inspect for training loops and logging.
	- `trojanvision/models/` — Models and helpers (e.g., `Partitioner` referenced by stateful attacks).
- `examples/` — Example scripts for running attacks and evaluations (e.g., `examples/backdoor_attack.py`). Use these as canonical entrypoints for experiments.
- `tests/` — Small scripts for validating behaviors (e.g., `tests/prob_attack_test.py`).
- `data/` — Dataset and model artifacts used by examples and tests.
- `docs/` — Documentation sources.
- `scripts/` — Utility scripts (Kaggle setup, evaluation helpers).
- `IBAU/`, `NAD/` — Experimental and legacy code; review before changing.

Files to reference often
- `examples/backdoor_attack.py` — entrypoint for many experiments.
- `trojanvision/attacks/backdoor/prob/stateful_prob.py` — main attack implementation (training loop, partitioner creation, saving, validation, XAI hooks).
- `trojanzoo/utils/output.py` — printing helpers (`prints`, `output_iter`, `ansi` helpers).
- `trojanzoo/utils/logger.py` — `MetricLogger`, `SmoothedValue`, and `log_every` helpers.
- `requirements.txt`, `pyproject.toml` — dependency and environment details.

## External Resources

- Issue tracking: use your organization’s issue tracker (e.g., Jira / GitHub Issues) to file reproducible bug reports specifying dataset, model, commandline, and exact commit.
- Knowledge Base: internal wiki or ServiceNow (link to org KB if available in your org).
- CI: configure CI to run critical scripts (smoke tests using `tests/` scripts) and linting.
- Visualization: use TensorBoard for losses and metrics (`tensorboard --logdir data/attack/.../tensorboard_logs`).

## Running & Debugging (quick reference)

Run a standard example (adjust flags as needed):

```bash
python examples/backdoor_attack.py --attack stateful_prob --dataset mnist --model net \
	--epoch 50 --batch_size 128 --lr 0.001 --mark_path square_white.png --tqdm --color --save
```

Run a simple test script:

```bash
python tests/prob_attack_test.py cpu
```

## Investigation checklist: progress/printing helpers (why iterations print on new lines)

Follow these steps when investigating or fixing in-place progress output:

1. Inspect `trojanzoo/utils/logger.py` for `MetricLogger` and `log_every`:
	 - Confirm whether `log_every` wraps iterables with `tqdm` only when `tqdm` is enabled in `env`.
2. Inspect `trojanzoo/utils/output.py` for `prints`, `output_iter`, `ansi`, and `get_ansi_len`:
	 - Confirm `prints` calls Python `print()` (which emits newlines) and does not use `tqdm.write()`.
3. Search repo for `--tqdm` flag registration and for references to `env['tqdm']` to see how it is enabled from CLI (commonly added in environ/arg parsing).
4. Trace call sites that emit per-iteration logging lines (for example `trojanvision/attacks/backdoor/prob/stateful_prob.py:train`) and confirm whether they rely on `MetricLogger.log_every` for in-place updates.
5. Root-cause hypothesis (common): If `env['tqdm']` is False, `MetricLogger` will not wrap the iterator with `tqdm` and `prints()` will write new lines repeatedly; fixes either enable `tqdm` in the environment or make `prints()` use `tqdm.write()` when a bar is active.

Recommended minimal fixes:
- Option A (preferred, low-risk): Ensure callers enable `--tqdm` for interactive runs (CLI flag). Update documentation and examples to include `--tqdm` where an in-place bar is desired.
- Option B (library change): Update `trojanzoo/utils/output.py::prints` to call `tqdm.write()` instead of `print()` when a `tqdm` bar is active (detect via `hasattr(sys.stdout, 'write')` and `tqdm` state). This is a central, single-file change that prevents bar-breaks without changing callers.

Files to examine first for this issue:
- `trojanzoo/utils/logger.py` (MetricLogger, log_every)
- `trojanzoo/utils/output.py` (prints, output_iter)
- `examples/backdoor_attack.py` (entrypoint that passes `--tqdm`)
- `trojanvision/attacks/backdoor/prob/stateful_prob.py` (training loop that logs per-iter lines)

## Contribution & PR guidelines

- Keep PRs small and focused. Include a reproducible command-line example and expected vs actual output.
- Run the relevant `tests/` script locally and attach TensorBoard screenshots when reporting metric changes.
