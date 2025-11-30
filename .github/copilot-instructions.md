# GitHub Copilot Instructions for Stateful Projan (TrojanZoo)

This document provides context and guidelines for AI agents working on the Stateful Projan codebase, built on the TrojanZoo framework.

## 1. Architecture Overview

The project is split into two main packages:
- **`trojanzoo`**: The core library providing base classes, environment management, datasets, models, and training utilities. It is domain-agnostic.
- **`trojanvision`**: An extension of `trojanzoo` specifically for computer vision tasks. It contains implementations of attacks (like Projan), defenses, and marks (triggers).

### Key Components
- **Environment (`trojanzoo.environ.Env`)**: A singleton (`env`) that manages global state like device (CPU/GPU), random seeds, and configuration. Accessed via `trojanzoo.environ.env`.
- **Factory Pattern**: Objects are rarely instantiated directly. Instead, use `create(**kwargs)` factory functions found in each module (e.g., `trojanvision.models.create`, `trojanvision.attacks.create`).
- **Distributed Configuration**: Argument parsing is decentralized. Each module (`environ`, `datasets`, `models`, `trainer`, `attacks`) has an `add_argument(parser)` class method to register its own CLI arguments.

## 2. Developer Workflows

### Setup
- Install dependencies: `pip install -r requirements.txt`
- Install in editable mode: `pip install -e .`

### Running Attacks
Attacks are typically run via scripts in `examples/`. The standard pattern is:
```bash
python examples/backdoor_attack.py --attack <attack_name> --dataset <dataset_name> --model <model_name> [args...]
```
Example (Stateful Projan):
```bash
python examples/backdoor_attack.py --attack stateful_prob --dataset mnist --model lenet --epochs 50
```

### Testing
- The project uses custom test scripts in `tests/` rather than a standard runner like `pytest`.
- Run tests manually: `python tests/prob_attack_test.py <device>` (e.g., `cpu` or `cuda`).

### Debugging
- Use `--verbose <level>` (e.g., `--verbose 1`) to increase logging output.
- TensorBoard logging is integrated. Logs are saved to `results/<experiment>/tensorboard_logs`.
- XAI (Explainable AI) features (Captum) are available in `stateful_prob.py` via `--xai_enable`.
- Do not make .md files as a review after every change. Only if requested.
## 3. Coding Conventions

### Argument Parsing & Configuration
- **Do not** hardcode parameters. Add them to the `add_argument` method of the relevant class.
- Access arguments via `kwargs` in `__init__` or `create` methods.
- Use `trojanzoo.configs` for default configuration values.

### Device Management
- Always use `env['device']` to get the current torch device.
- Example: `tensor.to(env['device'])`.

### Logging
- Use `trojanzoo.utils.output.prints` for console output.
- Use `trojanzoo.utils.logger.MetricLogger` for tracking training metrics.

### Type Hinting
- Use standard Python type hints (e.g., `list[Watermark]`, `torch.Tensor`).
- `trojanzoo` uses `typing.TYPE_CHECKING` to avoid circular imports for type hints.

## 4. Project-Specific Patterns

### Attack Implementation
- Attacks inherit from `trojanvision.attacks.BackdoorAttack` (or subclasses like `BadNet`).
- The `attack()` method controls the training loop. It can override the default trainer logic.
- **Stateful Projan** (`Prob` class in `trojanvision/attacks/backdoor/prob/stateful_prob.py`) implements a custom training loop with partitioner networks and multiple loss terms.

### Marks (Triggers)
- Triggers are defined as `Watermark` objects in `trojanvision.marks`.
- Multiple triggers are handled by passing lists of marks (e.g., `marks=[mark1, mark2]`).

### Model Definitions
- Models are defined in `trojanvision/models` (e.g., `resnet.py`, `lenet.py`).
- Custom models should implement `get_features()` if they are to be used with the Stateful Projan partitioner.

## 5. Integration Points
- **PyTorch**: The core deep learning framework.
- **TensorBoard**: For visualization of loss and accuracy.
- **Captum**: For model interpretability (Integrated Gradients, Grad-CAM).
