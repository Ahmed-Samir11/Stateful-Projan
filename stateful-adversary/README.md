# Stateful Projan Evaluation

This module implements a stateful adversary attack and replicates the defense evaluation protocol from the Projan paper, using all standard backdoor detection algorithms.

## Usage

1. **Install dependencies:** Ensure all required packages from the Projan codebase are installed.
2. **Prepare models and triggers:** Place your trained backdoored models and trigger definitions in accessible locations.
3. **Run evaluation:**

```bash
python stateful-adversary/src/evaluate_stateful.py
```

## What it does
- Runs a stateful adversary attack with configurable query budgets.
- Evaluates the effectiveness of all standard defenses (Activation Clustering, Neural Cleanse, Spectral Signature, Fine-Pruning, DeepInspect, ABS, TABOR, STRIP, NewSTRIP, ImageTransform, AdvTrain, MagNet, NEO).
- Outputs a Markdown table of success probabilities, probes-to-success, and detection rates for each defense.

## Customization
- Edit `evaluate_stateful.py` to specify model paths, dataset, and trigger logic as needed for your experiments.

## Reference
- Projan: A probabilistic trojan attack on deep neural networks (Saremi et al., 2024)
# Stateful Adversary Project

This project implements a stateful adversary within the Projan codebase. The adversary is designed to manage its belief state and query budget while executing attacks based on a defined threat model. The implementation includes various components for analysis, evaluation, and utility functions.

## Project Structure

- **src/**: Contains the main implementation of the stateful adversary and related functionalities.
  - **adversary.py**: Implementation of the Adversary class, which manages the belief state and attack logic.
  - **analysis.py**: Functions for analyzing the performance of the adversary.
  - **implementation_plan.md**: Detailed implementation plan for the stateful adversary.
  - **code_modifications.md**: Documentation of changes made to the Projan codebase.
  - **evaluation/**: Contains the evaluation framework and metrics calculation.
    - **framework.py**: Implements the evaluation framework for testing the adversary.
    - **metrics.py**: Functions for calculating and reporting key metrics.
  - **utils.py**: Utility functions supporting the adversary and evaluation framework.

- **tests/**: Contains unit tests for the stateful adversary implementation.
  - **test_adversary.py**: Unit tests ensuring core functionalities work as expected.

- **README.md**: Overview of the project, setup instructions, and evaluation script execution.

- **requirements.txt**: Lists necessary Python packages and dependencies.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stateful-adversary
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the evaluation script:
   ```
   python src/evaluation/framework.py
   ```

## Usage

To use the stateful adversary, instantiate the `Adversary` class from `src/adversary.py` and call the relevant methods to initialize the belief state, update it, and run attacks based on the threat model.

## Evaluation

The evaluation framework allows for running simulations and logging results. Key metrics can be calculated using the functions provided in `src/evaluation/metrics.py`.

## Contribution

Contributions to enhance the functionality or performance of the stateful adversary are welcome. Please submit a pull request with a description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.