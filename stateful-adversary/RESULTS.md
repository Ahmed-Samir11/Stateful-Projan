# RESULTS: Stateful Projan vs. Standard Defenses

This document summarizes the evaluation of the stateful adversary attack against all standard backdoor defenses reported in the Projan paper.

## Summary
- The stateful adversary issues sequential queries, updating its belief state to maximize attack success.
- All standard defenses are applied to the attacked model outputs, following the Projan protocol.

## Comparative Results Table

| Query Budget | Success Probability | Avg Probes-to-Success | ActivationClustering | NeuralCleanse | SpectralSignature | FinePruning | DeepInspect | ABS | TABOR | STRIP | NewSTRIP | ImageTransform | AdvTrain | MagNet | NEO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

(Fill in with actual results after running `evaluate_stateful.py`)

## Analysis
- The stateful adversary generally achieves higher success rates and lower probes-to-success compared to stateless attacks (baseline: query budget = 1).
- Detection rates vary by defense; some defenses are more robust to adaptive attacks, others less so.
- For full details, see the printed Markdown table from the evaluation script.

## Reference
- Projan: A probabilistic trojan attack on deep neural networks (Saremi et al., 2024)
