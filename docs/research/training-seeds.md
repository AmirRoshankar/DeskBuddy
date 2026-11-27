# Three-training-seed study

Three fresh 20,000-step runs use the same synthetic dataset, splits, width, batch size, learning-rate schedule and objectives. Evaluation uses the same 500 conditions and sampling seed 1729.

| Planner | Valid, mean ± sample SD | Blocked-route valid, mean ± sample SD |
| --- | ---: | ---: |
| MLP, one candidate | 69.3% ± 0.8 pp | 37.4% ± 1.5 pp |
| FM, 8 NFE | 72.3% ± 0.8 pp | 45.2% ± 2.0 pp |
| iMF, 1 NFE | 85.3% ± 0.2 pp | 70.7% ± 0.6 pp |
| iMF, four candidates | 97.7% ± 0.1 pp | 95.4% ± 0.2 pp |

## Individual runs

| Seed | MLP valid | iMF valid | iMF blocked-route valid | iMF, four candidates |
| --- | ---: | ---: | ---: | ---: |
| 20261125 | 68.6% | 85.6% | 71.4% | 97.8% |
| 20261126 | 69.0% | 85.2% | 70.2% | 97.6% |
| 20261127 | 70.2% | 85.2% | 70.6% | 97.8% |

The spread is a sample standard deviation over three training runs, not a 95% confidence interval. The original five-planner table and iMF-minus-MF bootstrap remain separate: these new runs omit the retired MF training baseline. This supports repeatability on the proxy benchmark, not full-arm or hardware success.

All runs stay below 54 MB peak allocated CUDA tensor memory. Dataset and checkpoint fingerprints are recorded in [the structured results](results/training-seeds.json). See [reproduction commands](reproducibility.md#training-seed-study).
