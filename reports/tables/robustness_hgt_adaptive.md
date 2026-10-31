# HGT robustness to scripted shill campaigns

> **SYNTHETIC RESULT.** Shills are scripted by `SyntheticAnomalyInjector` and added to a
> copy of the training split. Evaluation is on the clean test split. These numbers measure
> resistance to this attack, not detection of real Amazon fraud.

| Injection rate | Defence | seeds | NDCG@10 | users with a promoted item in top-10 |
| ---: | --- | ---: | ---: | ---: |
| 0% | none | 3 | 0.0278 ± 0.0018 | 0.00% |
| 0% | gate_v2 | 2 | 0.0266 ± 0.0004 | 0.00% |
| 20% | none | 4 | 0.0246 ± 0.0017 | 1.54% |
| 20% | trust+gate | 4 | 0.0235 ± 0.0033 | 1.37% |
| 20% | gate_v2 | 4 | 0.0266 ± 0.0018 | 0.94% |
| 20% | oracle | 2 | 0.0236 ± 0.0029 | 0.27% |
| 20% | oracle_gate | 4 | 0.0269 ± 0.0014 | 0.28% |
