# HGT robustness to scripted shill campaigns

> **SYNTHETIC RESULT.** Shills are scripted by `SyntheticAnomalyInjector` and added to a
> copy of the training split. Evaluation is on the clean test split. These numbers measure
> resistance to this attack, not detection of real Amazon fraud.

| Injection rate | Defence | seeds | NDCG@10 | users with a promoted item in top-10 |
| ---: | --- | ---: | ---: | ---: |
| 0% | none | 3 | 0.0278 ± 0.0018 | 0.00% |
| 0% | trust | 2 | 0.0267 ± 0.0009 | 0.00% |
| 0% | trust+gate | 2 | 0.0244 ± 0.0003 | 0.00% |
| 20% | none | 3 | 0.0250 ± 0.0020 | 0.20% |
| 20% | cola | 4 | 0.0253 ± 0.0022 | 0.59% |
| 20% | trust | 3 | 0.0245 ± 0.0012 | 0.32% |
| 20% | trust+gate | 3 | 0.0259 ± 0.0014 | 1.20% |
| 20% | oracle | 3 | 0.0261 ± 0.0014 | 0.59% |
| 20% | inverted | 3 | 0.0223 ± 0.0033 | 0.18% |
