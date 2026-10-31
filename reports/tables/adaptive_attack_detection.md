# Detecting a campaign that knows the detector (SYNTHETIC)

Injection rate 10%. ROC-AUC against the injected labels (chance = 0.5).

| Detector | standard (templates, 30-minute bursts) | adaptive (paraphrased, spread over 30 days) |
| --- | ---: | ---: |
| version 1 (mean of ranks) | 0.9997 | 0.5190 |
| version 2 (mean of ranks) | 0.9997 | 0.7273 |
| version 2 (strongest indicator) | 0.9897 | 0.9445 |
| indicator: shared_text | 0.9997 | 0.8736 |
| indicator: own_repeat | 0.9893 | 0.4643 |
| indicator: burstiness | 0.9195 | 0.1293 |
| indicator: burst_7d | 0.9142 | 0.4493 |
| indicator: semantic_shared | 0.9938 | 0.9937 |
