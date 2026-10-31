# Confidence usefulness (test split)

Seed ensemble of 3 runs of `E_hgt_sbert_trust_recency`; 18,620 (user, top-10 item) pairs over 1,862 users; base hit rate 0.0087.

AUROC for predicting whether a recommended item was actually interacted with (0.5 = no signal):

| Signal | AUROC |
| --- | ---: |
| confidence | 0.456 |
| stability | 0.470 |
| volume | 0.458 |
| trust | 0.507 |
| score | 0.556 |

Hit rate by confidence quintile (low to high):

| Quintile | pairs | mean confidence | hit rate |
| ---: | ---: | ---: | ---: |
| 1 | 3,724 | 0.283 | 0.0134 |
| 2 | 3,724 | 0.438 | 0.0073 |
| 3 | 3,724 | 0.599 | 0.0075 |
| 4 | 3,724 | 0.658 | 0.0059 |
| 5 | 3,724 | 0.753 | 0.0094 |

## A calibrated hit probability (fitted on validation recommendations, scored once on test)

AUROC 0.607; Brier score 0.00869 against 0.00868 for always predicting the base rate; expected calibration error 0.00767 (base rate 0.0087).

| Quintile of predicted probability | pairs | mean predicted | observed hit rate |
| ---: | ---: | ---: | ---: |
| 1 | 3,724 | 0.0087 | 0.0059 |
| 2 | 3,724 | 0.0111 | 0.0059 |
| 3 | 3,724 | 0.0136 | 0.0070 |
| 4 | 3,724 | 0.0180 | 0.0086 |
| 5 | 3,724 | 0.0305 | 0.0161 |

Logistic-regression weights on standardised features: score +0.20, log(rank) -0.32, stability +0.07, log(history) -0.09, trust -0.05.

Explanation coverage: 83.6% of recommendations had at least one retrieved evidence item (mean 1.84 items per recommendation). Hit rate 0.0081 with evidence against 0.0118 without.

Cold-start check: mean confidence 0.292 for users with fewer than 3 training interactions (5,340 pairs) against 0.648 for the rest.
