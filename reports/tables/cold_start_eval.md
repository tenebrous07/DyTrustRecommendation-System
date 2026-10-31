# Cold-start items (test split)

Embeddings: 3-seed ensemble of `E_hgt_sbert_trust_recency`. 3,801 known users with a test interaction, on a catalogue of 6,619 warm and 1,073 new products; 5,308 of 8,317 test rows are on new products.

Blend weights (text weight 0.25, new-product bonus 0.0) were chosen on validation (NDCG@10 0.0385).

## Reserved new-arrival slots (last r of the 10 places are the best new products by text)

r was chosen on validation: r = 3.

| r | NDCG@10 | Recall@10 |
| ---: | ---: | ---: |
| 0 | 0.0126 | 0.0182 |
| 1 | 0.0135 | 0.0209 |
| 2 | 0.0142 | 0.0231 |
| 3 (chosen) | 0.0154 | 0.0262 |
| 4 | 0.0160 | 0.0271 |
| 5 | 0.0166 | 0.0285 |

Chosen r against r = 0, paired bootstrap over users: +0.0028 NDCG@10, 95% CI [+0.0017, +0.0040].

## Mixed catalogue, one score for everything (a single blended score; kept as a negative result)

| Model | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| blend: model + text (chosen on validation) | 0.0114 | 0.0172 |
| model only (cannot score new products) | 0.0126 | 0.0182 |
| text similarity only | 0.0054 | 0.0097 |
| popularity | 0.0048 | 0.0095 |
| random | 0.0006 | 0.0013 |

Gain of the blend over the model alone, paired bootstrap over users: -0.0012 NDCG@10, 95% CI [-0.0028, +0.0004].

## New products only (2937 users, 1073 candidates)

| Model | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| text similarity | 0.0247 | 0.0436 |
| model embedding mapped from text | 0.0099 | 0.0190 |
| random | 0.0040 | 0.0075 |

Mapping text into the embedding space (the first idea) has a 5-fold CV R^2 of 0.321 and ranks new products worse than plain text similarity.
