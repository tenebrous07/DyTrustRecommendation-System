# Model Comparison

Mean ± standard deviation over seeds. All models share one chronological training
split, were tuned on the validation split only, stopped by validation NDCG@10, and
were scored once on the test split by the same evaluator (full-catalogue ranking,
training positives masked). Sorted by ndcg@10.

| Model | seeds | precision@10 | recall@10 | ndcg@10 | map@10 | mrr | hit_rate@10 | coverage@10 | diversity@10 | novelty@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E_hgt_sbert_trust_recency | 1 | 0.0082 ± 0.0000 | 0.0502 ± 0.0000 | 0.0295 ± 0.0000 | 0.0190 ± 0.0000 | 0.0321 ± 0.0000 | 0.0768 ± 0.0000 | 0.0275 ± 0.0000 | 0.3073 ± 0.0000 | 10.0528 ± 0.0000 |
