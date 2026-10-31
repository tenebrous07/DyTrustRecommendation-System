# Model Comparison

Mean ± standard deviation over seeds. All models share one chronological training
split, were tuned on the validation split only, stopped by validation NDCG@10, and
were scored once on the test split by the same evaluator (full-catalogue ranking,
training positives masked). Sorted by ndcg@10.

| Model | seeds | precision@10 | recall@10 | ndcg@10 | map@10 | mrr | hit_rate@10 | coverage@10 | diversity@10 | novelty@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B_hgt_sbert | 2 | 0.0067 ± 0.0002 | 0.0430 ± 0.0021 | 0.0267 ± 0.0011 | 0.0184 ± 0.0008 | 0.0290 ± 0.0007 | 0.0631 ± 0.0019 | 0.0645 ± 0.0278 | 0.4177 ± 0.0481 | 9.4617 ± 0.2379 |
