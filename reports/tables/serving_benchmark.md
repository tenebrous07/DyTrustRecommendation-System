# Serving and training cost

Hardware: Apple M3 Pro, 12 cores, 18 GB, CPU numpy for serving.

| Catalogue | items | ms per user (one at a time) | ms per user (batched) |
| --- | ---: | ---: | ---: |
| this catalogue (real embeddings) | 6,619 | 0.08 | 0.052 |
| synthetic catalogue (random vectors) | 100,000 | 1.55 | 0.687 |
| synthetic catalogue (random vectors) | 1,000,000 | 16.35 | 8.389 |

One user's embedding is 512 bytes, so 1 million users is 0.51 GB.

Mean training time per run, seconds (includes validation passes):

| Model | seconds |
| --- | ---: |
| A_hgt_only | 274 |
| B_hgt_sbert | 377 |
| C_hgt_sbert_trust | 342 |
| D_hgt_sbert_recency | 229 |
| E_hgt_sbert_trust_recency | 211 |
| F_dytrustrec_full | 214 |
| G_dytrustrec_cola_trust | 300 |
| H_hgt_sbert_unmasked | 111 |
| I_dytrustrec_cosine | 501 |
| item_knn_cf | 1 |
| lightgcn | 12 |
| matrix_factorization | 17 |
| popularity | 0 |
| sasrec | 31 |
