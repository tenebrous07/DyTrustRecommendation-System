# Cleaning Report

## Summary

| Metric | Value |
| --- | ---: |
| Raw interactions | 262,547 |
| Retained interactions | 144,178 |
| Removed | 118,369 |
| Retention | 54.91% |
| Users (raw -> final) | 30,000 -> 17,223 |
| Items (raw -> final) | 47,089 -> 7,692 |

## Pipeline steps

| Step | Before | After | Removed | % | Detail |
| --- | ---: | ---: | ---: | ---: | --- |
| normalise_ids | 262,547 | 262,547 | 0 | 0.00 | 0 item_id values do not match the 10-char ASIN pattern (kept) |
| drop_null_keys | 262,547 | 262,547 | 0 | 0.00 | rows missing user_id or item_id |
| validate_ratings | 262,547 | 262,547 | 0 | 0.00 | outside [1.0, 5.0] or null (0 null) |
| validate_timestamps | 262,547 | 262,547 | 0 | 0.00 | outside [1996-01-01, 2023-12-31] or unparseable |
| deduplicate_exact | 262,547 | 257,497 | 5,050 | 1.92 | identical (user_id, item_id, timestamp) |
| collapse_user_item | 257,497 | 255,021 | 2,476 | 0.96 | repeat reviews of the same item by the same user; kept most recent |
| flag_suspicious | 255,021 | 255,021 | 0 | 0.00 | 66,629 rows flagged, 0 removed |
| k_core_filter | 255,021 | 144,178 | 110,843 | 43.46 | users >= 5, items >= 5 interactions, converged in 10 pass(es) |

## Missing values (after cleaning)

| Column | Null count | Null % |
| --- | ---: | ---: |
| event_time | 0 | 0.00 |
| helpful_votes | 0 | 0.00 |
| item_id | 0 | 0.00 |
| outlier_flag | 0 | 0.00 |
| outlier_reasons | 0 | 0.00 |
| rating | 0 | 0.00 |
| review_text | 12 | 0.01 |
| review_title | 1 | 0.00 |
| timestamp | 0 | 0.00 |
| user_id | 0 | 0.00 |
| variant_id | 0 | 0.00 |
| verified_purchase | 0 | 0.00 |

## Rows flagged (retained, not deleted)

| Flag reason | Rows |
| --- | ---: |
| zero_rating_variance | 25,762 |
| burst_under_60s | 20,886 |
| repeated_text_across_items | 6,251 |
| extreme_review_length | 194 |
| empty_review_text | 12 |

## Notes and caveats

- Time span after timestamp validation (before k-core): 1999-10-22 to 2023-09-02.
- k-core (5-user / 5-item) uses interaction counts over the full timeline, including the future test period. This is standard practice in recommender-systems evaluation but is a mild selection leakage; it is disclosed rather than hidden. Set both thresholds to 1 to disable.
- k-core removes the extreme long tail. Coverage and novelty metrics (Module 19) are therefore computed over the filtered catalogue, not the full Amazon catalogue, and must be reported as such.
- `categories` is empty for 190 of 7,692 products (2.5%). Category nodes are built primarily from `main_category`.
- `price` is missing for 1,962 of 7,692 products (25.5%). It is never imputed; a `price_missing` indicator is carried instead.
- Final retained time span: 1999-11-18 to 2023-08-29.
