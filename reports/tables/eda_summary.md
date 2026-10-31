# Exploratory Data Analysis

## Headline figures

| Metric | Value |
| --- | ---: |
| n interactions | 144,178 |
| n users | 17,223 |
| interactions per user mean | 8.3712 |
| interactions per user median | 6.0000 |
| interactions per user max | 185 |
| heavy users p99 | 178 |
| heavy user threshold p99 | 35.0000 |
| users with le 5 interactions | 5,664 |
| mean rating given | 4.2958 |
| users with zero rating variance | 4,159 |
| n items | 7,692 |
| interactions per item mean | 18.7439 |
| interactions per item median | 10.0000 |
| interactions per item max | 656 |
| share held by top 20% items | 0.5947 |
| popularity gini | 0.5206 |
| mean rating overall | 4.2933 |
| share five star | 0.6476 |
| review length mean chars | 507.5153 |
| review length median chars | 155.0000 |
| share verified purchase | 0.7839 |
| verified minus unverified rating | 0.5028 |
| first interaction | 1999-11-18 |
| last interaction | 2023-08-29 |
| n active months | 286 |
| peak month | 2014-12-31 |
| peak month interactions | 2,305 |
| n categories | 39 |
| mean user category drift | 0.4334 |
| density | 0.0011 |

## Figures

| Figure | Path |
| --- | --- |
| category trends | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/category_trends.png` |
| category trends interactive | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/category_trends.html` |
| long tail | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/long_tail.png` |
| long tail interactive | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/long_tail.html` |
| review distributions | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/review_distributions.png` |
| reviews per month interactive | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/reviews_per_month.html` |
| temporal trends | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/temporal_trends.png` |
| user distributions | `/Users/monishraja/Documents/FDA/dytrustrec/reports/figures/user_distributions.png` |

## Notes and caveats

- 4,159 users (24.1%) gave the same rating to everything. That is an anomaly indicator for Module 4, not a fraud finding - most are simply users with very short histories.
- The most popular 20% of items hold 59.5% of all interactions (Gini 0.521). A popularity baseline will therefore be strong, and any model that fails to beat it is not learning personalisation.
- 64.8% of retained interactions are 5-star. This skew is characteristic of e-commerce review data and is why rating value is treated as a feature, not as a graded relevance label.
- Sentiment distribution is not plotted: sentiment is produced by Module 6 in Phase 3, which has not run. Re-running EDA afterwards adds the panel.
- Verified purchases rate +0.503 stars differently from unverified ones. `verified_purchase` is therefore kept as a review feature and as a trust input.
- Trend panels start at 2008-01-31: 115 early periods hold fewer than 144 interactions each and would otherwise dominate the axis. Full history is retained in the table, only the plots are trimmed.
- Mean per-user category drift between first and second half of history is 0.433 (0 = identical mix, 1 = disjoint). This is the measured premise for recency weighting in Module 13, not an assumption.
