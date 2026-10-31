# Outlier Indicators (Module 4)

> These are anomaly **indicators**, not fraud labels. This dataset has no
> manipulation ground truth, so none of the numbers below can be validated
> against the truth. They rank users for inspection; they do not convict them.

## Summary

| Metric | Value |
| --- | ---: |
| n users scored | 17,223 |
| n flagged iqr | 9,457 |
| n flagged zscore | 2,441 |
| n flagged iforest | 1,695 |
| n flagged lof | 543 |
| n corroborated | 3,218 |
| corroborated share | 0.1868 |
| min indicators for corroboration | 2 |

## Features used (12)

`n_interactions`, `n_distinct_items`, `mean_rating`, `rating_std`, `frac_five_star`, `frac_one_star`, `mean_review_length`, `reviews_per_active_day`, `median_gap_hours`, `min_gap_hours`, `frac_verified`, `frac_duplicate_text`

## Top 50 users by agreeing indicators

| user_id | n_indicators | iforest_score | lof_score | iqr_flags | zscore_flags | flag_reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `AHI5HIYDBYF22CL3OQWBZ2BVB6JA` | 4 | 0.720 | 1.744 | 5 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:reviews_per_active_day;iqr:frac_verified;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AED7RGWNPJ7SBBUN6UVNM4XLQRHQ` | 4 | 0.720 | 1.678 | 5 | 5 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_rating;iqr:frac_one_star;iqr:frac_verified;z>3:n_interactions;z>3:n_distinct_items;z>3:mean_rating;z>3:frac_one_star;z>3:frac_verified;isolation_forest;local_outlier_factor |
| `AEDQEB3TPK2Z666NRC6WSCIQO5VA` | 4 | 0.715 | 1.938 | 4 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:median_gap_hours;iqr:min_gap_hours;z>3:mean_rating;z>3:frac_one_star;z>3:min_gap_hours;isolation_forest;local_outlier_factor |
| `AFGEE4ZB5NC5ODGROG7T5MWUPPQQ` | 4 | 0.713 | 1.743 | 6 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:reviews_per_active_day;iqr:frac_verified;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGEU4JUG22336QDEYI6J3DSE2GTQ` | 4 | 0.706 | 1.812 | 5 | 4 | iqr:mean_rating;iqr:frac_one_star;iqr:reviews_per_active_day;iqr:frac_verified;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_verified;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AHK5IQX6MXRKJDVSOPHNQ4N2CAUQ` | 4 | 0.706 | 1.901 | 5 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_rating;iqr:frac_one_star;iqr:frac_verified;z>3:n_interactions;z>3:n_distinct_items;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AG4S5TGV54EEU2CPZISLGT5LSNVA` | 4 | 0.702 | 1.653 | 6 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;iqr:frac_verified;z>3:mean_rating;z>3:frac_one_star;z>3:min_gap_hours;isolation_forest;local_outlier_factor |
| `AHA22N4CZDHGUVCACGF7FVPHQCHA` | 4 | 0.686 | 1.588 | 6 | 2 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;iqr:frac_verified;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AE6YOYTJIFFPJXCCPSP55QUJCTSA` | 4 | 0.683 | 1.708 | 4 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AHFA76LWJOOVLKL4WMAN7KXRHHEQ` | 4 | 0.681 | 1.511 | 6 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;iqr:frac_verified;z>3:mean_rating;z>3:frac_one_star;z>3:frac_verified;isolation_forest;local_outlier_factor |
| `AGLDW5ZY7MIVGH65NZ3BWSZ75EYA` | 4 | 0.680 | 1.673 | 4 | 4 | iqr:mean_rating;iqr:frac_one_star;iqr:frac_verified;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_verified;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AF7Q5PFMNXP6ULUE6JP5Z2XF7PCQ` | 4 | 0.675 | 1.725 | 4 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGTRATJL2IH2BOA5GF3MBQD3EJFQ` | 4 | 0.672 | 2.001 | 4 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AFWOBQ44IBZCTR7HZGLKQ2NX7PAQ` | 4 | 0.667 | 1.636 | 5 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_review_length;iqr:min_gap_hours;iqr:frac_verified;z>3:n_interactions;z>3:n_distinct_items;z>3:frac_verified;isolation_forest;local_outlier_factor |
| `AEYTBNAWM3JWMCXTOOLK3DBXRMQA` | 4 | 0.662 | 1.596 | 4 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_verified;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;z>3:frac_verified;isolation_forest;local_outlier_factor |
| `AEAP2QKWA6GHFG47ERFDYXIT4WGQ` | 4 | 0.662 | 1.626 | 5 | 2 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AFP2NVVQNQC5WOWV7HWHWLDHTE2Q` | 4 | 0.660 | 1.577 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;isolation_forest;local_outlier_factor |
| `AE2FSBAB3Z5FBOWHKTNJHBZMRCKQ` | 4 | 0.660 | 1.533 | 4 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AF4T43HDTE7P3FJXQMZYVGHDMLRQ` | 4 | 0.660 | 1.660 | 4 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:frac_verified;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AE2UUOJBJFJLUV7KFD5MKEJK5PBA` | 4 | 0.660 | 2.303 | 5 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_review_length;iqr:reviews_per_active_day;iqr:frac_verified;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;isolation_forest;local_outlier_factor |
| `AEHI3XLWYDNZRN3GQK5BWXN7I7QA` | 4 | 0.659 | 1.542 | 6 | 2 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;iqr:frac_verified;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AGZ2DHO7BDXDKLYH3Y7QVOO5R6OA` | 4 | 0.657 | 1.784 | 4 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AFAVYSHL6NZXFMVTRSGUTYIZ2WUQ` | 4 | 0.651 | 1.511 | 5 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:frac_one_star;iqr:frac_verified;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;isolation_forest;local_outlier_factor |
| `AFBUTPXCZODRFIW3V4KYF5N4FL7Q` | 4 | 0.646 | 1.559 | 4 | 2 | iqr:mean_rating;iqr:frac_one_star;iqr:median_gap_hours;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AFL7GC37E3E4U7EC74JZ6JOGH2EA` | 4 | 0.644 | 2.295 | 4 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGSKQLOUYY5GDR5WGV6SSYIDKMOQ` | 4 | 0.644 | 1.814 | 4 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AFWF34IGLZICGWPAXD6ARNH7EK2Q` | 4 | 0.636 | 2.177 | 3 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;isolation_forest;local_outlier_factor |
| `AF4PKFGRMXMA4ZGWWLNQ6LLVSBOQ` | 4 | 0.634 | 1.673 | 4 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;isolation_forest;local_outlier_factor |
| `AGIITBN5ZOGYZXG4Z2X35U5XYD4Q` | 4 | 0.633 | 2.184 | 4 | 4 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AHRO7UV2HLK6EJCHBNHRL2WKYYCA` | 4 | 0.632 | 1.816 | 3 | 1 | iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;z>3:min_gap_hours;isolation_forest;local_outlier_factor |
| `AEC3MEHHKQXQT26ZZ55GOMAZMHJQ` | 4 | 0.629 | 1.634 | 3 | 1 | iqr:frac_one_star;iqr:median_gap_hours;iqr:min_gap_hours;z>3:min_gap_hours;isolation_forest;local_outlier_factor |
| `AEXMCM6MKEM2VKAJ37XKPDIEMLTA` | 4 | 0.629 | 3.575 | 3 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;isolation_forest;local_outlier_factor |
| `AEMBVEZ6QOAFFO3Z2YMSHM6EBZUA` | 4 | 0.626 | 1.521 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_rating;iqr:frac_one_star;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AEECVGK7YKKKH572X45MXQ6VI2WQ` | 4 | 0.624 | 1.788 | 3 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:frac_duplicate_text;z>3:n_interactions;z>3:n_distinct_items;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGP3UYNXJQDKNWLW3IPJJNQJKNXQ` | 4 | 0.624 | 1.647 | 3 | 3 | iqr:mean_rating;iqr:frac_one_star;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGWMT2QORZIH3ITWUO5CRPK2OH5A` | 4 | 0.622 | 1.752 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_review_length;iqr:reviews_per_active_day;z>3:n_interactions;z>3:n_distinct_items;isolation_forest;local_outlier_factor |
| `AFVX3CYYUVS5KUTNXAHHO7FXA7PQ` | 4 | 0.622 | 1.578 | 5 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_rating;iqr:frac_one_star;iqr:frac_verified;z>3:n_interactions;z>3:n_distinct_items;z>3:mean_rating;isolation_forest;local_outlier_factor |
| `AHN7LSSDO54P3DSUATJOJUUHMLIA` | 4 | 0.622 | 1.598 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:mean_review_length;iqr:min_gap_hours;z>3:n_interactions;z>3:n_distinct_items;isolation_forest;local_outlier_factor |
| `AE3DVMCFNZTGTSN3RKCGVW774OSQ` | 4 | 0.621 | 1.726 | 3 | 3 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;z>3:n_interactions;z>3:n_distinct_items;z>3:reviews_per_active_day;isolation_forest;local_outlier_factor |
| `AFVOW2EKNL5O25KG6VA4AIZKSB2A` | 4 | 0.615 | 3.481 | 3 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;z>3:n_interactions;z>3:n_distinct_items;isolation_forest;local_outlier_factor |
| `AHRK66U55GMWRNBIQGHGJXOZ6AOQ` | 4 | 0.609 | 1.987 | 5 | 1 | iqr:mean_rating;iqr:frac_one_star;iqr:mean_review_length;iqr:median_gap_hours;iqr:min_gap_hours;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AGZSQLBTFBCKEIT5PLI2BCUD64RQ` | 4 | 0.604 | 1.554 | 3 | 1 | iqr:frac_one_star;iqr:median_gap_hours;iqr:min_gap_hours;z>3:min_gap_hours;isolation_forest;local_outlier_factor |
| `AEPVSRTQ5CYJGOIG44577MURUYXQ` | 4 | 0.602 | 1.509 | 3 | 2 | iqr:mean_rating;iqr:frac_one_star;iqr:frac_duplicate_text;z>3:mean_rating;z>3:frac_one_star;isolation_forest;local_outlier_factor |
| `AEO262JTZSKXY4UGODOEB4OVYN4Q` | 4 | 0.598 | 1.862 | 2 | 1 | iqr:median_gap_hours;iqr:min_gap_hours;z>3:min_gap_hours;isolation_forest;local_outlier_factor |
| `AFPF3E4WXBWDSSAODAMO3N6ZQX4A` | 4 | 0.596 | 1.822 | 3 | 2 | iqr:mean_review_length;iqr:frac_verified;iqr:frac_duplicate_text;z>3:frac_verified;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AEXSOPA7UIYWBRS2KCD6MVQZH4DQ` | 4 | 0.595 | 1.586 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AFZSRJP6J54PUNWKUUO6YTABDGCQ` | 4 | 0.594 | 1.587 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGYJCJE47MSIPL77M75HLJ7A5NIQ` | 4 | 0.594 | 2.455 | 3 | 2 | iqr:mean_rating;iqr:frac_one_star;iqr:reviews_per_active_day;z>3:frac_one_star;z>3:reviews_per_active_day;isolation_forest;local_outlier_factor |
| `AHUHWVQ3MPIH7GSGCPMYJNAHXFAA` | 4 | 0.593 | 1.587 | 4 | 2 | iqr:n_interactions;iqr:n_distinct_items;iqr:reviews_per_active_day;iqr:frac_duplicate_text;z>3:reviews_per_active_day;z>3:frac_duplicate_text;isolation_forest;local_outlier_factor |
| `AGHDTZHD42MLV6QUD26QX4JYJD4A` | 4 | 0.593 | 1.542 | 3 | 1 | iqr:mean_rating;iqr:frac_one_star;iqr:frac_duplicate_text;z>3:mean_rating;isolation_forest;local_outlier_factor |

## Notes

- These are unsupervised anomaly indicators. The dataset carries no fraud labels, so precision and recall cannot be computed here. Quantitative validation happens only under the synthetic-injection protocol in Phase 16, where labels exist by construction.
- Detectors were fitted on: all cleaned interactions (exploratory; NOT train-only) (transductive). Any use of these scores as MODEL FEATURES must refit on the training split only, by setting outliers.fit_on_train_split_only=true.
- LOF ran with novelty=False because it is scoring the rows it was fitted on; scikit-learn documents novelty=True as invalid for that case.
- IQR alone flags 54.9% of users, but that is a multiple-comparisons artefact rather than a finding: the rule fires when ANY of 12 features falls outside its fence, so the per-user hit rate compounds with the feature count. Use `n_indicators` / `corroborated`, not the raw IQR count.
- Isolation Forest and LOF agree on 6.2% of their flags (Jaccard). Low agreement is expected - they model different notions of outlyingness - and is a reason to use the consensus count rather than any single detector.
