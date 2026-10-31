# Explanation fidelity by interaction removal (test users, top-1 recommendation)

Model: `E seed 42`. Score drop is in units of the user's own score spread; a larger drop means the recommendation depended more on what was removed.

| Cited evidence | users | rows removed | score drop, cited | score drop, same number of random rows | cited hurts more | difference, 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| shared_category | 81 | 3.4 | 0.314 | -0.127 | 94% | +0.441 [+0.322, +0.604] |
| shared_brand | 73 | 5.3 | 0.475 | -0.239 | 95% | +0.714 [+0.582, +0.853] |
