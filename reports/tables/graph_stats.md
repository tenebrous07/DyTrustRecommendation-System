# Graph Analytics

## Structural metrics

| Metric | Value |
| --- | ---: |
| n nodes | 24,915 |
| n edges | 144,178 |
| density | 0.000465 |
| mean degree | 11.5736 |
| n user nodes | 17,223 |
| n item nodes | 7,692 |
| bipartite density | 0.001088 |
| degree min | 5.0000 |
| degree median | 7.0000 |
| degree mean | 11.5736 |
| degree p99 | 83.0000 |
| degree max | 656.0000 |
| n connected components | 1 |
| largest component nodes | 24,915 |
| largest component share | 1.0000 |
| bipartite clustering mean | 0.058315 |
| triangle clustering mean | 0.000000 |
| pagerank max | 0.002046 |
| closeness mean | 0.237620 |
| n communities | 11 |
| largest community nodes | 5,261 |
| median community nodes | 2,037 |
| modularity | 0.458215 |
| degree assortativity | -0.137320 |

## Approximated metrics

| Metric | Method | Sample | Exact cost avoided |
| --- | --- | ---: | --- |
| betweenness | Brandes with pivot sampling | 500 of 24,915 source nodes | O(V*E), hours at this size |
| closeness | exact per-node closeness on a uniform node sample | 1000 of 24,915 nodes | O(V*E), hours at this size |

## Top nodes by pagerank

| node | type | value |
| --- | --- | ---: |
| `i:B01N3ASPNV` | item | 0.002046 |
| `i:B0086VPUHI` | item | 0.001628 |
| `i:B07YBXFDYN` | item | 0.001564 |
| `i:B00BGA9WK2` | item | 0.001313 |
| `i:B004RMK5QG` | item | 0.001258 |
| `i:B000N5Z2L4` | item | 0.001206 |
| `i:B077GG9D5D` | item | 0.001183 |
| `i:B07YBWT3PK` | item | 0.001118 |
| `i:B00C1TTF86` | item | 0.001086 |
| `i:B00BN5T30E` | item | 0.001055 |
| `i:B087NNPYP3` | item | 0.001007 |
| `i:B004LLHFAW` | item | 0.000948 |
| `i:B07624RBWB` | item | 0.000881 |
| `i:B00W9DHUBS` | item | 0.000878 |
| `i:B00KIWEMIG` | item | 0.000859 |
| `i:B00BHRD4BM` | item | 0.000853 |
| `i:B00CJ9OTNE` | item | 0.000805 |
| `i:B07K3KHFSY` | item | 0.000738 |
| `i:B019WRM1IA` | item | 0.000727 |
| `i:B00K0NV5J2` | item | 0.000699 |

## Top nodes by betweenness

| node | type | value |
| --- | --- | ---: |
| `i:B01N3ASPNV` | item | 0.046344 |
| `i:B07YBXFDYN` | item | 0.038076 |
| `i:B0086VPUHI` | item | 0.030391 |
| `i:B00BGA9WK2` | item | 0.025213 |
| `i:B004RMK5QG` | item | 0.021490 |
| `i:B000N5Z2L4` | item | 0.020834 |
| `i:B07YBWT3PK` | item | 0.020433 |
| `i:B00BN5T30E` | item | 0.016835 |
| `i:B077GG9D5D` | item | 0.016234 |
| `i:B087NNPYP3` | item | 0.015592 |
| `i:B07624RBWB` | item | 0.014242 |
| `i:B004LLHFAW` | item | 0.013723 |
| `i:B019WRM1IA` | item | 0.011522 |
| `u:AFVOW2EKNL5O25KG6VA4AIZKSB2A` | user | 0.011268 |
| `i:B00C1TTF86` | item | 0.010809 |
| `u:AH2ENAH2MUCHWBSJHZIP6GGXB4GA` | user | 0.009707 |
| `i:B00W9DHUBS` | item | 0.009667 |
| `i:B014R4KYMS` | item | 0.009606 |
| `i:B00BHRD4BM` | item | 0.009157 |
| `i:B00NOD0OTW` | item | 0.009130 |

## Top nodes by closeness

| node | type | value |
| --- | --- | ---: |
| `i:B01N3ASPNV` | item | 0.324414 |
| `i:B00BGA9X9W` | item | 0.300368 |
| `i:B003ZDOFF0` | item | 0.298604 |
| `i:B07YBX6T95` | item | 0.298482 |
| `i:B00YQ74LVM` | item | 0.294544 |
| `i:B0079V4GK8` | item | 0.292284 |
| `i:B07HYW4XMP` | item | 0.291818 |
| `i:B01MYUCFBK` | item | 0.291198 |
| `i:B01GY3601E` | item | 0.288467 |
| `i:B0C3KYVDWT` | item | 0.288313 |
| `i:B07N5LL4YW` | item | 0.284839 |
| `i:B003ZSN600` | item | 0.284267 |
| `i:B00CES8EFY` | item | 0.284053 |
| `i:B06ZZCCDP7` | item | 0.283123 |
| `i:B00RSXRLUE` | item | 0.280610 |
| `i:B00NQREKHS` | item | 0.279672 |
| `u:AE4IGGDPCICARTUOMUO6XVF4BTIA` | user | 0.279324 |
| `i:B08F4D36D9` | item | 0.279158 |
| `i:B0721GGGS9` | item | 0.277776 |
| `u:AHLD3BAM5XXNSUGS3QICCDK6PR6Q` | user | 0.277346 |

## Notes

- `density` uses the unipartite denominator C(n,2); `bipartite_density` uses |users|x|items|, which is the correct maximum for a bipartite graph. Quote the latter.
- Triangle clustering is exactly 0 for any bipartite graph (no odd cycles exist), so it is not a finding. The Latapy bipartite coefficient (mode='dot') is reported instead.
- Louvain found 11 communities with modularity 0.4582. Community structure is what a graph model can exploit and a matrix-factorisation baseline cannot.
- Degree assortativity is -0.137 (disassortative): active users attach disproportionately to popular items. Expected for e-commerce, and a reason popularity baselines are hard to beat.
- Computed with NetworkX for analysis only. Training uses PyTorch Geometric; NetworkX is never used at training scale (requirement 35).
