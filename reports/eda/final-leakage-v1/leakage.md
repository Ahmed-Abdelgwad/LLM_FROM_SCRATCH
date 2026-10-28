# Leakage report final-leakage-v1

scratchllm 0.1.0. Measure only: no document was removed.
Jaccard on 5-word shingles; near-duplicate = exact Jaccard >= 0.8 (A9); 128 permutations, 8 rows per band; at most 50 training documents offered per bucket.

Training pool (train split, after the cleaning rules): fw2_arb 48,232, fw2_arz 47,900, fw2_ary 9,649, fw2_ars 9,507, wiki_ar 19,351, fineweb_edu_10bt 47,880

Training pool 182,519 documents; evaluation 7,449 (val+test of the training sources) + 24,601 (evaluation-only sources). Counts reconcile with the manifests: True.

| evaluation group | documents | exact dup | near >= 0.8 | share | near >= 0.6 | best in same source | best in other source |
|---|---|---|---|---|---|---|---|
| fw2_arb (val+test) | 1,730 | 0 | 0 | 0.00% | 0 (0.00%) | 0 | 0 |
| fw2_arz (val+test) | 2,014 | 0 | 0 | 0.00% | 1 (0.05%) | 0 | 0 |
| fw2_ary (val+test) | 350 | 0 | 0 | 0.00% | 0 (0.00%) | 0 | 0 |
| fw2_ars (val+test) | 473 | 0 | 0 | 0.00% | 0 (0.00%) | 0 | 0 |
| wiki_ar (val+test) | 823 | 0 | 0 | 0.00% | 3 (0.36%) | 0 | 0 |
| fineweb_edu_10bt (val+test) | 2,059 | 0 | 0 | 0.00% | 0 (0.00%) | 0 | 0 |
| fw2_acm (evaluation-only source) | 6,990 | 0 | 0 | 0.00% | 0 (0.00%) | 0 | 0 |
| fw2_apc (evaluation-only source) | 9,618 | 0 | 0 | 0.00% | 0 (0.00%) | 0 | 0 |
| wiki_ary (evaluation-only source) | 7,993 | 0 | 0 | 0.00% | 2 (0.03%) | 0 | 0 |

## Sample coverage (a twin in the full corpus is in the sample with this chance)

| source | rows in source | sampled | share |
|---|---|---|---|
| fw2_arb | 61,977,525 | 50,000 | 0.0807% |
| fw2_arz | 853,290 | 50,000 | 5.8597% |
| fw2_ary | 2,365,405 | 10,000 | 0.4228% |
| fw2_ars | 298,167 | 10,000 | 3.3538% |
| wiki_ar | 1,219,201 | 50,000 | 4.1010% |
| fineweb_edu_10bt | 9,672,101 | 50,000 | 0.5170% |
| fw2_acm | 7,382 | 7,000 | 94.8253% |
| fw2_apc | 69,740 | 10,000 | 14.3390% |
| wiki_ary | 8,087 | 8,000 | 98.9242% |
