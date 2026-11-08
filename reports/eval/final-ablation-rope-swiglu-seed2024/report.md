# Evaluation of runs/ablation-rope-swiglu-seed2024/best.pt

Checkpoint step 5000, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.454 | 32 | 1.486 | 3.330 | 4.174 | 0.45 | 2.985 | 0.50 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.503 | 33 | 1.507 | 3.321 | 4.174 | 0.45 | 2.973 | 0.51 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.517 | 249 | 2.818 | 4.331 | 4.957 | 0.65 | 4.127 | 0.68 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.123 | 168 | 2.491 | 4.120 | 4.719 | 0.60 | 3.933 | 0.63 |
| fw2_arb | val | training | 692,916 | 4.369 | 79 | 1.771 | 3.448 | 3.934 | 0.51 | 3.302 | 0.54 |
| fw2_arb | test | training | 855,059 | 4.503 | 90 | 1.846 | 3.475 | 3.978 | 0.53 | 3.322 | 0.56 |
| fw2_ars | val | training | 189,459 | 5.211 | 183 | 2.438 | 3.990 | 4.539 | 0.61 | 3.803 | 0.64 |
| fw2_ars | test | training | 159,996 | 5.093 | 163 | 2.506 | 4.158 | 4.775 | 0.60 | 3.979 | 0.63 |
| fw2_ary | val | training | 64,808 | 4.561 | 96 | 1.801 | 3.419 | 3.832 | 0.53 | 3.255 | 0.55 |
| fw2_ary | test | training | 203,565 | 4.592 | 99 | 1.706 | 3.272 | 3.606 | 0.52 | 3.110 | 0.55 |
| fw2_arz | val | training | 571,707 | 4.643 | 104 | 2.052 | 3.761 | 4.289 | 0.55 | 3.605 | 0.57 |
| fw2_arz | test | training | 729,537 | 4.600 | 100 | 2.068 | 3.829 | 4.361 | 0.54 | 3.660 | 0.56 |
| wiki_ar | val | training | 277,136 | 3.993 | 54 | 1.704 | 3.585 | 4.140 | 0.48 | 3.426 | 0.50 |
| wiki_ar | test | training | 319,077 | 4.144 | 63 | 1.741 | 3.532 | 4.077 | 0.49 | 3.369 | 0.52 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.298 | 200 | 2.997 | 4.695 | 5.489 | 0.64 | 4.487 | 0.67 |
