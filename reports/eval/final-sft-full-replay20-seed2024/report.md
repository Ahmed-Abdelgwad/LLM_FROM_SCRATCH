# Evaluation of runs/sft-full-replay20-seed2024/merged.pt

Checkpoint step 1000, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.676 | 39 | 1.581 | 3.330 | 4.174 | 0.47 | 2.985 | 0.53 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.726 | 42 | 1.603 | 3.321 | 4.174 | 0.48 | 2.973 | 0.54 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.721 | 305 | 2.922 | 4.331 | 4.957 | 0.67 | 4.127 | 0.71 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.308 | 202 | 2.581 | 4.120 | 4.719 | 0.63 | 3.933 | 0.66 |
| fw2_arb | val | training | 692,916 | 4.551 | 95 | 1.845 | 3.448 | 3.934 | 0.54 | 3.302 | 0.56 |
| fw2_arb | test | training | 855,059 | 4.678 | 108 | 1.917 | 3.475 | 3.978 | 0.55 | 3.322 | 0.58 |
| fw2_ars | val | training | 189,459 | 5.417 | 225 | 2.534 | 3.990 | 4.539 | 0.64 | 3.803 | 0.67 |
| fw2_ars | test | training | 159,996 | 5.319 | 204 | 2.618 | 4.158 | 4.775 | 0.63 | 3.979 | 0.66 |
| fw2_ary | val | training | 64,808 | 4.787 | 120 | 1.890 | 3.419 | 3.832 | 0.55 | 3.255 | 0.58 |
| fw2_ary | test | training | 203,565 | 4.808 | 122 | 1.786 | 3.272 | 3.606 | 0.55 | 3.110 | 0.57 |
| fw2_arz | val | training | 571,707 | 4.843 | 127 | 2.140 | 3.761 | 4.289 | 0.57 | 3.605 | 0.59 |
| fw2_arz | test | training | 729,537 | 4.807 | 122 | 2.160 | 3.829 | 4.361 | 0.56 | 3.660 | 0.59 |
| wiki_ar | val | training | 277,136 | 4.171 | 65 | 1.780 | 3.585 | 4.140 | 0.50 | 3.426 | 0.52 |
| wiki_ar | test | training | 319,077 | 4.318 | 75 | 1.814 | 3.532 | 4.077 | 0.51 | 3.369 | 0.54 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.527 | 251 | 3.126 | 4.695 | 5.489 | 0.67 | 4.487 | 0.70 |
