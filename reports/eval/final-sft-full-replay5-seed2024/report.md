# Evaluation of runs/sft-full-replay5-seed2024/merged.pt

Checkpoint step 900, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.874 | 48 | 1.666 | 3.330 | 4.174 | 0.50 | 2.985 | 0.56 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.922 | 51 | 1.687 | 3.321 | 4.174 | 0.51 | 2.973 | 0.57 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.850 | 347 | 2.988 | 4.331 | 4.957 | 0.69 | 4.127 | 0.72 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.438 | 230 | 2.645 | 4.120 | 4.719 | 0.64 | 3.933 | 0.67 |
| fw2_arb | val | training | 692,916 | 4.658 | 105 | 1.889 | 3.448 | 3.934 | 0.55 | 3.302 | 0.57 |
| fw2_arb | test | training | 855,059 | 4.787 | 120 | 1.962 | 3.475 | 3.978 | 0.56 | 3.322 | 0.59 |
| fw2_ars | val | training | 189,459 | 5.532 | 253 | 2.587 | 3.990 | 4.539 | 0.65 | 3.803 | 0.68 |
| fw2_ars | test | training | 159,996 | 5.456 | 234 | 2.685 | 4.158 | 4.775 | 0.65 | 3.979 | 0.67 |
| fw2_ary | val | training | 64,808 | 4.911 | 136 | 1.940 | 3.419 | 3.832 | 0.57 | 3.255 | 0.60 |
| fw2_ary | test | training | 203,565 | 4.920 | 137 | 1.828 | 3.272 | 3.606 | 0.56 | 3.110 | 0.59 |
| fw2_arz | val | training | 571,707 | 4.966 | 143 | 2.195 | 3.761 | 4.289 | 0.58 | 3.605 | 0.61 |
| fw2_arz | test | training | 729,537 | 4.930 | 138 | 2.216 | 3.829 | 4.361 | 0.58 | 3.660 | 0.61 |
| wiki_ar | val | training | 277,136 | 4.277 | 72 | 1.825 | 3.585 | 4.140 | 0.51 | 3.426 | 0.53 |
| wiki_ar | test | training | 319,077 | 4.418 | 83 | 1.856 | 3.532 | 4.077 | 0.53 | 3.369 | 0.55 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.613 | 274 | 3.175 | 4.695 | 5.489 | 0.68 | 4.487 | 0.71 |
