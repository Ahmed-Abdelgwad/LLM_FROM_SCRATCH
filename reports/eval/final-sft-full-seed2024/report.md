# Evaluation of runs/sft-full-seed2024/merged.pt

Checkpoint step 900, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 4.078 | 59 | 1.754 | 3.330 | 4.174 | 0.53 | 2.985 | 0.59 |
| fineweb_edu_10bt | test | training | 1,380,127 | 4.125 | 62 | 1.775 | 3.321 | 4.174 | 0.53 | 2.973 | 0.60 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 6.018 | 411 | 3.074 | 4.331 | 4.957 | 0.71 | 4.127 | 0.74 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.640 | 281 | 2.742 | 4.120 | 4.719 | 0.67 | 3.933 | 0.70 |
| fw2_arb | val | training | 692,916 | 4.772 | 118 | 1.935 | 3.448 | 3.934 | 0.56 | 3.302 | 0.59 |
| fw2_arb | test | training | 855,059 | 4.916 | 136 | 2.015 | 3.475 | 3.978 | 0.58 | 3.322 | 0.61 |
| fw2_ars | val | training | 189,459 | 5.688 | 295 | 2.660 | 3.990 | 4.539 | 0.67 | 3.803 | 0.70 |
| fw2_ars | test | training | 159,996 | 5.639 | 281 | 2.775 | 4.158 | 4.775 | 0.67 | 3.979 | 0.70 |
| fw2_ary | val | training | 64,808 | 5.119 | 167 | 2.022 | 3.419 | 3.832 | 0.59 | 3.255 | 0.62 |
| fw2_ary | test | training | 203,565 | 5.238 | 188 | 1.946 | 3.272 | 3.606 | 0.59 | 3.110 | 0.63 |
| fw2_arz | val | training | 571,707 | 5.157 | 174 | 2.279 | 3.761 | 4.289 | 0.61 | 3.605 | 0.63 |
| fw2_arz | test | training | 729,537 | 5.107 | 165 | 2.295 | 3.829 | 4.361 | 0.60 | 3.660 | 0.63 |
| wiki_ar | val | training | 277,136 | 4.373 | 79 | 1.866 | 3.585 | 4.140 | 0.52 | 3.426 | 0.54 |
| wiki_ar | test | training | 319,077 | 4.509 | 91 | 1.894 | 3.532 | 4.077 | 0.54 | 3.369 | 0.56 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.747 | 313 | 3.251 | 4.695 | 5.489 | 0.69 | 4.487 | 0.72 |
