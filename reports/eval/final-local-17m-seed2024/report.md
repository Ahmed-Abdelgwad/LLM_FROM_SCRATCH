# Evaluation of runs/local-17m-seed2024/best.pt

Checkpoint step 5000, 17,109,888 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.626 | 38 | 1.560 | 3.330 | 4.174 | 0.47 | 2.985 | 0.52 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.672 | 39 | 1.580 | 3.321 | 4.174 | 0.48 | 2.973 | 0.53 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.566 | 261 | 2.843 | 4.331 | 4.957 | 0.66 | 4.127 | 0.69 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.222 | 185 | 2.539 | 4.120 | 4.719 | 0.62 | 3.933 | 0.65 |
| fw2_arb | val | training | 692,916 | 4.532 | 93 | 1.837 | 3.448 | 3.934 | 0.53 | 3.302 | 0.56 |
| fw2_arb | test | training | 855,059 | 4.659 | 106 | 1.910 | 3.475 | 3.978 | 0.55 | 3.322 | 0.57 |
| fw2_ars | val | training | 189,459 | 5.332 | 207 | 2.494 | 3.990 | 4.539 | 0.63 | 3.803 | 0.66 |
| fw2_ars | test | training | 159,996 | 5.210 | 183 | 2.564 | 4.158 | 4.775 | 0.62 | 3.979 | 0.64 |
| fw2_ary | val | training | 64,808 | 4.759 | 117 | 1.879 | 3.419 | 3.832 | 0.55 | 3.255 | 0.58 |
| fw2_ary | test | training | 203,565 | 4.786 | 120 | 1.778 | 3.272 | 3.606 | 0.54 | 3.110 | 0.57 |
| fw2_arz | val | training | 571,707 | 4.790 | 120 | 2.117 | 3.761 | 4.289 | 0.56 | 3.605 | 0.59 |
| fw2_arz | test | training | 729,537 | 4.743 | 115 | 2.132 | 3.829 | 4.361 | 0.56 | 3.660 | 0.58 |
| wiki_ar | val | training | 277,136 | 4.141 | 63 | 1.767 | 3.585 | 4.140 | 0.49 | 3.426 | 0.52 |
| wiki_ar | test | training | 319,077 | 4.293 | 73 | 1.804 | 3.532 | 4.077 | 0.51 | 3.369 | 0.54 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.152 | 173 | 2.914 | 4.695 | 5.489 | 0.62 | 4.487 | 0.65 |
