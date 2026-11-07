# Evaluation of runs/ablation-swiglu-only/best.pt

Checkpoint step 5000, 17,109,888 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.599 | 37 | 1.548 | 3.330 | 4.174 | 0.46 | 2.985 | 0.52 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.647 | 38 | 1.569 | 3.321 | 4.174 | 0.47 | 2.973 | 0.53 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.565 | 261 | 2.843 | 4.331 | 4.957 | 0.66 | 4.127 | 0.69 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.204 | 182 | 2.531 | 4.120 | 4.719 | 0.61 | 3.933 | 0.64 |
| fw2_arb | val | training | 692,916 | 4.503 | 90 | 1.826 | 3.448 | 3.934 | 0.53 | 3.302 | 0.55 |
| fw2_arb | test | training | 855,059 | 4.634 | 103 | 1.899 | 3.475 | 3.978 | 0.55 | 3.322 | 0.57 |
| fw2_ars | val | training | 189,459 | 5.319 | 204 | 2.488 | 3.990 | 4.539 | 0.62 | 3.803 | 0.65 |
| fw2_ars | test | training | 159,996 | 5.190 | 180 | 2.554 | 4.158 | 4.775 | 0.61 | 3.979 | 0.64 |
| fw2_ary | val | training | 64,808 | 4.707 | 111 | 1.859 | 3.419 | 3.832 | 0.54 | 3.255 | 0.57 |
| fw2_ary | test | training | 203,565 | 4.731 | 113 | 1.758 | 3.272 | 3.606 | 0.54 | 3.110 | 0.57 |
| fw2_arz | val | training | 571,707 | 4.766 | 117 | 2.106 | 3.761 | 4.289 | 0.56 | 3.605 | 0.58 |
| fw2_arz | test | training | 729,537 | 4.719 | 112 | 2.121 | 3.829 | 4.361 | 0.55 | 3.660 | 0.58 |
| wiki_ar | val | training | 277,136 | 4.114 | 61 | 1.755 | 3.585 | 4.140 | 0.49 | 3.426 | 0.51 |
| wiki_ar | test | training | 319,077 | 4.267 | 71 | 1.793 | 3.532 | 4.077 | 0.51 | 3.369 | 0.53 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.220 | 185 | 2.953 | 4.695 | 5.489 | 0.63 | 4.487 | 0.66 |
