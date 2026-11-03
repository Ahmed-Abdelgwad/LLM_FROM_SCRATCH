# Evaluation of runs/local-17m/best.pt

Checkpoint step 5000, 17,109,888 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.619 | 37 | 1.557 | 3.330 | 4.174 | 0.47 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.664 | 39 | 1.576 | 3.321 | 4.174 | 0.47 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.562 | 260 | 2.841 | 4.331 | 4.957 | 0.66 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.213 | 184 | 2.535 | 4.120 | 4.719 | 0.62 |
| fw2_arb | val | training | 692,916 | 4.530 | 93 | 1.836 | 3.448 | 3.934 | 0.53 |
| fw2_arb | test | training | 855,059 | 4.657 | 105 | 1.909 | 3.475 | 3.978 | 0.55 |
| fw2_ars | val | training | 189,459 | 5.332 | 207 | 2.494 | 3.990 | 4.539 | 0.63 |
| fw2_ars | test | training | 159,996 | 5.205 | 182 | 2.561 | 4.158 | 4.775 | 0.62 |
| fw2_ary | val | training | 64,808 | 4.742 | 115 | 1.873 | 3.419 | 3.832 | 0.55 |
| fw2_ary | test | training | 203,565 | 4.773 | 118 | 1.773 | 3.272 | 3.606 | 0.54 |
| fw2_arz | val | training | 571,707 | 4.785 | 120 | 2.115 | 3.761 | 4.289 | 0.56 |
| fw2_arz | test | training | 729,537 | 4.736 | 114 | 2.129 | 3.829 | 4.361 | 0.56 |
| wiki_ar | val | training | 277,136 | 4.135 | 62 | 1.764 | 3.585 | 4.140 | 0.49 |
| wiki_ar | test | training | 319,077 | 4.292 | 73 | 1.803 | 3.532 | 4.077 | 0.51 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.095 | 163 | 2.882 | 4.695 | 5.489 | 0.61 |
