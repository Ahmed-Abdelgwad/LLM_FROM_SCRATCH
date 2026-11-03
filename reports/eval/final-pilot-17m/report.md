# Evaluation of runs/pilot-17m/best.pt

Checkpoint step 1500, 17,109,888 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 4.450 | 86 | 1.914 | 3.330 | 4.174 | 0.57 |
| fineweb_edu_10bt | test | training | 1,380,127 | 4.472 | 88 | 1.924 | 3.321 | 4.174 | 0.58 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 6.289 | 539 | 3.212 | 4.331 | 4.957 | 0.74 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 6.065 | 430 | 2.949 | 4.120 | 4.719 | 0.72 |
| fw2_arb | val | training | 692,916 | 5.491 | 243 | 2.226 | 3.448 | 3.934 | 0.65 |
| fw2_arb | test | training | 855,059 | 5.552 | 258 | 2.276 | 3.475 | 3.978 | 0.65 |
| fw2_ars | val | training | 189,459 | 6.137 | 463 | 2.871 | 3.990 | 4.539 | 0.72 |
| fw2_ars | test | training | 159,996 | 5.928 | 375 | 2.917 | 4.158 | 4.775 | 0.70 |
| fw2_ary | val | training | 64,808 | 5.693 | 297 | 2.248 | 3.419 | 3.832 | 0.66 |
| fw2_ary | test | training | 203,565 | 5.611 | 273 | 2.085 | 3.272 | 3.606 | 0.64 |
| fw2_arz | val | training | 571,707 | 5.776 | 323 | 2.553 | 3.761 | 4.289 | 0.68 |
| fw2_arz | test | training | 729,537 | 5.741 | 311 | 2.580 | 3.829 | 4.361 | 0.67 |
| wiki_ar | val | training | 277,136 | 5.135 | 170 | 2.191 | 3.585 | 4.140 | 0.61 |
| wiki_ar | test | training | 319,077 | 5.261 | 193 | 2.211 | 3.532 | 4.077 | 0.63 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 6.477 | 650 | 3.664 | 4.695 | 5.489 | 0.78 |
