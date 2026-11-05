# Evaluation of runs/ablation-english-20-long/best.pt

Checkpoint step 5000, 17,109,888 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.795 | 44 | 1.632 | 3.330 | 4.174 | 0.49 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.839 | 46 | 1.651 | 3.321 | 4.174 | 0.50 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.508 | 247 | 2.814 | 4.331 | 4.957 | 0.65 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.130 | 169 | 2.494 | 4.120 | 4.719 | 0.61 |
| fw2_arb | val | training | 692,916 | 4.421 | 83 | 1.793 | 3.448 | 3.934 | 0.52 |
| fw2_arb | test | training | 855,059 | 4.556 | 95 | 1.867 | 3.475 | 3.978 | 0.54 |
| fw2_ars | val | training | 189,459 | 5.249 | 190 | 2.455 | 3.990 | 4.539 | 0.62 |
| fw2_ars | test | training | 159,996 | 5.127 | 168 | 2.523 | 4.158 | 4.775 | 0.61 |
| fw2_ary | val | training | 64,808 | 4.624 | 102 | 1.826 | 3.419 | 3.832 | 0.53 |
| fw2_ary | test | training | 203,565 | 4.662 | 106 | 1.732 | 3.272 | 3.606 | 0.53 |
| fw2_arz | val | training | 571,707 | 4.679 | 108 | 2.068 | 3.761 | 4.289 | 0.55 |
| fw2_arz | test | training | 729,537 | 4.628 | 102 | 2.080 | 3.829 | 4.361 | 0.54 |
| wiki_ar | val | training | 277,136 | 4.036 | 57 | 1.722 | 3.585 | 4.140 | 0.48 |
| wiki_ar | test | training | 319,077 | 4.188 | 66 | 1.760 | 3.532 | 4.077 | 0.50 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.081 | 161 | 2.874 | 4.695 | 5.489 | 0.61 |
