# Evaluation of runs/sft-full-replay10-seed1337/merged.pt

Checkpoint step 950, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.776 | 44 | 1.624 | 3.330 | 4.174 | 0.49 | 2.985 | 0.54 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.826 | 46 | 1.646 | 3.321 | 4.174 | 0.50 | 2.973 | 0.55 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.773 | 321 | 2.949 | 4.331 | 4.957 | 0.68 | 4.127 | 0.71 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.391 | 219 | 2.621 | 4.120 | 4.719 | 0.64 | 3.933 | 0.67 |
| fw2_arb | val | training | 692,916 | 4.608 | 100 | 1.868 | 3.448 | 3.934 | 0.54 | 3.302 | 0.57 |
| fw2_arb | test | training | 855,059 | 4.742 | 115 | 1.944 | 3.475 | 3.978 | 0.56 | 3.322 | 0.59 |
| fw2_ars | val | training | 189,459 | 5.476 | 239 | 2.561 | 3.990 | 4.539 | 0.64 | 3.803 | 0.67 |
| fw2_ars | test | training | 159,996 | 5.375 | 216 | 2.645 | 4.158 | 4.775 | 0.64 | 3.979 | 0.66 |
| fw2_ary | val | training | 64,808 | 4.872 | 131 | 1.924 | 3.419 | 3.832 | 0.56 | 3.255 | 0.59 |
| fw2_ary | test | training | 203,565 | 4.925 | 138 | 1.830 | 3.272 | 3.606 | 0.56 | 3.110 | 0.59 |
| fw2_arz | val | training | 571,707 | 4.911 | 136 | 2.170 | 3.761 | 4.289 | 0.58 | 3.605 | 0.60 |
| fw2_arz | test | training | 729,537 | 4.871 | 130 | 2.189 | 3.829 | 4.361 | 0.57 | 3.660 | 0.60 |
| wiki_ar | val | training | 277,136 | 4.242 | 70 | 1.810 | 3.585 | 4.140 | 0.50 | 3.426 | 0.53 |
| wiki_ar | test | training | 319,077 | 4.373 | 79 | 1.837 | 3.532 | 4.077 | 0.52 | 3.369 | 0.55 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.530 | 252 | 3.128 | 4.695 | 5.489 | 0.67 | 4.487 | 0.70 |
