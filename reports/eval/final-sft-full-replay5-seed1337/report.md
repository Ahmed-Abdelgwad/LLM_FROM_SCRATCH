# Evaluation of runs/sft-full-replay5-seed1337/merged.pt

Checkpoint step 950, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.877 | 48 | 1.668 | 3.330 | 4.174 | 0.50 | 2.985 | 0.56 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.926 | 51 | 1.689 | 3.321 | 4.174 | 0.51 | 2.973 | 0.57 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.819 | 337 | 2.972 | 4.331 | 4.957 | 0.69 | 4.127 | 0.72 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.448 | 232 | 2.649 | 4.120 | 4.719 | 0.64 | 3.933 | 0.67 |
| fw2_arb | val | training | 692,916 | 4.654 | 105 | 1.887 | 3.448 | 3.934 | 0.55 | 3.302 | 0.57 |
| fw2_arb | test | training | 855,059 | 4.795 | 121 | 1.966 | 3.475 | 3.978 | 0.57 | 3.322 | 0.59 |
| fw2_ars | val | training | 189,459 | 5.524 | 251 | 2.584 | 3.990 | 4.539 | 0.65 | 3.803 | 0.68 |
| fw2_ars | test | training | 159,996 | 5.433 | 229 | 2.673 | 4.158 | 4.775 | 0.64 | 3.979 | 0.67 |
| fw2_ary | val | training | 64,808 | 4.933 | 139 | 1.948 | 3.419 | 3.832 | 0.57 | 3.255 | 0.60 |
| fw2_ary | test | training | 203,565 | 5.013 | 150 | 1.862 | 3.272 | 3.606 | 0.57 | 3.110 | 0.60 |
| fw2_arz | val | training | 571,707 | 4.972 | 144 | 2.197 | 3.761 | 4.289 | 0.58 | 3.605 | 0.61 |
| fw2_arz | test | training | 729,537 | 4.934 | 139 | 2.218 | 3.829 | 4.361 | 0.58 | 3.660 | 0.61 |
| wiki_ar | val | training | 277,136 | 4.291 | 73 | 1.831 | 3.585 | 4.140 | 0.51 | 3.426 | 0.53 |
| wiki_ar | test | training | 319,077 | 4.416 | 83 | 1.855 | 3.532 | 4.077 | 0.53 | 3.369 | 0.55 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.566 | 261 | 3.148 | 4.695 | 5.489 | 0.67 | 4.487 | 0.70 |
