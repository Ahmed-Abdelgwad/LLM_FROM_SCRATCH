# Evaluation of runs/sft-full-replay20-seed1337/merged.pt

Checkpoint step 950, 16,913,280 parameters. Every non-overlapping window of each split is scored. *Bits per character* is the figure to compare across sources; *unigram* knows only token frequencies of the training data (both languages pooled), *same-language unigram* only those of the training sources in the source's own language (the fair baseline, D-026), and *uniform* knows nothing.

| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char | uniform bits/char | model / unigram | same-language unigram bits/char | model / same-language unigram |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | training | 1,548,223 | 3.664 | 39 | 1.576 | 3.330 | 4.174 | 0.47 | 2.985 | 0.53 |
| fineweb_edu_10bt | test | training | 1,380,127 | 3.717 | 41 | 1.599 | 3.321 | 4.174 | 0.48 | 2.973 | 0.54 |
| fw2_acm | eval | evaluation-only | 5,470,966 | 5.717 | 304 | 2.920 | 4.331 | 4.957 | 0.67 | 4.127 | 0.71 |
| fw2_apc | eval | evaluation-only | 4,472,543 | 5.327 | 206 | 2.590 | 4.120 | 4.719 | 0.63 | 3.933 | 0.66 |
| fw2_arb | val | training | 692,916 | 4.550 | 95 | 1.845 | 3.448 | 3.934 | 0.54 | 3.302 | 0.56 |
| fw2_arb | test | training | 855,059 | 4.682 | 108 | 1.919 | 3.475 | 3.978 | 0.55 | 3.322 | 0.58 |
| fw2_ars | val | training | 189,459 | 5.412 | 224 | 2.532 | 3.990 | 4.539 | 0.63 | 3.803 | 0.67 |
| fw2_ars | test | training | 159,996 | 5.307 | 202 | 2.612 | 4.158 | 4.775 | 0.63 | 3.979 | 0.66 |
| fw2_ary | val | training | 64,808 | 4.790 | 120 | 1.892 | 3.419 | 3.832 | 0.55 | 3.255 | 0.58 |
| fw2_ary | test | training | 203,565 | 4.829 | 125 | 1.794 | 3.272 | 3.606 | 0.55 | 3.110 | 0.58 |
| fw2_arz | val | training | 571,707 | 4.844 | 127 | 2.141 | 3.761 | 4.289 | 0.57 | 3.605 | 0.59 |
| fw2_arz | test | training | 729,537 | 4.801 | 122 | 2.158 | 3.829 | 4.361 | 0.56 | 3.660 | 0.59 |
| wiki_ar | val | training | 277,136 | 4.182 | 66 | 1.784 | 3.585 | 4.140 | 0.50 | 3.426 | 0.52 |
| wiki_ar | test | training | 319,077 | 4.318 | 75 | 1.814 | 3.532 | 4.077 | 0.51 | 3.369 | 0.54 |
| wiki_ary | eval | evaluation-only | 2,232,171 | 5.471 | 238 | 3.095 | 4.695 | 5.489 | 0.66 | 4.487 | 0.69 |
