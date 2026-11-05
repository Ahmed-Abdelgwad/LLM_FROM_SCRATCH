# Same-language unigram baselines (D-026)

Bits per character. *pooled unigram* is the baseline in the original reports (train splits of both
languages); *same-language unigram* counts only the training sources of the row's language.

## final-pilot-17m (step 1500)

| source | split | model | pooled unigram | model / pooled | same-language unigram | model / same-language |
|---|---|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | 1.914 | 3.330 | 0.57 | 2.985 | 0.64 |
| fineweb_edu_10bt | test | 1.924 | 3.321 | 0.58 | 2.973 | 0.65 |
| fw2_acm | eval | 3.212 | 4.331 | 0.74 | 4.127 | 0.78 |
| fw2_apc | eval | 2.949 | 4.120 | 0.72 | 3.933 | 0.75 |
| fw2_arb | val | 2.226 | 3.448 | 0.65 | 3.302 | 0.67 |
| fw2_arb | test | 2.276 | 3.475 | 0.65 | 3.322 | 0.69 |
| fw2_ars | val | 2.871 | 3.990 | 0.72 | 3.803 | 0.75 |
| fw2_ars | test | 2.917 | 4.158 | 0.70 | 3.979 | 0.73 |
| fw2_ary | val | 2.248 | 3.419 | 0.66 | 3.255 | 0.69 |
| fw2_ary | test | 2.085 | 3.272 | 0.64 | 3.110 | 0.67 |
| fw2_arz | val | 2.553 | 3.761 | 0.68 | 3.605 | 0.71 |
| fw2_arz | test | 2.580 | 3.829 | 0.67 | 3.660 | 0.71 |
| wiki_ar | val | 2.191 | 3.585 | 0.61 | 3.426 | 0.64 |
| wiki_ar | test | 2.211 | 3.532 | 0.63 | 3.369 | 0.66 |
| wiki_ary | eval | 3.664 | 4.695 | 0.78 | 4.487 | 0.82 |

| group (mean) | model | pooled unigram | model / pooled | same-language unigram | model / same-language |
|---|---:|---:|---:|---:|---:|
| trained Arabic, val | 2.418 | 3.641 | 0.66 | 3.478 | 0.70 |
| unseen dialects | 3.275 | 4.382 | 0.75 | 4.183 | 0.78 |
| English, val | 1.914 | 3.330 | 0.57 | 2.985 | 0.64 |

## final-local-17m (step 5000)

| source | split | model | pooled unigram | model / pooled | same-language unigram | model / same-language |
|---|---|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | 1.557 | 3.330 | 0.47 | 2.985 | 0.52 |
| fineweb_edu_10bt | test | 1.576 | 3.321 | 0.47 | 2.973 | 0.53 |
| fw2_acm | eval | 2.841 | 4.331 | 0.66 | 4.127 | 0.69 |
| fw2_apc | eval | 2.535 | 4.120 | 0.62 | 3.933 | 0.64 |
| fw2_arb | val | 1.836 | 3.448 | 0.53 | 3.302 | 0.56 |
| fw2_arb | test | 1.909 | 3.475 | 0.55 | 3.322 | 0.57 |
| fw2_ars | val | 2.494 | 3.990 | 0.63 | 3.803 | 0.66 |
| fw2_ars | test | 2.561 | 4.158 | 0.62 | 3.979 | 0.64 |
| fw2_ary | val | 1.873 | 3.419 | 0.55 | 3.255 | 0.58 |
| fw2_ary | test | 1.773 | 3.272 | 0.54 | 3.110 | 0.57 |
| fw2_arz | val | 2.115 | 3.761 | 0.56 | 3.605 | 0.59 |
| fw2_arz | test | 2.129 | 3.829 | 0.56 | 3.660 | 0.58 |
| wiki_ar | val | 1.764 | 3.585 | 0.49 | 3.426 | 0.51 |
| wiki_ar | test | 1.803 | 3.532 | 0.51 | 3.369 | 0.54 |
| wiki_ary | eval | 2.882 | 4.695 | 0.61 | 4.487 | 0.64 |

| group (mean) | model | pooled unigram | model / pooled | same-language unigram | model / same-language |
|---|---:|---:|---:|---:|---:|
| trained Arabic, val | 2.016 | 3.641 | 0.55 | 3.478 | 0.58 |
| unseen dialects | 2.753 | 4.382 | 0.63 | 4.183 | 0.66 |
| English, val | 1.557 | 3.330 | 0.47 | 2.985 | 0.52 |

## final-ablation-english-20-long (step 5000)

| source | split | model | pooled unigram | model / pooled | same-language unigram | model / same-language |
|---|---|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | 1.632 | 3.330 | 0.49 | 2.985 | 0.55 |
| fineweb_edu_10bt | test | 1.651 | 3.321 | 0.50 | 2.973 | 0.56 |
| fw2_acm | eval | 2.814 | 4.331 | 0.65 | 4.127 | 0.68 |
| fw2_apc | eval | 2.494 | 4.120 | 0.61 | 3.933 | 0.63 |
| fw2_arb | val | 1.793 | 3.448 | 0.52 | 3.302 | 0.54 |
| fw2_arb | test | 1.867 | 3.475 | 0.54 | 3.322 | 0.56 |
| fw2_ars | val | 2.455 | 3.990 | 0.62 | 3.803 | 0.65 |
| fw2_ars | test | 2.523 | 4.158 | 0.61 | 3.979 | 0.63 |
| fw2_ary | val | 1.826 | 3.419 | 0.53 | 3.255 | 0.56 |
| fw2_ary | test | 1.732 | 3.272 | 0.53 | 3.110 | 0.56 |
| fw2_arz | val | 2.068 | 3.761 | 0.55 | 3.605 | 0.57 |
| fw2_arz | test | 2.080 | 3.829 | 0.54 | 3.660 | 0.57 |
| wiki_ar | val | 1.722 | 3.585 | 0.48 | 3.426 | 0.50 |
| wiki_ar | test | 1.760 | 3.532 | 0.50 | 3.369 | 0.52 |
| wiki_ary | eval | 2.874 | 4.695 | 0.61 | 4.487 | 0.64 |

| group (mean) | model | pooled unigram | model / pooled | same-language unigram | model / same-language |
|---|---:|---:|---:|---:|---:|
| trained Arabic, val | 1.973 | 3.641 | 0.54 | 3.478 | 0.57 |
| unseen dialects | 2.727 | 4.382 | 0.62 | 4.183 | 0.65 |
| English, val | 1.632 | 3.330 | 0.49 | 2.985 | 0.55 |
