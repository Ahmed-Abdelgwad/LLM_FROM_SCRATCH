# Human rubric scores

30 prompts, scored blind.

| Model | Prompts | Overall (1-7) | 95% interval | Fails instruction | Incorrect, of the replies judged |
|---|---|---|---|---|---|
| base-1337 (all) | 30 | 1.00 | 1.00 to 1.00 | 100% (30/30) | 100% (30/30 judged) |
| base-1337 (egyptian) | 6 | 1.00 | 1.00 to 1.00 | 100% (6/6) | 100% (6/6 judged) |
| base-1337 (english) | 12 | 1.00 | 1.00 to 1.00 | 100% (12/12) | 100% (12/12 judged) |
| base-1337 (msa) | 12 | 1.00 | 1.00 to 1.00 | 100% (12/12) | 100% (12/12 judged) |
| full-1337 (all) | 30 | 1.17 | 1.03 to 1.30 | 100% (30/30) | 100% (30/30 judged) |
| full-1337 (egyptian) | 6 | 1.17 | 1.00 to 1.50 | 100% (6/6) | 100% (6/6 judged) |
| full-1337 (english) | 12 | 1.08 | 1.00 to 1.25 | 100% (12/12) | 100% (12/12 judged) |
| full-1337 (msa) | 12 | 1.25 | 1.00 to 1.50 | 100% (12/12) | 100% (12/12 judged) |
| lora-1337 (all) | 30 | 1.03 | 1.00 to 1.10 | 100% (30/30) | 100% (29/29 judged; 1 not judged) |
| lora-1337 (egyptian) | 6 | 1.00 | 1.00 to 1.00 | 100% (6/6) | 100% (6/6 judged) |
| lora-1337 (english) | 12 | 1.00 | 1.00 to 1.00 | 100% (12/12) | 100% (11/11 judged; 1 not judged) |
| lora-1337 (msa) | 12 | 1.08 | 1.00 to 1.25 | 100% (12/12) | 100% (12/12 judged) |

| Pair | Mean difference | 95% interval | Wins / ties / losses |
|---|---|---|---|
| base-1337 - full-1337 | -0.17 | -0.30 to -0.03 | 0 / 25 / 5 |
| base-1337 - lora-1337 | -0.03 | -0.10 to +0.00 | 0 / 29 / 1 |
| full-1337 - lora-1337 | +0.13 | +0.03 to +0.27 | 4 / 26 / 0 |
