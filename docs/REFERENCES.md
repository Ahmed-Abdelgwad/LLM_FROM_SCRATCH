# References

The research papers and books behind this repository, what each one supports, and where in the
code it is used. Line numbers refer to the code as of 2026-11-11 (commit `9b4ba51` plus the
uncommitted `finetune/chat.py`).

**How this list was checked.** Every technique was located in the code first, then matched to the
paper that introduced it. Each arXiv identifier was resolved on 2026-11-11 through the Semantic
Scholar API, and its title and year compared with the entry below. Every other link returned
HTTP 200, or a redirect to the publisher for the DOI. Papers with more than three authors are
cited by first author only, because the author order returned by the API was wrong for two of them. A paper is listed only if the code
implements what it describes; a citation does not mean the code follows the paper in every detail,
and each entry says where it departs.

**Three groups:**

1. **Implemented and cited.** The code names the paper in a docstring or comment.
2. **Implemented, not cited.** The code does what the paper describes, but nothing in the
   repository names it. These are the gaps a reader would expect to see filled.
3. **Planned or in progress.** Item 4 (LoRA/SFT) and the GQA part of item 2 in
   [PROBLEM.md](PROBLEM.md).

## 1. Implemented and cited

| Paper | Link | What it supports | Where |
|---|---|---|---|
| Sennrich, Haddow and Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016 | [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) | The BPE training algorithm: repeatedly merge the most frequent adjacent pair. The repository applies it to bytes, as GPT-2 does, not to characters as in the paper | [tokenizer/bpe.py:1](../src/scratchllm/tokenizer/bpe.py#L1), `train_merges` at [:64](../src/scratchllm/tokenizer/bpe.py#L64), `train_bpe` at [:118](../src/scratchllm/tokenizer/bpe.py#L118); decisions D-014, D-015 |
| Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI technical report, 2019 | [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Byte-level BPE and the pre-tokenization pattern, cited in the code. Also used without citing it at those places: normalization before each sublayer, the residual-scaling of the initialization, and the architecture of the baseline model | Cited: [tokenizer/pretokenize.py:4](../src/scratchllm/tokenizer/pretokenize.py#L4), pattern at [:23](../src/scratchllm/tokenizer/pretokenize.py#L23). Named as "GPT-2" only: [model/gpt.py:9](../src/scratchllm/model/gpt.py#L9) (pre-norm), [:137](../src/scratchllm/model/gpt.py#L137) (init scaled by 1/sqrt(2 n_layers)); D-014, D-019 |
| Rust et al. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021 | [arXiv:2012.15613](https://arxiv.org/abs/2012.15613) | Tokenizer fertility, tokens per word, the primary tokenizer metric | [tokenizer/fertility.py:3](../src/scratchllm/tokenizer/fertility.py#L3), `measure` at [:51](../src/scratchllm/tokenizer/fertility.py#L51); [tokenizer/report.py](../src/scratchllm/tokenizer/report.py); D-016, D-017 |
| Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing 2024 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | Rotary position embeddings, part of the reference architecture. The code pairs dimension i with i + head_dim/2, the GPT-NeoX/Llama layout, not the interleaved pairs of the paper; the two are equivalent up to a fixed permutation of dimensions | [model/attention.py:42](../src/scratchllm/model/attention.py#L42), `rotary_cos_sin` at [:57](../src/scratchllm/model/attention.py#L57), `apply_rotary` at [:76](../src/scratchllm/model/attention.py#L76); D-027, D-035 |
| Zhang and Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019 | [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | RMSNorm. Built as a switch, tested and dropped from the reference architecture | [model/layers.py:46](../src/scratchllm/model/layers.py#L46), citation at [:50](../src/scratchllm/model/layers.py#L50); D-027, D-034, D-036 |
| Shazeer. *GLU Variants Improve Transformer.* 2020 | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | The SwiGLU feed-forward, and its width chosen to keep the parameter count (the paper's 2/3 rule, here rounded to a multiple of 64) | [model/layers.py:98](../src/scratchllm/model/layers.py#L98) `swiglu_hidden`, [:108](../src/scratchllm/model/layers.py#L108) `SwiGLUFeedForward`, citation at [:109](../src/scratchllm/model/layers.py#L109); D-027, D-033 |
| McCandlish et al. *An Empirical Model of Large-Batch Training.* 2018 | [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) | The gradient noise scale, used to judge whether the batch size was too small | [scripts/diagnose_checkpoint.py:92](../scripts/diagnose_checkpoint.py#L92); [scripts/archive/d023_entry.py:364](../scripts/archive/d023_entry.py#L364); D-023 |
| Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla). NeurIPS 2022 | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | The rule of thumb of about 20 tokens per parameter, and the fitted loss law used to put the 5,000-step result in scale. Analysis only, no code | [DECISIONS.md:766](DECISIONS.md) (D-017), D-023 at [DECISIONS.md:1224](DECISIONS.md); [scripts/archive/d023_entry.py:394](../scripts/archive/d023_entry.py#L394) |

Also cited by name, as the course the project follows: Raschka, *Build a Large Language Model
(From Scratch)*, Manning 2024, [publisher page](https://www.manning.com/books/build-a-large-language-model-from-scratch).
Named in [README.md](../README.md), [model/attention.py:1](../src/scratchllm/model/attention.py#L1),
[model/layers.py:1](../src/scratchllm/model/layers.py#L1), [model/gpt.py:1](../src/scratchllm/model/gpt.py#L1),
[training/trainer.py:1](../src/scratchllm/training/trainer.py#L1) and
[training/dataset.py:6](../src/scratchllm/training/dataset.py#L6), by chapter and without a link.
Its chapter 7 (instruction fine-tuning) and appendix E (LoRA) are the book's counterparts of
`finetune/`. Its LoRA layer, in the [companion notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb),
initializes A with Kaiming-uniform rather than the paper's Gaussian, and scales by alpha / rank
(the printed chapter scaled by alpha only); chapter 7 leaves masking the instruction as an exercise.

## 2. Implemented, not cited

### Model

| Paper | Link | What it supports | Where |
|---|---|---|---|
| Vaswani et al. *Attention Is All You Need.* NeurIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Scaled dot-product attention, multiple heads, the causal mask of the decoder, the transformer block | [model/attention.py:88](../src/scratchllm/model/attention.py#L88) `CausalSelfAttention`, explicit mask at [:196](../src/scratchllm/model/attention.py#L196); D-018 |
| Radford et al. *Improving Language Understanding by Generative Pre-Training* (GPT). OpenAI, 2018 | [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | The decoder-only language model trained on next-token prediction, which GPT-2 inherits | [model/gpt.py:119](../src/scratchllm/model/gpt.py#L119) `GPT`; D-019 |
| Ba, Kiros and Hinton. *Layer Normalization.* 2016 | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) | LayerNorm, kept in the reference architecture | [model/layers.py:17](../src/scratchllm/model/layers.py#L17); D-019, D-036 |
| Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML 2020 | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) | Why normalizing before each sublayer (pre-norm) trains more stably than after it. The docstring states this claim without its source | [model/gpt.py:9](../src/scratchllm/model/gpt.py#L9), [:115](../src/scratchllm/model/gpt.py#L115) |
| Hendrycks and Gimpel. *Gaussian Error Linear Units (GELUs).* 2016 | [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | GELU, in the tanh approximation the paper gives and GPT-2 uses | [model/layers.py:67](../src/scratchllm/model/layers.py#L67), formula at [:76](../src/scratchllm/model/layers.py#L76) |
| Press and Wolf. *Using the Output Embedding to Improve Language Models.* EACL 2017 | [arXiv:1608.05859](https://arxiv.org/abs/1608.05859) | Weight tying: the output map is the token embedding matrix | [model/gpt.py:132](../src/scratchllm/model/gpt.py#L132) |
| Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | The `fast_attention` path calls PyTorch's `scaled_dot_product_attention`, which can dispatch to a FlashAttention kernel. The repository does not implement the algorithm; it uses PyTorch's | [model/attention.py:138](../src/scratchllm/model/attention.py#L138), [:166](../src/scratchllm/model/attention.py#L166)–[:171](../src/scratchllm/model/attention.py#L171) |

### Training

| Paper | Link | What it supports | Where |
|---|---|---|---|
| Loshchilov and Hutter. *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019 | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | The optimizer. The rule that skips decay on norms and biases is common practice, not from the paper | [training/trainer.py:165](../src/scratchllm/training/trainer.py#L165) `make_optimizer`, [:177](../src/scratchllm/training/trainer.py#L177) |
| Loshchilov and Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017 | [arXiv:1608.03983](https://arxiv.org/abs/1608.03983) | Cosine decay of the learning rate. Used once, with no restarts, after a linear warmup | [training/schedule.py](../src/scratchllm/training/schedule.py) `lr_at` |
| Pascanu, Mikolov and Bengio. *On the Difficulty of Training Recurrent Neural Networks.* ICML 2013 | [arXiv:1211.5063](https://arxiv.org/abs/1211.5063) | Clipping the global gradient norm | [training/trainer.py:390](../src/scratchllm/training/trainer.py#L390) |
| Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training.* 2019 | [arXiv:1905.12322](https://arxiv.org/abs/1905.12322) | Training under bfloat16 without loss scaling | [training/trainer.py:153](../src/scratchllm/training/trainer.py#L153) `autocast_factory`; D-020 |

### Generation

| Paper | Link | What it supports | Where |
|---|---|---|---|
| Fan, Lewis and Dauphin. *Hierarchical Neural Story Generation.* ACL 2018 | [arXiv:1805.04833](https://arxiv.org/abs/1805.04833) | Top-k sampling | [model/generate.py:65](../src/scratchllm/model/generate.py#L65) `_pick`, [:72](../src/scratchllm/model/generate.py#L72)–[:75](../src/scratchllm/model/generate.py#L75); D-022 |

### Data

| Paper | Link | What it supports | Where |
|---|---|---|---|
| Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 | [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) | The English source, FineWeb-Edu `sample-10BT` | [data/sources.py:68](../src/scratchllm/data/sources.py#L68); [docs/data/SOURCES.md](data/SOURCES.md) |
| Penedo et al. *FineWeb2: One Pipeline to Scale Them All.* 2025 | [arXiv:2506.20920](https://arxiv.org/abs/2506.20920) | The Arabic web sources (Standard, Egyptian, Moroccan, Najdi, Levantine, Mesopotamian), and their language-ID labels that the EDA tests instead of trusting | [data/sources.py:24](../src/scratchllm/data/sources.py#L24); [docs/data/SOURCES.md](data/SOURCES.md) |
| Broder. *On the Resemblance and Containment of Documents.* Compression and Complexity of Sequences 1997 | [DOI:10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900) | MinHash: the probability that two signatures agree at a position equals the Jaccard similarity of the shingle sets | [eda/checks_learner.py:287](../src/scratchllm/eda/checks_learner.py#L287) `word_shingles`, [:297](../src/scratchllm/eda/checks_learner.py#L297) `NearDuplicates`; [eda/leakage.py:62](../src/scratchllm/eda/leakage.py#L62)–[:83](../src/scratchllm/eda/leakage.py#L83); D-012 |
| Leskovec, Rajaraman and Ullman. *Mining of Massive Datasets*, chapter 3. Cambridge University Press | [mmds.org](http://www.mmds.org/) | LSH banding over the MinHash signatures, to find candidate pairs without comparing every pair. A book, not a paper | [eda/checks_learner.py:304](../src/scratchllm/eda/checks_learner.py#L304) |

Wikipedia (`wikimedia/wikipedia`) is also a source; it has no paper.

### Techniques with no single originating paper

These are used and named in the code, but no one paper introduced them, so none is listed:

- **Key/value cache** for generation, [model/cache.py:22](../src/scratchllm/model/cache.py#L22) (D-030).
- **Bits per character** for comparing models across tokenizers, [training/batches.py:181](../src/scratchllm/training/batches.py#L181) (D-007, D-023).
- **Learning-rate warmup** and **temperature sampling**.

## 3. Planned or in progress

| Paper | Link | What it would support | State |
|---|---|---|---|
| Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | Fine-tuning through a trainable low-rank change beside frozen weights, B initialized at zero so training starts from the base model | **Not built.** [PROBLEM.md](PROBLEM.md) item 4. `src/scratchllm/finetune/lora.py` does not exist |
| Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT). NeurIPS 2022 | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) | Supervised fine-tuning on demonstrations, the first of its three stages; its observation that SFT "overfit[s] on validation loss after 1 epoch" (section 3.5). The paper does not say whether the prompt is masked from the loss, so it is not the source of that choice. The reward model and RL stages are not planned | **In progress, uncommitted.** [finetune/chat.py](../src/scratchllm/finetune/chat.py) |
| Shi et al. *Instruction Tuning With Loss Over Instructions.* NeurIPS 2024 | [arXiv:2405.14394](https://arxiv.org/abs/2405.14394) | The two losses: on the response only (their equation 2, used here) and on prompt and response without the template tokens (equation 4). They find the second helps most with short outputs or few examples | **In progress, uncommitted.** The loss mask in [finetune/chat.py](../src/scratchllm/finetune/chat.py) |
| Singh et al. *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.* ACL 2024 | [arXiv:2402.06619](https://arxiv.org/abs/2402.06619), [dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | SFT data: human-written, Apache 2.0. The Standard Arabic, Egyptian and English subsets are used; the Moroccan subset is not (median response 19 characters) | **In progress** |
| Alyafeai et al. *CIDAR: Culturally Relevant Instruction Dataset For Arabic.* ACL 2024 | [arXiv:2402.03177](https://arxiv.org/abs/2402.03177), [dataset](https://huggingface.co/datasets/arbml/CIDAR) | SFT data replacing Aya's Moroccan subset: 10,000 Arabic pairs, an Alpaca subset translated with ChatGPT plus 891 native grammar questions, all reviewed by people. Card license Apache 2.0; the Alpaca source is CC BY-NC 4.0 | **In progress** |
| Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023 | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | Grouped-query attention: several query heads share one key/value head, shrinking the KV cache | **Not built.** [PROBLEM.md](PROBLEM.md) item 2 |
| Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019 | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | Multi-query attention, the one-group extreme of GQA | **Not built**, background for GQA |

## Code the project learned from

Neither is cited by link in the repository.

- Raschka, [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch): the code that accompanies the book above.
- [bakrianoo/jabarti-llm-from-scratch](https://github.com/bakrianoo/jabarti-llm-from-scratch): an Arabic LLM built from scratch, one of the two projects this one set out to combine.
