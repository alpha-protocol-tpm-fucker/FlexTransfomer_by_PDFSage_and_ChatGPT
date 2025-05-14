Below is a practical-engineering menu of everything you can tweak in a Transformer’s math to squeeze speed, stability, or context length. Citations mark the key papers, blog posts, or docs you’d inspect before implementing.

---

## TL;DR

* **LayerNorm + Residuals:** You can move, replace, scale, gate, or even drop them. Pre-LN trains faster and is now default; Post-LN still beats it in very deep decoders with **DeepNorm** or **RMSNorm** scalings.
* **Positional Encoding:** Absolute (sinusoid / learned), relative (Shaw-style), rotary (RoPE), bias-only (ALiBi), or hybrid schemes each trade extrapolation vs. parameter cost.
* **Attention Variants:** Full O(n²) → sparse window+global (Longformer, Big Bird, XAttention), low-rank projection (Linformer), kernelized linear (Performer), hashing (Reformer), or mixed block libraries (DeepSpeed SA).
* **Training JSON & regex tokens:** Keep only `{"request": "...", "response": "..."}`; add any task-specific placeholders (e.g. `<TABLE_1>`, `<BOLD>`) to your tokenizer and filter with regex during preprocessing.

---

## 1 Layer Normalization & Residual Connections

| Variant                     | What you change                                       | Why you’d use it                                                                                       |
| --------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Post-LN (original)**      | `y = LN(x + Sublayer(x))`                             | Stable for moderate depth; reference Vaswani et al. 2017 ([NeurIPS Proceedings][1])                    |
| **Pre-LN**                  | `y = x + Sublayer(LN(x))`                             | Removes warm-up, converges faster, more numerically stable on long schedules ([Medium][2], [arXiv][3]) |
| **DeepNorm / ScaleNorm**    | Multiply residual by learned or fixed α before adding | Enables >1 k layers without blow-up; see arXiv 2002.04745 section 4 ([arXiv][3])                       |
| **RMSNorm**                 | L2-norm without centering mean                        | Slightly cheaper, keeps magnitude information, common in LLaMA-family                                  |
| **ReZero / Gated Residual** | Start residual weight at 0 or sigmoid-gate it         | Lets gradients flow early, then gradually trusts residual path                                         |
| **NormFormer additions**    | Extra LayerNorm after attention or FFN outputs        | Empirically boosts BLEU & perplexity for same compute                                                  |

Implementation toggle is literally where you call `nn.LayerNorm` and how you scale the residual term.

---

## 2 Positional / Sequence Encoding Options

1. **Sinusoidal absolute (fixed)** – original formula, no learned parameters; extrapolates but can degrade beyond trained length. ([Medium][4], [NeurIPS Proceedings][1])
2. **Learned absolute embeddings** – treat position like a word ID; best on short, fixed-length inputs. ([NeurIPS Proceedings][1])
3. **Relative position (Shaw)** – add pair-wise bias `a_{ij}` to attention logits; generalises length, widely used in T5/BERT-Style models. ([arXiv][5])
4. **RoPE (Rotary)** – rotate Q/K vectors by angle proportional to position; natural for extrapolation & efficient in FlashAttention-style kernels. ([Medium][6])
5. **ALiBi (Linear Bias)** – no embedding; subtract `m·|i-j|` slope in logits, enabling 32 k+ contexts with no extra memory. ([Medium][7])
6. **Hybrid / Compounded** – mix RoPE inside each head with ALiBi slope across heads for best of both worlds (used in Llama 3).

Choose based on maximum context you need and whether you must load weights into devices with different max-len.

---

## 3 Attention Mechanism Families

| Complexity      | Mechanism                                    | Core trick                                                 | Key refs                             |
| --------------- | -------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| **O(n²)**       | Full                                         | All-pairs dot product                                      | Baseline                             |
| **O(n · w)**    | Window + global (Longformer)                 | Sliding window  + select global tokens                     | ([arXiv][8])                         |
| **O(n · r)**    | Block + random + global (Big Bird)           | Graph-theoretic sparse pattern, proves Turing completeness | ([arXiv][9])                         |
| **≈O(n log n)** | Reformer                                     | LSH buckets similar queries; reversible layers save memory | ([arXiv][10], [Google Research][11]) |
| **O(n)**        | Linformer                                    | Low-rank K/V projection with shared `E∈ℝ^{n×k}`            | ([arXiv][12])                        |
| **O(n)**        | Performer                                    | Kernel trick (FAVOR+) to approximate softmax               | ([arXiv][13])                        |
| **O(s·n)**      | Block-sparse libs (DeepSpeed SA, XAttention) | Hardware-native  or antidiagonal scoring to drop blocks    | ([DeepSpeed][14], [arXiv][15])       |

Practical notes:

* **Window + global**: choose window size `w`; add `[CLS]` or task tokens to `global` set.
* **Block sparse**: pick block size (e.g., 128×128) and pattern (band, dilated, triangular). Use Han-Lab’s kernels for A100/H100 GPUs. ([hanlab.mit.edu][16])
* **Kernel / linear**: must pick positive random features `φ(x)` dimension; `m≈64–256` suffices for 1 % relative error in 2048-token LM. ([arXiv][13])

---

## 4 Special Tokens & Regex Filtering in Training JSON

Minimal record:

```json
{"request": "<USER> What is 5+7?", "response": "<ASSISTANT> 12"}
```

Add task-specific placeholders -- e.g.:

* `<URL>`, `<CODE_BLOCK>`, `<TABLE_#>`, `<IMG_#>` – inserted by a pre-tokeniser.
* Regex such as `/<IMG_\d+>/` lets you mask or down-sample examples during cleaning. ([JSON Schema][17])

Guidelines:

1. Enumerate every special string in tokenizer’s `added_tokens.json`.
2. Enforce the regex during dataset streaming so corrupted lines are skipped, not padded.
3. Keep exactly two keys (`request`, `response`) to simplify SFT/ILQL loaders.

---

## 5 Putting It Together

```python
class FlexTransformer(nn.Module):
    """(∀x)(Encoder(x) → Decoder(x))"""
    def __init__(self, cfg):
        super().__init__()
        self.pos = select_positional(cfg.pos)        # sinusoid / rope / alibi…
        self.blocks = nn.ModuleList([
            FlexBlock(cfg) for _ in range(cfg.L)
        ])
        self.norm_final = maybe_norm(cfg)
    def forward(self, x, **kw):
        x = x + self.pos(x)
        for blk in self.blocks:
            x = blk(x, **kw)
        return self.norm_final(x)
```

Here `FlexBlock` plugs in any attention kernel from §3 and any residual/normalization choice from §1; the positional module comes from §2. Building the menu this way lets you benchmark combinations quickly on your MIT cluster and pick the sweet spot for mission-critical throughput.

---

### Further Hands-On References

Long survey of 2024 designs ([Nathan Brake][18]) • Sparse-attention tutorial and OpenAI code ([llmmodels.org][19]) • Block-sparse patterns explainer ([DEV Community][20]).

Good luck, Bo — may your attentions be fast and your gradients stable!

[1]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf?utm_source=chatgpt.com "[PDF] Attention is All you Need - NIPS papers"
[2]: https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab?utm_source=chatgpt.com "Review — Pre-LN Transformer: On Layer Normalization in the ..."
[3]: https://arxiv.org/pdf/2002.04745?utm_source=chatgpt.com "[PDF] On Layer Normalization in the Transformer Architecture - arXiv"
[4]: https://medium.com/%40shravankoninti/transformers-attention-is-all-you-need-positional-encoding-485dcc1019fe?utm_source=chatgpt.com "Transformers: Attention is all you need — Positional Encoding"
[5]: https://arxiv.org/abs/1803.02155?utm_source=chatgpt.com "Self-Attention with Relative Position Representations"
[6]: https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83?utm_source=chatgpt.com "Rotary Positional Embeddings: A Detailed Look and ... - Medium"
[7]: https://medium.com/%40pajakamy/alibi-attention-with-linear-biases-942abe042e9f?utm_source=chatgpt.com "ALiBi: Attention with Linear Biases | by Amy Pajak - Medium"
[8]: https://arxiv.org/abs/2004.05150?utm_source=chatgpt.com "Longformer: The Long-Document Transformer"
[9]: https://arxiv.org/abs/2007.14062?utm_source=chatgpt.com "Big Bird: Transformers for Longer Sequences"
[10]: https://arxiv.org/abs/2001.04451?utm_source=chatgpt.com "Reformer: The Efficient Transformer"
[11]: https://research.google/blog/reformer-the-efficient-transformer/?utm_source=chatgpt.com "Reformer: The Efficient Transformer - Google Research"
[12]: https://arxiv.org/abs/2006.04768?utm_source=chatgpt.com "[2006.04768] Linformer: Self-Attention with Linear Complexity - arXiv"
[13]: https://arxiv.org/abs/2009.14794?utm_source=chatgpt.com "[2009.14794] Rethinking Attention with Performers - arXiv"
[14]: https://www.deepspeed.ai/tutorials/sparse-attention/?utm_source=chatgpt.com "DeepSpeed Sparse Attention"
[15]: https://arxiv.org/html/2503.16428v1?utm_source=chatgpt.com "XAttention: Block Sparse Attention with Antidiagonal Scoring - arXiv"
[16]: https://hanlab.mit.edu/blog/block-sparse-attention?utm_source=chatgpt.com "Block Sparse Attention - MIT HAN Lab"
[17]: https://json-schema.org/understanding-json-schema/reference/regular_expressions?utm_source=chatgpt.com "Regular Expressions - JSON Schema"
[18]: https://www.natebrake.com/blog/2024/07-24-survey-of-attention?utm_source=chatgpt.com "Survey of Current Modified Transformer Attention Designs"
[19]: https://llmmodels.org/blog/sparse-attention-in-transformers-step-by-step-implementation/?utm_source=chatgpt.com "Sparse Attention in Transformers: Step-by-Step Implementation"
[20]: https://dev.to/nareshnishad/day-29-sparse-transformers-efficient-scaling-for-large-language-models-59j5?utm_source=chatgpt.com "Sparse Transformers: Efficient Scaling for Large Language Models"
