---
layout: post
title: "FlashSpec: Exact speculative sampling without materialized draft/target probability tensors"
date: 2026-04-21 10:00:00+0800
description: A series of specialized kernels designed to accelerate the speculative sampling process.
tags: [speculative-decoding, sampling]
categories: [blog]
related_posts: false
giscus_comments: false
---

# The Memory I/O Bottleneck in SD

Speculative Decoding has gained widespread popularity because it can significantly accelerate LLM inference without losing any output quality. However, when the model's vocabulary size is large, the expected speedup drops significantly.

This is mainly because a large vocabulary introduces a huge memory I/O (read/write) burden. Specifically, traditional frameworks suffer from severe inefficiencies in two main areas:

**1. The Costly Materialization of Logits and Probs (Draft & Verify Stages)**

In both the draft and verify stages, standard speculative sampling explicitly computes the full `logits` and `probabilities (probs)` tensors. Take generating a single token in the draft stage as an example, the resulting memory access pattern is as follows:

```python
# Traditional Draft Sampling (High Memory Traffic)
logits = matmul(hidden_states, LM_head)  # WRITE [V] to GPU Memory (HBM)
probs = softmax(logits)                  # READ [V], WRITE [V] to HBM
token_id = multinomial(probs)            # READ [V], WRITE [1] to HBM
```
For modern models with large vocabularies (like 128k or 256k), repeatedly reading and writing these massive $O(V)$ tensors back and forth creates a severe latency overhead.

**2. The Fragmented Residual Sampling (Resample Stage)**

When a draft token is rejected by the target model, we must resample from a modified probability distribution: `norm(max(P - Q, 0))`. The underlying implementation entails a highly fragmented memory access pattern:

```python
# Traditional Residual Sampling (Severe Kernel Fragmentation)
diff = P - Q                                  # READ P [V], Q [V], WRITE [V] to HBM
truncated = max(diff, 0)                      # READ [V], WRITE [V] to HBM
residual_probs = truncated / sum(truncated)   # READ [V] (x2), WRITE [V] to HBM
token_id = multinomial(residual_probs)        # READ [V], WRITE [1] to HBM
```

Consistent with the first point, these redundant $O(V)$ memory transfers incur substantial latency overhead.

To eliminate these memory I/O bottlenecks, we introduce **FlashSpec**. Through SD-specific kernel simplification, metadata reuse, and kernel fusion, FlashSpec largely avoids explicitly reading and writing these large tensors to the GPU memory.

# FlashSpec

To completely shatter the memory bandwidth wall, FlashSpec deeply refactors every stage of Speculative Decoding (SD). Before diving into our method, let's briefly review a key mathematical concept: the Gumbel-Max Trick.

## Prerequisite: Efficient Sampling via Gumbel Noise

In standard sampling pipelines, we typically apply a Softmax operation over the logits to obtain a normalized probability distribution, and then perform multinomial sampling. Given the logits $x_i$, the standard approach computes the Softmax probabilities $p_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$.

However, by utilizing the **Gumbel-Max Trick**, we can perform equivalent sampling directly from the **unnormalized** logits. Specifically, we add independent and identically distributed Gumbel noise $g_i \sim \text{Gumbel}(0, 1)$ to each logit $x_i$, and simply take the `argmax` of the resulting array:

$$k = \arg\max_i (x_i + g_i)$$

This trick is crucial because it allows us to draw a sample without ever computing the partition function (the denominator $\sum_j \exp(x_j)$).

## Draft Stage: FlashSampling-style Lazy Draft

During the draft stage, the draft model consecutively generates $\gamma$ tokens. In traditional SD implementations, it is common practice to instantiate the full logits and probs tensors for each step. But do we really need the full `probs` tensor?

If we break down the SD pipeline, the draft `probs` tensor is only used in three places:
1. **Draft Stage**: To sample the draft tokens.
2. **Verify Stage**: In the acceptance criterion, to calculate the ratio $P_{\text{target}}(x) / P_{\text{draft}}(x)$.
3. **Resample Stage**: If a token is rejected, to compute the residual distribution $\text{norm}(\max(P_{\text{target}} - P_{\text{draft}}, 0))$ for resampling.

However, generating and preserving the full `probs` tensor is extremely wasteful:
1. As demonstrated by the Gumbel-Max trick (and similar to the FlashSampling approach), we can sample without materializing the logits and probs at all.
2. In the Verify stage, out of that massive $V$-dimensional `probs` vector, **only the probability of the sampled draft token (exactly 1 scalar value)** is used. The remaining $V-1$ values are completely ignored.
3. In the Resample stage, the probability distribution of **at most one** draft step is needed. If all draft tokens are accepted, this tensor is never used. Even if a rejection occurs, we only need the distribution corresponding to the *first* rejected token. Computing the full distributions for all $\gamma$ steps upfront wastes a massive amount of memory I/O.

Based on these insights, we designed an **SD-specific Lazy Draft Kernel** in FlashSpec. It introduces two core improvements:

**1. Fused Sampling with Metadata Extraction**

We implemented a FlashSampling-style fused kernel that reads directly from the Hidden States. Inside the GPU SRAM, it performs chunked matrix multiplications, adds Gumbel noise, and finds the local maximum. 
The key difference from standard FlashSampling is that **our kernel returns two extra critical scalars: the raw `logit` of the selected token and the local `LogSumExp (LSE)` of the current step.** With these two scalars, we can effortlessly compute the exact generation probability of that token during the verify stage ($\log P = \text{logit} - \text{LSE}$) at zero cost.

The pseudo-code for the algorithm is as follows:

```python
# Pseudo-code: FlashSpec Draft Kernel (Tile-level in SRAM)
def flashspec_lazy_draft_kernel(hidden_state, LM_head):
    global_best_score = -inf
    global_best_token = -1
    global_max = -inf
    global_sumexp = 0.0
    global_selected_logit = 0.0

    # Split the massive vocabulary into tiles; all operations stay in ultra-fast SRAM
    for tile_W in split_into_tiles(LM_head):
        # 1. Compute local logits
        logits_tile = dot(hidden_state, tile_W) 
        
        # 2. Local LSE statistics (for later reuse)
        local_max = max(logits_tile)
        local_sumexp = sum(exp(logits_tile - local_max))
        global_sumexp = global_sumexp * exp(global_max - max(global_max, local_max)) + \
                        local_sumexp * exp(local_max - max(global_max, local_max))
        global_max = max(global_max, local_max)

        # 3. Local Gumbel-Max Sampling
        noise_tile = generate_gumbel_noise()
        scores_tile = logits_tile + noise_tile
        tile_best_score, tile_best_token = max(scores_tile)

        # 4. Maintain global optimum
        if tile_best_score > global_best_score:
            global_best_score = tile_best_score
            global_best_token = tile_best_token
            global_selected_logit = logits_tile[tile_best_token]

    # Finalize global LSE
    global_lse = log(global_sumexp) + global_max

    # Only write 3 lightweight scalars back to HBM, completely avoiding [V]-dimensional I/O!
    return global_best_token, global_selected_logit, global_lse
```

**2. Lazy Recompute for Resample**

Because the draft stage does not materialize the full logits, we adopt a "Lazy Recompute" strategy for rejections. Specifically, when the resample stage requires the full probability distribution, we perform a lightweight matrix multiplication using the saved hidden state to recompute the logits *solely* for the specific step that was rejected. This trades a negligible amount of compute for a massive reduction in memory bandwidth.