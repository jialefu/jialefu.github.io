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

To overcome the memory I/O bottleneck, FlashSpec systematically re-architects the speculative sampling process. To understand how we eliminate redundant tensor materialization, let's briefly review a key mathematical concept: the Gumbel-Max Trick.

## Prerequisite: Efficient Sampling via Gumbel Noise

In standard sampling pipelines, we typically apply a Softmax operation over the logits to obtain a normalized probability distribution, and then perform multinomial sampling. Given the logits $x_i$, the standard approach computes the Softmax probabilities $p_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$.

However, by utilizing the **Gumbel-Max Trick**, we can perform equivalent sampling directly from the logits. Specifically, we add independent and identically distributed Gumbel noise $g_i \sim \text{Gumbel}(0, 1)$ to each logit $x_i$, and simply take the `argmax` of the resulting array:

$$k = \arg\max_i (x_i + g_i)$$

This trick is crucial because it allows us to draw a sample without ever computing the partition function (the denominator $\sum_j \exp(x_j)$).

## FlashSpec-Draft: A FlashSampling-style Metadata-Aware Fused Kernel

During the drafting phase, the model autoregressively generates a sequence of $\gamma$ candidate tokens. Conventional implementations typically materialize the full $V$-dimensional logit and probability tensors for every single step. But is this high-resolution view of the entire vocabulary truly necessary?

If we break down the SD pipeline, the draft logits and probability tensor are only used in three places:
1. **Draft Stage**: To sample the next draft token.
2. **Verify Stage**: To retrieve the probability of the specific drafted token, $P_{\text{draft}}(x)$, for the acceptance check.
3. **Resample Stage**: To compute the residual distribution $\text{norm}(\max(P_{\text{target}} - P_{\text{draft}}, 0))$ if a rejection occurs.

However, generating and preserving these full $O(V)$ tensors is extremely wasteful:
1. As demonstrated by the Gumbel-Max trick (and similar to the FlashSampling approach), we can sample without materializing the logits and probs at all.
2. In the Verify stage, out of that massive $V$-dimensional `probs` vector, **only the probability of the sampled draft token (exactly 1 scalar value)** is used. The remaining $V-1$ values are completely ignored.
3. In the Resample stage, the probability distribution of **at most one** draft step is needed. If all draft tokens are accepted, this tensor is never used. Even if a rejection occurs, we only need the distribution corresponding to the *first* rejected token. Computing the full distributions for all $\gamma$ steps upfront wastes a massive amount of memory I/O.

Based on these insights, we developed the **FlashSpec-Draft Kernel**. It introduces two core improvements:

**1. Fused Sampling with Metadata Extraction**

**FlashSpec-Draft** fuses hidden state projection, Gumbel noise injection, and max-tracking into a single SRAM-resident kernel. Unlike standard FlashSampling, it specifically retains the **selected logit** and the **LogSumExp (LSE)** for each step. This minimal metadata allows the subsequent Verify stage to reconstruct **exact probabilities** ($\log P = \text{logit} - \text{LSE}$) with zero additional I/O, bypassing the need to store or re-access the massive $O(V)$ logit tensors.

The implementation logic of the **FlashSpec-Draft** kernel is outlined below:

```python
# Pseudo-code: FlashSpec-Draft Kernel (Metadata-Aware Fused Sampling)
def flashspec_draft_kernel(hidden_state, LM_head):
    # Register-level accumulators
    selected_id = -1
    selected_score = -inf
    selected_logit = 0.0
    
    # Online LSE statistics for numerical stability
    curr_max = -inf
    curr_sumexp = 0.0

    # Iterate through vocabulary tiles (all intermediate steps stay in SRAM)
    for tile_W in tiles(LM_head):
        # 1. On-the-fly Projection: Compute logits for the current tile
        logits_tile = matmul(hidden_state, tile_W) 
        
        # 2. Online LSE Update: Maintains global partition function without full materialization
        tile_max = max(logits_tile)
        new_max = max(curr_max, tile_max)
        curr_sumexp = curr_sumexp * exp(curr_max - new_max) + \
                      sum(exp(logits_tile - new_max))
        curr_max = new_max

        # 3. Fused Gumbel-Max: Sampling integrated into the tile-loop
        scores_tile = logits_tile + generate_gumbel_noise(tile_W.shape)
        tile_best_score, tile_best_idx = max_with_index(scores_tile)

        # 4. Global Reduction: Update the winning token and its raw logit
        if tile_best_score > selected_score:
            selected_score = tile_best_score
            selected_id = global_index(tile_best_idx)
            selected_logit = logits_tile[tile_best_idx]

    # Finalize Metadata: Combine max and sumexp into a single LSE scalar
    lse = log(curr_sumexp) + curr_max

    # I/O Efficiency: Write only 3 scalars back to HBM (O(1) vs O(V))
    return selected_id, selected_logit, lse
```

**2. Lazy Recompute for Resample**

Since the **FlashSpec-Draft** kernel avoids materializing full logits to save I/O, a natural question arises: what happens if the Resample stage actually needs the full distribution?

To resolve this, we adopt a **Lazy Recompute** strategy. If a draft token is rejected, instead of fetching a massive $O(V)$ tensor from HBM (which was never stored anyway), we re-trigger a lightweight projection using the preserved hidden state. 

This process is designed with two key efficiency principles:

1.  **Compute-for-I/O Trade-off**: The recomputation is fused directly within our Resample kernel. The logits are generated and consumed entirely within **SRAM** for immediate resampling; they are never explicitly written back to HBM. This turns a slow memory-bound task into a fast compute-bound one.

2.  **Minimal Triggering**: In speculative decoding, we only need to resample for the **first rejected token** in a sequence. This means that for the vast majority of steps (the accepted ones), no recomputation occurs. Even on a "bad" step, we only perform this for a single token, making the overhead nearly invisible.

*(For a deeper look at the implementation, see the FlashSpec-Resample Kernel section below.)*


## FlashSpec-Verify & Resample

Stay tuned!