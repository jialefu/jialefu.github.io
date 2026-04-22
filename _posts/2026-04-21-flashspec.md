---
layout: post
title: "FlashSpec: Exact speculative sampling without materialized draft/target probability tensors"
date: 2026-04-21 10:00:00+0800
description: Coming soon.
tags: [speculative-decoding, sampling]
categories: [blog]
related_posts: false
giscus_comments: false
---

# Background

Speculative Decoding has gained widespread popularity because it can significantly accelerate LLM inference without losing any output quality. However, when the model's vocabulary size is large, the expected speedup drops significantly.

This is mainly because a large vocabulary introduces a huge memory I/O (read/write) burden. Specifically, traditional frameworks suffer from severe inefficiencies in two main areas:

**1. The Costly Materialization of Logits and Probs (Draft & Verify Stages)**

In both the draft and verify stages, standard speculative sampling explicitly computes the full `logits` and `probabilities (probs)` tensors. Take generating a single token in the draft stage as an example, the data flow looks like this:

- Read `hidden states` and `LM head` weights -> compute matrix multiplication -> **write** the `logits` tensor to GPU memory (HBM).

- Read `logits` -> compute Softmax -> **write** the `probs` tensor to memory.

- Read `probs` -> perform sampling algorithm -> **write** the final Token ID.

  For modern models with large vocabularies (like 128k or 256k), repeatedly reading and writing these large tensors back and forth creates a severe latency overhead.

**2. The Fragmented Residual Sampling (Resample Stage)**

When a draft token is rejected by the target model, we must resample from a modified probability distribution: `norm(max(P - Q, 0))`. The underlying data flow involves highly fragmented read/write steps:

- Read probabilities `P` and `Q` -> compute the difference `P - Q` -> **write** the difference to memory.

- Read the difference -> apply the `max(..., 0)` truncation -> **write** the truncated distribution.

- Read the truncated distribution -> sum it and normalize -> **write** the new `probs`.

- Read the new `probs` -> perform multinomial sampling -> **write** the final Token ID.

  Similar to the first point, forcing the GPU to move $O(V)$ data in and out of the memory multiple times across these steps causes significant latency overhead.

**Enter FlashSpec**

To eliminate these memory I/O bottlenecks, we introduce **FlashSpec**. Through SD-specific kernel simplification, metadata reuse, and kernel fusion, FlashSpec largely avoids explicitly reading and writing these large tensors to the GPU memory.

# FlashSpec

To completely shatter the memory bandwidth wall, FlashSpec deeply refactors every stage of Speculative Decoding (SD). Before diving into our method, let's briefly review a key mathematical concept: the Gumbel-Max Trick.

**a. Prerequisite: Efficient Sampling via Gumbel Noise**

In standard sampling pipelines, we typically apply a Softmax operation over the logits to obtain a normalized probability distribution, and then perform multinomial sampling. Given the logits $x_i$, the standard approach computes the Softmax probabilities $p_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$.

However, by utilizing the **Gumbel-Max Trick**, we can perform equivalent sampling directly from the **unnormalized** logits. Specifically, we add independent and identically distributed Gumbel noise $g_i \sim \text{Gumbel}(0, 1)$ to each logit $x_i$, and simply take the `argmax` of the resulting array:

$$k = \arg\max_i (x_i + g_i)$$

This trick is crucial because it allows us to draw a sample without ever computing the partition function (the denominator $\sum_j \exp(x_j)$).