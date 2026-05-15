---
layout: post
title: "FlashSpec (Part I): One-Pass Verify-and-Resample for Greedy Drafts"
date: 2026-05-14 10:00:00+0800
description: A one-pass verify-and-resample for greedy draft speculative decoding that avoids materializing logits, probabilities, and residual distributions.
tags: [speculative-decoding, sampling, vllm, flashspec]
categories: [blog]
related_posts: false
giscus_comments: false
pretty_table: true
---

# The Memory I/O Bottleneck in Speculative Decoding

Speculative Decoding accelerates LLM inference by using a lightweight drafter to propose several future tokens, followed by one target-model forward pass to verify them. When the draft is mostly accepted, one target step can produce multiple output tokens.

However, standard implementations still materialize vocabulary-sized tensors in the draft and verify/resample stages: logits, probabilities, and residual distributions. For vocabularies with 128k, 150k, or 250k tokens, these $O(V)$ reads and writes become a real HBM bottleneck.

FlashSpec asks whether this materialization is necessary: can we verify draft tokens and recover from rejection without writing full logits or probabilities to memory? In Part 1, we study the most structured case, **greedy draft speculative decoding**, where the draft model is greedy while the target model can still sample from its own distribution.

# Greedy Draft Speculative Decoding

In this post, "greedy" only refers to the **draft model**. The target model is still allowed to sample. This is the setting used by many practical speculative decoding systems, for example:

1. **Prompt lookup / n-gram speculative decoding**
2. **MTP draft models**
3. **EAGLE-style draft models**

A conventional greedy draft step looks like this:

```python
# Standard greedy draft step
def standard_greedy_draft(hidden, lm_head):
    logits = hidden @ lm_head.T        # WRITE [V]
    token = argmax(logits)             # READ [V]
    return token
```

A conventional verify/resample stage then computes target probabilities and checks whether each proposed token should be accepted:

```python
# Standard greedy-draft verify/resample
def standard_greedy_draft_verify(target_hidden, target_lm_head, draft_tokens, uniforms):
    target_logits = target_hidden @ target_lm_head.T     # WRITE [gamma, V]
    target_probs = softmax(target_logits)                # READ/WRITE [gamma, V]

    for i in range(gamma):
        x = draft_tokens[i]

        # Since q(x) = 1, the acceptance probability is p(x).
        accept_prob = target_probs[i, x]

        if uniforms[i] <= accept_prob:
            output.append(x)
        else:
            residual_probs = target_probs[i].clone()     # READ/WRITE [V]
            residual_probs[x] = 0
            residual_probs /= residual_probs.sum()
            recovered = sample(residual_probs)           # READ [V]
            output.append(recovered)
            break

    return output
```

This standard implementation is inefficient in two ways. First, both the draft and target stages materialize vocabulary-sized logits and probability tensors. Second, when a token is rejected, we need to construct a residual distribution and sample from it. This requires another pass over a $[V]$-sized vector for masking, renormalization, and sampling, usually through extra GPU kernels.

FlashSpec targets these materializations directly. For greedy drafts, both verification and recovery can be implemented as a single pass over the target vocabulary, without writing full logits, probabilities, or residual probabilities to HBM.

# FlashSpec-Draft: Greedy Argmax without Writing Logits

The draft side is straightforward. Instead of writing full logits to HBM, FlashSpec scans the vocabulary in tiles and keeps only the current maximum token.

```python
# FlashSpec greedy draft
def flashspec_greedy_draft(hidden, lm_head):
    best_token = -1
    best_logit = -inf

    for W_tile, token_range in tiles(lm_head):
        logits = hidden @ W_tile.T       # stays in SRAM/registers
        tile_token, tile_logit = max(logits)

        if tile_logit > best_logit:
            best_logit = tile_logit
            best_token = global_id(tile_token, token_range)

    return best_token                    # WRITE O(1)
```

This removes the full $[V]$ logits write from the draft stage.

# FlashSpec-Verify: Acceptance Needs Only One Probability

For a drafted token $x$, the speculative acceptance probability is:

$$
\alpha(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right).
$$

For greedy drafts, $q(x)=1$, so the rule reduces to:

$$
\alpha(x) = p(x).
$$

In a standard verify implementation, the system applies softmax over the complete vocabulary and then indexes the drafted token to obtain $p(x)$. This softmax is itself a multi-pass operation over the vocabulary: one pass to compute the normalization term, and another pass to write the normalized probabilities.

However, verify only needs $p(x)$. It does not use the target probabilities of the other tokens. This means we can avoid materializing the full probability vector and compute the needed scalar directly. Specifically, the target probability of the drafted token is:

$$
p(x) = \frac{\exp(\ell_x)}{\sum_j \exp(\ell_j)}.
$$

Therefore, during a single pass over the target vocabulary, FlashSpec only needs to record two values:

1. the target logit of the drafted token, $\ell_x$
2. the row log-sum-exp, $\mathrm{LSE} = \log \sum_j \exp(\ell_j)$

Then:

$$
\log p(x) = \ell_x - \mathrm{LSE}.
$$

Concretely, FlashSpec records the drafted token's logit and maintains an online LSE accumulator:

```python
# Target summary needed for acceptance
for W_tile, token_range in tiles(target_lm_head):
    logits = target_hidden @ W_tile.T

    # Online LSE over the full vocabulary.
    update_lse(logits)

    # Save only the drafted token's target logit.
    if draft_token in token_range:
        selected_logit = logits[draft_token]

accept_prob = exp(selected_logit - lse)
```

This gives the exact acceptance probability without writing target logits or target probabilities to HBM.

# One-Pass Residual Sampling with Gumbel-Max

There is still one challenge. If the drafted token is rejected, we need to sample from the residual distribution.

In the greedy-draft case, the residual distribution is simply the target distribution with the drafted token removed and renormalized:

$$
p_{\mathrm{res}}(i) \propto
\begin{cases}
p(i), & i \ne x, \\
0, & i = x.
\end{cases}
$$

Naively, this seems to require materializing the full target probability vector. FlashSpec avoids this with the Gumbel-Max trick.

Sampling from a categorical distribution with logits $\ell_i$ is equivalent to:

$$
y = \arg\max_i(\ell_i + g_i),
\qquad
g_i \sim \mathrm{Gumbel}(0, 1).
$$

Therefore, while scanning the logits for LSE and $\ell_x$, we can also maintain:

$$
y_{\mathrm{recovered}} = \arg\max_{i \ne x}(\ell_i + g_i).
$$

This gives the token that would be sampled from the residual distribution, again without materializing logits, probabilities, or residual probabilities.

# FlashSpec One-Pass Verify-and-Resample

The complete greedy-draft FlashSpec verify/resample kernel looks like this:

```python
# FlashSpec one-pass verify-and-resample for greedy drafts
def flashspec_greedy_draft_verify_resample(
    target_hidden,
    target_lm_head,
    draft_tokens,
    uniforms,
):
    output = []

    for pos in range(gamma):
        x = draft_tokens[pos]
        h = target_hidden[pos]

        selected_logit = -inf

        lse_max = -inf
        lse_sum = 0.0

        recovered_token = -1
        recovered_score = -inf

        for W_tile, token_range in tiles(target_lm_head):
            logits = h @ W_tile.T

            # 1. Online LSE for the denominator of p(x).
            lse_max, lse_sum = online_lse_update(lse_max, lse_sum, logits)

            # 2. Record the target logit at the drafted token x.
            if x in token_range:
                selected_logit = logits[x]

            # 3. In the same scan, sample from target excluding x.
            gumbels = gumbel_noise(token_range)
            scores = logits + gumbels

            if x in token_range:
                scores[x] = -inf

            tile_token, tile_score = max(scores)

            if tile_score > recovered_score:
                recovered_score = tile_score
                recovered_token = global_id(tile_token, token_range)

        lse = log(lse_sum) + lse_max
        accept_prob = exp(selected_logit - lse)

        if uniforms[pos] <= accept_prob:
            output.append(x)
        else:
            output.append(recovered_token)
            break

    return output
```

The important point is that verify and resample are no longer two separate memory-heavy stages. FlashSpec performs both in a **single pass over the vocabulary**:

1. compute $p(x)$ for acceptance,
2. prepare the recovered token in case of rejection,
3. emit either the draft token or the recovered token with a simple branch.

In practice, the bonus token for the all-accepted case can be fused into the same target-side kernel as well. We omit it from the pseudocode here to keep the core idea clean.

# Experiments

We integrated FlashSpec into vLLM's greedy-draft speculative decoding paths and evaluated three representative settings: n-gram, MTP, and EAGLE3. All runs use target sampling with `temperature=1.0`, `top_p=1.0`, batch size 1, and CUDA graph decode-only mode.

Representative results:

<div class="table-responsive">
  <table class="table table-sm table-striped">
    <thead>
      <tr>
        <th>Setting</th>
        <th style="text-align: right;">Stock tok/s</th>
        <th style="text-align: right;">FlashSpec tok/s</th>
        <th style="text-align: right;">Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Qwen3.5-9B n-gram</td>
        <td style="text-align: right;">198.7</td>
        <td style="text-align: right;">217.4</td>
        <td style="text-align: right;">+9.5%</td>
      </tr>
      <tr>
        <td>Qwen3.5-9B MTP6 decode512</td>
        <td style="text-align: right;">292.8</td>
        <td style="text-align: right;">312.7</td>
        <td style="text-align: right;">+6.9%</td>
      </tr>
      <tr>
        <td>Qwen3.5-9B MTP8 decode256</td>
        <td style="text-align: right;">261.0</td>
        <td style="text-align: right;">278.8</td>
        <td style="text-align: right;">+6.9%</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B EAGLE3 k=3 decode256</td>
        <td style="text-align: right;">310.5</td>
        <td style="text-align: right;">330.8</td>
        <td style="text-align: right;">+6.5%</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B EAGLE3 k=4 decode512</td>
        <td style="text-align: right;">326.7</td>
        <td style="text-align: right;">340.4</td>
        <td style="text-align: right;">+4.2%</td>
      </tr>
    </tbody>
  </table>
</div>

The speedup does not come from changing the model, the acceptance rule, or the output distribution. FlashSpec preserves the greedy-draft speculative decoding semantics. The gain comes from avoiding unnecessary vocabulary-sized memory traffic and fusing verify/resample into one pass.

# Takeaway

For greedy draft speculative decoding, the draft distribution is a delta distribution:

$$
q(x)=1.
$$

This makes the acceptance rule especially simple:

$$
\alpha(x)=p(x).
$$

FlashSpec exploits this structure. During a single pass over the target vocabulary, it computes the drafted token's acceptance probability and simultaneously prepares the recovered token for rejection. No full logits tensor, probability tensor, or residual distribution needs to be materialized.

This gives a practical fast path for n-gram, MTP, and EAGLE-style speculative decoding. In Part 2, we will move beyond delta drafts and discuss top-k draft distributions, where the draft distribution is no longer a single token but is still sparse enough to avoid full probability materialization.
