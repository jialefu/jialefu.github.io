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
toc:
  sidebar: left
_styles: |
  @media (min-width: 992px) {
    .container[role="main"] {
      max-width: 1180px;
    }

    .container[role="main"] > .row > .col-sm-3 {
      flex: 0 0 22%;
      max-width: 22%;
    }

    .container[role="main"] > .row > .col-sm-9 {
      flex: 0 0 78%;
      max-width: 78%;
    }
  }

  #toc-sidebar {
    top: 5rem;
    max-height: calc(100vh - 7rem);
    overflow-y: auto;
    padding: 0.75rem 0 0.75rem 0.25rem;
    border-left: 1px solid var(--global-divider-color);
  }

  #toc-sidebar::before {
    content: "Contents";
    display: block;
    margin: 0 0 0.65rem 1.15rem;
    color: var(--global-text-color);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  #toc-sidebar .nav > li > a {
    padding-top: 0.28rem;
    padding-bottom: 0.28rem;
    color: var(--global-text-color-light);
    font-size: 0.86rem;
    line-height: 1.35;
  }

  #toc-sidebar .nav-link.active,
  #toc-sidebar .nav-link.active:focus,
  #toc-sidebar .nav-link.active:hover {
    color: var(--global-theme-color);
    border-left-color: var(--global-theme-color);
  }

  body {
    padding-bottom: 0;
  }

  footer.fixed-bottom {
    position: static !important;
    margin-top: 3rem;
  }

  .post {
    max-width: 880px;
    margin: 0 auto 3.5rem;
  }

  .post-header {
    margin-bottom: 2.25rem;
    padding-bottom: 1.35rem;
    border-bottom: 1px solid var(--global-divider-color);
  }

  .post-title {
    max-width: 820px;
    margin-bottom: 0.85rem;
    font-size: 2.35rem;
    font-weight: 500;
    line-height: 1.14;
  }

  .post-meta,
  .post-tags {
    color: var(--global-text-color-light);
    font-size: 0.9rem;
  }

  .post-tags a {
    white-space: nowrap;
  }

  #markdown-content {
    font-size: 1.02rem;
    line-height: 1.74;
  }

  #markdown-content p {
    margin-bottom: 1.05rem;
  }

  #markdown-content ol,
  #markdown-content ul {
    margin-bottom: 1.25rem;
  }

  #markdown-content li {
    margin-bottom: 0.35rem;
  }

  #markdown-content > h1 {
    margin-top: 3rem;
    margin-bottom: 1rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--global-divider-color);
    font-size: 1.72rem;
    font-weight: 500;
    line-height: 1.25;
  }

  #markdown-content > h1:first-child {
    margin-top: 0;
    padding-top: 0;
    border-top: 0;
  }

  #markdown-content > h2 {
    margin-top: 2.2rem;
    margin-bottom: 0.75rem;
    font-size: 1.32rem;
    font-weight: 500;
    line-height: 1.3;
  }

  #markdown-content blockquote {
    margin: 1.4rem 0;
    padding: 0.8rem 1.1rem;
    border-left: 4px solid var(--global-theme-color);
    color: var(--global-text-color);
    background: var(--global-code-bg-color);
  }

  #markdown-content :not(pre) > code {
    padding: 0.1rem 0.28rem;
    border-radius: 4px;
    color: var(--global-theme-color);
    background: var(--global-code-bg-color);
    font-size: 0.92em;
  }

  #markdown-content div.highlighter-rouge,
  #markdown-content figure.highlight {
    margin: 1.45rem 0;
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    background: var(--global-code-bg-color);
    overflow: hidden;
  }

  #markdown-content div.highlighter-rouge .highlight,
  #markdown-content figure.highlight .highlight {
    margin: 0;
    background: transparent;
  }

  #markdown-content div.highlighter-rouge pre,
  #markdown-content figure.highlight pre {
    margin: 0;
    padding: 1rem 1.1rem;
    border: 0;
    background: transparent;
    font-size: 0.86rem;
    line-height: 1.55;
  }

  #markdown-content .mjx-container[jax="CHTML"][display="true"] {
    margin: 1.25rem 0;
  }

  #markdown-content .table-responsive {
    margin: 1.5rem 0 1.8rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    overflow: hidden;
  }

  #markdown-content table.table {
    margin-bottom: 0;
    font-size: 0.94rem;
  }

  #markdown-content table.table thead th {
    border-bottom: 1px solid var(--global-divider-color);
    background: var(--global-code-bg-color);
    font-weight: 600;
  }

  #markdown-content table.table td,
  #markdown-content table.table th {
    padding: 0.72rem 0.85rem;
    vertical-align: middle;
  }

  #markdown-content table.table-striped tbody tr:nth-of-type(odd) {
    background-color: var(--global-bg-color);
  }

  .flashspec-figure {
    margin: 1.8rem 0 2.1rem;
    padding: 1.05rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    background: color-mix(in srgb, var(--global-code-bg-color) 72%, var(--global-bg-color));
    overflow-x: auto;
  }

  .flashspec-figure-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .flashspec-kicker {
    margin-bottom: 0.18rem;
    color: var(--global-theme-color);
    font-size: 0.74rem;
    font-weight: 650;
    letter-spacing: 0;
    text-transform: uppercase;
  }

  .flashspec-figure-title {
    color: var(--global-text-color);
    font-size: 1.02rem;
    font-weight: 600;
    line-height: 1.35;
  }

  .flashspec-caption {
    margin: 0.95rem 0 0 !important;
    color: var(--global-text-color-light);
    font-size: 0.88rem;
    line-height: 1.55;
  }

  .flashspec-panel {
    padding: 0.9rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 7px;
    background: var(--global-bg-color);
  }

  .flashspec-panel-title {
    margin-bottom: 0.65rem;
    color: var(--global-text-color);
    font-size: 0.86rem;
    font-weight: 650;
  }

  .flashspec-panel-note {
    margin-top: 0.65rem;
    color: var(--global-text-color-light);
    font-size: 0.78rem;
    line-height: 1.45;
  }

  .flashspec-control-row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .flashspec-control {
    display: grid;
    gap: 0.4rem;
    color: var(--global-text-color);
    font-size: 0.83rem;
    font-weight: 600;
  }

  .flashspec-control span {
    color: var(--global-text-color-light);
    font-weight: 500;
  }

  .flashspec-control input[type="range"] {
    width: 100%;
    accent-color: var(--global-theme-color);
  }

  .flashspec-compare-grid,
  .flashspec-inspector-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
  }

  .flashspec-badge {
    flex: 0 0 auto;
    padding: 0.28rem 0.55rem;
    border: 1px solid color-mix(in srgb, var(--global-theme-color) 45%, var(--global-divider-color));
    border-radius: 999px;
    color: var(--global-theme-color);
    background: var(--global-bg-color);
    font-size: 0.78rem;
    font-weight: 650;
    white-space: nowrap;
  }

  .flashspec-pipeline {
    display: grid;
    gap: 0.45rem;
  }

  .flashspec-pipeline-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    min-height: 2rem;
  }

  .flashspec-node {
    min-width: 0;
    padding: 0.42rem 0.5rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 6px;
    color: var(--global-text-color);
    background: var(--global-code-bg-color);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.74rem;
    line-height: 1.2;
    text-align: center;
  }

  .flashspec-node.is-hot {
    border-color: color-mix(in srgb, var(--global-theme-color) 55%, var(--global-divider-color));
    color: var(--global-theme-color);
  }

  .flashspec-arrow {
    color: var(--global-text-color-light);
    font-size: 0.85rem;
  }

  .flashspec-meter {
    margin-top: 0.75rem;
  }

  .flashspec-meter-track {
    height: 0.56rem;
    border-radius: 999px;
    background: var(--global-code-bg-color);
    overflow: hidden;
  }

  .flashspec-meter-fill {
    width: var(--flashspec-meter, 100%);
    height: 100%;
    border-radius: inherit;
    background: var(--global-theme-color);
    transition: width 160ms ease;
  }

  .flashspec-meter-fill.is-muted {
    background: color-mix(in srgb, var(--global-theme-color) 38%, var(--global-divider-color));
  }

  .flashspec-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.45rem;
    margin-top: 0.75rem;
  }

  .flashspec-metric {
    padding: 0.5rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 6px;
    background: var(--global-code-bg-color);
  }

  .flashspec-metric-label {
    color: var(--global-text-color-light);
    font-size: 0.68rem;
    line-height: 1.25;
  }

  .flashspec-metric-value {
    margin-top: 0.22rem;
    color: var(--global-text-color);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.84rem;
    font-weight: 650;
  }

  .flashspec-svg {
    display: block;
    width: 100%;
    height: auto;
  }

  .flashspec-svg text {
    fill: var(--global-text-color);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  }

  .flashspec-svg .muted {
    fill: var(--global-text-color-light);
  }

  .flashspec-svg .box {
    fill: var(--global-bg-color);
    stroke: var(--global-divider-color);
    stroke-width: 1.3;
  }

  .flashspec-svg .soft-box {
    fill: var(--global-code-bg-color);
    stroke: var(--global-divider-color);
    stroke-width: 1.2;
  }

  .flashspec-svg .accent-box {
    fill: color-mix(in srgb, var(--global-theme-color) 11%, var(--global-bg-color));
    stroke: var(--global-theme-color);
    stroke-width: 1.35;
  }

  .flashspec-svg .line {
    fill: none;
    stroke: var(--global-text-color-light);
    stroke-width: 1.4;
  }

  .flashspec-svg .accent-line {
    fill: none;
    stroke: var(--global-theme-color);
    stroke-width: 1.7;
  }

  .flashspec-button-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-bottom: 0.85rem;
  }

  .flashspec-button {
    padding: 0.38rem 0.62rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 6px;
    color: var(--global-text-color);
    background: var(--global-bg-color);
    font-size: 0.78rem;
    font-weight: 600;
  }

  .flashspec-button:hover,
  .flashspec-button:focus {
    border-color: var(--global-theme-color);
    color: var(--global-theme-color);
  }

  .flashspec-button.is-active {
    border-color: var(--global-theme-color);
    color: var(--global-theme-color);
    background: color-mix(in srgb, var(--global-theme-color) 9%, var(--global-bg-color));
  }

  .flashspec-token-bars {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    align-items: stretch;
    gap: 0.35rem;
    min-height: 190px;
    padding-top: 0.4rem;
  }

  .flashspec-token {
    display: grid;
    grid-template-rows: 140px auto auto;
    align-items: end;
    gap: 0.24rem;
    min-width: 0;
    color: var(--global-text-color-light);
    font-size: 0.68rem;
    text-align: center;
  }

  .flashspec-token-bar {
    align-self: end;
    width: 100%;
    height: var(--flashspec-bar, 50%);
    min-height: 0.35rem;
    border-radius: 4px 4px 2px 2px;
    background: color-mix(in srgb, var(--global-text-color-light) 30%, var(--global-code-bg-color));
    transition: height 180ms ease, background 180ms ease, opacity 180ms ease;
  }

  .flashspec-token.is-draft .flashspec-token-bar {
    background: var(--global-theme-color);
  }

  .flashspec-token.is-recovered .flashspec-token-bar {
    background: #2f9e44;
  }

  .flashspec-token.is-masked {
    opacity: 0.45;
  }

  .flashspec-token-label,
  .flashspec-token-value {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .flashspec-token-label {
    color: var(--global-text-color);
    font-weight: 650;
  }

  .flashspec-summary-list {
    display: grid;
    gap: 0.5rem;
  }

  .flashspec-summary-row {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--global-divider-color);
    color: var(--global-text-color-light);
    font-size: 0.8rem;
  }

  .flashspec-summary-row:last-child {
    padding-bottom: 0;
    border-bottom: 0;
  }

  .flashspec-summary-row strong {
    color: var(--global-text-color);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-weight: 650;
  }

  @media (max-width: 767px) {
    #toc-sidebar {
      max-height: none;
      margin-bottom: 1.5rem;
      padding: 0.85rem 0;
      border-left: 0;
      border-bottom: 1px solid var(--global-divider-color);
    }

    #toc-sidebar::before {
      margin-left: 0;
    }

    .post-title {
      font-size: 2rem;
    }

    #markdown-content {
      font-size: 1rem;
    }

    #markdown-content > h1 {
      font-size: 1.52rem;
    }

    .flashspec-figure {
      padding: 0.85rem;
    }

    .flashspec-figure-header,
    .flashspec-compare-grid,
    .flashspec-inspector-grid,
    .flashspec-control-row {
      grid-template-columns: 1fr;
    }

    .flashspec-figure-header {
      display: grid;
    }

    .flashspec-badge {
      width: fit-content;
      white-space: normal;
    }

    .flashspec-svg {
      min-width: 620px;
    }

    .flashspec-metrics {
      grid-template-columns: 1fr;
    }

    .flashspec-token-bars {
      gap: 0.22rem;
      min-height: 160px;
    }

    .flashspec-token {
      grid-template-rows: 112px auto auto;
      font-size: 0.62rem;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .flashspec-meter-fill,
    .flashspec-token-bar {
      transition: none;
    }
  }
---

# The Memory I/O Bottleneck in Speculative Decoding

Speculative Decoding accelerates LLM inference by using a lightweight drafter to propose several future tokens, followed by one target-model forward pass to verify them. When the draft is mostly accepted, one target step can produce multiple output tokens.

However, standard implementations still materialize vocabulary-sized tensors in the draft and verify/resample stages: logits, probabilities, and residual distributions. For vocabularies with 128k, 150k, or 250k tokens, these $O(V)$ reads and writes become a real HBM bottleneck.

FlashSpec asks whether this materialization is necessary: can we verify draft tokens and recover from rejection without writing full logits or probabilities to memory? In Part 1, we study the most structured case, **greedy draft speculative decoding**, where the draft model is greedy while the target model can still sample from its own distribution.

<div class="flashspec-figure flashspec-interactive" id="flashspec-memory-explorer" aria-labelledby="flashspec-memory-title">
  <div class="flashspec-figure-header">
    <div>
      <div class="flashspec-kicker">Interactive sketch</div>
      <div class="flashspec-figure-title" id="flashspec-memory-title">Where the vocabulary-sized traffic appears</div>
    </div>
    <div class="flashspec-badge" aria-live="polite"><span data-memory-ratio>4,675x</span> fewer materialized values</div>
  </div>

  <div class="flashspec-control-row">
    <label class="flashspec-control">
      Vocabulary size <span><output data-memory-vocab>150k</output> tokens</span>
      <input type="range" min="128000" max="250000" step="1000" value="150000" aria-label="Vocabulary size" data-memory-vocab-input>
    </label>
    <label class="flashspec-control">
      Draft length <span><output data-memory-gamma>5</output> positions</span>
      <input type="range" min="2" max="8" step="1" value="5" aria-label="Draft length" data-memory-gamma-input>
    </label>
  </div>

  <div class="flashspec-compare-grid">
    <div class="flashspec-panel">
      <div class="flashspec-panel-title">Standard path</div>
      <div class="flashspec-pipeline" aria-hidden="true">
        <div class="flashspec-pipeline-row">
          <div class="flashspec-node is-hot">draft logits<br>[V]</div>
          <div class="flashspec-arrow">-&gt;</div>
          <div class="flashspec-node is-hot">target logits<br>[gamma,V]</div>
        </div>
        <div class="flashspec-pipeline-row">
          <div class="flashspec-node is-hot">target probs<br>[gamma,V]</div>
          <div class="flashspec-arrow">-&gt;</div>
          <div class="flashspec-node is-hot">residual probs<br>[V]</div>
        </div>
      </div>
      <div class="flashspec-meter" aria-hidden="true">
        <div class="flashspec-meter-track"><div class="flashspec-meter-fill" data-memory-standard-bar></div></div>
      </div>
      <div class="flashspec-metrics">
        <div class="flashspec-metric">
          <div class="flashspec-metric-label">Full-vector values</div>
          <div class="flashspec-metric-value" data-memory-standard-values>1.80M</div>
        </div>
        <div class="flashspec-metric">
          <div class="flashspec-metric-label">Approx. bytes</div>
          <div class="flashspec-metric-value" data-memory-standard-bytes>3.4 MB</div>
        </div>
        <div class="flashspec-metric">
          <div class="flashspec-metric-label">Large tensors</div>
          <div class="flashspec-metric-value" data-memory-standard-tensors>12</div>
        </div>
      </div>
    </div>

    <div class="flashspec-panel">
      <div class="flashspec-panel-title">FlashSpec path</div>
      <div class="flashspec-pipeline" aria-hidden="true">
        <div class="flashspec-pipeline-row">
          <div class="flashspec-node">tile LSE</div>
          <div class="flashspec-arrow">+</div>
          <div class="flashspec-node">selected logit</div>
          <div class="flashspec-arrow">+</div>
          <div class="flashspec-node">Gumbel max</div>
        </div>
        <div class="flashspec-pipeline-row">
          <div class="flashspec-node is-hot">compact summaries</div>
          <div class="flashspec-arrow">-&gt;</div>
          <div class="flashspec-node is-hot">one output token</div>
        </div>
      </div>
      <div class="flashspec-meter" aria-hidden="true">
        <div class="flashspec-meter-track"><div class="flashspec-meter-fill is-muted" data-memory-flashspec-bar></div></div>
      </div>
      <div class="flashspec-metrics">
        <div class="flashspec-metric">
          <div class="flashspec-metric-label">Summary values</div>
          <div class="flashspec-metric-value" data-memory-flashspec-values>385</div>
        </div>
        <div class="flashspec-metric">
          <div class="flashspec-metric-label">Approx. payload</div>
          <div class="flashspec-metric-value" data-memory-flashspec-bytes>3.0 KB</div>
        </div>
        <div class="flashspec-metric">
          <div class="flashspec-metric-label">Full tensors</div>
          <div class="flashspec-metric-value">0</div>
        </div>
      </div>
    </div>
  </div>

  <p class="flashspec-caption">This is a schematic accounting, not a benchmark model. The point is the shape of the work: the standard path repeatedly writes vectors with one entry per vocabulary token, while FlashSpec returns only tile-level summaries and the few scalars needed downstream.</p>
</div>

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

<div class="flashspec-figure" aria-labelledby="flashspec-draft-diagram-title">
  <div class="flashspec-figure-header">
    <div>
      <div class="flashspec-kicker">Tiled argmax</div>
      <div class="flashspec-figure-title" id="flashspec-draft-diagram-title">Keep the running maximum, not the whole logits vector</div>
    </div>
  </div>

  <svg class="flashspec-svg" viewBox="0 0 860 300" role="img" aria-labelledby="flashspec-draft-svg-title">
    <title id="flashspec-draft-svg-title">A hidden state is multiplied with vocabulary tiles. Each tile emits a local maximum, and a final reduction returns the best token.</title>
    <defs>
      <marker id="flashspec-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="currentColor"></path>
      </marker>
    </defs>
    <rect class="box" x="24" y="108" width="112" height="72" rx="8"></rect>
    <text x="80" y="138" text-anchor="middle" font-size="16">hidden</text>
    <text class="muted" x="80" y="162" text-anchor="middle" font-size="13">h</text>

    <path class="line" d="M144 144 C180 144, 186 72, 225 72" marker-end="url(#flashspec-arrowhead)"></path>
    <path class="line" d="M144 144 C180 144, 186 120, 225 120" marker-end="url(#flashspec-arrowhead)"></path>
    <path class="line" d="M144 144 C180 144, 186 168, 225 168" marker-end="url(#flashspec-arrowhead)"></path>
    <path class="line" d="M144 144 C180 144, 186 216, 225 216" marker-end="url(#flashspec-arrowhead)"></path>

    <g>
      <rect class="soft-box" x="230" y="46" width="150" height="52" rx="7"></rect>
      <text x="305" y="68" text-anchor="middle" font-size="13">W tile 0</text>
      <text class="muted" x="305" y="87" text-anchor="middle" font-size="12">local max</text>
      <rect class="soft-box" x="230" y="94" width="150" height="52" rx="7"></rect>
      <text x="305" y="116" text-anchor="middle" font-size="13">W tile 1</text>
      <text class="muted" x="305" y="135" text-anchor="middle" font-size="12">local max</text>
      <rect class="accent-box" x="230" y="142" width="150" height="52" rx="7"></rect>
      <text x="305" y="164" text-anchor="middle" font-size="13">W tile 2</text>
      <text class="muted" x="305" y="183" text-anchor="middle" font-size="12">local max</text>
      <rect class="soft-box" x="230" y="190" width="150" height="52" rx="7"></rect>
      <text x="305" y="212" text-anchor="middle" font-size="13">W tile 3</text>
      <text class="muted" x="305" y="231" text-anchor="middle" font-size="12">local max</text>
    </g>

    <path class="line" d="M388 72 C430 72, 435 144, 480 144" marker-end="url(#flashspec-arrowhead)"></path>
    <path class="line" d="M388 120 C430 120, 435 144, 480 144" marker-end="url(#flashspec-arrowhead)"></path>
    <path class="accent-line" d="M388 168 C430 168, 435 144, 480 144" marker-end="url(#flashspec-arrowhead)"></path>
    <path class="line" d="M388 216 C430 216, 435 144, 480 144" marker-end="url(#flashspec-arrowhead)"></path>

    <rect class="accent-box" x="490" y="108" width="144" height="72" rx="8"></rect>
    <text x="562" y="137" text-anchor="middle" font-size="15">reduce max</text>
    <text class="muted" x="562" y="161" text-anchor="middle" font-size="12">token, logit</text>

    <path class="accent-line" d="M642 144 C680 144, 690 144, 728 144" marker-end="url(#flashspec-arrowhead)"></path>
    <rect class="box" x="738" y="108" width="98" height="72" rx="8"></rect>
    <text x="787" y="138" text-anchor="middle" font-size="15">best</text>
    <text class="muted" x="787" y="162" text-anchor="middle" font-size="12">O(1) write</text>
  </svg>

  <p class="flashspec-caption">Each tile can produce a local maximum in registers or shared memory. The global reduction only needs the winning token id and its logit.</p>
</div>

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

<div class="flashspec-figure flashspec-interactive" id="flashspec-one-pass-inspector" aria-labelledby="flashspec-inspector-title">
  <div class="flashspec-figure-header">
    <div>
      <div class="flashspec-kicker">One-pass inspector</div>
      <div class="flashspec-figure-title" id="flashspec-inspector-title">The same vocabulary pass prepares acceptance and recovery</div>
    </div>
    <div class="flashspec-badge" aria-live="polite">decision: <span data-inspector-decision>accept</span></div>
  </div>

  <div class="flashspec-button-row" role="group" aria-label="Inspector mode">
    <button class="flashspec-button is-active" type="button" data-inspector-mode="acceptance" aria-pressed="true">Acceptance view</button>
    <button class="flashspec-button" type="button" data-inspector-mode="recovery" aria-pressed="false">Recovery view</button>
    <button class="flashspec-button" type="button" data-inspector-resample aria-label="Resample Gumbel noise">Resample Gumbels</button>
  </div>

  <div class="flashspec-control-row">
    <label class="flashspec-control">
      Uniform threshold <span>u = <output data-inspector-uniform>0.18</output></span>
      <input type="range" min="0" max="0.95" step="0.01" value="0.18" aria-label="Uniform threshold" data-inspector-uniform-input>
    </label>
    <div class="flashspec-panel">
      <div class="flashspec-panel-title">Draft token</div>
      <div class="flashspec-summary-row"><span>x</span><strong data-inspector-draft-token>tok_4</strong></div>
    </div>
  </div>

  <div class="flashspec-inspector-grid">
    <div class="flashspec-panel">
      <div class="flashspec-panel-title" data-inspector-bars-title>Toy target logits</div>
      <div class="flashspec-token-bars" data-inspector-bars aria-label="Toy vocabulary token bars"></div>
      <div class="flashspec-panel-note">Blue marks the drafted token. Green marks the residual Gumbel-Max winner when recovery is shown.</div>
    </div>

    <div class="flashspec-panel">
      <div class="flashspec-panel-title">Compact summaries</div>
      <div class="flashspec-summary-list">
        <div class="flashspec-summary-row"><span>selected_logit</span><strong data-inspector-selected-logit>2.00</strong></div>
        <div class="flashspec-summary-row"><span>LSE</span><strong data-inspector-lse>3.42</strong></div>
        <div class="flashspec-summary-row"><span>p(x)</span><strong data-inspector-prob>0.24</strong></div>
        <div class="flashspec-summary-row"><span>uniform u</span><strong data-inspector-uniform-summary>0.18</strong></div>
        <div class="flashspec-summary-row"><span>recovered token</span><strong data-inspector-recovered>tok_6</strong></div>
      </div>
    </div>
  </div>

  <p class="flashspec-caption">Acceptance uses only the selected logit and the row LSE. If the token is rejected, the residual sample has already been prepared by the Gumbel-Max candidate that excluded the drafted token.</p>
</div>

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

# FlashSpec: One-Pass Verify-and-Resample

<div class="flashspec-figure" aria-labelledby="flashspec-stage-title">
  <div class="flashspec-figure-header">
    <div>
      <div class="flashspec-kicker">Kernel structure</div>
      <div class="flashspec-figure-title" id="flashspec-stage-title">A wide parallel summary pass followed by a tiny prefix finalizer</div>
    </div>
  </div>

  <svg class="flashspec-svg" viewBox="0 0 860 260" role="img" aria-labelledby="flashspec-stage-svg-title">
    <title id="flashspec-stage-svg-title">Stage 1 scans positions and vocabulary tiles in parallel, emits compact summaries, and Stage 2 reduces them to produce the accepted prefix or recovered token.</title>
    <defs>
      <marker id="flashspec-stage-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="currentColor"></path>
      </marker>
    </defs>
    <rect class="accent-box" x="26" y="58" width="210" height="132" rx="9"></rect>
    <text x="131" y="88" text-anchor="middle" font-size="15">Stage 1</text>
    <text class="muted" x="131" y="113" text-anchor="middle" font-size="12">parallel over</text>
    <text class="muted" x="131" y="133" text-anchor="middle" font-size="12">positions x tiles</text>
    <text x="131" y="164" text-anchor="middle" font-size="13">LM-head scan</text>

    <path class="accent-line" d="M248 124 C280 124, 296 124, 328 124" marker-end="url(#flashspec-stage-arrowhead)"></path>

    <rect class="box" x="340" y="32" width="196" height="64" rx="8"></rect>
    <text x="438" y="58" text-anchor="middle" font-size="13">tile_lse</text>
    <text class="muted" x="438" y="78" text-anchor="middle" font-size="12">for acceptance</text>

    <rect class="box" x="340" y="104" width="196" height="64" rx="8"></rect>
    <text x="438" y="130" text-anchor="middle" font-size="13">selected_logit</text>
    <text class="muted" x="438" y="150" text-anchor="middle" font-size="12">only at draft token</text>

    <rect class="box" x="340" y="176" width="196" height="64" rx="8"></rect>
    <text x="438" y="202" text-anchor="middle" font-size="13">recovered_candidate</text>
    <text class="muted" x="438" y="222" text-anchor="middle" font-size="12">Gumbel-Max per tile</text>

    <path class="line" d="M548 64 C590 64, 590 124, 632 124" marker-end="url(#flashspec-stage-arrowhead)"></path>
    <path class="line" d="M548 136 C590 136, 590 124, 632 124" marker-end="url(#flashspec-stage-arrowhead)"></path>
    <path class="line" d="M548 208 C590 208, 590 124, 632 124" marker-end="url(#flashspec-stage-arrowhead)"></path>

    <rect class="accent-box" x="644" y="58" width="190" height="132" rx="9"></rect>
    <text x="739" y="88" text-anchor="middle" font-size="15">Stage 2</text>
    <text class="muted" x="739" y="113" text-anchor="middle" font-size="12">compact reductions</text>
    <text class="muted" x="739" y="133" text-anchor="middle" font-size="12">and prefix decision</text>
    <text x="739" y="164" text-anchor="middle" font-size="13">emit draft or recovered</text>
  </svg>

  <p class="flashspec-caption">The heavy part is the parallel vocabulary scan. The sequential prefix logic only touches compact summaries, not vocabulary-sized probability vectors.</p>
</div>

The complete greedy-draft FlashSpec verify/resample kernel looks like this:

```python
# FlashSpec one-pass verify-and-resample for greedy drafts
def flashspec_greedy_draft_verify_resample(
    target_hidden,
    target_lm_head,
    draft_tokens,
    uniforms,
):
    # Stage 1: fused LM-head summary.
    # Parallel over draft positions and vocabulary tiles.
    for pos in range(gamma):
        x = draft_tokens[pos]
        h = target_hidden[pos]

        for tile_id, (W_tile, token_range) in enumerate(tiles(target_lm_head)):
            logits = h @ W_tile.T

            # 1. For acceptance: keep a tile-level LSE summary.
            tile_lse[pos, tile_id] = online_lse_summary(logits)

            # 2. Also keep the target logit at the drafted token x.
            if x in token_range:
                selected_logit[pos] = logits[x]

            # 3. For rejection: keep a residual Gumbel-Max candidate.
            gumbels = gumbel_noise(token_range)
            scores = logits + gumbels

            if x in token_range:
                scores[x] = -inf

            recovered_candidate[pos, tile_id] = max_with_index(scores, token_range)

    # Stage 2: tiny finalizer over compact summaries.
    # This is the only sequential prefix logic.
    output = []

    for pos in range(gamma):
        x = draft_tokens[pos]
        lse = reduce_lse(tile_lse[pos])
        recovered_token = reduce_max(recovered_candidate[pos])
        accept_prob = exp(selected_logit[pos] - lse)

        if uniforms[pos] <= accept_prob:
            output.append(x)
        else:
            output.append(recovered_token)
            break

    return output
```

This bypasses the large tensor reads and writes in the standard path. The vocabulary pass directly returns compact summaries for acceptance and recovery, instead of materializing full logits, probabilities, or residual probabilities.

In practice, the bonus token for the all-accepted case can be fused into the same target-side kernel as well. We omit it from the pseudocode here to keep the core idea clean.

# Experiments

We integrated FlashSpec into vLLM's greedy-draft speculative decoding paths and evaluated three representative settings: n-gram, MTP, and EAGLE3. All runs use target sampling with `temperature=1.0`, `top_p=1.0`, batch size 1, and CUDA graph decode-only mode.

Representative results:

<div class="table-responsive">
  <table class="table table-sm table-striped">
    <thead>
      <tr>
        <th>Setting</th>
        <th style="text-align: center;">Stock tok/s</th>
        <th style="text-align: center;">FlashSpec tok/s</th>
        <th style="text-align: center;">Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Qwen3.5-9B n-gram</td>
        <td style="text-align: center;">198.7</td>
        <td style="text-align: center;">217.4</td>
        <td style="text-align: center;">+9.5%</td>
      </tr>
      <tr>
        <td>Qwen3.5-9B MTP6 decode512</td>
        <td style="text-align: center;">292.8</td>
        <td style="text-align: center;">312.7</td>
        <td style="text-align: center;">+6.9%</td>
      </tr>
      <tr>
        <td>Qwen3.5-9B MTP8 decode256</td>
        <td style="text-align: center;">261.0</td>
        <td style="text-align: center;">278.8</td>
        <td style="text-align: center;">+6.9%</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B EAGLE3 k=3 decode256</td>
        <td style="text-align: center;">310.5</td>
        <td style="text-align: center;">330.8</td>
        <td style="text-align: center;">+6.5%</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B EAGLE3 k=4 decode512</td>
        <td style="text-align: center;">326.7</td>
        <td style="text-align: center;">340.4</td>
        <td style="text-align: center;">+4.2%</td>
      </tr>
    </tbody>
  </table>
</div>

The speedup does not come from changing the model, the acceptance rule, or the output distribution. FlashSpec preserves the greedy-draft speculative decoding semantics. The gain comes from avoiding unnecessary vocabulary-sized memory traffic and fusing verify/resample into one pass.

<script defer src="{{ '/assets/js/flashspec-blog.js' | relative_url | bust_file_cache }}"></script>
