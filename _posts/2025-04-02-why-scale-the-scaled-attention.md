---
title: Why scale the Scaled Dot-Product Attention?
description: Mathematics and Logic Behind the Scale
author: pranay
date: 2025-03-11
categories: [Blogging,Foundation]
tags: [Transformer, Attention]
comments: false
math: true
mermaid: false
---
![Scaled Attention with Sherlock Holmes](/assets/img/why_scale_the_scaled_attention.png){: width="800" }
_Figure: Created with GPT._

## Overview

The groundbreaking paper *"Attention Is All You Need"*[^Paper] (2017) introduced the **Transformer architecture**, marking a major breakthrough in the field of language modeling.

Among the many innovations in that paper, the **Scaled Dot-Product Attention** mechanism stood out as a core component of the Transformer. Here's what it looks like mathematically:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The term **"scaled"** in this equation isn’t just a fancy name—it comes with solid mathematical reasoning.

Unlike many hyperparameters in deep learning that are often tuned through trial and error or heuristics (rules of thumb), the scaling factor $\sqrt{d_k}$ has a theoretical motivation.

## Logic

{: .mt-4 .mb-0 }

The paper states:

> “We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale dot products by $\sqrt{d_k}$”

Let’s dive deeper into this.

During initialization, weights are typically assigned randomly from a distribution with **zero mean** and **unit variance**. So, when we generate the query vector $\( q \)$ and key vector $\( k \)$, the dot product $\( q \cdot k \)$ will have a variance **proportional to $\( d_k \)$**, where $\( d_k \)$ is the embedding dimension.

Because,

$$
\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right)
$$

Each $\( q_i \)$ and $\( k_i \)$ are **independent**, with $\( \mathbb{E}[q_i] = \mathbb{E}[k_i] = 0 \)$ and $\( \text{Var}(q_i) = \text{Var}(k_i) = 1 \)$. So each pair $\(q_i \cdot k_i\)$ is also independent.

Therefore:

$$
\text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)
$$

And for each term:

$$
\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - \left(\mathbb{E}[q_i k_i]\right)^2 = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] = 1 \cdot 1 = 1
$$

Hence, the total variance becomes:

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} 1 = d_k
$$

This shows that the variance grows linearly with $\( d_k \)$, while the mean remains **0**.

As $d_k$ increases, the variance of the dot product grows—roughly scaling as $\sim d_k$. A higher variance causes the output of the softmax to become **very skewed**. In other words, the softmax outputs are extremely close to 1 for one position and near 0 for others.
This can be understood with a simple example:

Suppose we have the vector:

$$
[y=0.2,\ 0.4,\ 0.1,\ 0.8]
$$

Applying softmax to this would produce moderately spread probabilities. But now, consider multiplying the vector by 8:

$$
y=[1.6,\ 3.2,\ 0.8,\ 6.4]
$$

Since variance scales with the **square of the scaling factor**, the new vector has $\( 8^2 = 64 \)$ times the original variance.

Now, applying softmax to this scaled vector would result in something like:

$$
s=[0.004,\ 0.054,\ 0.001,\ 0.941]
$$

Clearly, the output becomes extremely **peaked**, with one value dominating.

Now, when we compute gradients during backpropagation, we rely on the derivative of the softmax function. This derivative is captured by a **Jacobian matrix**:

- For $i = j$: $\text{softmax}' = s_i(1 - s_i)$  
- For $i \ne j$: $\text{softmax}' = -s_i s_j$

In highly skewed distributions, most $s_i$ values are close to 0, so their derivatives become **very small**.

This leads to **vanishing gradients**, meaning the model’s parameters update very little (or not at all). As a result, **learning slows down dramatically or fails altogether**.

Now, the scaling factor $\( 1/\sqrt{d_k} \)$ is introduced to counter this effect. When we divide the dot product $\( q \cdot k \)$ by $\( \sqrt{d_k} \)$, the variance becomes:

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1
$$

This normalization keeps the variance of the scaled dot product around 1, regardless of the input dimension $\( d_k \)$. As a result, the **softmax stays in a stable, responsive region**, and the **gradient flow is preserved**.


## References

[^Paper]: ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762)
