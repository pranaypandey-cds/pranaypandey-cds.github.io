---
title: Why scale the Scaled Dot-Product Attention?
description: Mathematics and Logic Behind the Scale
author: pranay
date: 2025-03-11
categories: [Blogging,Foundation]
tags: [BERT, Recommendation]
comments: false
math: true
mermaid: false
---

## Why Scale the Scaled Dot-Product Attention?
{: .mt-4 .mb-0 }

## The Mathematics and Logic Behind the Scale
{: .mt-4 .mb-0 }

The groundbreaking paper *"Attention Is All You Need"* (2017) introduced the **Transformer architecture**, marking a major breakthrough in the field of language modeling.

Among the many innovations in that paper, the **Scaled Dot-Product Attention** mechanism stood out as a core component of the Transformer. Here's what it looks like mathematically:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The term **"scaled"** in this equation isn’t just a fancy name—it comes with solid mathematical reasoning.

Unlike many hyperparameters in deep learning that are often tuned through trial and error or heuristics (rules of thumb), the scaling factor $\sqrt{d_k}$ has a theoretical motivation.

### Logic
{: .mt-4 .mb-0 }

The paper states:

> “We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\sqrt{d_k}$.”

Let’s dive deeper into this.

During initialization, weights are typically assigned randomly from a distribution with **zero mean** and **unit variance**. So, when we generate the query vector $q$ and key vector $k$, the dot product $q \cdot k$ will have a variance proportional to $d_k$.

As $d_k$ increases, the variance of the dot product grows—roughly scaling as $\sim d_k$. A higher variance causes the output of the softmax to become **very skewed**. In other words, the softmax outputs are extremely close to 1 for one position and near 0 for others.

Now, when we compute gradients during backpropagation, we rely on the derivative of the softmax function. This derivative is captured by a **Jacobian matrix**:

- For $i = j$: $\text{softmax}' = s_i(1 - s_i)$  
- For $i \ne j$: $\text{softmax}' = -s_i s_j$

In highly skewed distributions, most $s_i$ values are close to 0, so their derivatives become **very small**.

This leads to **vanishing gradients**, meaning the model’s parameters update very little (or not at all). As a result, **learning slows down dramatically or fails altogether**.
