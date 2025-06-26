---
title: BERT for Recommendation - BERT4Rec
description: Exploring how transformer-based models are advancing sequential recommendation systems.
author: pranay
date: 2025-03-11
categories: [Blogging, Papers]
tags: [BERT, Recommendation]
comments: false
math: true
mermaid: true
# image:
#   path: /assests/img/BERT4Rec.PNG
#   lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
#   alt: Responsive rendering of Chirpy theme on multiple devices.
---
![Transformers for Recommendation](/assets/img/bert4rec.png){: width="800" }
_Figure: Created with GPT._


## Introduction

Recommendation engines are at the heart of many e-commerce and content platforms. Building an effective recommendation system requires understanding the dynamic behavior of users. Sequential recommender systems aim to:

- Understand what a user generally likes (based on past behavior), and  
- Adapt recommendations based on their recent activity — thereby handling the dynamic nature of preferences.

**Example:** Joey usually watches light-hearted movies, but recently he watched *Die Hard*. Recommending *Die Hard 2* next is a good decision, showing why the **sequence** of user interactions matters.

Various approaches have been proposed for sequential recommendation prior to this paper — including **Markov Chains (MCs), RNNs, GRUs, LSTMs**, and deep learning models like **CNNs** and even **Transformer Decoders**. Most of these follow a similar paradigm: they encode a user’s historical interactions **left-to-right** into a hidden representation and use that for making recommendations.

**BERT4Rec**[^1] differs by using **BERT**[^2], a bidirectional model. Its bidirectional nature brings two key advantages:

- It enhances the hidden representation of items in a user’s sequence by encoding from both directions, unlike left-to-right models.
- It avoids the assumption of strictly ordered sequences, which isn’t always practical in real-world data.

## Model

The model tackles the classic sequential recommendation task:  
Given Joey’s viewing history, predict the next movie he’s likely to enjoy.

**BERT** is used to extract hidden representations of items in the user's list.

## Model Training

One key innovation in this paper is the **training objective**. Traditional sequential models are trained to **predict the next item at each step** in the input sequence. However, this approach doesn't suit a **bidirectional model**, since conditioning jointly from both directions would let items indirectly "see" the target item — defeating the purpose.

Instead, the paper uses the **Cloze task objective** (aka **Masked Language Model**). Just like BERT's original training setup, some items in the user's interaction sequence are randomly **masked**, and the model is trained to predict them using **both left and right context**.

In some cases, the **last item** is specifically masked (explained later).  
The masked item is replaced with a `<mask>` token. Its hidden representation is passed through a softmax layer over the entire item set to get a distribution.

The **loss function** used is **negative log likelihood**.

An advantage of this Cloze task is that it generates **more training samples**:  
If the sequence length is `n` and we mask `k` items, we get `nCk` possible samples.  
In contrast, next-item prediction can only create up to `t-1` samples (for a sequence of length `t`).

## Model Inference

The final goal is to **predict the next item**, but the Cloze task predicts masked items in general. To bridge this mismatch, the model adds a `<mask>` token at the **end** of the user's item sequence during inference. The prediction is then made based on the **final hidden representation** of that `<mask>` token.

## Evaluation

For evaluation:

- The full sequence is used for training, **excluding the last two items**.
- The **second last** item is used for **validation**.
- The **last** item is used for **testing**.

To evaluate the model:

- **100 negative samples** are randomly selected and combined with the correct next item.
- The model must rank all items, and the performance is measured using:

  - **Hit Ratio**
  - **NDCG**
  - **MRR**

In recommendation systems, metrics are calculated a bit differently compared to traditional classification problems.

Let’s say Joey has a list of movies he wants to watch — for example: `[Rambo, Fight Club]`. This list is the **ground truth**. The model assigns scores to all **unwatched movies**, and based on those scores (e.g., from predicted probabilities or log-likelihood), it generates a **ranked list**.

We don't evaluate performance over all unwatched movies. Instead, we take the **top-K** movies from the ranked list and calculate the metrics based on this subset.

> 🧠 **K** can be considered as the size of the app’s recommendation block (e.g., top 5 or top 10).

We also follow the **leave-one-out evaluation strategy**:
- The **last interaction** is used for **testing**
- The **second-last** is for **validation**
- The rest are used for **training**

---

#### 📌 Recall@K

Recall in classification is defined as:  
**TP / (TP + FN)** — i.e., out of all actual positives, how many were correctly predicted.

In recommendation terms, **Recall@K** means:
> Out of the actual ground truth items, how many appeared in the top-K predicted items.

---

#### 📌 Hit Ratio@K

**Hit Ratio@K** is:
> `1` if **any** of the ground truth items appears in the top-K list, otherwise `0`.

In our setup — where the ground truth has only **1 item** — **Hit Ratio** is effectively the same as **Recall**, because the denominator in Recall is 1.

---

#### 📌 Normalized Discounted Cumulative Gain (NDCG@K)

**NDCG** accounts not only for whether the ground truth item appears, but also **where** it appears in the top-K list. It rewards higher placement (top of the list) more than lower placement.

It compares the actual ranked list to an **ideal ranking**, where all relevant items are at the top.

- **DCG (Discounted Cumulative Gain)**: Measures relevance, penalizing lower-ranked relevant items using a logarithmic discount.
- **IDCG (Ideal DCG)**: The best possible DCG for that user — i.e., if all relevant items were ranked perfectly.

> **NDCG = DCG / IDCG**

This gives a normalized score between `0` and `1`:
- `1` → perfect ranking
- `0` → worst ranking

Because we use binary relevance (an item is either relevant or not), and since different users may have different numbers of relevant items, NDCG provides a consistent way to compare ranking quality across users.

---

## References

[^1]:["BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"](https://arxiv.org/pdf/1904.06690)
[^2]:["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/pdf/1810.04805)
