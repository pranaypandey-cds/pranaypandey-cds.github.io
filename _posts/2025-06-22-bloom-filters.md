---
title: Bloom Filter – Your Memory Friendly Liar
description: Dissecting the Lies and Guarantees
author: pranay
date: 2025-07-11
categories: [Blogging,Foundation]
tags: [bloom-filter, data-structures, hashing]
comments: true
math: true
mermaid: false
pin: true
---

![Bloom Filter - Your Memory-Friendly Liar](/assets/img/Bloom_filter.png){: width="800" }
_Figure: Created with GPT._

## Introduction

Imagine a data structure that can confidently tell you **"No, this item has never been seen before"**, but might lie with a **"Yes"** even if the item isn’t really there. Welcome to the world of **Bloom Filters**, a **probabilistic data structure** that trades a bit of truth for a lot of speed and memory efficiency.

A Bloom Filter is designed to answer **set membership queries** like "Is this item in the set?" without storing the actual items. Instead, it stores hash values. It **guarantees no false negatives**, meaning if it says an item is *not* present, it definitely isn't. But it may give **false positives**, i.e., say an item is present when it actually isn't.

This makes it perfect for situations where **space is tight and speed is essential** like web caches, databases, or distributed systems.

## How It Works

At the core, a Bloom Filter consists of:

- A **bit array** of size `m`, initialized to all 0s
- `k` independent **hash functions** that uniformly distribute inputs

Let’s say we want to insert the string `"Home"`:

1. Hash `"Home"` with each of the `k=3` hash functions.
2. Suppose the hash outputs are indices: `4, 5, 6`.
3. We set the bits at these positions in the bit array to 1.

Next, we insert `"Cake"` and its hash indices are `1, 3, 4`. We set those bits to 1 too.

Now, when we check for membership of `"Home"` again:

- We hash it using the same 3 hash functions → `4, 5, 6`
- We check if all these bits are 1
  - If **yes** → *“maybe present”*
  - If **any one is 0** → *definitely not present*

### Why No False Negatives?

If a string was truly inserted, then all of its hash indices would have been set to 1. So if even one of those bits is 0, we **can be certain** it was never added. Hence, **no false negatives**.

### Why False Positives?

Because different strings may produce overlapping hash indices. For example, `"Cake"` and `"Home"` both set index `4`. A new query might coincidentally find all required bits set to 1 (by other elements), even if it was never added. Thus, **false positives** occur.

### Can We Delete Items?

Here’s the problem: If we try to remove `"Home"` by resetting its bits (4, 5, 6) back to 0, we might accidentally unset a bit shared with another item (like index `4`, which `"Cake"` also uses). This leads to **false negatives** for the `"Cake"`, which violates the guarantee.

But in standard Bloom Filters, **deletion isn't supported**. If false positives become too frequent (as the bit array fills up), the filter is typically reset and rebuilt.

## Analysis

Let’s analyze the error probability:

Let:
- `m` = size of the bit array
- `k` = number of hash functions
- `n` = number of inserted elements
 
The probability that a particular bit set to 1 by a hash function:

$$
P(\text{bit = 1}) = \left(\frac{1}{m}\right)
$$

And the probability that the bit remain 0 by k hash function will be:

$$
P(\text{bit = 0}) = \left(1-\frac{1}{m}\right)^{k}
$$

The probability that a particular bit remains 0 after `n` insertions:

$$
P(\text{bit = 0}) = \left(1 - \frac{1}{m}\right)^{kn}
$$

So, probability a bit is 1:

$$
P(\text{bit = 1}) = 1 - \left(1 - \frac{1}{m} \right)^{kn}
$$

And for the Bloom filter to say it is positive all the bit from k function should be 1. Thus, the **false positive probability (FPP)** becomes:

$$
FPP = \left(1 - \left(1 - \frac{1}{m}\right)^{kn}\right)^k
$$

<!-- For large `m`, this simplifies to:

$$
FPP \approx \left(1 - e^{-kn/m}\right)^k
$$ -->

### Optimal `k` and `m`

If you want a false positive probability `p`, and you plan to store `n` items:

- **Optimal size of bit array**:
  $$
  m = -\frac{n \ln p}{(\ln 2)^2}
  $$

- **Optimal number of hash functions**:
  $$
  k = \frac{m}{n} \ln 2
  $$

<!-- This minimizes the false positive rate. -->

## Choice of Hashing Functions

Hash functions for Bloom Filters should be:
- **Independent**, or at least statistically uncorrelated
- **Uniform**, i.e., they distribute values evenly
- **Efficient**, to minimize insert/query time

In practice, people use variations like `MurmurHash`, `xxHash`, or multiple seeds of `MD5/SHA256` truncated to array size.

<!-- A common trick is **double hashing** to derive `k` hash functions from two:

$$
h_i(x) = h_1(x) + i \cdot h_2(x)
$$ -->

## Applications

Bloom Filters are widely used in:

- **Web Caches** (e.g., checking if a URL is already cached)
- **Databases** (e.g., Cassandra, HBase use them to reduce disk lookups)
- **Email Spam Filters** (checking if an email has been seen)
- **Networking** (routing, duplicate suppression)
- **Distributed Systems** (e.g., tracking resource usage or membership)

<!-- ## Code Example

```python
import hashlib
import numpy as np

class BloomFilter:
    def __init__(self, size=1000, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = np.zeros(size, dtype=bool)

    def _get_hashes(self, item):
        hashes = []
        for i in range(self.hash_count):
            data = f"{item}_{i}".encode('utf-8')
            digest = hashlib.sha256(data).hexdigest()
            index = int(digest, 16) % self.size
            hashes.append(index)
        return hashes

    def add(self, item):
        for index in self._get_hashes(item):
            self.bit_array[index] = True

    def __contains__(self, item):
        return all(self.bit_array[index] for index in self._get_hashes(item))

# Example Usage
bf = BloomFilter(size=100, hash_count=3)
bf.add("Home")
print("Home" in bf)  # True
print("Cake" in bf)  # Maybe True or False -->
