# Attention Is All You Need

Implementing the Transformer architecture from the **Attention Is All You Need** paper step by step using **PyTorch**.

The goal is to understand each component by implementing it myself rather than using PyTorch's high-level Transformer implementations.

## Current Progress

* [x] Character-level dataset
* [x] Token & positional embeddings
* [x] Scaled dot-product self-attention
* [x] Causal masking
* [x] Feed-forward network
* [x] Residual connections
* [x] Layer normalization
* [x] Language modeling head
* [x] Cross-entropy loss
* [x] Training loop
* [x] Basic autoregressive generation

## Architecture

Current implementation:

```text
Token Embedding + Positional Embedding
              ↓
       Self-Attention
              ↓
      Residual + LayerNorm
              ↓
        Feed Forward
              ↓
      Residual + LayerNorm
              ↓
       Language Model Head
```

Currently this is a small single-head, single-block implementation.

## Roadmap

* [ ] Multi-head attention
* [ ] Multiple Transformer blocks
* [ ] Improved training
* [ ] Temperature & probabilistic sampling
* [ ] Larger-scale training
* [ ] GPT-style architecture

## Tech

* Python
* PyTorch

No high-level Transformer implementation is used.

## Dataset

Initial experiments use a character-level Shakespeare dataset.

## Reference

Vaswani et al. — *Attention Is All You Need* (2017)
