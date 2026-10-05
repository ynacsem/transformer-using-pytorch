# Attention Is All You Need

Implementing the Transformer architecture step by step using **PyTorch**, with the goal of understanding the architecture from first principles rather than relying on PyTorch's high-level Transformer implementations.

The project started as a character-level Transformer language model and is progressively being extended toward a small GPT-style decoder-only architecture.

## Current Progress

* Character-level dataset
* Token & positional embeddings
* Scaled dot-product self-attention
* Causal masking
* Multi-head self-attention
* Feed-forward network
* Residual connections
* Layer normalization
* Multiple stacked Transformer blocks
* Language modeling head
* Cross-entropy loss
* Training loop
* Basic autoregressive generation

## Architecture

Current implementation:

```text
Token + Positional Embedding
            ↓
   Transformer Block × N
            │
            ├── Multi-Head Self-Attention
            │
            ├── Residual + LayerNorm
            │
            ├── Feed-Forward Network
            │
            └── Residual + LayerNorm
            ↓
       Language Model Head
            ↓
          Logits
```

Each Transformer block has its own parameters and is registered as a PyTorch module.

The model uses causal self-attention so that each position can only attend to previous positions and itself.

## Roadmap

* Improve training and optimization
* Temperature & probabilistic sampling
* Experiment with model depth, number of heads, context size and embedding dimensions
* Larger-scale training
* More extensive autoregressive generation experiments
* Further GPT-style improvements

## Implementation Philosophy

The main goal of this project is **learning by implementation**.

Rather than using:

```text
nn.Transformer
nn.TransformerEncoder
nn.MultiheadAttention
```

the core Transformer components are implemented manually using PyTorch tensor operations and basic neural network modules.

This allows each component of the architecture to be understood and experimented with independently.

## Tech

* Python
* PyTorch

No high-level Transformer implementation is used.

## Dataset

Initial experiments use a character-level Shakespeare dataset.

The model learns to predict the next character given a fixed context.

## Reference

Vaswani et al. — *Attention Is All You Need* (2017)
