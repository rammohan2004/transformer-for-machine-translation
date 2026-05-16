## Wandb Report Link
You can find the Report on [WandB Report](
    https://wandb.ai/cs25m017-indian-institute-of-technology-madras/da6401-a3/reports/Assignment-3-Report-CS25M017--VmlldzoxNjg4OTQ0Mw?accessToken=4krlnrm66z4ll4fwg4j2vr4pfq0ghr541vj3u9flvhljngo2o11dh3h7z7utg511
)

# Transformer from Scratch: Machine Translation (German to English)

This repository contains a complete from-scratch implementation of the Transformer architecture as described in the paper *"Attention Is All You Need"* (Vaswani et al., 2017). 



## Project Overview

The model is trained to translate German sentences into English using the **Multi30k dataset**. The implementation completely avoids high-level PyTorch abstraction modules like `nn.Transformer` or `nn.MultiheadAttention`, building the core components—including Scaled Dot-Product Attention, Multi-Head Attention, and Positional Encodings—entirely from foundational tensor operations.

## Key Features Implemented

* **Custom Multi-Head Attention:** Built from scratch with manual head splitting, merging, and scaled dot-product attention.
* **Noam Learning Rate Scheduler:** Custom learning rate scheduler featuring a linear warmup phase followed by an inverse square root decay to stabilize early self-attention gradients.
* **Label Smoothing Loss:** Custom loss function that acts as a regularizer, redistributing target probabilities to prevent model overconfidence and overfitting.
* **Greedy Autoregressive Decoding:** Token-by-token generation with custom detokenization (regex-based contraction and punctuation fixing) to maximize BLEU evaluation scores.
* **Sinusoidal Positional Encodings:** Mathematical positional representations allowing for sequence length extrapolation, benchmarked against learned embeddings.
* **Weights & Biases Integration:** Full MLOps integration for tracking training loss, validation BLEU, learning rate curves, and interactive attention rollouts.

## File Structure

* `model.py` - Contains the core Transformer architecture, including `Encoder`, `Decoder`, `MultiHeadAttention`, `PositionwiseFeedForward`, and `PositionalEncoding`.
* `dataset.py` - Handles downloading the Multi30k dataset from HuggingFace, tokenization using `spaCy`, vocabulary building, and batch collation with dynamic padding.
* `train.py` - The main execution script. Contains the training loop, label smoothing loss, greedy decoding, BLEU evaluation (via HuggingFace `evaluate`), W&B logging, and checkpointing.
* `lr_scheduler.py` - Contains the `NoamScheduler` implementation.

## Installation & Requirements

Ensure you have Python 3.10+ installed. Install the required dependencies: