# NanoGPT from Scratch

A from-scratch implementation of a GPT language model in Python, built by following along with Andrej Karpathy's tutorial.

## Attribution

This project is based on the video:

**Let's build GPT: from scratch, in code, spelled out**
by [Andrej Karpathy](https://github.com/karpathy)
https://www.youtube.com/watch?v=kCc8FmEb1nY

## Project Structure

- `tokenizer.py` — character-level tokenizer and dataset preparation
- `model.py` — GPT model definition
- `train.py` — training loop
- `data/` — training text data

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```
