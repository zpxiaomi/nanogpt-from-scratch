# NanoGPT from Scratch

A from-scratch implementation of a GPT language model in Python, built by following along with Andrej Karpathy's tutorial.

## Goal

Train a character-level language model on a given text corpus so that it can generate new text that resembles the style and content of the original input — coherent enough to feel like it belongs, even if not perfectly meaningful.

## Attribution

This project is based on the video:

**Let's build GPT: from scratch, in code, spelled out**
by [Andrej Karpathy](https://github.com/karpathy)
https://www.youtube.com/watch?v=kCc8FmEb1nY

## Project Structure

- `tokenizer.py` — character-level tokenizer and dataset preparation
- `model.py` — GPT model definition
- `train.py` — training loop
- `run.py` — Modal entrypoint for remote GPU training (see below)
- `data/` — training text data

## Remote Training with Modal

Training is run on [Modal](https://modal.com) to access GPU hardware (A100). `run.py` defines the Modal app that:

1. Builds a remote image with the required dependencies (`torch`, `numpy`) and uploads the training script and data.
2. Runs the training script on an A100 GPU with a 30-minute timeout.
3. Returns the saved checkpoint (`checkpoint_step3000.pt`) back to your local machine.

To run training remotely:

```bash
pip install modal
modal run run.py
```

The checkpoint will be saved locally as `checkpoint_step3000.pt` after the run completes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```
