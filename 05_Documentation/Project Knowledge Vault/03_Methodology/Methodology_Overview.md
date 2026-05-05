# Methodology Overview

## Overall methodology

The current project methodology is:

1. Store a domain-specific FLAN-T5 checkpoint for seq2seq answer generation.
2. Keep optimized FLAN-T5 binaries for lighter local deployment.
3. Maintain a retrieval corpus and FAISS index for grounded answer support.
4. Keep Whisper checkpoints as the speech-recognition baseline.
5. Benchmark and integrate these assets into a single local assistant workflow.

## Current modeling stack

- text generation: `FLAN-T5`
- optimization: `FP16` and `dynamic INT8` exports
- speech recognition: `Whisper`
- retrieval: JSON records plus `FAISS`

## Current gaps

- no committed end-to-end runtime pipeline
- no scripted artifact conversion pipeline
- limited benchmark evidence for latency, RAM, and answer quality
